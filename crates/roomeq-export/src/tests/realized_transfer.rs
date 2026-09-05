//! Independent realized-transfer conformance for routed multi-sub exports.
//!
//! The reference transfer is computed from the canonical [`DspGraph`]
//! (plugin parameters, routing coefficients, delays, polarity). The realized
//! transfer is reconstructed from the emitted CamillaDSP YAML text: filter
//! parameters are re-parsed from the rendered document (including its decimal
//! quantization), the mixer stages are re-parsed, and the pipeline order is
//! checked. Comparing the two proves the exporter transcribes gain, signed
//! polarity, delay units, all-pass phase, routing coefficients, and
//! convolution content faithfully instead of dropping or rescaling them.
//!
//! Absolute correctness against a running CamillaDSP binary (its internal
//! biquad/interpolation implementation) is out of scope: no backend binary is
//! exercised here. The anchor properties asserted absolutely (LR −6.02 dB at
//! fc, all-pass unity magnitude) hold for any correct implementation.

use super::super::export_format::ExportFormat;
use super::super::misc::parse_biquad_filter_type;
use super::super::render_dsp_graph as render_dsp_chain;
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use num_complex::Complex64;
use roomeq_model::{
    BassManagementReport, BassManagementRoute, BassManagementRoutingGraph,
    BassManagementSubOutputReport, ChannelDspChain, DspGraph, OptimizationMetadata,
    PluginConfigWrapper,
};
use serde_json::json;
use std::collections::HashMap;
use std::f64::consts::TAU;

const BUTTERWORTH_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;
const SUB_IR_TAPS: [f64; 4] = [0.5, 0.25, 0.125, 0.0625];
const SUB_IR_FILE: &str = "sub_ir.wav";

/// Full optimizer-shaped multi-sub graph: mains plus `sub_count` sub outputs.
///
/// Every value is chosen to survive the exporter's decimal quantization
/// exactly (delays with 3 decimals, frequencies with 1, Q with 4, gains with
/// 2) so the dense-grid comparison can use tight tolerances; precision loss
/// itself is covered separately by
/// [`multisub_delay_precision_contract`].
fn multisub_fixture(sub_count: usize) -> (DspGraph, HashMap<String, Vec<f64>>) {
    assert!((2..=8).contains(&sub_count));
    let sub_names: Vec<String> = (1..=sub_count).map(|index| format!("SUB{index}")).collect();
    let mut input_channels = vec!["L".to_string(), "R".to_string()];
    input_channels.extend(sub_names.iter().cloned());

    let mut channels = HashMap::new();
    channels.insert(
        "L".to_string(),
        staged_chain(
            "L",
            vec![
                PluginConfigWrapper {
                    plugin_type: "gain".to_string(),
                    parameters: json!({"gain_db": -1.5, "room_eq_stage": "pre_route"}),
                },
                PluginConfigWrapper {
                    plugin_type: "delay".to_string(),
                    parameters: json!({"delay_ms": 1.234, "room_eq_stage": "pre_route"}),
                },
            ],
            vec![
                eq_filter("peak", 1000.0, 1.0, 3.0),
                eq_filter("allpass", 200.0, 1.0, 0.0),
                eq_filter("highshelf", 8000.0, 0.7, -2.0),
            ],
        ),
    );
    channels.insert(
        "R".to_string(),
        staged_chain(
            "R",
            vec![
                PluginConfigWrapper {
                    plugin_type: "gain".to_string(),
                    parameters: json!({"gain_db": -1.5, "invert": true, "room_eq_stage": "pre_route"}),
                },
                PluginConfigWrapper {
                    plugin_type: "delay".to_string(),
                    parameters: json!({"delay_ms": 0.01, "room_eq_stage": "pre_route"}),
                },
            ],
            vec![
                eq_filter("peak", 500.0, 2.0, -4.0),
                eq_filter("lowshelf", 100.0, 0.7, 1.5),
            ],
        ),
    );
    for sub in &sub_names {
        channels.insert(
            sub.clone(),
            staged_chain(
                sub,
                Vec::new(),
                vec![eq_filter("peak", 50.0, 1.0, -2.0)],
            ),
        );
        channels.get_mut(sub).unwrap().plugins.push(PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": SUB_IR_FILE, "room_eq_stage": "post_route"}),
        });
    }

    let redirected_gain_db = -6.0206;
    let redirected_gain_linear = 10.0_f64.powf(redirected_gain_db / 20.0);
    let mut routes = vec![
        full_range_route("L", 0, "L", 0, "LR24", Some(80.0), None, 0.0),
        full_range_route("R", 1, "R", 1, "LR24", Some(80.0), None, 0.0),
    ];
    for (index, sub) in sub_names.iter().enumerate() {
        let destination_index = index + 2;
        // Alternate crossover orders across subs so both LR24 and LR48 bass
        // crossovers are exercised at every sample rate.
        let (crossover_type, crossover_hz) = if index % 2 == 0 {
            ("LR24", 80.0)
        } else {
            ("LR48", 60.0)
        };
        // Alternate polarity pairing: even-index subs sum L+R coherently
        // (headroom case), odd-index subs oppose them (relative-phase case).
        // The L leg never inverts; the R leg inverts on odd-index subs.
        routes.push(redirected_route(
            "L",
            0,
            sub,
            destination_index,
            crossover_type,
            crossover_hz,
            redirected_gain_db,
            redirected_gain_linear,
            2.5,
            false,
        ));
        routes.push(redirected_route(
            "R",
            1,
            sub,
            destination_index,
            crossover_type,
            crossover_hz,
            redirected_gain_db,
            redirected_gain_linear,
            2.512,
            index % 2 == 0,
        ));
    }

    let routing_graph = BassManagementRoutingGraph {
        physical_sub_output: "SUB1".to_string(),
        input_channels: input_channels.clone(),
        output_channels: input_channels,
        routes,
        matrix: None,
        input_trim_db: Default::default(),
        advisories: vec!["ok".to_string()],
    };
    let metadata = OptimizationMetadata {
        pre_score: 5.0,
        post_score: 2.0,
        algorithm: "test".to_string(),
        loss_type: Some("flat".to_string()),
        iterations: 1,
        timestamp: "2026-01-01T00:00:00Z".to_string(),
        inter_channel_deviation: None,
        epa_per_channel: None,
        epa_multichannel: None,
        group_delay: None,
        mixed_phase_per_channel: None,
        perceptual_metrics: None,
        home_cinema_layout: None,
        multi_seat_coverage: None,
        multi_seat_correction: None,
        bass_management: Some(BassManagementReport {
            enabled: true,
            crossover_type: "LR24".to_string(),
            crossover_frequency_hz: Some(80.0),
            lfe_low_pass_hz: 120.0,
            redirected_bass_enabled: true,
            lfe_channel: "LFE".to_string(),
            lfe_playback_gain_db: 0.0,
            lfe_gain_applied_to_chain: false,
            sub_trim_db: 0.0,
            max_sub_boost_db: 6.0,
            headroom_margin_db: 6.0,
            applied_sub_gain_db: Some(0.0),
            gain_limited: false,
            physical_sub_output: "SUB1".to_string(),
            redirected_bass_channel_count: 2,
            main_high_pass_hz: Some(80.0),
            sub_low_pass_hz: Some(80.0),
            lfe_headroom_required_db: 16.0,
            signal_flow: Vec::new(),
            signal_flow_advisories: Vec::new(),
            routing_graph: Some(routing_graph),
            optimization: None,
            groups: Vec::new(),
            sub_outputs: vec![BassManagementSubOutputReport {
                output_role: "SUB1".to_string(),
                gain_db: 0.0,
                delay_ms: 0.0,
                polarity_inverted: false,
                strategy_source: "single".to_string(),
                headroom_contribution_db: 0.0,
            }],
            headroom_simulation: None,
            advisory: "ok".to_string(),
        }),
        timing_diagnostics: None,
        ctc: None,
        perceptual_policy: None,
        bootstrap_uncertainty: None,
        validation_bundle: None,
        supporting_source: None,
        correction_acceptance: None,
        optimizer_evidence: None,
        stage_outcomes: Vec::new(),
        effective_config: None,
    };
    let graph = DspGraph {
        deployed_source_curves: Default::default(),
        version: "1.3.0".to_string(),
        global_plugins: Vec::new(),
        channels,
        metadata: Some(metadata),
    };
    let registry = HashMap::from([(SUB_IR_FILE.to_string(), SUB_IR_TAPS.to_vec())]);
    (graph, registry)
}

/// Complex response of one canonical plugin at `frequency` Hz.
fn plugin_response(
    plugin: &PluginConfigWrapper,
    sample_rate: f64,
    frequency: f64,
    ir_registry: &HashMap<String, Vec<f64>>,
) -> Complex64 {
    let parameters = plugin.parameters.as_object().unwrap();
    match plugin.plugin_type.as_str() {
        "gain" => {
            let gain_db = parameters.get("gain_db").unwrap().as_f64().unwrap();
            let sign = if parameters
                .get("invert")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
            {
                -1.0
            } else {
                1.0
            };
            Complex64::new(sign * 10.0_f64.powf(gain_db / 20.0), 0.0)
        }
        "delay" => {
            let delay_ms = parameters.get("delay_ms").unwrap().as_f64().unwrap();
            Complex64::from_polar(1.0, -TAU * frequency * delay_ms / 1000.0)
        }
        "eq" => parameters
            .get("filters")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|filter| {
                let filter = filter.as_object().unwrap();
                let filter_type = parse_biquad_filter_type(
                    filter.get("filter_type").unwrap().as_str().unwrap(),
                )
                .unwrap();
                Biquad::new(
                    filter_type,
                    filter.get("freq").unwrap().as_f64().unwrap(),
                    sample_rate,
                    filter.get("q").unwrap().as_f64().unwrap(),
                    filter.get("db_gain").unwrap().as_f64().unwrap(),
                )
                .complex_response(frequency)
            })
            .product(),
        "crossover" => crossover_response(
            parameters.get("type").unwrap().as_str().unwrap(),
            parameters.get("frequency").unwrap().as_f64().unwrap(),
            parameters.get("output").unwrap().as_str().unwrap(),
            sample_rate,
            frequency,
        ),
        "convolution" => {
            let ir_file = parameters.get("ir_file").unwrap().as_str().unwrap();
            let taps = ir_registry
                .get(ir_file)
                .unwrap_or_else(|| panic!("unknown convolution reference '{ir_file}'"));
            fir_response(taps, sample_rate, frequency)
        }
        unsupported => panic!("fixture uses unsupported plugin '{unsupported}'"),
    }
}

/// Complex response of an LR/Butterworth crossover branch.
fn crossover_response(
    crossover_type: &str,
    frequency_hz: f64,
    output: &str,
    sample_rate: f64,
    frequency: f64,
) -> Complex64 {
    let lowpass = matches!(output.to_ascii_lowercase().as_str(), "low" | "lowpass");
    // Sections per branch: LR24 and Butterworth-24 use two cascaded
    // second-order sections, LR48 uses four.
    let sections = match crossover_type.to_ascii_lowercase().as_str() {
        "lr24" | "lr4" | "butterworth24" | "bw24" => 2,
        "lr48" | "lr8" => 4,
        other => panic!("fixture uses unsupported crossover '{other}'"),
    };
    (0..sections)
        .map(|_| {
            let filter_type = if lowpass {
                BiquadFilterType::Lowpass
            } else {
                BiquadFilterType::Highpass
            };
            Biquad::new(
                filter_type,
                frequency_hz,
                sample_rate,
                BUTTERWORTH_Q,
                0.0,
            )
            .complex_response(frequency)
        })
        .product()
}

fn fir_response(taps: &[f64], sample_rate: f64, frequency: f64) -> Complex64 {
    taps.iter()
        .enumerate()
        .map(|(index, tap)| {
            Complex64::from_polar(*tap, -TAU * frequency * index as f64 / sample_rate)
        })
        .sum()
}

fn chain_response(
    plugins: &[PluginConfigWrapper],
    stage: &str,
    sample_rate: f64,
    frequency: f64,
    ir_registry: &HashMap<String, Vec<f64>>,
) -> Complex64 {
    plugins
        .iter()
        .filter(|plugin| {
            plugin
                .parameters
                .get("room_eq_stage")
                .and_then(|value| value.as_str())
                == Some(stage)
        })
        .map(|plugin| plugin_response(plugin, sample_rate, frequency, ir_registry))
        .product()
}

/// Reference transfer from the canonical graph: output -> input -> response.
fn reference_transfer(
    graph: &DspGraph,
    sample_rate: f64,
    frequencies: &[f64],
    ir_registry: &HashMap<String, Vec<f64>>,
) -> HashMap<String, HashMap<String, Vec<Complex64>>> {
    let routing = graph
        .metadata
        .as_ref()
        .unwrap()
        .bass_management
        .as_ref()
        .unwrap()
        .routing_graph
        .as_ref()
        .unwrap();
    let mut transfer = HashMap::new();
    for destination in &routing.output_channels {
        let post: Vec<Complex64> = frequencies
            .iter()
            .map(|frequency| {
                chain_response(
                    &graph.channels[destination].plugins,
                    "post_route",
                    sample_rate,
                    *frequency,
                    ir_registry,
                )
            })
            .collect();
        let mut inputs = HashMap::new();
        for source in &routing.input_channels {
            let pre: Vec<Complex64> = frequencies
                .iter()
                .map(|frequency| {
                    chain_response(
                        &graph.channels[source].plugins,
                        "pre_route",
                        sample_rate,
                        *frequency,
                        ir_registry,
                    )
                })
                .collect();
            let response: Vec<Complex64> = frequencies
                .iter()
                .enumerate()
                .map(|(index, frequency)| {
                    let bus: Complex64 = routing
                        .routes
                        .iter()
                        .filter(|route| {
                            route.source_channel == *source && route.destination == *destination
                        })
                        .map(|route| {
                            let sign = if route.polarity_inverted { -1.0 } else { 1.0 };
                            let gain = sign * 10.0_f64.powf(route.gain_db / 20.0);
                            let branch = route
                                .high_pass_hz
                                .map(|fc| {
                                    crossover_response(
                                        &route.crossover_type,
                                        fc,
                                        "high",
                                        sample_rate,
                                        *frequency,
                                    )
                                })
                                .unwrap_or(Complex64::new(1.0, 0.0))
                                * route
                                    .low_pass_hz
                                    .map(|fc| {
                                        crossover_response(
                                            &route.crossover_type,
                                            fc,
                                            "low",
                                            sample_rate,
                                            *frequency,
                                        )
                                    })
                                    .unwrap_or(Complex64::new(1.0, 0.0));
                            let delay = Complex64::from_polar(
                                1.0,
                                -TAU * frequency * route.delay_ms / 1000.0,
                            );
                            Complex64::new(gain, 0.0) * branch * delay
                        })
                        .sum();
                    post[index] * pre[index] * bus
                })
                .collect();
            inputs.insert(source.clone(), response);
        }
        transfer.insert(destination.clone(), inputs);
    }
    transfer
}

/// One re-parsed CamillaDSP filter stage.
enum RealizedFilter {
    Gain { db: f64, inverted: bool },
    DelayMs(f64),
    Biquad {
        filter_type: BiquadFilterType,
        freq: f64,
        q: f64,
        gain_db: f64,
    },
    Crossover {
        lowpass: bool,
        freq: f64,
        sections: usize,
    },
    Conv { file: String },
}

impl RealizedFilter {
    fn response(
        &self,
        sample_rate: f64,
        frequency: f64,
        ir_registry: &HashMap<String, Vec<f64>>,
    ) -> Complex64 {
        match self {
            Self::Gain { db, inverted } => {
                let sign = if *inverted { -1.0 } else { 1.0 };
                Complex64::new(sign * 10.0_f64.powf(db / 20.0), 0.0)
            }
            Self::DelayMs(delay_ms) => {
                Complex64::from_polar(1.0, -TAU * frequency * delay_ms / 1000.0)
            }
            Self::Biquad {
                filter_type,
                freq,
                q,
                gain_db,
            } => Biquad::new(*filter_type, *freq, sample_rate, *q, *gain_db)
                .complex_response(frequency),
            Self::Crossover {
                lowpass,
                freq,
                sections,
            } => (0..*sections)
                .map(|_| {
                    Biquad::new(
                        if *lowpass {
                            BiquadFilterType::Lowpass
                        } else {
                            BiquadFilterType::Highpass
                        },
                        *freq,
                        sample_rate,
                        BUTTERWORTH_Q,
                        0.0,
                    )
                    .complex_response(frequency)
                })
                .product(),
            Self::Conv { file } => {
                let taps = ir_registry
                    .get(file)
                    .unwrap_or_else(|| panic!("exported unknown convolution file '{file}'"));
                fir_response(taps, sample_rate, frequency)
            }
        }
    }
}

fn parse_number(value: &str) -> f64 {
    value.trim().parse().unwrap_or_else(|_| panic!("expected number, got '{value}'"))
}

fn camilladsp_biquad_type(name: &str) -> BiquadFilterType {
    match name {
        "Peaking" => BiquadFilterType::Peak,
        "Lowshelf" => BiquadFilterType::Lowshelf,
        "Highshelf" => BiquadFilterType::Highshelf,
        "Lowpass" => BiquadFilterType::Lowpass,
        "Highpass" => BiquadFilterType::Highpass,
        "Notch" => BiquadFilterType::Notch,
        "Bandpass" => BiquadFilterType::Bandpass,
        "Allpass" => BiquadFilterType::AllPass,
        other => panic!("unexpected CamillaDSP biquad type '{other}'"),
    }
}

/// Split the rendered document into its top-level sections.
fn document_sections(yaml: &str) -> HashMap<String, Vec<String>> {
    let mut sections = HashMap::new();
    let mut current: Option<String> = None;
    for line in yaml.lines() {
        if !line.starts_with(' ') && !line.starts_with('#') && line.ends_with(':') && !line.is_empty()
        {
            current = Some(line.trim_end_matches(':').to_string());
            sections.entry(current.clone().unwrap()).or_insert_with(Vec::new);
        } else if let Some(section) = current.as_ref() {
            sections.get_mut(section).unwrap().push(line.to_string());
        }
    }
    sections
}

/// Parse the `filters:` section into name -> stage.
fn parse_filters(lines: &[String]) -> HashMap<String, RealizedFilter> {
    let mut filters = HashMap::new();
    let mut index = 0;
    while index < lines.len() {
        let line = &lines[index];
        let is_entry = line.starts_with("  ")
            && !line.starts_with("   ")
            && line.trim_end().ends_with(':');
        if !is_entry {
            index += 1;
            continue;
        }
        let name = line.trim().trim_end_matches(':').to_string();
        index += 1;
        let mut node_type = String::new();
        let mut params: HashMap<String, String> = HashMap::new();
        while index < lines.len()
            && (lines[index].starts_with("    ") || lines[index].trim().is_empty())
        {
            // Node attributes use 4-space indent; filter parameters use 6, so
            // a parameter literally named `type` (biquad subtype) must not
            // overwrite the node type.
            let raw = &lines[index];
            let depth = raw.len() - raw.trim_start().len();
            let entry = raw.trim();
            if depth <= 4 {
                if let Some(kind) = entry.strip_prefix("type: ") {
                    node_type = kind.trim().to_string();
                }
            } else if let Some((key, value)) = entry.split_once(':') {
                params.insert(key.trim().to_string(), value.trim().to_string());
            }
            index += 1;
        }
        let filter = match node_type.as_str() {
            "Gain" => RealizedFilter::Gain {
                db: parse_number(&params["gain"]),
                inverted: params.get("inverted").is_some_and(|v| v == "true"),
            },
            "Delay" => {
                assert_eq!(
                    params.get("unit").map(String::as_str),
                    Some("ms"),
                    "CamillaDSP delay for '{name}' must use millisecond units"
                );
                RealizedFilter::DelayMs(parse_number(&params["delay"]))
            }
            "Biquad" => RealizedFilter::Biquad {
                filter_type: camilladsp_biquad_type(&params["type"]),
                freq: parse_number(&params["freq"]),
                q: parse_number(&params["q"]),
                gain_db: params
                    .get("gain")
                    .map(|value| parse_number(value))
                    .unwrap_or(0.0),
            },
            "BiquadCombo" => {
                let (lowpass, order) = match params["type"].as_str() {
                    "LinkwitzRileyLowpass" => (true, parse_number(&params["order"]) as usize),
                    "LinkwitzRileyHighpass" => (false, parse_number(&params["order"]) as usize),
                    "ButterworthLowpass" => (true, parse_number(&params["order"]) as usize),
                    "ButterworthHighpass" => (false, parse_number(&params["order"]) as usize),
                    other => panic!("unexpected CamillaDSP combo type '{other}'"),
                };
                assert!(
                    order % 2 == 0,
                    "combo order {order} is not a biquad cascade"
                );
                RealizedFilter::Crossover {
                    lowpass,
                    freq: parse_number(&params["freq"]),
                    sections: order / 2,
                }
            }
            "Conv" => {
                let file: String = serde_json::from_str(&params["filename"])
                    .unwrap_or_else(|_| panic!("bad conv filename {}", params["filename"]));
                RealizedFilter::Conv { file }
            }
            other => panic!("unexpected CamillaDSP filter node '{other}'"),
        };
        filters.insert(name, filter);
    }
    filters
}

/// One `- channel:` source row inside a mixer mapping.
struct MixerSource {
    channel: usize,
    gain_db: f64,
    inverted: bool,
}

/// Parse one `roomeq_route_*` mixer entry: dest index -> source rows.
fn parse_mixer_mapping(lines: &[String], entry: &str) -> Vec<(usize, Vec<MixerSource>)> {
    let header = format!("  {entry}:");
    let start = lines
        .iter()
        .position(|line| line == &header)
        .unwrap_or_else(|| panic!("missing mixer entry '{entry}'"));
    let mut mapping: Vec<(usize, Vec<MixerSource>)> = Vec::new();
    let mut current_source: Option<MixerSource> = None;
    let flush_source = |mapping: &mut Vec<(usize, Vec<MixerSource>)>,
                        source: &mut Option<MixerSource>| {
        if let Some(source) = source.take() {
            mapping
                .last_mut()
                .expect("mixer source without destination")
                .1
                .push(source);
        }
    };
    for line in lines.iter().skip(start + 1) {
        if line.starts_with("  ") && !line.starts_with("   ") {
            break;
        }
        let entry = line.trim();
        if let Some(dest) = entry.strip_prefix("- dest: ") {
            flush_source(&mut mapping, &mut current_source);
            mapping.push((dest.trim().parse().unwrap(), Vec::new()));
        } else if let Some(channel) = entry.strip_prefix("- channel: ") {
            flush_source(&mut mapping, &mut current_source);
            current_source = Some(MixerSource {
                channel: channel.trim().parse().unwrap(),
                gain_db: 0.0,
                inverted: false,
            });
        } else if let Some(gain) = entry.strip_prefix("gain: ") {
            current_source.as_mut().expect("mixer gain without source").gain_db =
                parse_number(gain);
        } else if let Some(inverted) = entry.strip_prefix("inverted: ") {
            current_source
                .as_mut()
                .expect("mixer polarity without source")
                .inverted = inverted.trim() == "true";
        } else if let Some(scale) = entry.strip_prefix("scale: ") {
            assert_eq!(scale.trim(), "dB", "route mixer must use dB scale");
        }
    }
    flush_source(&mut mapping, &mut current_source);
    mapping
}

enum PipelineStep {
    Filter { channel: usize, names: Vec<String> },
    Mixer { name: String },
}

/// Parse the `pipeline:` section into ordered steps, rejecting any layout
/// that does not match the routed exporter's documented skeleton.
fn parse_pipeline(lines: &[String]) -> Vec<PipelineStep> {
    let meaningful: Vec<&str> = lines
        .iter()
        .map(|line| line.trim())
        .filter(|entry| !entry.is_empty())
        .collect();
    let mut steps = Vec::new();
    let mut index = 0;
    while index < meaningful.len() {
        assert_eq!(
            meaningful[index], "- bypassed: null",
            "unexpected pipeline step start '{}'",
            meaningful[index]
        );
        index += 1;
        if meaningful[index] == "channels:" {
            index += 1;
            let channel: usize = meaningful[index]
                .strip_prefix("- ")
                .unwrap_or_else(|| panic!("expected pipeline channel, got '{}'", meaningful[index]))
                .trim()
                .parse()
                .unwrap();
            index += 1;
            assert_eq!(meaningful[index], "names:", "expected pipeline names");
            index += 1;
            let mut names = Vec::new();
            while meaningful[index] != "type: Filter" {
                names.push(
                    meaningful[index]
                        .strip_prefix("- ")
                        .unwrap_or_else(|| {
                            panic!("expected pipeline filter name, got '{}'", meaningful[index])
                        })
                        .trim()
                        .to_string(),
                );
                index += 1;
            }
            index += 1;
            steps.push(PipelineStep::Filter { channel, names });
        } else if let Some(name) = meaningful[index].strip_prefix("name: ") {
            let name = name.trim().to_string();
            index += 1;
            assert_eq!(meaningful[index], "type: Mixer", "expected mixer step");
            index += 1;
            steps.push(PipelineStep::Mixer { name });
        } else {
            panic!("unexpected pipeline step body '{}'", meaningful[index]);
        }
    }
    steps
}

/// Everything reconstructed from the rendered YAML document.
struct RealizedDocument {
    /// output -> input -> response per frequency.
    transfer: HashMap<String, HashMap<String, Vec<Complex64>>>,
    /// Per-input pre-route chain responses (drive normalization).
    pre: Vec<Vec<Complex64>>,
    /// Per-route signed linear expand gain and source input index.
    expand: Vec<(usize, f64)>,
    /// Per-destination route indices from the sum mixer.
    sum: Vec<Vec<usize>>,
    /// Per-route filter chain responses.
    route: Vec<Vec<Complex64>>,
    /// Per-output post-route chain responses.
    post: Vec<Vec<Complex64>>,
}

/// Transfer reconstructed from the rendered YAML document.
fn realized_transfer(
    yaml: &str,
    sample_rate: f64,
    frequencies: &[f64],
    input_channels: &[String],
    output_channels: &[String],
    ir_registry: &HashMap<String, Vec<f64>>,
) -> RealizedDocument {
    let sections = document_sections(yaml);
    let filters = parse_filters(&sections["filters"]);
    let expand = parse_mixer_mapping(&sections["mixers"], "roomeq_route_matrix");
    let sum = parse_mixer_mapping(&sections["mixers"], "roomeq_route_sum");
    let steps = parse_pipeline(&sections["pipeline"]);

    let chain = |names: &[String]| -> Vec<Complex64> {
        frequencies
            .iter()
            .map(|frequency| {
                names
                    .iter()
                    .map(|name| {
                        filters
                            .get(name)
                            .unwrap_or_else(|| panic!("pipeline references unknown filter '{name}'"))
                            .response(sample_rate, *frequency, ir_registry)
                    })
                    .product()
            })
            .collect()
    };

    // Walk the documented routed skeleton: pre-route filters, expand mixer,
    // per-route filters, sum mixer, post-route filters. Empty chains emit no
    // pipeline step and default to unity.
    let mut pre = vec![Vec::new(); input_channels.len()];
    let mut route = vec![Vec::new(); expand.len()];
    let mut post = vec![Vec::new(); output_channels.len()];
    let mut phase = 0;
    for step in &steps {
        match step {
            PipelineStep::Filter { channel, names } => match phase {
                0 => pre[*channel] = names.clone(),
                2 => route[*channel] = names.clone(),
                4 => post[*channel] = names.clone(),
                _ => panic!("filter step outside routed skeleton phase {phase}"),
            },
            PipelineStep::Mixer { name } if name == "roomeq_route_matrix" && phase == 0 => {
                phase = 2;
            }
            PipelineStep::Mixer { name } if name == "roomeq_route_sum" && phase == 2 => phase = 4,
            PipelineStep::Mixer { name } => panic!("unexpected mixer step '{name}'"),
        }
    }
    assert_eq!(phase, 4, "pipeline never reached the post-route stage");

    for (route_index, (dest, sources)) in expand.iter().enumerate() {
        assert_eq!(*dest, route_index, "expand mixer must list one bus per route");
        assert_eq!(sources.len(), 1, "expand mixer must map one source per route");
    }
    for (dest_index, (dest, _)) in sum.iter().enumerate() {
        assert_eq!(*dest, dest_index, "sum mixer must cover every destination");
    }
    for (_, sources) in &sum {
        for source in sources {
            assert_eq!(source.gain_db, 0.0, "sum mixer must sum at unity");
            assert!(!source.inverted, "sum mixer must not invert");
        }
    }

    let pre_response: Vec<Vec<Complex64>> =
        pre.iter().map(|names| chain(names)).collect();
    let route_response: Vec<Vec<Complex64>> =
        route.iter().map(|names| chain(names)).collect();
    let post_response: Vec<Vec<Complex64>> =
        post.iter().map(|names| chain(names)).collect();

    let expand_signed: Vec<(usize, f64)> = expand
        .iter()
        .map(|(_, sources)| {
            let source = &sources[0];
            let sign = if source.inverted { -1.0 } else { 1.0 };
            (source.channel, sign * 10.0_f64.powf(source.gain_db / 20.0))
        })
        .collect();
    let sum_routes: Vec<Vec<usize>> = sum
        .iter()
        .map(|(_, sources)| sources.iter().map(|row| row.channel).collect())
        .collect();

    let mut transfer = HashMap::new();
    for (dest_index, destination) in output_channels.iter().enumerate() {
        let mut inputs = HashMap::new();
        for (source_index, source) in input_channels.iter().enumerate() {
            let response: Vec<Complex64> = frequencies
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    let mut bus = Complex64::new(0.0, 0.0);
                    for (route_index, (route_source, gain)) in expand_signed.iter().enumerate() {
                        if *route_source != source_index {
                            continue;
                        }
                        if !sum_routes[dest_index].contains(&route_index) {
                            continue;
                        }
                        bus += Complex64::new(*gain, 0.0) * route_response[route_index][index];
                    }
                    post_response[dest_index][index]
                        * pre_response[source_index][index]
                        * bus
                })
                .collect();
            inputs.insert(source.clone(), response);
        }
        transfer.insert(destination.clone(), inputs);
    }
    RealizedDocument {
        transfer,
        pre: pre_response,
        expand: expand_signed,
        sum: sum_routes,
        route: route_response,
        post: post_response,
    }
}

fn log_grid() -> Vec<f64> {
    // Dense log grid from 10 Hz to 20 kHz.
    let points = 192;
    (0..points)
        .map(|index| 10.0 * (2000.0_f64).powf(index as f64 / (points - 1) as f64))
        .collect()
}

/// Compare realized against reference on a dense grid.
///
/// Fixture values round-trip through the exporter's decimals exactly, so any
/// transcription defect (wrong units, dropped stage, swapped route, rescaled
/// convolution) exceeds these tolerances by orders of magnitude.
fn assert_transfer_matches(
    reference: &HashMap<String, HashMap<String, Vec<Complex64>>>,
    realized: &HashMap<String, HashMap<String, Vec<Complex64>>>,
    frequencies: &[f64],
    context: &str,
) {
    assert_eq!(reference.len(), realized.len(), "{context}: output channel set");
    for (destination, reference_inputs) in reference {
        let realized_inputs = &realized[destination];
        assert_eq!(
            reference_inputs.len(),
            realized_inputs.len(),
            "{context}: input channel set for '{destination}'"
        );
        for (source, reference_response) in reference_inputs {
            let realized_response = &realized_inputs[source];
            assert_eq!(reference_response.len(), frequencies.len());
            assert_eq!(realized_response.len(), frequencies.len());
            for (index, frequency) in frequencies.iter().enumerate() {
                let expected = reference_response[index];
                let actual = realized_response[index];
                let magnitude = expected.norm();
                if magnitude > 1e-8 {
                    let expected_db = 20.0 * magnitude.log10();
                    let actual_db = 20.0 * actual.norm().max(1e-300).log10();
                    assert!(
                        (expected_db - actual_db).abs() < 1e-6,
                        "{context}: {source} -> {destination} at {frequency} Hz: \
                         {expected_db:.9} dB vs {actual_db:.9} dB"
                    );
                    let relative = (expected - actual).norm() / magnitude;
                    assert!(
                        relative < 1e-9,
                        "{context}: {source} -> {destination} at {frequency} Hz: \
                         complex relative error {relative:.3e}"
                    );
                } else {
                    assert!(
                        actual.norm() < 1e-6,
                        "{context}: {source} -> {destination} at {frequency} Hz: \
                         expected silence, realized {}",
                        actual.norm()
                    );
                }
            }
        }
    }
}

fn render_routed(
    sub_count: usize,
    sample_rate: f64,
) -> (
    DspGraph,
    HashMap<String, Vec<f64>>,
    String,
    Vec<String>,
    Vec<String>,
) {
    let (graph, registry) = multisub_fixture(sub_count);
    let yaml = render_dsp_chain(&graph, ExportFormat::CamillaDsp, sample_rate)
        .expect("supported multi-sub fixture must render");
    let routing = graph
        .metadata
        .as_ref()
        .unwrap()
        .bass_management
        .as_ref()
        .unwrap()
        .routing_graph
        .as_ref()
        .unwrap();
    (
        graph,
        registry,
        yaml,
        routing.input_channels.clone(),
        routing.output_channels.clone(),
    )
}

#[test]
fn multisub_routed_transfer_matches_canonical() {
    // Full 2/4/8-sub fixtures, optimization-shaped (staged plugins, routing
    // graph, single-sub report metadata), at both export sample rates:
    // dense-grid complex comparison (primary-seat semantics) of the
    // reconstructed preset against the canonical graph.
    for sub_count in [2, 4, 8] {
        for sample_rate in [44_100.0, 48_000.0] {
            let (graph, registry, yaml, inputs, outputs) =
                render_routed(sub_count, sample_rate);
            let frequencies = log_grid();
            let reference = reference_transfer(&graph, sample_rate, &frequencies, &registry);
            let realized = realized_transfer(
                &yaml,
                sample_rate,
                &frequencies,
                &inputs,
                &outputs,
                &registry,
            );
            assert_transfer_matches(
                &reference,
                &realized.transfer,
                &frequencies,
                &format!("{sub_count}-sub @{sample_rate}Hz"),
            );
        }
    }
}

#[test]
fn multisub_crossover_anchor_holds_across_sample_rates() {
    // Bass-crossover SRC angle: the LR branches must sit at -6.02 dB on their
    // design frequency at every export rate, pass bass below, and stop above,
    // proving frequency, order, and low/high transcription (not just shape).
    for sample_rate in [44_100.0, 48_000.0] {
        let (graph, registry, yaml, inputs, outputs) = render_routed(4, sample_rate);
        // SUB1 hangs off the LR24 80 Hz branch, SUB2 off the LR48 60 Hz one.
        let probes = [
            ("SUB1", "L", 80.0, 0.5, 0.99, 0.1),
            ("SUB2", "L", 60.0, 0.5, 0.999, 0.01),
        ];
        let frequencies: Vec<f64> = probes
            .iter()
            .flat_map(|(_, _, fc, _, _, _)| [*fc / 2.0, *fc, *fc * 2.0])
            .collect();
        let realized = realized_transfer(
            &yaml,
            sample_rate,
            &frequencies,
            &inputs,
            &outputs,
            &registry,
        );
        let _ = (graph, registry);
        for (destination, source, fc, at_fc, below, above) in probes {
            let dest_index = outputs.iter().position(|name| name == destination).unwrap();
            let source_index = inputs.iter().position(|name| name == source).unwrap();
            let response = &realized.transfer[destination][source];
            let anchor = |frequency: f64| {
                let index = frequencies
                    .iter()
                    .position(|candidate| candidate == &frequency)
                    .unwrap();
                // Divide out the pre/post chains; route legs contribute only
                // the crossover (|delay| == 1), so this isolates |LP(fc)|.
                response[index].norm()
                    / (realized.pre[source_index][index].norm()
                        * realized.post[dest_index][index].norm())
                    / 10.0_f64.powf(-6.0206 / 20.0)
            };
            let measured = anchor(fc) * 10.0_f64.powf(-6.0206 / 20.0);
            assert!(
                (measured - at_fc).abs() < 1e-9,
                "{destination} LP at {fc} Hz @{sample_rate}Hz: {measured}"
            );
            assert!(
                anchor(fc / 2.0) * 10.0_f64.powf(-6.0206 / 20.0) > below,
                "{destination} LP must pass bass below {fc} Hz"
            );
            assert!(
                anchor(fc * 2.0) * 10.0_f64.powf(-6.0206 / 20.0) < above,
                "{destination} LP must stop above {fc} Hz"
            );
        }
    }
}

#[test]
fn multisub_allpass_is_phase_only() {
    // An exported all-pass section must be magnitude-transparent while moving
    // phase; a dropped or magnitude-coupled all-pass fails here.
    let output = DspGraph {
        deployed_source_curves: Default::default(),
        version: "1.3.0".to_string(),
        global_plugins: Vec::new(),
        channels: HashMap::from([(
            "left".to_string(),
            ChannelDspChain {
                channel: "left".to_string(),
                plugins: vec![PluginConfigWrapper {
                    plugin_type: "eq".to_string(),
                    parameters: json!({"filters": [{
                        "filter_type": "allpass",
                        "freq": 200.0,
                        "q": 1.0,
                        "db_gain": 0.0,
                    }]}),
                }],
                drivers: None,
                initial_curve: None,
                final_curve: None,
                eq_response: None,
                target_curve: None,
                pre_ir: None,
                post_ir: None,
                fir_temporal_masking: None,
                direct_early_late_correction: None,
            },
        )]),
        metadata: None,
    };
    let yaml = render_dsp_chain(&output, ExportFormat::CamillaDsp, 48_000.0).unwrap();
    let sections = document_sections(&yaml);
    let filters = parse_filters(&sections["filters"]);
    assert_eq!(filters.len(), 1);
    let filter = filters.values().next().unwrap();
    let frequencies = log_grid();
    let registry = HashMap::new();
    let mut peak_phase = 0.0;
    for frequency in &frequencies {
        let response = filter.response(48_000.0, *frequency, &registry);
        assert!(
            (response.norm() - 1.0).abs() < 1e-12,
            "all-pass magnitude at {frequency} Hz: {}",
            response.norm()
        );
        peak_phase = peak_phase.max(response.arg().abs());
    }
    assert!(peak_phase > 0.5, "all-pass must move phase ({peak_phase})");
}

#[test]
fn multisub_relative_phase_and_headroom_through_routing() {
    // SUB1 sums L+R coherently (headroom case), SUB2 opposes them
    // (relative-phase case). Drive-normalized bus sums isolate the routing
    // matrix: coherent legs must add linearly (not RMS), opposed legs must
    // cancel in the bass band.
    let sample_rate = 48_000.0;
    let (_graph, _registry, yaml, inputs, outputs) = render_routed(2, sample_rate);
    let frequencies: Vec<f64> = (0..24).map(|index| 10.0 * 20.0_f64.powf(index as f64 / 23.0)).collect();
    let registry = HashMap::from([(SUB_IR_FILE.to_string(), SUB_IR_TAPS.to_vec())]);
    let realized = realized_transfer(&yaml, sample_rate, &frequencies, &inputs, &outputs, &registry);
    let bus = |destination: &str| {
        frequencies
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let l = realized.transfer[destination]["L"][index] / realized.pre[0][index];
                let r = realized.transfer[destination]["R"][index] / realized.pre[1][index];
                (l, r)
            })
            .collect::<Vec<_>>()
    };
    for (index, _frequency) in frequencies.iter().enumerate() {
        let (l, r) = bus("SUB1")[index];
        let coherent = (l + r).norm() / (l.norm() + r.norm());
        assert!(
            coherent > 0.99,
            "SUB1 legs must sum coherently (headroom), ratio {coherent}"
        );
        let (l, r) = bus("SUB2")[index];
        let residual = (l + r).norm() / (l.norm() + r.norm());
        assert!(
            residual < 0.02,
            "SUB2 legs must cancel through signed polarity, residual {residual}"
        );
    }
    // Headroom basis at 10 Hz (delays contribute no phase yet): the coherent
    // peak gain equals the algebraic sum of the legs, ~+6 dB over one leg.
    let (l, r) = bus("SUB1")[0];
    let peak = (l + r).norm();
    let leg = l.norm();
    assert!(
        (peak / leg - 2.0).abs() < 1e-3,
        "coherent peak {peak} must equal twice one leg {leg}"
    );
}

#[test]
fn multisub_impulse_excitation_matches() {
    // Time-domain view of the same contract: excite with an impulse (IDFT of
    // the realized transfer) and require identical responses, including peak
    // position, so delay realization is checked in samples, not just phase.
    for sample_rate in [44_100.0, 48_000.0] {
        let (graph, registry, yaml, inputs, outputs) = render_routed(2, sample_rate);
        let points = 512;
        let frequencies: Vec<f64> = (0..=points / 2)
            .map(|bin| bin as f64 * sample_rate / points as f64)
            .collect();
        let reference = reference_transfer(&graph, sample_rate, &frequencies, &registry);
        let realized = realized_transfer(
            &yaml,
            sample_rate,
            &frequencies,
            &inputs,
            &outputs,
            &registry,
        )
        .transfer;
        for destination in &outputs {
            let expected = impulse(&reference[destination]["L"], points);
            let actual = impulse(&realized[destination]["L"], points);
            let peak = expected
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
                .map(|(index, _)| index)
                .unwrap();
            let actual_peak = actual
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
                .map(|(index, _)| index)
                .unwrap();
            assert_eq!(
                peak, actual_peak,
                "{destination} impulse peak position @{sample_rate}Hz"
            );
            let scale = expected[peak].norm().max(1e-12);
            for (index, (a, b)) in expected.iter().zip(actual.iter()).enumerate() {
                assert!(
                    (a - b).norm() / scale < 1e-9,
                    "{destination} impulse sample {index} @{sample_rate}Hz"
                );
            }
        }
    }
}

/// Inverse DFT of a one-sided spectrum evaluated at `frequencies[k] =
/// k * fs / points`, driving the L input with an impulse.
fn impulse(spectrum: &[Complex64], points: usize) -> Vec<Complex64> {
    assert_eq!(spectrum.len(), points / 2 + 1);
    (0..points)
        .map(|n| {
            let mut sample = spectrum[0] + spectrum[points / 2] * if n % 2 == 0 { 1.0 } else { -1.0 };
            for k in 1..points / 2 {
                let angle = TAU * k as f64 * n as f64 / points as f64;
                sample += (spectrum[k] * Complex64::from_polar(1.0, angle)) * 2.0;
            }
            sample /= points as f64;
            sample
        })
        .collect()
}

#[test]
fn multisub_spatial_magnitude_semantics() {
    // Spatial checks average magnitudes across sub seats (phase varies by
    // seat), while the primary seat keeps the complex comparison from
    // `multisub_routed_transfer_matches_canonical`.
    let sample_rate = 48_000.0;
    let (graph, registry, yaml, inputs, outputs) = render_routed(8, sample_rate);
    let frequencies = log_grid();
    let reference = reference_transfer(&graph, sample_rate, &frequencies, &registry);
    let realized = realized_transfer(&yaml, sample_rate, &frequencies, &inputs, &outputs, &registry)
        .transfer;
    let subs: Vec<String> = (1..=8).map(|index| format!("SUB{index}")).collect();
    for (index, frequency) in frequencies.iter().enumerate() {
        let reference_mean: f64 = subs
            .iter()
            .map(|sub| reference[sub]["L"][index].norm())
            .sum::<f64>()
            / subs.len() as f64;
        let realized_mean: f64 = subs
            .iter()
            .map(|sub| realized[sub]["L"][index].norm())
            .sum::<f64>()
            / subs.len() as f64;
        assert!(
            (reference_mean - realized_mean).abs() / reference_mean.max(1e-12) < 1e-9,
            "spatial sub mean at {frequency} Hz: {reference_mean} vs {realized_mean}"
        );
    }
    // Mixed polarities make the coherent sum far smaller than the spatial
    // mean; the harness records that distinction instead of conflating it.
    let coherent: f64 = subs
        .iter()
        .map(|sub| realized[sub]["L"][0])
        .sum::<Complex64>()
        .norm()
        / subs.len() as f64;
    let mean: f64 = subs
        .iter()
        .map(|sub| realized[sub]["L"][0].norm())
        .sum::<f64>()
        / subs.len() as f64;
    assert!(
        mean / coherent.max(1e-12) > 1.5,
        "spatial mean {mean} must exceed coherent sum {coherent}"
    );
}

#[test]
fn multisub_delay_precision_contract() {
    // Delays render with millisecond decimals: 1.23456 ms becomes 1.235 ms.
    // The contract pins the rounding bound and the realized phase slope.
    let output = DspGraph {
        deployed_source_curves: Default::default(),
        version: "1.3.0".to_string(),
        global_plugins: Vec::new(),
        channels: HashMap::from([(
            "left".to_string(),
            ChannelDspChain {
                channel: "left".to_string(),
                plugins: vec![PluginConfigWrapper {
                    plugin_type: "delay".to_string(),
                    parameters: json!({"delay_ms": 1.23456}),
                }],
                drivers: None,
                initial_curve: None,
                final_curve: None,
                eq_response: None,
                target_curve: None,
                pre_ir: None,
                post_ir: None,
                fir_temporal_masking: None,
                direct_early_late_correction: None,
            },
        )]),
        metadata: None,
    };
    let yaml = render_dsp_chain(&output, ExportFormat::CamillaDsp, 48_000.0).unwrap();
    assert!(yaml.contains("delay: 1.235"), "unexpected delay line:\n{yaml}");
    assert!((1.235 - 1.23456).abs() <= 5e-4 + 1e-12);
    let sections = document_sections(&yaml);
    let filters = parse_filters(&sections["filters"]);
    let registry = HashMap::new();
    let response = filters["left_delay"].response(48_000.0, 1000.0, &registry);
    let expected = Complex64::from_polar(1.0, -TAU * 1000.0 * 0.001_235);
    assert!((response - expected).norm() < 1e-12);
}

#[test]
fn camilladsp_rejects_shared_global_eq() {
    // Shared/global EQ has no preset stage: it must error, never silently
    // drop, so a partial preset cannot read as complete.
    let (mut graph, _registry) = multisub_fixture(2);
    graph.global_plugins.push(PluginConfigWrapper {
        plugin_type: "eq".to_string(),
        parameters: json!({"filters": [{
            "filter_type": "peak", "freq": 1000.0, "q": 1.0, "db_gain": -1.0,
        }]}),
    });
    let error = render_dsp_chain(&graph, ExportFormat::CamillaDsp, 48_000.0)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("does not support global plugin"),
        "unexpected error: {error}"
    );
}

fn eq_filter(filter_type: &str, freq: f64, q: f64, db_gain: f64) -> serde_json::Value {
    json!({"filter_type": filter_type, "freq": freq, "q": q, "db_gain": db_gain})
}

fn staged_chain(
    name: &str,
    pre_route: Vec<PluginConfigWrapper>,
    post_filters: Vec<serde_json::Value>,
) -> ChannelDspChain {
    let mut plugins = pre_route;
    if !post_filters.is_empty() {
        plugins.push(PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: json!({"filters": post_filters, "room_eq_stage": "post_route"}),
        });
    }
    ChannelDspChain {
        channel: name.to_string(),
        plugins,
        drivers: None,
        initial_curve: None,
        final_curve: None,
        eq_response: None,
        target_curve: None,
        pre_ir: None,
        post_ir: None,
        fir_temporal_masking: None,
        direct_early_late_correction: None,
    }
}

fn full_range_route(
    source: &str,
    source_index: usize,
    destination: &str,
    destination_index: usize,
    crossover_type: &str,
    high_pass_hz: Option<f64>,
    low_pass_hz: Option<f64>,
    delay_ms: f64,
) -> BassManagementRoute {
    BassManagementRoute {
        group_id: Some("mains".to_string()),
        source_channel: source.to_string(),
        source_index,
        destination: destination.to_string(),
        destination_index,
        pre_chain_channel: Some(source.to_string()),
        post_chain_channel: Some(destination.to_string()),
        route_kind: "main_highpass_to_self".to_string(),
        crossover_type: crossover_type.to_string(),
        high_pass_hz: high_pass_hz,
        low_pass_hz: low_pass_hz,
        gain_db: 0.0,
        gain_linear: 1.0,
        matrix_gain: 1.0,
        delay_ms,
        polarity_inverted: false,
    }
}

fn redirected_route(
    source: &str,
    source_index: usize,
    destination: &str,
    destination_index: usize,
    crossover_type: &str,
    crossover_hz: f64,
    gain_db: f64,
    gain_linear: f64,
    delay_ms: f64,
    polarity_inverted: bool,
) -> BassManagementRoute {
    BassManagementRoute {
        group_id: Some("mains".to_string()),
        source_channel: source.to_string(),
        source_index,
        destination: destination.to_string(),
        destination_index,
        pre_chain_channel: Some(source.to_string()),
        post_chain_channel: Some(destination.to_string()),
        route_kind: "redirected_bass_lowpass_to_sub".to_string(),
        crossover_type: crossover_type.to_string(),
        high_pass_hz: None,
        low_pass_hz: Some(crossover_hz),
        gain_db,
        gain_linear,
        matrix_gain: gain_linear,
        delay_ms,
        polarity_inverted,
    }
}
