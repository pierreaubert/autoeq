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
//! The opt-in `tool_contract_camilladsp` test also runs an actual backend for
//! sampled steady-state physical-sub peaks. This does not cover every matrix
//! bundle, broadband transients, clipping, or acoustic/listening outcomes.
//! The analytic anchors (LR −6.02 dB at fc, all-pass unity magnitude) are
//! checked independently of backend availability.

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

const SUB_IR_TAPS: [f64; 4] = [0.5, 0.25, 0.125, 0.0625];
const SUB_IR_FILE: &str = "sub_ir.wav";

/// Diagnostic only: distinguish integer rounding from the backend's optional
/// subsample approximation. Printed errors are evidence, not acceptance.
#[test]
#[ignore = "requires ROOMEQ_CAMILLADSP_BIN; diagnostic, not a conformance gate"]
fn characterize_matrix_fractional_delay_backend() {
    std::env::var("ROOMEQ_CAMILLADSP_BIN").expect("backend is required");
    let rate = 48_000.0;
    let delay_ms = 0.10478137820238476;
    for subsample in [false, true] {
        // Isolate backend realization from exporter decimal quantization.
        let yaml = format!("devices:\n  samplerate: 48000\n  chunksize: 4096\n  capture:\n    type: Stdin\n    channels: 1\n    format: S32_LE\n  playback:\n    type: Stdout\n    channels: 1\n    format: S32_LE\nfilters:\n  delay:\n    type: Delay\n    parameters:\n      delay: {delay_ms:.16}\n      unit: ms\n      subsample: {subsample}\npipeline:\n  - type: Filter\n    channels: [0]\n    names: [delay]\n");
        let amplitude = 1 << 26;
        let mut input = vec![0_i32; 65_536];
        input[0] = amplitude;
        let rendered = super::conformance::run_optional_pcm_backend_contract(
            "ROOMEQ_CAMILLADSP_BIN", "yaml", &yaml, &input, |_| {},
        ).expect("backend is required");
        for frequency in [100.0, 3079.853052118983, 10_000.0, 20_000.0] {
            let actual: Complex64 = rendered.iter().enumerate().map(|(index, sample)| {
                Complex64::from_polar(*sample as f64 / amplitude as f64,
                    -TAU * frequency * index as f64 / rate)
            }).sum();
            let ideal = Complex64::from_polar(1.0, -TAU * frequency * delay_ms / 1000.0);
            eprintln!("delay diagnostic subsample={subsample}, {frequency} Hz: magnitude {} dB, phase error {} rad, complex error {}",
                20.0 * actual.norm().log10(), (actual / ideal).arg(), (actual - ideal).norm());
        }
    }
}

/// Preserve terminal failure evidence without turning an assertion into success.
/// Process termination cannot unwind; the required wrapper owns timeout evidence.
fn record_backend_row_failure(
    evidence: &mut serde_json::Value,
    save: impl Fn(&serde_json::Value),
    run: impl FnOnce(&mut serde_json::Value),
) {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(evidence)));
    if let Err(payload) = outcome {
        let message = payload.downcast_ref::<String>().map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("non-string backend replay panic");
        evidence["status"] = json!("failed");
        evidence["failure"] = json!({"kind": "backend_replay_assertion_or_setup_failure", "message": message});
        save(evidence);
        std::panic::resume_unwind(payload);
    }
}

#[test]
fn backend_row_failure_preserves_context_and_still_fails() {
    let saved = std::cell::RefCell::new(None);
    let mut evidence = json!({"status": "running", "row": 3, "run_id": "fault-test",
        "comparisons": [{"input": "L", "output": "L", "frequency_count": 49}]});
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        record_backend_row_failure(&mut evidence, |record| {
            saved.replace(Some(record.clone()));
        }, |record| {
            record["active_comparison"] = json!({"input": "R", "output": "SUB1", "frequency_hz": 80.0});
            panic!("injected missing sub route");
        });
    }));
    assert!(outcome.is_err(), "failure must propagate to the required runner");
    let record = saved.into_inner().expect("terminal evidence was saved");
    assert_eq!(record["status"], "failed");
    assert_eq!(record["run_id"], "fault-test");
    assert_eq!(record["row"], 3);
    assert_eq!(record["comparisons"].as_array().unwrap().len(), 1);
    assert_eq!(record["active_comparison"]["frequency_hz"], 80.0);
    assert_eq!(record["failure"]["message"], "injected missing sub route");
}

/// Replay the actual selected optimizer artifacts, not a hand-built export.
/// Explicitly ignored because this requires the completed matrix and backend.
#[test]
#[ignore = "requires ROOMEQ_PARAMETER_MATRIX and ROOMEQ_CAMILLADSP_BIN"]
fn parameter_matrix_backend_complex_transfer() {
    use std::path::Path;
    std::env::var("ROOMEQ_CAMILLADSP_BIN").expect("backend is required");
    let matrix = std::env::var("ROOMEQ_PARAMETER_MATRIX").expect("matrix is required");
    let backend = std::env::var("ROOMEQ_CAMILLADSP_BIN").unwrap();
    let version = std::process::Command::new(&backend).arg("--version").output().unwrap();
    assert!(version.status.success());
    let rows: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(matrix).unwrap()).unwrap();
    assert_eq!(rows.len(), 16, "only a completed matrix can be replayed");
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row["row"].as_u64(), Some(index as u64));
        let bundle = &row["replay_bundle"];
        let directory = Path::new(bundle["directory"].as_str().unwrap());
        let evidence_path = directory.join("backend-complex-transfer.json");
        let save_evidence = |record: &serde_json::Value| {
            let temporary = tempfile::NamedTempFile::new_in(directory).unwrap();
            std::fs::write(temporary.path(), serde_json::to_vec_pretty(record).unwrap()).unwrap();
            temporary.persist(&evidence_path).unwrap();
        };
        let mut evidence = json!({"status": "running", "row": index,
            "run_id": std::env::var("ROOMEQ_BACKEND_RUN_ID").unwrap_or_else(|_| "manual".into()),
            "scope": "sampled_linear_electrical_complex_transfer_not_acoustic_or_clipping_certification",
            "backend": backend, "backend_version": String::from_utf8_lossy(&version.stdout),
            "requested_axes": row["requested_axes"], "unexecuted_axes": row["unexecuted_axes"]});
        save_evidence(&evidence);
        record_backend_row_failure(&mut evidence, &save_evidence, |evidence| {
        evidence["phase"] = json!("load_and_validate_artifacts");
        let graph: DspGraph = serde_json::from_slice(&std::fs::read(
            directory.join(bundle["selected_output"].as_str().unwrap()),
        ).unwrap()).unwrap();
        let rate = row["sample_rate_hz"].as_f64().unwrap();
        let mut irs = HashMap::new();
        for chain in graph.channels.values() {
            assert!(chain.drivers.as_ref().is_none_or(|drivers| drivers.is_empty()),
                "driver graphs require a separate physical-output oracle");
            for plugin in &chain.plugins {
                if plugin.plugin_type == "convolution" {
                    let name = plugin.parameters["ir_file"].as_str().unwrap();
                    let path = directory.join(name).canonicalize().unwrap();
                    assert!(path.starts_with(directory.canonicalize().unwrap()));
                    let mut reader = hound::WavReader::open(path).unwrap();
                    assert_eq!(reader.spec().sample_rate, rate as u32);
                    assert_eq!(reader.spec().channels, 1);
                    let taps: Vec<f64> = reader.samples::<f32>()
                        .map(|sample| sample.unwrap() as f64).collect();
                    assert!(!taps.is_empty() && taps.iter().all(|tap| tap.is_finite()));
                    irs.insert(name.to_owned(), taps);
                }
            }
        }
        let frequencies: Vec<f64> = (0..49)
            .map(|bin| 20.0 * 1000.0_f64.powf(bin as f64 / 48.0)).collect();
        let routing = graph.metadata.as_ref().and_then(|m| m.bass_management.as_ref())
            .and_then(|b| b.routing_graph.as_ref()).filter(|r| !r.routes.is_empty());
        let (inputs, outputs, expected) = if let Some(routing) = routing {
            (routing.input_channels.clone(), routing.output_channels.clone(),
                reference_transfer_with_delay_model(&graph, rate, &frequencies, &irs, false))
        } else {
            let mut names: Vec<String> = graph.channels.keys().cloned().collect();
            names.sort();
            let transfer = names.iter().map(|destination| {
                let by_source = names.iter().map(|source| {
                    let response = frequencies.iter().map(|frequency| {
                        if source == destination {
                            graph.channels[destination].plugins.iter().map(|plugin|
                                plugin_response(plugin, rate, *frequency, &irs)).product()
                        } else { Complex64::new(0.0, 0.0) }
                    }).collect::<Vec<_>>();
                    (source.clone(), response)
                }).collect::<HashMap<_, _>>();
                (destination.clone(), by_source)
            }).collect::<HashMap<_, _>>();
            (names.clone(), names, transfer)
        };
        let yaml = render_dsp_chain(&graph, ExportFormat::CamillaDsp, rate).unwrap();
        let common_padding: f64 = yaml.lines().find_map(|line|
            line.strip_prefix("# roomeq_common_delay_padding_samples: "))
            .expect("export must declare its common causal latency").parse().unwrap();
        assert_eq!(common_padding, reference_padding_samples(&graph, rate));
        let padding_to_apply = if routing.is_some() { 0.0 } else { common_padding };
        let mut hashes = serde_json::Map::new();
        for filename in irs.keys().map(String::as_str).chain([
            bundle["selected_output"].as_str().unwrap(), bundle["request"].as_str().unwrap()]) {
            hashes.insert(filename.to_owned(), json!(crate::hash::sha256_hex(&std::fs::read(directory.join(filename)).unwrap())));
        }
        evidence["artifact_sha256"] = json!(hashes);
        evidence["yaml_sha256"] = json!(crate::hash::sha256_hex(yaml.as_bytes()));
        std::fs::write(directory.join("backend-camilladsp.yaml"), &yaml).unwrap();
        evidence["sample_rate_hz"] = json!(rate);
        evidence["common_padding_samples"] = json!(common_padding);
        evidence["frequencies_hz"] = json!(frequencies);
        evidence["input_channels"] = json!(inputs);
        evidence["output_channels"] = json!(outputs);
        save_evidence(&evidence);
        let frames = 131_072;
        let amplitude = 1 << 26;
        evidence["frames_per_input"] = json!(frames);
        evidence["impulse_amplitude_s32"] = json!(amplitude);
        evidence["absolute_error_allowance"] = json!(0.0001);
        evidence["relative_error_allowance"] = json!(0.01);
        let mut max_error = 0.0_f64;
        let mut comparisons = Vec::new();
        for (source_index, source) in inputs.iter().enumerate() {
            evidence["phase"] = json!("render_backend_input");
            evidence["active_comparison"] = json!({"input": source});
            let mut input = vec![0_i32; frames * inputs.len()];
            input[source_index] = amplitude;
            let rendered = super::conformance::run_optional_pcm_backend_contract(
                "ROOMEQ_CAMILLADSP_BIN", "yaml", &yaml, &input, |dir| {
                    for name in irs.keys() {
                        std::fs::copy(directory.join(name), dir.join(name)).unwrap();
                    }
                },
            ).expect("required backend did not execute");
            assert!(rendered.len() >= frames * outputs.len());
            for (output_index, destination) in outputs.iter().enumerate() {
                let mut path_max_error = 0.0_f64;
                let mut complex_samples = Vec::with_capacity(frequencies.len());
                for (bin, frequency) in frequencies.iter().enumerate() {
                    evidence["phase"] = json!("compare_complex_transfer");
                    evidence["active_comparison"] = json!({"input": source, "output": destination, "frequency_hz": frequency});
                    let step = Complex64::from_polar(1.0, -TAU * frequency / rate);
                    let mut phase = Complex64::new(1.0, 0.0);
                    let mut actual = Complex64::new(0.0, 0.0);
                    for frame in rendered.chunks_exact(outputs.len()).take(frames) {
                        actual += phase * (frame[output_index] as f64 / amplitude as f64);
                        phase *= step;
                    }
                    let target = expected[destination][source][bin]
                        * Complex64::from_polar(1.0, -TAU * frequency * padding_to_apply / rate);
                    let error = (actual - target).norm();
                    evidence["active_comparison"]["actual_complex"] = json!([actual.re, actual.im]);
                    evidence["active_comparison"]["expected_complex"] = json!([target.re, target.im]);
                    evidence["active_comparison"]["absolute_complex_error"] = json!(error);
                    max_error = max_error.max(error);
                    path_max_error = path_max_error.max(error);
                    complex_samples.push(json!({"actual": [actual.re, actual.im], "expected": [target.re, target.im]}));
                    assert!(error <= 0.0001 + 0.01 * target.norm(),
                        "row {index} {source}->{destination} at {frequency} Hz: actual {actual}, expected {target}, error {error}");
                }
                comparisons.push(json!({"input": source, "output": destination,
                    "max_absolute_complex_error": path_max_error, "frequency_count": frequencies.len(), "complex_samples": complex_samples}));
                evidence["comparisons"] = json!(comparisons);
            }
        }
        evidence["status"] = json!("passed");
        evidence["phase"] = json!("complete");
        evidence.as_object_mut().unwrap().remove("active_comparison");
        evidence["comparisons"] = json!(comparisons);
        evidence["frames_per_input"] = json!(frames);
        evidence["impulse_amplitude_s32"] = json!(amplitude);
        evidence["absolute_error_allowance"] = json!(0.0001);
        evidence["relative_error_allowance"] = json!(0.01);
        save_evidence(&evidence);
        eprintln!("matrix backend row {index}: {} inputs, {} outputs, {rate} Hz, max complex error {max_error}", inputs.len(), outputs.len());
        });
    }
}

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
            staged_chain(sub, Vec::new(), vec![eq_filter("peak", 50.0, 1.0, -2.0)]),
        );
        channels
            .get_mut(sub)
            .unwrap()
            .plugins
            .push(PluginConfigWrapper {
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
                selected_low_pass_hz: None,
            }],
            headroom_simulation: None,
            advisory: "ok".to_string(),
        }),
        timing_diagnostics: None,
        ctc: None,
        perceptual_policy: None,
        bootstrap_uncertainty: None,
        validation_bundle: None,
        final_convolution_sha256: None,
        supporting_source: None,
        correction_acceptance: None,
        optimizer_evidence: None,
        stage_outcomes: Vec::new(),
        qa_seed_distribution: None,
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
                let filter_type =
                    parse_biquad_filter_type(filter.get("filter_type").unwrap().as_str().unwrap())
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

/// Independent analog Butterworth pole pairs, doubled for an LR branch.
fn oracle_crossover_q(order: usize, linkwitz_riley: bool) -> Vec<f64> {
    let butterworth_order = if linkwitz_riley { order / 2 } else { order };
    assert!(butterworth_order >= 2 && butterworth_order % 2 == 0);
    let mut q: Vec<_> = (0..butterworth_order / 2).map(|index| {
        1.0 / (2.0 * (((2 * index + 1) as f64 * std::f64::consts::PI)
            / (2 * butterworth_order) as f64).cos())
    }).collect();
    if linkwitz_riley { q.extend(q.clone()); }
    q
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
    // LR branches square a half-order Butterworth response; Butterworth
    // branches use the full order without duplicating the pole pairs.
    let (order, linkwitz_riley) = match crossover_type.to_ascii_lowercase().as_str() {
        "lr24" | "lr4" => (4, true),
        "lr48" | "lr8" => (8, true),
        "butterworth24" | "bw24" => (4, false),
        other => panic!("fixture uses unsupported crossover '{other}'"),
    };
    oracle_crossover_q(order, linkwitz_riley).into_iter()
        .map(|q| {
            let filter_type = if lowpass {
                BiquadFilterType::Lowpass
            } else {
                BiquadFilterType::Highpass
            };
            Biquad::new(filter_type, frequency_hz, sample_rate, q, 0.0)
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

/// Independent support calculation for the specified 129-tap delay kernel.
/// Every parallel stage includes its zero-delay branches. Do not read the
/// export's latency declaration to determine the expected padding.
fn reference_padding_samples(graph: &DspGraph, rate: f64) -> f64 {
    let stage_padding = |delays: Vec<f64>| delays.into_iter().map(|ms| {
        let samples = ms * rate / 1000.0;
        if (samples - samples.round()).abs() <= 1e-9 { 0.0 }
        else { (64.0 - samples.floor()).max(0.0) }
    }).fold(0.0_f64, f64::max);
    let chain_delay = |name: &String, stage: Option<&str>| graph.channels[name].plugins.iter()
        .filter(|p| p.plugin_type == "delay" && stage.is_none_or(|s|
            p.parameters["room_eq_stage"].as_str() == Some(s)))
        .map(|p| p.parameters["delay_ms"].as_f64().unwrap()).sum::<f64>();
    if let Some(routing) = graph.metadata.as_ref().and_then(|m| m.bass_management.as_ref())
        .and_then(|b| b.routing_graph.as_ref()).filter(|r| !r.routes.is_empty()) {
        stage_padding(routing.input_channels.iter().map(|name| chain_delay(name, Some("pre_route"))).collect())
            + stage_padding(routing.routes.iter().map(|r| r.delay_ms).collect())
            + stage_padding(routing.output_channels.iter().map(|name| chain_delay(name, Some("post_route"))).collect())
    } else {
        stage_padding(graph.channels.keys().map(|name| chain_delay(name, None)).collect())
    }
}

/// Reference graph transfer plus independently calculated common backend latency.
fn reference_transfer(
    graph: &DspGraph,
    sample_rate: f64,
    frequencies: &[f64],
    ir_registry: &HashMap<String, Vec<f64>>,
) -> HashMap<String, HashMap<String, Vec<Complex64>>> {
    reference_transfer_with_delay_model(graph, sample_rate, frequencies, ir_registry, true)
}

/// Shape of the specified finite Blackman-windowed sinc relative to an ideal
/// delay. Independently evaluated here, without reading exported coefficients
/// or calling the production kernel. Integer delay and common support cancel
/// in this ratio. Ideal-delay acceptance remains a separate backend test.
fn reference_delay_shape(delay_ms: f64, rate: f64, frequency: f64) -> Complex64 {
    let samples = delay_ms * rate / 1000.0;
    if (samples - samples.round()).abs() <= 1e-9 {
        return Complex64::new(1.0, 0.0);
    }
    let center = 64.0 + samples - samples.floor();
    let mut dc = 0.0;
    let mut response = Complex64::new(0.0, 0.0);
    for index in 0..129 {
        let distance = index as f64 - center;
        let angle = std::f64::consts::PI * distance;
        let coefficient = if distance.abs() > 64.0 { 0.0 } else {
            (angle.sin() / angle) * (0.42 + 0.5 * (angle / 64.0).cos()
                + 0.08 * (angle / 32.0).cos())
        };
        dc += coefficient;
        response += Complex64::from_polar(coefficient, -TAU * frequency * index as f64 / rate);
    }
    response / dc / Complex64::from_polar(1.0, -TAU * frequency * center / rate)
}

fn reference_transfer_with_delay_model(
    graph: &DspGraph,
    sample_rate: f64,
    frequencies: &[f64],
    ir_registry: &HashMap<String, Vec<f64>>,
    finite_delay: bool,
) -> HashMap<String, HashMap<String, Vec<Complex64>>> {
    let shape = |delay, frequency| if finite_delay {
        reference_delay_shape(delay, sample_rate, frequency)
    } else { Complex64::new(1.0, 0.0) };
    let chain_delay = |name: &String, stage: &str| graph.channels[name].plugins.iter()
        .filter(|p| p.plugin_type == "delay" && p.parameters["room_eq_stage"].as_str() == Some(stage))
        .map(|p| p.parameters["delay_ms"].as_f64().unwrap()).sum::<f64>();
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
                            Complex64::new(gain, 0.0) * branch * delay * shape(route.delay_ms, *frequency)
                        })
                        .sum();
                    post[index] * pre[index] * bus
                        * shape(chain_delay(source, "pre_route"), *frequency)
                        * shape(chain_delay(destination, "post_route"), *frequency)
                        * Complex64::from_polar(
                        1.0, -TAU * frequency * reference_padding_samples(graph, sample_rate) / sample_rate)
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
    Gain {
        db: f64,
        inverted: bool,
    },
    DelayMs(f64),
    DelaySamples(f64),
    InlineConv(Vec<f64>),
    Biquad {
        filter_type: BiquadFilterType,
        freq: f64,
        q: f64,
        gain_db: f64,
    },
    Crossover {
        lowpass: bool,
        freq: f64,
        q_values: Vec<f64>,
    },
    Conv {
        file: String,
    },
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
            Self::DelaySamples(samples) => Complex64::from_polar(1.0, -TAU * frequency * samples / sample_rate),
            Self::InlineConv(taps) => fir_response(taps, sample_rate, frequency),
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
                q_values,
            } => q_values.iter()
                .map(|q| {
                    Biquad::new(
                        if *lowpass {
                            BiquadFilterType::Lowpass
                        } else {
                            BiquadFilterType::Highpass
                        },
                        *freq,
                        sample_rate,
                        *q,
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
    value
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("expected number, got '{value}'"))
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
        if !line.starts_with(' ')
            && !line.starts_with('#')
            && line.ends_with(':')
            && !line.is_empty()
        {
            current = Some(line.trim_end_matches(':').to_string());
            sections
                .entry(current.clone().unwrap())
                .or_insert_with(Vec::new);
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
        let is_entry =
            line.starts_with("  ") && !line.starts_with("   ") && line.trim_end().ends_with(':');
        if !is_entry {
            index += 1;
            continue;
        }
        let name = line.trim().trim_end_matches(':').to_string();
        index += 1;
        let mut node_type = String::new();
        let mut params: HashMap<String, String> = HashMap::new();
        let mut inline_values = Vec::new();
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
            } else if let Some(value) = entry.strip_prefix("- ") {
                inline_values.push(parse_number(value));
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
                match params["unit"].as_str() {
                    "ms" => RealizedFilter::DelayMs(parse_number(&params["delay"])),
                    "samples" => RealizedFilter::DelaySamples(parse_number(&params["delay"])),
                    unit => panic!("unsupported delay unit {unit}"),
                }
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
                    q_values: oracle_crossover_q(order, params["type"].starts_with("LinkwitzRiley")),
                }
            }
            "Conv" if params.get("type").is_some_and(|kind| kind == "Values") => {
                assert!(!inline_values.is_empty());
                RealizedFilter::InlineConv(inline_values)
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
            current_source
                .as_mut()
                .expect("mixer gain without source")
                .gain_db = parse_number(gain);
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
                            .unwrap_or_else(|| {
                                panic!("pipeline references unknown filter '{name}'")
                            })
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
        assert_eq!(
            *dest, route_index,
            "expand mixer must list one bus per route"
        );
        assert_eq!(
            sources.len(),
            1,
            "expand mixer must map one source per route"
        );
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

    let pre_response: Vec<Vec<Complex64>> = pre.iter().map(|names| chain(names)).collect();
    let route_response: Vec<Vec<Complex64>> = route.iter().map(|names| chain(names)).collect();
    let post_response: Vec<Vec<Complex64>> = post.iter().map(|names| chain(names)).collect();

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
                    post_response[dest_index][index] * pre_response[source_index][index] * bus
                })
                .collect();
            inputs.insert(source.clone(), response);
        }
        transfer.insert(destination.clone(), inputs);
    }
    RealizedDocument {
        transfer,
        pre: pre_response,
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
    assert_eq!(
        reference.len(),
        realized.len(),
        "{context}: output channel set"
    );
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

/// Rendered multi-sub fixture with its IR registry and channel order.
type RoutedFixture = (
    DspGraph,
    HashMap<String, Vec<f64>>,
    String,
    Vec<String>,
    Vec<String>,
);

fn render_routed(sub_count: usize, sample_rate: f64) -> RoutedFixture {
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
    let input_channels = routing.input_channels.clone();
    let output_channels = routing.output_channels.clone();
    (graph, registry, yaml, input_channels, output_channels)
}

#[test]
fn multisub_routed_transfer_matches_canonical() {
    // Full 2/4/8-sub fixtures, optimization-shaped (staged plugins, routing
    // graph, single-sub report metadata), at both export sample rates:
    // dense-grid complex comparison (primary-seat semantics) of the
    // reconstructed preset against the canonical graph.
    for sub_count in [2, 4, 8] {
        for sample_rate in [44_100.0, 48_000.0] {
            let (graph, registry, yaml, inputs, outputs) = render_routed(sub_count, sample_rate);
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
fn electrical_headroom_matches_independently_reparsed_camilladsp_outputs() {
    use roomeq_workflow::electrical_headroom::{
        SerializedElectricalPath, canonical_electrical_routing, expand_routed_electrical_paths,
        replay_sampled_electrical_headroom,
    };
    use std::collections::BTreeMap;
    for sub_count in [2, 4, 8] {
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            let (graph, irs, yaml, inputs, outputs) = render_routed(sub_count, sample_rate);
            let frequencies = log_grid();
            let document =
                realized_transfer(&yaml, sample_rate, &frequencies, &inputs, &outputs, &irs);
            let routing = canonical_electrical_routing(&graph).unwrap().unwrap();
            let expanded = expand_routed_electrical_paths(&graph.channels, routing).unwrap();
            let stages: Vec<Vec<_>> = expanded.iter().map(|p| p.stages.iter().collect()).collect();
            let paths: Vec<_> = expanded
                .iter()
                .zip(&stages)
                .map(|(p, stages)| SerializedElectricalPath {
                    input: &p.input,
                    output: &p.output,
                    stages,
                })
                .collect();
            let limits: BTreeMap<_, _> = inputs
                .iter()
                .enumerate()
                .map(|(index, name)| (name.clone(), if index == 0 { 0.5 } else { 1.0 }))
                .collect();
            let assessed = replay_sampled_electrical_headroom(
                &paths,
                &frequencies,
                sample_rate,
                &limits,
                std::path::Path::new("."),
                &irs,
            )
            .unwrap();
            assert_eq!(assessed.len(), outputs.len());
            for peak in assessed {
                // Independent oracle: walk the emitted YAML mixer/filter
                // pipeline above, then form the permitted independent-input
                // magnitude sum here. No production headroom reducer is used.
                let transfers = &document.transfer[&peak.output];
                let expected_peak = (0..frequencies.len())
                    .map(|index| {
                        inputs
                            .iter()
                            .map(|input| transfers[input][index].norm() * limits[input])
                            .sum::<f64>()
                    })
                    .fold(0.0_f64, f64::max);
                let expected_db = (20.0 * expected_peak.log10()).max(0.0);
                assert!(
                    (peak.required_attenuation_db - expected_db).abs() < 0.01,
                    "{sub_count} subs at {sample_rate} Hz, {}: assessment {} dB vs emitted document {expected_db} dB",
                    peak.output,
                    peak.required_attenuation_db
                );
                assert!(
                    (peak.peak_amplitude - expected_peak).abs() < 1e-3,
                    "{} electrical amplitude differs from emitted document",
                    peak.output
                );
            }
        }
    }
}

#[test]
fn tool_contract_camilladsp_multisub_coherent_peak_at_all_rates() {
    if std::env::var("ROOMEQ_CAMILLADSP_BIN").is_err() {
        eprintln!("set ROOMEQ_CAMILLADSP_BIN for actual multi-sub PCM replay");
        return;
    }
    for sub_count in [2, 4, 8] {
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let (_graph, irs, yaml, inputs, outputs) = render_routed(sub_count, rate);
            let frequencies = log_grid();
            let predicted = realized_transfer(&yaml, rate, &frequencies, &inputs, &outputs, &irs);
            let physical_subs: Vec<_> = outputs.iter().enumerate().filter(|(_, name)| name.starts_with("SUB")).collect();
            assert_eq!(physical_subs.len(), sub_count);
            for (output_index, target) in physical_subs {
            let (bin, expected) = (0..frequencies.len()).map(|bin| {
                (bin, inputs.iter().map(|input| predicted.transfer[target][input][bin].norm()).sum::<f64>())
            }).max_by(|left, right| left.1.total_cmp(&right.1)).unwrap();
            let frequency = frequencies[bin];
            let frames = 65_536;
            let scale = 0.005;
            let mut samples = Vec::with_capacity(frames * inputs.len());
            for frame in 0..frames {
                for input in &inputs {
                    let phase = predicted.transfer[target][input][bin].arg();
                    let value = scale * (std::f64::consts::TAU * frequency * frame as f64 / rate - phase).sin();
                    samples.push((value * i32::MAX as f64).round() as i32);
                }
            }
            let rendered = super::conformance::run_optional_pcm_backend_contract(
                "ROOMEQ_CAMILLADSP_BIN", "yaml", &yaml, &samples, |dir| {
                    for (name, coefficients) in &irs {
                        let spec = hound::WavSpec { channels: 1, sample_rate: rate as u32,
                            bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
                        let mut writer = hound::WavWriter::create(dir.join(name), spec).unwrap();
                        for coefficient in coefficients { writer.write_sample(*coefficient as f32).unwrap(); }
                        writer.finalize().unwrap();
                    }
                },
            ).expect("explicitly enabled backend must execute");
            assert!(rendered.len() >= frames * outputs.len());
            let peak = rendered.chunks_exact(outputs.len()).skip(frames / 2).take(frames / 2)
                .map(|frame| frame[output_index].unsigned_abs() as f64 / i32::MAX as f64)
                .fold(0.0_f64, f64::max);
            let error_db = 20.0 * (peak / (scale * expected)).log10();
            eprintln!("CamillaDSP: {sub_count} subs, {rate} Hz, {target}, {frequency} Hz tone: peak error {error_db:.6} dB");
            assert!(error_db.abs() < 0.05,
                "{sub_count} subs at {rate} Hz, {target}: actual {peak}, predicted {}, error {error_db} dB", scale * expected);
            }
        }
    }
}

#[test]
fn driver_owned_low_pass_exports_without_an_extra_redirected_filter() {
    let (mut graph, registry) = multisub_fixture(2);
    let frequencies = log_grid();
    let sample_rate = 48_000.0;
    let original = reference_transfer(&graph, sample_rate, &frequencies, &registry);
    let routing = graph
        .metadata
        .as_mut()
        .unwrap()
        .bass_management
        .as_mut()
        .unwrap()
        .routing_graph
        .as_mut()
        .unwrap();
    for route in &mut routing.routes {
        if route.route_kind == "redirected_bass_lowpass_to_sub" {
            route.low_pass_hz = None;
        }
    }
    let inputs = routing.input_channels.clone();
    let outputs = routing.output_channels.clone();
    for (name, kind, frequency) in [("SUB1", "LR24", 80.0), ("SUB2", "LR48", 60.0)] {
        graph.channels.get_mut(name).unwrap().plugins.push(PluginConfigWrapper {
            plugin_type: "crossover".into(),
            parameters: json!({"type": kind, "frequency": frequency, "output": "low", "room_eq_stage": "post_route"}),
        });
    }
    let yaml = render_dsp_chain(&graph, ExportFormat::CamillaDsp, sample_rate).unwrap();
    let realized = realized_transfer(
        &yaml,
        sample_rate,
        &frequencies,
        &inputs,
        &outputs,
        &registry,
    );
    let mut expected = original;
    let mut actual = realized.transfer;
    // LFE has its independent route cutoff plus the physical driver's LP.
    // Redirected L/R must retain exactly one low-pass transfer per driver.
    for by_source in expected.values_mut().chain(actual.values_mut()) {
        by_source.retain(|source, _| source == "L" || source == "R");
    }
    assert_transfer_matches(&expected, &actual, &frequencies, "driver-owned low-pass");
}

/// Closed-form Linkwitz-Riley low-pass magnitude, independent of the exporter.
///
/// An order-N LR branch squares an order-N/2 Butterworth magnitude:
/// `1 / (1 + x^N)` with `x = f/fc`. Both LR24 and LR48 are exactly 0.5
/// (-6.02 dB) at fc. LR48 has distinct Butterworth pole-pair Q values;
/// four identical Q=1/sqrt(2) sections are not an LR48 crossover.
/// A bilinear-transformed biquad matches this closed form up to frequency
/// warping (~2e-6 off-center here), so the anchor tolerance is 1e-5: still
/// orders of magnitude below any transcription defect (wrong order, type,
/// or a 1% fc shift moves these anchors by >1e-2).
fn analytic_lr_lowpass(crossover_type: &str, frequency: f64, fc: f64) -> f64 {
    let order = match crossover_type {
        "LR24" => 4,
        "LR48" => 8,
        other => panic!("anchor covers LR24/LR48, not '{other}'"),
    };
    let x = frequency / fc;
    1.0 / (1.0 + x.powi(order))
}

#[test]
fn multisub_crossover_anchor_holds_across_sample_rates() {
    // Bass-crossover SRC angle: each LR branch must reproduce its analytic
    // magnitude at fc/2, fc, and 2fc at every export rate, proving frequency,
    // order, and low/high transcription (not just shape). SUB1 hangs off the
    // LR24 80 Hz branch, SUB2 off the LR48 60 Hz one.
    for sample_rate in [44_100.0, 48_000.0] {
        let (graph, registry, yaml, inputs, outputs) = render_routed(4, sample_rate);
        let probes = [("SUB1", "L", "LR24", 80.0), ("SUB2", "L", "LR48", 60.0)];
        let frequencies: Vec<f64> = probes
            .iter()
            .flat_map(|(_, _, _, fc)| [*fc / 2.0, *fc, *fc * 2.0])
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
        // Redirected-leg gain from the fixture, recomputed from its dB
        // value so the anchor does not depend on how the route stores it.
        let redirected_gain = 10.0_f64.powf(-6.0206 / 20.0);
        for (destination, source, crossover_type, fc) in probes {
            let dest_index = outputs.iter().position(|name| name == destination).unwrap();
            let source_index = inputs.iter().position(|name| name == source).unwrap();
            let response = &realized.transfer[destination][source];
            for frequency in [fc / 2.0, fc, fc * 2.0] {
                let index = frequencies
                    .iter()
                    .position(|candidate| candidate == &frequency)
                    .unwrap();
                // Divide out the pre/post chains, the redirected-leg gain,
                // and the route delay (|delay| == 1): what remains is the
                // crossover branch magnitude alone.
                let measured = response[index].norm()
                    / (realized.pre[source_index][index].norm()
                        * realized.post[dest_index][index].norm()
                        * redirected_gain);
                let expected = analytic_lr_lowpass(crossover_type, frequency, fc);
                assert!(
                    (measured - expected).abs() < 1e-5,
                    "{destination} {crossover_type} LP at {frequency} Hz @{sample_rate}Hz: \
                     {measured} vs analytic {expected}"
                );
            }
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
    let mut peak_phase = 0.0f64;
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
    // Route-matrix transcription: dividing out the pre AND post chains
    // isolates the pure route-bus legs (gain, signed polarity, crossover,
    // delay) per input. At route level SUB1's legs oppose (its R route
    // inverts) while SUB2's legs agree; end to end, SUB1 still sums
    // constructively because its R pre-chain inverts too (double inversion).
    // Dividing out only the pre chains cannot work: it removes the R invert
    // that makes SUB1 coherent, while keeping the pre chains keeps their
    // 1.224 ms delay skew, which alone destroys >0.99 coherence above ~20 Hz.
    let sample_rate = 48_000.0;
    let (_graph, _registry, yaml, inputs, outputs) = render_routed(2, sample_rate);
    let frequencies: Vec<f64> = (0..24)
        .map(|index| 10.0 * 20.0_f64.powf(index as f64 / 23.0))
        .collect();
    let registry = HashMap::from([(SUB_IR_FILE.to_string(), SUB_IR_TAPS.to_vec())]);
    let realized = realized_transfer(
        &yaml,
        sample_rate,
        &frequencies,
        &inputs,
        &outputs,
        &registry,
    );
    // Pure route-bus legs per destination: transfer with both chain stages
    // divided out. Both legs of one sub share the crossover branch and the
    // redirected gain; only signed polarity and the 12 us route-delay skew
    // distinguish them.
    let bus = |destination: &str| {
        let dest_index = outputs.iter().position(|name| name == destination).unwrap();
        frequencies
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let legs = ["L", "R"].map(|source| {
                    let source_index = inputs.iter().position(|name| name == source).unwrap();
                    realized.transfer[destination][source][index]
                        / (realized.pre[source_index][index] * realized.post[dest_index][index])
                });
                (legs[0], legs[1])
            })
            .collect::<Vec<_>>()
    };
    for (index, frequency) in frequencies.iter().enumerate() {
        // SUB1 route legs oppose through signed polarity: the residual is
        // bounded by the 12 us route-delay skew (~0.0075 at 200 Hz).
        let (l, r) = bus("SUB1")[index];
        let residual = (l + r).norm() / (l.norm() + r.norm());
        assert!(
            residual < 0.02,
            "SUB1 route legs must cancel through signed polarity at {frequency} Hz, residual {residual}"
        );
        // SUB2 route legs agree: same bound, constructive.
        let (l, r) = bus("SUB2")[index];
        let coherent = (l + r).norm() / (l.norm() + r.norm());
        assert!(
            coherent > 0.999,
            "SUB2 route legs must sum coherently at {frequency} Hz, ratio {coherent}"
        );
    }
    // Headroom basis on the agreeing SUB2 route legs at 10 Hz: the coherent
    // peak gain equals the algebraic sum of the legs, +6.02 dB over one leg.
    let (l, r) = bus("SUB2")[0];
    let peak = (l + r).norm();
    let leg = l.norm();
    assert!(
        (peak / leg - 2.0).abs() < 1e-6,
        "coherent peak {peak} must equal twice one leg {leg}"
    );
    // End-to-end composition at 10 Hz: SUB1's full-chain legs (pre chains
    // included) sum constructively — the R pre-chain invert composes with
    // the R route invert. Analytic ratio is ~1.9986 (1.212 ms residual
    // skew); anything below ~1.9 would mean a dropped inversion.
    let full_l = realized.transfer["SUB1"]["L"][0];
    let full_r = realized.transfer["SUB1"]["R"][0];
    let ratio = (full_l + full_r).norm() / full_l.norm();
    assert!(
        ratio > 1.9,
        "SUB1 full-chain legs must compose to coherence at 10 Hz, ratio {ratio}"
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
            let mut sample =
                spectrum[0] + spectrum[points / 2] * if n % 2 == 0 { 1.0 } else { -1.0 };
            for (k, component) in spectrum.iter().enumerate().take(points / 2).skip(1) {
                let angle = TAU * k as f64 * n as f64 / points as f64;
                sample += (*component * Complex64::from_polar(1.0, angle)) * 2.0;
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
    let realized = realized_transfer(
        &yaml,
        sample_rate,
        &frequencies,
        &inputs,
        &outputs,
        &registry,
    )
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
    // The L legs share polarity across subs (they sum to ~1.05x the mean),
    // so this probes the R column instead: the R route inverts on
    // even-index subs while the R pre-chain inverts everywhere, leaving
    // full-chain R legs that alternate +, -, +, ... across the eight subs
    // and nearly cancel in the coherent sum (up to the alternating 80/60 Hz
    // crossover branches).
    let coherent: f64 = subs
        .iter()
        .map(|sub| realized[sub]["R"][0])
        .sum::<Complex64>()
        .norm()
        / subs.len() as f64;
    let mean: f64 = subs
        .iter()
        .map(|sub| realized[sub]["R"][0].norm())
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
    assert!(yaml.contains("type: Values"));
    assert!(!yaml.contains("delay: 1.235"), "must not quantize milliseconds");
    let sections = document_sections(&yaml);
    let filters = parse_filters(&sections["filters"]);
    let registry = HashMap::new();
    let padding = reference_padding_samples(&output, 48_000.0);
    assert_eq!(padding, 5.0);
    for bin in 0..=128 {
        let frequency = 0.46 * 48_000.0 * bin as f64 / 128.0;
        let response = filters["left_delay"].response(48_000.0, frequency, &registry);
        let expected = Complex64::from_polar(1.0,
            -TAU * frequency * (0.001_234_56 + padding / 48_000.0));
        assert!((20.0 * response.norm().log10()).abs() <= 0.01,
            "magnitude contract at {frequency} Hz");
        assert!((response / expected).arg().abs() <= 0.001,
            "phase contract at {frequency} Hz");
    }
}

#[test]
fn tiny_route_delay_survives_export_and_missing_branch_padding_is_detected() {
    let (mut graph, irs) = multisub_fixture(2);
    for chain in graph.channels.values_mut() {
        chain.plugins.retain(|plugin| plugin.plugin_type != "delay");
    }
    let routing = graph.metadata.as_mut().unwrap().bass_management.as_mut().unwrap()
        .routing_graph.as_mut().unwrap();
    for route in &mut routing.routes { route.delay_ms = 0.0; }
    let index = routing.routes.iter().position(|route|
        route.source_channel == "L" && route.destination == "SUB1").unwrap();
    // The former route writer silently omitted delays <= 0.001 ms.
    routing.routes[index].delay_ms = 0.0005;
    let inputs = routing.input_channels.clone();
    let outputs = routing.output_channels.clone();
    let rate = 48_000.0;
    let frequencies = [100.0, 1000.0];
    let expected = reference_transfer(&graph, rate, &frequencies, &irs);
    let report = crate::camilladsp_delay_realization(&graph, rate).unwrap();
    assert_eq!(report.route_padding_samples, 64);
    assert_eq!(report.common_padding_samples, 64);
    let yaml = render_dsp_chain(&graph, ExportFormat::CamillaDsp, rate).unwrap();
    let actual = realized_transfer(&yaml, rate, &frequencies, &inputs, &outputs, &irs);
    assert_transfer_matches(&expected, &actual.transfer, &frequencies, "tiny route delay");
    let name = format!("  - route_{index}_L_to_SUB1_delay\n");
    assert_eq!(yaml.matches(&name).count(), 1);
    let omitted = yaml.replace(&name, "");
    let faulty = realized_transfer(&omitted, rate, &frequencies, &inputs, &outputs, &irs);
    let reference = expected["SUB1"]["L"][0];
    let error = (faulty.transfer["SUB1"]["L"][0] - reference).norm() / reference.norm();
    assert!(error > 0.1, "omitting one branch's support must be detected: {error}");
}

#[test]
fn camilladsp_rejects_shared_global_eq() {
    // Shared/global EQ has no preset stage: it must error, never silently
    // drop, so a partial preset cannot read as complete. The fixture is
    // deliberately non-routed (no bass-management graph): a routed graph
    // reaches the routed-graph gate first, which is a different rejection.
    // Two layers are asserted: the public render entry (generic preserved-
    // DSP error) and the specific global-plugin conformance gate.
    use super::super::conformance::validate_camilladsp_input;
    let graph = DspGraph {
        deployed_source_curves: Default::default(),
        version: "1.3.0".to_string(),
        global_plugins: vec![PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: json!({"filters": [{
                "filter_type": "peak", "freq": 1000.0, "q": 1.0, "db_gain": -1.0,
            }]}),
        }],
        channels: HashMap::from([(
            "left".to_string(),
            ChannelDspChain {
                channel: "left".to_string(),
                plugins: vec![PluginConfigWrapper {
                    plugin_type: "gain".to_string(),
                    parameters: json!({"gain_db": 0.0}),
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
    let error = render_dsp_chain(&graph, ExportFormat::CamillaDsp, 48_000.0)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("global_plugins"),
        "shared global EQ must fail loudly at the render entry, got: {error}"
    );
    let gate_error = validate_camilladsp_input(&graph, Some(48_000.0))
        .unwrap_err()
        .to_string();
    assert!(
        gate_error.contains("does not support global plugin #0 ('eq')"),
        "unexpected gate error: {gate_error}"
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

#[allow(clippy::too_many_arguments)]
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
        high_pass_hz,
        low_pass_hz,
        gain_db: 0.0,
        gain_linear: 1.0,
        matrix_gain: 1.0,
        delay_ms,
        polarity_inverted: false,
    }
}

#[allow(clippy::too_many_arguments)]
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
