use super::super::export_format::ExportFormat;
use super::super::render_dsp_chain;
use super::make::{make_routed_bass_output, make_test_output};
use crate::roomeq::types::{ChannelDspChain, DspChainOutput, PluginConfigWrapper};
use serde_json::json;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

fn channel(name: &str, plugins: Vec<PluginConfigWrapper>) -> ChannelDspChain {
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

fn output_with_plugins(plugins: Vec<PluginConfigWrapper>) -> DspChainOutput {
    DspChainOutput {
        version: "1.3.0".to_string(),
        global_plugins: Vec::new(),
        channels: HashMap::from([("left".to_string(), channel("left", plugins))]),
        metadata: None,
    }
}

fn camilladsp_error(output: &DspChainOutput) -> String {
    render_dsp_chain(output, ExportFormat::CamillaDsp, 48_000.0)
        .unwrap_err()
        .to_string()
}

/// Run a backend CLI against raw interleaved S32_LE PCM. The command contract
/// is deliberately backend-neutral: the executable receives the generated
/// config path and communicates PCM through stdin/stdout.
fn run_optional_pcm_backend_contract<F>(
    binary_env: &str,
    extension: &str,
    config: &str,
    input: &[i32],
    prepare_sidecars: F,
) -> Option<Vec<i32>>
where
    F: FnOnce(&Path),
{
    let binary = match std::env::var(binary_env) {
        Ok(binary) if !binary.trim().is_empty() => binary,
        _ => {
            eprintln!("skipping optional PCM backend contract; set {binary_env} to enable it");
            return None;
        }
    };

    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join(format!("room_eq.{extension}"));
    std::fs::write(&config_path, config).unwrap();
    prepare_sidecars(dir.path());

    let mut child = Command::new(&binary)
        .arg(&config_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("failed to run PCM backend '{binary}': {error}"));
    {
        let stdin = child.stdin.as_mut().unwrap();
        for sample in input {
            stdin.write_all(&sample.to_le_bytes()).unwrap();
        }
    }
    drop(child.stdin.take());

    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "PCM backend failed: {binary}\nstatus: {}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stdout.len() % 4,
        0,
        "PCM backend produced a partial S32_LE sample"
    );
    Some(
        output
            .stdout
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
    )
}

#[test]
fn generic_renderer_accepts_supported_camilladsp_fixture_in_source_order() {
    let yaml = render_dsp_chain(&make_test_output(), ExportFormat::CamillaDsp, 48_000.0)
        .expect("supported CamillaDSP fixture should pass conformance validation");

    assert!(yaml.contains(
        "channels:\n  - 0\n  names:\n  - left_gain\n  - left_delay\n  - left_peq_0\n  - left_peq_1\n  - left_peq_2"
    ));
}

#[test]
fn tool_contract_camilladsp_pcm_preserves_polarity_and_delay() {
    let output = output_with_plugins(vec![
        PluginConfigWrapper {
            plugin_type: "gain".to_string(),
            parameters: json!({"gain_db": 0.0, "invert": true}),
        },
        PluginConfigWrapper {
            plugin_type: "delay".to_string(),
            parameters: json!({"delay_ms": 1.0}),
        },
    ]);
    let config = render_dsp_chain(&output, ExportFormat::CamillaDsp, 48_000.0).unwrap();
    let mut input = vec![0_i32; 8192];
    let impulse_index = 128;
    let amplitude = 1 << 28;
    input[impulse_index] = amplitude;

    let Some(rendered) =
        run_optional_pcm_backend_contract("ROOMEQ_CAMILLADSP_BIN", "yaml", &config, &input, |_| {})
    else {
        return;
    };

    let (peak_index, peak) = rendered
        .iter()
        .copied()
        .enumerate()
        .max_by_key(|(_, sample)| sample.unsigned_abs())
        .unwrap();
    assert_eq!(peak_index, impulse_index + 48);
    assert!((peak + amplitude).abs() <= 2, "unexpected peak {peak}");
}

#[test]
fn tool_contract_camilladsp_pcm_preserves_routed_channel_matrix() {
    let mut output = make_routed_bass_output();
    for chain in output.channels.values_mut() {
        chain.plugins.clear();
    }
    let routes = &mut output
        .metadata
        .as_mut()
        .unwrap()
        .bass_management
        .as_mut()
        .unwrap()
        .routing_graph
        .as_mut()
        .unwrap()
        .routes;
    for route in routes.iter_mut() {
        route.high_pass_hz = None;
        route.low_pass_hz = None;
        route.delay_ms = 0.0;
        route.polarity_inverted = false;
        route.gain_db = 0.0;
        route.gain_linear = 1.0;
        route.matrix_gain = 1.0;
    }
    // L -> LFE is exactly one half in amplitude.
    routes[1].gain_db = -6.020_599_913;
    routes[1].gain_linear = 0.5;
    routes[1].matrix_gain = 0.5;

    let config = render_dsp_chain(&output, ExportFormat::CamillaDsp, 48_000.0).unwrap();
    let frame_count = 8192;
    let channel_count = 3;
    let impulse_frame = 128;
    let amplitude = 1 << 27;
    let mut input = vec![0_i32; frame_count * channel_count];
    input[impulse_frame * channel_count] = amplitude;

    let Some(rendered) =
        run_optional_pcm_backend_contract("ROOMEQ_CAMILLADSP_BIN", "yaml", &config, &input, |_| {})
    else {
        return;
    };

    let frame = &rendered[impulse_frame * channel_count..(impulse_frame + 1) * channel_count];
    assert!((frame[0] - amplitude).abs() <= 2, "unexpected L sample");
    assert_eq!(frame[1], 0, "L input leaked into R output");
    assert!(
        (frame[2] - amplitude / 2).abs() <= 4,
        "unexpected LFE sample {}",
        frame[2]
    );
}

#[test]
fn tool_contract_camilladsp_pcm_matches_peaking_filter_gain() {
    let output = output_with_plugins(vec![PluginConfigWrapper {
        plugin_type: "eq".to_string(),
        parameters: json!({
            "filters": [{
                "filter_type": "peak",
                "freq": 1000.0,
                "q": 1.0,
                "db_gain": 6.0,
            }]
        }),
    }]);
    let config = render_dsp_chain(&output, ExportFormat::CamillaDsp, 48_000.0).unwrap();
    let frame_count = 12_288;
    let amplitude = (1 << 26) as f64;
    let input: Vec<i32> = (0..frame_count)
        .map(|index| {
            let phase = std::f64::consts::TAU * 1000.0 * index as f64 / 48_000.0;
            (amplitude * phase.sin()).round() as i32
        })
        .collect();

    let Some(rendered) =
        run_optional_pcm_backend_contract("ROOMEQ_CAMILLADSP_BIN", "yaml", &config, &input, |_| {})
    else {
        return;
    };

    let settled_input = &input[4096..];
    let settled_output = &rendered[4096..frame_count];
    let input_rms = rms(settled_input);
    let output_rms = rms(settled_output);
    let measured_gain = output_rms / input_rms;
    let expected_gain = 10.0_f64.powf(6.0 / 20.0);
    assert!(
        (measured_gain - expected_gain).abs() < 0.002,
        "expected gain {expected_gain}, measured {measured_gain}"
    );
}

#[test]
fn tool_contract_camilladsp_pcm_matches_linkwitz_riley_crossover_gain() {
    let output = output_with_plugins(vec![PluginConfigWrapper {
        plugin_type: "crossover".to_string(),
        parameters: json!({
            "type": "LR24",
            "frequency": 80.0,
            "output": "high",
        }),
    }]);
    let config = render_dsp_chain(&output, ExportFormat::CamillaDsp, 48_000.0).unwrap();
    let frame_count = 24_576;
    let amplitude = (1 << 27) as f64;
    let input: Vec<i32> = (0..frame_count)
        .map(|index| {
            let phase = std::f64::consts::TAU * 80.0 * index as f64 / 48_000.0;
            (amplitude * phase.sin()).round() as i32
        })
        .collect();

    let Some(rendered) =
        run_optional_pcm_backend_contract("ROOMEQ_CAMILLADSP_BIN", "yaml", &config, &input, |_| {})
    else {
        return;
    };

    let settled_input = &input[12_000..24_000];
    let settled_output = &rendered[12_000..24_000];
    let measured_gain = rms(settled_output) / rms(settled_input);
    assert!(
        (measured_gain - 0.5).abs() < 0.002,
        "expected -6.0206 dB at the LR24 crossover, measured gain {measured_gain}"
    );
}

fn rms(samples: &[i32]) -> f64 {
    (samples
        .iter()
        .map(|sample| (*sample as f64).powi(2))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt()
}

#[test]
fn tool_contract_equalizer_apo_benchmark_processes_real_pcm() {
    let command = match std::env::var("ROOMEQ_EQUALIZER_APO_PCM_CMD") {
        Ok(command) if !command.trim().is_empty() => command,
        _ => {
            eprintln!("skipping Equalizer APO PCM contract; set ROOMEQ_EQUALIZER_APO_PCM_CMD");
            return;
        }
    };
    let output = output_with_plugins(vec![
        PluginConfigWrapper {
            plugin_type: "gain".to_string(),
            parameters: json!({"gain_db": -3.0}),
        },
        PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: json!({"filters": [{
                "filter_type": "peak", "freq": 1000.0, "q": 1.0, "db_gain": 6.0
            }]}),
        },
    ]);
    let config = render_dsp_chain(&output, ExportFormat::EqualizerApo, 48_000.0).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.txt");
    let input_path = dir.path().join("input.wav");
    let output_path = dir.path().join("output.wav");
    std::fs::write(&config_path, config).unwrap();

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&input_path, spec).unwrap();
    let amplitude = 0.05_f32;
    for index in 0..16_384 {
        let phase = std::f32::consts::TAU * 1000.0 * index as f32 / 48_000.0;
        writer.write_sample(amplitude * phase.sin()).unwrap();
        writer.write_sample(0.0_f32).unwrap();
    }
    writer.finalize().unwrap();

    let config_string = config_path.to_string_lossy();
    let input_string = input_path.to_string_lossy();
    let output_string = output_path.to_string_lossy();
    let expanded = command
        .replace("{config}", &config_string)
        .replace("{input}", &input_string)
        .replace("{output}", &output_string);
    #[cfg(windows)]
    let result = Command::new("cmd")
        .args(["/C", &expanded])
        .output()
        .unwrap();
    #[cfg(not(windows))]
    let result = Command::new("sh").args(["-c", &expanded]).output().unwrap();
    assert!(
        result.status.success(),
        "Equalizer APO PCM command failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );

    let mut reader = hound::WavReader::open(output_path).unwrap();
    let samples: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
    let left: Vec<f64> = samples
        .chunks_exact(2)
        .skip(4096)
        .map(|frame| frame[0] as f64 / i16::MAX as f64)
        .collect();
    let measured_rms =
        (left.iter().map(|sample| sample * sample).sum::<f64>() / left.len() as f64).sqrt();
    let input_rms = amplitude as f64 / std::f64::consts::SQRT_2;
    let expected_gain = 10.0_f64.powf(3.0 / 20.0);
    let measured_gain = measured_rms / input_rms;
    assert!(
        (measured_gain - expected_gain).abs() < 0.003,
        "real Equalizer APO gain {measured_gain}, expected {expected_gain}"
    );
}

#[test]
fn tool_contract_camilladsp_pcm_processes_convolution_sidecar() {
    let output = output_with_plugins(vec![PluginConfigWrapper {
        plugin_type: "convolution".to_string(),
        parameters: json!({"ir_file": "identity.wav"}),
    }]);
    let config = render_dsp_chain(&output, ExportFormat::CamillaDsp, 48_000.0).unwrap();
    let mut input = vec![0_i32; 8192];
    let impulse_index = 128;
    let amplitude = 1 << 27;
    input[impulse_index] = amplitude;

    let Some(rendered) = run_optional_pcm_backend_contract(
        "ROOMEQ_CAMILLADSP_BIN",
        "yaml",
        &config,
        &input,
        |dir| write_identity_wav(&dir.join("identity.wav")),
    ) else {
        return;
    };

    let (peak_index, peak) = rendered
        .iter()
        .copied()
        .enumerate()
        .max_by_key(|(_, sample)| sample.unsigned_abs())
        .unwrap();
    assert_eq!(peak_index, impulse_index);
    assert!((peak - amplitude).abs() <= 2, "unexpected peak {peak}");
}

fn write_identity_wav(path: &Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    writer.write_sample(i32::MAX).unwrap();
    for _ in 1..64 {
        writer.write_sample(0_i32).unwrap();
    }
    writer.finalize().unwrap();
}

#[test]
fn camilladsp_rejects_unsupported_plugins_instead_of_dropping_them() {
    let output = output_with_plugins(vec![PluginConfigWrapper {
        plugin_type: "xtc".to_string(),
        parameters: json!({}),
    }]);

    let error = camilladsp_error(&output);
    assert!(error.contains("does not support channel 'left' plugin #0 ('xtc')"));
}

#[test]
fn every_external_export_rejects_unknown_plugins_instead_of_dropping_them() {
    let output = output_with_plugins(vec![PluginConfigWrapper {
        plugin_type: "xtc".to_string(),
        parameters: json!({}),
    }]);
    for format in [
        ExportFormat::CamillaDsp,
        ExportFormat::EqualizerApo,
        ExportFormat::EasyEffects,
        ExportFormat::Wavelet,
        ExportFormat::PipeWire,
        ExportFormat::RoonDsp,
    ] {
        let error = render_dsp_chain(&output, format, 48_000.0)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("does not support channel 'left' plugin #0 ('xtc')"),
            "unexpected {format:?} error: {error}"
        );
    }
}

#[test]
fn systemwide_exports_reject_time_domain_processing_and_channel_collapse() {
    let convolution = output_with_plugins(vec![PluginConfigWrapper {
        plugin_type: "convolution".to_string(),
        parameters: json!({"ir_file": "room.wav"}),
    }]);
    for format in [ExportFormat::EasyEffects, ExportFormat::Wavelet] {
        let error = render_dsp_chain(&convolution, format, 48_000.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not support channel 'left' plugin #0 ('convolution')"));
    }

    let mut different_channels = output_with_plugins(vec![PluginConfigWrapper {
        plugin_type: "gain".to_string(),
        parameters: json!({"gain_db": -1.0}),
    }]);
    different_channels.channels.insert(
        "right".to_string(),
        channel(
            "right",
            vec![PluginConfigWrapper {
                plugin_type: "gain".to_string(),
                parameters: json!({"gain_db": -2.0}),
            }],
        ),
    );
    for format in [ExportFormat::EasyEffects, ExportFormat::Wavelet] {
        let error = render_dsp_chain(&different_channels, format, 48_000.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot preserve different per-channel DSP chains"));
    }
}

#[test]
fn roon_rejects_lossy_filter_substitution_and_truncation() {
    let allpass = output_with_plugins(vec![PluginConfigWrapper {
        plugin_type: "eq".to_string(),
        parameters: json!({"filters": [{
            "filter_type": "allpass", "freq": 1000.0, "q": 1.0, "db_gain": 0.0
        }]}),
    }]);
    let error = render_dsp_chain(&allpass, ExportFormat::RoonDsp, 48_000.0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cannot represent all-pass filter"));

    let filters: Vec<_> = (0..21)
        .map(|index| {
            json!({
                "filter_type": "peak", "freq": 100.0 + index as f64 * 10.0,
                "q": 1.0, "db_gain": -1.0
            })
        })
        .collect();
    let too_many = output_with_plugins(vec![PluginConfigWrapper {
        plugin_type: "eq".to_string(),
        parameters: json!({"filters": filters}),
    }]);
    let error = render_dsp_chain(&too_many, ExportFormat::RoonDsp, 48_000.0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("at most 20 PEQ filters"));

    for (parameters, expected) in [
        (
            json!({"ir_file": "/tmp/room.wav"}),
            "safe relative impulse path",
        ),
        (
            json!({"ir_file": "room.wav", "mix": 0.5}),
            "does not support convolution parameter 'mix'",
        ),
    ] {
        let convolution = output_with_plugins(vec![PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters,
        }]);
        let error = render_dsp_chain(&convolution, ExportFormat::RoonDsp, 48_000.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn camilladsp_rejects_missing_or_malformed_plugin_parameters() {
    let cases = [
        (
            PluginConfigWrapper {
                plugin_type: "gain".to_string(),
                parameters: json!({}),
            },
            "requires numeric field 'gain_db'",
        ),
        (
            PluginConfigWrapper {
                plugin_type: "delay".to_string(),
                parameters: json!({"delay_ms": "late"}),
            },
            "requires numeric field 'delay_ms'",
        ),
        (
            PluginConfigWrapper {
                plugin_type: "gain".to_string(),
                parameters: json!({"gain_db": 151.0}),
            },
            "requires gain_db between -150 and 150",
        ),
        (
            PluginConfigWrapper {
                plugin_type: "eq".to_string(),
                parameters: json!({
                    "filters": [{
                        "filter_type": "lowsehlf",
                        "freq": 100.0,
                        "q": 0.7,
                        "db_gain": 1.0,
                    }]
                }),
            },
            "does not support filter type 'lowsehlf'",
        ),
        (
            PluginConfigWrapper {
                plugin_type: "convolution".to_string(),
                parameters: json!({}),
            },
            "requires string field 'ir_file'",
        ),
    ];

    for (plugin, expected) in cases {
        let error = camilladsp_error(&output_with_plugins(vec![plugin]));
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn camilladsp_rejects_filter_frequencies_at_or_above_nyquist() {
    let output = output_with_plugins(vec![PluginConfigWrapper {
        plugin_type: "eq".to_string(),
        parameters: json!({
            "filters": [{
                "filter_type": "peak",
                "freq": 24000.0,
                "q": 1.0,
                "db_gain": -1.0,
            }]
        }),
    }]);

    let error = camilladsp_error(&output);
    assert!(error.contains("must be below the 24000 Hz Nyquist frequency"));
}

#[test]
fn camilladsp_rejects_nonstandard_eq_topologies() {
    for topology in ["warped_biquad", "kautz_filter"] {
        let output = output_with_plugins(vec![PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: json!({
                "filters": [{
                    "topology": topology,
                    "filter_type": "peak",
                    "freq": 100.0,
                    "q": 1.0,
                    "db_gain": -1.0,
                }]
            }),
        }]);

        let error = camilladsp_error(&output);
        assert!(
            error.contains(&format!("does not support EQ topology '{topology}'")),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn camilladsp_rejects_fractional_sample_rates_before_truncation() {
    let error = render_dsp_chain(&make_test_output(), ExportFormat::CamillaDsp, 48_000.5)
        .unwrap_err()
        .to_string();

    assert!(error.contains("requires an integer sample rate"));
}

#[test]
fn camilladsp_rejects_channel_identifier_collisions() {
    let output = DspChainOutput {
        version: "1.3.0".to_string(),
        global_plugins: Vec::new(),
        channels: HashMap::from([
            ("left A".to_string(), channel("left A", Vec::new())),
            ("left-A".to_string(), channel("left-A", Vec::new())),
        ]),
        metadata: None,
    };

    let error = camilladsp_error(&output);
    assert!(error.contains("both normalize to 'left_A'"));
}

#[test]
fn camilladsp_routed_export_requires_every_plugin_to_have_a_stage() {
    let mut output = make_routed_bass_output();
    output.channels.get_mut("L").unwrap().plugins[0]
        .parameters
        .as_object_mut()
        .unwrap()
        .remove("room_eq_stage");

    let error = camilladsp_error(&output);
    assert!(error.contains("routed export requires room_eq_stage"));
}

#[test]
fn camilladsp_routed_export_validates_route_channel_indices() {
    let mut output = make_routed_bass_output();
    output
        .metadata
        .as_mut()
        .unwrap()
        .bass_management
        .as_mut()
        .unwrap()
        .routing_graph
        .as_mut()
        .unwrap()
        .routes[0]
        .source_index = 1;

    let error = camilladsp_error(&output);
    assert!(error.contains("source index 1 names 'R', not 'L'"));
}
