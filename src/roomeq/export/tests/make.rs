use super::super::super::types::{ChannelDspChain, DspChainOutput, PluginConfigWrapper};
use super::super::export_camilladsp;
use super::super::export_dsp_chain_with_convolution_sidecars;
use super::super::export_easyeffects;
use super::super::export_equalizer_apo;
use super::super::export_format::ExportFormat;
use super::super::export_format::external_export_supported;
use super::super::export_pipewire;
use super::super::export_roon;
use super::super::export_wavelet;
use super::super::package::package_convolution_sidecars;
use crate::roomeq::{
    BassManagementMatrix, BassManagementReport, BassManagementRoute, BassManagementRoutingGraph,
};
use std::collections::HashMap;
use std::process::Command;

use crate::roomeq::types::*;
use serde_json::json;

/// Build a test DspChainOutput with 2 channels, each having gain + 3 PEQ + delay
fn make_test_output() -> DspChainOutput {
    let mut channels = HashMap::new();

    // Left channel: gain -2.5 dB, delay 1.5 ms, 3 PEQ bands
    channels.insert(
        "left".to_string(),
        ChannelDspChain {
            channel: "left".to_string(),
            plugins: vec![
                PluginConfigWrapper {
                    plugin_type: "gain".to_string(),
                    parameters: json!({"gain_db": -2.5}),
                },
                PluginConfigWrapper {
                    plugin_type: "delay".to_string(),
                    parameters: json!({"delay_ms": 1.5}),
                },
                PluginConfigWrapper {
                    plugin_type: "eq".to_string(),
                    parameters: json!({
                        "filters": [
                            {"filter_type": "peak", "freq": 100.0, "q": 2.0, "db_gain": -5.0},
                            {"filter_type": "peak", "freq": 1000.0, "q": 1.5, "db_gain": 3.0},
                            {"filter_type": "highshelf", "freq": 8000.0, "q": 0.7, "db_gain": -2.0},
                        ]
                    }),
                },
            ],
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
    );

    // Right channel: gain -1.0 dB, 2 PEQ bands
    channels.insert(
        "right".to_string(),
        ChannelDspChain {
            channel: "right".to_string(),
            plugins: vec![
                PluginConfigWrapper {
                    plugin_type: "gain".to_string(),
                    parameters: json!({"gain_db": -1.0}),
                },
                PluginConfigWrapper {
                    plugin_type: "eq".to_string(),
                    parameters: json!({
                        "filters": [
                            {"filter_type": "peak", "freq": 200.0, "q": 1.0, "db_gain": -3.0},
                            {"filter_type": "lowshelf", "freq": 80.0, "q": 0.71, "db_gain": 4.0},
                        ]
                    }),
                },
            ],
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
    );

    DspChainOutput {
        version: "1.3.0".to_string(),
        global_plugins: Vec::new(),
        channels,
        metadata: Some(OptimizationMetadata {
            pre_score: 5.0,
            post_score: 2.0,
            algorithm: "cobyla".to_string(),
            loss_type: Some("flat".to_string()),
            iterations: 1000,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            inter_channel_deviation: None,
            epa_per_channel: None,
            epa_multichannel: None,
            group_delay: None,
            perceptual_metrics: None,
            home_cinema_layout: None,
            multi_seat_coverage: None,
            multi_seat_correction: None,
            bass_management: None,
            timing_diagnostics: None,
            ctc: None,
            perceptual_policy: None,
            bootstrap_uncertainty: None,
            validation_bundle: None,
            supporting_source: None,
        }),
    }
}

fn make_single_filter_output(filter_type: &str, gain_db: f64) -> DspChainOutput {
    let mut channels = HashMap::new();
    channels.insert(
        "left".to_string(),
        ChannelDspChain {
            channel: "left".to_string(),
            plugins: vec![PluginConfigWrapper {
                plugin_type: "eq".to_string(),
                parameters: json!({
                    "filters": [
                        {
                            "filter_type": filter_type,
                            "freq": 80.0,
                            "q": 0.707,
                            "db_gain": gain_db,
                        }
                    ]
                }),
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
    );

    DspChainOutput {
        version: "1.3.0".to_string(),
        global_plugins: Vec::new(),
        channels,
        metadata: None,
    }
}

fn make_routed_bass_output() -> DspChainOutput {
    let mut channels = HashMap::new();
    for channel in ["L", "R", "LFE"] {
        let post_filter = if channel == "LFE" {
            json!({"filter_type": "peak", "freq": 50.0, "q": 1.0, "db_gain": -2.0})
        } else {
            json!({"filter_type": "peak", "freq": 1000.0, "q": 1.0, "db_gain": -1.0})
        };
        channels.insert(
            channel.to_string(),
            ChannelDspChain {
                channel: channel.to_string(),
                plugins: vec![
                    PluginConfigWrapper {
                        plugin_type: "gain".to_string(),
                        parameters: json!({"gain_db": -0.5, "room_eq_stage": "pre_route"}),
                    },
                    PluginConfigWrapper {
                        plugin_type: "eq".to_string(),
                        parameters: json!({
                            "filters": [post_filter],
                            "room_eq_stage": "post_route",
                        }),
                    },
                ],
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
        );
    }

    let routes = vec![
        BassManagementRoute {
            group_id: Some("lcr".to_string()),
            source_channel: "L".to_string(),
            source_index: 0,
            destination: "L".to_string(),
            destination_index: 0,
            pre_chain_channel: Some("L".to_string()),
            post_chain_channel: Some("L".to_string()),
            route_kind: "main_highpass_to_self".to_string(),
            crossover_type: "LR24".to_string(),
            high_pass_hz: Some(80.0),
            low_pass_hz: None,
            gain_db: 0.0,
            gain_linear: 1.0,
            matrix_gain: 1.0,
            delay_ms: 1.25,
            polarity_inverted: false,
        },
        BassManagementRoute {
            group_id: Some("lcr".to_string()),
            source_channel: "L".to_string(),
            source_index: 0,
            destination: "LFE".to_string(),
            destination_index: 2,
            pre_chain_channel: Some("LFE".to_string()),
            post_chain_channel: Some("LFE".to_string()),
            route_kind: "redirected_bass_lowpass_to_sub".to_string(),
            crossover_type: "LR24".to_string(),
            high_pass_hz: None,
            low_pass_hz: Some(80.0),
            gain_db: -6.0,
            gain_linear: 10.0_f64.powf(-6.0 / 20.0),
            matrix_gain: 10.0_f64.powf(-6.0 / 20.0),
            delay_ms: 2.5,
            polarity_inverted: false,
        },
        BassManagementRoute {
            group_id: Some("lcr".to_string()),
            source_channel: "R".to_string(),
            source_index: 1,
            destination: "R".to_string(),
            destination_index: 1,
            pre_chain_channel: Some("R".to_string()),
            post_chain_channel: Some("R".to_string()),
            route_kind: "main_highpass_to_self".to_string(),
            crossover_type: "LR24".to_string(),
            high_pass_hz: Some(80.0),
            low_pass_hz: None,
            gain_db: 0.0,
            gain_linear: 1.0,
            matrix_gain: 1.0,
            delay_ms: 1.25,
            polarity_inverted: false,
        },
        BassManagementRoute {
            group_id: Some("lcr".to_string()),
            source_channel: "R".to_string(),
            source_index: 1,
            destination: "LFE".to_string(),
            destination_index: 2,
            pre_chain_channel: Some("LFE".to_string()),
            post_chain_channel: Some("LFE".to_string()),
            route_kind: "redirected_bass_lowpass_to_sub".to_string(),
            crossover_type: "LR24".to_string(),
            high_pass_hz: None,
            low_pass_hz: Some(80.0),
            gain_db: -6.0,
            gain_linear: 10.0_f64.powf(-6.0 / 20.0),
            matrix_gain: 10.0_f64.powf(-6.0 / 20.0),
            delay_ms: 2.5,
            polarity_inverted: true,
        },
        BassManagementRoute {
            group_id: Some("lfe".to_string()),
            source_channel: "LFE".to_string(),
            source_index: 2,
            destination: "LFE".to_string(),
            destination_index: 2,
            pre_chain_channel: Some("LFE".to_string()),
            post_chain_channel: Some("LFE".to_string()),
            route_kind: "lfe_lowpass_to_sub".to_string(),
            crossover_type: "LR24".to_string(),
            high_pass_hz: None,
            low_pass_hz: Some(80.0),
            gain_db: -3.0,
            gain_linear: 10.0_f64.powf(-3.0 / 20.0),
            matrix_gain: 10.0_f64.powf(-3.0 / 20.0),
            delay_ms: 2.0,
            polarity_inverted: false,
        },
    ];

    let routing_graph = BassManagementRoutingGraph {
        physical_sub_output: "LFE".to_string(),
        input_channels: vec!["L".to_string(), "R".to_string(), "LFE".to_string()],
        output_channels: vec!["L".to_string(), "R".to_string(), "LFE".to_string()],
        routes,
        matrix: Some(BassManagementMatrix {
            input_channel_map: vec![0, 1, 2],
            output_channel_map: vec![2],
            matrix: vec![
                10.0_f32.powf(-6.0 / 20.0),
                10.0_f32.powf(-6.0 / 20.0),
                10.0_f32.powf(-3.0 / 20.0),
            ],
            route_count: 3,
        }),
        advisories: vec!["ok".to_string()],
    };

    DspChainOutput {
        version: "1.3.0".to_string(),
        global_plugins: Vec::new(),
        channels,
        metadata: Some(OptimizationMetadata {
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
            perceptual_metrics: None,
            home_cinema_layout: None,
            multi_seat_coverage: None,
            multi_seat_correction: None,
            bass_management: Some(BassManagementReport {
                enabled: true,
                crossover_type: "LR24".to_string(),
                crossover_frequency_hz: Some(80.0),
                redirected_bass_enabled: true,
                lfe_channel: "LFE".to_string(),
                lfe_playback_gain_db: 10.0,
                lfe_gain_applied_to_chain: false,
                sub_trim_db: 0.0,
                max_sub_boost_db: 6.0,
                headroom_margin_db: 6.0,
                applied_sub_gain_db: Some(0.0),
                gain_limited: false,
                physical_sub_output: "LFE".to_string(),
                redirected_bass_channel_count: 2,
                main_high_pass_hz: Some(80.0),
                sub_low_pass_hz: Some(80.0),
                lfe_headroom_required_db: 16.0,
                signal_flow: Vec::new(),
                signal_flow_advisories: Vec::new(),
                routing_graph: Some(routing_graph),
                optimization: None,
                groups: Vec::new(),
                sub_outputs: Vec::new(),
                headroom_simulation: None,
                advisory: "ok".to_string(),
            }),
            timing_diagnostics: None,
            ctc: None,
            perceptual_policy: None,
            bootstrap_uncertainty: None,
            validation_bundle: None,
            supporting_source: None,
        }),
    }
}

fn run_optional_export_validator(env_var: &str, extension: &str, content: &str) {
    let template = match std::env::var(env_var) {
        Ok(template) if !template.trim().is_empty() => template,
        _ => {
            eprintln!("skipping optional export validator; set {env_var} to enable it");
            return;
        }
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("room_eq.{extension}"));
    std::fs::write(&path, content).unwrap();
    let path_str = path.to_string_lossy();
    let mut parts = template.split_whitespace();
    let Some(program) = parts.next() else {
        panic!("{env_var} did not contain a command");
    };
    let mut saw_placeholder = false;
    let mut command = Command::new(program);
    for part in parts {
        if part.contains("{config}") || part.contains("{file}") {
            saw_placeholder = true;
        }
        command.arg(
            part.replace("{config}", &path_str)
                .replace("{file}", &path_str),
        );
    }
    if !saw_placeholder {
        command.arg(&path);
    }
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("failed to run {env_var} validator '{template}': {err}"));
    assert!(
        output.status.success(),
        "{env_var} validator failed: {template}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn external_exports_reject_routed_bass_management() {
    let output = make_routed_bass_output();
    assert!(external_export_supported(&output, ExportFormat::CamillaDsp).is_ok());

    for format in [
        ExportFormat::EqualizerApo,
        ExportFormat::EasyEffects,
        ExportFormat::Wavelet,
        ExportFormat::PipeWire,
        ExportFormat::RoonDsp,
    ] {
        let err = external_export_supported(&output, format).unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot represent routed home-cinema bass management safely"),
            "unexpected error for {format:?}: {err}"
        );
    }
}

#[test]
fn package_convolution_sidecars_copies_and_rewrites_relative_paths() {
    let source_dir = tempfile::tempdir().unwrap();
    let dest_dir = tempfile::tempdir().unwrap();
    std::fs::write(source_dir.path().join("L_fir_96000hz.wav"), b"wav").unwrap();

    let mut output = make_test_output();
    output
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": "L_fir_96000hz.wav"}),
        });

    let packaged =
        package_convolution_sidecars(&output, source_dir.path(), dest_dir.path()).unwrap();

    assert_eq!(
        std::fs::read(dest_dir.path().join("L_fir_96000hz.wav")).unwrap(),
        b"wav"
    );
    let ir_file = packaged.channels["left"]
        .plugins
        .iter()
        .find(|plugin| plugin.plugin_type == "convolution")
        .unwrap()
        .parameters
        .get("ir_file")
        .and_then(|value| value.as_str())
        .unwrap();
    assert_eq!(ir_file, "L_fir_96000hz.wav");
}

#[test]
fn package_convolution_sidecars_avoids_destination_collisions() {
    let source_dir = tempfile::tempdir().unwrap();
    let dest_dir = tempfile::tempdir().unwrap();
    std::fs::write(source_dir.path().join("L_fir_96000hz.wav"), b"new").unwrap();
    std::fs::write(dest_dir.path().join("L_fir_96000hz.wav"), b"old").unwrap();

    let mut output = make_test_output();
    output
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": "L_fir_96000hz.wav"}),
        });

    let packaged =
        package_convolution_sidecars(&output, source_dir.path(), dest_dir.path()).unwrap();

    assert_eq!(
        std::fs::read(dest_dir.path().join("L_fir_96000hz_002.wav")).unwrap(),
        b"new"
    );
    let ir_file = packaged.channels["left"]
        .plugins
        .iter()
        .find(|plugin| plugin.plugin_type == "convolution")
        .unwrap()
        .parameters
        .get("ir_file")
        .and_then(|value| value.as_str())
        .unwrap();
    assert_eq!(ir_file, "L_fir_96000hz_002.wav");
}

#[test]
fn export_with_convolution_sidecars_uses_selected_sample_rate() {
    let source_dir = tempfile::tempdir().unwrap();
    let dest_dir = tempfile::tempdir().unwrap();
    std::fs::write(source_dir.path().join("L_fir_96000hz.wav"), b"wav").unwrap();

    let mut output = make_test_output();
    output
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": "L_fir_96000hz.wav"}),
        });

    let export_path = dest_dir.path().join("room_eq_cdsp.yaml");
    export_dsp_chain_with_convolution_sidecars(
        &output,
        ExportFormat::CamillaDsp,
        &export_path,
        96_000.0,
        source_dir.path(),
    )
    .unwrap();

    let yaml = std::fs::read_to_string(&export_path).unwrap();
    assert!(yaml.contains("samplerate: 96000"));
    assert!(yaml.contains("filename: L_fir_96000hz.wav"));
    assert!(dest_dir.path().join("L_fir_96000hz.wav").is_file());
}

#[test]
fn test_export_camilladsp() {
    let output = make_test_output();
    let result = export_camilladsp(&output, 48000.0).unwrap();

    assert!(result.contains("samplerate: 48000"));
    assert!(result.contains("left_gain:"));
    assert!(result.contains("left_delay:"));
    assert!(result.contains("left_peq_0:"));
    assert!(result.contains("left_peq_1:"));
    assert!(result.contains("left_peq_2:"));
    assert!(result.contains("right_gain:"));
    assert!(result.contains("right_peq_0:"));
    assert!(result.contains("type: Biquad"));
    assert!(result.contains("type: Peaking"));
    assert!(result.contains("type: Highshelf"));
    assert!(result.contains("type: Gain"));
    assert!(result.contains("type: Delay"));
    assert!(result.contains("unit: ms"));
    assert!(result.contains("pipeline:"));
}

#[test]
fn test_export_camilladsp_routed_bass_management_graph() {
    let output = make_routed_bass_output();
    let result = export_camilladsp(&output, 48000.0).unwrap();

    assert!(result.contains("# Routed bass-management graph export"));
    assert!(result.contains("capture:\n    type: File\n    channels: 3"));
    assert!(result.contains("playback:\n    type: File\n    channels: 3"));
    assert!(result.contains("mixers:"));
    assert!(result.contains("  roomeq_route_matrix:"));
    assert!(result.contains("  roomeq_route_sum:"));
    assert!(result.contains("  name: roomeq_route_matrix\n  type: Mixer"));
    assert!(result.contains("  name: roomeq_route_sum\n  type: Mixer"));
    assert!(result.contains("route_0_L_to_L_crossover:"));
    assert!(result.contains("type: LinkwitzRileyHighpass"));
    assert!(result.contains("route_1_L_to_LFE_crossover:"));
    assert!(result.contains("type: LinkwitzRileyLowpass"));
    assert!(result.contains("route_1_L_to_LFE_delay:"));
    assert!(result.contains("delay: 2.500"));
    assert!(result.contains("gain: -6.000000"));
    assert!(result.contains("inverted: true"));
    assert!(result.contains("post_LFE_peq_0:"));
    assert!(result.contains("  - post_LFE_peq_0"));
}

#[test]
fn tool_contract_camilladsp_routed_export_can_be_validated_locally() {
    let output = make_routed_bass_output();
    let result = export_camilladsp(&output, 48000.0).unwrap();

    let route_matrix = result.find("  roomeq_route_matrix:").unwrap();
    let route_sum = result.find("  roomeq_route_sum:").unwrap();
    let pipeline = result.find("pipeline:").unwrap();
    assert!(route_matrix < route_sum);
    assert!(route_sum < pipeline);
    assert_eq!(result.matches("type: Mixer").count(), 2);
    assert_eq!(result.matches("type: BiquadCombo").count(), 5);
    assert!(result.contains("channels:\n      in: 3\n      out: 5"));
    assert!(result.contains("channels:\n      in: 5\n      out: 3"));

    run_optional_export_validator("ROOMEQ_CAMILLADSP_VALIDATE_CMD", "yaml", &result);
}

#[test]
fn test_export_camilladsp_writes_crossover_plugins() {
    let mut output = make_test_output();
    output
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "crossover".to_string(),
            parameters: json!({
                "type": "LR48",
                "frequency": 95.0,
                "output": "high",
            }),
        });

    let result = export_camilladsp(&output, 48000.0).unwrap();
    assert!(result.contains("left_crossover:"));
    assert!(result.contains("type: BiquadCombo"));
    assert!(result.contains("type: LinkwitzRileyHighpass"));
    assert!(result.contains("order: 8"));
    assert!(result.contains("  - left_crossover"));
}

#[test]
fn test_camilladsp_pipeline_uses_gui_friendly_steps() {
    let output = make_test_output();
    let result = export_camilladsp(&output, 48000.0).unwrap();

    assert!(
        result.contains("pipeline:\n- bypassed: null\n  channels:\n  - 0\n  names:\n  - left_gain"),
        "Expected pipeline entries to start with bypassed null, got:\n{result}"
    );
    assert!(
        result.contains("  type: Filter\n- bypassed: null"),
        "Expected type line inside the pipeline step, got:\n{result}"
    );
    assert!(
        !result.contains("  - type: Filter"),
        "Pipeline step should not start with a dashed type line"
    );
    assert!(result.contains("  - left_delay"));
    assert!(result.contains("  - left_peq_0"));
    assert!(result.contains("  - right_gain"));
}

#[test]
fn test_export_equalizer_apo() {
    let output = make_test_output();
    let result = export_equalizer_apo(&output).unwrap();

    assert!(result.contains("Channel: L"));
    assert!(result.contains("Channel: R"));
    assert!(result.contains("Preamp: -2.5 dB"));
    assert!(result.contains("Delay: 1.500 ms"));
    assert!(result.contains("Filter  1: ON PK Fc 100 Hz Gain -5.00 dB Q 2.0000"));
    assert!(result.contains("Filter  3: ON HSC Fc 8000 Hz Gain -2.00 dB Q 0.7000"));
    assert!(result.contains("Filter  1: ON PK Fc 200 Hz Gain -3.00 dB Q 1.0000"));
    assert!(result.contains("Filter  2: ON LSC Fc 80 Hz Gain +4.00 dB Q 0.7100"));
}

#[test]
fn tool_contract_equalizer_apo_text_has_channel_scoped_filters() {
    let output = make_test_output();
    let result = export_equalizer_apo(&output).unwrap();

    let mut current_channel = None;
    let mut left_filters = 0;
    let mut right_filters = 0;
    for line in result.lines() {
        if line == "Channel: L" {
            current_channel = Some("L");
        } else if line == "Channel: R" {
            current_channel = Some("R");
        } else if line.starts_with("Filter") {
            assert!(line.contains(" ON "));
            assert!(line.contains(" Fc "));
            match current_channel {
                Some("L") => left_filters += 1,
                Some("R") => right_filters += 1,
                _ => panic!("filter emitted before channel header: {line}"),
            }
        }
    }
    assert_eq!(left_filters, 3);
    assert_eq!(right_filters, 2);

    run_optional_export_validator("ROOMEQ_EQUALIZER_APO_VALIDATE_CMD", "txt", &result);
}

#[test]
fn test_export_easyeffects() {
    let output = make_test_output();
    let result = export_easyeffects(&output).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let eq = &parsed["output"]["equalizer#0"];
    assert_eq!(eq["num-bands"].as_u64().unwrap(), 5);
    assert!(eq["left"]["band0"]["frequency"].as_f64().unwrap() > 0.0);

    // Check filter types
    let band0_type = eq["left"]["band0"]["type"].as_str().unwrap();
    assert_eq!(band0_type, "Bell");
}

#[test]
fn tool_contract_easyeffects_json_has_mirrored_stereo_preset() {
    let output = make_test_output();
    let result = export_easyeffects(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let eq = &parsed["output"]["equalizer#0"];

    assert_eq!(eq["split-channels"], serde_json::json!(false));
    assert_eq!(eq["num-bands"].as_u64().unwrap(), 5);
    assert_eq!(eq["left"], eq["right"]);
    for band_idx in 0..eq["num-bands"].as_u64().unwrap() {
        let band = &eq["left"][format!("band{band_idx}")];
        assert!(band["frequency"].as_f64().unwrap().is_finite());
        assert!(band["q"].as_f64().unwrap().is_finite());
        assert!(band["gain"].as_f64().unwrap().is_finite());
        assert_eq!(band["solo"], serde_json::json!(false));
        assert_eq!(band["mute"], serde_json::json!(false));
    }

    run_optional_export_validator("ROOMEQ_EASYEFFECTS_VALIDATE_CMD", "json", &result);
}

#[test]
fn test_export_wavelet() {
    let output = make_test_output();
    let result = export_wavelet(&output, 48000.0).unwrap();

    assert!(result.contains("GraphicEQ:"));
    // Should have 9 frequency/gain pairs
    let line = result
        .lines()
        .find(|l| l.starts_with("GraphicEQ:"))
        .unwrap();
    let parts: Vec<&str> = line.trim_start_matches("GraphicEQ:").split(';').collect();
    assert_eq!(parts.len(), 9);
}

#[test]
fn tool_contract_wavelet_graphiceq_has_numeric_band_pairs() {
    let output = make_test_output();
    let result = export_wavelet(&output, 48000.0).unwrap();
    let line = result
        .lines()
        .find(|line| line.starts_with("GraphicEQ:"))
        .unwrap();
    let mut previous_freq = 0.0;
    for pair in line.trim_start_matches("GraphicEQ:").split(';') {
        let fields: Vec<_> = pair.split_whitespace().collect();
        assert_eq!(fields.len(), 2, "unexpected Wavelet band pair: {pair}");
        let freq: f64 = fields[0].parse().unwrap();
        let gain: f64 = fields[1].parse().unwrap();
        assert!(freq > previous_freq);
        assert!(gain.is_finite());
        previous_freq = freq;
    }

    run_optional_export_validator("ROOMEQ_WAVELET_VALIDATE_CMD", "txt", &result);
}

#[test]
fn test_export_pipewire() {
    let output = make_test_output();
    let result = export_pipewire(&output, 48000.0).unwrap();

    assert!(result.contains("libpipewire-module-filter-chain"));
    assert!(result.contains("bq_peaking"));
    assert!(result.contains("bq_highshelf"));
    assert!(result.contains("filter.graph"));
    assert!(result.contains("nodes ="));
    assert!(result.contains("links ="));
    assert!(result.contains("audio.channels = 2"));
    assert!(result.contains("\"FL\""));
    assert!(result.contains("\"FR\""));
}

#[test]
fn tool_contract_pipewire_filter_chain_has_nodes_links_and_positions() {
    let output = make_test_output();
    let result = export_pipewire(&output, 48000.0).unwrap();

    assert!(result.contains("filter.graph = {"));
    assert!(result.contains("nodes = ["));
    assert!(result.contains("links = ["));
    assert!(result.contains("inputs  = ["));
    assert!(result.contains("outputs = ["));
    assert!(result.contains("audio.position = [ \"FL\", \"FR\" ]"));
    assert!(result.contains("label = delay"));
    assert!(result.contains("label = bq_peaking"));
    assert!(result.contains("label = bq_highshelf"));

    run_optional_export_validator("ROOMEQ_PIPEWIRE_VALIDATE_CMD", "conf", &result);
}

#[test]
fn test_export_roon() {
    let output = make_test_output();
    let result = export_roon(&output).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let channels = &parsed["channels"];

    // Left channel
    let left = &channels["left"];
    assert!(left["headroom_gain_db"].as_f64().unwrap() < 0.0);
    assert!((left["delay_ms"].as_f64().unwrap() - 1.5).abs() < 0.01);

    let left_bands = left["parametric_eq"]["bands"].as_array().unwrap();
    assert_eq!(left_bands.len(), 3);
    assert_eq!(left_bands[0]["type"].as_str().unwrap(), "Peak/Dip");
    assert_eq!(left_bands[0]["frequency"].as_f64().unwrap(), 100.0);
    assert_eq!(left_bands[2]["type"].as_str().unwrap(), "High Shelf");

    // Right channel
    let right = &channels["right"];
    assert!(right["headroom_gain_db"].as_f64().unwrap() < 0.0);
    assert!(right.get("delay_ms").is_none()); // no delay on right

    let right_bands = right["parametric_eq"]["bands"].as_array().unwrap();
    assert_eq!(right_bands.len(), 2);
    assert_eq!(right_bands[1]["type"].as_str().unwrap(), "Low Shelf");
    assert!(right_bands[0]["enabled"].as_bool().unwrap());
}

#[test]
fn tool_contract_roon_json_keeps_per_channel_manual_setup_data() {
    let output = make_test_output();
    let result = export_roon(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let channels = parsed["channels"].as_object().unwrap();

    assert_eq!(channels.len(), 2);
    for (name, channel) in channels {
        assert!(channel["headroom_gain_db"].as_f64().unwrap().is_finite());
        if name == "left" {
            assert!(channel["delay_ms"].as_f64().unwrap().is_finite());
        }
        let bands = channel["parametric_eq"]["bands"].as_array().unwrap();
        assert!(!bands.is_empty());
        assert!(bands.len() <= 20);
        for band in bands {
            assert_eq!(band["enabled"], serde_json::json!(true));
            assert!(band["frequency"].as_f64().unwrap().is_finite());
            assert!(band["q"].as_f64().unwrap().is_finite());
        }
    }

    run_optional_export_validator("ROOMEQ_ROON_VALIDATE_CMD", "json", &result);
}

#[test]
fn test_wavelet_rejects_unknown_filter_type() {
    let output = make_single_filter_output("lowsehlf", 3.0);

    let err = export_wavelet(&output, 48_000.0).unwrap_err();

    assert!(
        err.to_string().contains("Unsupported biquad filter type"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pipewire_rejects_unknown_filter_type() {
    let output = make_single_filter_output("lowsehlf", 3.0);

    let err = export_pipewire(&output, 48_000.0).unwrap_err();

    assert!(
        err.to_string()
            .contains("Unsupported PipeWire biquad filter type"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pipewire_highpassvariableq_omits_gain_control() {
    let output = make_single_filter_output("highpassvariableq", -6.0);

    let conf = export_pipewire(&output, 48_000.0).unwrap();

    assert!(conf.contains("label = bq_highpass"));
    assert!(
        conf.contains("control = { \"Freq\" = 80.0  \"Q\" = 0.7070 }"),
        "highpassvariableq should emit only Freq/Q controls:\n{conf}"
    );
    assert!(
        !conf.contains("\"Gain\" = -6.00"),
        "PipeWire highpassvariableq must not emit unsupported Gain control:\n{conf}"
    );
}
