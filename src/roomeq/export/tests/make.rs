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
use super::super::extract::extract_eq_filters;
use super::super::extract_gain_db;
use super::super::misc::parse_biquad_filter_type;
use super::super::package::package_convolution_sidecars;
use super::super::roon_convolver::package_roon_convolution_archive;
use crate::roomeq::{
    BassManagementMatrix, BassManagementReport, BassManagementRoute, BassManagementRoutingGraph,
};
use std::collections::HashMap;
use std::process::Command;

use crate::roomeq::types::*;
use serde_json::json;

/// Build a test DspChainOutput with 2 channels, each having gain + 3 PEQ + delay
pub(super) fn make_test_output() -> DspChainOutput {
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
            correction_acceptance: None,
            stage_outcomes: Vec::new(),
        }),
    }
}

pub(super) fn make_systemwide_test_output() -> DspChainOutput {
    let mut output = make_test_output();
    let plugins: Vec<_> = output.channels["left"]
        .plugins
        .iter()
        .filter(|plugin| plugin.plugin_type != "delay")
        .cloned()
        .collect();
    for chain in output.channels.values_mut() {
        chain.plugins.clone_from(&plugins);
    }
    output
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

pub(super) fn make_routed_bass_output() -> DspChainOutput {
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
            correction_acceptance: None,
            stage_outcomes: Vec::new(),
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

include!("make/packaging.rs");
include!("make/camilladsp.rs");
include!("make/formats.rs");
