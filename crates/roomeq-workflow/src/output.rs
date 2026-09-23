//! RoomEQ output persistence adapters.

use std::path::Path;

use roomeq_model::DspChainOutput;

/// Save a DSP chain as pretty-printed JSON.
///
/// The written number representation matches the value used for payload
/// binding, including widened `f32` routing-matrix coefficients.
///
/// # Errors
///
/// Returns serialization or filesystem write errors.
pub fn save_dsp_chain(
    output: &DspChainOutput,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let normalized = serde_json::to_value(output)?;
    let json = serde_json::to_string_pretty(&normalized)?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saves_pretty_json() {
        let directory = tempfile::TempDir::new().expect("temp output directory");
        let path = directory.path().join("dsp.json");
        let output = DspChainOutput {
            version: "1".to_string(),
            global_plugins: Vec::new(),
            channels: std::collections::HashMap::new(),
            deployed_source_curves: Default::default(),
            metadata: None,
            correction_decisions: None,
        };

        save_dsp_chain(&output, &path).expect("save DSP chain");

        let json = std::fs::read_to_string(path).expect("read DSP chain");
        assert!(json.contains("\n  \"version\": \"1\""));
    }

    #[test]
    fn saved_routing_matrix_matches_bound_json_value() {
        let directory = tempfile::TempDir::new().expect("temp output directory");
        let path = directory.path().join("dsp.json");
        let metadata = serde_json::from_value(serde_json::json!({
            "pre_score": 0.0,
            "post_score": 0.0,
            "algorithm": "fixture",
            "iterations": 0,
            "timestamp": "fixture",
            "bass_management": {
                "routing_title": "fixture",
                "enabled": true,
                "crossover_type": "LR24",
                "crossover_frequency_hz": 80.0,
                "redirected_bass_enabled": true,
                "sub_trim_db": 0.0,
                "max_sub_boost_db": 0.0,
                "headroom_margin_db": 0.0,
                "applied_sub_gain_db": null,
                "gain_limited": false,
                "physical_sub_outputs": ["Sub1"],
                "redirected_bass_channel_count": 1,
                "main_high_pass_hz": 80.0,
                "sub_low_pass_hz": 80.0,
                "lfe_headroom_required_db": 0.0,
                "signal_flow": [],
                "signal_flow_advisories": [],
                "routing_graph": {
                    "physical_sub_outputs": ["Sub1"],
                    "input_channels": ["L"],
                    "output_channels": ["Sub1"],
                    "routes": [],
                    "matrix": {
                        "input_channel_map": [0],
                        "output_channel_map": [0],
                        "matrix": [0.54963374],
                        "route_count": 0
                    },
                    "advisories": []
                },
                "advisory": "fixture"
            }
        }))
        .expect("routing metadata");
        let output = DspChainOutput {
            version: "1".to_string(),
            global_plugins: Vec::new(),
            channels: std::collections::HashMap::new(),
            deployed_source_curves: Default::default(),
            metadata: Some(metadata),
            correction_decisions: None,
        };
        let bound_value = serde_json::to_value(&output).expect("bound JSON value");

        save_dsp_chain(&output, &path).expect("save DSP chain");
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).expect("read DSP chain"))
                .expect("parse saved DSP chain");
        assert_eq!(saved, bound_value);
    }
}
