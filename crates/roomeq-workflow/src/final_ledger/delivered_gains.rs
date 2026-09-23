//! Derive final gain observations from canonical serialized electrical paths.
//!
//! Each row describes a gain stage on a named input-to-output branch, not a
//! measured benefit or the net gain of coherently summed parallel branches.

use super::{DecisionAction, DecisionRecord, DecisionStage, DecisionStatus, observed};
use roomeq_model::{DspGraph, PluginConfigWrapper};

pub(super) const ID_PREFIX: &str = "delivered-gain:";

fn final_gain_label(plugin: &PluginConfigWrapper) -> Option<&str> {
    if plugin.plugin_type != "gain" {
        return None;
    }
    plugin.parameters.get("label").and_then(|label| {
        label.as_str().filter(|label| {
            matches!(
                *label,
                "final_electrical_headroom" | "final_channel_level_alignment"
            )
        })
    })
}

pub(super) fn records(graph: &DspGraph) -> Result<Vec<DecisionRecord>, String> {
    // Preserve legacy/unsupported graph behavior when there is no gain claim
    // to resolve. Never guess output ownership for a gain we do report.
    let has_gains = graph.channels.values().any(|channel| {
        channel
            .plugins
            .iter()
            .chain(
                channel
                    .drivers
                    .iter()
                    .flatten()
                    .flat_map(|driver| &driver.plugins),
            )
            .any(|plugin| final_gain_label(plugin).is_some())
    });
    if !has_gains {
        return Ok(Vec::new());
    }
    let routing = crate::electrical_headroom::canonical_electrical_routing(graph)
        .map_err(|error| error.to_string())?;
    let paths = if let Some(routing) = routing {
        crate::electrical_headroom::expand_routed_electrical_paths(&graph.channels, routing)
    } else {
        crate::electrical_headroom::expand_independent_electrical_paths(
            &graph.channels,
            &crate::electrical_headroom::independent_graph_output_ports(&graph.channels),
        )
    }
    .map_err(|error| error.to_string())?;
    let mut records = Vec::new();
    for (path_index, path) in paths.iter().enumerate() {
        for (stage_index, stage) in path.stages.iter().enumerate() {
            for (plugin_index, plugin) in stage.plugins.iter().enumerate() {
                let Some(label) = final_gain_label(plugin) else {
                    continue;
                };
                let gain_db = plugin
                    .parameters
                    .get("gain_db")
                    .and_then(serde_json::Value::as_f64)
                    .filter(|gain| gain.is_finite())
                    .ok_or_else(|| {
                        format!(
                            "final gain on {} -> {} requires finite gain_db",
                            path.input, path.output
                        )
                    })?;
                let location = serde_json::json!([
                    path.input,
                    path.output,
                    path_index,
                    stage_index,
                    plugin_index,
                    plugin
                ]);
                let fingerprint = super::canonical_value_identity(&location).fingerprint;
                records.push(DecisionRecord {
                    decision_id: format!("{ID_PREFIX}{fingerprint}"),
                    ledger_version: super::DECISION_LEDGER_VERSION.to_owned(),
                    stage: DecisionStage::Provisional,
                    logical_input: path.input.clone(),
                    physical_output: path.output.clone(),
                    measurement_refs: Vec::new(),
                    seat_refs: Vec::new(),
                    // Broadband scalar DSP, not an observed acoustic interval.
                    frequency_band_hz: None,
                    filter_center_hz: None,
                    action: DecisionAction::GainAdjust,
                    status: DecisionStatus::Applied,
                    reason_codes: vec![
                        "serialized_final_gain".into(),
                        label.into(),
                        "branch_gain_not_acoustic_benefit_or_net_parallel_gain".into(),
                    ],
                    observed: vec![observed("delivered_gain_db", gain_db, "dB")],
                    limits: Vec::new(),
                    evidence_refs: vec![format!(
                        "serialized-electrical-path:{path_index}:stage:{stage_index}:plugin:{plugin_index}"
                    )],
                    confidence: roomeq_model::AssessmentConfidence::High,
                    related_decision_ids: Vec::new(),
                    supersedes_ids: Vec::new(),
                    final_graph_identity: None,
                });
            }
        }
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_ledger::{ReconciliationEvents, finalize_output_ledger};

    fn gain(db: f64, label: &str) -> PluginConfigWrapper {
        let mut plugin = roomeq_engine::output::create_gain_plugin(db);
        plugin.parameters["label"] = serde_json::json!(label);
        plugin
    }

    fn graph() -> DspGraph {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let chain = result.channels.get_mut("L").unwrap();
        chain.plugins = vec![gain(-2.0, "final_channel_level_alignment")];
        chain.drivers = Some(vec![
            roomeq_model::DriverDspChain {
                name: "woofer".into(),
                index: 0,
                plugins: vec![gain(-3.0, "final_electrical_headroom")],
                initial_curve: None,
                measured_band_hz: None,
            },
            roomeq_model::DriverDspChain {
                name: "tweeter".into(),
                index: 1,
                plugins: vec![gain(-1.0, "ordinary_gain")],
                initial_curve: None,
                measured_band_hz: None,
            },
        ]);
        result.to_dsp_chain_output()
    }

    #[test]
    fn final_gain_records_follow_physical_driver_branches() {
        let records = records(&graph()).unwrap();
        assert_eq!(records.len(), 3);
        let mut gains = std::collections::BTreeMap::<String, Vec<f64>>::new();
        for record in records {
            assert_eq!(record.logical_input, "L");
            assert!(record.frequency_band_hz.is_none());
            assert!(record.seat_refs.is_empty());
            gains
                .entry(record.physical_output)
                .or_default()
                .push(record.observed[0].value);
        }
        assert_eq!(gains["[\"driver\",\"L\",0,\"woofer\"]"], [-2.0, -3.0]);
        assert_eq!(gains["[\"driver\",\"L\",1,\"tweeter\"]"], [-2.0]);
    }

    #[test]
    fn final_gain_records_keep_routed_input_and_output_ownership() {
        let mut graph = crate::test_fixtures::single_channel_room_result("L").to_dsp_chain_output();
        for (name, db, stage) in [
            ("L", -2.0, "pre_route"),
            ("R", -4.0, "pre_route"),
            ("sub", -3.0, "post_route"),
        ] {
            let mut channel = crate::test_fixtures::single_channel_room_result(name)
                .channels
                .remove(name)
                .unwrap();
            let mut plugin = gain(db, "final_electrical_headroom");
            plugin.parameters["room_eq_stage"] = serde_json::json!(stage);
            channel.plugins = vec![plugin];
            graph.channels.insert(name.into(), channel);
        }
        let routes: Vec<_> = ["L", "R"]
            .iter()
            .enumerate()
            .map(|(index, name)| {
                serde_json::json!({
                    "source_channel": name, "source_index": index,
                    "destination": "sub", "destination_index": 0,
                    "pre_chain_channel": name, "post_chain_channel": "sub",
                    "route_kind": "low", "crossover_type": "LR24",
                    "gain_db": -6.0, "gain_linear": 10.0_f64.powf(-6.0 / 20.0),
                    "matrix_gain": 0.5, "delay_ms": 0.0, "polarity_inverted": index == 1
                })
            })
            .collect();
        graph.metadata.as_mut().unwrap().bass_management = Some(
            serde_json::from_value(serde_json::json!({
                "routing_title": "test", "enabled": true, "crossover_type": "LR24",
                "redirected_bass_enabled": true, "sub_trim_db": 0.0,
                "max_sub_boost_db": 0.0, "headroom_margin_db": 0.0, "gain_limited": false,
                "physical_sub_outputs": ["sub"], "redirected_bass_channel_count": 2,
                "lfe_headroom_required_db": 0.0, "signal_flow": [],
                "signal_flow_advisories": [], "advisory": "synthetic routing control",
                "routing_graph": {
                    "physical_sub_output": "sub", "input_channels": ["L", "R"],
                    "output_channels": ["sub"], "routes": routes,
                    "input_trim_db": {}, "advisories": []
                }
            }))
            .unwrap(),
        );
        let records = records(&graph).unwrap();
        assert_eq!(records.len(), 4);
        for (input, expected) in [("L", vec![-2.0, -3.0]), ("R", vec![-4.0, -3.0])] {
            let branch: Vec<_> = records
                .iter()
                .filter(|record| record.logical_input == input)
                .collect();
            assert!(branch.iter().all(|record| record.physical_output == "sub"));
            assert_eq!(
                branch
                    .iter()
                    .map(|record| record.observed[0].value)
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(
                branch
                    .iter()
                    .all(|record| record.reason_codes.iter().any(|reason| {
                        reason == "branch_gain_not_acoustic_benefit_or_net_parallel_gain"
                    }))
            );
        }
    }

    #[test]
    fn final_gain_records_refinalize_without_duplicates_and_refuse_stale_claims() {
        let mut graph = graph();
        let events = ReconciliationEvents::default();
        finalize_output_ledger(&mut graph, &[], &events).unwrap();
        let first = graph
            .correction_decisions
            .as_ref()
            .unwrap()
            .decisions
            .clone();
        assert_eq!(first.len(), 3);
        finalize_output_ledger(&mut graph, &first, &events).unwrap();
        assert_eq!(
            graph.correction_decisions.as_ref().unwrap().decisions,
            first
        );
        let mut forged = first.clone();
        forged[0].observed[0].value = 12.0;
        let error = finalize_output_ledger(&mut graph, &forged, &events).unwrap_err();
        assert!(
            error.contains("conflicting delivered-gain history"),
            "{error}"
        );
        graph.channels.get_mut("L").unwrap().plugins.clear();
        let error = finalize_output_ledger(&mut graph, &first, &events).unwrap_err();
        assert!(error.contains("stale delivered gain record"), "{error}");
        // A fresh production finalization derives only gains still delivered.
        finalize_output_ledger(&mut graph, &[], &events).unwrap();
        assert_eq!(
            graph.correction_decisions.as_ref().unwrap().decisions.len(),
            1
        );
    }

    #[test]
    fn final_gain_records_refuse_missing_or_nonfinite_values() {
        for value in [serde_json::Value::Null, serde_json::json!("NaN")] {
            let mut graph = graph();
            graph.channels.get_mut("L").unwrap().plugins[0].parameters["gain_db"] = value;
            assert!(records(&graph).is_err());
        }
    }
}
