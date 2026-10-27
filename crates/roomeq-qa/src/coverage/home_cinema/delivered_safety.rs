//! Check delivered electrical safety independently of pre-protection diagnostics.

// Rust guideline compliant 2026-02-21
use roomeq_model::{DspGraph, FinalizationConfig};
use std::{collections::BTreeSet, path::Path};

// Match the production finalizer's sampled-output roundoff allowance.
const PEAK_ROUNDOFF_DB: f64 = 1e-6;

pub(super) fn validate(
    graph: &DspGraph,
    sample_rate: f64,
    sidecar_dir: &Path,
    policy: &FinalizationConfig,
) -> Result<(), String> {
    policy.validate()?;
    let routing = roomeq_workflow::electrical_headroom::canonical_electrical_routing(graph)
        .map_err(|error| error.to_string())?;
    let mut protected = BTreeSet::new();
    if let Some(routing) = routing {
        let physical =
            roomeq_engine::physical_routing::resolve_physical_routing(&graph.channels, routing)
                .map_err(|error| error.to_string())?;
        let sub_outputs: BTreeSet<_> = routing
            .routes
            .iter()
            .filter(|route| {
                matches!(
                    route.route_kind.as_str(),
                    "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
                )
            })
            .map(|route| route.destination.as_str())
            .collect();
        if physical.inputs.iter().any(|input| {
            input
                .plugins
                .iter()
                .any(|plugin| plugin.plugin_type == "limiter")
        }) {
            return Err("output limiter precedes route summation".into());
        }
        for output in &physical.outputs {
            for (index, plugin) in output
                .plugins
                .iter()
                .enumerate()
                .filter(|(_, plugin)| plugin.plugin_type == "limiter")
            {
                let ceiling = roomeq_engine::runtime_limiter::ceiling(plugin)
                    .map_err(|error| error.to_string())?;
                if !policy.subwoofer_limiter
                    || !sub_outputs.contains(output.name.as_str())
                    || index + 1 != output.plugins.len()
                    || ceiling > policy.output_ceiling_dbfs
                {
                    return Err(format!(
                        "unauthorized or nonterminal limiter on '{}'",
                        output.name
                    ));
                }
                protected.insert(output.name.clone());
            }
        }
        if policy.subwoofer_limiter
            && protected
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                != sub_outputs
        {
            return Err("required physical sub-output protection is missing".into());
        }
    } else if graph.channels.values().any(|channel| {
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
            .any(|plugin| plugin.plugin_type == "limiter")
    }) {
        return Err("output limiter has no explicit authorized bass routing".into());
    }
    let outputs = roomeq_workflow::electrical_headroom::assess_final_graph(
        graph,
        sample_rate,
        sidecar_dir,
        policy,
    )
    .map_err(|error| error.to_string())?;
    if outputs.is_empty() {
        return Err("delivered electrical output coverage is empty".into());
    }
    for output in outputs {
        if !protected.contains(&output.output)
            && output.peak_dbfs.is_some_and(|peak| {
                !peak.is_finite() || peak > policy.output_ceiling_dbfs + PEAK_ROUNDOFF_DB
            })
        {
            return Err(format!(
                "delivered output '{}' exceeds {:.3} dBFS ceiling: {:?}",
                output.output, policy.output_ceiling_dbfs, output.peak_dbfs
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::PluginConfigWrapper;

    fn graph(gain_db: f64) -> DspGraph {
        let mut graph = DspGraph::new("qa-fixture");
        graph.add_channel("L", vec![]);
        graph
            .channels
            .get_mut("L")
            .unwrap()
            .plugins
            .push(PluginConfigWrapper {
                plugin_type: "gain".into(),
                parameters: serde_json::json!({"gain_db": gain_db}),
            });
        graph
    }

    #[test]
    fn replays_delivered_static_attenuation() {
        assert!(
            validate(
                &graph(-3.0),
                48_000.0,
                Path::new("."),
                &FinalizationConfig::default()
            )
            .is_ok()
        );
    }

    #[test]
    fn changed_delivered_gain_cannot_reuse_a_safe_result() {
        let mut delivered = graph(-3.0);
        assert!(
            validate(
                &delivered,
                48_000.0,
                Path::new("."),
                &FinalizationConfig::default()
            )
            .is_ok()
        );
        delivered.channels.get_mut("L").unwrap().plugins[0].parameters["gain_db"] =
            serde_json::json!(6.0);
        assert!(
            validate(
                &delivered,
                48_000.0,
                Path::new("."),
                &FinalizationConfig::default()
            )
            .is_err()
        );
    }

    #[test]
    fn checks_every_physical_output() {
        let mut delivered = graph(-3.0);
        let mut unsafe_output = delivered.channels["L"].clone();
        unsafe_output.channel = "R".into();
        unsafe_output.plugins[0].parameters["gain_db"] = serde_json::json!(6.0);
        delivered.channels.insert("R".into(), unsafe_output);
        assert!(
            validate(
                &delivered,
                48_000.0,
                Path::new("."),
                &FinalizationConfig::default()
            )
            .unwrap_err()
            .contains('R')
        );
    }

    #[test]
    fn missing_fir_resource_is_not_safety_evidence() {
        let mut delivered = graph(-3.0);
        delivered
            .channels
            .get_mut("L")
            .unwrap()
            .plugins
            .push(PluginConfigWrapper {
                plugin_type: "convolution".into(),
                parameters: serde_json::json!({"ir_file": "missing-qa-fixture.wav"}),
            });
        assert!(
            validate(
                &delivered,
                48_000.0,
                Path::new("."),
                &FinalizationConfig::default()
            )
            .is_err()
        );
    }

    #[test]
    fn limiter_without_authorized_routing_is_rejected() {
        let mut delivered = graph(6.0);
        delivered
            .channels
            .get_mut("L")
            .unwrap()
            .plugins
            .push(roomeq_engine::runtime_limiter::plugin(-1.0));
        assert!(
            validate(
                &delivered,
                48_000.0,
                Path::new("."),
                &FinalizationConfig::default()
            )
            .is_err()
        );
    }
}
