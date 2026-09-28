//! Physical sub-output protection, installed after route summation and output EQ.

use super::*;
use std::collections::BTreeSet;

const COMPENSATION: &str = "room_eq_limiter_latency";

pub(super) fn sub_outputs(result: &RoomOptimizationResult) -> BTreeSet<String> {
    result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
        .map(|graph| {
            graph
                .routes
                .iter()
                .filter(|route| {
                    matches!(
                        route.route_kind.as_str(),
                        "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
                    )
                })
                .map(|route| route.destination.clone())
                .collect()
        })
        .unwrap_or_default()
}

pub(super) fn install(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
    fs: f64,
) -> Result<()> {
    if !config.optimizer.finalization.subwoofer_limiter {
        return Ok(());
    }
    let subs = sub_outputs(result);
    if subs.is_empty() {
        return Ok(());
    }
    let graph = result
        .metadata
        .bass_management
        .as_ref()
        .unwrap()
        .routing_graph
        .as_ref()
        .unwrap();
    let outputs: BTreeSet<_> = graph
        .routes
        .iter()
        .map(|route| route.destination.clone())
        .collect();
    let delay_ms = roomeq_engine::runtime_limiter::latency_samples(fs) as f64 * 1000.0 / fs;
    for output in outputs {
        let plugin = if subs.contains(&output) {
            roomeq_engine::runtime_limiter::plugin(
                config.optimizer.finalization.output_ceiling_dbfs,
            )
        } else {
            PluginConfigWrapper {
                plugin_type: "delay".into(),
                parameters: serde_json::json!({
                    "delay_ms": delay_ms, "room_eq_stage": "post_route", "label": COMPENSATION
                }),
            }
        };
        let mut installed = false;
        for (name, chain) in &mut result.channels {
            let plugins = if let Some(drivers) = &mut chain.drivers {
                drivers
                    .iter_mut()
                    .find(|driver| driver.name == output)
                    .map(|driver| &mut driver.plugins)
            } else if *name == output {
                Some(&mut chain.plugins)
            } else {
                None
            };
            if let Some(plugins) = plugins {
                if plugins.iter().any(|p| {
                    p.plugin_type == "limiter"
                        || p.parameters["label"].as_str() == Some(COMPENSATION)
                }) {
                    return Err(failed("runtime output protection already present"));
                }
                plugins.push(plugin.clone());
                installed = true;
            }
        }
        if !installed {
            return Err(failed(format!("missing limiter output owner '{output}'")));
        }
    }
    protected_outputs(result, &config.optimizer.finalization)?;
    Ok(())
}

/// Validate placement as well as settings before trusting dynamic protection.
pub(super) fn protected_outputs(
    result: &RoomOptimizationResult,
    policy: &roomeq_model::FinalizationConfig,
) -> Result<BTreeSet<String>> {
    let Some(graph) = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
    else {
        return Ok(BTreeSet::new());
    };
    let physical =
        roomeq_engine::physical_routing::resolve_physical_routing(&result.channels, graph)?;
    let subs = sub_outputs(result);
    let mut protected = BTreeSet::new();
    for input in &physical.inputs {
        if input.plugins.iter().any(|p| p.plugin_type == "limiter") {
            return Err(failed(
                "sub-output limiter must follow summation, not precede routing",
            ));
        }
    }
    for output in &physical.outputs {
        for (index, plugin) in output
            .plugins
            .iter()
            .enumerate()
            .filter(|(_, p)| p.plugin_type == "limiter")
        {
            let ceiling = roomeq_engine::runtime_limiter::ceiling(plugin)?;
            if !policy.subwoofer_limiter
                || !subs.contains(&output.name)
                || index + 1 != output.plugins.len()
                || ceiling > policy.output_ceiling_dbfs
            {
                return Err(failed(
                    "runtime limiter must be the terminal processor on an authorized sub output",
                ));
            }
            protected.insert(output.name.clone());
        }
    }
    if policy.subwoofer_limiter && protected != subs {
        return Err(failed("runtime sub-output protection is missing"));
    }
    Ok(protected)
}
