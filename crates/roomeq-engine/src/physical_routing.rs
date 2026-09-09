//! Resolve legacy RoomEQ ownership into the shared physical routing contract.
//!
//! This is independent of native plugin-node types. Rendering and electrical
//! assessment must consume this same contract, rather than infer ownership again.

use roomeq_model::{
    AutoeqError, BassManagementRoutingGraph, ChannelDspChain, PhysicalRoute,
    PhysicalRouteCrossover, PhysicalRoutePass, PhysicalRoutingGraph, PhysicalRoutingPort,
    PluginConfigWrapper, Result,
};
use std::collections::{BTreeSet, HashMap};

fn invalid(message: impl Into<String>) -> AutoeqError {
    AutoeqError::InvalidMeasurement {
        message: message.into(),
    }
}

fn stage(plugin: &PluginConfigWrapper) -> Option<&str> {
    plugin
        .parameters
        .get("room_eq_stage")
        .and_then(|v| v.as_str())
}

fn selected(chain: &ChannelDspChain, owner: &str) -> Vec<PluginConfigWrapper> {
    chain
        .plugins
        .iter()
        .filter(|p| stage(p) == Some(owner))
        .cloned()
        .collect()
}

/// Resolve already expanded channels, or the home-cinema hierarchy whose common
/// sub chain is explicitly named by `physical_sub_output`. In that hierarchy the
/// producer has already baked driver gain/delay/polarity into each bass route.
/// Driver EQ/FIR processing remains physical-output-owned and is not discarded.
/// Unrecognized hierarchy/ownership is an error, not guessed routing.
pub fn resolve_physical_routing(
    channels: &HashMap<String, ChannelDspChain>,
    graph: &BassManagementRoutingGraph,
) -> Result<PhysicalRoutingGraph> {
    let shared = channels.get(&graph.physical_sub_output);
    let shared_drivers = shared.and_then(|c| c.drivers.as_deref()).unwrap_or(&[]);
    let hierarchical = !shared_drivers.is_empty();
    for (name, chain) in channels {
        if name != &chain.channel {
            return Err(invalid(
                "channel key disagrees with embedded physical identity",
            ));
        }
        if chain.drivers.as_ref().is_some_and(|d| !d.is_empty())
            && (!hierarchical || name != &graph.physical_sub_output)
        {
            return Err(invalid("unsupported physical driver hierarchy"));
        }
        for plugin in &chain.plugins {
            match stage(plugin) {
                Some("pre_route" | "post_route") => {}
                Some("route_owned")
                    if matches!(plugin.plugin_type.as_str(), "gain" | "delay" | "crossover") => {}
                // Legacy exports left redundant crossovers untagged. A matching
                // explicit route must prove ownership before we omit one.
                None if legacy_route_crossover(plugin, name, graph) => {}
                _ => {
                    return Err(invalid(format!(
                        "channel {name} has unresolved plugin ownership"
                    )));
                }
            }
        }
    }
    let mut driver_names = BTreeSet::new();
    let mut driver_indices = BTreeSet::new();
    for driver in shared_drivers {
        if driver.name.is_empty()
            || !driver_names.insert(&driver.name)
            || !driver_indices.insert(driver.index)
            || !graph.output_channels.contains(&driver.name)
        {
            return Err(invalid(
                "physical sub drivers require unique, routed identities",
            ));
        }
    }
    let inputs = graph
        .input_channels
        .iter()
        .map(|name| {
            let chain = channels.get(name);
            if chain.is_none()
                && (graph
                    .routes
                    .iter()
                    .any(|route| &route.source_channel == name)
                    || !graph.output_channels.contains(name))
            {
                return Err(invalid("missing physical input chain"));
            }
            Ok(PhysicalRoutingPort {
                name: name.clone(),
                plugins: chain.map(|c| selected(c, "pre_route")).unwrap_or_default(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let outputs = graph
        .output_channels
        .iter()
        .map(|name| {
            let plugins = if let Some(driver) = shared_drivers.iter().find(|d| &d.name == name) {
                if channels.contains_key(name) {
                    return Err(invalid(
                        "physical output has both driver and channel ownership",
                    ));
                }
                let mut plugins =
                    selected(shared.expect("drivers require shared chain"), "post_route");
                for plugin in &driver.plugins {
                    if stage(plugin) != Some("post_route") {
                        return Err(invalid(
                            "physical sub driver has unresolved plugin ownership",
                        ));
                    }
                    match plugin.plugin_type.as_str() {
                        // These controls are already included in the producer's route values.
                        "gain" | "delay" => (),
                        // Linear residual processing stays after the common sub correction.
                        "eq" | "convolution" => plugins.push(plugin.clone()),
                        _ => return Err(invalid("unsupported physical sub driver processing")),
                    }
                }
                plugins
            } else {
                let chain = channels
                    .get(name)
                    .ok_or_else(|| invalid("missing expanded physical output chain"))?;
                if hierarchical && name == &graph.physical_sub_output {
                    if graph.routes.iter().any(|r| &r.destination == name) {
                        return Err(invalid(
                            "hierarchical sub parent cannot also be a physical destination",
                        ));
                    }
                    // Legacy channel order retains the now-unused logical LFE slot.
                    Vec::new()
                } else {
                    selected(chain, "post_route")
                }
            };
            Ok(PhysicalRoutingPort {
                name: name.clone(),
                plugins,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let routes = graph
        .routes
        .iter()
        .map(|route| {
            let is_sub = matches!(
                route.route_kind.as_str(),
                "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
            );
            if graph.input_channels.get(route.source_index) != Some(&route.source_channel)
                || graph.output_channels.get(route.destination_index) != Some(&route.destination)
                || route
                    .post_chain_channel
                    .as_ref()
                    .is_some_and(|c| c != &route.destination)
                || route.pre_chain_channel.as_ref().is_some_and(|c| {
                    c != &route.source_channel && !(is_sub && c == &graph.physical_sub_output)
                })
            {
                return Err(invalid(
                    "physical route has inconsistent endpoint chain ownership",
                ));
            }
            if hierarchical && driver_names.contains(&route.destination) && !is_sub {
                return Err(invalid(
                    "driver controls are only resolved for home-cinema bass routes",
                ));
            }
            if !route.gain_linear.is_finite()
                || route.gain_linear <= 0.0
                || !route.gain_db.is_finite()
                || (20.0 * route.gain_linear.log10() - route.gain_db).abs() > 1e-6
                || !route.matrix_gain.is_finite()
                || (route.high_pass_hz.is_some() && route.low_pass_hz.is_some())
            {
                return Err(invalid(
                    "physical route has invalid or contradictory gain/crossover",
                ));
            }
            let crossover = route
                .high_pass_hz
                .or(route.low_pass_hz)
                .map(|frequency_hz| PhysicalRouteCrossover {
                    crossover_type: route.crossover_type.clone(),
                    frequency_hz,
                    pass: if route.high_pass_hz.is_some() {
                        PhysicalRoutePass::High
                    } else {
                        PhysicalRoutePass::Low
                    },
                });
            Ok(PhysicalRoute {
                input_index: route.source_index,
                output_index: route.destination_index,
                gain_db: route.gain_db,
                polarity_inverted: route.polarity_inverted,
                delay_ms: route.delay_ms,
                crossover,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut physical = PhysicalRoutingGraph {
        inputs,
        outputs,
        routes,
    };
    physical.canonicalize().map_err(invalid)?;
    for (index, output) in physical.outputs.iter().enumerate() {
        if !physical.routes.iter().any(|r| r.output_index == index)
            && !(hierarchical && output.name == graph.physical_sub_output)
        {
            return Err(invalid(format!(
                "physical output {} has no incoming route",
                output.name
            )));
        }
    }
    Ok(physical)
}

fn legacy_route_crossover(
    plugin: &PluginConfigWrapper,
    channel: &str,
    graph: &BassManagementRoutingGraph,
) -> bool {
    if plugin.plugin_type != "crossover" {
        return false;
    }
    let parameters = &plugin.parameters;
    graph.routes.iter().any(|route| {
        let high = parameters["output"] == "high";
        let low = parameters["output"] == "low";
        let frequency = if high {
            route.high_pass_hz
        } else if low {
            route.low_pass_hz
        } else {
            None
        };
        let owns = if high {
            route.source_channel == channel && route.destination == channel
        } else {
            channel == graph.physical_sub_output
                && matches!(
                    route.route_kind.as_str(),
                    "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
                )
        };
        owns && parameters["type"].as_str() == Some(route.crossover_type.as_str())
            && frequency.is_some()
            && parameters["frequency"].as_f64() == frequency
    })
}

/// Serialize only the complete per-route transfer for the existing DSP evaluator.
/// Port processing is owned separately by `PhysicalRoutingGraph`.
pub fn physical_route_plugins(route: &PhysicalRoute) -> Vec<PluginConfigWrapper> {
    let mut plugins = vec![crate::output::create_gain_plugin_with_invert(
        route.gain_db,
        route.polarity_inverted,
    )];
    if let Some(crossover) = &route.crossover {
        plugins.push(crate::output::create_crossover_plugin(
            &crossover.crossover_type,
            crossover.frequency_hz,
            match crossover.pass {
                PhysicalRoutePass::High => "high",
                PhysicalRoutePass::Low => "low",
            },
        ));
    }
    // Preserve even sub-microsecond delays; no consumer-specific threshold.
    if route.delay_ms > 0.0 {
        plugins.push(crate::output::create_delay_plugin(route.delay_ms));
    }
    plugins
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::{BassManagementRoute, DriverDspChain};
    use serde_json::json;

    fn tagged(kind: &str, mut parameters: serde_json::Value, owner: &str) -> PluginConfigWrapper {
        parameters["room_eq_stage"] = json!(owner);
        PluginConfigWrapper {
            plugin_type: kind.into(),
            parameters,
        }
    }

    fn fixture() -> (HashMap<String, ChannelDspChain>, BassManagementRoutingGraph) {
        let mut channels: HashMap<String, ChannelDspChain> = ["L", "R", "LFE"]
            .into_iter()
            .map(|name| {
                let chain =
                    serde_json::from_value(json!({"channel": name, "plugins": []})).unwrap();
                (name.into(), chain)
            })
            .collect();
        channels.get_mut("L").unwrap().plugins.push(tagged(
            "gain",
            json!({"gain_db": -3.0}),
            "pre_route",
        ));
        let sub = channels.get_mut("LFE").unwrap();
        sub.plugins
            .push(tagged("gain", json!({"gain_db": -7.0}), "pre_route"));
        for count in [9, 4] {
            sub.plugins.push(tagged(
                "eq",
                json!({"filters": vec![
                    json!({"filter_type": "peak", "freq": 70.0, "q": 1.0, "db_gain": 0.0}); count
                ]}),
                "post_route",
            ));
        }
        sub.plugins
            .push(tagged("gain", json!({"gain_db": -6.0}), "post_route"));
        sub.plugins
            .push(tagged("gain", json!({"gain_db": 4.0}), "route_owned"));
        sub.drivers = Some(vec![
            DriverDspChain {
                name: "subs_1".into(),
                index: 0,
                initial_curve: None,
                plugins: vec![tagged("delay", json!({"delay_ms": 9.0}), "post_route")],
            },
            DriverDspChain {
                name: "subs_2".into(),
                index: 1,
                initial_curve: None,
                plugins: vec![tagged("gain", json!({"gain_db": -5.0}), "post_route")],
            },
        ]);
        let mut graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".into(),
            input_channels: vec!["L".into(), "R".into(), "LFE".into()],
            output_channels: vec![
                "L".into(),
                "R".into(),
                "LFE".into(),
                "subs_1".into(),
                "subs_2".into(),
            ],
            routes: vec![],
            matrix: None,
            input_trim_db: HashMap::from([("L".into(), -3.0)]),
            advisories: vec![],
        };
        for (source_index, source) in ["L", "R", "LFE"].into_iter().enumerate() {
            for (destination_index, destination, gain_db, delay_ms) in
                [(3, "subs_1", 4.0, 9.0), (4, "subs_2", -1.0, 0.0)]
            {
                graph.routes.push(BassManagementRoute {
                    group_id: None,
                    source_channel: source.into(),
                    source_index,
                    destination: destination.into(),
                    destination_index,
                    pre_chain_channel: Some("LFE".into()),
                    post_chain_channel: Some(destination.into()),
                    route_kind: if source == "LFE" {
                        "lfe_lowpass_to_sub"
                    } else {
                        "redirected_bass_lowpass_to_sub"
                    }
                    .into(),
                    crossover_type: "LR48".into(),
                    high_pass_hz: None,
                    low_pass_hz: Some(if source == "LFE" { 120.0 } else { 80.0 }),
                    gain_db,
                    gain_linear: 10.0_f64.powf(gain_db / 20.0),
                    matrix_gain: 10.0_f64.powf(gain_db / 20.0),
                    delay_ms,
                    polarity_inverted: destination_index == 4,
                });
            }
        }
        for index in 0..2 {
            let mut route = graph.routes[index * 2].clone();
            route.destination = route.source_channel.clone();
            route.destination_index = index;
            route.pre_chain_channel = Some(route.source_channel.clone());
            route.post_chain_channel = Some(route.source_channel.clone());
            route.route_kind = "main_highpass_to_self".into();
            route.high_pass_hz = route.low_pass_hz.take();
            route.gain_db = 0.0;
            route.gain_linear = 1.0;
            route.matrix_gain = 1.0;
            route.delay_ms = 0.0;
            graph.routes.push(route);
        }
        (channels, graph)
    }

    #[test]
    fn physical_routing_preserves_sub_delays_eq_and_single_control_ownership() {
        let (channels, graph) = fixture();
        let physical = resolve_physical_routing(&channels, &graph).unwrap();
        assert_eq!(physical.inputs[0].plugins[0].parameters["gain_db"], -3.0);
        assert_eq!(physical.inputs[2].plugins[0].parameters["gain_db"], -7.0);
        assert!(
            physical.outputs[2].plugins.is_empty(),
            "legacy LFE slot is silent"
        );
        for index in [3, 4] {
            let plugins = &physical.outputs[index].plugins;
            let eq_count: usize = plugins
                .iter()
                .filter(|p| p.plugin_type == "eq")
                .map(|p| p.parameters["filters"].as_array().unwrap().len())
                .sum();
            assert_eq!(eq_count, 13);
            assert_eq!(
                plugins.len(),
                3,
                "no duplicate driver or route-owned controls"
            );
            assert_eq!(plugins[2].parameters["gain_db"], -6.0);
        }
        assert_eq!(physical.routes.len(), 8);
        for route in physical.routes.iter().filter(|r| r.output_index >= 3) {
            assert_eq!(
                route.delay_ms,
                if route.output_index == 3 { 9.0 } else { 0.0 }
            );
            assert_eq!(
                route.gain_db,
                if route.output_index == 3 { 4.0 } else { -1.0 }
            );
            assert_eq!(route.polarity_inverted, route.output_index == 4);
            assert_eq!(route.crossover.as_ref().unwrap().crossover_type, "LR48");
        }
    }

    #[test]
    fn physical_routing_is_permutation_invariant_and_roundtrips() {
        let (channels, mut graph) = fixture();
        let before =
            serde_json::to_value(resolve_physical_routing(&channels, &graph).unwrap()).unwrap();
        graph.routes.reverse();
        let after =
            serde_json::to_value(resolve_physical_routing(&channels, &graph).unwrap()).unwrap();
        assert_eq!(before, after);
        let decoded: PhysicalRoutingGraph = serde_json::from_value(before.clone()).unwrap();
        decoded.validate().unwrap();
        assert_eq!(before, serde_json::to_value(decoded).unwrap());
    }

    #[test]
    fn physical_routing_retains_driver_eq_and_tiny_route_delay() {
        let (mut channels, mut graph) = fixture();
        let driver = &mut channels.get_mut("LFE").unwrap().drivers.as_mut().unwrap()[0];
        driver
            .plugins
            .push(tagged("eq", json!({"filters": []}), "post_route"));
        graph.routes[0].delay_ms = 0.0001;
        let physical = resolve_physical_routing(&channels, &graph).unwrap();
        assert_eq!(physical.outputs[3].plugins.len(), 4);
        let plugins = physical_route_plugins(
            physical
                .routes
                .iter()
                .find(|r| r.input_index == 0 && r.output_index == 3)
                .unwrap(),
        );
        assert_eq!(plugins.last().unwrap().parameters["delay_ms"], 0.0001);
        assert_eq!(plugins[1].parameters["type"], "LR48");
    }

    #[test]
    fn physical_routing_rejects_ambiguous_ownership_and_invalid_controls() {
        for mutation in 0..8 {
            let (mut channels, mut graph) = fixture();
            match mutation {
                0 => graph.routes[0].source_index = 99,
                1 => graph.routes[0].gain_linear = 9.0,
                2 => graph.routes[0].delay_ms = -1.0,
                3 => graph.routes[0].pre_chain_channel = Some("R".into()),
                4 => graph.routes[0].high_pass_hz = Some(80.0),
                5 => {
                    channels.get_mut("LFE").unwrap().drivers.as_mut().unwrap()[1].name =
                        "subs_1".into()
                }
                6 => {
                    channels.get_mut("L").unwrap().plugins[0].parameters["room_eq_stage"] =
                        json!("unknown")
                }
                _ => graph.output_channels[4] = "subs_1".into(),
            }
            assert!(
                resolve_physical_routing(&channels, &graph).is_err(),
                "mutation {mutation}"
            );
        }
    }

    #[test]
    fn physical_routing_migrates_only_proven_legacy_crossover_ownership() {
        let (mut channels, mut graph) = fixture();
        let crossover = PluginConfigWrapper {
            plugin_type: "crossover".into(),
            parameters: json!({"type": "LR48", "frequency": 80.0, "output": "high"}),
        };
        channels.get_mut("L").unwrap().plugins.push(crossover);
        // Legacy native exports padded the input map with destination-only slots.
        graph.input_channels.push("subs_1".into());
        let resolved = resolve_physical_routing(&channels, &graph).unwrap();
        assert!(resolved.inputs[3].plugins.is_empty());
        assert_eq!(
            resolved.inputs[0].plugins.len(),
            1,
            "crossover belongs only to its route"
        );
        channels
            .get_mut("L")
            .unwrap()
            .plugins
            .last_mut()
            .unwrap()
            .parameters["frequency"] = json!(90.0);
        assert!(resolve_physical_routing(&channels, &graph).is_err());
    }
}
