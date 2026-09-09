//! Backend-neutral, fully resolved physical routing.
//!
//! Each input is processed once, then fanned out through its routes. Contributions
//! are summed at each output **before** that output's plugins run. Port order is
//! significant; route record order is not. A declared output with no routes is
//! silent. Names identify graph ports, not operating-system audio devices.
//!
//! Gains, polarity, crossovers and delays in a route are the complete route
//! controls: consumers must not additionally apply legacy matrix gains, input
//! trim metadata, route-owned channel plugins, or baked-in driver controls.

use crate::PluginConfigWrapper;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhysicalRoutingGraph {
    pub inputs: Vec<PhysicalRoutingPort>,
    pub outputs: Vec<PhysicalRoutingPort>,
    pub routes: Vec<PhysicalRoute>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhysicalRoutingPort {
    pub name: String,
    /// Ordered backend-neutral processing. No nested drivers or implicit stages.
    pub plugins: Vec<PluginConfigWrapper>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhysicalRoute {
    pub input_index: usize,
    pub output_index: usize,
    pub gain_db: f64,
    pub polarity_inverted: bool,
    pub delay_ms: f64,
    pub crossover: Option<PhysicalRouteCrossover>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhysicalRouteCrossover {
    /// The requested family/order; consumers must not substitute LR24.
    pub crossover_type: String,
    pub frequency_hz: f64,
    pub pass: PhysicalRoutePass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalRoutePass {
    High,
    Low,
}

impl PhysicalRoutingGraph {
    /// Structural validation only. Backend support, filter stability and sampled
    /// or continuous electrical/acoustic safety require separate checks.
    pub fn validate(&self) -> Result<(), String> {
        for (kind, ports) in [("input", &self.inputs), ("output", &self.outputs)] {
            let mut names = BTreeSet::new();
            if ports.is_empty() {
                return Err(format!("physical routing requires {kind} ports"));
            }
            for port in ports {
                if port.name.trim().is_empty() || !names.insert(&port.name) {
                    return Err(format!(
                        "physical routing has empty/duplicate {kind} identity"
                    ));
                }
                if port.plugins.iter().any(|plugin| {
                    plugin.plugin_type.trim().is_empty() || !plugin.parameters.is_object()
                }) {
                    return Err(format!(
                        "physical {kind} {} has an invalid plugin",
                        port.name
                    ));
                }
            }
        }
        if self.routes.is_empty() {
            return Err("physical routing requires routes".into());
        }
        for route in &self.routes {
            if route.input_index >= self.inputs.len() || route.output_index >= self.outputs.len() {
                return Err("physical route endpoint index is out of range".into());
            }
            if !route.gain_db.is_finite() || !route.delay_ms.is_finite() || route.delay_ms < 0.0 {
                return Err("physical route has invalid gain/delay".into());
            }
            if let Some(crossover) = &route.crossover
                && (crossover.crossover_type.trim().is_empty()
                    || !crossover.frequency_hz.is_finite()
                    || crossover.frequency_hz <= 0.0)
            {
                return Err("physical route has invalid crossover".into());
            }
        }
        Ok(())
    }

    /// Produce deterministic route order without merging distinct transfers or
    /// reordering processing within ports. Duplicate paths remain additive.
    pub fn canonicalize(&mut self) -> Result<(), String> {
        self.validate()?;
        self.routes.sort_by(|a, b| {
            fn crossover_key(route: &PhysicalRoute) -> Option<(&str, bool)> {
                route
                    .crossover
                    .as_ref()
                    .map(|c| (c.crossover_type.as_str(), c.pass == PhysicalRoutePass::High))
            }
            (a.input_index, a.output_index)
                .cmp(&(b.input_index, b.output_index))
                .then_with(|| crossover_key(a).cmp(&crossover_key(b)))
                .then_with(|| {
                    a.crossover
                        .as_ref()
                        .map_or(0.0, |c| c.frequency_hz)
                        .total_cmp(&b.crossover.as_ref().map_or(0.0, |c| c.frequency_hz))
                })
                .then_with(|| a.gain_db.total_cmp(&b.gain_db))
                .then_with(|| a.delay_ms.total_cmp(&b.delay_ms))
                .then_with(|| a.polarity_inverted.cmp(&b.polarity_inverted))
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn graph() -> PhysicalRoutingGraph {
        PhysicalRoutingGraph {
            inputs: vec![PhysicalRoutingPort {
                name: "L".into(),
                plugins: vec![],
            }],
            outputs: vec![PhysicalRoutingPort {
                name: "sub".into(),
                plugins: vec![],
            }],
            routes: vec![PhysicalRoute {
                input_index: 0,
                output_index: 0,
                gain_db: -3.0,
                polarity_inverted: false,
                delay_ms: 0.0,
                crossover: Some(PhysicalRouteCrossover {
                    crossover_type: "LR24".into(),
                    frequency_hz: 80.0,
                    pass: PhysicalRoutePass::Low,
                }),
            }],
        }
    }

    #[test]
    fn physical_contract_canonicalization_preserves_distinct_and_duplicate_routes() {
        let mut graph = graph();
        let mut delayed = graph.routes[0].clone();
        delayed.delay_ms = 9.0;
        graph.routes.push(delayed);
        graph.routes.push(graph.routes[0].clone());
        graph.canonicalize().unwrap();
        let before = serde_json::to_value(&graph).unwrap();
        graph.routes.reverse();
        graph.canonicalize().unwrap();
        assert_eq!(before, serde_json::to_value(&graph).unwrap());
        assert_eq!(
            graph.routes.len(),
            3,
            "duplicate contributions are additive"
        );
        assert_eq!(graph.routes[2].delay_ms, 9.0);
    }

    #[test]
    fn physical_contract_rejects_invalid_ports_controls_and_indices() {
        for mutation in 0..10 {
            let mut graph = graph();
            match mutation {
                0 => graph.inputs.clear(),
                1 => graph.outputs[0].name.clear(),
                2 => graph.outputs.push(graph.outputs[0].clone()),
                3 => graph.routes[0].input_index = 1,
                4 => graph.routes[0].output_index = 1,
                5 => graph.routes[0].gain_db = f64::INFINITY,
                6 => graph.routes[0].delay_ms = -1.0,
                7 => graph.routes[0].crossover.as_mut().unwrap().frequency_hz = f64::NAN,
                8 => graph.routes.clear(),
                _ => graph.outputs[0].plugins.push(PluginConfigWrapper {
                    plugin_type: "gain".into(),
                    parameters: json!(null),
                }),
            }
            assert!(graph.canonicalize().is_err(), "mutation {mutation}");
        }
    }

    #[test]
    fn physical_contract_refuses_unknown_ownership_fields() {
        let mut value = serde_json::to_value(graph()).unwrap();
        value["outputs"][0]["drivers"] = json!([{"name": "hidden"}]);
        assert!(serde_json::from_value::<PhysicalRoutingGraph>(value).is_err());
    }
}
