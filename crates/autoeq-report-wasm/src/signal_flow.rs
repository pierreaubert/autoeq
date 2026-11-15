//! Saved plugin and routing diagrams for DSP reports.

// Rust guideline compliant 2026-02-21

use serde_json::{Map, Value, json};
use std::collections::HashMap;

use crate::report_document::{
    DiagnosticBlock, SignalFlowStage, diagnostic_section, paragraph_section, project_section,
};

const LINK_COLOR: &str = "rgba(74,144,217,0.25)";

/// Builds report sections that show saved DSP stages and routing ownership.
///
/// The first returned section explains the diagrams. Each graph view is
/// followed by an HTML parameter table. Channel views use the report's
/// canonical channel order, and routed views require explicit plugin ownership
/// before assigning stages.
///
/// # Examples
///
/// ```
/// use autoeq_report_wasm::signal_flow_sections;
/// use serde_json::json;
///
/// let sections = signal_flow_sections(&json!({"channels": {"L": {"plugins": []}}}));
/// assert!(sections.iter().any(|section| section["kind"] == "sankey"));
/// ```
///
/// # Errors
///
/// Invalid or incomplete topology is represented by an HTML section that
/// states why signal flow is unavailable.
///
/// # Panics
///
/// This function does not panic for malformed report values.
pub fn signal_flow_sections(data: &Value) -> Vec<Value> {
    let mut sections = signal_flow_document_sections(data);
    for section in &mut sections {
        project_section(section).expect("signal-flow sections are serialized by AutoEQ");
    }
    sections
}

pub(crate) fn signal_flow_document_sections(data: &Value) -> Vec<Value> {
    let sections = vec![flow_paragraph(
        None,
        "Read left to right. Each numbered node is a saved processing stage; expand its row below for parameters. Branches join at Σ before output processing. Link widths show connections, not signal level. Select All channels for the complete graph, or an output for its inputs.",
    )];

    let channels = data.get("channels").and_then(Value::as_object);
    let bass = data
        .pointer("/metadata/bass_management")
        .unwrap_or(&Value::Null);
    let mut routing = bass.pointer("/routing_graph").unwrap_or(&Value::Null);
    let global_plugins = array(data.get("global_plugins"));

    if array(routing.get("routes")).is_empty() {
        for plugin in &global_plugins {
            let metadata = plugin
                .pointer("/parameters/metadata")
                .unwrap_or(&Value::Null);
            if metadata.get("routes").is_some() {
                routing = metadata;
                break;
            }
        }
    }

    let routes = array(routing.get("routes"));
    if routes
        .iter()
        .any(|route| route.get("destination").and_then(Value::as_str).is_none())
    {
        return vec![unavailable_section(
            None,
            "a route has no destination output",
            true,
        )];
    }
    if !routes.is_empty()
        && global_plugins
            .iter()
            .any(|plugin| plugin.pointer("/parameters/metadata/routes").is_none())
    {
        return vec![flow_paragraph(
            None,
            "Signal flow unavailable: additional global processing has no explicit position in the saved routing graph.",
        )];
    }
    let Some(channels) = channels.filter(|channels| !channels.is_empty()) else {
        return vec![flow_paragraph(
            None,
            "DSP signal flow unavailable: no saved channel chains.",
        )];
    };

    if routes.is_empty() {
        return build_unrouted(channels, &global_plugins, sections);
    }
    build_routed(data, channels, bass, &routes, sections)
}

fn build_unrouted(
    channels: &Map<String, Value>,
    global_plugins: &[&Value],
    mut sections: Vec<Value>,
) -> Vec<Value> {
    let mut names: Vec<_> = channels.keys().map(String::as_str).collect();
    names.sort_by(|left, right| {
        crate::report_helpers::channel_sort_key(left)
            .cmp(&crate::report_helpers::channel_sort_key(right))
    });
    let mut views = vec![("All channels".to_string(), names.clone())];
    views.extend(names.iter().map(|name| ((*name).to_string(), vec![*name])));

    for (tab, outputs) in views {
        let mut diagram = Diagram::new(format!("Saved DSP: {tab}"), tab.clone());
        let mut unavailable = None;
        for name in outputs {
            let channel = channels.get(name).unwrap_or(&Value::Null);
            if !array(channel.get("drivers")).is_empty() || !global_plugins.is_empty() {
                unavailable = Some("this saved graph has driver or global processing without explicit routing. No connection order was guessed".to_string());
                break;
            }
            let start = diagram.node(&format!("Input {name}"), None);
            let end = diagram.chain(start, array(channel.get("plugins")));
            let output = diagram.node(&format!("Output {name}"), None);
            diagram.edge(end, output);
        }
        if let Some(reason) = unavailable {
            sections.push(unavailable_section(Some(&tab), &reason, false));
        } else {
            sections.extend(diagram.sections());
        }
    }
    sections
}

fn build_routed(
    data: &Value,
    channels: &Map<String, Value>,
    bass: &Value,
    routes: &[&Value],
    mut sections: Vec<Value>,
) -> Vec<Value> {
    let routing = data
        .pointer("/metadata/bass_management/routing_graph")
        .unwrap_or(&Value::Null);
    let shared_name = routing
        .get("physical_sub_output")
        .or_else(|| bass.get("physical_sub_output"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            let parents: Vec<_> = channels
                .iter()
                .filter(|(_, channel)| !array(channel.get("drivers")).is_empty())
                .map(|(name, _)| name.clone())
                .collect();
            (parents.len() == 1).then(|| parents[0].clone())
        });
    let shared = shared_name
        .as_deref()
        .and_then(|name| channels.get(name))
        .unwrap_or(&Value::Null);
    let drivers: Map<String, Value> = array(shared.get("drivers"))
        .iter()
        .filter_map(|driver| Some((driver.get("name")?.as_str()?.to_owned(), (*driver).clone())))
        .collect();

    let mut destination_names = Vec::<String>::new();
    for route in routes {
        if let Some(destination) = route.get("destination").and_then(Value::as_str)
            && !destination_names
                .iter()
                .any(|existing| existing == destination)
        {
            destination_names.push(destination.to_owned());
        }
    }
    let mut ordered_destinations = destination_names.clone();
    ordered_destinations.sort_by(|left, right| {
        crate::report_helpers::channel_sort_key(left)
            .cmp(&crate::report_helpers::channel_sort_key(right))
    });
    let mut views = vec![("All channels".to_string(), ordered_destinations)];
    views.extend(
        destination_names
            .iter()
            .map(|name| (name.clone(), vec![name.clone()])),
    );

    for (tab, outputs) in views {
        let result = (|| {
            let mut diagram = Diagram::new(format!("Saved DSP: {tab}"), tab.clone());
            let mut inputs = HashMap::<String, usize>::new();
            for destination in outputs {
                add_destination(
                    &mut diagram,
                    &destination,
                    &mut inputs,
                    channels,
                    routes,
                    &drivers,
                    shared,
                )?;
            }
            Ok::<_, String>(diagram.sections())
        })();
        match result {
            Ok(view_sections) => sections.extend(view_sections),
            Err(reason) => sections.push(unavailable_section(Some(&tab), &reason, true)),
        }
    }
    sections
}

fn add_destination(
    diagram: &mut Diagram,
    destination: &str,
    inputs: &mut HashMap<String, usize>,
    channels: &Map<String, Value>,
    routes: &[&Value],
    drivers: &Map<String, Value>,
    shared: &Value,
) -> Result<(), String> {
    let incoming: Vec<_> = routes
        .iter()
        .copied()
        .filter(|route| route.get("destination").and_then(Value::as_str) == Some(destination))
        .collect();
    let mut tails = Vec::new();
    for route in incoming {
        let source = route
            .get("source_channel")
            .and_then(Value::as_str)
            .ok_or_else(|| "a route has no source channel".to_string())?;
        if !channels.contains_key(source) && !source.eq_ignore_ascii_case("LFE") {
            return Err(format!("input chain {source} is missing"));
        }
        let input_tail = if let Some(tail) = inputs.get(source) {
            *tail
        } else {
            let channel = channels.get(source).unwrap_or(&Value::Null);
            let start = diagram.node(&format!("Input {source}"), None);
            let tail = diagram.chain(start, owned_plugins(channel, "pre_route")?);
            inputs.insert(source.to_owned(), tail);
            tail
        };
        let mut transfer = vec![json!({
            "plugin_type": "gain",
            "parameters": {
                "gain_db": route.get("gain_db").cloned().unwrap_or(json!(0)),
                "invert": route.get("polarity_inverted").cloned().unwrap_or(json!(false)),
                "label": format!("Route {source} → {destination}")
            }
        })];
        let high = route.get("high_pass_hz");
        let low = route.get("low_pass_hz");
        if high.is_some_and(|value| !value.is_null()) || low.is_some_and(|value| !value.is_null()) {
            transfer.push(json!({
                "plugin_type": "crossover",
                "parameters": {
                    "type": route.get("crossover_type").cloned().unwrap_or(Value::Null),
                    "frequency": high.filter(|value| !value.is_null()).or(low).cloned().unwrap_or(Value::Null),
                    "output": if high.is_some_and(|value| !value.is_null()) { "high" } else { "low" }
                }
            }));
        }
        if truthy(route.get("delay_ms")) {
            transfer.push(
                json!({"plugin_type": "delay", "parameters": {"delay_ms": route["delay_ms"]}}),
            );
        }
        tails.push(diagram.chain(input_tail, transfer.iter()));
    }

    let junction = diagram.node(
        &format!(
            "{} {destination}",
            if tails.len() > 1 { "Σ" } else { "Bus" }
        ),
        None,
    );
    for tail in tails {
        diagram.edge(tail, junction);
    }

    let output_plugins = if let Some(driver) = drivers.get(destination) {
        let mut plugins = owned_plugins(shared, "post_route")?;
        for plugin in owned_plugins(driver, "post_route")? {
            let kind = plugin
                .get("plugin_type")
                .and_then(Value::as_str)
                .unwrap_or("");
            let params = plugin.get("parameters").unwrap_or(&Value::Null);
            let skip = (kind == "gain" && !truthy(params.get("room_eq_correction_gain")))
                || (kind == "delay" && !truthy(params.get("room_eq_correction_delay")));
            if !skip {
                plugins.push(plugin);
            }
        }
        plugins
    } else {
        let channel = channels
            .get(destination)
            .ok_or_else(|| "the physical output chain is missing".to_string())?;
        owned_plugins(channel, "post_route")?
    };
    let end = diagram.chain(junction, output_plugins);
    let output = diagram.node(&format!("Output {destination}"), None);
    diagram.edge(end, output);
    Ok(())
}

fn owned_plugins<'a>(channel: &'a Value, stage: &str) -> Result<Vec<&'a Value>, String> {
    let mut selected = Vec::new();
    for plugin in array(channel.get("plugins")) {
        let owner = plugin
            .pointer("/parameters/room_eq_stage")
            .and_then(Value::as_str);
        let Some(owner) =
            owner.filter(|owner| ["pre_route", "post_route", "route_owned"].contains(owner))
        else {
            return Err("a plugin has no explicit pre-route/post-route ownership".to_string());
        };
        if owner == stage {
            selected.push(plugin);
        }
    }
    Ok(selected)
}

struct Diagram {
    title: String,
    tab: String,
    nodes: Vec<String>,
    links: Vec<Value>,
    rows: Vec<(usize, String, Value)>,
    depths: Vec<usize>,
}

impl Diagram {
    fn new(title: String, tab: String) -> Self {
        Self {
            title,
            tab,
            nodes: Vec::new(),
            links: Vec::new(),
            rows: Vec::new(),
            depths: Vec::new(),
        }
    }

    fn node(&mut self, label: &str, detail: Option<&Value>) -> usize {
        let index = self.nodes.len();
        let number = index + 1;
        self.nodes.push(format!("{number}: {label}"));
        self.rows.push((
            number,
            label.to_owned(),
            detail.cloned().unwrap_or_else(|| json!({})),
        ));
        self.depths.push(0);
        index
    }

    fn edge(&mut self, source: usize, target: usize) {
        self.depths[target] = self.depths[target].max(self.depths[source].saturating_add(1));
        self.links
            .push(json!({"source": source, "target": target, "value": 1.0, "color": LINK_COLOR}));
    }

    fn chain<'a>(
        &mut self,
        mut source: usize,
        plugins: impl IntoIterator<Item = &'a Value>,
    ) -> usize {
        for plugin in plugins {
            let label = plugin_name(plugin);
            let node = self.node(&label, Some(plugin));
            self.edge(source, node);
            source = node;
        }
        source
    }

    fn sections(self) -> Vec<Value> {
        let stages = self
            .rows
            .into_iter()
            .map(|(number, label, detail)| {
                let heading = detail
                    .pointer("/parameters/label")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .unwrap_or(&label)
                    .to_owned();
                SignalFlowStage {
                    number,
                    label,
                    heading,
                    parameters: serde_json::to_string_pretty(&detail)
                        .unwrap_or_else(|_| "{}".to_string()),
                }
            })
            .collect();
        let mut stage_section = diagnostic_section(vec![DiagnosticBlock::SignalFlowStages(stages)]);
        stage_section["tab"] = json!(self.tab);
        let mut depths = self.depths;
        for link in &self.links {
            if let (Some(source), Some(target)) = (
                link.get("source").and_then(Value::as_u64),
                link.get("target").and_then(Value::as_u64),
            ) {
                let (source, target) = (source as usize, target as usize);
                if target < depths.len() && source < depths.len() {
                    depths[target] = depths[target].max(depths[source].saturating_add(1));
                }
            }
        }
        let max_depth = depths.iter().copied().max().unwrap_or(0);
        let mut layers = vec![0usize; max_depth + usize::from(!depths.is_empty())];
        for depth in depths {
            layers[depth] += 1;
        }
        let min_width = 1000usize.max((max_depth + 1) * 190);
        let min_height = 320usize.max(layers.iter().copied().max().unwrap_or(1) * 70 + 100);
        let chart = json!({"title": self.title, "nodes": self.nodes, "links": self.links, "node_boxes": true});
        vec![
            json!({"kind": "sankey", "chart": chart, "tab": self.tab, "min_width": min_width, "min_height": min_height}),
            stage_section,
        ]
    }
}

fn plugin_name(plugin: &Value) -> String {
    let kind = plugin
        .get("plugin_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    match kind {
        "eq" => "EQ",
        "gain" => "Gain",
        "delay" => "Delay",
        "crossover" => "Crossover",
        "limiter" => "Limiter",
        "convolution" => "FIR",
        _ => kind,
    }
    .to_owned()
}

fn flow_paragraph(tab: Option<&str>, text: &str) -> Value {
    let mut section = paragraph_section(&[text]);
    section["tab"] = json!(tab);
    section
}

fn unavailable_section(tab: Option<&str>, reason: &str, regenerate: bool) -> Value {
    let suffix = if regenerate {
        " Regenerate this legacy output with explicit routing ownership."
    } else {
        ""
    };
    flow_paragraph(tab, &format!("Signal flow unavailable: {reason}.{suffix}"))
}

fn array(value: Option<&Value>) -> Vec<&Value> {
    value
        .and_then(Value::as_array)
        .map(|items| items.iter().collect())
        .unwrap_or_default()
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_f64().unwrap_or(0.0) != 0.0,
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::signal_flow_sections;
    use serde_json::{Value, json};

    fn charts(sections: &[Value]) -> Vec<&Value> {
        sections
            .iter()
            .filter(|section| section["kind"] == "sankey")
            .collect()
    }

    fn plugin(kind: &str, parameters: Value) -> Value {
        json!({"plugin_type": kind, "parameters": parameters})
    }

    #[test]
    fn serial_graph_preserves_stage_order_and_duplicate_plugins() {
        let data = json!({"channels": {"L": {"plugins": [
            plugin("gain", json!({"gain_db": -2})),
            plugin("delay", json!({"delay_ms": 1})),
            plugin("eq", json!({})),
            plugin("eq", json!({})),
            plugin("limiter", json!({}))
        ]}}});
        let sections = signal_flow_sections(&data);
        let chart = charts(&sections);
        assert_eq!(chart.len(), 2); // all channels and L
        assert_eq!(
            chart[1]["chart"]["nodes"],
            json!([
                "1: Input L",
                "2: Gain",
                "3: Delay",
                "4: EQ",
                "5: EQ",
                "6: Limiter",
                "7: Output L"
            ])
        );
        assert_eq!(chart[1]["chart"]["node_boxes"], true);
        assert_eq!(chart[1]["min_width"], 1330);
    }

    #[test]
    fn routed_fanout_sums_before_post_route_processing() {
        let data = json!({
            "channels": {
                "L": {"plugins": [plugin("eq", json!({"room_eq_stage": "pre_route"}))]},
                "R": {"plugins": [plugin("delay", json!({"room_eq_stage": "pre_route"}))]},
                "Sub": {"plugins": [
                    plugin("crossover", json!({"room_eq_stage": "route_owned"})),
                    plugin("eq", json!({"room_eq_stage": "post_route"})),
                    plugin("limiter", json!({"room_eq_stage": "post_route"}))
                ]}
            },
            "metadata": {"bass_management": {"routing_graph": {"routes": [
                {"source_channel": "L", "destination": "Sub", "gain_db": -3, "low_pass_hz": 80, "crossover_type": "LR24", "delay_ms": 1},
                {"source_channel": "R", "destination": "Sub", "gain_db": -3, "low_pass_hz": 80, "crossover_type": "LR24", "delay_ms": 1}
            ]}}}
        });
        let sections = signal_flow_sections(&data);
        let chart = charts(&sections);
        assert_eq!(chart.len(), 2);
        let nodes = chart[1]["chart"]["nodes"].as_array().unwrap();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| node.as_str().unwrap().ends_with(": Crossover"))
                .count(),
            2
        );
        let sum = nodes
            .iter()
            .position(|node| node.as_str().is_some_and(|node| node.ends_with(": Σ Sub")))
            .unwrap();
        assert_eq!(
            chart[1]["chart"]["links"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|link| link["target"] == sum)
                .count(),
            2
        );
        assert_eq!(
            nodes[sum + 1..]
                .iter()
                .map(|node| node.as_str().unwrap().split_once(": ").unwrap().1)
                .collect::<Vec<_>>(),
            ["EQ", "Limiter", "Output Sub"]
        );
        assert!(chart[0]["min_height"].as_u64().unwrap() >= 320);
    }

    #[test]
    fn ambiguous_ownership_and_missing_chains_are_unavailable() {
        let data = json!({"channels": {"L": {"plugins": [plugin("eq", json!({}))]}}, "metadata": {"bass_management": {"routing_graph": {"routes": [
            {"source_channel": "L", "destination": "L"}
        ]}}}});
        let sections = signal_flow_sections(&data);
        assert!(charts(&sections).is_empty());
        assert!(
            sections
                .iter()
                .any(|section| section["html"].as_str().unwrap_or("").contains("ownership"))
        );

        let missing = json!({"channels": {"L": {"plugins": []}}, "metadata": {"bass_management": {"routing_graph": {"routes": [
            {"source_channel": "Missing", "destination": "L"}
        ]}}}});
        let sections = signal_flow_sections(&missing);
        assert!(charts(&sections).is_empty());
        assert!(sections.iter().any(|section| {
            section["html"]
                .as_str()
                .unwrap_or("")
                .contains("input chain Missing is missing")
        }));
    }

    #[test]
    fn parameter_and_unavailable_html_escape_untrusted_labels() {
        let data = json!({"channels": {"<script>alert(1)</script>": {"plugins": [
            plugin("eq", json!({"label": "<script>bad</script>"}))
        ]}}});
        let sections = signal_flow_sections(&data);
        let html = sections
            .iter()
            .filter_map(|section| section["html"].as_str())
            .collect::<String>();
        assert!(html.contains("&lt;script&gt;bad&lt;/script&gt;"));
        assert!(!html.contains("<script>"));
        assert!(charts(&sections).iter().any(|chart| {
            chart["chart"]["nodes"][0]
                .as_str()
                .unwrap()
                .contains("<script>")
        }));
    }
}
