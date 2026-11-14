//! Builds the canonical deployed-crossover and routing-cutoff report section.
//!
//! The input is a RoomEQ result document. This module presents every deployed
//! channel and driver crossover even when its measured response is external.
//!
//! # Examples
//!
//! ```
//! use autoeq_report_wasm::crossover_config_html;
//! assert_eq!(crossover_config_html(&serde_json::json!({})), "");
//! ```

// Rust guideline compliant 2026-02-21

use std::fmt::Write as _;

use serde_json::Value;

use crate::report_table::{ReportTable, ReportTableGroup};

use crate::report_helpers::{channel_sort_key, escape_html, python_string};

/// Renders deployed crossover plugins and routing-graph cutoff rows.
///
/// Channel rows use the report's established classical channel ordering;
/// driver and route rows preserve their input order. Driver visibility depends
/// on the deployed plugin chain, not whether its measured curve is embedded.
/// If the input contains no crossover plugins or routes with cutoffs, the
/// returned fragment is empty.
///
/// # Examples
///
/// ```
/// use autoeq_report_wasm::crossover_config_html;
/// let data = serde_json::json!({
///     "channels": {"L": {"plugins": [{"plugin_type":"crossover",
///         "parameters":{"type":"LR24", "output":"high", "frequency":80}}]}}
/// });
/// assert!(crossover_config_html(&data).contains("80.0 Hz"));
/// ```
///
/// # Panics
///
/// This function does not panic for JSON input values.
pub fn crossover_config_html(data: &Value) -> String {
    crossover_tables(data).map_or_else(String::new, |group| render_crossover_tables(&group))
}

pub(crate) fn crossover_tables(data: &Value) -> Option<ReportTableGroup> {
    let channels = data.get("channels").and_then(Value::as_object)?;

    let mut channel_names: Vec<&str> = channels.keys().map(String::as_str).collect();
    channel_names.sort_by(|left, right| channel_sort_key(left).cmp(&channel_sort_key(right)));

    let mut plugin_rows = Vec::new();
    for channel_name in channel_names {
        let Some(channel) = channels.get(channel_name).and_then(Value::as_object) else {
            continue;
        };
        let mut owners = vec![(format!("Channel {channel_name}"), channel.get("plugins"))];
        let drivers = object_drivers(channel.get("drivers"));
        let names = driver_display_names(data, channel_name, &drivers);
        for (display_index, (source_index, driver)) in drivers.iter().enumerate() {
            let label = names
                .get(display_index)
                .cloned()
                .or_else(|| driver.get("name").and_then(truthy).map(python_string))
                .unwrap_or_else(|| source_index.to_string());
            owners.push((format!("Driver {label}"), driver.get("plugins")));
        }

        for (owner, plugins) in owners {
            for plugin in array_or_empty(plugins) {
                let Some(plugin) = plugin.as_object() else {
                    continue;
                };
                let kind = plugin
                    .get("plugin_type")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if kind != "crossover" && kind != "band_split" {
                    continue;
                }
                let params = plugin.get("parameters").and_then(Value::as_object);
                let frequency = params.and_then(|params| params.get("frequency"));
                let frequency_text = format_frequency(frequency);
                let type_text = params
                    .and_then(|params| params.get("type"))
                    .map(python_string)
                    .unwrap_or_else(|| {
                        if kind == "band_split" {
                            "band_split"
                        } else {
                            "-"
                        }
                        .to_string()
                    });
                let output_text = if kind == "band_split" {
                    "-".to_string()
                } else {
                    params
                        .and_then(|params| params.get("output"))
                        .map(python_string)
                        .unwrap_or_else(|| "-".to_string())
                };
                let stage_text = params
                    .and_then(|params| params.get("room_eq_stage"))
                    .filter(|value| !value.is_null())
                    .map(python_string)
                    .unwrap_or_else(|| "-".to_string());
                plugin_rows.push(vec![
                    owner.clone(),
                    type_text,
                    output_text,
                    frequency_text,
                    stage_text,
                ]);
            }
        }
    }

    let mut route_rows = Vec::new();
    let bass_management = data
        .get("metadata")
        .and_then(|metadata| metadata.get("bass_management"));
    let routes = bass_management
        .and_then(|bass| bass.get("routing_graph"))
        .and_then(|graph| graph.get("routes"));
    for route in array_or_empty(routes) {
        let Some(route) = route.as_object() else {
            continue;
        };
        let low = route.get("low_pass_hz");
        let high = route.get("high_pass_hz");
        if low.is_none_or(Value::is_null) && high.is_none_or(Value::is_null) {
            continue;
        }
        let source = route
            .get("source_channel")
            .map(python_string)
            .unwrap_or_else(|| "-".to_string());
        let destination = route
            .get("destination")
            .map(python_string)
            .unwrap_or_else(|| "-".to_string());
        let kind = route
            .get("route_kind")
            .map(python_string)
            .unwrap_or_else(|| "-".to_string());
        let crossover_type = route
            .get("crossover_type")
            .filter(|value| !value.is_null())
            .map(python_string)
            .unwrap_or_else(|| "-".to_string());
        route_rows.push(vec![
            format!("{source} → {destination}"),
            kind,
            crossover_type,
            format_frequency(low),
            format_frequency(high),
        ]);
    }

    let mut tables = Vec::new();
    if !plugin_rows.is_empty() {
        tables.push(ReportTable {
            title: "Deployed Crossover DSP".into(),
            note: String::new(),
            columns: ["Owner", "Type", "Output", "Frequency", "Stage"]
                .map(String::from)
                .into(),
            rows: plugin_rows,
        });
    }
    if !route_rows.is_empty() {
        tables.push(ReportTable {
            title: "Routing-Graph Crossovers".into(),
            note: String::new(),
            columns: ["Route", "Kind", "Type", "Low-pass", "High-pass"]
                .map(String::from)
                .into(),
            rows: route_rows,
        });
    }
    (!tables.is_empty()).then_some(ReportTableGroup {
        title: "Crossover Configuration".into(),
        tables,
    })
}

pub(crate) fn render_crossover_tables(group: &ReportTableGroup) -> String {
    let mut html = String::new();
    let _ = write!(
        html,
        "        <div class=\"plot-container\">\n            <h2>{}</h2>\n",
        escape_html(&group.title)
    );
    for table in &group.tables {
        let _ = write!(
            html,
            "            <h3>{}</h3>\n            <table class=\"bm-table\">\n                <thead><tr>",
            escape_html(&table.title)
        );
        for column in &table.columns {
            let _ = write!(html, "<th>{}</th>", escape_html(column));
        }
        html.push_str("</tr></thead>\n                <tbody>\n");
        for row in &table.rows {
            html.push_str("                    <tr>");
            for cell in row {
                let _ = write!(html, "<td>{}</td>", escape_html(cell));
            }
            html.push_str("</tr>\n");
        }
        html.push_str("                </tbody>\n            </table>\n");
    }
    html.push_str("        </div>\n");
    html
}

fn array_or_empty(value: Option<&Value>) -> &[Value] {
    value.and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

fn truthy(value: &Value) -> Option<&Value> {
    match value {
        Value::Null => None,
        Value::Bool(false) => None,
        Value::Number(number) if number.as_f64() == Some(0.0) => None,
        Value::String(text) if text.is_empty() => None,
        Value::Array(items) if items.is_empty() => None,
        Value::Object(items) if items.is_empty() => None,
        _ => Some(value),
    }
}

fn format_frequency(value: Option<&Value>) -> String {
    value
        .and_then(|value| {
            value.as_f64().or_else(|| {
                value
                    .as_bool()
                    .map(|boolean| if boolean { 1.0 } else { 0.0 })
            })
        })
        .map(|frequency| format!("{frequency:.1} Hz"))
        .unwrap_or_else(|| "-".to_string())
}

fn object_drivers(drivers: Option<&Value>) -> Vec<(usize, &Value)> {
    array_or_empty(drivers)
        .iter()
        .enumerate()
        .filter(|(_, driver)| driver.is_object())
        .collect()
}

fn driver_display_names(
    data: &Value,
    channel_name: &str,
    drivers: &[(usize, &Value)],
) -> Vec<String> {
    let driver_slots = data
        .get("channels")
        .and_then(|channels| channels.get(channel_name))
        .and_then(|channel| channel.get("drivers"))
        .and_then(Value::as_array)
        .map(Vec::len);
    let configured_subwoofers = data
        .get("metadata")
        .and_then(|metadata| metadata.get("effective_config"))
        .and_then(|config| config.get("system"))
        .and_then(|system| system.get("speakers"))
        .and_then(|speakers| speakers.get(channel_name))
        .and_then(Value::as_str)
        .and_then(|speaker_key| {
            data.get("metadata")
                .and_then(|metadata| metadata.get("effective_config"))
                .and_then(|config| config.get("speakers"))
                .and_then(|speakers| speakers.get(speaker_key))
        })
        .and_then(|group| group.get("subwoofers"))
        .and_then(Value::as_array);
    let configured_names_match_slots = configured_subwoofers
        .zip(driver_slots)
        .is_some_and(|(subs, slot_count)| subs.len() == slot_count);
    drivers
        .iter()
        .map(|(source_index, driver)| {
            let configured_name = configured_subwoofers
                .filter(|_| configured_names_match_slots)
                .and_then(|subs| subs.get(*source_index))
                .and_then(|sub| sub.get("name"))
                .and_then(truthy);
            configured_name
                .or_else(|| driver.get("name").and_then(truthy))
                .or_else(|| driver.get("index").filter(|value| !value.is_null()))
                .map(python_string)
                .unwrap_or_else(|| source_index.to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_SUB_GOLDEN: &str = concat!(
        "        <div class=\"plot-container\">\n",
        "            <h2>Crossover Configuration</h2>\n",
        "            <h3>Deployed Crossover DSP</h3>\n",
        "            <table class=\"bm-table\">\n",
        "                <thead><tr><th>Owner</th><th>Type</th><th>Output</th><th>Frequency</th><th>Stage</th></tr></thead>\n",
        "                <tbody>\n",
        "                    <tr><td>Driver Left Sub</td><td>LR24</td><td>low</td><td>80.0 Hz</td><td>post_route</td></tr>\n",
        "                    <tr><td>Driver Right Sub</td><td>LR24</td><td>low</td><td>90.0 Hz</td><td>post_route</td></tr>\n",
        "                </tbody>\n            </table>\n",
        "            <h3>Routing-Graph Crossovers</h3>\n",
        "            <table class=\"bm-table\">\n",
        "                <thead><tr><th>Route</th><th>Kind</th><th>Type</th><th>Low-pass</th><th>High-pass</th></tr></thead>\n",
        "                <tbody>\n",
        "                    <tr><td>LFE → Two subs_1</td><td>lfe_lowpass_to_sub</td><td>LR24</td><td>120.0 Hz</td><td>-</td></tr>\n",
        "                    <tr><td>LFE → Two subs_2</td><td>lfe_lowpass_to_sub</td><td>LR24</td><td>120.0 Hz</td><td>-</td></tr>\n",
        "                </tbody>\n            </table>\n",
        "        </div>\n"
    );

    #[test]
    fn matches_captured_python_golden_for_two_sub_plugin_and_route_tables() {
        let data = serde_json::json!({
            "channels": {
                "LFE": {"drivers": [
                    {"name": "ignored", "initial_curve": {"freq": [20], "spl": [0]},
                     "plugins": [{"plugin_type":"crossover", "parameters": {
                         "type":"LR24", "output":"low", "frequency":80,
                         "room_eq_stage":"post_route"}}]},
                    {"initial_curve": {"freq": [20], "spl": [0]}, "name":"Right Sub",
                     "plugins": [{"plugin_type":"crossover", "parameters": {
                         "type":"LR24", "output":"low", "frequency":90,
                         "room_eq_stage":"post_route"}}]}
                ]},
                "R": {}, "L": {}
            },
            "metadata": {
                "effective_config": {"system":{"speakers":{"LFE":"subs"}},
                    "speakers":{"subs":{"subwoofers":[{"name":"Left Sub"},{"name":"Right Sub"}]}}},
                "bass_management": {"routing_graph":{"routes":[
                    {"source_channel":"LFE","destination":"Two subs_1","route_kind":"lfe_lowpass_to_sub","crossover_type":"LR24","low_pass_hz":120},
                    {"source_channel":"LFE","destination":"Two subs_2","route_kind":"lfe_lowpass_to_sub","crossover_type":"LR24","low_pass_hz":120}
                ]}}
            }
        });
        assert_eq!(crossover_config_html(&data), TWO_SUB_GOLDEN);
    }

    #[test]
    fn escapes_labels_and_preserves_missing_and_mixed_cutoff_rows() {
        let data = serde_json::json!({
            "channels": {
                "L <script>": {"plugins": [{"plugin_type":"band_split",
                    "parameters":{"type":"<x>", "frequency": 55.25, "room_eq_stage": null}}]},
                "R": {"plugins": [{"plugin_type":"crossover",
                    "parameters":{"type": true, "frequency": true}}]}
            },
            "metadata":{"bass_management":{"routing_graph":{"routes":[
                {"source_channel":"A'\"&", "destination":"<out>", "route_kind":"route", "high_pass_hz":12.25, "crossover_type":"LR"},
                {"source_channel":"drop", "destination":"drop", "low_pass_hz":null, "high_pass_hz":null},
                null,
                "non-object"
            ]}}}
        });
        let html = crossover_config_html(&data);
        assert!(html.contains("Channel L &lt;script&gt;"));
        assert!(html.contains("&lt;x&gt;</td><td>-</td><td>55.2 Hz</td><td>-</td>"));
        assert!(html.contains("A&#x27;&quot;&amp; → &lt;out&gt;"));
        assert!(html.contains("<td>route</td><td>LR</td><td>-</td><td>12.2 Hz</td>"));
        assert!(html.contains("<td>Channel R</td><td>True</td><td>-</td><td>1.0 Hz</td>"));
        assert!(!html.contains("<script>"));
    }

    #[test]
    fn raw_slim_driver_records_render_without_embedded_measurement_curves() {
        let data = serde_json::json!({
            "channels": {"LFE": {"drivers": [
                null,
                {"name":"Raw Front", "initial_curve": null, "plugins": [{"plugin_type":"crossover",
                    "parameters":{"type":"LR24", "output":"low", "frequency":80}}]},
                {"name":"", "index":9, "initial_curve": null, "plugins": [{"plugin_type":"crossover",
                    "parameters":{"type":"BW12", "output":"high", "frequency":90}}]}
            ]}},
            "metadata":{"effective_config":{
                "system":{"speakers":{"LFE":"sub_group"}},
                "speakers":{"sub_group":{"subwoofers":[
                    {"name":"Front <Sub>"}, {"name":"Rear Sub"}
                ]}}
            }}
        });
        let html = crossover_config_html(&data);
        assert!(html.contains("Driver Raw Front</td><td>LR24</td>"));
        assert!(html.contains("Driver 9</td><td>BW12</td>"));
        assert!(!html.contains("Driver 0"));

        let partial_config = serde_json::json!({
            "channels":{"LFE":{"drivers":[
                {"name":"Raw A", "plugins":[{"plugin_type":"crossover", "parameters":{"frequency":80}}]},
                {"name":"Raw B", "plugins":[{"plugin_type":"crossover", "parameters":{"frequency":90}}]}
            ]}},
            "metadata":{"effective_config":{
                "system":{"speakers":{"LFE":"sub_group"}},
                "speakers":{"sub_group":{"subwoofers":[{"name":"Configured A"},{}]}}
            }}
        });
        let partial_html = crossover_config_html(&partial_config);
        assert!(partial_html.contains("Driver Configured A"));
        assert!(partial_html.contains("Driver Raw B"));

        let aligned_sparse = serde_json::json!({
            "channels":{"LFE":{"drivers":[
                {"plugins":[{"plugin_type":"crossover", "parameters":{"frequency":80}}]},
                null,
                {"plugins":[{"plugin_type":"crossover", "parameters":{"frequency":90}}]}
            ]}},
            "metadata":{"effective_config":{
                "system":{"speakers":{"LFE":"sub_group"}},
                "speakers":{"sub_group":{"subwoofers":[
                    {"name":"Configured A"}, {}, {"name":"Configured C"}
                ]}}
            }}
        });
        let aligned_html = crossover_config_html(&aligned_sparse);
        assert!(aligned_html.contains("Driver Configured A"));
        assert!(aligned_html.contains("Driver Configured C"));
    }

    #[test]
    fn returns_empty_when_no_crossover_evidence_exists() {
        assert_eq!(crossover_config_html(&serde_json::json!({})), "");
        assert_eq!(
            crossover_config_html(&serde_json::json!({"channels":{"L":{}}})),
            ""
        );
    }
}
