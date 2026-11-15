//! Canonical HTML summaries and tables for bass-management metadata.

// Rust guideline compliant 2026-02-21

use std::fmt::Write as _;

use serde_json::Value;

use crate::report_helpers::{escape_html, format_db, format_hz, format_ms, python_string};
use crate::report_table::ReportTable;

/// The canonical HTML fragments for bass-management metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct BassManagementSections {
    pub summary_html: String,
    pub groups_table_html: String,
    pub sub_outputs_table_html: String,
}

/// Renders the bass-management summary and its two metadata tables.
pub fn bass_management_sections(report: &Value) -> BassManagementSections {
    BassManagementSections {
        summary_html: bass_management_summary_html(report),
        groups_table_html: bass_management_groups_table_html(report),
        sub_outputs_table_html: bass_management_sub_outputs_table_html(report),
    }
}

pub fn bass_management_summary_html(report: &Value) -> String {
    bass_management_summary(report).map_or_else(String::new, |table| render_bass_summary(&table))
}

pub(crate) fn bass_management_summary(report: &Value) -> Option<ReportTable> {
    if !is_truthy(report) {
        return None;
    }
    let routing = report.get("routing_graph").filter(|value| is_truthy(value));
    let routes = routing
        .and_then(|value| value.get("routes"))
        .filter(|value| is_truthy(value))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let route_count = routes.len();
    let mut physical_outputs = routes
        .iter()
        .filter(|route| {
            let kind = route.get("route_kind").and_then(Value::as_str);
            matches!(
                kind,
                Some("redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub")
            ) && route.get("destination").is_some_and(is_truthy)
        })
        .map(|route| python_string(&route["destination"]))
        .collect::<Vec<_>>();
    physical_outputs.sort();
    physical_outputs.dedup();

    let mut advisories = Vec::new();
    if let Some(advisory) = report.get("advisory").filter(|value| is_truthy(value)) {
        advisories.push(python_string(advisory));
    }
    if let Some(items) = routing
        .and_then(|value| value.get("advisories"))
        .filter(|value| is_truthy(value))
        .and_then(Value::as_array)
    {
        advisories.extend(items.iter().map(python_string));
    }
    advisories.sort();
    advisories.dedup();
    advisories.retain(|item| !item.is_empty() && item != "ok");
    let enabled = report.get("enabled").is_some_and(is_truthy);
    let crossover_type = report
        .get("crossover_type")
        .map(python_string)
        .unwrap_or_else(|| "-".to_string());
    let physical = if physical_outputs.is_empty() {
        report
            .get("physical_sub_output")
            .map(python_string)
            .unwrap_or_else(|| "-".to_string())
    } else {
        physical_outputs.join(", ")
    };
    let items = [
        ("Enabled", if enabled { "yes" } else { "no" }.to_string()),
        (
            "Crossover",
            format!(
                "{} @ {}",
                crossover_type,
                format_hz(report.get("crossover_frequency_hz"))
            ),
        ),
        ("LFE gain", format_db(report.get("lfe_playback_gain_db"))),
        (
            "Shared sub gain",
            format_db(report.get("applied_sub_gain_db")),
        ),
        ("Physical bass outputs", physical),
        ("Route count", route_count.to_string()),
        (
            "Graph mode",
            if route_count > 0 {
                "route branches".to_string()
            } else {
                "linear / none".to_string()
            },
        ),
    ];
    Some(ReportTable {
        title: "Bass Management".into(),
        note: advisories.join("; "),
        columns: ["Setting", "Value"].map(String::from).into(),
        rows: items
            .into_iter()
            .map(|(label, value)| vec![label.into(), value])
            .collect(),
    })
}

pub(crate) fn render_bass_summary(table: &ReportTable) -> String {
    let mut html = String::new();
    let _ = write!(
        html,
        "<div class=\"metadata bass-management-section\">\n<h2>{}</h2>\n<div class=\"metadata-grid\">",
        escape_html(&table.title)
    );
    for row in &table.rows {
        if let [label, value] = row.as_slice() {
            let _ = writeln!(
                html,
                "<div class=\"metadata-item\"><span class=\"metadata-label\">{}:</span> <span class=\"metadata-value\">{}</span></div>",
                escape_html(label),
                escape_html(value)
            );
        }
    }
    html.push_str("</div>\n");
    if !table.note.is_empty() {
        let _ = write!(
            html,
            "<br><span style=\"color:#b36b00\">{}</span>",
            escape_html(&table.note)
        );
    }
    html.push_str("\n</div>\n");
    html
}

pub fn bass_management_groups_table_html(report: &Value) -> String {
    bass_management_groups(report).map_or_else(String::new, |table| render_bass_table(&table))
}

pub(crate) fn bass_management_groups(report: &Value) -> Option<ReportTable> {
    let groups = selected_array(report, "groups", "group_results");
    if groups.is_empty() {
        return None;
    }
    let mut rows = Vec::new();
    for group in groups {
        let advisories = group
            .get("advisories")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter(|item| item != &&Value::String("ok".to_string()))
                    .map(python_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        let roles = group
            .get("roles")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(python_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        rows.push(vec![
            field_string(group, "group_id", "-"),
            roles,
            field_string(group, "crossover_type", "-"),
            format_hz(group.get("selected_crossover_hz")),
            format_ms(group.get("main_delay_ms")),
            format_ms(group.get("bass_route_delay_ms")),
            if group.get("polarity_inverted").is_some_and(is_truthy) {
                "yes".into()
            } else {
                "no".into()
            },
            format_db(group.get("trim_db")),
            if advisories.is_empty() {
                "-".into()
            } else {
                advisories
            },
        ]);
    }
    Some(ReportTable {
        title: "Per-Speaker-Group Crossovers".into(),
        note: String::new(),
        columns: [
            "Group",
            "Roles",
            "Type",
            "Selected XO",
            "Main delay",
            "Bass delay",
            "Invert bass",
            "Trim",
            "Advisories",
        ]
        .map(String::from)
        .into(),
        rows,
    })
}

pub fn bass_management_sub_outputs_table_html(report: &Value) -> String {
    bass_management_outputs(report).map_or_else(String::new, |table| render_bass_table(&table))
}

pub(crate) fn bass_management_outputs(report: &Value) -> Option<ReportTable> {
    let outputs = selected_array(report, "sub_outputs", "sub_output_results");
    if outputs.is_empty() {
        return None;
    }
    let mut rows = Vec::new();
    for output in outputs {
        rows.push(vec![
            field_string(output, "output_role", "-"),
            field_string(output, "strategy_source", "-"),
            format_db(output.get("gain_db")),
            format_ms(output.get("delay_ms")),
            if output.get("polarity_inverted").is_some_and(is_truthy) {
                "yes".into()
            } else {
                "no".into()
            },
            format_db(output.get("headroom_contribution_db")),
        ]);
    }
    Some(ReportTable {
        title: "Physical Bass Outputs".into(),
        note: String::new(),
        columns: [
            "Output",
            "Strategy",
            "Gain",
            "Delay",
            "Invert",
            "Headroom contribution",
        ]
        .map(String::from)
        .into(),
        rows,
    })
}

pub(crate) fn render_bass_table(table: &ReportTable) -> String {
    let mut html = String::new();
    let _ = write!(
        html,
        "<div class=\"filters-section bass-management-section\">\n<h3>{}</h3>\n<table class=\"bm-table\">\n<thead><tr>",
        escape_html(&table.title)
    );
    for column in &table.columns {
        let _ = write!(html, "<th>{}</th>", escape_html(column));
    }
    html.push_str("</tr></thead>\n<tbody>");
    for row in &table.rows {
        html.push_str("<tr>");
        for cell in row {
            let _ = write!(html, "<td>{}</td>", escape_html(cell));
        }
        html.push_str("</tr>\n");
    }
    html.push_str("</tbody>\n</table>\n</div>\n");
    html
}

fn selected_array<'a>(report: &'a Value, primary: &str, fallback: &str) -> Vec<&'a Value> {
    let values = report
        .get(primary)
        .filter(|value| is_truthy(value))
        .and_then(Value::as_array)
        .or_else(|| {
            report
                .get("optimization")
                .and_then(|optimization| optimization.get(fallback))
                .filter(|value| is_truthy(value))
                .and_then(Value::as_array)
        });
    values
        .map(|items| items.iter().filter(|item| item.is_object()).collect())
        .unwrap_or_default()
}

fn field_string(value: &Value, field: &str, default: &str) -> String {
    value
        .get(field)
        .map(python_string)
        .unwrap_or_else(|| default.to_string())
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(items) => !items.is_empty(),
        Value::Bool(true) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bass_management_groups_table_html, bass_management_sections,
        bass_management_sub_outputs_table_html, bass_management_summary_html,
    };

    const SUMMARY_GOLDEN: &str = concat!(
        "<div class=\"metadata bass-management-section\">\n<h2>Bass Management</h2>\n<div class=\"metadata-grid\">",
        "<div class=\"metadata-item\"><span class=\"metadata-label\">Enabled:</span> <span class=\"metadata-value\">yes</span></div>\n",
        "<div class=\"metadata-item\"><span class=\"metadata-label\">Crossover:</span> <span class=\"metadata-value\">Linkwitz-Riley @ 80.0 Hz</span></div>\n",
        "<div class=\"metadata-item\"><span class=\"metadata-label\">LFE gain:</span> <span class=\"metadata-value\">-2.50 dB</span></div>\n",
        "<div class=\"metadata-item\"><span class=\"metadata-label\">Shared sub gain:</span> <span class=\"metadata-value\">+1.25 dB</span></div>\n",
        "<div class=\"metadata-item\"><span class=\"metadata-label\">Physical bass outputs:</span> <span class=\"metadata-value\">Sub A, Sub B</span></div>\n",
        "<div class=\"metadata-item\"><span class=\"metadata-label\">Route count:</span> <span class=\"metadata-value\">3</span></div>\n",
        "<div class=\"metadata-item\"><span class=\"metadata-label\">Graph mode:</span> <span class=\"metadata-value\">route branches</span></div>\n",
        "</div>\n<br><span style=\"color:#b36b00\">check &lt;routing&gt;; delay mismatch</span>\n</div>\n"
    );
    const GROUPS_GOLDEN: &str = concat!(
        "<div class=\"filters-section bass-management-section\">\n<h3>Per-Speaker-Group Crossovers</h3>\n<table class=\"bm-table\">\n",
        "<thead><tr><th>Group</th><th>Roles</th><th>Type</th><th>Selected XO</th><th>Main delay</th><th>Bass delay</th><th>Invert bass</th><th>Trim</th><th>Advisories</th></tr></thead>\n",
        "<tbody><tr><td>main&lt;script&gt;</td><td>L, R</td><td>LR4</td><td>80.0 Hz</td><td>-1.250 ms</td><td>0.500 ms</td><td>yes</td><td>-3.12 dB</td><td>review &lt;phase&gt;</td></tr>\n</tbody>\n</table>\n</div>\n"
    );
    const OUTPUTS_GOLDEN: &str = concat!(
        "<div class=\"filters-section bass-management-section\">\n<h3>Physical Bass Outputs</h3>\n<table class=\"bm-table\">\n",
        "<thead><tr><th>Output</th><th>Strategy</th><th>Gain</th><th>Delay</th><th>Invert</th><th>Headroom contribution</th></tr></thead>\n",
        "<tbody><tr><td>Sub A</td><td>shared &lt;bass&gt;</td><td>-1.50 dB</td><td>2.250 ms</td><td>no</td><td>-6.75 dB</td></tr>\n</tbody>\n</table>\n</div>\n"
    );

    fn report() -> serde_json::Value {
        serde_json::json!({
            "enabled": true, "crossover_type": "Linkwitz-Riley", "crossover_frequency_hz": 80.0,
            "lfe_playback_gain_db": -2.5, "applied_sub_gain_db": 1.25,
            "physical_sub_output": "Sub A", "advisory": "check <routing>",
            "routing_graph": {"routes": [
                {"route_kind":"redirected_bass_lowpass_to_sub", "destination":"Sub B"},
                {"route_kind":"lfe_lowpass_to_sub", "destination":"Sub A"},
                {"route_kind":"other", "destination":"Ignored"}], "advisories":["ok", "delay mismatch"]},
            "groups": [{"group_id":"main<script>", "roles":["L","R"], "crossover_type":"LR4", "selected_crossover_hz":80.0, "main_delay_ms":-1.25, "bass_route_delay_ms":0.5, "polarity_inverted":true, "trim_db":-3.125, "advisories":["ok","review <phase>"]}],
            "sub_outputs": [{"output_role":"Sub A", "strategy_source":"shared <bass>", "gain_db":-1.5, "delay_ms":2.25, "polarity_inverted":false, "headroom_contribution_db":-6.75}]
        })
    }

    #[test]
    fn exactly_matches_captured_python_bass_management_goldens() {
        let report = report();
        let sections = bass_management_sections(&report);
        assert_eq!(sections.summary_html, SUMMARY_GOLDEN);
        assert_eq!(sections.groups_table_html, GROUPS_GOLDEN);
        assert_eq!(sections.sub_outputs_table_html, OUTPUTS_GOLDEN);
        assert_eq!(bass_management_summary_html(&report), SUMMARY_GOLDEN);
        assert_eq!(bass_management_groups_table_html(&report), GROUPS_GOLDEN);
        assert_eq!(
            bass_management_sub_outputs_table_html(&report),
            OUTPUTS_GOLDEN
        );
    }

    #[test]
    fn empty_sections_and_optimization_fallback_match_python_contract() {
        assert_eq!(bass_management_summary_html(&serde_json::json!({})), "");
        assert_eq!(
            bass_management_groups_table_html(&serde_json::json!({})),
            ""
        );
        assert_eq!(
            bass_management_sub_outputs_table_html(&serde_json::json!({})),
            ""
        );
        let fallback = serde_json::json!({"optimization": {
            "group_results": [{"group_id":"fallback"}],
            "sub_output_results": [{"output_role":"fallback"}]
        }});
        assert!(bass_management_groups_table_html(&fallback).contains("<td>fallback</td>"));
        assert!(bass_management_sub_outputs_table_html(&fallback).contains("<td>fallback</td>"));
    }
}
