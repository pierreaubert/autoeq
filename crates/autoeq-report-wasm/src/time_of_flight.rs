//! Builds the canonical Time of Flight report section.
//!
//! The input is `metadata` from a RoomEQ output document. This module only
//! presents the already-computed `timing_diagnostics.channels` records; it
//! does not derive arrival times or DSP delays. Rows retain their source
//! order, and unavailable or non-finite values display as `n/a`.
//!
//! # Examples
//!
//! ```
//! use autoeq_report_wasm::time_of_flight_html;
//! let metadata = serde_json::json!({"timing_diagnostics": {"channels": []}});
//! assert_eq!(time_of_flight_html(&metadata), "");
//! ```

// Rust guideline compliant 2026-02-21

use std::fmt::Write as _;

use serde_json::Value;

use crate::report_helpers::escape_html;
use crate::report_table::{ReportTable, write_table_html};

/// Builds the Time of Flight HTML fragment from RoomEQ metadata.
///
/// Each input channel becomes one row in the original order. Missing or
/// invalid timing fields display as `n/a`; an absent or empty channel list
/// produces an empty fragment.
///
/// # Examples
///
/// ```
/// use autoeq_report_wasm::time_of_flight_html;
/// let metadata = serde_json::json!({"timing_diagnostics": {"channels": [
///     {"name": "Left", "measured_arrival_ms": 1.25,
///      "applied_delay_ms": -0.5, "final_arrival_ms": 0.75,
///      "final_offset_from_reference_ms": -2.0}
/// ]}});
/// let html = time_of_flight_html(&metadata);
/// assert!(html.contains("<td>Left</td><td>1.250</td><td>-0.500</td>"));
/// ```
///
/// # Panics
///
/// This function does not panic for JSON input values.
pub fn time_of_flight_html(metadata: &Value) -> String {
    let Some(table) = time_of_flight_table(metadata) else {
        return String::new();
    };
    render_timing_html(&table)
}

pub(crate) fn render_timing_html(table: &ReportTable) -> String {
    let mut html = String::new();
    let _ = write!(
        html,
        "<div class=\"filters-section\">\n<h3>{}</h3>\n<p class=\"epa-footer\">{}</p>",
        escape_html(&table.title),
        escape_html(&table.note),
    );
    write_table_html(&mut html, table, "epa-table", "\n");
    html.push_str("</div>\n");
    html
}

/// Builds the ordered timing table for the report.
pub fn time_of_flight_table(metadata: &Value) -> Option<ReportTable> {
    let rows = time_of_flight_rows(metadata);
    let rows = rows.as_array().filter(|rows| !rows.is_empty())?;
    Some(ReportTable {
        title: "Time of flight".into(),
        note: "After-DSP arrival adds the deployed chain delay to the measured input arrival; it is not a post-playback capture.".into(),
        columns: [
            "Monitor",
            "Measured arrival before DSP (ms)",
            "Applied delay (ms)",
            "Calculated arrival after DSP (ms)",
            "Offset vs reference (ms)",
        ]
        .map(String::from)
        .into(),
        rows: rows
            .iter()
            .map(|row| {
                vec![
                    row["name"].as_str().unwrap_or("?").to_owned(),
                    format_number(row.get("before_ms")),
                    format_number(row.get("applied_ms")),
                    format_number(row.get("after_ms")),
                    format_number(row.get("offset_ms")),
                ]
            })
            .collect(),
    })
}

/// Extracts canonical Time of Flight rows without formatting them.
///
/// The returned rows match the keys and null behavior of AutoEQ's former
/// `tof_table`: `name`, `before_ms`, `applied_ms`, `after_ms`, and
/// `offset_ms`. Rows with non-object channel entries are skipped, and input
/// order is preserved. Numeric values are copied without calculating new
/// acoustic quantities.
///
/// # Examples
///
/// ```
/// use autoeq_report_wasm::time_of_flight_rows;
/// let metadata = serde_json::json!({"timing_diagnostics": {"channels": [
///     {"name": "Left", "measured_arrival_ms": 3.2}
/// ]}});
/// let rows = time_of_flight_rows(&metadata);
/// assert_eq!(rows[0]["before_ms"], 3.2);
/// assert_eq!(rows[0]["applied_ms"], serde_json::Value::Null);
/// ```
///
/// # Panics
///
/// This function does not panic for JSON input values.
pub fn time_of_flight_rows(metadata: &Value) -> Value {
    let Some(channels) = metadata
        .get("timing_diagnostics")
        .and_then(Value::as_object)
        .and_then(|timing| timing.get("channels"))
        .and_then(Value::as_array)
    else {
        return Value::Array(Vec::new());
    };
    let rows = channels
        .iter()
        .filter_map(Value::as_object)
        .map(|channel| {
            serde_json::json!({
                "name": channel.get("name").map(python_string).unwrap_or_else(|| "?".to_string()),
                "before_ms": channel.get("measured_arrival_ms").cloned().unwrap_or(Value::Null),
                "applied_ms": channel.get("applied_delay_ms").cloned().unwrap_or(Value::Null),
                "after_ms": channel.get("final_arrival_ms").cloned().unwrap_or(Value::Null),
                "offset_ms": channel.get("final_offset_from_reference_ms").cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    Value::Array(rows)
}

fn format_number(value: Option<&Value>) -> String {
    value
        .and_then(Value::as_number)
        .and_then(|number| number.as_f64())
        .filter(|number| number.is_finite())
        .map(|number| format!("{number:.3}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("{}: {}", python_repr_string(key), python_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::String(text) => python_repr_string(text),
        _ => python_string(value),
    }
}

fn python_repr_string(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut rendered = String::with_capacity(text.len() + 2);
    rendered.push(quote);
    for character in text.chars() {
        match character {
            '\\' => rendered.push_str("\\\\"),
            '\n' => rendered.push_str("\\n"),
            '\r' => rendered.push_str("\\r"),
            '\t' => rendered.push_str("\\t"),
            character if character == quote => {
                rendered.push('\\');
                rendered.push(character);
            }
            character if character.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(rendered, "\\x{:02x}", character as u32);
            }
            character => rendered.push(character),
        }
    }
    rendered.push(quote);
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOF_HEADING_AND_DESCRIPTION: &str = concat!(
        "<div class=\"filters-section\">\n<h3>Time of flight</h3>\n",
        "<p class=\"epa-footer\">After-DSP arrival adds the deployed chain delay ",
        "to the measured input arrival; it is not a post-playback capture.</p>",
        "<table class=\"epa-table\"><thead><tr><th>Monitor</th>",
        "<th>Measured arrival before DSP (ms)</th>",
        "<th>Applied delay (ms)</th>",
        "<th>Calculated arrival after DSP (ms)</th>",
        "<th>Offset vs reference (ms)</th>",
        "</tr></thead><tbody>\n"
    );

    #[test]
    fn formats_rows_exactly_and_preserves_source_order() {
        let metadata = serde_json::json!({
            "timing_diagnostics": {"channels": [
                {"name": "Surround <R>", "measured_arrival_ms": 8.12549,
                 "applied_delay_ms": -4.5, "final_arrival_ms": 3.62549,
                 "final_offset_from_reference_ms": -1.23456},
                {"name": "Front L", "measured_arrival_ms": 2,
                 "applied_delay_ms": 1.25, "final_arrival_ms": 3.25,
                 "final_offset_from_reference_ms": 0}
            ]}
        });
        let html = time_of_flight_html(&metadata);
        let expected = concat!(
            "<div class=\"filters-section\">\n<h3>Time of flight</h3>\n",
            "<p class=\"epa-footer\">After-DSP arrival adds the deployed chain delay ",
            "to the measured input arrival; it is not a post-playback capture.</p>",
            "<table class=\"epa-table\"><thead><tr><th>Monitor</th>",
            "<th>Measured arrival before DSP (ms)</th>",
            "<th>Applied delay (ms)</th>",
            "<th>Calculated arrival after DSP (ms)</th>",
            "<th>Offset vs reference (ms)</th>",
            "</tr></thead><tbody>\n",
            "<tr><td>Surround &lt;R&gt;</td><td>8.125</td><td>-4.500</td>",
            "<td>3.625</td><td>-1.235</td></tr>\n",
            "<tr><td>Front L</td><td>2.000</td><td>1.250</td>",
            "<td>3.250</td><td>0.000</td></tr>\n",
            "</tbody></table></div>\n"
        );
        assert_eq!(html, expected);
    }

    #[test]
    fn missing_values_and_non_object_rows_are_not_invented() {
        let metadata = serde_json::json!({
            "timing_diagnostics": {"channels": [
                {"measured_arrival_ms": null, "applied_delay_ms": true,
                 "final_arrival_ms": "5", "final_offset_from_reference_ms": null},
                null,
                {"name": "Unknown", "applied_delay_ms": -0.0004}
            ]}
        });
        let html = time_of_flight_html(&metadata);
        assert!(
            html.contains("<tr><td>?</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr>\n")
        );
        assert!(html.contains(
            "<tr><td>Unknown</td><td>n/a</td><td>-0.000</td><td>n/a</td><td>n/a</td></tr>\n"
        ));
        assert_eq!(html.matches("<tr>").count(), 3, "header plus two data rows");
        assert!(html.starts_with(TOF_HEADING_AND_DESCRIPTION));
    }

    #[test]
    fn absent_and_empty_timing_rows_produce_no_section() {
        for metadata in [
            serde_json::json!({}),
            serde_json::json!({"timing_diagnostics": null}),
            serde_json::json!({"timing_diagnostics": {"channels": []}}),
        ] {
            assert_eq!(time_of_flight_html(&metadata), "");
            assert_eq!(time_of_flight_rows(&metadata), serde_json::json!([]));
        }
    }

    #[test]
    fn extracted_rows_match_legacy_keys_and_preserve_topology_order() {
        let metadata = serde_json::json!({"timing_diagnostics": {"channels": [
            {"name": "Sub 2", "measured_arrival_ms": -0.25,
             "applied_delay_ms": 1.75, "final_arrival_ms": 1.5,
             "final_offset_from_reference_ms": -3.0},
            {"name": "Front R", "final_arrival_ms": 0.3333},
            false,
            {"measured_arrival_ms": 7}
        ]}});
        assert_eq!(
            time_of_flight_rows(&metadata),
            serde_json::json!([
                {"name":"Sub 2", "before_ms":-0.25, "applied_ms":1.75,
                 "after_ms":1.5, "offset_ms":-3.0},
                {"name":"Front R", "before_ms":null, "applied_ms":null,
                 "after_ms":0.3333, "offset_ms":null},
                {"name":"?", "before_ms":7, "applied_ms":null,
                 "after_ms":null, "offset_ms":null}
            ])
        );
    }

    #[test]
    fn channel_names_use_python_string_conversion_and_html_escaping() {
        let metadata = serde_json::json!({"timing_diagnostics": {"channels": [
            {"name": "L &'\"", "measured_arrival_ms": 1},
            {"name": null, "measured_arrival_ms": 2},
            {"name": ["x", null], "measured_arrival_ms": 3},
            {"name": {"x": "y"}, "measured_arrival_ms": 4}
        ]}});
        let html = time_of_flight_html(&metadata);
        assert!(html.contains("<tr><td>L &amp;&#x27;&quot;</td><td>1.000</td>"));
        assert!(html.contains("<tr><td>None</td><td>2.000</td>"));
        assert!(html.contains("<tr><td>[&#x27;x&#x27;, None]</td><td>3.000</td>"));
        assert!(html.contains("<tr><td>{&#x27;x&#x27;: &#x27;y&#x27;}</td><td>4.000</td>"));
    }

    #[test]
    fn decimal_rounding_matches_python_fixed_point_format() {
        let metadata = serde_json::json!({"timing_diagnostics": {"channels": [
            {"name": "a", "measured_arrival_ms": 1.2345},
            {"name": "b", "measured_arrival_ms": 1.2355},
            {"name": "c", "measured_arrival_ms": 2.6755},
            {"name": "d", "measured_arrival_ms": 0.0005},
            {"name": "e", "measured_arrival_ms": -0.0005}
        ]}});
        let html = time_of_flight_html(&metadata);
        for row in [
            "<tr><td>a</td><td>1.234</td>",
            "<tr><td>b</td><td>1.236</td>",
            "<tr><td>c</td><td>2.675</td>",
            "<tr><td>d</td><td>0.001</td>",
            "<tr><td>e</td><td>-0.001</td>",
        ] {
            assert!(
                html.contains(row),
                "missing Python-compatible formatted row {row}"
            );
        }
    }
}
