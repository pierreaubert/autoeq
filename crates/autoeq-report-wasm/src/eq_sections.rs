//! Renders prepared EQ filter listings and summary HTML.
//!
//! Filter ownership and Kautz section validation remain in the report data
//! preparation boundary. This module formats those descriptors into the exact
//! filter rows, per-origin tabs, and all-filter tables used by report sections.
// Rust guideline compliant 2026-02-21

use std::fmt::Write as _;

use crate::report_helpers::escape_html;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Failure while selecting an EQ section HTML builder.
pub enum EqSectionError {
    /// The requested builder name is not supported by this module.
    UnsupportedBuilder(String),
}

impl std::fmt::Display for EqSectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedBuilder(builder) => {
                write!(formatter, "unsupported EQ section builder: {builder}")
            }
        }
    }
}

impl std::error::Error for EqSectionError {}

/// Renders one EQ filter-listing component from prepared descriptors.
///
/// `builder` selects `filter_line`, `passes`, `filter_section`,
/// `filter_table`, `driver_filters`, `all_filters`, or `summary`. The JSON value
/// contains that builder's prepared input.
/// Unknown builders return an error because they indicate a caller contract
/// bug, rather than absent optional report evidence.
///
/// # Examples
/// ```
/// use serde_json::json;
/// use autoeq_report_wasm::eq_sections::eq_section_html;
///
/// let html = eq_section_html("filter_line", &json!({
///     "number": 1,
///     "filter": {"filter_type":"peak", "freq": 1000, "q": 1, "db_gain": 2}
/// })).unwrap();
/// assert!(html.contains("Filter 1: PEAK @ 1000.0 Hz"));
/// ```
///
/// # Errors
/// Returns [`EqSectionError::UnsupportedBuilder`] when `builder` is unknown.
///
/// # Panics
/// This function does not panic for malformed or incomplete JSON values.
pub fn eq_section_html(builder: &str, prepared: &Value) -> Result<String, EqSectionError> {
    let html = match builder {
        "filter_line" => format_eq_filter_line(
            &prepared["filter"],
            usize_value(&prepared["number"]).unwrap_or(1),
        ),
        "passes" => eq_passes_list_html(&prepared["passes"]),
        "filter_section" => filter_section_html(prepared),
        "filter_table" => eq_filter_table_html(&prepared["passes"]),
        "driver_filters" => driver_eq_filters_html(prepared),
        "all_filters" => all_eq_filters_html(&prepared["panels"]),
        "summary" => eq_filter_summary_html(&prepared["counts"]),
        _ => return Err(EqSectionError::UnsupportedBuilder(builder.to_owned())),
    };
    Ok(html)
}

/// Prepared EQ filter text grouped by origin and processing pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqFilterListing {
    /// Listing heading.
    pub title: String,
    /// Explanatory note shown before the filter origins.
    pub note: Option<String>,
    /// Driver and shared filter origins, or a single unlabelled origin.
    pub origins: Vec<EqFilterOrigin>,
    style: EqFilterStyle,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum EqFilterStyle {
    Channel,
    Driver,
}

/// A filter origin within the listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqFilterOrigin {
    /// Display label, including the number of filters when supplied.
    pub label: String,
    /// Ordered filter passes.
    pub passes: Vec<EqFilterPass>,
    id: String,
}

/// A group of filters from one processing pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqFilterPass {
    /// Pass label and color when the source describes a labelled pipeline.
    pub heading: Option<(String, String)>,
    /// Filter descriptions in execution order.
    pub filters: Vec<EqFilterLine>,
}

/// A filter description and any Kautz bank sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqFilterLine {
    /// Main filter description.
    pub summary: String,
    /// Pole, Q, and weight for each Kautz section.
    pub details: Vec<String>,
    kautz: bool,
}

impl EqFilterLine {
    fn write_html(&self, output: &mut String) {
        output.push_str(&escape_html(&self.summary));
        if self.kautz {
            output.push_str("; ");
            output.push_str(&escape_html(&self.details.join("; ")));
        }
        output.push_str("<br>\n");
    }
}

fn filter_section_html(prepared: &Value) -> String {
    channel_filter_listing(prepared).html()
}

pub(crate) fn channel_filter_listing(prepared: &Value) -> EqFilterListing {
    EqFilterListing {
        title: prepared["title"].as_str().unwrap_or("EQ Filters").into(),
        note: None,
        origins: vec![EqFilterOrigin {
            label: String::new(),
            passes: prepare_passes(
                &prepared["passes"],
                prepared["labeled"].as_bool().unwrap_or(false),
            ),
            id: String::new(),
        }],
        style: EqFilterStyle::Channel,
    }
}

impl EqFilterListing {
    pub(crate) fn html(&self) -> String {
        match self.style {
            EqFilterStyle::Channel => {
                let mut html = format!(
                    "<div class=\"filters-section\">\n    <h3>{}</h3>\n",
                    escape_html(&self.title)
                );
                if let Some(origin) = self.origins.first() {
                    write_passes(&mut html, &origin.passes, "    ", "", "");
                }
                html.push_str("</div>\n");
                html
            }
            EqFilterStyle::Driver => {
                let mut html = format!(
                    "\n                <div class=\"filters-section\">\n                    <h3>{}</h3>\n",
                    escape_html(&self.title)
                );
                if let Some(note) = &self.note {
                    let _ = writeln!(
                        html,
                        "                    <p class=\"epa-footer\">{}</p>",
                        escape_html(note)
                    );
                }
                if self.origins.len() > 1 {
                    html.push_str("                    <div class=\"eq-tabs\">\n                        <div class=\"eq-tab-header\">\n");
                    for (index, origin) in self.origins.iter().enumerate() {
                        let active = if index == 0 { " active" } else { "" };
                        let _ = writeln!(
                            html,
                            "                            <button class=\"eq-tab-btn{active}\" onclick=\"openEqTab(event, '{}')\">{}</button>",
                            safe_id_prefix(&origin.id),
                            escape_html(&origin.label)
                        );
                    }
                    html.push_str("                        </div>\n");
                    for (index, origin) in self.origins.iter().enumerate() {
                        let active = if index == 0 { " active" } else { "" };
                        let _ = writeln!(
                            html,
                            "                        <div id=\"{}\" class=\"eq-tab-panel{active}\">",
                            safe_id_prefix(&origin.id)
                        );
                        write_passes(
                            &mut html,
                            &origin.passes,
                            "                    ",
                            "\n",
                            "                    ",
                        );
                        html.push_str("                        </div>\n");
                    }
                    html.push_str("                    </div>\n");
                } else if let Some(origin) = self.origins.first() {
                    let _ = writeln!(
                        html,
                        "                    <h4>{}</h4>",
                        escape_html(&origin.label)
                    );
                    write_passes(
                        &mut html,
                        &origin.passes,
                        "                    ",
                        "\n",
                        "                    ",
                    );
                }
                html.push_str("                </div>\n");
                html
            }
        }
    }
}

fn eq_filter_summary_html(value: &Value) -> String {
    let counts: Vec<u64> = value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .collect();
    if counts.is_empty() {
        return "<td style=\"color:#999\">-</td>".to_string();
    }
    let minimum = counts.iter().min().copied().unwrap_or(0);
    let maximum = counts.iter().max().copied().unwrap_or(0);
    if minimum == maximum {
        return format!("<td>{minimum}</td>");
    }
    let average = counts.iter().sum::<u64>() as f64 / counts.len() as f64;
    format!("<td>{minimum}-{maximum} (avg {average:.1})</td>")
}

fn format_eq_filter_line(filter: &Value, number: usize) -> String {
    let mut html = String::new();
    prepare_filter_line(filter, number).write_html(&mut html);
    html
}

fn prepare_filter_line(filter: &Value, number: usize) -> EqFilterLine {
    if filter["topology"] == "kautz_filter" {
        let entries = filter["sections"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(index, section)| {
                format!(
                    "section {}: pole={} Hz, Q={}, weight={} (linear)",
                    index + 1,
                    fixed(number_value(&section["pole"]).unwrap_or(0.0), 1),
                    fixed(number_value(&section["q"]).unwrap_or(1.0), 2),
                    significant(number_value(&section["weight"]).unwrap_or(0.0), 6),
                )
            })
            .collect();
        return EqFilterLine {
            summary: format!("Filter {number}: KAUTZ bank (unity dry path)"),
            details: entries,
            kautz: true,
        };
    }

    let mut filter_type = filter["filter_type"]
        .as_str()
        .unwrap_or("peak")
        .to_uppercase();
    if filter["topology"] == "warped_biquad" {
        filter_type = format!("WARPED {filter_type}");
    }
    EqFilterLine {
        summary: format!(
            "Filter {number}: {} @ {} Hz, Q={}, Gain={} dB",
            filter_type,
            fixed(number_value(&filter["freq"]).unwrap_or(0.0), 1),
            fixed(number_value(&filter["q"]).unwrap_or(1.0), 2),
            signed_fixed(number_value(&filter["db_gain"]).unwrap_or(0.0), 1),
        ),
        details: Vec::new(),
        kautz: false,
    }
}

fn has_labeled_passes(value: &Value) -> bool {
    value
        .as_array()
        .into_iter()
        .flatten()
        .any(|pass| truthy(&pass["label"]))
}

fn prepare_passes(value: &Value, labeled: bool) -> Vec<EqFilterPass> {
    let passes = value.as_array().into_iter().flatten();
    if labeled {
        passes
            .map(|pass| EqFilterPass {
                heading: Some((
                    pass["display_name"].as_str().unwrap_or("").into(),
                    pass["color"].as_str().unwrap_or("").into(),
                )),
                filters: pass["filters"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .enumerate()
                    .map(|(index, filter)| prepare_filter_line(filter, index + 1))
                    .collect(),
            })
            .collect()
    } else {
        vec![EqFilterPass {
            heading: None,
            filters: passes
                .flat_map(|pass| pass["filters"].as_array().into_iter().flatten())
                .enumerate()
                .map(|(index, filter)| prepare_filter_line(filter, index + 1))
                .collect(),
        }]
    }
}

fn write_passes(
    html: &mut String,
    passes: &[EqFilterPass],
    indent: &str,
    prefix: &str,
    close_indent: &str,
) {
    for pass in passes {
        if let Some((name, color)) = &pass.heading {
            let color = escape_html(color);
            let name = escape_html(name);
            let _ = write!(
                html,
                "{prefix}{indent}<h4 style=\"color: {color}; margin-bottom: 5px;\">{name}</h4>\n{indent}<div class=\"filter-list\" style=\"margin-bottom: 10px;\">\n"
            );
        } else {
            let _ = writeln!(html, "{prefix}{indent}<div class=\"filter-list\">");
        }
        for filter in &pass.filters {
            filter.write_html(html);
        }
        let _ = writeln!(html, "{close_indent}</div>");
    }
}

fn eq_passes_list_html(value: &Value) -> String {
    let mut html = String::new();
    write_passes(
        &mut html,
        &prepare_passes(value, has_labeled_passes(value)),
        "                    ",
        "\n",
        "                    ",
    );
    html
}

fn eq_filter_table_html(value: &Value) -> String {
    let passes = value.as_array().cloned().unwrap_or_default();
    let has_labeled = passes.iter().any(|pass| truthy(&pass["label"]));
    let mut output = String::new();
    for pass in &passes {
        let filters = pass["filters"].as_array().cloned().unwrap_or_default();
        if filters.is_empty() {
            continue;
        }
        if has_labeled {
            let color = escape_html(pass["color"].as_str().unwrap_or(""));
            let display_name = escape_html(pass["display_name"].as_str().unwrap_or(""));
            let _ = writeln!(
                output,
                "            <h4 style=\"color: {color}; margin-bottom: 5px;\">{display_name}</h4>"
            );
        }
        output.push_str(
            "            <table class=\"bm-table\">\n                <thead><tr><th>#</th><th>Type</th><th>Freq</th><th>Q</th><th>Gain</th></tr></thead>\n                <tbody>\n",
        );
        for (index, filter) in filters.iter().enumerate() {
            let number = index + 1;
            if filter["topology"] == "kautz_filter" || filter["topology"] == "warped_biquad" {
                let _ = writeln!(
                    output,
                    "<tr><td>{number}</td><td colspan=\"4\">{}</td></tr>",
                    format_eq_filter_line(filter, number)
                );
                continue;
            }
            let filter_type = filter["filter_type"]
                .as_str()
                .unwrap_or("peak")
                .to_uppercase();
            let frequency = numeric_table_cell(&filter["freq"], 1, " Hz");
            let q = numeric_table_cell(&filter["q"], 2, "");
            let gain = numeric_table_signed_cell(&filter["db_gain"], 1, " dB");
            let _ = writeln!(
                output,
                "                    <tr><td>{number}</td><td>{}</td><td>{frequency}</td><td>{q}</td><td>{gain}</td></tr>",
                escape_html(&filter_type)
            );
        }
        output.push_str("                </tbody>\n            </table>\n");
    }
    output
}

fn driver_eq_filters_html(prepared: &Value) -> String {
    driver_filter_listing(prepared).map_or_else(String::new, |listing| listing.html())
}

pub(crate) fn driver_filter_listing(prepared: &Value) -> Option<EqFilterListing> {
    let driver_passes = &prepared["driver_passes"];
    let shared_passes = &prepared["shared_passes"];
    let driver_count = filter_count(driver_passes);
    let shared_count = filter_count(shared_passes);
    if driver_count == 0 && shared_count == 0 {
        return None;
    }
    let total = driver_count + shared_count;
    let driver = prepared["driver_name"].as_str().unwrap_or("");
    let channel = prepared["channel_name"].as_str().unwrap_or("");
    let prefix = safe_id_prefix(prepared["id_prefix"].as_str().unwrap_or(""));
    let mut origins = Vec::new();
    if driver_count > 0 {
        origins.push(EqFilterOrigin {
            label: format!("Driver: {driver} ({driver_count})"),
            passes: prepare_passes(driver_passes, has_labeled_passes(driver_passes)),
            id: format!("{prefix}_drv"),
        });
    }
    if shared_count > 0 {
        origins.push(EqFilterOrigin {
            label: format!("Shared channel {channel} ({shared_count})"),
            passes: prepare_passes(shared_passes, has_labeled_passes(shared_passes)),
            id: format!("{prefix}_sh"),
        });
    }
    Some(EqFilterListing {
        title: "EQ Filters".into(),
        note: Some(format!(
            "{total} PEQ filter(s) total: {driver_count} driver + {shared_count} shared channel. Numbering restarts in each tab; the EQ plot shows the combined response."
        )),
        origins,
        style: EqFilterStyle::Driver,
    })
}

fn all_eq_filters_html(value: &Value) -> String {
    let panels = value.as_array().cloned().unwrap_or_default();
    if panels.is_empty() {
        return String::new();
    }
    let totals: Vec<usize> = panels
        .iter()
        .map(|panel| {
            panel["groups"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|group| filter_count(&group["passes"]))
                .sum()
        })
        .collect();
    if totals.iter().all(|total| *total == 0) {
        return String::new();
    }
    let mut output = String::from(
        "        <div class=\"plot-container\">\n            <h2>All EQ Filters</h2>\n            <p class=\"epa-footer\">Complete PEQ listing for every channel and driver (same data and numbering as the per-tab sections below). Select a channel to show its filters.</p>\n            <div class=\"eq-tabs\">\n                <div class=\"eq-tab-header\">\n",
    );
    for (index, panel) in panels.iter().enumerate() {
        let label = escape_html(panel["label"].as_str().unwrap_or(""));
        let active = if index == 0 { " active" } else { "" };
        let _ = writeln!(
            output,
            "                    <button class=\"eq-tab-btn{active}\" onclick=\"openEqTab(event, 'alleq_{index}')\">{label} ({})</button>",
            totals[index]
        );
    }
    output.push_str("                </div>\n");
    for (index, panel) in panels.iter().enumerate() {
        let active = if index == 0 { " active" } else { "" };
        let _ = writeln!(
            output,
            "                <div id=\"alleq_{index}\" class=\"eq-tab-panel{active}\">"
        );
        let label = escape_html(panel["label"].as_str().unwrap_or(""));
        let _ = writeln!(output, "            <h3>{label}</h3>");
        for group in panel["groups"].as_array().into_iter().flatten() {
            let origin = escape_html(group["origin"].as_str().unwrap_or(""));
            let passes = &group["passes"];
            let count = filter_count(passes);
            if count == 0 {
                continue;
            }
            let _ = writeln!(output, "            <h4>{origin} ({count})</h4>");
            output.push_str(&eq_filter_table_html(passes));
        }
        output.push_str("                </div>\n");
    }
    output.push_str("            </div>\n        </div>\n");
    output
}

fn filter_count(passes: &Value) -> usize {
    passes
        .as_array()
        .into_iter()
        .flatten()
        .map(|pass| pass["filters"].as_array().map_or(0, Vec::len))
        .sum()
}

fn numeric_table_cell(value: &Value, precision: usize, suffix: &str) -> String {
    number_value(value)
        .map(|number| format!("{}{}", fixed(number, precision), suffix))
        .unwrap_or_else(|| "-".to_string())
}

fn numeric_table_signed_cell(value: &Value, precision: usize, suffix: &str) -> String {
    number_value(value)
        .map(|number| format!("{}{}", signed_fixed(number, precision), suffix))
        .unwrap_or_else(|| "-".to_string())
}

fn number_value(value: &Value) -> Option<f64> {
    value.as_f64().filter(|value| value.is_finite())
}

fn usize_value(value: &Value) -> Option<usize> {
    value.as_u64().and_then(|value| usize::try_from(value).ok())
}

fn fixed(value: f64, precision: usize) -> String {
    format!("{value:.precision$}")
}

fn signed_fixed(value: f64, precision: usize) -> String {
    format!("{value:+.precision$}")
}

fn significant(value: f64, digits: usize) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "+0" }.to_string();
    }
    let exponent = value.abs().log10().floor() as i32;
    if exponent >= digits as i32 || exponent < -4 {
        let scientific = format!("{value:+.5e}");
        return normalize_exponent(scientific);
    }
    let decimal_places = (digits as i32 - 1 - exponent).max(0) as usize;
    signed_fixed(value, decimal_places)
}

fn normalize_exponent(value: String) -> String {
    let Some((mantissa, exponent)) = value.split_once('e') else {
        return value;
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return value;
    };
    format!("{mantissa}e{exponent:+03}")
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.is_empty(),
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn safe_id_prefix(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::eq_section_html;

    #[test]
    fn formats_biquad_and_kautz_filter_lines() {
        let peak = eq_section_html(
            "filter_line",
            &json!({"number":2,"filter":{"filter_type":"peak<script>","freq":55,"q":2,"db_gain":1.5}}),
        ).unwrap();
        assert_eq!(
            peak,
            "Filter 2: PEAK&lt;SCRIPT&gt; @ 55.0 Hz, Q=2.00, Gain=+1.5 dB<br>\n"
        );
        let kautz = eq_section_html(
            "filter_line",
            &json!({"number":1,"filter":{"topology":"kautz_filter","sections":[{"pole":500,"q":0.707,"weight":0.123456789}]}}),
        ).unwrap();
        assert!(kautz.contains("pole=500.0 Hz, Q=0.71, weight=+0.123457 (linear)"));
    }

    #[test]
    fn renders_driver_and_all_filter_tabs_with_safe_labels() {
        let passes = json!([{"label":"","display_name":"","color":"#abc","filters":[{"filter_type":"peak","freq":40,"q":1,"db_gain":-2.5}]}]);
        let driver = eq_section_html(
            "driver_filters",
            &json!({"driver_name":"woofer<script>","channel_name":"LFE","id_prefix":"eq_1","driver_passes":passes,"shared_passes":[]}),
        ).unwrap();
        assert!(driver.contains("woofer&lt;script&gt;"));
        let all = eq_section_html(
            "all_filters",
            &json!({"panels":[{"label":"Left<script>","groups":[{"origin":"Channel L","passes":passes}]}]}),
        ).unwrap();
        assert!(all.contains("Left&lt;script&gt;"));
        assert!(all.contains("-2.5 dB"));
    }

    #[test]
    fn shared_and_driver_filter_origins_keep_distinct_numbering() {
        let prepared = json!({
            "driver_name":"Woofer",
            "channel_name":"L",
            "id_prefix":"eq_1",
            "driver_passes":[{"filters":[{"topology":"kautz_filter","sections":[
                {"pole":80,"q":0.7,"weight":0.25}
            ]}]}],
            "shared_passes":[{"filters":[{"freq":1000,"q":1,"db_gain":-2.5}]}],
        });
        let listing = super::driver_filter_listing(&prepared).unwrap();
        assert_eq!(listing.origins[0].label, "Driver: Woofer (1)");
        assert_eq!(listing.origins[1].label, "Shared channel L (1)");
        let bank = &listing.origins[0].passes[0].filters[0];
        assert_eq!(bank.summary, "Filter 1: KAUTZ bank (unity dry path)");
        assert_eq!(
            bank.details,
            ["section 1: pole=80.0 Hz, Q=0.70, weight=+0.250000 (linear)"]
        );
        assert!(
            listing.origins[1].passes[0].filters[0]
                .summary
                .starts_with("Filter 1: PEAK")
        );
        let html = listing.html();
        assert!(html.contains("openEqTab(event, 'eq_1_drv')"));
        assert!(html.contains("openEqTab(event, 'eq_1_sh')"));
    }

    #[test]
    fn rejects_unknown_builder_and_escapes_prepared_html_values() {
        assert!(matches!(
            eq_section_html("surprise", &serde_json::json!({})),
            Err(super::EqSectionError::UnsupportedBuilder(builder)) if builder == "surprise"
        ));

        let pass = serde_json::json!([{
            "label":"labeled",
            "display_name":"<img src=x onerror=alert(1)>",
            "color":"red\" onmouseover=\"alert(1)",
            "filters":[{"filter_type":"peak","freq":40,"q":1,"db_gain":-2.5}]
        }]);
        for builder in ["passes", "filter_section", "filter_table"] {
            let input = if builder == "filter_section" {
                serde_json::json!({"title":"EQ","labeled":true,"passes":pass})
            } else {
                serde_json::json!({"passes":pass})
            };
            let html = eq_section_html(builder, &input).unwrap();
            assert!(!html.contains("<img"), "{builder}: {html}");
            assert!(!html.contains("onmouseover=\"alert"), "{builder}: {html}");
            assert!(html.contains("&lt;img"), "{builder}: {html}");
            assert!(
                html.contains("&quot; onmouseover=&quot;"),
                "{builder}: {html}"
            );
        }

        let driver = eq_section_html(
            "driver_filters",
            &serde_json::json!({
                "driver_name":"x<script>",
                "channel_name":"LFE",
                "id_prefix":"eq_';alert(1)//",
                "driver_passes":pass,
                "shared_passes":[{"filters":[{"filter_type":"peak","freq":40,"q":1,"db_gain":-2.5}]}]
            }),
        ).unwrap();
        assert!(driver.contains("x&lt;script&gt;"));
        assert!(!driver.contains("eq_';alert(1)//"));
        assert!(driver.contains("id=\"eq_alert1_drv\""));
    }
}
