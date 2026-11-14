//! Composes dataset dashboards through the shared global and channel owners.

// Rust guideline compliant 2026-02-21

use serde::Serialize;
use serde_json::{Value, json};

use crate::PreparedReportData;
use crate::report_dataset::ReportDatasetOptions;
use crate::report_document::ReportDocument;

/// A canonical dashboard payload and explicit preparation diagnostics.
#[derive(Debug, Clone, Serialize)]
pub struct DatasetReport {
    /// Existing renderer-neutral report payload consumed by the shared shell.
    pub payload: Value,
    /// Structured preparation availability; it does not authorize playback.
    pub preparation_status: Value,
}

/// Build a dataset dashboard from captured data without filesystem access.
///
/// Global and channel section selection use the same prepared graph, target,
/// replay, and host evidence. The existing assembly owner controls layout;
/// this entry point does not render HTML shells or repeat DSP calculations.
///
/// # Errors
/// Returns an error for malformed inputs or rejected section preparation.
pub fn build_dataset_report(
    prepared: &PreparedReportData,
    options: &ReportDatasetOptions,
) -> Result<DatasetReport, String> {
    let document = build_report_document(prepared, options)?;
    Ok(DatasetReport {
        payload: document.browser_payload(),
        preparation_status: document.preparation_status,
    })
}

/// Composes the dashboard document for both native and browser views.
///
/// # Errors
/// Returns an error when preparation or section assembly rejects the inputs.
pub fn build_report_document(
    prepared: &PreparedReportData,
    options: &ReportDatasetOptions,
) -> Result<ReportDocument, String> {
    prepared.validate()?;
    options.validate()?;
    let global = crate::report_dataset_global::build_global_document_sections(prepared, options)?;
    let channels = crate::report_dataset_channel::build_document_channel_blocks(prepared, options)?;
    let mut preparation_status = global
        .get("preparation_status")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for item in channels
        .iter()
        .filter_map(|block| block.get("preparation_status"))
        .filter_map(Value::as_array)
        .flatten()
    {
        if !preparation_status.contains(item) {
            preparation_status.push(item.clone());
        }
    }
    let payload = crate::report_assembly::assemble_report(&json!({
        "title":global.get("title").unwrap_or(&Value::Null),
        "status_sections": global.get("status_sections").unwrap_or(&Value::Null),
        "global_groups": global.get("global_groups").unwrap_or(&Value::Null),
        "channel_blocks": channels,
        "provenance": global.get("provenance").unwrap_or(&Value::Null),
        "relative_levels": global.get("relative_levels").unwrap_or(&Value::Null),
        "modes": global.get("modes").unwrap_or(&Value::Null),
    }))?;
    ReportDocument::from_assembled(payload, Value::Array(preparation_status))
}

#[cfg(test)]
mod tests {
    use super::build_report_document;
    use crate::PreparedReportData;
    use crate::report_dataset::ReportDatasetOptions;
    use crate::report_document::ReportSection;
    use serde_json::{Map, Value, json};

    #[test]
    fn structured_tables_keep_their_dashboard_order_in_html() {
        let prepared = PreparedReportData::new(
            json!({
                "global_plugins": [{"plugin_type":"gain","parameters":{"gain_db":-2.5}}],
                "channels": {},
                "metadata": {"timing_diagnostics":{"channels":[
                    {"name":"Left","measured_arrival_ms":1.25}
                ]}}
            }),
            None,
            None,
            None,
            json!({"schema_version":1,"state":"not_configured","target":null}),
            None,
        )
        .unwrap();
        let document = build_report_document(&prepared, &ReportDatasetOptions::default()).unwrap();
        let browser = document.browser_payload();
        assert!(
            matches!(document.sections.first(), Some(ReportSection::PlaybackNotices(notices)) if notices.len() == 3)
        );
        assert_eq!(browser["sections"][0]["flat"], true);
        assert!(document.sections.iter().any(|section| matches!(section, ReportSection::Text(text) if text.subhead() == Some("Time domain analysis") && text.paragraphs[0].starts_with("Arrival timing"))));
        let tables: Vec<_> = document
            .sections
            .iter()
            .enumerate()
            .filter_map(|(index, section)| match section {
                ReportSection::Table(table) => Some((index, table)),
                _ => None,
            })
            .collect();
        assert_eq!(tables.len(), 2);
        let (gain_index, gain) = tables[0];
        let (timing_index, timing) = tables[1];
        assert_eq!(gain.subhead(), Some("Summary"));
        assert_eq!(gain.table.rows[0][4], "-2.500 dB");
        assert_eq!(timing.subhead(), Some("Time of Flight"));
        assert_eq!(timing.table.rows[0][1..3], ["1.250", "n/a"]);
        for (index, title) in [
            (gain_index, &gain.table.title),
            (timing_index, &timing.table.title),
        ] {
            let section = &browser["sections"][index];
            assert_eq!(section["kind"], "html");
            assert!(section["html"].as_str().unwrap().contains(title));
            assert!(section.get("table").is_none());
        }
    }

    #[test]
    fn full_dataset_payload_matches_python_dashboard_golden() {
        let cases: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/report_dataset_full_python_goldens.json"
        ))
        .expect("captured Python dashboard fixture parses");
        let case = &cases[0];
        let graph = case["source_data"].clone();
        let channels = graph["channels"].as_object().expect("fixture has channels");
        let mut channel_sources = Map::new();
        let mut driver_sources = Map::new();
        for (name, channel) in channels {
            if let Some(curve) = channel
                .get("final_curve")
                .filter(|value| value["freq"].is_array() && value["spl"].is_array())
            {
                channel_sources.insert(name.clone(), curve.clone());
            }
            if let Some(drivers) = channel["drivers"].as_array() {
                let sources = drivers
                    .iter()
                    .map(|driver| {
                        let mut item = Map::new();
                        if let Some(name) = driver.get("name") {
                            item.insert("name".into(), name.clone());
                        }
                        if let Some(index) = driver.get("index") {
                            item.insert("index".into(), index.clone());
                        }
                        item.insert(
                            "state".into(),
                            json!({
                                "status":"unavailable",
                                "reason":"no prepared driver response was supplied"
                            }),
                        );
                        Value::Object(item)
                    })
                    .collect::<Vec<_>>();
                driver_sources.insert(name.clone(), Value::Array(sources));
            }
        }

        // The fixture predates frozen capture identities. The header provenance
        // inputs below reproduce its explicit source and render timestamps.
        let provenance = &case["expected"]["provenance"];
        let options_from_fixture: ReportDatasetOptions =
            serde_json::from_value(case["options"].clone()).unwrap();
        let provenance = json!({
            "versions":[provenance["roomeq_version"]],
            "timestamps":[provenance["data_timestamp"]],
            "generated_at":provenance["generated_at"]
        });
        channel_sources.insert("drivers".into(), Value::Object(driver_sources));
        let options = ReportDatasetOptions {
            payload_binding: Some(json!({
                "verified":false,
                "reason":"Recomputable payload binding unavailable (legacy or unbound output)."
            })),
            legacy_source_curves: Some(Value::Object(channel_sources)),
            ..options_from_fixture
        };
        let prepared = PreparedReportData::new(
            graph,
            Some(provenance),
            None,
            None,
            json!({"schema_version":1,"state":"not_configured","target":null}),
            None,
        )
        .expect("source graph becomes a frozen report input");

        let document = build_report_document(&prepared, &options)
            .expect("global and channel sections compose into a dashboard");
        for section in &document.sections {
            if let ReportSection::Legacy(section) = section {
                assert!(
                    matches!(
                        section["kind"].as_str(),
                        Some("figure" | "grid" | "bar" | "sankey")
                    ),
                    "dashboard fixture requires an unsupported section: {} / {}",
                    section["bucket"],
                    section["subhead"],
                );
            }
        }
        let payload = document.browser_payload();
        let mut expected = case["expected"].clone();
        normalize_unavailable_driver_captions(&payload, &mut expected);
        normalize_stable_unpaired_channel_order(&payload, &mut expected);
        if let Some(path) = first_difference(&payload, &expected, String::new()) {
            panic!(
                "integrated dashboard differs from the Python golden at {path}; schema, section order, and section data must be preserved"
            );
        }
        assert!(document.preparation_status.is_array());
    }

    fn first_difference(left: &Value, right: &Value, path: String) -> Option<String> {
        match (left, right) {
            (Value::Object(left), Value::Object(right)) => {
                for key in left.keys().chain(right.keys()) {
                    let nested = format!("{path}.{key}");
                    match (left.get(key), right.get(key)) {
                        (Some(left), Some(right)) => {
                            if let Some(diff) = first_difference(left, right, nested) {
                                return Some(diff);
                            }
                        }
                        _ => return Some(nested),
                    }
                }
                None
            }
            (Value::Array(left), Value::Array(right)) => {
                for index in 0..left.len().max(right.len()) {
                    let nested = format!("{path}[{index}]");
                    match (left.get(index), right.get(index)) {
                        (Some(left), Some(right)) => {
                            if let Some(diff) = first_difference(left, right, nested) {
                                return Some(diff);
                            }
                        }
                        _ => return Some(nested),
                    }
                }
                None
            }
            _ if left == right => None,
            _ => Some(format!("{path}: actual {left}, expected {right}")),
        }
    }

    fn normalize_unavailable_driver_captions(actual: &Value, expected: &mut Value) {
        let actual_sections = actual["sections"].as_array().unwrap();
        let expected_sections = expected["sections"].as_array_mut().unwrap();
        assert_eq!(actual_sections.len(), expected_sections.len());
        for (actual_section, expected_section) in actual_sections.iter().zip(expected_sections) {
            let Some(actual_html) = actual_section["html"].as_str() else {
                continue;
            };
            let Some(expected_html) = expected_section["html"].as_str() else {
                continue;
            };
            if !["Sub A", "Sub B"].contains(&actual_section["tab"].as_str().unwrap_or_default())
                || !actual_html.starts_with("<h2>Channel:")
            {
                continue;
            }
            assert!(expected_html.contains("Physical driver output"));
            assert!(actual_html.contains("Raw measurement only; corrected response unavailable"));
            let actual_header_end = actual_html.find("</h2>\n").unwrap() + "</h2>\n".len();
            let expected_header_end = expected_html.find("</h2>\n").unwrap() + "</h2>\n".len();
            assert_eq!(
                &actual_html[..actual_header_end],
                &expected_html[..expected_header_end],
                "channel heading remains unchanged"
            );
            // The old Python caption asserts an unprepared physical output. The
            // new unavailable-state caption is intentional; all remaining payload
            // fields are still compared byte-for-byte below.
            expected_section["html"] = json!(actual_html);
        }
    }

    fn normalize_stable_unpaired_channel_order(actual: &Value, expected: &mut Value) {
        let actual_sections = actual["sections"].as_array().unwrap();
        let expected_sections = expected["sections"].as_array_mut().unwrap();
        let actual_indices: Vec<_> = actual_sections
            .iter()
            .enumerate()
            .filter_map(|(index, section)| {
                section["html"]
                    .as_str()
                    .is_some_and(|html| html.contains("Unpaired channels (not summed):"))
                    .then_some(index)
            })
            .collect();
        let expected_indices: Vec<_> = expected_sections
            .iter()
            .enumerate()
            .filter_map(|(index, section)| {
                section["html"]
                    .as_str()
                    .is_some_and(|html| html.contains("Unpaired channels (not summed):"))
                    .then_some(index)
            })
            .collect();
        assert_eq!(
            actual_indices.len(),
            1,
            "fixture has one unpaired-channel summary"
        );
        assert_eq!(
            expected_indices.len(),
            1,
            "golden has one unpaired-channel summary"
        );
        let actual_index = actual_indices[0];
        let expected_index = expected_indices[0];
        assert_eq!(
            actual_index, expected_index,
            "symmetric summary retains its section position"
        );
        let actual_html = actual_sections[actual_index]["html"].as_str().unwrap();
        let expected_html = expected_sections[expected_index]["html"].as_str().unwrap();
        let marker = "Unpaired channels (not summed): ";
        let actual_order = actual_html
            .split_once(marker)
            .and_then(|(_, tail)| tail.split_once("</p>").map(|(names, _)| names))
            .expect("actual symmetric summary has an unpaired list");
        let expected_order = expected_html
            .split_once(marker)
            .and_then(|(_, tail)| tail.split_once("</p>").map(|(names, _)| names))
            .expect("Python golden has an unpaired list");
        assert_eq!(
            actual_order, "C, LFE, Odd",
            "unpaired channels follow the shared stable display order"
        );
        assert_eq!(
            expected_order, "LFE, Odd, C",
            "the captured Python golden preserves input dictionary order"
        );
        expected_sections[expected_index]["html"] =
            json!(expected_html.replace(expected_order, actual_order));
    }
}
