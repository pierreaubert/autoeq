//! Builds the shared upstream report fragments consumed by native and Python callers.

// Rust guideline compliant 2026-02-21

use serde_json::{Map, Value};

use crate::bass_management::{
    bass_management_groups, bass_management_outputs, bass_management_summary,
};
use crate::crossover_config::crossover_tables;
use crate::gain_summary::gain_summary;
use crate::report_document::{
    TableStyle, crossover_section, project_section, section_fragment, table_section,
};
use crate::signal_flow::signal_flow_document_sections;
use crate::time_of_flight::time_of_flight_table;

/// Builds all renderer-neutral report sections from one full RoomEQ result value.
///
/// Strings hold HTML fragments. `signal_flow_sections` remains an array of
/// ordinary report section objects so the shared report shell can render it
/// through its existing section schema.
pub fn build_shared_sections(data: &Value) -> Map<String, Value> {
    let mut sections = build_shared_document_sections(data);
    project_shared_sections(&mut sections).expect("shared tables are serialized by AutoEQ");
    sections
}

pub(crate) fn project_shared_sections(sections: &mut Map<String, Value>) -> Result<(), String> {
    for key in [
        "gain_summary_html",
        "crossover_config_html",
        "time_of_flight_html",
        "bass_management_summary_html",
        "bass_management_groups_table_html",
        "bass_management_sub_outputs_table_html",
    ] {
        if let Some(value) = sections.get_mut(key) {
            *value = Value::String(section_fragment(value.take())?);
        }
    }
    if let Some(flow) = sections
        .get_mut("signal_flow_sections")
        .and_then(Value::as_array_mut)
    {
        for section in flow {
            project_section(section)?;
        }
    }
    Ok(())
}

pub(crate) fn build_shared_document_sections(data: &Value) -> Map<String, Value> {
    let mut sections = Map::new();
    let metadata = data.get("metadata").unwrap_or(&Value::Null);
    let bass = metadata.get("bass_management").unwrap_or(&Value::Null);

    sections.insert(
        "time_of_flight_html".to_string(),
        table_section(time_of_flight_table(metadata), TableStyle::Timing),
    );
    sections.insert(
        "crossover_config_html".to_string(),
        crossover_section(crossover_tables(data)),
    );
    sections.insert(
        "gain_summary_html".to_string(),
        table_section(gain_summary(data), TableStyle::Gain),
    );
    sections.insert(
        "bass_management_summary_html".to_string(),
        table_section(bass_management_summary(bass), TableStyle::BassSummary),
    );
    sections.insert(
        "bass_management_groups_table_html".to_string(),
        table_section(bass_management_groups(bass), TableStyle::BassTable),
    );
    sections.insert(
        "bass_management_sub_outputs_table_html".to_string(),
        table_section(bass_management_outputs(bass), TableStyle::BassTable),
    );
    sections.insert(
        "signal_flow_sections".to_string(),
        Value::Array(signal_flow_document_sections(data)),
    );
    sections
}

/// Builds global evidence HTML from an already prepared graph and its
/// externally verified payload-binding result.
pub fn build_global_evidence_sections(
    data: &Value,
    label: &str,
    payload_binding: &Value,
) -> Map<String, Value> {
    crate::evidence_sections::global_evidence_sections(data, label, payload_binding)
}

#[cfg(test)]
mod tests {
    use super::build_shared_sections;

    #[test]
    fn builds_all_shared_report_sections_in_one_value() {
        let data = serde_json::json!({
            "global_plugins": [{"plugin_type":"gain","parameters":{"gain_db":-1.0}}],
            "channels":{"L":{"plugins":[{"plugin_type":"gain","parameters":{"gain_db":-2.0}}]}},
            "metadata":{"time_of_flight":{"measurements":[]}, "bass_management":{"enabled":true}}
        });
        let sections = build_shared_sections(&data);
        assert!(
            sections["gain_summary_html"]
                .as_str()
                .unwrap()
                .contains("-1.000 dB")
        );
        assert!(
            sections["bass_management_summary_html"]
                .as_str()
                .unwrap()
                .contains("Bass Management")
        );
        assert!(sections["time_of_flight_html"].is_string());
        assert!(sections["crossover_config_html"].is_string());
        assert!(sections["signal_flow_sections"].is_array());
    }
}
