//! Assembles prepared dataset sections into the RoomEQ dashboard layout.
//!
//! This module owns only the presentation boundary between prepared report
//! sections and the shared dashboard payload. It does not calculate curves,
//! render figures, or derive acoustic evidence. Cross-mode comparison reports
//! remain assembled by their dedicated comparison path.

// Rust guideline compliant 2026-02-21

use serde_json::{Map, Value, json};

const TAB_GROUPS: &[&str] = &["speakers", "time_domain", "epa"];

fn section_array_mut<'a>(
    groups: &'a mut Map<String, Value>,
    role: &str,
) -> Result<&'a mut Vec<Value>, String> {
    match groups
        .entry(role.to_owned())
        .or_insert_with(|| Value::Array(Vec::new()))
    {
        Value::Array(values) => Ok(values),
        _ => Err(format!(
            "report assembly global group `{role}` must be an array"
        )),
    }
}

fn add_tab_sections(
    groups: &mut Map<String, Value>,
    block: &Value,
    block_index: usize,
) -> Result<(), String> {
    let label = block.get("label").and_then(Value::as_str).ok_or_else(|| {
        format!("report assembly channel block {block_index} requires a string label")
    })?;
    let block_groups = block
        .get("groups")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            format!("report assembly channel block {block_index} requires a groups object")
        })?;

    for (role, value) in block_groups {
        if !TAB_GROUPS.contains(&role.as_str()) {
            return Err(format!(
                "report assembly channel block {block_index} has unsupported group `{role}`"
            ));
        }
        let sections = value.as_array().ok_or_else(|| {
            format!("report assembly channel block {block_index} group `{role}` must be an array")
        })?;
        for (section_index, section) in sections.iter().enumerate() {
            let object = section.as_object().ok_or_else(|| {
                format!(
                    "report assembly channel block {block_index} group `{role}` section {section_index} must be an object"
                )
            })?;
            if let Some(existing) = object.get("tab") {
                match existing {
                    Value::Null => {}
                    Value::String(existing) if existing == label => {}
                    _ => {
                        return Err(format!(
                            "report assembly channel block {block_index} group `{role}` section {section_index} has a mismatched tab"
                        ));
                    }
                }
            }

            let mut section = section.clone();
            section
                .as_object_mut()
                .ok_or("validated report section must remain an object")?
                .insert("tab".into(), json!(label));
            section_array_mut(groups, role)?.push(section);
        }
    }
    Ok(())
}

/// Orders prepared channel sections before assembling the canonical dashboard.
///
/// `global_groups` contains prepared role sections that are not scoped to a
/// channel tab. `channel_blocks` carries per-tab section arrays in their
/// established display order; the block label is applied to each section while
/// all other section fields are preserved. Global sections are emitted before
/// channel sections within each role. The final role order, notes, empty states,
/// and provenance filtering remain owned by [`crate::dashboard_payload`].
///
/// # Examples
/// ```
/// use autoeq_report_wasm::report_assembly::report_assembly;
/// use serde_json::json;
/// let result = report_assembly(&json!({
///     "title": "Room EQ",
///     "global_groups": {},
///     "channel_blocks": []
/// })).unwrap();
/// assert_eq!(result["schema"], "autoeq-report-data-v1");
/// ```
///
/// # Errors
/// Returns an error when groups, blocks, or prepared section shapes are invalid.
pub fn report_assembly(request: &Value) -> Result<Value, String> {
    let mut payload = assemble_report(request)?;
    crate::report_document::project_sections(&mut payload)?;
    Ok(payload)
}

pub(crate) fn assemble_report(request: &Value) -> Result<Value, String> {
    let mut groups = match request.get("global_groups") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(groups)) => groups.clone(),
        _ => return Err("report assembly global_groups must be an object".into()),
    };

    let blocks = match request.get("channel_blocks") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(blocks)) => blocks.as_slice(),
        _ => return Err("report assembly channel_blocks must be an array".into()),
    };
    for (block_index, block) in blocks.iter().enumerate() {
        if !block.is_object() {
            return Err(format!(
                "report assembly channel block {block_index} must be an object"
            ));
        }
        add_tab_sections(&mut groups, block, block_index)?;
    }

    let dashboard_request = json!({
        "title": request.get("title").unwrap_or(&Value::Null),
        "status_sections": request.get("status_sections").unwrap_or(&Value::Null),
        "groups": Value::Object(groups),
        "provenance": request.get("provenance").unwrap_or(&Value::Null),
        "relative_levels_html": request.get("relative_levels_html").unwrap_or(&Value::Null),
        "relative_levels": request.get("relative_levels").unwrap_or(&Value::Null),
        "modes_html": request.get("modes_html").unwrap_or(&Value::Null),
        "modes": request.get("modes").unwrap_or(&Value::Null),
    });
    crate::dashboard_payload::assemble_dashboard(&dashboard_request)
}

#[cfg(test)]
mod tests {
    use super::report_assembly;
    use serde_json::{Value, json};

    #[test]
    fn captured_python_dataset_sections_match_dashboard_payload() {
        let cases: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/report_assembly_python_goldens.json"
        ))
        .expect("Python assembly goldens must be valid JSON");
        for case in cases.as_array().expect("goldens must be an array") {
            assert_eq!(
                report_assembly(&case["request"]).expect("prepared report should assemble"),
                case["expected"]
            );
        }
    }

    #[test]
    fn tab_blocks_apply_labels_and_preserve_other_section_fields() {
        let output = report_assembly(&json!({
            "global_groups": {"speakers": [{"kind":"html","html":"global"}]},
            "channel_blocks": [{
                "label": "L",
                "groups": {"speakers": [{"kind":"figure","figure":{"title":"prepared"},"custom":true}]}
            }]
        }))
        .expect("valid sections should assemble");
        let speaker_sections: Vec<_> = output["sections"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|section| section["subhead"] == "Details per speaker")
            .collect();
        assert_eq!(speaker_sections[0]["html"], "global");
        assert_eq!(speaker_sections[1]["tab"], "L");
        assert_eq!(speaker_sections[1]["custom"], true);
    }

    #[test]
    fn rejects_mismatched_tabs_and_unknown_tab_roles() {
        assert!(report_assembly(&json!({
            "channel_blocks": [{"label":"L","groups":{"speakers":[{"kind":"html","html":"x","tab":"R"}]}}]
        }))
        .is_err());
        assert!(
            report_assembly(&json!({
                "channel_blocks": [{"label":"L","groups":{"overview":[{"kind":"html","html":"x"}]}}]
            }))
            .is_err()
        );
    }
}
