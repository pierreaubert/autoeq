//! Groups prepared report sections into the canonical RoomEQ dashboard layout.
//!
//! Section generation remains separate from layout. Native and WASM consumers
//! share dashboard order, explanatory notes, unavailable states, and provenance.

// Rust guideline compliant 2026-02-21

use serde_json::{Value, json};

use crate::report_document::{paragraph_section, project_sections};

// Keep dashboard ordering consistent with the established standalone report.
const GROUPS: &[(&str, &str, &str)] = &[
    ("why", "DSP analysis", "Why this correction?"),
    ("overview", "Acoustics analysis", "Summary"),
    ("speakers", "Acoustics analysis", "Details per speaker"),
    ("timing", "Acoustics analysis", "Time of Flight"),
    ("symmetric", "Acoustics analysis", "Symmetric monitors"),
    ("time_domain", "Acoustics analysis", "Time domain analysis"),
    ("epa", "Psychoacoustic report", "Section 6: EPA scores"),
    ("flow", "DSP analysis", "DSP signal flow"),
];

fn html_section(html: &str) -> Value {
    json!({"kind":"html","html":html,"tab":null})
}

fn sections(value: Option<&Value>, name: &str) -> Result<Vec<Value>, String> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(values)) if values.iter().all(Value::is_object) => Ok(values.clone()),
        _ => Err(format!(
            "dashboard {name} must be an array of section objects"
        )),
    }
}

/// Assembles prepared sections with canonical grouping, notes, and provenance.
///
/// This function does not load files or generate missing measurement evidence.
/// `groups` contains prepared sections keyed by dashboard role; `status_sections`
/// appear first. Display timestamps must be supplied by the calling boundary.
///
/// # Examples
/// ```
/// use autoeq_report_wasm::dashboard_payload::dashboard_payload;
/// use serde_json::json;
/// let payload = dashboard_payload(&json!({"title":"Room EQ","groups":{}})).unwrap();
/// assert_eq!(payload["schema"], "autoeq-report-data-v1");
/// ```
///
/// # Errors
/// Returns an error when supplied groups or sections have invalid shapes.
pub fn dashboard_payload(request: &Value) -> Result<Value, String> {
    let mut payload = assemble_dashboard(request)?;
    project_sections(&mut payload)?;
    Ok(payload)
}

pub(crate) fn assemble_dashboard(request: &Value) -> Result<Value, String> {
    if request
        .get("groups")
        .is_some_and(|value| !value.is_null() && !value.is_object())
    {
        return Err("dashboard groups must be an object".into());
    }
    let groups = &request["groups"];
    let mut result = sections(request.get("status_sections"), "status_sections")?;
    for (role, bucket, subhead) in GROUPS {
        let mut contents = sections(groups.get(*role), role)?;
        if *role == "time_domain" {
            contents.insert(0,paragraph_section(&["Arrival timing and applied delays are in Time of Flight. The diagnostics below describe the measured room impulse responses."]));
            if request["modes"].is_object() {
                let mut section = request["modes"].clone();
                section["footer"] = json!(true);
                contents.push(section);
            } else if let Some(modes) = request["modes_html"]
                .as_str()
                .filter(|value| !value.is_empty())
            {
                contents.push(json!({"kind":"html","html":modes,"tab":null,"footer":true}));
            }
        } else if *role == "epa" {
            contents.insert(0,paragraph_section(&["EPA scores are model-based psychoacoustic estimates before and after DSP. They are not listening-test results or proof of improved audibility. Compare each metric using its stated direction and units."]));
        } else if *role == "speakers" {
            if request["relative_levels"].is_object() {
                contents.insert(0, request["relative_levels"].clone());
            } else if let Some(levels) = request["relative_levels_html"]
                .as_str()
                .filter(|value| !value.is_empty())
            {
                contents.insert(0, html_section(levels));
            }
        }
        if contents.is_empty() {
            contents.push(paragraph_section(&["No data available in this output."]));
        }
        for mut section in contents {
            // The section shape was validated before adding canonical grouping.
            let object = section
                .as_object_mut()
                .ok_or("dashboard section must be an object")?;
            object.insert("bucket".into(), json!(bucket));
            object.insert("subhead".into(), json!(subhead));
            result.push(section);
        }
    }
    let mut payload = json!({"schema":"autoeq-report-data-v1","title":request["title"].as_str().unwrap_or("Room EQ"),"sections":result});
    let mut provenance = serde_json::Map::new();
    for key in ["roomeq_version", "data_timestamp", "generated_at"] {
        if let Some(value) = request["provenance"].get(key).filter(|value| match value {
            Value::Null => false,
            Value::Bool(value) => *value,
            Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
            Value::String(value) => !value.is_empty(),
            Value::Array(value) => !value.is_empty(),
            Value::Object(value) => !value.is_empty(),
        }) {
            provenance.insert(key.into(), value.clone());
        }
    }
    if !provenance.is_empty() {
        payload["provenance"] = Value::Object(provenance);
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_full_python_dashboard_layouts_match() {
        let cases: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/dashboard_payload_python.json"
        ))
        .unwrap();
        for case in cases.as_array().unwrap() {
            assert_eq!(
                dashboard_payload(&case["request"]).unwrap(),
                case["expected"]
            );
        }
    }

    #[test]
    fn rejects_bad_group_shapes_and_does_not_mutate_prepared_input() {
        let request = json!({"groups":{"why":[{"kind":"html","html":"evidence","custom":true} ]},"provenance":{"generated_at":"fixed","unknown":"drop"}});
        let before = request.clone();
        let output = dashboard_payload(&request).unwrap();
        assert_eq!(request, before);
        assert_eq!(output["sections"][0]["custom"], true);
        assert!(output["provenance"].get("unknown").is_none());
        assert!(dashboard_payload(&json!({"groups":{"why":[null]}})).is_err());
        assert!(dashboard_payload(&json!({"groups":[]})).is_err());
    }
}
