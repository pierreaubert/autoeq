//! Composes saved acoustic evidence and decision-ledger summaries.
//!
//! This module formats the report's operational response table using only
//! emitted ledger and analyzer values. It delegates acoustic metrics to the
//! canonical curve, reflection, and measured-T60 modules.

// Rust guideline compliant 2026-02-21

use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

use crate::acoustic_curves::{deepest_notch_db, notch_metric};
use crate::acoustic_reflections::reflection_metrics;
use crate::acoustic_t60::summary_metrics;
use crate::report_document::{
    DiagnosticBlock, MetricCell, MetricTone, ResultsRow, diagnostic_html,
};

const DELIVERED_STATUSES: [&str; 3] = ["applied", "already_acceptable", "constrained"];

/// Builds operational ledger shares indexed by physical output channel.
///
/// Emitted `channel_summaries` are preferred. Legacy `decisions` are reduced
/// using final equalization scope, excluding superseded and provisional rows.
/// Each value is `[percentage, decided_count, delivered_count]`.
///
/// # Panics
///
/// This function does not panic for malformed JSON values.
pub fn operational_summaries_by_channel(data: &Value) -> Value {
    let ledger = data.get("correction_decisions").and_then(Value::as_object);
    let mut indexed = Map::new();
    if let Some(summaries) = ledger
        .and_then(|ledger| ledger.get("channel_summaries"))
        .and_then(Value::as_array)
    {
        for entry in summaries {
            let Some(entry) = entry.as_object() else {
                continue;
            };
            let Some(channel) = entry.get("channel").and_then(Value::as_str) else {
                continue;
            };
            if channel.is_empty() {
                continue;
            }
            let Some(pct) = entry
                .get("operational_response_pct")
                .and_then(finite_number)
            else {
                continue;
            };
            indexed.insert(
                channel.to_owned(),
                Value::Array(vec![
                    json!(pct),
                    entry
                        .get("decided_equalize")
                        .cloned()
                        .unwrap_or(Value::Null),
                    entry
                        .get("delivered_equalize")
                        .cloned()
                        .unwrap_or(Value::Null),
                ]),
            );
        }
    }
    if !indexed.is_empty() {
        return Value::Object(indexed);
    }
    let decisions = ledger
        .and_then(|ledger| ledger.get("decisions"))
        .and_then(Value::as_array);
    decisions
        .filter(|decisions| !decisions.is_empty())
        .map(|decisions| derive_summaries_from_decisions(decisions.as_slice()))
        .unwrap_or(Value::Object(indexed))
}

fn derive_summaries_from_decisions(decisions: &[Value]) -> Value {
    let superseded: Vec<&str> = decisions
        .iter()
        .filter_map(|record| record.get("supersedes_ids").and_then(Value::as_array))
        .flat_map(|ids| ids.iter().filter_map(Value::as_str))
        .collect();
    let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for record in decisions {
        let Some(record) = record.as_object() else {
            continue;
        };
        if record.get("stage").and_then(Value::as_str) != Some("final")
            || record.get("action").and_then(Value::as_str) != Some("equalize")
            || !record
                .get("final_graph_identity")
                .and_then(Value::as_str)
                .is_some_and(|identity| !identity.trim().is_empty())
        {
            continue;
        }
        let decision_id = record.get("decision_id").and_then(Value::as_str);
        if decision_id.is_some_and(|decision_id| superseded.contains(&decision_id)) {
            continue;
        }
        let channel = record
            .get("physical_output")
            .filter(|value| is_truthy(value))
            .or_else(|| record.get("logical_input"));
        let Some(channel) = channel
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        else {
            continue;
        };
        let counter = counts.entry(channel.to_owned()).or_insert((0, 0));
        counter.0 += 1;
        if record
            .get("status")
            .and_then(Value::as_str)
            .is_some_and(|status| DELIVERED_STATUSES.contains(&status))
        {
            counter.1 += 1;
        }
    }

    Value::Object(
        counts
            .into_iter()
            .filter_map(|(channel, (decided, delivered))| {
                if delivered > decided {
                    return None;
                }
                Some((
                    channel,
                    json!([
                        100.0 * delivered as f64 / decided as f64,
                        decided,
                        delivered
                    ]),
                ))
            })
            .collect(),
    )
}

/// Renders the operational-response cell for one channel.
///
/// The summaries value uses the array representation returned by
/// [`operational_summaries_by_channel`]. Missing evidence remains unassessed.
///
/// # Panics
///
/// This function does not panic for malformed JSON values.
pub fn operational_response_cell(summaries_by_channel: &Value, name: &str) -> String {
    operational_response_metric(summaries_by_channel, name).browser_html()
}

fn operational_response_metric(summaries_by_channel: &Value, name: &str) -> MetricCell {
    let hit = summaries_by_channel.get(name).and_then(Value::as_array);
    let Some((hit, pct)) = hit.and_then(|hit| Some((hit, finite_number(hit.first()?)?))) else {
        return MetricCell {
            value: "Not assessed".into(),
            tone: MetricTone::Unassessed,
            detail: Some("No final EQ assessment is present in the saved decision ledger".into()),
        };
    };
    let decided = hit.get(1).unwrap_or(&Value::Null);
    let delivered = hit.get(2).unwrap_or(&Value::Null);
    let tone = if pct > 90.0 {
        MetricTone::Good
    } else if pct >= 80.0 {
        MetricTone::Warning
    } else {
        MetricTone::Poor
    };
    let detail = if is_python_integer(decided) && is_python_integer(delivered) {
        format!(
            "{}/{} final EQ scope delivered",
            python_string(delivered),
            python_string(decided)
        )
    } else {
        "final EQ scope share".to_owned()
    };
    MetricCell {
        value: format!("{pct:.1}"),
        tone,
        detail: Some(format!(
            "operational room response: {detail} (share of decided final equalization records delivered: Applied/AlreadyAcceptable/Constrained; empty scope stays pending, never 100%)"
        )),
    }
}

/// Escapes a missing-evidence field name in the report's pending cell.
///
/// # Panics
///
/// This function does not panic.
pub fn pending_cell(field: &str) -> String {
    MetricCell::pending(field).browser_html()
}

/// Composes the acoustic summary table from serialized report evidence.
///
/// Acoustic cells delegate to their established report modules; absent metrics
/// remain pending and channels are displayed in lexical order.
///
/// # Panics
///
/// This function does not panic for malformed or incomplete JSON values.
pub fn summary_table_html(data: &Value) -> String {
    let summaries = operational_summaries_by_channel(data);
    diagnostic_html(&[summary_block(data, &summaries)])
}

/// Builds the complete operational summary report response.
///
/// The response contains operational summaries, individual operational cells,
/// and the exact HTML table used by the standalone report.
///
/// # Panics
///
/// This function does not panic for malformed or incomplete JSON values.
pub fn operational_summary_report(data: &Value) -> Value {
    let summaries = operational_summaries_by_channel(data);
    let response_cells = data
        .get("channels")
        .and_then(Value::as_object)
        .map(|channels| {
            channels
                .keys()
                .map(|name| {
                    (
                        name.clone(),
                        Value::String(operational_response_cell(&summaries, name)),
                    )
                })
                .collect::<Map<_, _>>()
        })
        .unwrap_or_default();
    let table_html = diagnostic_html(&[summary_block(data, &summaries)]);
    json!({
        "operational_summaries_by_channel": summaries,
        "operational_response_cells": response_cells,
        "summary_table_html": table_html,
    })
}

pub(crate) fn results_summary(data: &Value) -> DiagnosticBlock {
    summary_block(data, &operational_summaries_by_channel(data))
}

fn summary_block(data: &Value, summaries: &Value) -> DiagnosticBlock {
    let Some(channels) = data["channels"].as_object() else {
        return DiagnosticBlock::ResultsSummary(Vec::new());
    };
    let rows = summary_metrics(data, channels)
        .into_iter()
        .map(|(name, [flatness, itu])| {
            let channel = &channels[&name];
            let [level, ratio] = reflection_metrics(channel);
            let operational = operational_response_metric(summaries, &name);
            ResultsRow {
                speaker: name,
                metrics: [
                    operational,
                    level,
                    ratio,
                    flatness,
                    itu,
                    notch_metric(deepest_notch_db(channel)),
                ],
            }
        })
        .collect();
    DiagnosticBlock::ResultsSummary(rows)
}

fn finite_number(value: &Value) -> Option<f64> {
    let number = value.as_f64()?;
    number.is_finite().then_some(number)
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn is_python_integer(value: &Value) -> bool {
    value.is_boolean() || value.as_i64().is_some() || value.as_u64().is_some()
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn golden() -> Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../scripts/tests/fixtures/acoustic_summary_python_goldens.json"
        );
        serde_json::from_str(&fs::read_to_string(path).expect("Python golden fixture"))
            .expect("valid fixture JSON")
    }

    #[test]
    fn ledger_summaries_and_summary_html_match_pre_port_python_goldens() {
        let fixture = golden();
        for case in fixture["cases"].as_object().unwrap().values() {
            let data = &case["input"];
            let expected = case;
            let report = operational_summary_report(data);
            assert_eq!(report["summary_table_html"], expected["summary_table_html"]);
            assert_eq!(
                report["operational_summaries_by_channel"],
                expected["summaries"]
            );
            for (name, cell) in expected["cells"].as_object().unwrap() {
                assert_eq!(
                    operational_response_cell(&report["operational_summaries_by_channel"], name),
                    *cell
                );
            }
        }
        assert_eq!(
            pending_cell(fixture["pending_cell"]["field"].as_str().unwrap()),
            fixture["pending_cell"]["html"]
        );
        assert_eq!(
            operational_response_cell(&json!({}), "missing"),
            fixture["unmatched_cell"]
        );
    }

    #[test]
    fn emitted_summary_precedence_and_missing_scope_remain_pending() {
        let fixture = golden();
        let emitted = &fixture["cases"]["emitted_summaries"]["input"];
        let summaries = operational_summaries_by_channel(emitted);
        assert_eq!(summaries["R"][0], 80.0);
        assert_eq!(summaries["D"][0], 0.0);
        assert_eq!(
            operational_response_cell(&summaries, "missing"),
            fixture["unmatched_cell"]
        );
    }
}
