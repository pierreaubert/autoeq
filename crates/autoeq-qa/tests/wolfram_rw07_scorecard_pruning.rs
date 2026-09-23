//! Wolfram cross-check: pruning-matrix audit reconciliation (RW07).
//!
//! Oracle: `wolfram/rw07_scorecard_pruning.wls` (independent
//! re-derivation of the audit rules from the workflow contract).
//! The Rust test rebuilds the row table from the golden inputs and
//! calls the real `roomeq_workflow::pruning_audit::audit_pruning_matrix`.
//! Exact identity on counts and reason strings (class I/X).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_workflow::pruning_audit::{
    MatrixRowOutcome, PruningFilterClass, PruningMatrixRow, PruningMode, PruningRoutedVariant,
    audit_pruning_matrix,
};

const CASE: &str = "rw07_scorecard_pruning";
const CASE_ID: &str = "autoeq-qa.rw07-scorecard-pruning.v1";

fn parse_row(v: &serde_json::Value) -> PruningMatrixRow {
    let mode = match v["mode"].as_str().unwrap() {
        "Ordinary" => PruningMode::Ordinary,
        "Adaptive" => PruningMode::Adaptive,
        "Pareto" => PruningMode::Pareto,
        "Refinement" => PruningMode::Refinement,
        other => panic!("{CASE}: unknown mode {other}"),
    };
    let class = match v["class"].as_str().unwrap() {
        "Iir" => PruningFilterClass::Iir,
        "Fir" => PruningFilterClass::Fir,
        "Hybrid" => PruningFilterClass::Hybrid,
        other => panic!("{CASE}: unknown class {other}"),
    };
    let variant = match v["variant"].as_str().unwrap() {
        "Unrouted" => PruningRoutedVariant::Unrouted,
        "Routed" => PruningRoutedVariant::Routed,
        other => panic!("{CASE}: unknown variant {other}"),
    };
    let conditions: Vec<String> = serde_json::from_value(v["conditions"].clone()).unwrap();
    let reason = v["reason"].as_str().unwrap().to_string();
    let outcome = match v["outcome"].as_str().unwrap() {
        "Ran" => MatrixRowOutcome::Ran,
        "Stale" => MatrixRowOutcome::Stale { reason },
        "Skipped" => MatrixRowOutcome::Skipped { reason },
        other => panic!("{CASE}: unknown outcome {other}"),
    };
    PruningMatrixRow {
        mode,
        filter_class: class,
        routed_variant: variant,
        condition_ids: conditions,
        outcome,
    }
}

#[test]
fn wolfram_rw07_scorecard_pruning() {
    let ref_json = require_reference(CASE, "rw07_scorecard_pruning.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let rows: Vec<PruningMatrixRow> = ref_json["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(parse_row)
        .collect();
    assert_eq!(rows.len(), 4, "{CASE}: expected 4 matrix rows");
    let expected_ran: usize = serde_json::from_value(ref_json["expected_ran"].clone()).unwrap();
    let expected_stale: Vec<String> =
        serde_json::from_value(ref_json["expected_stale"].clone()).unwrap();
    let expected_skipped: Vec<String> =
        serde_json::from_value(ref_json["expected_skipped"].clone()).unwrap();

    let report = audit_pruning_matrix(&rows);
    assert_eq!(
        report.audit_version,
        ref_json["audit_version"].as_str().unwrap(),
        "{CASE}: audit version mismatch"
    );
    assert_eq!(report.ran, expected_ran, "{CASE}: ran count mismatch");
    assert_eq!(report.stale, expected_stale, "{CASE}: stale paths mismatch");
    assert_eq!(
        report.skipped, expected_skipped,
        "{CASE}: skipped paths mismatch"
    );
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: 0.0,
        tolerance: 0.0,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
