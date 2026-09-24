//! Wolfram cross-check: evidence-intake grid overlap (RW01).
//!
//! Oracle: `wolfram/rw01_intake_overlap.wls` (closed-form support
//! intersection `[max of minima, min of maxima]`). Exercises the real
//! intake path (`supported_overlap`, `validate_intake_grids`): shared
//! support must be found by value, and disjoint supports must refuse
//! intake instead of being silently index-averaged. Overlap endpoints
//! absolute 1e-12 Hz (class A); disjoint refusal is exact (X).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_model::decision_ledger::CaptureKind;
use roomeq_workflow::evidence_intake::{RawCaptureRef, supported_overlap, validate_intake_grids};

const CASE: &str = "rw01_intake_overlap";
const CASE_ID: &str = "autoeq-qa.rw01-intake-overlap.v1";
const TOL: f64 = 1e-12;

fn capture(id: &str, grid: Vec<f64>) -> RawCaptureRef {
    RawCaptureRef {
        measurement_id: id.to_string(),
        source_id: "left".to_string(),
        seat_id: "seat-a".to_string(),
        take_id: "take-0".to_string(),
        capture_kind: CaptureKind::StationaryIr,
        calibration_id: None,
        artifact_hash: None,
        grid_hz: grid,
        validity_mask: None,
    }
}

#[test]
fn wolfram_rw01_intake_overlap() {
    let ref_json = require_reference(CASE, "rw01_intake_overlap.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let grid_a: Vec<f64> = serde_json::from_value(ref_json["grid_a_hz"].clone()).unwrap();
    let grid_b: Vec<f64> = serde_json::from_value(ref_json["grid_b_hz"].clone()).unwrap();
    let grid_c: Vec<f64> = serde_json::from_value(ref_json["grid_c_hz"].clone()).unwrap();
    let expected: [f64; 2] =
        serde_json::from_value(ref_json["expected_overlap_ab_hz"].clone()).unwrap();
    let expected_disjoint: bool =
        serde_json::from_value(ref_json["expected_disjoint_ac"].clone()).unwrap();
    assert_eq!((grid_a.len(), grid_b.len(), grid_c.len()), (8, 5, 2));
    assert!(
        expected_disjoint,
        "{CASE}: oracle must stage a disjoint pair"
    );

    // Same grids flow into overlap and intake validation: no resampling
    // or zip of unequal grids is possible; assert lengths anyway.
    let overlap = supported_overlap(&grid_a, &grid_b)
        .unwrap_or_else(|| panic!("{CASE}: grids A/B must share support"));
    let mut max_err = 0.0f64;
    for (i, (got, want)) in overlap.iter().zip(expected.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: overlap[{i}]: rust={got:.12e} expected={want:.12e} err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    validate_intake_grids(&[
        capture("meas-a", grid_a.clone()),
        capture("meas-b", grid_b.clone()),
    ])
    .unwrap_or_else(|error| panic!("{CASE}: overlapping intake must validate: {error}"));
    assert!(
        supported_overlap(&grid_a, &grid_c).is_none(),
        "{CASE}: grids A/C must have no shared support"
    );
    let disjoint = validate_intake_grids(&[
        capture("meas-a", grid_a.clone()),
        capture("meas-c", grid_c.clone()),
    ]);
    let reason = disjoint.expect_err(&format!("{CASE}: disjoint intake must refuse"));
    assert!(
        reason.contains("disjoint"),
        "{CASE}: refusal must name disjoint support, got: {reason}"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_err,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
