//! Wolfram cross-check: flat-magnitude minimum phase is zero (C05).
//!
//! Oracle: `wolfram/c05_minphase_flat.wls` (analytic claim: the Hilbert
//! transform of a constant log-magnitude is zero, so minimum phase is
//! identically 0 deg). Covers magnitude/min-phase agreement ONLY — not
//! excess-phase, time-domain, or spatial claims. The 0.5 deg bound is the
//! N-class arithmetic budget of the finite log-aware Hilbert pipeline.

use autoeq_core::phase_utils::reconstruct_minimum_phase;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "c05_minphase_flat";
const CASE_ID: &str = "autoeq-qa.c05-minphase-flat.v1";
const TOL: f64 = 0.5;

#[test]
fn wolfram_c05_minphase_flat() {
    let ref_json = require_reference(CASE, "c05_minphase_flat.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let spl: Vec<f64> = serde_json::from_value(ref_json["spl_db"].clone()).unwrap();
    let zeros: Vec<f64> = serde_json::from_value(ref_json["min_phase_deg"].clone()).unwrap();
    let bound: f64 = serde_json::from_value(ref_json["arithmetic_bound_deg"].clone()).unwrap();
    assert_eq!(freqs.len(), 16, "{CASE}: expected 16 grid points");
    assert_eq!(spl.len(), freqs.len(), "{CASE}: spl/grid length zip");
    assert_eq!(zeros.len(), freqs.len(), "{CASE}: phase/grid length zip");
    assert!(
        zeros.iter().all(|v| *v == 0.0),
        "{CASE}: analytic claim is zero"
    );
    assert_eq!(bound, TOL, "{CASE}: bound must match the manifest");

    let rust = reconstruct_minimum_phase(&Array1::from_vec(freqs), &Array1::from_vec(spl));
    assert_eq!(rust.len(), zeros.len());
    let mut max_abs = 0.0f64;
    for (i, (got, want)) in rust.iter().zip(zeros.iter()).enumerate() {
        assert!(got.is_finite(), "{CASE}: non-finite min phase at {i}");
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "min_phase[{i}]: rust={got:.6} deg expected={want} abs_err={err:.3e} tol={TOL}"
        );
        max_abs = max_abs.max(err);
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_abs,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
