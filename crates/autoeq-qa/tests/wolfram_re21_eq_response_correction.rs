//! Wolfram cross-check: assembled graph correction curve (RE21).
//!
//! Oracle: `wolfram/re21_eq_response_correction.wls` (independent
//! final-minus-initial subtraction on the shared grid; never calls the
//! Rust graph code). The grid must pass through unchanged: aligned
//! grids only, never zipped across unequal grids. Tolerance 1e-12
//! absolute dB (exact f64 subtraction).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_engine::output::compute_eq_response;
use roomeq_model::CurveData;

const CASE: &str = "re21_eq_response_correction";
const CASE_ID: &str = "autoeq-qa.re21-eq-response-correction.v1";
const TOL: f64 = 1e-12;

#[test]
fn wolfram_re21_eq_response_correction() {
    let ref_json = require_reference(CASE, "re21_eq_response_correction.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let initial: Vec<f64> = serde_json::from_value(ref_json["initial_spl_db"].clone()).unwrap();
    let final_: Vec<f64> = serde_json::from_value(ref_json["final_spl_db"].clone()).unwrap();
    let expected: Vec<f64> = serde_json::from_value(ref_json["correction_spl_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 7, "{CASE}: expected 7 frequency points");
    assert_eq!(initial.len(), freqs.len());
    assert_eq!(final_.len(), freqs.len());
    assert_eq!(
        expected.len(),
        freqs.len(),
        "{CASE}: correction length must match grid length"
    );

    let initial_curve = CurveData {
        freq: freqs.clone(),
        spl: initial,
        ..CurveData::default()
    };
    let final_curve = CurveData {
        freq: freqs.clone(),
        spl: final_,
        ..CurveData::default()
    };
    let rust = compute_eq_response(&initial_curve, &final_curve);
    assert_eq!(
        rust.freq, freqs,
        "{CASE}: correction grid must pass through unchanged"
    );
    assert_eq!(rust.spl.len(), freqs.len());

    let mut max_err = 0.0f64;
    for ((f, want), got) in freqs.iter().zip(expected.iter()).zip(rust.spl.iter()) {
        assert!(want.is_finite(), "{CASE}: non-finite reference at {f} Hz");
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "correction({f} Hz): rust={got:.12e} expected={want:.12e} abs_err={err:.3e} tol={TOL:.1e}"
        );
        max_err = max_err.max(err);
    }
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
