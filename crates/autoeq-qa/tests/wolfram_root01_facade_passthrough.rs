//! Wolfram cross-check: root facade pass-through (ROOT01).
//!
//! Oracle: `wolfram/root01_facade_passthrough.wls` (independent RBJ
//! cookbook evaluation at a 44100 Hz C01 fixture). The Rust test
//! routes the fixture through the engine root facade
//! (`roomeq_engine::response`) and separately through the direct
//! kernel (`autoeq_core::response`); the two must agree bit-for-bit
//! (numeric args unchanged) and both must reproduce the oracle.
//! Tolerance 1e-12 relative on complex values (class A/I).

use autoeq_core::iir::{Biquad, BiquadFilterType};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;

const CASE: &str = "root01_facade_passthrough";
const CASE_ID: &str = "autoeq-qa.root01-facade-passthrough.v1";
const TOL: f64 = 1e-12;

#[test]
fn wolfram_root01_facade_passthrough() {
    let ref_json = require_reference(CASE, "root01_facade_passthrough.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();
    let sr: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let fc: f64 = serde_json::from_value(ref_json["center_hz"].clone()).unwrap();
    let q: f64 = serde_json::from_value(ref_json["q"].clone()).unwrap();
    let gain: f64 = serde_json::from_value(ref_json["gain_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 frequency points");
    assert_eq!(pairs.len(), freqs.len());

    let filter = Biquad::new(BiquadFilterType::Peak, fc, sr, q, gain);
    let grid = Array1::from_vec(freqs.clone());
    let via_facade = roomeq_engine::response::compute_peq_complex_response(
        std::slice::from_ref(&filter),
        &grid,
        sr,
    );
    let direct = autoeq_core::response::compute_peq_complex_response(
        std::slice::from_ref(&filter),
        &grid,
        sr,
    );
    assert_eq!(via_facade.len(), freqs.len());
    assert_eq!(direct.len(), freqs.len());
    for (a, b) in via_facade.iter().zip(direct.iter()) {
        assert!(
            a == b,
            "{CASE}: facade altered pass-through args: {a:?} vs {b:?}"
        );
    }

    let mut max_err = 0.0f64;
    for ((f, pair), value) in freqs.iter().zip(pairs.iter()).zip(via_facade.iter()) {
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(
            expected.norm().is_finite(),
            "{CASE}: non-finite reference at {f} Hz"
        );
        let err = complex_rel_error(*value, expected);
        assert!(
            err <= TOL,
            "H({f} Hz): rust={value:?} expected={expected:?} rel_err={err:.3e} tol={TOL:.1e}"
        );
        max_err = max_err.max(err);
    }
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        max_abs_error: 0.0,
        tolerance: TOL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
