//! Wolfram cross-check: FIR direct-DTFT complex response (C01).
//!
//! Oracle: `wolfram/fir_response.wls` (explicit finite DTFT sum with tap
//! zero at time zero). Compared at exact frequencies — not via the
//! padded-FFT nearest-bin approximation used by some FIR test helpers.
//! Tolerance 1e-9 relative on the complex values.

use autoeq_core::response::compute_fir_complex_response;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;

const CASE: &str = "fir_response";
const CASE_ID: &str = "autoeq-qa.fir-response.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_fir_response() {
    let ref_json = require_reference(CASE, "fir_response.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let taps: Vec<f64> = serde_json::from_value(ref_json["taps"].clone()).unwrap();
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();
    assert_eq!(taps.len(), 5, "{CASE}: expected 5 taps");
    assert_eq!(freqs.len(), 6, "{CASE}: expected 6 frequency points");
    assert_eq!(
        pairs.len(),
        freqs.len(),
        "{CASE}: response length must match grid length"
    );

    let grid = Array1::from_vec(freqs.clone());
    let rust = compute_fir_complex_response(&taps, &grid, 48000.0);
    assert_eq!(rust.len(), freqs.len());

    let mut max_err = 0.0f64;
    for ((f, pair), value) in freqs.iter().zip(pairs.iter()).zip(rust.iter()) {
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
