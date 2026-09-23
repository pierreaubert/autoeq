//! Wolfram cross-check: integer FIR delay realization vs direct DTFT (RE06).
//!
//! Oracle: `wolfram/re06_fir_integer_delay.wls` (independent zero-pad
//! construction and direct DTFT sum; never calls the Rust delay code).
//! The comparison checks the stored tap vector exactly (length and
//! placement catch truncation, wraparound, and double-delay defects)
//! and the delayed transfer at 1e-9 complex-relative tolerance.

use autoeq_core::response::compute_fir_complex_response;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, complex_rel_error, emit_result, provenance};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_engine::fir::{gd_delay_padding_samples, realize_gd_fir_delay};

const CASE: &str = "re06_fir_integer_delay";
const CASE_ID: &str = "autoeq-qa.re06-fir-integer-delay.v1";
const TOL: f64 = 1e-9;
const SAMPLE_RATE: f64 = 48000.0;

#[test]
fn wolfram_re06_fir_integer_delay() {
    let ref_json = require_reference(CASE, "re06_fir_integer_delay.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let input_taps: Vec<f64> = serde_json::from_value(ref_json["input_taps"].clone()).unwrap();
    let stored: Vec<f64> = serde_json::from_value(ref_json["stored_taps"].clone()).unwrap();
    let shift: usize = serde_json::from_value(ref_json["shift_samples"].clone()).unwrap();
    let delay_ms: f64 = serde_json::from_value(ref_json["delay_ms"].clone()).unwrap();
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();
    assert_eq!(input_taps.len(), 3, "{CASE}: expected 3 input taps");
    assert_eq!(freqs.len(), 6, "{CASE}: expected 6 frequency points");
    assert_eq!(
        pairs.len(),
        freqs.len(),
        "{CASE}: response length must match grid length"
    );

    // Integer 1 ms shift at 48 kHz needs no fractional kernel support.
    let padding = gd_delay_padding_samples(&[delay_ms], SAMPLE_RATE);
    assert_eq!(padding, 0, "{CASE}: integer shift must need no padding");
    let realized = realize_gd_fir_delay(&input_taps, delay_ms, SAMPLE_RATE, padding).unwrap();
    assert_eq!(
        realized.common_padding_samples, 0,
        "{CASE}: padding metadata must agree"
    );
    assert!(
        (realized.effective_delay_ms - delay_ms).abs() <= 1e-12,
        "{CASE}: effective delay must equal the requested delay exactly once"
    );
    assert_eq!(
        realized.coefficients.len(),
        input_taps.len() + shift,
        "{CASE}: stored length must grow by exactly the shift (no truncation)"
    );
    assert_eq!(
        realized.coefficients, stored,
        "{CASE}: stored taps must equal the zero-padded reference exactly"
    );

    let grid = Array1::from_vec(freqs.clone());
    let rust = compute_fir_complex_response(&realized.coefficients, &grid, SAMPLE_RATE);
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
