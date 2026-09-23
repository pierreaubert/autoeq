//! Wolfram cross-check: backend delay quantization with shared
//! parallel-stage padding (EX02).
//!
//! Oracle: `wolfram/ex02_delay_padding_matrix.wls` (independent
//! evaluation of the stage-padding rule from its documented
//! definition; never calls the Rust FIR helpers). The test calls
//! `roomeq_engine::fir::gd_delay_padding_samples` for three delay
//! groups, realizes an integer delay with `realize_gd_fir_delay`
//! (exact sample shift), and realizes a fractional unit impulse
//! (DC-normalized kernel of the documented length). Tolerance: exact
//! integers for padding/indices (I), 1e-12 relative on realized taps
//! and effective delay (A).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_engine::fir::{gd_delay_padding_samples, realize_gd_fir_delay};

const CASE: &str = "ex02_delay_padding_matrix";
const CASE_ID: &str = "autoeq-qa.ex02-delay-padding-matrix.v1";
const TOL: f64 = 1e-12;

#[test]
fn wolfram_ex02_delay_padding_matrix() {
    let ref_json = require_reference(CASE, "ex02_delay_padding_matrix.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let groups: Vec<serde_json::Value> =
        serde_json::from_value(ref_json["groups"].clone()).unwrap();
    assert_eq!(groups.len(), 3, "{CASE}: expected 3 delay groups");

    // Padding maximization per parallel stage, including the
    // zero-delay reference branch and a negative delay.
    for group in &groups {
        let delays: Vec<f64> = serde_json::from_value(group["delays_ms"].clone()).unwrap();
        let expected_pad: usize = serde_json::from_value(group["pad_samples"].clone()).unwrap();
        let rust_pad = gd_delay_padding_samples(&delays, rate);
        assert_eq!(
            rust_pad, expected_pad,
            "{CASE}: padding for {delays:?} must be {expected_pad} samples"
        );
    }

    // Integer delays shift taps exactly, with no fractional kernel.
    let int = &ref_json["integer_delay"];
    let int_delay_ms: f64 = serde_json::from_value(int["delay_ms"].clone()).unwrap();
    let int_pad: usize = serde_json::from_value(int["pad_samples"].clone()).unwrap();
    let int_taps: Vec<f64> = serde_json::from_value(int["input_taps"].clone()).unwrap();
    let want_taps: Vec<f64> = serde_json::from_value(int["realized_taps"].clone()).unwrap();
    let realized = realize_gd_fir_delay(&int_taps, int_delay_ms, rate, int_pad).unwrap();
    assert_eq!(
        realized.coefficients.len(),
        want_taps.len(),
        "{CASE}: integer realization length"
    );
    let mut max_err = 0.0f64;
    for (index, (got, want)) in realized.coefficients.iter().zip(&want_taps).enumerate() {
        let err = (got - want).abs() / want.abs().max(1e-300);
        assert!(
            err <= TOL,
            "{CASE}: integer tap {index}: {got:.17e} vs {want:.17e} err={err:.3e}"
        );
        max_err = max_err.max(err);
    }
    let want_ms: f64 = serde_json::from_value(int["effective_delay_ms"].clone()).unwrap();
    let ms_err = (realized.effective_delay_ms - want_ms).abs() / want_ms.abs();
    assert!(
        ms_err <= TOL,
        "{CASE}: effective delay {} vs {want_ms} err={ms_err:.3e}",
        realized.effective_delay_ms
    );
    max_err = max_err.max(ms_err);
    assert_eq!(realized.common_padding_samples, int_pad);

    // Fractional unit impulses realize as unity-DC kernels of the
    // documented causal length on shared padding.
    let frac = &ref_json["fractional_unit"];
    let frac_delay_ms: f64 = serde_json::from_value(frac["delay_ms"].clone()).unwrap();
    let frac_pad: usize = serde_json::from_value(frac["pad_samples"].clone()).unwrap();
    let frac_len: usize = serde_json::from_value(frac["kernel_length"].clone()).unwrap();
    let frac_shift: f64 = serde_json::from_value(frac["shift_samples"].clone()).unwrap();
    let unit = realize_gd_fir_delay(&[1.0], frac_delay_ms, rate, frac_pad).unwrap();
    assert_eq!(
        unit.coefficients.len(),
        frac_len,
        "{CASE}: fractional kernel length"
    );
    let dc: f64 = unit.coefficients.iter().sum();
    assert!(
        (dc - 1.0).abs() <= TOL,
        "{CASE}: fractional kernel DC {dc:.17e} must be unity"
    );
    let want_frac_ms = frac_shift * 1000.0 / rate;
    let frac_ms_err = (unit.effective_delay_ms - want_frac_ms).abs() / want_frac_ms.abs();
    assert!(
        frac_ms_err <= TOL,
        "{CASE}: fractional effective delay err={frac_ms_err:.3e}"
    );
    max_err = max_err.max((dc - 1.0).abs()).max(frac_ms_err);

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
