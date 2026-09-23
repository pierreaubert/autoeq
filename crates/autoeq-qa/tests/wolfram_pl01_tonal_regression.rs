//! Wolfram cross-check: tonal-balance regression (PL01).
//!
//! Oracle: `wolfram/pl01_tonal_regression.wls` (closed-form normal
//! equations for SPL against log2(f)). Compares
//! `calculate_tonal_balance` (slope dB/octave, intercept at 1 kHz)
//! and `generate_regression_line` on an exact-line fixture.
//! Tolerance 1e-9 absolute (class A: direct f64 algebra).

use autoeq_plot::{calculate_tonal_balance, generate_regression_line};
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "pl01_tonal_regression";
const CASE_ID: &str = "autoeq-qa.pl01-tonal-regression.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_pl01_tonal_regression() {
    let ref_json = require_reference(CASE, "pl01_tonal_regression.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let spl: Vec<f64> = serde_json::from_value(ref_json["spl_db"].clone()).unwrap();
    let fmin: f64 = serde_json::from_value(ref_json["fmin_hz"].clone()).unwrap();
    let fmax: f64 = serde_json::from_value(ref_json["fmax_hz"].clone()).unwrap();
    let slope: f64 = serde_json::from_value(ref_json["slope_db_per_oct"].clone()).unwrap();
    let icept: f64 = serde_json::from_value(ref_json["intercept_db_at_1k"].clone()).unwrap();
    let line_freqs: Vec<f64> = serde_json::from_value(ref_json["line_freqs_hz"].clone()).unwrap();
    let line: Vec<f64> = serde_json::from_value(ref_json["line_spl_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 6, "{CASE}: expected 6 fixture points");
    assert_eq!(spl.len(), freqs.len());
    assert_eq!(line.len(), line_freqs.len());
    for v in [slope, icept] {
        assert!(v.is_finite(), "{CASE}: non-finite reference scalar");
    }

    let freq_arr = Array1::from_vec(freqs);
    let spl_arr = Array1::from_vec(spl);
    let (rust_slope, rust_icept) =
        calculate_tonal_balance(&freq_arr, &spl_arr, fmin, fmax).expect("fixture must fit");
    let slope_err = (rust_slope - slope).abs();
    let icept_err = (rust_icept - icept).abs();
    assert!(
        slope_err <= TOL,
        "{CASE}: slope: rust={rust_slope:.12e} expected={slope:.12e} err={slope_err:.3e}"
    );
    assert!(
        icept_err <= TOL,
        "{CASE}: intercept: rust={rust_icept:.12e} expected={icept:.12e} err={icept_err:.3e}"
    );

    let line_grid = Array1::from_vec(line_freqs);
    let rust_line = generate_regression_line(rust_slope, rust_icept, &line_grid);
    assert_eq!(rust_line.len(), line.len());
    let mut max_err = slope_err.max(icept_err);
    for (i, (got, want)) in rust_line.iter().zip(line.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: line [{i}]: rust={got:.12e} expected={want:.12e} err={err:.3e}"
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
