//! Wolfram cross-check: cumulative pruning against frozen full chain (RE04).
//!
//! Oracle: `wolfram/re04_pruning_cumulative.wls` (independent RBJ cookbook
//! evaluation of a two-biquad chain and every removal subset, plus
//! ERB-weighted with/without dB differences). Complex tolerance 1e-9
//! relative; dB differences 1e-8 absolute.

use autoeq_core::erb_rate_cell_widths;
use autoeq_core::iir::{Biquad, BiquadFilterType};
use autoeq_core::response::compute_peq_complex_response;
use autoeq_qa::{
    QaResult, assert_case_id, complex_rel_error, emit_result, provenance, rel_error,
    require_reference,
};
use ndarray::Array1;
use num_complex::Complex64;

const CASE: &str = "re04_pruning_cumulative";
const CASE_ID: &str = "autoeq-qa.re04-pruning-cumulative.v1";
const TOL_REL: f64 = 1e-9;
const TOL_DB: f64 = 1e-8;

fn pairs(value: &serde_json::Value) -> Vec<Complex64> {
    let raw: Vec<[f64; 2]> = serde_json::from_value(value.clone()).unwrap();
    raw.iter().map(|p| Complex64::new(p[0], p[1])).collect()
}

fn mags_db(values: &[Complex64]) -> Vec<f64> {
    values.iter().map(|v| 20.0 * v.norm().log10()).collect()
}

fn erb_rms_db(diffs: &[f64], grid: &Array1<f64>) -> f64 {
    let weights = erb_rate_cell_widths(grid);
    let total: f64 = weights.iter().sum();
    let sum: f64 = diffs
        .iter()
        .zip(weights.iter())
        .map(|(d, w)| d * d * w)
        .sum();
    (sum / total).sqrt()
}

#[test]
fn wolfram_re04_pruning_cumulative() {
    let ref_json = require_reference(CASE, "re04_pruning_cumulative.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 grid points");
    let grid = Array1::from_vec(freqs.clone());

    let fa = Biquad::new(BiquadFilterType::Peak, 1000.0, 48000.0, 1.0, 6.0);
    let fb = Biquad::new(BiquadFilterType::Peak, 100.0, 48000.0, 2.0, -4.0);
    let rust_f0 = compute_peq_complex_response(&[fa.clone(), fb.clone()], &grid, 48000.0);
    let rust_minus_a = compute_peq_complex_response(std::slice::from_ref(&fb), &grid, 48000.0);
    let rust_minus_b = compute_peq_complex_response(std::slice::from_ref(&fa), &grid, 48000.0);

    let exp_f0 = pairs(&ref_json["f0_re_im"]);
    let exp_minus_a = pairs(&ref_json["minus_a_re_im"]);
    let exp_minus_b = pairs(&ref_json["minus_b_re_im"]);
    assert_eq!(exp_f0.len(), grid.len());
    assert_eq!(exp_minus_a.len(), grid.len());
    assert_eq!(exp_minus_b.len(), grid.len());

    let mut max_rel = 0.0f64;
    for (rust, exp) in [&rust_f0, &rust_minus_a, &rust_minus_b].into_iter().zip([
        &exp_f0,
        &exp_minus_a,
        &exp_minus_b,
    ]) {
        for (value, expected) in rust.iter().zip(exp.iter()) {
            assert!(expected.norm().is_finite(), "{CASE}: non-finite reference");
            max_rel = max_rel.max(complex_rel_error(*value, *expected));
        }
    }
    assert!(
        max_rel <= TOL_REL,
        "{CASE}: complex max_rel_err={max_rel:.3e} tol={TOL_REL:.1e}"
    );

    // With/without-filter dB differences and their ERB-weighted impact.
    let mag_f0 = mags_db(&rust_f0);
    let diff_a: Vec<f64> = mag_f0
        .iter()
        .zip(mags_db(&rust_minus_a).iter())
        .map(|(full, without)| full - without)
        .collect();
    let diff_b: Vec<f64> = mag_f0
        .iter()
        .zip(mags_db(&rust_minus_b).iter())
        .map(|(full, without)| full - without)
        .collect();
    let exp_diff_a: Vec<f64> =
        serde_json::from_value(ref_json["diff_remove_a_db"].clone()).unwrap();
    let exp_diff_b: Vec<f64> =
        serde_json::from_value(ref_json["diff_remove_b_db"].clone()).unwrap();
    let mut max_abs: f64 = 0.0;
    for (got, exp) in diff_a
        .iter()
        .chain(diff_b.iter())
        .zip(exp_diff_a.iter().chain(exp_diff_b.iter()))
    {
        max_abs = max_abs.max((got - exp).abs());
    }
    assert!(
        max_abs <= TOL_DB,
        "{CASE}: dB max_abs_err={max_abs:.3e} tol={TOL_DB:.1e}"
    );

    let exp_erb_a: f64 = serde_json::from_value(ref_json["erb_rms_remove_a_db"].clone()).unwrap();
    let exp_erb_b: f64 = serde_json::from_value(ref_json["erb_rms_remove_b_db"].clone()).unwrap();
    let erb_a = erb_rms_db(&diff_a, &grid);
    let erb_b = erb_rms_db(&diff_b, &grid);
    let erb_err = rel_error(erb_a, exp_erb_a).max(rel_error(erb_b, exp_erb_b));
    assert!(
        erb_err <= TOL_REL,
        "{CASE}: ERB impact rel_err={erb_err:.3e} tol={TOL_REL:.1e}"
    );
    max_rel = max_rel.max(erb_err);

    // Both removals must be audible-scale: otherwise the fixture cannot
    // distinguish cumulative subsets.
    assert!(
        exp_erb_a > 0.1 && exp_erb_b > 0.1,
        "{CASE}: fixture removals too small ({exp_erb_a:.3e}, {exp_erb_b:.3e})"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_rel,
        max_abs_error: max_abs,
        tolerance: TOL_REL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
