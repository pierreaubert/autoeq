//! Wolfram cross-check: gain/delay/polarity branch sum + level delta (RE10).
//!
//! Oracle: `wolfram/re10_branch_sum_levels.wls` (closed-form complex
//! pressure sum with no crossover filters: inverted 84 dB sub branch
//! with a 1 ms bulk delay, 80 dB main branch with the missing-phase 0
//! deg convention; the -4 dB sub adjustment is the declared difference
//! of the two reference-band means). Tolerance 1e-9 complex-relative
//! (1e-8 absolute on combined dB); the delta definition is checked
//! exactly. No target-shape claim beyond the band means is made.

use autoeq_optim::loss::{
    CrossoverType, DriverMeasurement, DriversLossData, compute_drivers_combined_response_complex,
};
use autoeq_qa::{
    QaResult, assert_case_id, complex_rel_error, emit_result, provenance, require_reference,
};
use ndarray::Array1;
use num_complex::Complex64;

const CASE: &str = "re10_branch_sum_levels";
const CASE_ID: &str = "autoeq-qa.re10-branch-sum-levels.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_re10_branch_sum_levels() {
    let ref_json = require_reference(CASE, "re10_branch_sum_levels.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(ref_json["response_re_im"].clone()).unwrap();
    let expected_db: Vec<f64> = serde_json::from_value(ref_json["combined_db"].clone()).unwrap();
    let sub_mean: f64 = serde_json::from_value(ref_json["sub_band_mean_db"].clone()).unwrap();
    let main_mean: f64 = serde_json::from_value(ref_json["main_band_mean_db"].clone()).unwrap();
    let delta: f64 = serde_json::from_value(ref_json["delta_sub_db"].clone()).unwrap();
    let gains: Vec<f64> = serde_json::from_value(ref_json["gains_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 100, "{CASE}: expected 100 grid points");
    assert_eq!(pairs.len(), freqs.len());
    assert_eq!(expected_db.len(), freqs.len());
    assert!(
        freqs.iter().all(|v| v.is_finite()),
        "{CASE}: non-finite grid"
    );

    // The two-band adjustment is defined by the reference-band means.
    assert!(
        (delta - (main_mean - sub_mean)).abs() == 0.0,
        "{CASE}: delta {delta} must equal main - sub band means ({main_mean} - {sub_mean})"
    );
    assert_eq!(
        gains,
        vec![delta, 0.0],
        "{CASE}: applied gains carry the delta"
    );

    // Unequal passbands, inverted sub (+180 deg), missing main phase (0 deg).
    let sub = DriverMeasurement {
        freq: Array1::from_vec(vec![20.0, 300.0]),
        spl: Array1::from_vec(vec![84.0, 84.0]),
        phase: Some(Array1::from_vec(vec![180.0, 180.0])),
    };
    let main = DriverMeasurement {
        freq: Array1::from_vec(vec![20.0, 20000.0]),
        spl: Array1::from_vec(vec![80.0, 80.0]),
        phase: None,
    };
    let data = DriversLossData::new_ordered(vec![sub, main], CrossoverType::None);

    // Explicit grid alignment before any elementwise comparison.
    assert_eq!(data.freq_grid.len(), freqs.len());
    for (i, (&got, &want)) in data.freq_grid.iter().zip(freqs.iter()).enumerate() {
        let err = ((got - want) / want).abs();
        assert!(
            err <= 1e-12,
            "{CASE}: grid[{i}] mismatch: rust={got:.12e} oracle={want:.12e}"
        );
    }

    let rust = compute_drivers_combined_response_complex(
        &data,
        &[delta, 0.0],
        &[],
        Some(&[1.0, 0.0]),
        48000.0,
    );
    assert_eq!(rust.len(), freqs.len());

    let mut max_rel = 0.0f64;
    let mut max_abs_db = 0.0f64;
    for ((pair, want_db), value) in pairs.iter().zip(expected_db.iter()).zip(rust.iter()) {
        let expected = Complex64::new(pair[0], pair[1]);
        assert!(expected.norm().is_finite(), "{CASE}: non-finite reference");
        let err = complex_rel_error(*value, expected);
        assert!(
            err <= TOL,
            "{CASE}: complex mismatch rust={value:?} expected={expected:?} rel_err={err:.3e}"
        );
        max_rel = max_rel.max(err);
        let rust_db = 20.0 * value.norm().max(1e-12).log10();
        assert!(rust_db.is_finite(), "{CASE}: non-finite rust dB");
        max_abs_db = max_abs_db.max((rust_db - want_db).abs());
    }
    assert!(
        max_abs_db <= 1e-8,
        "{CASE}: combined dB abs err {max_abs_db:.3e} exceeds 1e-8"
    );
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_rel,
        max_abs_error: max_abs_db,
        tolerance: TOL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
