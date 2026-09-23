//! Wolfram cross-check: flat scalar-weight loss blend (O01).
//!
//! Oracle: `wolfram/o01_scalar_weights.wls` — independent ERB-rate +
//! log-quadrature + band-weight evaluation from the published
//! Glasberg-Moore definition and the declared 0.7/0.3 blend, plus the
//! dip-only null-suppression mask. Tolerance 1e-9 absolute (dB).

use autoeq_optim::loss::enhanced_weights::{
    FrequencyBandWeights, band_weighted_loss, erb_weighted_loss,
};
use autoeq_optim::loss::{PreparedFlatLoss, flat_loss};
use autoeq_qa::{QaResult, assert_case_id, assert_close_abs, emit_result, provenance};
use autoeq_qa::{assert_all_finite, require_reference};
use ndarray::Array1;
use std::sync::Arc;

const CASE: &str = "o01_scalar_weights";
const CASE_ID: &str = "autoeq-qa.o01-scalar-weights.v1";
const TOL: f64 = 1e-9;

fn vec_f64(value: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(value[key].clone())
        .unwrap_or_else(|_| panic!("{CASE}: golden is missing numeric array `{key}`"))
}

#[test]
fn wolfram_o01_scalar_weights() {
    let ref_json = require_reference(CASE, "o01_scalar_weights.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs = vec_f64(&ref_json, "grid_hz");
    let errors = vec_f64(&ref_json, "errors_db");
    let mask = vec_f64(&ref_json, "null_mask");
    // The oracle grid is the comparison grid: identical grids, no resampling.
    assert_eq!(
        freqs,
        vec![20.0, 100.0, 300.0, 1000.0, 4000.0, 20000.0],
        "{CASE}: grid identity"
    );
    assert_eq!(errors.len(), freqs.len());
    assert_eq!(mask.len(), freqs.len());

    let freq_arr = Array1::from_vec(freqs);
    let err_arr = Array1::from_vec(errors);
    assert_all_finite(freq_arr.as_slice().unwrap(), CASE);
    assert_all_finite(err_arr.as_slice().unwrap(), CASE);

    let num = |key: &str| {
        ref_json[key]
            .as_f64()
            .unwrap_or_else(|| panic!("{CASE}: golden lacks `{key}`"))
    };

    // Component losses against the independent closed forms.
    let rust_erb = erb_weighted_loss(&freq_arr, &err_arr);
    let rust_band = band_weighted_loss(&freq_arr, &err_arr, &FrequencyBandWeights::default());
    let rust_combined = flat_loss(&freq_arr, &err_arr, 20.0, 20000.0);
    assert_close_abs(rust_erb, num("erb_loss"), TOL, "ERB-weighted loss");
    assert_close_abs(rust_band, num("band_loss"), TOL, "band-weighted loss");
    assert_close_abs(rust_combined, num("combined_loss"), TOL, "combined loss");

    // The golden's per-band RMS triple pins the band split (bass/mid/treble
    // with inclusive endpoints); the band blend above already scores it.
    let band_rms = vec_f64(&ref_json, "band_rms");
    assert_eq!(band_rms.len(), 3, "{CASE}: expected 3 band RMS values");
    assert_all_finite(&band_rms, CASE);
    let mut max_err: f64 = [rust_erb - num("erb_loss"), rust_band - num("band_loss")]
        .iter()
        .map(|e| e.abs())
        .fold(0.0, f64::max)
        .max((rust_combined - num("combined_loss")).abs());

    // Cached kernel must agree with the allocating reference.
    let prepared = PreparedFlatLoss::new(&freq_arr, 20.0, 20000.0);
    let cached = prepared.evaluate(&err_arr);
    assert_close_abs(cached, rust_combined, 1e-12, "prepared flat loss");

    // Dip-only null-suppression mask: masked dips score lower, peaks untouched.
    let masked = PreparedFlatLoss::new_with_null_suppression(
        &freq_arr,
        20.0,
        20000.0,
        Arc::new(Array1::from_vec(mask)),
    )
    .evaluate(&err_arr);
    assert_close_abs(masked, num("combined_loss_masked"), TOL, "masked loss");
    assert!(
        masked < rust_combined,
        "{CASE}: masked loss {masked} must be below unmasked {rust_combined}"
    );
    max_err = max_err.max((masked - num("combined_loss_masked")).abs());

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
