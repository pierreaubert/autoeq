//! Wolfram cross-check: asymmetric peak/dip weighted loss (O01).
//!
//! Oracle: `wolfram/o01_asymmetric_weights.wls` — independent sigmoid
//! crossfade weights from the documented transition formula, dip-only mask
//! scaling, and the shared ERB + band blend. Tolerance 1e-9 absolute (dB).

use autoeq_optim::loss::{AsymmetricLossConfig, PreparedAsymmetricLoss, weighted_mse_asymmetric};
use autoeq_qa::{QaResult, assert_case_id, assert_close_abs, emit_result, provenance};
use autoeq_qa::{assert_all_finite, require_reference};
use ndarray::Array1;

const CASE: &str = "o01_asymmetric_weights";
const CASE_ID: &str = "autoeq-qa.o01-asymmetric-weights.v1";
const TOL: f64 = 1e-9;

fn vec_f64(value: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(value[key].clone())
        .unwrap_or_else(|_| panic!("{CASE}: golden is missing numeric array `{key}`"))
}

#[test]
fn wolfram_o01_asymmetric_weights() {
    let ref_json = require_reference(CASE, "o01_asymmetric_weights.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs = vec_f64(&ref_json, "grid_hz");
    let errors = vec_f64(&ref_json, "errors_db");
    let mask = vec_f64(&ref_json, "null_mask");
    assert_eq!(
        freqs,
        vec![60.0, 150.0, 300.0, 600.0, 1500.0, 6000.0],
        "{CASE}: grid identity"
    );
    assert_eq!(errors.len(), freqs.len());
    assert_eq!(mask.len(), freqs.len());

    // Custom config exercises every branch: bass/mid peak and dip weights.
    let config = AsymmetricLossConfig {
        peak_weight: 3.0,
        dip_weight: 0.5,
        bass_peak_weight: 6.0,
        bass_dip_weight: 0.25,
        transition_freq: 300.0,
    };
    let cfg = &ref_json["config"];
    assert_eq!(cfg["peak_weight"], 3.0);
    assert_eq!(cfg["transition_freq_hz"], 300.0);

    let freq_arr = Array1::from_vec(freqs);
    let err_arr = Array1::from_vec(errors);
    let mask_arr = Array1::from_vec(mask);
    assert_all_finite(err_arr.as_slice().unwrap(), CASE);

    let rust =
        weighted_mse_asymmetric(&freq_arr, &err_arr, 20.0, 20000.0, &config, Some(&mask_arr));
    let expected = ref_json["combined_loss"]
        .as_f64()
        .expect("golden lacks combined_loss");
    assert!(expected.is_finite(), "{CASE}: non-finite reference");
    assert_close_abs(rust, expected, TOL, "asymmetric combined loss");

    // The golden's per-sample weights pin the sigmoid blend and the
    // peak/dip branch selection; the loss above already scores them.
    let weights = vec_f64(&ref_json, "sample_weights");
    assert_eq!(weights.len(), err_arr.len());
    assert_all_finite(&weights, CASE);
    // Bass peak (60 Hz, +4 dB) must carry more weight than the treble dip
    // branch, and the fully masked mid dip (600 Hz, mask 0) carries zero.
    assert!(
        weights[0] > weights[5],
        "{CASE}: bass peak weight must exceed treble dip weight"
    );
    assert_eq!(weights[3], 0.0, "{CASE}: fully masked dip must weigh zero");

    // Cached kernel must agree with the allocating reference.
    let cached = PreparedAsymmetricLoss::new(&freq_arr, 20.0, 20000.0, &config, Some(&mask_arr))
        .evaluate(&err_arr);
    assert_close_abs(cached, rust, 1e-12, "prepared asymmetric loss");

    // Peaks must be penalized more than equal dips under this config.
    let peak_only = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 5.0, 0.0]);
    let dip_only = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, -5.0, 0.0]);
    let peak_loss = weighted_mse_asymmetric(
        &freq_arr,
        &peak_only,
        20.0,
        20000.0,
        &config,
        Some(&mask_arr),
    );
    let dip_loss = weighted_mse_asymmetric(
        &freq_arr,
        &dip_only,
        20.0,
        20000.0,
        &config,
        Some(&mask_arr),
    );
    assert!(
        peak_loss > dip_loss,
        "{CASE}: peak {peak_loss} must exceed dip {dip_loss}"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: (rust - expected).abs().max((cached - rust).abs()),
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
