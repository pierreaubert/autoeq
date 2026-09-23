//! Wolfram cross-check: audibility deadband thresholds (O04).
//!
//! Oracle: `wolfram/o04_deadband.wls` — independent piecewise threshold with
//! log-frequency mid/treble join, sub-Schroeder bypass, and soft-threshold
//! residual mapping (residuals exactly at +-threshold map to zero).
//! Compared through the public `ObjectiveContext` deadband/smoothing path.
//! Tolerance 1e-9 absolute (dB). Error smoothing gets exact
//! definition-derived checks (disabled path is identity; a constant error
//! vector is preserved by the log-frequency mean).

use autoeq_optim::PeqModel;
use autoeq_optim::optim::loss::ObjectiveContext;
use autoeq_optim::roomeq::AudibilityDeadbandConfig;
use autoeq_qa::{QaResult, assert_case_id, assert_close_abs, emit_result, provenance};
use autoeq_qa::{assert_all_finite, require_reference};
use ndarray::Array1;

const CASE: &str = "o04_deadband";
const CASE_ID: &str = "autoeq-qa.o04-deadband.v1";
const TOL: f64 = 1e-9;

fn vec_f64(value: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(value[key].clone())
        .unwrap_or_else(|_| panic!("{CASE}: golden is missing numeric array `{key}`"))
}

#[test]
fn wolfram_o04_deadband() {
    let ref_json = require_reference(CASE, "o04_deadband.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs = vec_f64(&ref_json, "grid_hz");
    assert_eq!(
        freqs,
        vec![50.0, 250.0, 300.0, 1000.0, 2000.0, 8000.0],
        "{CASE}: grid identity"
    );
    let errors = vec_f64(&ref_json, "errors_db");
    let thresholds = vec_f64(&ref_json, "thresholds_db");
    let expected = vec_f64(&ref_json, "outputs_db");
    assert_eq!(errors.len(), freqs.len());
    assert_all_finite(&thresholds, CASE);

    let cfg = AudibilityDeadbandConfig {
        enabled: true,
        bass_db: 0.25,
        mid_db: 0.75,
        treble_db: 1.0,
        bass_mid_hz: 250.0,
        mid_treble_hz: 2000.0,
        disable_below_schroeder: true,
        schroeder_hz: 300.0,
    };
    let golden_cfg = &ref_json["config"];
    assert_eq!(golden_cfg["bass_db"], 0.25);
    assert_eq!(golden_cfg["schroeder_hz"], 300.0);

    let freq_arr = Array1::from_vec(freqs);
    let err_arr = Array1::from_vec(errors);
    let ctx = ObjectiveContext {
        freqs: &freq_arr,
        target: &err_arr,
        deviation: &err_arr,
        srate: 48000.0,
        peq_model: PeqModel::Pk,
        min_freq: 20.0,
        max_freq: 20000.0,
        smooth: false,
        smooth_n: 3,
        audibility_deadband: Some(&cfg),
        smoothness_penalty: None,
    };
    let rust = ctx.apply_deadband(&err_arr);

    let mut max_err = 0.0f64;
    for (i, (actual, want)) in rust.iter().zip(expected.iter()).enumerate() {
        assert_close_abs(*actual, *want, TOL, &format!("deadband output [{i}]"));
        max_err = max_err.max((actual - want).abs());
    }

    // Exact-boundary residuals (300 Hz at +T, 1000 Hz at -T) must vanish.
    assert_close_abs(rust[2], 0.0, TOL, "exact +threshold maps to zero");
    assert_close_abs(rust[3], 0.0, TOL, "exact -threshold maps to zero");
    // Sub-Schroeder points pass through untouched even below threshold.
    assert_close_abs(rust[0], 0.1, TOL, "sub-Schroeder passthrough");
    assert_close_abs(
        rust[1],
        0.25,
        TOL,
        "bass-mid join below Schroeder passes through",
    );

    // Disabled config is the identity, exactly.
    let off = AudibilityDeadbandConfig {
        enabled: false,
        ..cfg
    };
    let ctx_off = ObjectiveContext {
        audibility_deadband: Some(&off),
        ..ctx
    };
    let identity = ctx_off.apply_deadband(&err_arr);
    for (actual, want) in identity.iter().zip(err_arr.iter()) {
        assert_close_abs(*actual, *want, 0.0, "disabled deadband identity");
    }

    // Error smoothing: disabled path is identity; the 1/N-octave
    // log-frequency mean preserves a constant error vector.
    let constant = Array1::from_elem(freq_arr.len(), 1.5);
    let noop = ctx.smooth_error(constant.clone());
    for (actual, want) in noop.iter().zip(constant.iter()) {
        assert_close_abs(*actual, *want, 0.0, "smoothing disabled identity");
    }
    let ctx_smooth = ObjectiveContext {
        smooth: true,
        ..ctx
    };
    let smoothed = ctx_smooth.smooth_error(constant.clone());
    for actual in smoothed.iter() {
        assert_close_abs(*actual, 1.5, 1e-9, "smoothing preserves constants");
        max_err = max_err.max((actual - 1.5).abs());
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
