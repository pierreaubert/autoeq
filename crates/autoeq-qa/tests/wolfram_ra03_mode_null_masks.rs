//! Wolfram cross-check: RA03 mode/null detection + suppression mask.
//!
//! Oracle: `wolfram/ra03_mode_null_masks.wls` — synthetic Lorentzian
//! resonance (60 Hz) and notch (140 Hz) with independently computed
//! median-baseline prominence, interpolated half-power crossings, Q and
//! raised-cosine null mask. The oracle grid is the comparison grid:
//! identical grids, no resampling. Tolerance class A (1e-9 relative on
//! Q/prominence/depth; 1e-12 absolute on the unit mask).

use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, rel_error, require_reference};
use ndarray::Array1;
use roomeq_analysis::impulse_analysis::{
    DecomposedCorrectionConfig, NullDetectionConfig, build_null_suppression_mask,
    detect_narrow_nulls, detect_room_modes,
};

const CASE: &str = "ra03_mode_null_masks";
const CASE_ID: &str = "autoeq-qa.ra03-mode-null-masks.v1";
const TOL_REL: f64 = 1e-9;
const TOL_MASK_ABS: f64 = 1e-12;

fn vec_f64(value: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(value[key].clone())
        .unwrap_or_else(|_| panic!("{CASE}: golden is missing numeric array `{key}`"))
}

#[test]
fn wolfram_ra03_mode_null_masks() {
    let ref_json = require_reference(CASE, "ra03_mode_null_masks.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs = vec_f64(&ref_json, "grid_hz");
    let spl = vec_f64(&ref_json, "spl_db");
    assert_eq!(freqs.len(), 400, "{CASE}: expected 400 grid points");
    assert_eq!(spl.len(), freqs.len(), "{CASE}: grid/SPL length zip");
    assert!(
        freqs.iter().all(|f| f.is_finite()) && spl.iter().all(|s| s.is_finite()),
        "{CASE}: non-finite fixture"
    );

    let freq_arr = Array1::from_vec(freqs);
    let spl_arr = Array1::from_vec(spl);
    let detection = DecomposedCorrectionConfig::default();
    let null_cfg = NullDetectionConfig::default();

    // --- Resonance: exactly one mode at the injected 60 Hz peak. ---
    let modes = detect_room_modes(&freq_arr, &spl_arr, &detection);
    assert_eq!(
        modes.len(),
        1,
        "{CASE}: expected exactly one mode, got {modes:?}"
    );
    let exp_modes: Vec<[f64; 3]> =
        serde_json::from_value(ref_json["modes_f_q_prom"].clone()).unwrap();
    assert_eq!(exp_modes.len(), 1, "{CASE}: oracle must pin one mode");
    let mode = &modes[0];
    let mut max_rel: f64 = 0.0;
    let err_f = (mode.frequency - exp_modes[0][0]).abs();
    assert!(err_f == 0.0, "{CASE}: mode frequency {}", mode.frequency);
    for (actual, expected, what) in [
        (mode.q, exp_modes[0][1], "mode Q"),
        (mode.prominence_db, exp_modes[0][2], "mode prominence"),
    ] {
        let err = rel_error(actual, expected);
        assert!(
            err <= TOL_REL,
            "{what}: rust={actual:.12e} expected={expected:.12e} rel_err={err:.3e}"
        );
        max_rel = max_rel.max(err);
    }

    // --- Notch: exactly one null at the injected 140 Hz dip. ---
    let nulls = detect_narrow_nulls(&freq_arr, &spl_arr, &null_cfg);
    assert_eq!(
        nulls.len(),
        1,
        "{CASE}: expected exactly one null, got {nulls:?}"
    );
    let exp_nulls: Vec<[f64; 3]> =
        serde_json::from_value(ref_json["nulls_f_q_depth"].clone()).unwrap();
    assert_eq!(exp_nulls.len(), 1, "{CASE}: oracle must pin one null");
    let null = &nulls[0];
    assert!(
        (null.frequency - exp_nulls[0][0]).abs() == 0.0,
        "{CASE}: null frequency {}",
        null.frequency
    );
    for (actual, expected, what) in [
        (null.q, exp_nulls[0][1], "null Q"),
        (null.depth_db, exp_nulls[0][2], "null depth"),
    ] {
        let err = rel_error(actual, expected);
        assert!(
            err <= TOL_REL,
            "{what}: rust={actual:.12e} expected={expected:.12e} rel_err={err:.3e}"
        );
        max_rel = max_rel.max(err);
    }

    // --- Mask: raised-cosine suppression, C0-continuous, full depth. ---
    let mask = build_null_suppression_mask(&freq_arr, &nulls);
    let exp_mask = vec_f64(&ref_json, "null_mask");
    assert_eq!(mask.len(), exp_mask.len(), "{CASE}: mask length zip");
    let mut max_abs: f64 = 0.0;
    for (i, (&actual, &expected)) in mask.iter().zip(exp_mask.iter()).enumerate() {
        let err = (actual - expected).abs();
        assert!(
            err <= TOL_MASK_ABS,
            "mask[{i}] @ {} Hz: rust={actual:.12e} expected={expected:.12e} abs_err={err:.3e}",
            freq_arr[i]
        );
        max_abs = max_abs.max(err);
    }
    assert!(
        (mask.iter().cloned().fold(1.0f64, f64::min) - 0.0).abs() <= TOL_MASK_ABS,
        "{CASE}: mask must reach full suppression at the nadir"
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
