//! Wolfram cross-check: ingestion adapters (M04).
//!
//! Oracle: `wolfram/m04_ingest_adapters.wls` (independent log-frequency
//! interpolation with geometric-midpoint probes, branch-cut phase
//! handling, positive-only clamping, and the normalize-and-interpolate
//! display shift evaluated through the public CSV/API/record interfaces'
//! shared numeric kernels). Tolerance 1e-9 absolute in dB / degrees
//! (catalogue class A/I).

use autoeq_measurements::{
    Curve, clamp_positive_only, interpolate_log_space, interpolate_response,
    normalize_and_interpolate_response,
};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use ndarray::Array1;

const CASE: &str = "m04_ingest_adapters";
const CASE_ID: &str = "autoeq-qa.m04-ingest-adapters.v1";
const TOL: f64 = 1e-9;

fn vec_f64(v: &serde_json::Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}

#[test]
fn wolfram_m04_ingest_adapters() {
    let ref_json = require_reference(CASE, "m04_ingest_adapters.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let mut max_err = 0.0f64;

    // --- (a) log-frequency SPL interpolation through the public kernel. ---
    let input = Curve {
        freq: Array1::from_vec(vec_f64(&ref_json["input_freqs_hz"])),
        spl: Array1::from_vec(vec_f64(&ref_json["input_spl_db"])),
        phase: None,
        ..Default::default()
    };
    let targets = Array1::from_vec(vec_f64(&ref_json["interp_freqs_hz"]));
    let spl_ref = vec_f64(&ref_json["interp_spl_db"]);
    let got = interpolate_log_space(&targets, &input);
    assert_eq!(
        got.spl.len(),
        targets.len(),
        "{CASE}: output length must match target grid"
    );
    for (i, (g, w)) in got.spl.iter().zip(spl_ref.iter()).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= TOL,
            "{CASE}: interp[{i}]: rust={g:.12e} expected={w:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // Phase across the +/-180-degree cut unwraps before interpolation.
    let phased = Curve {
        freq: Array1::from_vec(vec![100.0, 10000.0]),
        spl: Array1::from_vec(vec![0.0, 0.0]),
        phase: Some(Array1::from_vec(vec![170.0, -170.0])),
        ..Default::default()
    };
    let mid = interpolate_log_space(&Array1::from_vec(vec![1000.0]), &phased);
    let mid_phase = mid.phase.as_ref().expect("{CASE}: phase must be preserved")[0];
    let want_phase = ref_json["phase_midpoint_deg"].as_f64().unwrap();
    let errp = (mid_phase - want_phase).abs();
    assert!(
        errp <= TOL,
        "{CASE}: branch-cut midpoint: rust={mid_phase:.12e} expected={want_phase:.12e} abs_err={errp:.3e}"
    );
    max_err = max_err.max(errp);

    // --- (b) positive-only clamp: negatives pass through untouched. ---
    let clamped = clamp_positive_only(
        &Array1::from_vec(vec_f64(&ref_json["clamp_input_db"])),
        ref_json["clamp_max_db"].as_f64().unwrap(),
    );
    let clamp_ref = vec_f64(&ref_json["clamp_output_db"]);
    for (i, (g, w)) in clamped.iter().zip(clamp_ref.iter()).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= TOL,
            "{CASE}: clamp[{i}]: rust={g:.12e} expected={w:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // --- (c) display shift applied exactly once. ---
    let grid = Array1::from_vec(vec_f64(&ref_json["norm_freqs_hz"]));
    let normalized = normalize_and_interpolate_response(&grid, &input);
    let norm_ref = vec_f64(&ref_json["norm_output_db"]);
    for (i, (g, w)) in normalized.spl.iter().zip(norm_ref.iter()).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= TOL,
            "{CASE}: norm[{i}]: rust={g:.12e} expected={w:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }
    // Shift-once ledger check: normalized + band mean == plain interpolation.
    let plain = interpolate_response(&grid, &input);
    let band_mean = ref_json["norm_band_mean_db"].as_f64().unwrap();
    for (i, (n, p)) in normalized.spl.iter().zip(plain.spl.iter()).enumerate() {
        let err = ((n + band_mean) - p).abs();
        assert!(
            err <= TOL,
            "{CASE}: shift-once[{i}]: norm+mean={:.12e} interp={p:.12e} abs_err={err:.3e}",
            n + band_mean
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
