//! Wolfram cross-check: speaker workflow composition (AW01).
//!
//! Oracle: `wolfram/aw01_workflow_correction.wls` (independent RBJ SOS
//! sums + published flat/harman target definitions). Exercises the
//! public in-memory workflow path
//! (`autoeq_workflow::build_target_curve_by_name`) together with the
//! realized response (`compute_peq_response_from_x`, the same kernel
//! the plot/workflow visualization paths use): target construction,
//! grid alignment, corrected response and residual RMS. Tolerance
//! 1e-6 dB absolute (class A; cross-libm budget).

use autoeq_core::{Curve, PeqModel};
use autoeq_plot::x2peq::compute_peq_response_from_x;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_all_finite, assert_case_id, emit_result, provenance};
use autoeq_workflow::build_target_curve_by_name;
use ndarray::Array1;

const CASE: &str = "aw01_workflow_correction";
const CASE_ID: &str = "autoeq-qa.aw01-workflow-correction.v1";
const TOL_DB: f64 = 1e-6;
const SR: f64 = 48000.0;

#[test]
fn wolfram_aw01_workflow_correction() {
    let ref_json = require_reference(CASE, "aw01_workflow_correction.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let input: Vec<f64> = serde_json::from_value(ref_json["input_spl_db"].clone()).unwrap();
    let flat: Vec<f64> = serde_json::from_value(ref_json["flat_target_db"].clone()).unwrap();
    let harman: Vec<f64> = serde_json::from_value(ref_json["harman_target_db"].clone()).unwrap();
    let eq: Vec<f64> = serde_json::from_value(ref_json["eq_response_db"].clone()).unwrap();
    let corrected: Vec<f64> = serde_json::from_value(ref_json["corrected_spl_db"].clone()).unwrap();
    let rms: f64 = serde_json::from_value(ref_json["residual_rms_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 grid points");
    for (name, v) in [
        ("input", &input),
        ("flat", &flat),
        ("harman", &harman),
        ("eq", &eq),
    ] {
        assert_eq!(
            v.len(),
            freqs.len(),
            "{CASE}: {name} length must match grid"
        );
    }
    assert_all_finite(&eq, CASE);
    assert_all_finite(&corrected, CASE);
    assert!(rms.is_finite(), "{CASE}: non-finite reference RMS");

    // Same grid object flows into target construction and response, so no
    // resampling/zip of unequal grids is possible here; assert it anyway.
    let grid = Array1::from_vec(freqs.clone());
    let input_curve = Curve {
        freq: grid.clone(),
        spl: Array1::from_vec(input.clone()),
        ..Default::default()
    };

    let flat_rust = build_target_curve_by_name("flat", &grid, &input_curve).unwrap();
    assert_eq!(flat_rust.freq.len(), grid.len());
    assert_eq!(flat_rust.spl.len(), grid.len());
    for (i, (got, want)) in flat_rust.spl.iter().zip(flat.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(err <= TOL_DB, "{CASE}: flat target [{i}]: err={err:.3e}");
    }

    let harman_rust = build_target_curve_by_name("harman", &grid, &input_curve).unwrap();
    let mut max_err = 0.0f64;
    for (i, (got, want)) in harman_rust.spl.iter().zip(harman.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL_DB,
            "{CASE}: harman target [{i}]: rust={got:.12e} expected={want:.12e} err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // Fixed candidate: one Peak at 1 kHz, Q 1, -6 dB (Pk layout).
    let x = vec![1000.0f64.log10(), 1.0, -6.0];
    let eq_rust = compute_peq_response_from_x(&grid, &x, SR, PeqModel::Pk);
    assert_eq!(eq_rust.len(), grid.len());
    for (i, (got, want)) in eq_rust.iter().zip(eq.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL_DB,
            "{CASE}: eq [{i}]: rust={got:.12e} expected={want:.12e} err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    let corrected_rust: Vec<f64> = input
        .iter()
        .zip(eq_rust.iter())
        .map(|(a, b)| a + b)
        .collect();
    for (i, ((got, want), f)) in corrected_rust
        .iter()
        .zip(corrected.iter())
        .zip(freqs.iter())
        .enumerate()
    {
        let err = (got - want).abs();
        assert!(
            err <= TOL_DB,
            "{CASE}: corrected({f} Hz) [{i}]: rust={got:.12e} expected={want:.12e} err={err:.3e}"
        );
        max_err = max_err.max(err);
    }
    let rms_rust =
        (corrected_rust.iter().map(|v| v * v).sum::<f64>() / corrected_rust.len() as f64).sqrt();
    let rms_err = (rms_rust - rms).abs();
    assert!(
        rms_err <= TOL_DB,
        "{CASE}: residual RMS: rust={rms_rust:.12e} expected={rms:.12e} err={rms_err:.3e}"
    );
    max_err = max_err.max(rms_err);

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_err,
        tolerance: TOL_DB,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
