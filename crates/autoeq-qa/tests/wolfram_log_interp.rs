//! Wolfram cross-check: log-frequency interpolation (C03).
//!
//! Oracle: `wolfram/log_interp.wls` (piecewise-affine SPL on the Log
//! axis with slope extrapolation outside the knot range). Covers exact
//! knots, interior points, and both out-of-band extrapolation sides.
//! Tolerance 1e-9 dB absolute.

use autoeq_core::Curve;
use autoeq_core::curve_transforms::interpolate_log_space;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_all_finite, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "log_interp";
const CASE_ID: &str = "autoeq-qa.log-interp.v1";
const TOL_DB: f64 = 1e-9;

#[test]
fn wolfram_log_interp() {
    let ref_json = require_reference(CASE, "log_interp.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let knots: Vec<f64> = serde_json::from_value(ref_json["knot_freqs_hz"].clone()).unwrap();
    let spl: Vec<f64> = serde_json::from_value(ref_json["knot_spl_db"].clone()).unwrap();
    let outputs: Vec<f64> = serde_json::from_value(ref_json["output_freqs_hz"].clone()).unwrap();
    let expected: Vec<f64> = serde_json::from_value(ref_json["interp_spl_db"].clone()).unwrap();
    assert_eq!(knots.len(), 5, "{CASE}: expected 5 knots");
    assert_eq!(outputs.len(), 10, "{CASE}: expected 10 output points");
    assert_eq!(
        expected.len(),
        outputs.len(),
        "{CASE}: reference length must match output grid"
    );
    assert_all_finite(&expected, CASE);

    let curve = Curve {
        freq: Array1::from_vec(knots),
        spl: Array1::from_vec(spl),
        ..Default::default()
    };
    let grid = Array1::from_vec(outputs.clone());
    let rust = interpolate_log_space(&grid, &curve);
    assert_eq!(rust.spl.len(), outputs.len());

    let mut max_err = 0.0f64;
    for ((f, want), got) in outputs.iter().zip(expected.iter()).zip(rust.spl.iter()) {
        let err = (got - want).abs();
        assert!(
            err <= TOL_DB,
            "SPL({f} Hz): rust={got:.12e} expected={want:.12e} abs_err={err:.3e} tol={TOL_DB:.1e}"
        );
        max_err = max_err.max(err);
    }
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
