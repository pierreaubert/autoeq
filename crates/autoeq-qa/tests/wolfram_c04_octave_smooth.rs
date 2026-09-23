//! Wolfram cross-check: 1/1-octave log-frequency smoothing (C04).
//!
//! Oracle: `wolfram/c04_octave_smooth.wls` (direct per-window integral of
//! the piecewise-linear log-frequency interpolant; independent of the Rust
//! prefix-integral path). Tolerance 1e-9 absolute dB.

use autoeq_core::Curve;
use autoeq_core::curve_transforms::smooth_one_over_n_octave;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "c04_octave_smooth";
const CASE_ID: &str = "autoeq-qa.c04-octave-smooth.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_c04_octave_smooth() {
    let ref_json = require_reference(CASE, "c04_octave_smooth.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let knots: Vec<f64> = serde_json::from_value(ref_json["knot_freqs_hz"].clone()).unwrap();
    let values: Vec<f64> = serde_json::from_value(ref_json["knot_spl_db"].clone()).unwrap();
    let expected: Vec<f64> = serde_json::from_value(ref_json["smoothed_spl_db"].clone()).unwrap();
    assert_eq!(knots.len(), 9, "{CASE}: expected 9 knots");
    assert_eq!(values.len(), knots.len(), "{CASE}: knot/value length zip");
    assert_eq!(
        expected.len(),
        knots.len(),
        "{CASE}: output/grid length zip"
    );
    for v in expected.iter() {
        assert!(v.is_finite(), "{CASE}: non-finite reference");
    }
    let bpo_float: f64 = serde_json::from_value(ref_json["bands_per_octave"].clone()).unwrap();
    assert_eq!(bpo_float, 1.0);
    let bpo = bpo_float as usize;

    let curve = Curve {
        freq: Array1::from_vec(knots),
        spl: Array1::from_vec(values),
        ..Default::default()
    };
    let rust = smooth_one_over_n_octave(&curve, bpo);
    assert_eq!(rust.spl.len(), expected.len());

    let mut max_err = 0.0f64;
    for (i, (got, want)) in rust.spl.iter().zip(expected.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "bin {i}: rust={got:.12e} expected={want:.12e} abs_err={err:.3e} tol={TOL:.1e}"
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
