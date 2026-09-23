//! Wolfram cross-check: index-domain Gaussian smoothing (C04).
//!
//! Oracle: `wolfram/c04_gaussian_smooth.wls` (direct double sum with
//! Exp[-0.5 (k/sigma)^2], radius Ceil[3 sigma], edge renormalization;
//! sigma is in samples). Tolerance 1e-12 absolute; sigma <= 0 identity.

use autoeq_core::curve_transforms::smooth_gaussian;
use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;

const CASE: &str = "c04_gaussian_smooth";
const CASE_ID: &str = "autoeq-qa.c04-gaussian-smooth.v1";
const TOL: f64 = 1e-12;

#[test]
fn wolfram_c04_gaussian_smooth() {
    let ref_json = require_reference(CASE, "c04_gaussian_smooth.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let input: Vec<f64> = serde_json::from_value(ref_json["input"].clone()).unwrap();
    let expected: Vec<f64> = serde_json::from_value(ref_json["smoothed"].clone()).unwrap();
    let identity: Vec<f64> = serde_json::from_value(ref_json["identity_expected"].clone()).unwrap();
    assert_eq!(input.len(), 7, "{CASE}: expected 7 samples");
    assert_eq!(
        expected.len(),
        input.len(),
        "{CASE}: output/signal length zip"
    );
    let sigma: f64 = serde_json::from_value(ref_json["sigma_samples"].clone()).unwrap();
    assert_eq!(sigma, 1.0, "{CASE}: sigma is in samples");
    let radius: usize = serde_json::from_value(ref_json["kernel_radius"].clone()).unwrap();
    assert_eq!(radius, 3, "{CASE}: radius is Ceil[3 sigma]");

    let signal = Array1::from_vec(input);
    let rust = smooth_gaussian(&signal, sigma);
    assert_eq!(rust.len(), expected.len());
    let mut max_err = 0.0f64;
    for (i, (got, want)) in rust.iter().zip(expected.iter()).enumerate() {
        assert!(want.is_finite(), "{CASE}: non-finite reference at {i}");
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "sample {i}: rust={got:.12e} expected={want:.12e} abs_err={err:.3e} tol={TOL:.1e}"
        );
        max_err = max_err.max(err);
    }

    // sigma <= 0 is the identity (separate contract clause).
    for dead in [0.0, -1.0] {
        let id = smooth_gaussian(&signal, dead);
        assert_eq!(id.len(), identity.len());
        for (i, (got, want)) in id.iter().zip(identity.iter()).enumerate() {
            assert!(
                (got - want).abs() == 0.0,
                "{CASE}: sigma={dead} must be the identity at {i}"
            );
        }
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
