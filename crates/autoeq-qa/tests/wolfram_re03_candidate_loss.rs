//! Wolfram cross-check: prepared-EQ candidate-loss recomputation (RE03).
//!
//! Oracle: `wolfram/re03_candidate_loss.wls` (independent direct-sum
//! evaluation of the ERB-rate + log-band weighted RMS composition used by
//! `roomeq_engine::group::target_error_score`). Tolerance 1e-9 relative.

use autoeq_core::Curve;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, rel_error, require_reference};
use ndarray::Array1;

const CASE: &str = "re03_candidate_loss";
const CASE_ID: &str = "autoeq-qa.re03-candidate-loss.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_re03_candidate_loss() {
    let ref_json = require_reference(CASE, "re03_candidate_loss.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let meas: Vec<f64> = serde_json::from_value(ref_json["meas_db"].clone()).unwrap();
    let target: Vec<f64> = serde_json::from_value(ref_json["target_db"].clone()).unwrap();
    let expected_loss: f64 = serde_json::from_value(ref_json["loss_db"].clone()).unwrap();
    assert_eq!(freqs.len(), 12, "{CASE}: expected 12 grid points");
    assert_eq!(
        meas.len(),
        freqs.len(),
        "{CASE}: meas length must match grid"
    );
    assert_eq!(
        target.len(),
        freqs.len(),
        "{CASE}: target length must match grid"
    );
    assert!(
        expected_loss.is_finite(),
        "{CASE}: non-finite reference loss"
    );

    let meas_curve = Curve {
        freq: Array1::from_vec(freqs.clone()),
        spl: Array1::from_vec(meas.clone()),
        ..Default::default()
    };
    let target_curve = Curve {
        freq: Array1::from_vec(freqs.clone()),
        spl: Array1::from_vec(target.clone()),
        ..Default::default()
    };
    let rust = roomeq_engine::group::target_error_score(&meas_curve, &target_curve, 20.0, 20_000.0);
    assert!(rust.is_finite(), "{CASE}: non-finite candidate loss");
    let err = rel_error(rust, expected_loss);
    assert!(
        err <= TOL,
        "{CASE}: rust={rust:.12e} expected={expected_loss:.12e} rel_err={err:.3e} tol={TOL:.1e}"
    );

    // The fixture must exercise a non-trivial residual: a zero loss would
    // pass against any broken weighting.
    assert!(
        expected_loss > 0.5,
        "{CASE}: fixture residual too small to exercise weighting"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: err,
        max_abs_error: 0.0,
        tolerance: TOL,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
