//! Wolfram cross-check: nonuniform group-delay stencil + duration (O09).
//!
//! Oracle: `wolfram/o09_group_delay.wls` (NumPy-style unwrap, backward
//! tau_g = -dphi/domega stencil in ms with gd[0] = gd[1], group-delay
//! standard deviation, detrended phase deviation, magnitude/phase loss
//! combination). Tolerances: 1e-9 absolute in ms/degrees.

use autoeq_optim::loss::phase_aware::{
    compute_group_delay, impulse_response_duration, magnitude_phase_loss, phase_deviation,
};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use ndarray::Array1;

const CASE: &str = "o09_group_delay";
const CASE_ID: &str = "autoeq-qa.o09-group-delay.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_o09_group_delay() {
    let ref_json = require_reference(CASE, "o09_group_delay.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let wrapped: Vec<f64> = serde_json::from_value(ref_json["phase_wrapped_deg"].clone()).unwrap();
    let want_gd: Vec<f64> = serde_json::from_value(ref_json["group_delay_ms"].clone()).unwrap();
    assert_eq!(freqs.len(), 5, "{CASE}: expected five grid points");

    let freqs_a = Array1::from_vec(freqs);
    let phase_a = Array1::from_vec(wrapped);
    let gd = compute_group_delay(&freqs_a, &phase_a);
    // The stencil must see the wrap: raw wrapped steps differ from the
    // unwrapped truth by 360 deg at the wrapped jump.
    assert!(
        (want_gd[4] - 1.0).abs() < 0.2,
        "{CASE}: fixture lost its delay character"
    );

    let mut max_abs: f64 = 0.0;
    for (i, (&actual, &expected)) in gd.iter().zip(want_gd.iter()).enumerate() {
        assert!(actual.is_finite(), "{CASE}: non-finite gd[{i}]");
        let err = (actual - expected).abs();
        assert!(
            err <= TOL,
            "{CASE}: gd[{i}] rust={actual:.12e} expected={expected:.12e} abs_err={err:.3e}"
        );
        max_abs = max_abs.max(err);
    }

    let duration = impulse_response_duration(&freqs_a, &phase_a);
    let want_duration = ref_json["ir_duration_ms"].as_f64().unwrap();
    assert!(want_duration > 0.0, "{CASE}: degenerate duration");
    let err = (duration - want_duration).abs();
    assert!(
        err <= TOL,
        "{CASE}: duration rust={duration:.12e} expected={want_duration:.12e}"
    );
    max_abs = max_abs.max(err);

    let zeros = Array1::zeros(freqs_a.len());
    let dev = phase_deviation(&phase_a, &zeros, &freqs_a);
    let want_dev = ref_json["phase_deviation_deg"].as_f64().unwrap();
    let err = (dev - want_dev).abs();
    assert!(
        err <= TOL,
        "{CASE}: deviation rust={dev:.12e} expected={want_dev:.12e}"
    );
    max_abs = max_abs.max(err);

    let loss = magnitude_phase_loss(
        ref_json["magnitude_error"].as_f64().unwrap(),
        dev,
        ref_json["phase_weight"].as_f64().unwrap(),
    );
    let want_loss = ref_json["magnitude_phase_loss"].as_f64().unwrap();
    let err = (loss - want_loss).abs();
    assert!(
        err <= TOL,
        "{CASE}: mag-phase loss rust={loss:.12e} expected={want_loss:.12e}"
    );
    max_abs = max_abs.max(err);

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_abs,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
