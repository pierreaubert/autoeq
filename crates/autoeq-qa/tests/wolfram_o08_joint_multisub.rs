//! Wolfram cross-check: joint multi-sub scalarization (O08).
//!
//! Oracle: `wolfram/o08_joint_multisub.wls` (per-bin seat variance,
//! target MSE, unnormalized output penalty from the documented
//! definition; SPL never summed as pressure). Tolerance 1e-9 absolute
//! in dB^2 units.

use autoeq_optim::loss::{JointSubWeights, joint_multisub_loss};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};

const CASE: &str = "o08_joint_multisub";
const CASE_ID: &str = "autoeq-qa.o08-joint-multisub.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_o08_joint_multisub() {
    let ref_json = require_reference(CASE, "o08_joint_multisub.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let seats: Vec<Vec<f64>> = serde_json::from_value(ref_json["seat_levels_db"].clone()).unwrap();
    let reference: Vec<f64> =
        serde_json::from_value(ref_json["power_reference_db"].clone()).unwrap();
    let target: Vec<f64> = serde_json::from_value(ref_json["target_db"].clone()).unwrap();
    assert_eq!(seats.len(), 2, "{CASE}: expected two seats");
    assert_eq!(reference.len(), 4, "{CASE}: expected four bins");

    let got = joint_multisub_loss(&seats, &reference, &target, &JointSubWeights::default())
        .expect("fixture inputs must be valid");
    let want = |key: &str| ref_json[key].as_f64().unwrap();
    // Non-degenerate by design: every component is strictly positive.
    assert!(want("variation") > 0.0, "{CASE}: degenerate variation");
    assert!(
        want("output_drive") > 0.0,
        "{CASE}: degenerate output_drive"
    );
    assert!(
        want("target_error") > 0.0,
        "{CASE}: degenerate target_error"
    );

    let mut max_abs = 0.0f64;
    for (label, actual, expected) in [
        ("variation", got.variation, want("variation")),
        ("output_drive", got.output_drive, want("output_drive")),
        ("target_error", got.target_error, want("target_error")),
        ("total", got.total, want("total")),
    ] {
        assert!(actual.is_finite(), "{CASE}: non-finite {label}");
        let err = (actual - expected).abs();
        assert!(
            err <= TOL,
            "{CASE}: {label} rust={actual:.12e} expected={expected:.12e} abs_err={err:.3e}"
        );
        max_abs = max_abs.max(err);
    }
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
