//! Wolfram cross-check: phase serialization and thresholds (RM04).
//!
//! Oracle: `wolfram/rm04_phase_thresholds.wls` (phase wrap
//! mod(p+180,360)-180 with nonfinite passthrough; cancellation
//! acceptance with its 0.05 dB tolerance and exact boundaries;
//! residual min/max/RMS over finite entries). Phase wrap runs through
//! the real `CurveData` serialization; acceptance and residual are
//! exact decision/scalar checks with 1e-9 absolute tolerance.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_model::report_contracts::MixedPhaseCorrectionReport;
use roomeq_model::{CurveData, crossover_cancellation_accepted};

const CASE: &str = "rm04_phase_thresholds";
const CASE_ID: &str = "autoeq-qa.rm04-phase-thresholds.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_rm04_phase_thresholds() {
    let ref_json = require_reference(CASE, "rm04_phase_thresholds.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let phases: Vec<f64> = serde_json::from_value(ref_json["phases_deg"].clone()).unwrap();
    let want_wrapped: Vec<f64> = serde_json::from_value(ref_json["wrapped_deg"].clone()).unwrap();
    let cases: Vec<[f64; 3]> = serde_json::from_value(ref_json["accept_cases"].clone()).unwrap();
    let expected: Vec<bool> = serde_json::from_value(ref_json["accept_expected"].clone()).unwrap();
    let missing: Vec<(f64, Option<f64>, f64)> =
        serde_json::from_value(ref_json["accept_missing_baseline"].clone()).unwrap();
    let missing_expected: Vec<bool> =
        serde_json::from_value(ref_json["accept_missing_expected"].clone()).unwrap();
    let residual: Vec<f64> = serde_json::from_value(ref_json["residual_deg"].clone()).unwrap();
    let want_min: f64 = serde_json::from_value(ref_json["residual_min_deg"].clone()).unwrap();
    let want_max: f64 = serde_json::from_value(ref_json["residual_max_deg"].clone()).unwrap();
    let want_rms: f64 = serde_json::from_value(ref_json["residual_rms_deg"].clone()).unwrap();
    assert_eq!(phases.len(), want_wrapped.len());
    assert_eq!(cases.len(), expected.len());
    assert_eq!(missing.len(), missing_expected.len());

    // Phase wrap through the real serialization path, incl. +/-180.
    let data = CurveData {
        freq: phases.clone(),
        spl: vec![0.0; phases.len()],
        phase: Some(phases.clone()),
        ..Default::default()
    };
    let value = serde_json::to_value(&data).expect("CurveData serializes");
    let got_wrapped: Vec<f64> =
        serde_json::from_value(value["phase"].clone()).expect("wrapped phase array");
    let mut max_err = 0.0f64;
    for ((got, want), input) in got_wrapped
        .iter()
        .zip(want_wrapped.iter())
        .zip(phases.iter())
    {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "wrap({input}): rust={got:.12e} expected={want:.12e}"
        );
        max_err = max_err.max(err);
    }
    // Nonfinite inputs pass through (serialized as null), never wrapped.
    let nonfinite = CurveData {
        freq: vec![100.0; 3],
        spl: vec![0.0; 3],
        phase: Some(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]),
        ..Default::default()
    };
    let nf_value = serde_json::to_value(&nonfinite).expect("nonfinite serializes");
    assert!(
        nf_value["phase"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v.is_null()),
        "nonfinite phase must pass through as null, got {}",
        nf_value["phase"]
    );

    // Cancellation acceptance, including the exact limit+0.05 boundary.
    for ([candidate, baseline, limit], want) in cases.iter().zip(expected.iter()) {
        let got = crossover_cancellation_accepted(*candidate, Some(*baseline), *limit);
        assert_eq!(
            got, *want,
            "accepted({candidate}, {baseline}, {limit}): rust={got} expected={want}"
        );
    }
    for ((candidate, baseline, limit), want) in missing.iter().zip(missing_expected.iter()) {
        let got = crossover_cancellation_accepted(*candidate, *baseline, *limit);
        assert_eq!(
            got, *want,
            "accepted({candidate}, {baseline:?}, {limit}): rust={got} expected={want}"
        );
    }
    assert!(!crossover_cancellation_accepted(f64::NAN, Some(10.0), 3.0));

    // Residual min/max/RMS over finite entries; nonfinite is ignored.
    let mut with_junk = residual.clone();
    with_junk.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    let report = MixedPhaseCorrectionReport::from_residual(
        1.5,
        with_junk.len(),
        &Array1::from_vec(with_junk),
    );
    for (what, got, want) in [
        ("min", report.residual_excess_phase_min_deg, want_min),
        ("max", report.residual_excess_phase_max_deg, want_max),
        ("rms", report.residual_excess_phase_rms_deg, want_rms),
    ] {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{what}: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }
    // Empty and all-nonfinite residuals are exact zeros, not NaN.
    for empty in [
        Array1::from_vec(Vec::new()),
        Array1::from_vec(vec![f64::NAN, f64::INFINITY]),
    ] {
        let zero = MixedPhaseCorrectionReport::from_residual(0.0, 0, &empty);
        assert_eq!(zero.residual_excess_phase_min_deg, 0.0);
        assert_eq!(zero.residual_excess_phase_max_deg, 0.0);
        assert_eq!(zero.residual_excess_phase_rms_deg, 0.0);
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
