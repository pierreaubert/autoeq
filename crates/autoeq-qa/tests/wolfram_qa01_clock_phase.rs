//! Wolfram cross-check: QA clock/phase arithmetic (QA01).
//!
//! Oracle: `wolfram/qa01_clock_phase.wls` (signed clock offsets, phase
//! uncertainty, coherent sums with flagged cancellation, common-EQ
//! invariance with missing entries skipped). Tolerance 1e-9 absolute in
//! stated units; exact identities for decisions.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_qa::analytic::{
    CoherentCombination, clock_offset_seconds, coherent_combination_db,
    common_eq_relative_drift_db, phase_uncertainty_degrees,
};

const CASE: &str = "qa01_clock_phase";
const CASE_ID: &str = "autoeq-qa.qa01-clock-phase.v1";
const TOL: f64 = 1e-9;

#[test]
fn wolfram_qa01_clock_phase() {
    let ref_json = require_reference(CASE, "qa01_clock_phase.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let clock_cases: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["clock_cases_ppm_s"].clone()).unwrap();
    let clock_want: Vec<f64> = serde_json::from_value(ref_json["clock_offsets_s"].clone()).unwrap();
    let phase_cases: Vec<[f64; 2]> =
        serde_json::from_value(ref_json["phase_cases_hz_s"].clone()).unwrap();
    let phase_want: Vec<f64> = serde_json::from_value(ref_json["phase_degrees"].clone()).unwrap();
    assert_eq!(clock_cases.len(), clock_want.len());
    assert_eq!(phase_cases.len(), phase_want.len());

    let mut max_err = 0.0f64;
    for ([ppm, dur], want) in clock_cases.iter().zip(clock_want.iter()) {
        let got = clock_offset_seconds(*ppm, *dur);
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: clock({ppm} ppm, {dur} s) rust={got:.12e} expected={want:.12e}"
        );
        max_err = max_err.max(err);
    }
    for ([freq, dt], want) in phase_cases.iter().zip(phase_want.iter()) {
        let got = phase_uncertainty_degrees(*freq, *dt);
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: phase({freq} Hz, {dt} s) rust={got:.12e} expected={want:.12e}"
        );
        max_err = max_err.max(err);
    }

    let want_11: f64 = serde_json::from_value(ref_json["coherent_gain_11_db"].clone()).unwrap();
    let want_1111: f64 = serde_json::from_value(ref_json["coherent_gain_1111_db"].clone()).unwrap();
    match coherent_combination_db(&[1.0, 1.0]) {
        CoherentCombination::GainDb(got) => {
            let err = (got - want_11).abs();
            assert!(err <= TOL, "{CASE}: coherent [1,1] drift {err:.3e}");
            max_err = max_err.max(err);
        }
        CoherentCombination::Cancellation => panic!("{CASE}: [1,1] must not cancel"),
    }
    match coherent_combination_db(&[1.0, 1.0, 1.0, 1.0]) {
        CoherentCombination::GainDb(got) => {
            let err = (got - want_1111).abs();
            assert!(err <= TOL, "{CASE}: coherent [1,1,1,1] drift {err:.3e}");
            max_err = max_err.max(err);
        }
        CoherentCombination::Cancellation => panic!("{CASE}: [1,1,1,1] must not cancel"),
    }
    assert_eq!(
        coherent_combination_db(&[1.0, -1.0]),
        CoherentCombination::Cancellation,
        "{CASE}: opposite polarity must cancel, never score 0 dB"
    );

    let seat_a: Vec<Option<f64>> = serde_json::from_value(ref_json["seat_a_db"].clone()).unwrap();
    let seat_b: Vec<Option<f64>> = serde_json::from_value(ref_json["seat_b_db"].clone()).unwrap();
    let eq: Vec<Option<f64>> = serde_json::from_value(ref_json["common_eq_db"].clone()).unwrap();
    let want_drift: f64 = serde_json::from_value(ref_json["common_eq_drift_db"].clone()).unwrap();
    let got_drift = common_eq_relative_drift_db(&seat_a, &seat_b, &eq);
    assert!(
        (got_drift - want_drift).abs() <= TOL,
        "{CASE}: common-EQ drift rust={got_drift:.12e} expected={want_drift:.12e}"
    );
    // Invariance holds for any seat values: changing one seat leaves the
    // common-EQ drift at zero because the same EQ is added to both sides.
    let mut seat_a_changed = seat_a.clone();
    seat_a_changed[0] = Some(65.0);
    let want_changed: f64 =
        serde_json::from_value(ref_json["seat_changed_drift_db"].clone()).unwrap();
    let got_changed = common_eq_relative_drift_db(&seat_a_changed, &seat_b, &eq);
    assert!(
        (got_changed - want_changed).abs() <= TOL,
        "{CASE}: changed-seat drift rust={got_changed:.12e} expected={want_changed:.12e}"
    );
    max_err = max_err.max((got_changed - want_changed).abs());

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
