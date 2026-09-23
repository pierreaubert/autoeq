//! Wolfram cross-check: MSO combined curves, polarity, allpass (RE14).
//!
//! Oracle: `wolfram/re14_combined_curves.wls` (independent per-seat
//! complex sums with gain/polarity/delay-slope/RBJ-allpass path factors
//! on the replicated log-spaced evaluation grid; weighted seat mean and
//! population seat variance follow from the summed SPLs). SPL/phase
//! compare absolute (dB/degrees, catalogue class A/I).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use ndarray::Array1;
use roomeq_engine::Curve;
use roomeq_engine::multiseat::{
    MultiSeatMeasurements, MultiSeatOptimizationResult, compute_multiseat_combined_curves,
};
use roomeq_model::MultiSeatStrategy;

const CASE: &str = "re14_combined_curves";
const CASE_ID: &str = "autoeq-qa.re14-combined-curves.v1";
const TOL: f64 = 1e-8;
const TOL_STATS: f64 = 1e-9;

fn curve(spl: f64, phase: f64, freq: &Array1<f64>) -> Curve {
    Curve {
        freq: freq.clone(),
        spl: Array1::from_elem(freq.len(), spl),
        phase: Some(Array1::from_elem(freq.len(), phase)),
        ..Default::default()
    }
}

fn check_abs(actual: f64, expected: f64, what: &str, max_err: &mut f64) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{CASE}: non-finite {what}: actual={actual} expected={expected}"
    );
    let err = (actual - expected).abs();
    assert!(
        err <= TOL,
        "{what}: rust={actual:.12e} expected={expected:.12e} abs_err={err:.3e}"
    );
    *max_err = (*max_err).max(err);
}

#[test]
fn wolfram_re14_combined_curves() {
    let ref_json = require_reference(CASE, "re14_combined_curves.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let grid: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let meas_freqs: Vec<f64> = serde_json::from_value(ref_json["meas_freqs_hz"].clone()).unwrap();
    let seat_spl: Vec<Vec<f64>> = serde_json::from_value(ref_json["seat_spl_db"].clone()).unwrap();
    let seat_ph: Vec<Vec<f64>> =
        serde_json::from_value(ref_json["seat_phase_deg"].clone()).unwrap();
    assert_eq!(grid.len(), 89, "{CASE}: expected 89 evaluation points");
    assert_eq!(seat_spl.len(), 2);

    // All four measurements share one explicit grid.
    let meas = Array1::from_vec(meas_freqs);
    let sub_spl: Vec<Vec<f64>> =
        serde_json::from_value(ref_json["sub_seat_spl_db"].clone()).unwrap();
    let sub_ph: Vec<Vec<f64>> =
        serde_json::from_value(ref_json["sub_seat_phase_deg"].clone()).unwrap();
    let measurements = vec![
        vec![
            curve(sub_spl[0][0], sub_ph[0][0], &meas),
            curve(sub_spl[0][1], sub_ph[0][1], &meas),
        ],
        vec![
            curve(sub_spl[1][0], sub_ph[1][0], &meas),
            curve(sub_spl[1][1], sub_ph[1][1], &meas),
        ],
    ];
    let set = MultiSeatMeasurements::new(measurements).unwrap();
    let result = MultiSeatOptimizationResult {
        gains: vec![1.5, -2.5],
        delays: vec![0.5, 1.25],
        polarities: vec![false, true],
        allpass_filters: vec![vec![(60.0, 1.0)], vec![]],
        strategy: MultiSeatStrategy::Average,
        objective_name: "qa".to_string(),
        objective_before: 0.0,
        objective_after: 0.0,
        objective_improvement_db: 0.0,
        variance_before: 0.0,
        variance_after: 0.0,
        variance_improvement_db: 0.0,
        improvement_db: 0.0,
    };
    let seats = compute_multiseat_combined_curves(&set, &result, (20.0, 200.0), 48000.0).unwrap();
    assert_eq!(seats.len(), 2, "{CASE}: expected 2 seat curves");

    let mut max_err = 0.0f64;
    for (seat_idx, seat) in seats.iter().enumerate() {
        assert_eq!(seat.freq.len(), grid.len());
        for (i, (got, want)) in seat.freq.iter().zip(grid.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-9,
                "{CASE}: seat {seat_idx} grid[{i}] drifted: {got} vs {want}"
            );
        }
        let phase = seat.phase.as_ref().expect("{CASE}: seat must carry phase");
        for i in 0..grid.len() {
            check_abs(
                seat.spl[i],
                seat_spl[seat_idx][i],
                &format!("seat{seat_idx} SPL[{i}]"),
                &mut max_err,
            );
            check_abs(
                phase[i],
                seat_ph[seat_idx][i],
                &format!("seat{seat_idx} phase[{i}]"),
                &mut max_err,
            );
        }
    }

    // Weighted seat mean and population seat variance from the
    // implementation-returned seat SPLs (definitional arithmetic check).
    let want_mean: Vec<f64> =
        serde_json::from_value(ref_json["weighted_mean_spl_db"].clone()).unwrap();
    let want_var: Vec<f64> = serde_json::from_value(ref_json["seat_variance_db2"].clone()).unwrap();
    for i in 0..grid.len() {
        let mean = (0.7 * seats[0].spl[i] + 0.3 * seats[1].spl[i]) / 1.0;
        let mid = 0.5 * (seats[0].spl[i] + seats[1].spl[i]);
        let var = 0.5 * ((seats[0].spl[i] - mid).powi(2) + (seats[1].spl[i] - mid).powi(2));
        assert!(
            (mean - want_mean[i]).abs() <= TOL_STATS,
            "{CASE}: weighted mean[{i}]: {mean} vs {m}",
            m = want_mean[i]
        );
        assert!(
            (var - want_var[i]).abs() <= TOL_STATS,
            "{CASE}: seat variance[{i}]: {var} vs {m}",
            m = want_var[i]
        );
        max_err = max_err
            .max((mean - want_mean[i]).abs())
            .max((var - want_var[i]).abs());
    }

    // Contract checks: empty and ragged measurement sets refuse loudly.
    assert!(MultiSeatMeasurements::new(vec![]).is_err());
    assert!(
        MultiSeatMeasurements::new(vec![
            vec![curve(80.0, 0.0, &meas)],
            vec![curve(80.0, 0.0, &meas), curve(80.0, 0.0, &meas)],
        ])
        .is_err()
    );

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
