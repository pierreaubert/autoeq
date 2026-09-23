//! Wolfram cross-check: summation lattice + replay + delay ledger (RE11).
//!
//! Oracle: `wolfram/re11_summation_lattice.wls` (direct evaluation of
//! Hmain + g Hsub Exp[-j 2 Pi f tau] with the normalized band error,
//! full 12-candidate coarse enumeration, 5-point refinement, two seat
//! replays, and the ledger advance rules, all from closed forms).
//! Absolute tolerance 1e-9 on band errors; lattice identity (best
//! delay/polarity/gain, evaluated counts) and ledger outcomes match
//! exactly. No weighting-curve claim is made.

use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use roomeq_engine::summation_search::{
    DelayEntry, DelayLedger, SearchGrid, SeatCombinedInput, evaluate_candidate, reverify_combined,
    search_summation,
};

const CASE: &str = "re11_summation_lattice";
const CASE_ID: &str = "autoeq-qa.re11-summation-lattice.v1";
const TOL: f64 = 1e-9;

fn vec_f64(value: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(value[key].clone())
        .unwrap_or_else(|_| panic!("{CASE}: golden is missing `{key}`"))
}

#[test]
fn wolfram_re11_summation_lattice() {
    let ref_json = require_reference(CASE, "re11_summation_lattice.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs = vec_f64(&ref_json, "freqs_hz");
    let band = vec_f64(&ref_json, "band_hz");
    let coarse_delays = vec_f64(&ref_json, "coarse_delays_s");
    let coarse_gains = vec_f64(&ref_json, "coarse_gains_db");
    let coarse_errors = vec_f64(&ref_json, "coarse_errors");
    let fine_delays = vec_f64(&ref_json, "fine_delays_s");
    let fine_errors = vec_f64(&ref_json, "fine_errors");
    let main_db: f64 = serde_json::from_value(ref_json["main_db"].clone()).unwrap();
    let sub_db: f64 = serde_json::from_value(ref_json["sub_db"].clone()).unwrap();
    let lag: f64 = serde_json::from_value(ref_json["true_lag_s"].clone()).unwrap();
    assert_eq!(freqs.len(), 6, "{CASE}: expected 6 grid points");
    assert!(
        freqs.iter().all(|v| v.is_finite()),
        "{CASE}: non-finite grid"
    );

    // Shared grid, declared once: main carries the pure 3 ms lag.
    let main_mag = vec![main_db; freqs.len()];
    let main_phase: Vec<f64> = freqs.iter().map(|f| -360.0 * f * lag).collect();
    let sub_mag = vec![sub_db; freqs.len()];
    let sub_phase = vec![0.0; freqs.len()];
    let band_hz = [band[0], band[1]];
    let mut max_abs = 0.0f64;

    // Bounded coarse enumeration: identity and every candidate error.
    let grid = SearchGrid {
        delays_s: coarse_delays.clone(),
        gains_db: coarse_gains.clone(),
        include_polarity_inversion: true,
    };
    let outcome = search_summation(
        &freqs,
        &main_mag,
        &main_phase,
        &sub_mag,
        &sub_phase,
        band_hz,
        &grid,
    )
    .expect("valid lattice must search");
    let evaluated: usize = serde_json::from_value(ref_json["coarse_evaluated"].clone()).unwrap();
    assert_eq!(outcome.evaluated, evaluated, "{CASE}: evaluated count");
    assert_eq!(
        outcome.evaluated,
        coarse_errors.len(),
        "{CASE}: oracle error table length"
    );
    assert!(!outcome.best.polarity_inverted, "{CASE}: best polarity");
    assert!(
        (outcome.best.delay_s - 0.003).abs() == 0.0,
        "{CASE}: best delay {}, want 3 ms",
        outcome.best.delay_s
    );
    assert!(
        (outcome.best.gain_db - 0.0).abs() == 0.0,
        "{CASE}: best gain {}, want 0 dB",
        outcome.best.gain_db
    );
    let best_err: f64 = serde_json::from_value(ref_json["coarse_best_error"].clone()).unwrap();
    assert!(
        (outcome.best.band_error - best_err).abs() <= 1e-12,
        "{CASE}: best error {} vs oracle {best_err}",
        outcome.best.band_error
    );
    max_abs = max_abs.max((outcome.best.band_error - best_err).abs());

    // Candidate table spot checks across the lattice order.
    for (i, err) in coarse_errors.iter().enumerate() {
        let polarity = i >= coarse_delays.len() * coarse_gains.len();
        let rest = i % (coarse_delays.len() * coarse_gains.len());
        let delay = coarse_delays[rest / coarse_gains.len()];
        let gain = coarse_gains[rest % coarse_gains.len()];
        let rust = evaluate_candidate(
            &freqs,
            &main_mag,
            &main_phase,
            &sub_mag,
            &sub_phase,
            band_hz,
            polarity,
            delay,
            gain,
        )
        .expect("valid candidate must evaluate");
        assert!(
            (rust - err).abs() <= TOL,
            "{CASE}: candidate[{i}] (pol={polarity}, tau={delay}, g={gain}): rust={rust:.6e} oracle={err:.6e}"
        );
        max_abs = max_abs.max((rust - err).abs());
    }

    // Refinement around the known optimum.
    let fine = SearchGrid {
        delays_s: fine_delays.clone(),
        gains_db: vec![0.0],
        include_polarity_inversion: false,
    };
    let refined = search_summation(
        &freqs,
        &main_mag,
        &main_phase,
        &sub_mag,
        &sub_phase,
        band_hz,
        &fine,
    )
    .expect("valid refinement must search");
    assert_eq!(refined.evaluated, fine_errors.len());
    assert!(
        (refined.best.delay_s - 0.003).abs() == 0.0,
        "{CASE}: refined delay {}, want 3 ms",
        refined.best.delay_s
    );
    let fine_best: f64 = serde_json::from_value(ref_json["fine_best_error"].clone()).unwrap();
    assert!(
        (refined.best.band_error - fine_best).abs() <= 1e-12,
        "{CASE}: refined error mismatch"
    );

    // Direct off-lattice and inverted evaluations.
    let off_point: Vec<f64> =
        serde_json::from_value(ref_json["off_lattice_point"].clone()).unwrap();
    let off_err: f64 = serde_json::from_value(ref_json["off_lattice_error"].clone()).unwrap();
    let rust_off = evaluate_candidate(
        &freqs,
        &main_mag,
        &main_phase,
        &sub_mag,
        &sub_phase,
        band_hz,
        off_point[0] == 1.0,
        off_point[1],
        off_point[2],
    )
    .unwrap();
    assert!(
        (rust_off - off_err).abs() <= TOL,
        "{CASE}: off-lattice rust={rust_off:.6e} oracle={off_err:.6e}"
    );
    max_abs = max_abs.max((rust_off - off_err).abs());
    let inv_err: f64 = serde_json::from_value(ref_json["inverted_error"].clone()).unwrap();
    let rust_inv = evaluate_candidate(
        &freqs,
        &main_mag,
        &main_phase,
        &sub_mag,
        &sub_phase,
        band_hz,
        true,
        0.003,
        0.0,
    )
    .unwrap();
    assert!(
        (rust_inv - inv_err).abs() <= TOL,
        "{CASE}: inverted rust={rust_inv:.6e} oracle={inv_err:.6e}"
    );
    max_abs = max_abs.max((rust_inv - inv_err).abs());

    // Seat replay: MLP plus a seat with no relative delay.
    let seat2_main: f64 = serde_json::from_value(ref_json["seat2_main_db"].clone()).unwrap();
    let seat2_sub: f64 = serde_json::from_value(ref_json["seat2_sub_db"].clone()).unwrap();
    let seat2_tol: f64 = serde_json::from_value(ref_json["seat2_tolerance"].clone()).unwrap();
    let seat2_err: f64 = serde_json::from_value(ref_json["seat2_error"].clone()).unwrap();
    let seat2_main_mag = vec![seat2_main; freqs.len()];
    let seat2_sub_mag = vec![seat2_sub; freqs.len()];
    let seats = vec![
        SeatCombinedInput {
            seat_id: "mlp",
            freqs: &freqs,
            main_mag_db: &main_mag,
            main_phase_deg: &main_phase,
            sub_mag_db: &sub_mag,
            sub_phase_deg: &sub_phase,
        },
        SeatCombinedInput {
            seat_id: "seat-2",
            freqs: &freqs,
            main_mag_db: &seat2_main_mag,
            main_phase_deg: &sub_phase,
            sub_mag_db: &seat2_sub_mag,
            sub_phase_deg: &sub_phase,
        },
    ];
    let replays = reverify_combined(&seats, &outcome.best, band_hz, seat2_tol).unwrap();
    assert_eq!(replays.len(), 2);
    assert!(replays[0].replayed_ok, "{CASE}: MLP replay must pass");
    assert!(
        replays[0].band_error <= 1e-12,
        "{CASE}: MLP replay error {}",
        replays[0].band_error
    );
    assert!(replays[1].replayed_ok, "{CASE}: seat-2 replay must pass");
    assert!(
        (replays[1].band_error - seat2_err).abs() <= TOL,
        "{CASE}: seat-2 rust={} oracle={seat2_err}",
        replays[1].band_error
    );
    max_abs = max_abs.max((replays[1].band_error - seat2_err).abs());

    // Delay ledger: samples, reduce, then common latency.
    let entry = DelayEntry {
        label: "sub-1".to_string(),
        delay_s: 0.003,
        sample_rate_hz: 48000.0,
    };
    let want_samples: i64 =
        serde_json::from_value(ref_json["ledger_delay_samples"].clone()).unwrap();
    assert_eq!(entry.samples(), want_samples, "{CASE}: delay samples");
    let mut ledger = DelayLedger {
        entries: vec![entry],
        common_latency_s: 0.0,
    };
    let first = ledger.apply_advance("sub-1", 0.001).unwrap();
    assert!(
        matches!(
            first,
            roomeq_engine::summation_search::AdvanceOutcome::ReducedExistingDelay { .. }
        ),
        "{CASE}: first advance must reduce, got {first:?}"
    );
    assert!((ledger.entries[0].delay_s - 0.002).abs() == 0.0);
    let second = ledger.apply_advance("sub-1", 0.005).unwrap();
    assert!(
        matches!(
            second,
            roomeq_engine::summation_search::AdvanceOutcome::AddedCommonLatency { .. }
        ),
        "{CASE}: second advance must add common latency, got {second:?}"
    );
    assert!((ledger.entries[0].delay_s - 0.0).abs() == 0.0);
    assert!((ledger.common_latency_s - 0.003).abs() == 0.0);

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
