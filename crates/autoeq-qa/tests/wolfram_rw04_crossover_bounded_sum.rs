//! Wolfram cross-check: crossover bounded sum and replay (RW04).
//!
//! Oracle: `wolfram/rw04_crossover_bounded_sum.wls` (per-candidate
//! coherent sums from polar dB/degree transfers, the published band
//! error, and the triangle-inequality magnitude bound, all stated
//! from the definitions). Exercises the real workflow path
//! (`verify_crossover_alignment`): input-route-output DSP times each
//! measured source/seat transfer, summed only within the common
//! timing scope, with the accepted alignment replayed per seat.
//! Winner triple exact; band error absolute 1e-12 (class A);
//! per-bin magnitudes relative 1e-9 (class L).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, rel_error};
use roomeq_engine::summation_search::SearchGrid;
use roomeq_workflow::crossover_summation::{SeatSummationInput, verify_crossover_alignment};

const CASE: &str = "rw04_crossover_bounded_sum";
const CASE_ID: &str = "autoeq-qa.rw04-crossover-bounded-sum.v1";
const TOL_ABS: f64 = 1e-12;
const REPLAY_TOL: f64 = 1e-6;

#[test]
fn wolfram_rw04_crossover_bounded_sum() {
    let ref_json = require_reference(CASE, "rw04_crossover_bounded_sum.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let main_mag: Vec<f64> = serde_json::from_value(ref_json["main_mag_db"].clone()).unwrap();
    let main_ph: Vec<f64> = serde_json::from_value(ref_json["main_phase_deg"].clone()).unwrap();
    let sub_mag: Vec<f64> = serde_json::from_value(ref_json["sub_mag_db"].clone()).unwrap();
    let sub_ph: Vec<f64> = serde_json::from_value(ref_json["sub_phase_deg"].clone()).unwrap();
    let band: [f64; 2] = serde_json::from_value(ref_json["band_hz"].clone()).unwrap();
    let delays: Vec<f64> = serde_json::from_value(ref_json["delays_s"].clone()).unwrap();
    let gains: Vec<f64> = serde_json::from_value(ref_json["gains_db"].clone()).unwrap();
    let evaluated: usize = serde_json::from_value(ref_json["evaluated"].clone()).unwrap();
    let winner = ref_json["winner"].clone();
    let win_mag: Vec<f64> =
        serde_json::from_value(ref_json["winner_combined_mag"].clone()).unwrap();
    let win_ideal: Vec<f64> = serde_json::from_value(ref_json["winner_ideal_mag"].clone()).unwrap();
    for (name, v) in [
        ("main_mag", &main_mag),
        ("main_phase", &main_ph),
        ("sub_mag", &sub_mag),
        ("sub_phase", &sub_ph),
    ] {
        assert_eq!(
            v.len(),
            freqs.len(),
            "{CASE}: {name} length must match grid"
        );
    }
    assert_eq!((win_mag.len(), win_ideal.len()), (freqs.len(), freqs.len()));
    assert_eq!(
        evaluated, 12,
        "{CASE}: expected 3 delays x 2 gains x 2 polarities"
    );

    // One shared grid object feeds every response: no resampling or zip
    // of unequal grids is possible; the workflow asserts it anyway.
    let seat = SeatSummationInput {
        seat_id: "mlp".to_string(),
        is_mlp: true,
        freqs: freqs.clone(),
        main_mag_db: main_mag,
        main_phase_deg: main_ph,
        sub_mag_db: sub_mag,
        sub_phase_deg: sub_ph,
    };
    let grid = SearchGrid {
        delays_s: delays,
        gains_db: gains,
        include_polarity_inversion: true,
    };
    let report = verify_crossover_alignment(&[seat], band, &grid, REPLAY_TOL)
        .unwrap_or_else(|error| panic!("{CASE}: verification must accept: {error}"));
    assert_eq!(
        (
            report.accepted_polarity_inverted,
            report.accepted_delay_s,
            report.accepted_gain_db
        ),
        (
            winner["polarity_inverted"].as_bool().unwrap(),
            winner["delay_s"].as_f64().unwrap(),
            winner["gain_db"].as_f64().unwrap(),
        ),
        "{CASE}: accepted alignment must match the oracle winner"
    );
    let want_error = winner["band_error"].as_f64().unwrap();
    let err_abs = (report.mlp_band_error - want_error).abs();
    assert!(
        err_abs <= TOL_ABS,
        "{CASE}: band error rust={:.12e} expected={want_error:.12e} err={err_abs:.3e}",
        report.mlp_band_error
    );
    assert_eq!(report.evaluated, evaluated, "{CASE}: candidate count");
    assert_eq!(report.seat_replays.len(), 1);
    let replay = &report.seat_replays[0];
    assert!(
        replay.replayed_ok,
        "{CASE}: MLP replay must sit within tolerance"
    );
    let replay_abs = (replay.band_error - want_error).abs();
    assert!(
        replay_abs <= TOL_ABS,
        "{CASE}: replay band error err={replay_abs:.3e}"
    );

    // Unknown-phase magnitude bound: the coherent sum never exceeds the
    // incoherent ideal on any bin (triangle inequality).
    for (i, (mag, ideal)) in win_mag.iter().zip(win_ideal.iter()).enumerate() {
        assert!(
            *mag <= *ideal + 1e-12,
            "{CASE}: bin {i}: |Hsum|={mag:.12e} exceeds ideal={ideal:.12e}"
        );
    }
    let max_rel = rel_error(report.mlp_band_error, want_error);

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_rel,
        max_abs_error: err_abs.max(replay_abs),
        tolerance: TOL_ABS,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
