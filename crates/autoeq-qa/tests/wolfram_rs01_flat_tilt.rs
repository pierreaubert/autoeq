//! Wolfram cross-check: synthetic magnitude plants (RS01).
//!
//! Oracle: `wolfram/rs01_flat_tilt.wls` (log-spaced grid, flat zero plant,
//! -0.8 dB/octave tilt about 200 Hz, highpass rolloff below 80 Hz).
//! Tolerance 1e-9 absolute in dB, 1e-12 relative on grid frequencies.
//! Degenerate grids must yield empty curves, never NaNs.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_synthetic::{
    generate_flat_curve, generate_harman_tilt_curve, generate_speaker_rolloff_curve,
};

const CASE: &str = "rs01_flat_tilt";
const CASE_ID: &str = "autoeq-qa.rs01-flat-tilt.v1";
const TOL_DB: f64 = 1e-9;
const TOL_GRID: f64 = 1e-12;

#[test]
fn wolfram_rs01_flat_tilt() {
    let ref_json = require_reference(CASE, "rs01_flat_tilt.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let flat: Vec<f64> = serde_json::from_value(ref_json["flat_db"].clone()).unwrap();
    let tilt: Vec<f64> = serde_json::from_value(ref_json["tilt_db"].clone()).unwrap();
    let rolloff: Vec<f64> = serde_json::from_value(ref_json["rolloff_db"].clone()).unwrap();
    let xo: f64 = serde_json::from_value(ref_json["crossover_hz"].clone()).unwrap();
    let slope: f64 = serde_json::from_value(ref_json["slope_db_per_oct"].clone()).unwrap();
    assert_eq!(freqs.len(), 8, "{CASE}: expected 8 grid points");

    let mut max_err = 0.0f64;
    let flat_curve = generate_flat_curve(freqs[0], freqs[freqs.len() - 1], freqs.len());
    assert_eq!(flat_curve.freq.len(), freqs.len());
    for (i, ((g, w), s)) in flat_curve
        .freq
        .iter()
        .zip(freqs.iter())
        .zip(flat.iter())
        .enumerate()
    {
        let (g, w) = (*g, *w);
        let grid_err = (g - w).abs() / w.abs();
        assert!(
            grid_err <= TOL_GRID,
            "{CASE}: grid[{i}] rust={g:.12e} expected={w:.12e}"
        );
        max_err = max_err.max(grid_err);
        let spl = flat_curve.spl[i];
        assert!(spl.is_finite());
        let err = (spl - s).abs();
        assert!(err <= TOL_DB, "{CASE}: flat[{i}] drift {err:.3e}");
        max_err = max_err.max(err);
    }

    let tilt_curve = generate_harman_tilt_curve(freqs[0], freqs[freqs.len() - 1], freqs.len());
    for (i, s) in tilt.iter().enumerate() {
        let got = tilt_curve.spl[i];
        assert!(got.is_finite());
        let err = (got - s).abs();
        assert!(
            err <= TOL_DB,
            "{CASE}: tilt[{i}] rust={got:.12e} expected={s:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    let roll_curve =
        generate_speaker_rolloff_curve(freqs[0], freqs[freqs.len() - 1], freqs.len(), xo, slope);
    for (i, s) in rolloff.iter().enumerate() {
        let got = roll_curve.spl[i];
        assert!(got.is_finite());
        let err = (got - s).abs();
        assert!(
            err <= TOL_DB,
            "{CASE}: rolloff[{i}] rust={got:.12e} expected={s:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // Degenerate grids fail closed as empty curves, never NaN plants.
    let empty = generate_flat_curve(-20.0, 20000.0, 8);
    assert!(empty.freq.is_empty(), "{CASE}: negative band must be empty");
    let empty_tilt = generate_harman_tilt_curve(20.0, 20000.0, 1);
    assert!(
        empty_tilt.freq.is_empty(),
        "{CASE}: singleton grid must be empty"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_err,
        max_abs_error: 0.0,
        tolerance: TOL_GRID,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
