//! Wolfram cross-check: SPL-calibration level arithmetic (RM01).
//!
//! Oracle: `wolfram/rm01_spl_level.wls` (level doubling is exactly
//! 20*log10(2) dB; forward/inverse round trip; clamp and epsilon
//! floor). Complements the sibling `rm01_schroeder_spl` room-volume
//! relation with the dB-domain invariants. Tolerance 1e-4 absolute
//! (f32 calibration path).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_model::SplCalibration;

const CASE: &str = "rm01_spl_level";
const CASE_ID: &str = "autoeq-qa.rm01-spl-level.v1";
const TOL: f64 = 1e-4;

#[test]
fn wolfram_rm01_spl_level() {
    let ref_json = require_reference(CASE, "rm01_spl_level.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let offset: f32 = serde_json::from_value(ref_json["spl_offset_db"].clone()).unwrap();
    let pairs: Vec<[f32; 2]> = serde_json::from_value(ref_json["doubling_pairs"].clone()).unwrap();
    let diffs: Vec<f64> = serde_json::from_value(ref_json["doubling_diffs_db"].clone()).unwrap();
    let exact: f64 = serde_json::from_value(ref_json["exact_doubling_db"].clone()).unwrap();
    let targets: Vec<f32> = serde_json::from_value(ref_json["targets_db_spl"].clone()).unwrap();
    let want_peaks: Vec<f32> =
        serde_json::from_value(ref_json["peaks_for_target"].clone()).unwrap();
    let round_trip: Vec<f64> =
        serde_json::from_value(ref_json["round_trip_db_spl"].clone()).unwrap();
    let peak_at_offset: f32 = serde_json::from_value(ref_json["peak_at_offset"].clone()).unwrap();
    let peak_above: f32 = serde_json::from_value(ref_json["peak_above_clamp"].clone()).unwrap();
    let floor_db: f64 = serde_json::from_value(ref_json["floor_db_spl"].clone()).unwrap();
    assert_eq!(pairs.len(), diffs.len());
    assert_eq!(targets.len(), want_peaks.len());
    assert_eq!(targets.len(), round_trip.len());

    let cal = SplCalibration {
        reported_db_spl: 94.0,
        reference_freq_hz: 1000.0,
        peak_sample_level: 0.5,
        spl_offset_db: offset,
    };
    let mut max_err = 0.0f64;
    // Level doubling adds exactly 20*log10(2) dB (catches 10-vs-20 slips).
    for ([lo, hi], want) in pairs.iter().zip(diffs.iter()) {
        let got = (cal.dbspl_for_peak_level(*hi) - cal.dbspl_for_peak_level(*lo)) as f64;
        for (what, a, b) in [("doubling", got, *want), ("exact", *want, exact)] {
            let err = (a - b).abs();
            assert!(
                err <= TOL,
                "{what}: got={a:.9e} expected={b:.9e} abs_err={err:.3e}"
            );
            max_err = max_err.max(err);
        }
    }
    // Inverse values and full round trip on the representable path.
    for ((target, want_peak), want_back) in
        targets.iter().zip(want_peaks.iter()).zip(round_trip.iter())
    {
        let peak = cal.peak_level_for_dbspl(*target);
        let err = (peak - want_peak).abs() as f64;
        assert!(
            err <= TOL,
            "peak({target}): rust={peak:.9e} expected={want_peak:.9e}"
        );
        max_err = max_err.max(err);
        let back = cal.dbspl_for_peak_level(peak) as f64;
        let back_err = (back - want_back).abs();
        assert!(
            back_err <= TOL,
            "round_trip({target}): rust={back:.9e} expected={want_back:.9e}"
        );
        max_err = max_err.max(back_err);
    }
    // Clamp boundaries: exactly 1.0 at the offset, clamped above it.
    assert_eq!(cal.peak_level_for_dbspl(offset), peak_at_offset);
    assert_eq!(cal.peak_level_for_dbspl(offset), 1.0);
    assert_eq!(cal.peak_level_for_dbspl(140.0), peak_above);
    assert_eq!(cal.peak_level_for_dbspl(140.0), 1.0);
    // Sub-epsilon peaks floor at f32::EPSILON, never at zero or NaN.
    let tiny = cal.dbspl_for_peak_level(1e-9);
    assert!(tiny.is_finite());
    let floor_err = (tiny as f64 - floor_db).abs();
    assert!(
        floor_err <= TOL,
        "floor: rust={tiny:.9e} expected={floor_db:.9e}"
    );
    max_err = max_err.max(floor_err);
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
