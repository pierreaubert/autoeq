//! Wolfram cross-check: Schroeder frequency + SPL calibration (RM01).
//!
//! Oracle: `wolfram/rm01_schroeder_spl.wls` (engineering formula from
//! dimensions, closed-form forward/inverse calibration). Covers three
//! ordinary rooms, degenerate (zero/negative) inputs pinned to 0.0,
//! the f32-epsilon floor, and the [0, 1] output clamp. Tolerances:
//! 1e-9 absolute on Hz (f64 algebra), 1e-4 dB/linear on the f32
//! calibration path.

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_model::{RoomDimensions, SplCalibration};

const CASE: &str = "rm01_schroeder_spl";
const CASE_ID: &str = "autoeq-qa.rm01-schroeder-spl.v1";
const TOL_HZ: f64 = 1e-9;
const TOL_F32: f64 = 1e-4;

#[test]
fn wolfram_rm01_schroeder_spl() {
    let ref_json = require_reference(CASE, "rm01_schroeder_spl.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let rooms: Vec<[f64; 4]> = serde_json::from_value(ref_json["rooms_lwh_rt"].clone()).unwrap();
    let want_hz: Vec<f64> = serde_json::from_value(ref_json["schroeder_hz"].clone()).unwrap();
    let degenerate: Vec<[f64; 4]> =
        serde_json::from_value(ref_json["degenerate_rooms_lwh_rt"].clone()).unwrap();
    let offset: f32 = serde_json::from_value(ref_json["spl_offset_db"].clone()).unwrap();
    let peaks: Vec<f32> = serde_json::from_value(ref_json["peak_levels"].clone()).unwrap();
    let want_db: Vec<f32> = serde_json::from_value(ref_json["spl_db"].clone()).unwrap();
    let targets: Vec<f32> = serde_json::from_value(ref_json["spl_targets"].clone()).unwrap();
    let want_peaks: Vec<f32> = serde_json::from_value(ref_json["peak_for_target"].clone()).unwrap();
    assert_eq!(rooms.len(), want_hz.len());
    assert_eq!(peaks.len(), want_db.len());
    assert_eq!(targets.len(), want_peaks.len());

    let mut max_hz = 0.0f64;
    for ([l, w, h, rt], want) in rooms.iter().zip(want_hz.iter()) {
        let room = RoomDimensions {
            length: *l,
            width: *w,
            height: *h,
        };
        let got = room.schroeder_frequency_with_rt60(*rt);
        let err = (got - want).abs();
        assert!(
            err <= TOL_HZ,
            "schroeder({l}x{w}x{h}, {rt}s): rust={got:.12e} expected={want:.12e}"
        );
        max_hz = max_hz.max(err);
    }
    // Default-RT60 entry point must agree with the explicit 0.4 s call.
    let default_room = RoomDimensions {
        length: 5.2,
        width: 4.1,
        height: 2.8,
    };
    assert!(
        (default_room.schroeder_frequency() - default_room.schroeder_frequency_with_rt60(0.4))
            .abs()
            <= 1e-12
    );
    for [l, w, h, rt] in degenerate.iter() {
        let room = RoomDimensions {
            length: *l,
            width: *w,
            height: *h,
        };
        assert_eq!(
            room.schroeder_frequency_with_rt60(*rt),
            0.0,
            "degenerate room ({l}x{w}x{h}, {rt}s) must pin to 0.0"
        );
    }

    let cal = SplCalibration {
        reported_db_spl: 94.0,
        reference_freq_hz: 1000.0,
        peak_sample_level: 0.5,
        spl_offset_db: offset,
    };
    let mut max_f32 = 0.0f64;
    for (peak, want) in peaks.iter().zip(want_db.iter()) {
        let got = cal.dbspl_for_peak_level(*peak);
        let err = (got - want).abs() as f64;
        assert!(
            err <= TOL_F32,
            "dbspl({peak}): rust={got:.6e} expected={want:.6e} abs_err={err:.3e}"
        );
        max_f32 = max_f32.max(err);
    }
    for (target, want) in targets.iter().zip(want_peaks.iter()) {
        let got = cal.peak_level_for_dbspl(*target);
        let err = (got - want).abs() as f64;
        assert!(
            err <= TOL_F32,
            "peak({target} dB): rust={got:.6e} expected={want:.6e} abs_err={err:.3e}"
        );
        max_f32 = max_f32.max(err);
    }
    // Round trip on the representable path.
    for peak in [1.0f32, 0.5, 0.25, 0.01] {
        let back = cal.peak_level_for_dbspl(cal.dbspl_for_peak_level(peak));
        assert!(
            (back - peak).abs() < 1e-6,
            "round trip failed at peak={peak}: got {back}"
        );
    }

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: max_hz.max(max_f32),
        tolerance: TOL_F32,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
