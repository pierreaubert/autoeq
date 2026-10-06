//! Wolfram cross-check: measurement confidence + direct-sound geometry (C07).
//!
//! Oracle: `wolfram/c07_quality_geometry.wls` (order-statistic medians,
//! population seat variance, (rR-rD)/c interval, cycles/gate bound, and
//! the 9-degree timing bound f <= 25000/residual_us, all evaluated
//! independently in the engine). Tolerance class A/X: 1e-9 absolute on
//! native summaries, exact quality/decision labels.

use autoeq_core::Curve;
use autoeq_core::capture_provenance::{
    CaptureCorrection, CaptureGeometry, CaptureProvenance, CaptureTakeProvenance,
};
use autoeq_core::direct_sound::{
    DirectSoundCaptureFacts, reflection_free_interval_s, valid_lower_bound_hz,
};
use autoeq_core::measurement_quality::{
    MeasurementQuality, assess_measurement_quality, assess_multiple_measurement_quality,
};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use ndarray::Array1;

const CASE: &str = "c07_quality_geometry";
const CASE_ID: &str = "autoeq-qa.c07-quality-geometry.v1";
const TOL: f64 = 1e-9;

fn abs_err(a: f64, b: f64) -> f64 {
    (a - b).abs()
}

fn provenance_takes(residual_us: f64) -> CaptureProvenance {
    CaptureProvenance {
        geometry: CaptureGeometry::Compact,
        reflection_report: None,
        takes: (0..2)
            .map(|index| CaptureTakeProvenance {
                seat_id: None,
                microphone_id: format!("mic-{index}"),
                device_id: "aggregate-input".into(),
                offset_samples: Some(123.25),
                skew_ppm: Some(-80.0),
                residual_uncertainty_us: Some(residual_us),
                correction_applied: CaptureCorrection::Resampled,
                timing_reference_id: Some("fixed-emitter-1".into()),
                calibration_id: format!("cal-{index}"),
                gain_db: 0.0,
                calibration_orientation: "on_axis".into(),
                position_m: [index as f64 * 0.06, 0.0, 0.0],
                position_uncertainty_mm: 0.5,
                preserves_acoustic_delay: true,
                quality_passed: true,
            })
            .collect(),
    }
}

#[test]
fn wolfram_c07_quality_geometry() {
    let ref_json = require_reference(CASE, "c07_quality_geometry.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let freqs: Vec<f64> = serde_json::from_value(ref_json["freqs_hz"].clone()).unwrap();
    let spl: Vec<f64> = serde_json::from_value(ref_json["spl_db"].clone()).unwrap();
    let noise: Vec<f64> =
        serde_json::from_value(ref_json["calibration"]["noise_floor_db"].clone()).unwrap();
    let coherence: Vec<f64> = serde_json::from_value(ref_json["coherence_vals"].clone()).unwrap();
    assert_eq!(freqs.len(), 3, "{CASE}: expected 3 grid points");
    assert_eq!(spl.len(), freqs.len());
    assert_eq!(noise.len(), freqs.len());
    assert_eq!(coherence.len(), freqs.len());

    let curve = Curve {
        freq: Array1::from_vec(freqs.clone()),
        spl: Array1::from_vec(spl.clone()),
        coherence: Some(Array1::from_vec(coherence)),
        noise_floor_db: Some(Array1::from_vec(noise)),
        ..Default::default()
    };
    let report = assess_measurement_quality(&curve);
    let exp_snr: f64 = serde_json::from_value(ref_json["median_snr_db"].clone()).unwrap();
    let exp_min: f64 = serde_json::from_value(ref_json["min_coherence"].clone()).unwrap();
    let exp_med: f64 = serde_json::from_value(ref_json["median_coherence"].clone()).unwrap();
    let got_snr = report.median_snr_db.expect("median SNR must be present");
    let got_min = report.min_coherence.expect("min coherence must be present");
    let got_med = report
        .median_coherence
        .expect("median coherence must be present");
    assert!(
        abs_err(got_snr, exp_snr) <= TOL,
        "{CASE}: median SNR {got_snr} != {exp_snr}"
    );
    assert!(
        abs_err(got_min, exp_min) <= TOL,
        "{CASE}: min coherence {got_min} != {exp_min}"
    );
    assert!(
        abs_err(got_med, exp_med) <= TOL,
        "{CASE}: median coherence {got_med} != {exp_med}"
    );
    assert_eq!(
        report.quality,
        MeasurementQuality::Good,
        "{CASE}: expected Good quality"
    );
    assert_eq!(
        ref_json["quality"], "good",
        "{CASE}: golden quality label mismatch"
    );
    let exp_scale: f64 =
        serde_json::from_value(ref_json["correction_depth_scale"].clone()).unwrap();
    assert!(
        abs_err(report.correction_depth_scale, exp_scale) <= TOL,
        "{CASE}: depth scale mismatch"
    );
    assert!(
        report.advisories.is_empty(),
        "{CASE}: unexpected advisories"
    );

    // Multi-seat population variance on the identical grid (never zip
    // unequal grids: every seat reuses the golden frequency axis).
    let seats: Vec<Vec<f64>> = serde_json::from_value(ref_json["seats_spl_db"].clone()).unwrap();
    assert_eq!(seats.len(), 3, "{CASE}: expected 3 seats");
    assert_eq!(seats[0], spl, "{CASE}: seat 0 must equal the main curve");
    let curves: Vec<Curve> = seats
        .iter()
        .map(|seat| Curve {
            freq: Array1::from_vec(freqs.clone()),
            spl: Array1::from_vec(seat.clone()),
            coherence: curve.coherence.clone(),
            noise_floor_db: curve.noise_floor_db.clone(),
            ..Default::default()
        })
        .collect();
    let multi = assess_multiple_measurement_quality(&curves);
    let exp_mean: f64 = serde_json::from_value(ref_json["mean_seat_variance_db"].clone()).unwrap();
    let exp_max: f64 = serde_json::from_value(ref_json["max_seat_variance_db"].clone()).unwrap();
    let got_mean = multi.mean_seat_variance_db.expect("mean variance present");
    let got_max = multi.max_seat_variance_db.expect("max variance present");
    assert!(
        abs_err(got_mean, exp_mean) <= TOL,
        "{CASE}: mean seat variance {got_mean} != {exp_mean}"
    );
    assert!(
        abs_err(got_max, exp_max) <= TOL,
        "{CASE}: max seat variance {got_max} != {exp_max}"
    );
    assert_eq!(multi.quality, MeasurementQuality::Good);

    // Direct/reflected geometry and gate bound.
    let direct: f64 = serde_json::from_value(ref_json["direct_path_m"].clone()).unwrap();
    let refl: f64 = serde_json::from_value(ref_json["first_reflection_path_m"].clone()).unwrap();
    let speed: f64 =
        serde_json::from_value(ref_json["calibration"]["sound_speed_m_s"].clone()).unwrap();
    let gate: f64 = serde_json::from_value(ref_json["gate_s"].clone()).unwrap();
    let cycles: f64 = serde_json::from_value(ref_json["cycles_required"].clone()).unwrap();
    let exp_interval: f64 =
        serde_json::from_value(ref_json["reflection_free_interval_s"].clone()).unwrap();
    let exp_lower: f64 = serde_json::from_value(ref_json["valid_lower_bound_hz"].clone()).unwrap();
    let got_interval = reflection_free_interval_s(direct, refl, speed).expect("interval known");
    let got_lower = valid_lower_bound_hz(gate, cycles).expect("lower bound known");
    assert!(
        abs_err(got_interval, exp_interval) <= 1e-15,
        "{CASE}: interval {got_interval} != {exp_interval}"
    );
    assert!(
        abs_err(got_lower, exp_lower) <= TOL,
        "{CASE}: lower bound {got_lower} != {exp_lower}"
    );
    let facts = DirectSoundCaptureFacts {
        gate_s: Some(gate),
        direct_path_m: Some(direct),
        first_reflection_path_m: Some(refl),
        sound_speed_m_s: speed,
        ..Default::default()
    };
    assert_eq!(facts.gate_fits_interval(), Some(true));
    assert_eq!(ref_json["gate_fits_interval"], true);

    // Timing-derived coherence limit: 40 us supports 625 Hz (9 degrees)
    // but not 626 Hz. This pins the 360*f*dt phase formula boundary.
    let exp_max_hz: f64 = serde_json::from_value(ref_json["max_coherent_hz"].clone()).unwrap();
    assert!(
        abs_err(exp_max_hz, 625.0) <= TOL,
        "{CASE}: golden max coherent frequency must be 625 Hz"
    );
    let exp_phase: f64 =
        serde_json::from_value(ref_json["phase_uncertainty_deg_at_bound"].clone()).unwrap();
    assert!(
        abs_err(exp_phase, 9.0) <= TOL,
        "{CASE}: phase uncertainty at the bound must be 9 degrees"
    );
    let evidence = provenance_takes(40.0);
    assert!(evidence.coherent_reference_at_frequency(2, 625.0).is_ok());
    assert!(evidence.coherent_reference_at_frequency(2, 626.0).is_err());

    let worst = [abs_err(got_snr, exp_snr), abs_err(got_min, exp_min)]
        .into_iter()
        .fold(0.0f64, f64::max)
        .max(abs_err(got_med, exp_med))
        .max(abs_err(got_mean, exp_mean))
        .max(abs_err(got_max, exp_max));
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: worst,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
