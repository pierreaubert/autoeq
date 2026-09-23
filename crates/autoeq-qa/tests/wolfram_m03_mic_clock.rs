//! Wolfram cross-check: mic phase calibration and clock fit (M03).
//!
//! Oracle: `wolfram/m03_mic_clock.wls` (independent linear interpolation
//! of the calibration with the declared minus sign, closed-form affine
//! least squares for the clock markers, and the identity time-map).
//! Tolerance 1e-9 absolute (dB / degrees / seconds / ppm,
//! catalogue class A/N/X).

use autoeq_measurements::{
    ClockCorrectionMethod, Curve, MicPhaseCalibration, ReferenceMarker, apply_clock_correction,
    fit_clock_markers,
};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use ndarray::Array1;

const CASE: &str = "m03_mic_clock";
const CASE_ID: &str = "autoeq-qa.m03-mic-clock.v1";
const TOL: f64 = 1e-9;

fn vec_f64(v: &serde_json::Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}

#[test]
fn wolfram_m03_mic_clock() {
    let ref_json = require_reference(CASE, "m03_mic_clock.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let mut max_err = 0.0f64;

    // --- (a) mic calibration: corrected = measured - mic (minus sign). ---
    let cal = MicPhaseCalibration {
        freq: Array1::from_vec(vec_f64(&ref_json["cal_freqs_hz"])),
        mag_db: Array1::from_vec(vec_f64(&ref_json["cal_mag_db"])),
        phase_deg: Array1::from_vec(vec_f64(&ref_json["cal_phase_deg"])),
        coherence: Array1::from_vec(vec_f64(&ref_json["cal_coherence"])),
    };
    let mut curve = Curve {
        freq: Array1::from_vec(vec_f64(&ref_json["freqs_hz"])),
        spl: Array1::from_vec(vec_f64(&ref_json["curve_spl_db"])),
        phase: Some(Array1::from_vec(vec_f64(&ref_json["curve_phase_deg"]))),
        coherence: Some(Array1::from_vec(vec_f64(&ref_json["curve_coherence"]))),
        ..Default::default()
    };
    cal.apply_to_curve(&mut curve).unwrap();
    let spl_ref = vec_f64(&ref_json["corrected_spl_db"]);
    let ph_ref = vec_f64(&ref_json["corrected_phase_deg"]);
    let coh_ref = vec_f64(&ref_json["corrected_coherence"]);
    let phase = curve
        .phase
        .as_ref()
        .expect("{CASE}: corrected curve must keep phase");
    let coh = curve
        .coherence
        .as_ref()
        .expect("{CASE}: corrected curve must keep coherence");
    for (i, ((got, want), (gotp, wantp))) in curve
        .spl
        .iter()
        .zip(spl_ref.iter())
        .zip(phase.iter().zip(ph_ref.iter()))
        .enumerate()
    {
        let err = (got - want).abs();
        let errp = (gotp - wantp).abs();
        assert!(
            err <= TOL,
            "{CASE}: spl[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        assert!(
            errp <= TOL,
            "{CASE}: phase[{i}]: rust={gotp:.12e} expected={wantp:.12e} abs_err={errp:.3e}"
        );
        max_err = max_err.max(err).max(errp);
    }
    for (i, (got, want)) in coh.iter().zip(coh_ref.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: coherence[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // --- (b) affine clock-marker least squares. ---
    let refs = vec_f64(&ref_json["markers_reference_s"]);
    let obss = vec_f64(&ref_json["markers_observed_s"]);
    let markers: Vec<ReferenceMarker> = refs
        .iter()
        .zip(obss.iter())
        .map(|(r, o)| ReferenceMarker {
            reference_time_s: *r,
            observed_time_s: *o,
        })
        .collect();
    let fit = fit_clock_markers(&markers).unwrap();
    assert_eq!(fit.n_markers, 3, "{CASE}: expected 3 markers");
    let rate = fit.rate_ppm.expect("{CASE}: drift must be observable");
    for (name, got, want) in [
        (
            "offset_s",
            fit.offset_s,
            ref_json["fit_offset_s"].as_f64().unwrap(),
        ),
        ("rate_ppm", rate, ref_json["fit_rate_ppm"].as_f64().unwrap()),
        (
            "rms_residual_s",
            fit.rms_residual_s,
            ref_json["fit_rms_residual_s"].as_f64().unwrap(),
        ),
        (
            "accumulated_20s",
            fit.accumulated_offset_s(20.0).expect("rate known"),
            ref_json["accumulated_offset_20s"].as_f64().unwrap(),
        ),
    ] {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: {name}: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }
    let resid_ref = vec_f64(&ref_json["fit_residuals_s"]);
    for (i, (got, want)) in fit.residuals_s.iter().zip(resid_ref.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: residual[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
    }

    // --- (c) identity time-map reproduces the samples. ---
    let samples = vec_f64(&ref_json["identity_samples"]);
    let corrected_ref = vec_f64(&ref_json["identity_corrected"]);
    let zero_fit = fit_clock_markers(&[
        ReferenceMarker {
            reference_time_s: 0.0,
            observed_time_s: 0.0,
        },
        ReferenceMarker {
            reference_time_s: 1.0,
            observed_time_s: 1.0,
        },
    ])
    .unwrap();
    let artifact = apply_clock_correction(
        &samples,
        48000.0,
        &zero_fit,
        ClockCorrectionMethod::LinearResample,
    )
    .unwrap();
    for (i, (got, want)) in artifact
        .corrected_samples
        .iter()
        .zip(corrected_ref.iter())
        .enumerate()
    {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{CASE}: identity[{i}]: rust={got:.12e} expected={want:.12e} abs_err={err:.3e}"
        );
        max_err = max_err.max(err);
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
