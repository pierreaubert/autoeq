//! Wolfram cross-check: pressure-calibrated Welch noise with
//! band SPL (RQ03).
//!
//! Oracle: `wolfram/rq03_welch_noise_spl.wls` (independent periodic
//! Hann frames, half hops with end-anchored final frame, per-frame
//! mean removal, unnormalized direct-sum DFT, one-sided power with
//! the noise-power normalization, flat Pa/sample calibration, and
//! bin-summed octave SPL re 20 µPa; never calls Rust code). The test
//! calls `roomeq_quality::calibrated_capture_noise` on a bin-centered
//! tone and checks frame anchoring, the window/noise-power
//! convention (not coherent gain), Pa²/Hz scaling, and the octave
//! support rule. Tolerance 1e-9 relative on PSD (N), 1e-6 dB
//! absolute on SPL (Q).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_quality::{
    CapturedNoiseSettings, NoisePressureCalibration, ViewProvenance, calibrated_capture_noise,
};
use std::f64::consts::PI;

const CASE: &str = "rq03_welch_noise_spl";
const CASE_ID: &str = "autoeq-qa.rq03-welch-noise-spl.v1";
const TOL_PSD: f64 = 1e-9;
const TOL_SPL: f64 = 1e-6;

#[test]
fn wolfram_rq03_welch_noise_spl() {
    let ref_json = require_reference(CASE, "rq03_welch_noise_spl.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let frame: usize = serde_json::from_value(ref_json["frame_samples"].clone()).unwrap();
    let want_starts: Vec<usize> = serde_json::from_value(ref_json["frame_starts"].clone()).unwrap();
    let want_spacing: f64 = serde_json::from_value(ref_json["bin_spacing_hz"].clone()).unwrap();
    let tone: serde_json::Value = ref_json["tone"].clone();
    let tone_hz: f64 = serde_json::from_value(tone["freq_hz"].clone()).unwrap();
    let tone_amp: f64 = serde_json::from_value(tone["amplitude"].clone()).unwrap();
    let record_len: usize = serde_json::from_value(tone["record_samples"].clone()).unwrap();
    let pa_per_sample: f64 =
        serde_json::from_value(ref_json["pascals_per_sample"].clone()).unwrap();
    let band: [f64; 2] = serde_json::from_value(ref_json["valid_band_hz"].clone()).unwrap();
    let spot_bins: Vec<usize> = serde_json::from_value(ref_json["spot_bins"].clone()).unwrap();
    let spot_freqs: Vec<f64> =
        serde_json::from_value(ref_json["spot_bin_freqs_hz"].clone()).unwrap();
    let spot_psd: Vec<f64> = serde_json::from_value(ref_json["spot_pressure_psd"].clone()).unwrap();
    let want_spl: f64 = serde_json::from_value(ref_json["octave_2k_spl_db"].clone()).unwrap();

    let samples: Vec<f64> = (0..record_len)
        .map(|n| tone_amp * (2.0 * PI * tone_hz * n as f64 / rate).cos())
        .collect();
    assert!(samples.iter().all(|v| v.is_finite() && v.abs() < 1.0));

    let calibration = NoisePressureCalibration {
        calibration_id: "numeric-test".into(),
        pascals_per_sample: pa_per_sample,
        microphone_id: "synthetic".into(),
        orientation: "omnidirectional model".into(),
        acquisition_gain_id: "fixed-test-gain".into(),
        reference_conditions: "numeric oracle, not a real calibrator".into(),
        response_freqs_hz: vec![1.0, rate / 2.0],
        response_correction_db: vec![0.0, 0.0],
        uncertainty_db: None,
        self_noise_note: "not characterized in this synthetic test".into(),
    };
    let prov = ViewProvenance {
        measurement_ids: vec!["synthetic-noise".into()],
        graph_identity: "graph".into(),
        sample_rate_hz: rate,
        calibration: calibration.calibration_id.clone(),
        processing_chain: "test".into(),
    };
    let view = calibrated_capture_noise(
        &samples,
        CapturedNoiseSettings {
            frame_samples: frame,
            valid_band_hz: band,
        },
        calibration,
        prov,
        "settings".into(),
    )
    .unwrap();

    // Exact estimator bookkeeping: spacing, record, frame anchoring.
    assert!(
        (view.bin_spacing_hz - want_spacing).abs() == 0.0,
        "{CASE}: bin spacing"
    );
    assert_eq!(view.frame_starts, want_starts, "{CASE}: frame starts");
    assert_eq!(view.record_samples, record_len);
    assert!(
        (view.duration_seconds - record_len as f64 / rate).abs() == 0.0,
        "{CASE}: duration"
    );

    // Calibrated Pa²/Hz at the tone bins. The view retains only
    // bins inside the declared band (bin 0 and sub-band bins are
    // dropped), so each bin is located by its exact frequency.
    // Null bins (61, 62, 66, 67) sit ~30 orders below the tone bins;
    // a relative error there divides numerical zero by numerical zero.
    // Normalize by the peak spot PSD so null bins compare absolutely.
    let peak_psd = spot_psd.iter().cloned().fold(0.0f64, f64::max);
    assert!(peak_psd > 0.0, "{CASE}: oracle peak PSD must be positive");
    let mut max_psd_err = 0.0f64;
    for ((bin, want_f), want_p) in spot_bins.iter().zip(&spot_freqs).zip(&spot_psd) {
        let index = view
            .bin_freqs_hz
            .iter()
            .position(|f| f == want_f)
            .unwrap_or_else(|| panic!("{CASE}: bin {bin} frequency missing from view"));
        let got_f = view.bin_freqs_hz[index];
        assert!(
            (got_f - want_f).abs() == 0.0,
            "{CASE}: bin {bin} frequency {got_f} vs {want_f}"
        );
        let got_p = view.pressure_psd_pa2_per_hz[index];
        // Far-sidelobe bins are analytically zero for the Hann
        // window, so both sides report only FFT roundoff there; use
        // an absolute floor far below any physical content instead
        // of comparing two roundoff noises relatively.
        let abs_err = (got_p - want_p).abs();
        let err = abs_err / peak_psd;
        assert!(
            err <= TOL_PSD || abs_err <= 1e-18,
            "{CASE}: bin {bin} PSD {got_p:.12e} vs {want_p:.12e} err={err:.3e}"
        );
        max_psd_err = max_psd_err.max(err.min(1.0));
    }

    // Octave-band SPL and the display-support rule.
    let pos = view
        .spectrum
        .freqs
        .iter()
        .position(|c| *c == 2000.0)
        .expect("2 kHz octave must be available");
    let spl = view.spectrum.noise_spl_db[pos];
    assert!(
        (spl - want_spl).abs() <= TOL_SPL,
        "{CASE}: 2 kHz SPL {spl:.12e} vs {want_spl:.12e}"
    );
    let missing: String = serde_json::from_value(ref_json["unavailable_octave"].clone()).unwrap();
    assert!(
        view.unavailable_bands.contains_key(&missing),
        "{CASE}: octave {missing} must be unavailable"
    );

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: max_psd_err,
        max_abs_error: (spl - want_spl).abs(),
        tolerance: TOL_PSD,
        tolerance_kind: "rel".to_string(),
        provenance: provenance(),
    });
}
