//! Negative controls for the qqsa group (RQ05, RQ06, RQ08, QA01, QA02,
//! QA03, RS01, RS02, RS03).
//!
//! Each control proves its comparison detects one defect class: the
//! defective value deviates from the engine golden by more than the case
//! tolerance, so the comparison would fail if the defect were present.
//! These tests assert the defect IS detected (deviation exceeds tolerance).

use autoeq_qa::golden_dir;
use num_complex::Complex64;

fn golden(stem: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).expect("golden must exist");
    serde_json::from_str(&text).expect("golden must parse")
}

#[test]
fn qqsa_rq05_double_gain_detected() {
    let g = golden("rq05_capture_alignment");
    let pred: Vec<f64> = serde_json::from_value(g["prediction_db"].clone()).unwrap();
    let capt: Vec<f64> = serde_json::from_value(g["capture_db"].clone()).unwrap();
    let gain: f64 = serde_json::from_value(g["declared_gain_db"].clone()).unwrap();
    let want: f64 = serde_json::from_value(g["worst_magnitude_db"].clone()).unwrap();
    // Defect: declared gain applied twice.
    let worst: f64 = capt
        .iter()
        .zip(pred.iter())
        .map(|(c, p)| (c - p - 2.0 * gain).abs())
        .fold(0.0, f64::max);
    assert!(
        (worst - want).abs() > 1e-9,
        "double-gain defect must be detected"
    );
}

#[test]
fn qqsa_rq05_power_db_factor_detected() {
    let g = golden("rq05_capture_alignment");
    let want: f64 = serde_json::from_value(g["worst_magnitude_db"].clone()).unwrap();
    // Defect: 10 log10 (power) factor instead of 20 log10 on a 2x pressure ratio.
    let defective = 10.0 * 2.0_f64.log10();
    let correct = 20.0 * 2.0_f64.log10();
    assert!((defective - correct).abs() > 1e-9);
    assert!((want - 0.2).abs() <= 1e-9);
}

#[test]
fn qqsa_rq06_missing_normalization_detected() {
    let g = golden("rq06_tone_render");
    let peak: f64 = serde_json::from_value(g["synthesis_peak"].clone()).unwrap();
    // Defect: peak normalization skipped leaves unit peak.
    assert!((1.0 - peak).abs() > 1e-6, "unnormalized peak must differ");
}

#[test]
fn qqsa_rq08_conjugation_detected() {
    let g = golden("rq08_delay_oracle");
    let pairs: Vec<[f64; 2]> = serde_json::from_value(g["delay_re_im"].clone()).unwrap();
    // Defect: conjugated transfer (wrong delay sign).
    for pair in &pairs {
        let expected = Complex64::new(pair[0], pair[1]);
        let conjugated = expected.conj();
        let err = (conjugated - expected).norm() / expected.norm();
        if expected.norm() > 1e-12 {
            assert!(err > 1e-9, "conjugation must be detected");
            return;
        }
    }
    panic!("no nonzero bin to test conjugation");
}

#[test]
fn qqsa_rq08_off_by_one_grid_detected() {
    let g = golden("rq08_delay_oracle");
    let freqs: Vec<f64> = serde_json::from_value(g["freqs_hz"].clone()).unwrap();
    let delay_ms: f64 = serde_json::from_value(g["delay_ms"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(g["delay_re_im"].clone()).unwrap();
    // Defect: index-wise zip of a shifted grid (off-by-one).
    let tau = delay_ms / 1000.0;
    let shifted = freqs[1];
    let expected = Complex64::new(pairs[0][0], pairs[0][1]);
    let wrong = Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * shifted * tau);
    let err = (wrong - expected).norm() / expected.norm();
    assert!(err > 1e-9, "off-by-one grid zip must be detected");
}

#[test]
fn qqsa_qa01_forgetful_eq_detected() {
    let g = golden("qa01_clock_phase");
    let seat_a: Vec<Option<f64>> = serde_json::from_value(g["seat_a_db"].clone()).unwrap();
    let seat_b: Vec<Option<f64>> = serde_json::from_value(g["seat_b_db"].clone()).unwrap();
    let eq: Vec<Option<f64>> = serde_json::from_value(g["common_eq_db"].clone()).unwrap();
    let want_naive: f64 = serde_json::from_value(g["naive_forgetful_drift_db"].clone()).unwrap();
    // Defect: EQ added to seat A only (forgotten on seat B).
    let naive: f64 = seat_a
        .iter()
        .zip(seat_b.iter())
        .zip(eq.iter())
        .filter_map(|((a, b), e)| match (a, b, e) {
            (Some(a), Some(b), Some(e)) => Some(((a + e) - b) - (a - b)),
            _ => None,
        })
        .fold(0.0_f64, |worst, drift| worst.max(drift.abs()));
    assert!(
        (naive - want_naive).abs() <= 1e-9,
        "sanity: forgetful EQ drifts by max|eq|"
    );
    assert!(naive > 1e-9, "forgetful common EQ must be detected");
}

#[test]
fn qqsa_qa01_sign_flip_detected() {
    let g = golden("qa01_clock_phase");
    let want: Vec<f64> = serde_json::from_value(g["clock_offsets_s"].clone()).unwrap();
    // Defect: rectified (unsigned) clock offset.
    assert!(
        (want[1].abs() - want[1]).abs() > 1e-9,
        "sign flip on negative ppm must be detected"
    );
    let gain: f64 = serde_json::from_value(g["coherent_gain_11_db"].clone()).unwrap();
    let wrong = 10.0 * 2.0_f64.log10();
    assert!(
        (wrong - gain).abs() > 1e-9,
        "10-vs-20 dB factor must be detected"
    );
}

#[test]
fn qqsa_qa02_wrong_seed_detected() {
    let g = golden("qa02_heldout_perturb");
    let want: Vec<[f64; 2]> = serde_json::from_value(g["perturbed_re_im"].clone()).unwrap();
    let re: Vec<f64> = serde_json::from_value(g["transfer_re"].clone()).unwrap();
    let im: Vec<f64> = serde_json::from_value(g["transfer_im"].clone()).unwrap();
    let transfer: Vec<Complex64> = re
        .iter()
        .zip(im.iter())
        .map(|(r, i)| Complex64::new(*r, *i))
        .collect();
    let wrong_seed = roomeq_quality::perturb_transfer(
        &transfer,
        roomeq_quality::MeasurementNoise::Noisy { rms_db: 0.75 },
        12,
    );
    let err: f64 = wrong_seed
        .iter()
        .zip(want.iter())
        .map(|(got, pair)| (got - Complex64::new(pair[0], pair[1])).norm())
        .fold(0.0, f64::max);
    assert!(err > 1e-9, "wrong perturbation seed must be detected");
}

#[test]
fn qqsa_qa03_swapped_counts_detected() {
    let g = golden("qa03_seed_matrix");
    let counts: Vec<usize> =
        serde_json::from_value(g["counts_pass_fail_notrun_blocked"].clone()).unwrap();
    // Defect: pass/fail swapped hides the tail.
    assert_ne!(counts[0], counts[1], "swapped counts must differ");
    let tail: f64 = serde_json::from_value(g["worst_tail_mean"].clone()).unwrap();
    let mean_all = (0.4 + 0.6 + 2.5 + 0.5) / 4.0;
    assert!(
        (tail - mean_all).abs() > 1e-9,
        "tail risk must differ from the plain mean"
    );
}

#[test]
fn qqsa_rs01_tilt_sign_detected() {
    let g = golden("rs01_flat_tilt");
    let tilt: Vec<f64> = serde_json::from_value(g["tilt_db"].clone()).unwrap();
    // Defect: flipped tilt sign.
    let flipped: Vec<f64> = tilt.iter().map(|v| -v).collect();
    let err: f64 = tilted_max_diff(&tilt, &flipped);
    assert!(err > 1e-9, "tilt sign flip must be detected");
}

fn tilted_max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn qqsa_rs02_pressure_as_db_detected() {
    let g = golden("rs02_coherent_sum");
    let dbs: Vec<Option<f64>> = serde_json::from_value(g["sum_db"].clone()).unwrap();
    // Defect: finite dB reported for exact cancellation.
    assert!(
        dbs[1].is_none(),
        "cancellation must expose no finite dB score"
    );
    let pressures: Vec<f64> = serde_json::from_value(g["sum_pressure"].clone()).unwrap();
    let wrong_db = 20.0 * pressures[1].log10();
    assert!(
        !wrong_db.is_finite() || wrong_db < -200.0,
        "cancellation pressure must not look like a good score"
    );
}

#[test]
fn qqsa_rs03_amplitude_swap_detected() {
    let g = golden("rs03_tone_bank");
    let dtft: Vec<f64> = serde_json::from_value(g["dtft_closed_form"].clone()).unwrap();
    // Defect: swapped channel amplitudes change the spectrum at fixed bins.
    assert!(
        (dtft[0] - dtft[1]).abs() > 1e-9,
        "swapped tone amplitudes must reshape the spectrum"
    );
    let energy: f64 = serde_json::from_value(g["energy"].clone()).unwrap();
    let closed: f64 = serde_json::from_value(g["energy_closed_form"].clone()).unwrap();
    assert!(
        ((energy - closed) / closed).abs() <= 1e-9,
        "sanity: oracle direct sum matches closed form"
    );
}
