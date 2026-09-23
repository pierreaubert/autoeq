//! Negative controls for the enga group (RE01, RE02, RE06, RE21, RE22):
//! each control loads an engine-blessed golden, applies one classic
//! defect to a copy, and asserts the resulting error exceeds the case
//! tolerance. A control that cannot fail is worthless; these must keep
//! failing.

use autoeq_qa::{complex_rel_error, golden_dir, rel_error};
use num_complex::Complex64;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn pairs(value: &serde_json::Value) -> Vec<Complex64> {
    let raw: Vec<[f64; 2]> = serde_json::from_value(value.clone()).unwrap();
    raw.iter().map(|p| Complex64::new(p[0], p[1])).collect()
}

fn worst_complex(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| complex_rel_error(*x, *y))
        .fold(0.0f64, f64::max)
}

/// RE01 replay: a power-vs-amplitude dB slip (10^(dB/10) instead of
/// 10^(dB/20)) scales every bin by ~2x and must blow the 1e-9 tolerance.
#[test]
fn power_db_factor_fails_re01_gain_tolerance() {
    let payload = load_golden("re01_dsp_gain_delay");
    let reference = pairs(&payload["response_re_im"]);
    let gain_db: f64 = serde_json::from_value(payload["gain_db"].clone()).unwrap();
    let wrong: Vec<Complex64> = reference
        .iter()
        .map(|h| h * 10.0f64.powf(gain_db / 10.0) / 10.0f64.powf(gain_db / 20.0))
        .collect();
    let worst = worst_complex(&wrong, &reference);
    assert!(
        worst > 1e-9,
        "power-vs-amplitude dB slip must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RE01 replay: a delay sign flip (complex conjugation of the phase
/// term) must blow the 1e-9 tolerance wherever phase is nonzero.
#[test]
fn delay_sign_flip_fails_re01_phase_tolerance() {
    let payload = load_golden("re01_dsp_gain_delay");
    let reference = pairs(&payload["response_re_im"]);
    let wrong: Vec<Complex64> = reference.iter().map(|h| h.conj()).collect();
    let worst = worst_complex(&wrong, &reference);
    assert!(
        worst > 1e-9,
        "delay sign flip must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RE02 shelves: a 10-vs-20 log slip (10*log10 instead of 20*log10)
/// halves every correction and must blow the 1e-9 dB tolerance.
#[test]
fn ten_log_factor_fails_re02_shelf_tolerance() {
    let payload = load_golden("re02_preference_shelves");
    let reference: Vec<f64> = serde_json::from_value(payload["spl_db"].clone()).unwrap();
    let level: f64 = serde_json::from_value(payload["level_db"].clone()).unwrap();
    let wrong: Vec<f64> = reference
        .iter()
        .map(|s| level + (s - level) / 2.0)
        .collect();
    let worst = wrong
        .iter()
        .zip(reference.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "10-vs-20 log slip must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

/// RE02 shelves: negating the bass shelf correction must move the low
/// end by far more than 1e-9 dB.
#[test]
fn negated_bass_shelf_fails_re02_low_end() {
    let payload = load_golden("re02_preference_shelves");
    let reference: Vec<f64> = serde_json::from_value(payload["spl_db"].clone()).unwrap();
    let level: f64 = serde_json::from_value(payload["level_db"].clone()).unwrap();
    let worst = reference
        .iter()
        .map(|s| (2.0 * level - s - s).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "negated shelf must move the curve by far more than 1e-9 dB, got {worst:.3e}"
    );
}

/// RE06 delay: an off-by-one shift (47 instead of 48 leading zeros)
/// must change the stored tap vector.
#[test]
fn off_by_one_shift_fails_re06_stored_taps() {
    let payload = load_golden("re06_fir_integer_delay");
    let stored: Vec<f64> = serde_json::from_value(payload["stored_taps"].clone()).unwrap();
    let input: Vec<f64> = serde_json::from_value(payload["input_taps"].clone()).unwrap();
    let mut wrong = vec![0.0; 47];
    wrong.extend_from_slice(&input);
    assert_ne!(
        wrong.len(),
        stored.len(),
        "off-by-one shift must change the stored length"
    );
}

/// RE06 delay: applying the delay twice (96 leading zeros) must change
/// both the stored length and the transfer phase.
#[test]
fn double_delay_fails_re06_transfer() {
    let payload = load_golden("re06_fir_integer_delay");
    let stored: Vec<f64> = serde_json::from_value(payload["stored_taps"].clone()).unwrap();
    let input: Vec<f64> = serde_json::from_value(payload["input_taps"].clone()).unwrap();
    let mut wrong = vec![0.0; 96];
    wrong.extend_from_slice(&input);
    assert_ne!(
        wrong.len(),
        stored.len(),
        "double delay must change the stored length"
    );
    let freqs: Vec<f64> = serde_json::from_value(payload["freqs_hz"].clone()).unwrap();
    let reference = pairs(&payload["response_re_im"]);
    let dtft = |taps: &[f64], f: f64| {
        taps.iter()
            .enumerate()
            .fold(Complex64::new(0.0, 0.0), |acc, (n, h)| {
                let w = -2.0 * std::f64::consts::PI * f * n as f64 / 48000.0;
                acc + Complex64::from_polar(*h, w)
            })
    };
    let wrong_h: Vec<Complex64> = freqs.iter().map(|f| dtft(&wrong, *f)).collect();
    let worst = worst_complex(&wrong_h, &reference);
    assert!(
        worst > 1e-9,
        "double delay must exceed the 1e-9 transfer tolerance, got {worst:.3e}"
    );
}

/// RE21 correction: swapping initial and final (sign flip) must blow
/// the 1e-12 dB tolerance.
#[test]
fn swapped_curves_fail_re21_correction_tolerance() {
    let payload = load_golden("re21_eq_response_correction");
    let reference: Vec<f64> = serde_json::from_value(payload["correction_spl_db"].clone()).unwrap();
    let worst = reference
        .iter()
        .map(|c| rel_error(-c, *c))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "swapped initial/final must exceed the 1e-12 tolerance, got {worst:.3e}"
    );
}

/// RE21 correction: zipping against a truncated grid (dropped last bin)
/// must be caught by the length contract.
#[test]
fn truncated_grid_fails_re21_length_contract() {
    let payload = load_golden("re21_eq_response_correction");
    let freqs: Vec<f64> = serde_json::from_value(payload["freqs_hz"].clone()).unwrap();
    let correction: Vec<f64> =
        serde_json::from_value(payload["correction_spl_db"].clone()).unwrap();
    let truncated = &freqs[..freqs.len() - 1];
    assert_ne!(
        truncated.len(),
        correction.len(),
        "off-by-one grid must change the grid length"
    );
}

/// RE22 limiter: a ms-vs-samples slip (5 samples instead of 240 at
/// 48 kHz) must be caught by the exact latency contract.
#[test]
fn ms_sample_slip_fails_re22_latency_contract() {
    let payload = load_golden("re22_limiter_gain_envelope");
    let latencies: Vec<usize> = serde_json::from_value(payload["latency_samples"].clone()).unwrap();
    assert_eq!(latencies[0], 240);
    assert_ne!(
        5usize, latencies[0],
        "milliseconds must never pass as samples"
    );
}

/// RE22 limiter: a missing ceiling clamp (0.0 dBFS instead of -1.0)
/// must be caught by the exact ceiling contract.
#[test]
fn missing_clamp_fails_re22_ceiling_contract() {
    let payload = load_golden("re22_limiter_gain_envelope");
    let ceilings: Vec<f64> = serde_json::from_value(payload["ceilings_dbfs"].clone()).unwrap();
    assert_eq!(ceilings[1], -1.0);
    assert!(
        (0.0 - ceilings[1]).abs() > 1e-12,
        "unclamped ceiling must miss the -1.0 dBFS contract"
    );
}

/// RE22 envelope: reporting only the in-band peak (5.5 dB) while the
/// full-span peak (6.2 dB) breaches the cap must flip the flag.
#[test]
fn band_only_peak_fails_re22_envelope_flag() {
    let payload = load_golden("re22_limiter_gain_envelope");
    let within: bool = serde_json::from_value(payload["within_envelope"].clone()).unwrap();
    let band_peak: f64 = serde_json::from_value(payload["band_peak_gain_db"].clone()).unwrap();
    let max_gain: f64 = serde_json::from_value(payload["max_gain_db"].clone()).unwrap();
    assert!(!within, "6.2 dB span peak must breach the 6.0 dB cap");
    let band_only_within = band_peak <= max_gain;
    assert!(
        band_only_within != within,
        "a band-only checker would wrongly read within-envelope; the full-span flag must differ"
    );
}
