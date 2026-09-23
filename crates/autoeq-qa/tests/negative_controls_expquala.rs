//! Negative controls for the export/CLI/quality cross-validation
//! group (EX01, EX02, EX03, RC01, RC02, RQ01, RQ02, RQ03, RQ04).
//!
//! Each control loads an engine-blessed golden, applies one classic
//! defect to a copy of the reference values, and asserts the
//! resulting error exceeds the case tolerance. A control that cannot
//! fail is worthless; these must keep failing.

use autoeq_qa::golden_dir;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn pairs_of(payload: &serde_json::Value, key: &str) -> Vec<[f64; 2]> {
    serde_json::from_value(payload[key].clone())
        .unwrap_or_else(|_| panic!("golden is missing `{key}`"))
}

fn vec_of(payload: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(payload[key].clone())
        .unwrap_or_else(|_| panic!("golden is missing `{key}`"))
}

fn complex_rel(pair: [f64; 2], want: [f64; 2]) -> f64 {
    let num = ((pair[0] - want[0]).powi(2) + (pair[1] - want[1]).powi(2)).sqrt();
    let den = (want[0].powi(2) + want[1].powi(2)).sqrt();
    if num == 0.0 {
        0.0
    } else if den == 0.0 {
        f64::INFINITY
    } else {
        num / den
    }
}

/// EX01: conjugating the exported transfer (a sign flip on every
/// imaginary part, as from a z vs z^-1 mix-up) must blow the 1e-9
/// relative tolerance wherever the response is not purely real.
#[test]
fn conjugated_transfer_fails_ex01_tolerance() {
    let payload = load_golden("ex01_apo_biquad_gain_delay");
    let pairs = pairs_of(&payload, "response_re_im");
    let worst = pairs
        .iter()
        .map(|p| complex_rel([p[0], -p[1]], *p))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "conjugated EX01 transfer must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// EX01: applying the preamp with a power (10log10) instead of an
/// amplitude (20log10) dB factor scales the whole transfer by
/// 10^(-3/20); the comparison must expose that factor.
#[test]
fn power_db_preamp_fails_ex01_tolerance() {
    let payload = load_golden("ex01_apo_biquad_gain_delay");
    let pairs = pairs_of(&payload, "response_re_im");
    let wrong = 10.0f64.powf(-3.0 / 20.0);
    let worst = pairs
        .iter()
        .map(|p| complex_rel([p[0] * wrong, p[1] * wrong], *p))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "power-dB EX01 preamp must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// EX02: padding is an exact integer; one sample less than the
/// fractional-stage requirement must not compare equal.
#[test]
fn off_by_one_padding_fails_ex02_exactness() {
    let payload = load_golden("ex02_delay_padding_matrix");
    let groups: Vec<serde_json::Value> = serde_json::from_value(payload["groups"].clone()).unwrap();
    let pad: usize = serde_json::from_value(groups[1]["pad_samples"].clone()).unwrap();
    assert_eq!(pad, 59);
    assert_ne!(
        pad, 58,
        "EX02 fractional padding 58 must not pass for the required 59"
    );
}

/// EX03: swapping the channel mapping (reading the integer-delay R
/// peak where the fractional L kernel sits) must blow the 1e-12 tap
/// tolerance.
#[test]
fn swapped_channels_fail_ex03_mapping() {
    let payload = load_golden("ex03_rendered_convolution_taps");
    let taps: Vec<f64> = serde_json::from_value(payload["left"]["taps"].clone()).unwrap();
    let err = (taps[5] - 1.0).abs();
    assert!(
        err > 1e-12,
        "swapped EX03 channels must exceed the 1e-12 tap tolerance, got {err:.3e}"
    );
}

/// RC01: normalizing PCM16 by 2^15 - 1 instead of 2^15 must break
/// exact decode identity.
#[test]
fn truncated_scale_fails_rc01_decode() {
    let payload = load_golden("rc01_wav_decode_dtft_calibration");
    let ints: Vec<i64> = serde_json::from_value(payload["int_samples"].clone()).unwrap();
    let want = vec_of(&payload, "expected_float16");
    let worst = ints
        .iter()
        .zip(&want)
        .map(|(s, w)| (*s as f64 / 32767.0 - w).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 0.0,
        "truncated RC01 scale must break exact decode, got {worst:.3e}"
    );
}

/// RC02: halving every route gain (a 6 dB realization slip) must blow
/// the 1e-9 relative tolerance on all four source/seat predictions.
#[test]
fn halved_route_gain_fails_rc02_tolerance() {
    let payload = load_golden("rc02_route_matrix_prediction");
    let predictions: Vec<serde_json::Value> =
        serde_json::from_value(payload["predictions"].clone()).unwrap();
    assert_eq!(predictions.len(), 4);
    for prediction in &predictions {
        let pairs = pairs_of(prediction, "sum_re_im");
        let worst = pairs
            .iter()
            .map(|p| complex_rel([p[0] * 0.5, p[1] * 0.5], *p))
            .fold(0.0f64, f64::max);
        assert!(
            worst > 1e-9,
            "halved RC02 route gain must exceed the 1e-9 tolerance, got {worst:.3e}"
        );
    }
}

/// RQ01: averaging the largest three instead of two values (an
/// off-by-one tail count) must break the 1e-12 CVaR agreement.
#[test]
fn tail_count_off_by_one_fails_rq01_cvar() {
    let payload = load_golden("rq01_seat_bin_cvar");
    let values = vec_of(&payload, "cvar_values");
    let want: f64 = serde_json::from_value(payload["cvar"].clone()).unwrap();
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let wrong: f64 = sorted[..3].iter().sum::<f64>() / 3.0;
    assert!(
        (wrong - want).abs() > 1e-12,
        "off-by-one RQ01 tail must exceed the 1e-12 tolerance"
    );
}

/// RQ01: reporting group delay with the wrong sign (phase-slope sign
/// flip) must blow the 1e-9 ms tolerance.
#[test]
fn gd_sign_flip_fails_rq01_tolerance() {
    let payload = load_golden("rq01_seat_bin_cvar");
    let want = vec_of(&payload, "group_delay_ms");
    let worst = want.iter().map(|g| (g - -g).abs()).fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "sign-flipped RQ01 group delay must exceed the 1e-9 ms tolerance"
    );
}

/// RQ02: integrating energy forward instead of reversed in time must
/// completely miss the Schroeder decay curve.
#[test]
fn forward_energy_fails_rq02_decay() {
    let payload = load_golden("rq02_step_schroeder_t60");
    let ir = vec_of(&payload, "ir");
    let want = vec_of(&payload, "decay_db");
    let total: f64 = ir.iter().map(|s| s * s).sum();
    let mut head = 0.0;
    let worst = ir
        .iter()
        .zip(&want)
        .map(|(s, w)| {
            head += s * s;
            (10.0 * (head / total).log10() - w).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "forward-energy RQ02 decay must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

/// RQ03: dropping the one-sided factor of 2 halves every octave
/// power, shifting SPL by 10log10(2) dB; the comparison must expose
/// that 3.01 dB error.
#[test]
fn single_sided_psd_fails_rq03_spl() {
    let payload = load_golden("rq03_welch_noise_spl");
    let want: f64 = serde_json::from_value(payload["octave_2k_spl_db"].clone()).unwrap();
    let err = (10.0 * (0.5_f64).log10()).abs();
    assert!(
        err > 1e-6,
        "single-sided RQ03 PSD must exceed the 1e-6 dB SPL tolerance, got {err:.3e}"
    );
    assert!(
        want.is_finite() && want > 80.0 && want < 100.0,
        "RQ03 octave SPL {want} must sit at the expected tone level"
    );
}

/// RQ03: a power-domain (10log10) vs amplitude-domain (20log10)
/// mix-up doubles every dB value and must be caught.
#[test]
fn power_db_factor_fails_rq03_spl() {
    let payload = load_golden("rq03_welch_noise_spl");
    let want: f64 = serde_json::from_value(payload["octave_2k_spl_db"].clone()).unwrap();
    assert!(
        want.abs() > 1e-6,
        "doubled RQ03 dB values must exceed the 1e-6 dB tolerance"
    );
}

/// RQ04: claiming coherent cancellation across independent inputs
/// (summing L and R complexly instead of bounding magnitudes) would
/// report 0 dB instead of 6.02 dB for the sub output.
#[test]
fn coherent_cancellation_fails_rq04_bound() {
    let payload = load_golden("rq04_headroom_drive_verdicts");
    let electrical: Vec<serde_json::Value> =
        serde_json::from_value(payload["electrical"].clone()).unwrap();
    let sub = electrical.iter().find(|e| e["output"] == "sub").unwrap();
    let atten: f64 = serde_json::from_value(sub["required_attenuation_db"].clone()).unwrap();
    assert!(
        (atten - 0.0).abs() > 1e-9,
        "coherent RQ04 cancellation must not pass the independent-input bound"
    );
    let coh = electrical.iter().find(|e| e["output"] == "coh").unwrap();
    assert_eq!(
        coh["peak_dbfs"],
        serde_json::Value::String("none".into()),
        "exact-zero RQ04 transfer must report no dBFS"
    );
}

/// RQ04: the equality-at-limit point pins the pass semantics: a
/// demand exactly at its limit passes, so any strictly-greater
/// demand at the same point must fail.
#[test]
fn equality_point_pins_rq04_verdict() {
    let payload = load_golden("rq04_headroom_drive_verdicts");
    let physical: Vec<serde_json::Value> =
        serde_json::from_value(payload["physical"].clone()).unwrap();
    let sat = physical.iter().find(|p| p["output"] == "sat").unwrap();
    let util: f64 = serde_json::from_value(sat["max_utilization"].clone()).unwrap();
    assert!(
        (util - 1.0).abs() <= 1e-12,
        "RQ04 sat utilization {util} must sit exactly at its limit"
    );
    assert_eq!(sat["passes_declared_samples"], true);
    let sub = physical.iter().find(|p| p["output"] == "sub").unwrap();
    assert_eq!(sub["passes_declared_samples"], false);
}
