//! Negative controls for the rab group (families RA03-RA06).
//!
//! Each control loads an engine-blessed golden, applies one classic defect
//! (dB-factor mix-up, sign flip, off-by-one grid, wrong interpolation
//! axis, dropped factor of two, conjugation), and asserts the resulting
//! error exceeds the case tolerance. A control that cannot fail is
//! worthless; these must keep failing.

use autoeq_qa::golden_dir;
use num_complex::Complex64;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn vec_f64(payload: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(payload[key].clone()).expect("golden numeric array")
}

// --- RA03: Q/prominence/depth (rel 1e-9) and unit mask (abs 1e-12). ---

/// A -6 dB bandwidth read as -3 dB shrinks a Lorentzian Q by 1/sqrt(3);
/// the RA03 Q tolerance must expose that dB-factor defect.
#[test]
fn ra03_db_factor_in_q_fails_tolerance() {
    let payload = load_golden("ra03_mode_null_masks");
    let modes: Vec<[f64; 3]> = serde_json::from_value(payload["modes_f_q_prom"].clone()).unwrap();
    let defective = modes[0][1] / 3.0_f64.sqrt();
    let err = ((defective - modes[0][1]) / modes[0][1]).abs();
    assert!(
        err > 1e-9,
        "6 dB-for-3 dB Q mix-up must exceed the 1e-9 tolerance, got {err:.3e}"
    );
}

/// A sign-flipped dip depth (notch reported as a peak) must be visible.
#[test]
fn ra03_sign_flipped_depth_fails_tolerance() {
    let payload = load_golden("ra03_mode_null_masks");
    let nulls: Vec<[f64; 3]> = serde_json::from_value(payload["nulls_f_q_depth"].clone()).unwrap();
    let err = (2.0 * nulls[0][2]).abs() / nulls[0][2].abs();
    assert!(
        err > 1e-9,
        "sign-flipped null depth must exceed the 1e-9 tolerance, got {err:.3e}"
    );
}

/// An off-by-one-bin mask shift must break the 1e-12 mask agreement.
#[test]
fn ra03_off_by_one_mask_fails_tolerance() {
    let payload = load_golden("ra03_mode_null_masks");
    let mask = vec_f64(&payload, "null_mask");
    let worst = mask
        .iter()
        .zip(mask.iter().skip(1))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "one-bin mask shift must exceed the 1e-12 tolerance, got {worst:.3e}"
    );
}

// --- RA04: thresholds (abs 1e-12 s), severity (abs 1e-9 dB). ---

/// Linear-in-frequency interpolation instead of log-frequency must move
/// the 75 Hz threshold beyond the arithmetic tolerance.
#[test]
fn ra04_linear_for_log_interp_fails_tolerance() {
    let payload = load_golden("ra04_decay_severity");
    let art: Vec<[f64; 2]> = serde_json::from_value(payload["artificial_table"].clone()).unwrap();
    // Bracket 75 Hz between the 63 Hz and 100 Hz entries.
    let (f0, t0) = (art[2][0], art[2][1]);
    let (f1, t1) = (art[3][0], art[3][1]);
    let linear = t0 + (75.0 - f0) / (f1 - f0) * (t1 - t0);
    let probes: Vec<f64> = serde_json::from_value(payload["probe_freqs_hz"].clone()).unwrap();
    let thr: Vec<f64> = serde_json::from_value(payload["threshold_artificial_s"].clone()).unwrap();
    let idx = probes.iter().position(|&f| f == 75.0).unwrap();
    let err = (linear - thr[idx]).abs();
    assert!(
        err > 1e-12,
        "linear-for-log interpolation must exceed the 1e-12 s tolerance, got {err:.3e}"
    );
}

/// A 10 log10 (power) severity instead of 20 log10 (amplitude) halves every
/// dB value; the RA04 severity tolerance must expose that factor.
#[test]
fn ra04_power_vs_amplitude_db_fails_tolerance() {
    let payload = load_golden("ra04_decay_severity");
    let sev = payload["severity_above_db"].as_f64().unwrap();
    assert!(
        sev > 1e-9,
        "fixture severity must be non-trivial, got {sev:.3e}"
    );
    let err = (sev - sev / 2.0).abs();
    assert!(
        err > 1e-9,
        "10-vs-20 log severity factor must exceed the 1e-9 dB tolerance, got {err:.3e}"
    );
}

/// Endpoint clamping outside 32-250 Hz must not pass as a valid limit.
#[test]
fn ra04_out_of_domain_clamp_fails_gate() {
    use roomeq_analysis::temporal_targets::{max_acceptable_decay_time_checked, temporal_severity};
    for f in [20.0, 300.0] {
        assert!(
            max_acceptable_decay_time_checked(f, false).is_none(),
            "clamped lookup at {f} Hz must report unknown"
        );
        assert_eq!(
            temporal_severity(f, 30.0, false),
            0.0,
            "out-of-domain severity at {f} Hz must stay silent"
        );
    }
}

// --- RA05: bulk delay (abs 1e-6 s), all-pass GD (abs 0.35 ms). ---

/// A polarity-flipped delay (advance instead of lag) must miss the bulk
/// estimate by twice the planted delay.
#[test]
fn ra05_sign_flipped_delay_fails_tolerance() {
    use roomeq_analysis::excess_phase::{ExcessPhaseConfig, ExcessPhaseInput, assess_excess_phase};
    let payload = load_golden("ra05_delay_excess_phase");
    let freqs = vec_f64(&payload, "grid_hz");
    let phase: Vec<f64> = vec_f64(&payload, "phase_delay_deg");
    let band: Vec<f64> = serde_json::from_value(payload["analysis_band_hz"].clone()).unwrap();
    let cfg = ExcessPhaseConfig {
        taper_oct: payload["taper_oct"].as_f64().unwrap(),
        snr_floor_db: payload["snr_floor_db"].as_f64().unwrap(),
        min_valid_fraction: payload["min_valid_fraction"].as_f64().unwrap(),
        smooth_narrow_oct: payload["smooth_narrow_oct"].as_f64().unwrap(),
        smooth_wide_oct: payload["smooth_wide_oct"].as_f64().unwrap(),
        consistency_tol_ms: payload["consistency_tol_ms"].as_f64().unwrap(),
        strict_dips: false,
        dip_depth_db: payload["dip_depth_db"].as_f64().unwrap(),
        analysis_band_hz: (band[0], band[1]),
    };
    let flipped: Vec<f64> = phase.iter().map(|p| -p).collect();
    let input = ExcessPhaseInput {
        freqs_hz: freqs,
        magnitude_db: vec_f64(&payload, "magnitude_db"),
        phase_deg: Some(flipped),
        snr_db: vec_f64(&payload, "snr_db"),
        sample_rate_hz: payload["sample_rate_hz"].as_f64().unwrap(),
    };
    match assess_excess_phase(&input, &cfg) {
        roomeq_analysis::excess_phase::Assessment::Supported(report) => {
            let err =
                (report.bulk_delay_s - payload["expected_bulk_delay_s"].as_f64().unwrap()).abs();
            assert!(
                err > 1e-6,
                "sign-flipped delay must exceed the 1e-6 s tolerance, got {err:.3e}"
            );
        }
        // A polarity-flipped delay also fails the window-sensitivity gate
        // (Unknown): either way the defect is detected, never passed.
        roomeq_analysis::excess_phase::Assessment::Unknown { reason, .. } => {
            assert!(
                reason.contains("window") || reason.contains("bulk-delay"),
                "unexpected Unknown reason: {reason}"
            );
        }
        other => panic!("sign-flipped delay must not assess cleanly, got {other:?}"),
    }
}

/// A dropped factor of two in the all-pass group-delay formula must exceed
/// the 0.35 ms smoothing budget.
#[test]
fn ra05_dropped_factor_two_fails_gd_budget() {
    let payload = load_golden("ra05_delay_excess_phase");
    let expected = payload["expected_allpass_gd_ms"].as_f64().unwrap();
    let err = (expected - expected / 2.0).abs();
    assert!(
        err > 0.35,
        "halved all-pass GD must exceed the 0.35 ms budget, got {err:.3e}"
    );
}

/// Sub-floor SNR must gate to Unknown, never to a Supported verdict.
#[test]
fn ra05_low_snr_gates_to_unknown() {
    use roomeq_analysis::excess_phase::{Assessment, ExcessPhaseConfig, ExcessPhaseInput};
    let payload = load_golden("ra05_delay_excess_phase");
    let freqs = vec_f64(&payload, "grid_hz");
    let n = freqs.len();
    let band: Vec<f64> = serde_json::from_value(payload["analysis_band_hz"].clone()).unwrap();
    let cfg = ExcessPhaseConfig {
        taper_oct: 0.5,
        snr_floor_db: 10.0,
        min_valid_fraction: 0.8,
        smooth_narrow_oct: 1.0 / 6.0,
        smooth_wide_oct: 0.5,
        consistency_tol_ms: 0.5,
        strict_dips: false,
        dip_depth_db: 6.0,
        analysis_band_hz: (band[0], band[1]),
    };
    let input = ExcessPhaseInput {
        freqs_hz: freqs,
        magnitude_db: vec![0.0; n],
        phase_deg: Some(vec_f64(&payload, "phase_delay_deg")),
        snr_db: vec![-10.0; n],
        sample_rate_hz: 48000.0,
    };
    assert!(
        matches!(
            roomeq_analysis::excess_phase::assess_excess_phase(&input, &cfg),
            Assessment::Unknown { .. }
        ),
        "sub-floor SNR must gate to Unknown"
    );
}

// --- RA06: H (complex rel 1e-9), time samples (abs 1e-9). ---

fn ra06_complex() -> (Vec<f64>, Vec<Complex64>, f64, f64, usize) {
    let payload = load_golden("ra06_reflection_cancel");
    let freqs: Vec<f64> = serde_json::from_value(payload["grid_hz"].clone()).unwrap();
    let pairs: Vec<[f64; 2]> = serde_json::from_value(payload["response_re_im"].clone()).unwrap();
    let href: Vec<Complex64> = pairs.iter().map(|p| Complex64::new(p[0], p[1])).collect();
    let sr = payload["sample_rate_hz"].as_f64().unwrap();
    let g = payload["cancellation_gain"].as_f64().unwrap();
    let d = payload["delay_samples"].as_f64().unwrap() as usize;
    (freqs, href, sr, g, d)
}

/// A sign-flipped cancellation gain (addition instead of subtraction,
/// H = 2 - H_ref) must blow the 1e-9 transfer tolerance.
#[test]
fn ra06_sign_flipped_gain_fails_tolerance() {
    use autoeq_qa::complex_rel_error;
    let (_, href, _, _, _) = ra06_complex();
    let worst = href
        .iter()
        .map(|h| complex_rel_error(Complex64::new(2.0, 0.0) - h, *h))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "sign-flipped cancellation gain must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// An off-by-one-sample echo delay must be visible at the top probe.
#[test]
fn ra06_off_by_one_delay_fails_tolerance() {
    use autoeq_qa::complex_rel_error;
    let (freqs, href, sr, g, d) = ra06_complex();
    let worst = freqs
        .iter()
        .zip(href.iter())
        .map(|(&f, h)| {
            // Recover LP(f), then re-apply with d+1 samples of delay.
            let z = Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * f * d as f64 / sr);
            let lp = (Complex64::new(1.0, 0.0) - h) / (g * z);
            let z1 =
                Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * f * (d + 1) as f64 / sr);
            complex_rel_error(Complex64::new(1.0, 0.0) - g * z1 * lp, *h)
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "off-by-one echo delay must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// A conjugated transfer (Exp[+I w] instead of Exp[-I w]) must not hide
/// inside a magnitude-only comparison.
#[test]
fn ra06_wrong_conjugation_fails_tolerance() {
    use autoeq_qa::complex_rel_error;
    let (_, href, _, _, _) = ra06_complex();
    let worst = href
        .iter()
        .map(|h| complex_rel_error(h.conj(), *h))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "conjugated reflection transfer must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}
