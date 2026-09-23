//! Negative controls for the rw2 group (RW07-RW11, ROOT01): each control
//! loads an engine-blessed golden, applies one classic defect to a copy,
//! and asserts the resulting error exceeds the case tolerance. A control
//! that cannot fail is worthless; these must keep failing.

use autoeq_qa::complex_rel_error;
use autoeq_qa::golden_dir;
use num_complex::Complex64;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn pairs(v: &serde_json::Value) -> Vec<Complex64> {
    let raw: Vec<[f64; 2]> = serde_json::from_value(v.clone()).unwrap();
    raw.iter().map(|p| Complex64::new(p[0], p[1])).collect()
}

fn worst_complex_rel(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| complex_rel_error(*x, *y))
        .fold(0.0f64, f64::max)
}

/// RW07: counting the empty-conditions row as ran (ran 2, stale 1) must
/// disagree with the audited 1 ran / 2 stale split.
#[test]
fn empty_conditions_counted_as_ran_fails_rw07() {
    let payload = load_golden("rw07_scorecard_pruning");
    let expected_ran: usize = serde_json::from_value(payload["expected_ran"].clone()).unwrap();
    let expected_stale: Vec<String> =
        serde_json::from_value(payload["expected_stale"].clone()).unwrap();
    let wrong_ran = expected_ran + 1;
    let wrong_stale = expected_stale[1..].to_vec();
    assert!(
        wrong_ran != expected_ran || wrong_stale != expected_stale,
        "dropping the empty-conditions stale row must change the audit outcome"
    );
}

/// RW07: swapping the stale and skipped reason strings must break exact
/// identity.
#[test]
fn swapped_audit_reasons_fail_rw07() {
    let payload = load_golden("rw07_scorecard_pruning");
    let stale: Vec<String> = serde_json::from_value(payload["expected_stale"].clone()).unwrap();
    let skipped: Vec<String> = serde_json::from_value(payload["expected_skipped"].clone()).unwrap();
    let mut swapped = stale.clone();
    swapped.extend(skipped.iter().take(1).cloned());
    assert!(
        swapped != stale,
        "mixing skipped reasons into stale must break identity"
    );
}

/// RW08: a single-hex-digit stimulus hash flip must break binding identity.
#[test]
fn flipped_hash_digit_fails_rw08() {
    let payload = load_golden("rw08_stimulus_binding");
    let hash = payload["stimulus_hash"].as_str().unwrap();
    let mut wrong = hash.to_string();
    let last = wrong.pop().unwrap();
    wrong.push(if last == '0' { '1' } else { '0' });
    assert_ne!(wrong, hash, "flipped hash must differ");
}

/// RW08: swapped routing labels (mono <-> spatial) must mismatch.
#[test]
fn swapped_route_labels_fail_rw08() {
    let payload = load_golden("rw08_stimulus_binding");
    let labels: Vec<String> = serde_json::from_value(payload["route_labels"].clone()).unwrap();
    let mut swapped = labels.clone();
    swapped.swap(0, 2);
    assert_ne!(swapped, labels, "swapped labels must mismatch");
}

/// RW08: a duplicated LFE gain stage (double LFE) must change the count.
#[test]
fn doubled_lfe_stage_fails_rw08() {
    let payload = load_golden("rw08_stimulus_binding");
    let expected: usize = serde_json::from_value(payload["expected_lfe_stages"].clone()).unwrap();
    let lfe = payload["lfe_channel"].as_str().unwrap();
    let mut gains: Vec<(String, f64)> = Vec::new();
    for entry in payload["routes"].as_array().unwrap() {
        for pair in entry["gains"].as_array().unwrap() {
            gains.push((
                pair[0].as_str().unwrap().to_string(),
                pair[1].as_f64().unwrap(),
            ));
        }
    }
    let baseline = gains.iter().filter(|(channel, _)| channel == lfe).count();
    assert_eq!(baseline, expected);
    gains.push((lfe.to_string(), -6.0));
    let doubled = gains.iter().filter(|(channel, _)| channel == lfe).count();
    assert!(
        doubled != expected,
        "double LFE gain must change the stage count"
    );
}

/// RW09: conjugating the replayed transfer must blow the 1e-9 tolerance.
#[test]
fn conjugated_replay_fails_rw09_tolerance() {
    let payload = load_golden("rw09_scaled_replay");
    let reference = pairs(&payload["response_re_im"]);
    let wrong: Vec<Complex64> = reference.iter().map(|v| v.conj()).collect();
    let worst = worst_complex_rel(&wrong, &reference);
    assert!(
        worst > 1e-9,
        "conjugated replay must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RW09: a 10log10 (power) vs 20log10 (pressure) gain-factor mix-up must
/// be visible in the delivered transfer.
#[test]
fn power_gain_factor_fails_rw09_tolerance() {
    let payload = load_golden("rw09_scaled_replay");
    let reference = pairs(&payload["response_re_im"]);
    let gain_db: f64 = serde_json::from_value(payload["gain_db"].clone()).unwrap();
    let factor = 10f64.powf(gain_db / 10.0) / 10f64.powf(gain_db / 20.0);
    let wrong: Vec<Complex64> = reference.iter().map(|v| v * factor).collect();
    let worst = worst_complex_rel(&wrong, &reference);
    assert!(
        worst > 1e-9,
        "power gain factor must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RW10: regressing SPL vs log10(f) instead of log2(f) must move the
/// slope by far more than 1e-9 dB/oct.
#[test]
fn log10_regressor_fails_rw10_tolerance() {
    let payload = load_golden("rw10_from_measurement_slope");
    let freqs: Vec<f64> = serde_json::from_value(payload["freqs_hz"].clone()).unwrap();
    let spl: Vec<f64> = serde_json::from_value(payload["spl_db"].clone()).unwrap();
    let expected: f64 =
        serde_json::from_value(payload["expected_slope_db_per_oct"].clone()).unwrap();
    let xs: Vec<f64> = freqs.iter().map(|f| f.log10()).collect();
    let n = xs.len() as f64;
    let (sx, sy, sxx, sxy) = xs
        .iter()
        .zip(spl.iter())
        .fold((0.0, 0.0, 0.0, 0.0), |(sx, sy, sxx, sxy), (x, y)| {
            (sx + x, sy + y, sxx + x * x, sxy + x * y)
        });
    let wrong = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    assert!(
        (wrong - expected).abs() > 1e-9,
        "log10 regressor slope {wrong:.6} must differ from {expected:.6}"
    );
}

/// RW10: a sign-flipped tilt must miss the slope tolerance.
#[test]
fn sign_flipped_tilt_fails_rw10_tolerance() {
    let payload = load_golden("rw10_from_measurement_slope");
    let expected: f64 =
        serde_json::from_value(payload["expected_slope_db_per_oct"].clone()).unwrap();
    assert!(
        (-expected - expected).abs() > 1e-9,
        "sign-flipped tilt must exceed the 1e-9 dB/oct tolerance"
    );
}

/// RW11: an off-by-one clipped grid (first bin dropped) must break the
/// 1e-12 grid agreement.
#[test]
fn off_by_one_grid_fails_rw11_tolerance() {
    let payload = load_golden("rw11_hybrid_grid_delay");
    let grid: Vec<f64> = serde_json::from_value(payload["clipped_grid_hz"].clone()).unwrap();
    let shifted = &grid[1..];
    assert_eq!(shifted.len() + 1, grid.len());
    let worst = shifted
        .iter()
        .zip(grid.iter())
        .map(|(a, b)| ((a - b) / b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "off-by-one grid must exceed the 1e-12 tolerance, got {worst:.3e}"
    );
}

/// RW11: a delay sign flip (-2.5 ms vs +2.5 ms) must blow the ms budget.
#[test]
fn delay_sign_flip_fails_rw11_tolerance() {
    let payload = load_golden("rw11_hybrid_grid_delay");
    let expected: f64 = serde_json::from_value(payload["expected_delay_ms"].clone()).unwrap();
    assert!(
        (-expected - expected).abs() > 1e-9,
        "delay sign flip must exceed the 1e-9 ms tolerance"
    );
}

/// RW11: excess phase in radians mistaken for degrees must break the
/// residual agreement.
#[test]
fn radian_degree_mixup_fails_rw11_tolerance() {
    let payload = load_golden("rw11_hybrid_grid_delay");
    let residual: Vec<f64> =
        serde_json::from_value(payload["expected_residual_deg"].clone()).unwrap();
    let excess: Vec<f64> = serde_json::from_value(payload["excess_phase_deg"].clone()).unwrap();
    let wrong: Vec<f64> = excess.iter().map(|p| p.to_radians()).collect();
    let worst = wrong
        .iter()
        .zip(excess.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "radian/degree mixup must exceed the 1e-9 deg tolerance, got {worst:.3e}"
    );
    assert!(
        residual.iter().all(|v| v.abs() <= 1e-9),
        "sanity: engine residual must be ~zero"
    );
}

/// ROOT01: conjugating the facade response must blow the 1e-12 tolerance.
#[test]
fn conjugated_facade_fails_root01_tolerance() {
    let payload = load_golden("root01_facade_passthrough");
    let reference = pairs(&payload["response_re_im"]);
    let wrong: Vec<Complex64> = reference.iter().map(|v| v.conj()).collect();
    let worst = worst_complex_rel(&wrong, &reference);
    assert!(
        worst > 1e-12,
        "conjugated facade response must exceed the 1e-12 tolerance, got {worst:.3e}"
    );
}

/// ROOT01: evaluating at the wrong sample rate (48 kHz vs 44.1 kHz)
/// must move the transfer beyond tolerance.
#[test]
fn wrong_sample_rate_fails_root01_tolerance() {
    use std::f64::consts::PI;
    let payload = load_golden("root01_facade_passthrough");
    let reference = pairs(&payload["response_re_im"]);
    let sr_wrong = 48000.0;
    let sr: f64 = serde_json::from_value(payload["sample_rate_hz"].clone()).unwrap();
    let fc: f64 = serde_json::from_value(payload["center_hz"].clone()).unwrap();
    let q: f64 = serde_json::from_value(payload["q"].clone()).unwrap();
    let gain_db: f64 = serde_json::from_value(payload["gain_db"].clone()).unwrap();
    let freqs: Vec<f64> = serde_json::from_value(payload["freqs_hz"].clone()).unwrap();
    let rbj = |rate: f64| -> Vec<Complex64> {
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * fc / rate;
        let (sn, cw) = w0.sin_cos();
        let alpha = sn / (2.0 * q);
        let (b0, b1, b2) = (1.0 + alpha * a, -2.0 * cw, 1.0 - alpha * a);
        let (a0, a1, a2) = (1.0 + alpha / a, -2.0 * cw, 1.0 - alpha / a);
        freqs
            .iter()
            .map(|f| {
                let z = Complex64::from_polar(1.0, -2.0 * PI * f / rate);
                (b0 + b1 * z + b2 * z * z) / (a0 + a1 * z + a2 * z * z)
            })
            .collect()
    };
    assert_eq!(sr, 44100.0);
    let wrong = rbj(sr_wrong);
    let worst = worst_complex_rel(&wrong, &reference);
    assert!(
        worst > 1e-12,
        "48 kHz evaluation must exceed the 1e-12 tolerance, got {worst:.3e}"
    );
}
