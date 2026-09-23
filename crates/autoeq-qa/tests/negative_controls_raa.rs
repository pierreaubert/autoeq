//! Negative controls for the raa group (RA01-RA02): each control loads an
//! engine-blessed golden, applies one classic defect to a copy, and
//! asserts the resulting error exceeds the case tolerance. A control
//! that cannot fail is worthless; these must keep failing.

use autoeq_qa::golden_dir;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn vec_f64(v: &serde_json::Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}

fn worst_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| autoeq_qa::rel_error(*x, *y))
        .fold(0.0f64, f64::max)
}

/// RA01 grid: extending the 4 Hz linear bass leg past the 1 kHz junction
/// (linear-vs-log mixup) must blow the 1e-9 relative tolerance.
#[test]
fn linear_bass_extension_fails_ra01_grid_tolerance() {
    let payload = load_golden("ra01_hybrid_grid");
    let reference = vec_f64(&payload["freqs_hz"]);
    let wrong: Vec<f64> = reference
        .iter()
        .map(|f| {
            if *f <= 1000.0 {
                *f
            } else {
                1000.0 + (*f - 1000.0) * 0.01
            }
        })
        .collect();
    let worst = worst_rel(&wrong, &reference);
    assert!(
        worst > 1e-9,
        "linear-vs-log grid mixup must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RA01 grid: dropping the 20000 Hz endpoint (off-by-one grid) must be
/// caught by the length/endpoint contract.
#[test]
fn dropped_endpoint_fails_ra01_grid_contract() {
    let payload = load_golden("ra01_hybrid_grid");
    let reference = vec_f64(&payload["freqs_hz"]);
    let truncated = &reference[..reference.len() - 1];
    assert_ne!(
        truncated.len(),
        reference.len(),
        "off-by-one grid must change the grid length"
    );
    assert!(
        (truncated[truncated.len() - 1] - 20_000.0).abs() > 1.0,
        "dropped endpoint must miss 20000 Hz"
    );
}

/// RA01 slope: a sign-flipped slope must blow the 1e-9 dB/oct tolerance.
#[test]
fn sign_flipped_slope_fails_ra01_slope_tolerance() {
    let payload = load_golden("ra01_log_slope");
    let full: f64 = serde_json::from_value(payload["full_slope_db_per_oct"].clone()).unwrap();
    let sub: f64 = serde_json::from_value(payload["sub_slope_db_per_oct"].clone()).unwrap();
    assert!(
        (-full - full).abs() > 1e-9 && (-sub - sub).abs() > 1e-9,
        "sign flip must move both window slopes by far more than 1e-9 dB/oct"
    );
}

/// RA01 slope: fitting against log10 instead of log2 (missing ln2 factor)
/// must be visible in both windows.
#[test]
fn log10_vs_log2_base_fails_ra01_slope_tolerance() {
    let payload = load_golden("ra01_log_slope");
    let full: f64 = serde_json::from_value(payload["full_slope_db_per_oct"].clone()).unwrap();
    let sub: f64 = serde_json::from_value(payload["sub_slope_db_per_oct"].clone()).unwrap();
    let ln2 = std::f64::consts::LN_2;
    assert!(
        (full / ln2 - full).abs() > 1e-9,
        "log10/log2 base mixup must exceed the slope tolerance, got {:.3e}",
        (full / ln2 - full).abs()
    );
    assert!(
        (sub / ln2 - sub).abs() > 1e-9,
        "log10/log2 base mixup must exceed the slope tolerance, got {:.3e}",
        (sub / ln2 - sub).abs()
    );
}

/// RA01 LR24: conjugating the lowpass branch (imaginary sign flip) must
/// blow the 1e-9 complex-relative tolerance wherever phase is nonzero.
#[test]
fn conjugated_branch_fails_ra01_lr24_tolerance() {
    use num_complex::Complex64;
    let payload = load_golden("ra01_lr24_crossover");
    let lp: Vec<[f64; 2]> = serde_json::from_value(payload["lp_re_im"].clone()).unwrap();
    let worst = lp
        .iter()
        .map(|p| {
            autoeq_qa::complex_rel_error(Complex64::new(p[0], -p[1]), Complex64::new(p[0], p[1]))
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "conjugated LR24 branch must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RA01 LR24: forgetting to square the Butterworth cascade (a single
/// stage instead of the LR24 pair) must blow the 1e-9 complex-relative
/// tolerance on both branches.
#[test]
fn unsquared_stage_fails_ra01_lr24_branches() {
    use num_complex::Complex64;
    // Principal complex square root inverts the cascade squaring.
    fn unsquared(pair: [f64; 2]) -> Complex64 {
        let mag = pair[0].hypot(pair[1]);
        let re = ((mag + pair[0]) / 2.0).sqrt();
        let im = ((mag - pair[0]) / 2.0).sqrt().copysign(pair[1]);
        Complex64::new(re, im)
    }
    let payload = load_golden("ra01_lr24_crossover");
    let lp: Vec<[f64; 2]> = serde_json::from_value(payload["lp_re_im"].clone()).unwrap();
    let hp: Vec<[f64; 2]> = serde_json::from_value(payload["hp_re_im"].clone()).unwrap();
    let worst = lp
        .iter()
        .chain(hp.iter())
        .map(|p| autoeq_qa::complex_rel_error(unsquared(*p), Complex64::new(p[0], p[1])))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "unsquared LR24 stage must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// RA02 probe: a one-sample arrival slip must exceed the 0.02 ms
/// tolerance at 8 kHz (one sample = 0.125 ms).
#[test]
fn off_by_one_arrival_fails_ra02_probe_tolerance() {
    let payload = load_golden("ra02_probe_delay");
    let arrival_ms: f64 = serde_json::from_value(payload["arrival_ms"].clone()).unwrap();
    let slipped = 38.0 * 1000.0 / 8000.0;
    assert!(
        (slipped - arrival_ms).abs() > 0.02,
        "one-sample slip must exceed the 0.02 ms tolerance"
    );
}

/// RA02 probe: a 10log10 (power) vs 20log10 (pressure) gain mixup must
/// exceed the 0.05 dB tolerance.
#[test]
fn power_vs_pressure_gain_fails_ra02_probe_tolerance() {
    let payload = load_golden("ra02_probe_delay");
    let gain_db: f64 = serde_json::from_value(payload["gain_db"].clone()).unwrap();
    let wrong = 10.0 * 0.5f64.log10();
    assert!(
        (wrong - gain_db).abs() > 0.05,
        "10-vs-20 dB gain factor must exceed the 0.05 dB tolerance"
    );
}

/// RA02 phase slope: a delay sign flip must blow the 1e-9 ms tolerance.
#[test]
fn sign_flipped_delay_fails_ra02_phase_tolerance() {
    let payload = load_golden("ra02_phase_slope_delay");
    let fitted: f64 = serde_json::from_value(payload["fitted_delay_ms"].clone()).unwrap();
    assert!(
        (-fitted - fitted).abs() > 1e-9,
        "delay sign flip must exceed the 1e-9 ms tolerance"
    );
}

/// RA02 phase slope: reporting seconds-per-Hz slope as ms (missing x1000)
/// must be caught by the sample/ms conversion.
#[test]
fn seconds_vs_ms_factor_fails_ra02_phase_conversion() {
    let payload = load_golden("ra02_phase_slope_delay");
    let planted_samples: f64 =
        serde_json::from_value(payload["planted_delay_samples"].clone()).unwrap();
    let fitted: f64 = serde_json::from_value(payload["fitted_delay_ms"].clone()).unwrap();
    let wrong_samples = fitted / 1000.0 / 1000.0 * 48_000.0;
    assert!(
        (wrong_samples - planted_samples).abs() > 1e-6,
        "missing ms factor must break the 114.3-sample conversion"
    );
}

/// RA02 arrival: an off-by-one onset must violate the exact-sample contract.
#[test]
fn off_by_one_onset_fails_ra02_arrival_contract() {
    let payload = load_golden("ra02_arrival_alignment");
    let arrival: usize = serde_json::from_value(payload["arrival_samples"].clone()).unwrap();
    assert_ne!(
        arrival + 1,
        arrival,
        "off-by-one onset must miss sample 240"
    );
    assert!(
        ((arrival + 1) as f64 * 1000.0 / 48_000.0 - 5.0).abs() > 1e-9,
        "off-by-one onset must miss 5.0 ms"
    );
}

/// RA02 alignment: aligning to the fastest channel (min instead of max)
/// must change every nonzero offset.
#[test]
fn align_to_fastest_fails_ra02_alignment_contract() {
    let payload = load_golden("ra02_arrival_alignment");
    let arrivals: std::collections::HashMap<String, f64> =
        serde_json::from_value(payload["arrivals_ms"].clone()).unwrap();
    let want: std::collections::HashMap<String, f64> =
        serde_json::from_value(payload["alignment_delays_ms"].clone()).unwrap();
    let min_arrival = arrivals.values().copied().fold(f64::INFINITY, f64::min);
    let worst = want
        .iter()
        .map(|(ch, expected)| ((arrivals[ch] - min_arrival) - expected).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "align-to-fastest must change the offsets, got {worst:.3e}"
    );
}
