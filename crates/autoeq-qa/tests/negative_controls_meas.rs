//! Negative controls for the meas group (M01-M04): each control loads an
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

fn worst_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// M01: an equal-thirds PIR mix instead of the declared 0.12/0.44/0.44
/// weights must blow the 1e-9 dB tolerance.
#[test]
fn equal_weight_pir_mix_fails_m01_tolerance() {
    let payload = load_golden("m01_pir_prefscore");
    let lw = vec_f64(&payload["lw_db"]);
    let er = vec_f64(&payload["er_db"]);
    let sp = vec_f64(&payload["sp_db"]);
    let reference = vec_f64(&payload["pir_db"]);
    let wrong: Vec<f64> = lw
        .iter()
        .zip(er.iter())
        .zip(sp.iter())
        .map(|((l, e), s)| {
            let p = |v: f64| 10f64.powf((v - 105.0) / 20.0);
            20.0 * ((p(*l).powi(2) + p(*e).powi(2) + p(*s).powi(2)) / 3.0)
                .sqrt()
                .log10()
                + 105.0
        })
        .collect();
    let worst = worst_abs(&wrong, &reference);
    assert!(
        worst > 1e-9,
        "equal-weight PIR mix must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

/// M01: a 10log10 (power) vs 20log10 (pressure) mix-up in the
/// SPL-to-pressure conversion must be visible in the PIR.
#[test]
fn power_vs_pressure_conversion_fails_m01_tolerance() {
    let payload = load_golden("m01_pir_prefscore");
    let lw = vec_f64(&payload["lw_db"]);
    let er = vec_f64(&payload["er_db"]);
    let sp = vec_f64(&payload["sp_db"]);
    let reference = vec_f64(&payload["pir_db"]);
    // Defect: SPL-to-pressure with /10 (power) instead of /20
    // (pressure), mixed and converted back with the correct 20log10.
    let shifted: Vec<f64> = lw
        .iter()
        .zip(er.iter())
        .zip(sp.iter())
        .map(|((l, e), s)| {
            let q = |v: f64| 10f64.powf((v - 105.0) / 10.0);
            20.0 * (0.12 * q(*l).powi(2) + 0.44 * q(*e).powi(2) + 0.44 * q(*s).powi(2))
                .sqrt()
                .log10()
                + 105.0
        })
        .collect();
    let worst = worst_abs(&shifted, &reference);
    assert!(
        worst > 1e-9,
        "10-vs-20 dB conversion factor must be visible in M01 PIR, got {worst:.3e}"
    );
}

/// M02: negating the coherent angle (+45 deg instead of -45 deg, i.e. a
/// conjugation / sign flip) must blow the phase tolerance.
#[test]
fn conjugated_coherent_angle_fails_m02_tolerance() {
    let payload = load_golden("m02_avg_coherent");
    let reference = vec_f64(&payload["coherent_phase_deg"]);
    let conjugated: Vec<f64> = reference.iter().map(|p| -p).collect();
    let worst = worst_abs(&conjugated, &reference);
    assert!(
        worst > 1e-9,
        "conjugated coherent angle must exceed the 1e-9 deg tolerance, got {worst:.3e}"
    );
}

/// M02: reporting the power mean where the coherent mean belongs must
/// fail: quadrature takes cancel 3.01 dB under the RMS level.
#[test]
fn power_mean_reported_as_coherent_fails_m02_tolerance() {
    let payload = load_golden("m02_avg_coherent");
    let power = vec_f64(&payload["power_mean_db"]);
    let coherent = vec_f64(&payload["coherent_mean_db"]);
    let worst = worst_abs(&power, &coherent);
    assert!(
        worst > 1e-9,
        "power/coherent confusion must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

/// M03: ADDING the mic phase instead of subtracting it (wrong sign)
/// must blow the phase tolerance.
#[test]
fn added_mic_phase_fails_m03_tolerance() {
    let payload = load_golden("m03_mic_clock");
    let curve_phase = vec_f64(&payload["curve_phase_deg"]);
    let corrected = vec_f64(&payload["corrected_phase_deg"]);
    // Defect reconstruction: corrected_wrong = measured + mic phase, so
    // wrong - right = 2 * mic phase at every bin.
    let mic_phase: Vec<f64> = curve_phase
        .iter()
        .zip(corrected.iter())
        .map(|(m, c)| m - c)
        .collect();
    let wrong: Vec<f64> = curve_phase
        .iter()
        .zip(mic_phase.iter())
        .map(|(m, p)| m + p)
        .collect();
    let worst = worst_abs(&wrong, &corrected);
    assert!(
        worst > 1e-9,
        "added mic phase must exceed the 1e-9 deg tolerance, got {worst:.3e}"
    );
}

/// M03: a flipped clock-drift sign (slow clock reported for a fast one)
/// must blow the ppm tolerance.
#[test]
fn flipped_clock_drift_sign_fails_m03_tolerance() {
    let payload = load_golden("m03_mic_clock");
    let rate = payload["fit_rate_ppm"].as_f64().unwrap();
    assert!(
        (-rate - rate).abs() > 1e-9,
        "flipped +50 ppm drift must exceed the 1e-9 ppm tolerance"
    );
}

/// M04: linear-frequency interpolation at a geometric midpoint instead
/// of log-frequency interpolation must blow the dB tolerance.
#[test]
fn linear_instead_of_log_interp_fails_m04_tolerance() {
    let payload = load_golden("m04_ingest_adapters");
    let reference = vec_f64(&payload["interp_spl_db"]);
    let targets = vec_f64(&payload["interp_freqs_hz"]);
    // Defect: linear-in-f interpolation of the {100:0, 1000:10} segment.
    let wrong: Vec<f64> = targets
        .iter()
        .map(|f| {
            if *f <= 100.0 {
                0.0
            } else if *f <= 1000.0 {
                10.0 * (f - 100.0) / 900.0
            } else if *f <= 10000.0 {
                10.0 * (10000.0 - f) / 9000.0
            } else {
                0.0
            }
        })
        .collect();
    let worst = worst_abs(&wrong, &reference);
    assert!(
        worst > 1e-9,
        "linear-instead-of-log interpolation must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

/// M04: clamping BOTH sides (not just positive dB) must move the
/// negative fixture values beyond tolerance.
#[test]
fn two_sided_clamp_fails_m04_tolerance() {
    let payload = load_golden("m04_ingest_adapters");
    let input = vec_f64(&payload["clamp_input_db"]);
    let reference = vec_f64(&payload["clamp_output_db"]);
    let wrong: Vec<f64> = input.iter().map(|v| v.clamp(-12.0, 12.0)).collect();
    let worst = worst_abs(&wrong, &reference);
    assert!(
        worst > 1e-9,
        "two-sided clamp must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

/// M04: subtracting the display shift twice must move the normalized
/// output beyond tolerance (the shift applies exactly once).
#[test]
fn double_display_shift_fails_m04_tolerance() {
    let payload = load_golden("m04_ingest_adapters");
    let reference = vec_f64(&payload["norm_output_db"]);
    let mean = payload["norm_band_mean_db"].as_f64().unwrap();
    let wrong: Vec<f64> = reference.iter().map(|v| v - mean).collect();
    let worst = worst_abs(&wrong, &reference);
    assert!(
        worst > 1e-9,
        "double display shift must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}
