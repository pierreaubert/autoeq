//! Negative controls for the optimc group (O07-O12): prove each comparison
//! detects its defect class. Each control loads an engine-blessed golden,
//! applies one classic defect to a copy, and asserts the resulting error
//! exceeds the case tolerance. A control that cannot fail is worthless;
//! these must keep failing.

use autoeq_qa::{complex_rel_error, golden_dir};
use num_complex::Complex64;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn complex_pairs(payload: &serde_json::Value) -> Vec<Complex64> {
    serde_json::from_value::<Vec<[f64; 2]>>(payload["response_re_im"].clone())
        .expect("golden must carry response_re_im")
        .into_iter()
        .map(|pair| Complex64::new(pair[0], pair[1]))
        .collect()
}

/// O07: wrong conjugation (Exp[+I w] instead of Exp[-I w]) must blow the
/// 1e-9 complex tolerance, not hide inside a magnitude-only comparison.
#[test]
fn o07_wrong_conjugation_fails_crossover_tolerance() {
    let payload = load_golden("o07_crossover_sum");
    let reference = complex_pairs(&payload);
    let worst = reference
        .iter()
        .map(|z| complex_rel_error(z.conj(), *z))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "conjugated crossover sum must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// O07: an off-by-one grid (comparing shifted against unshifted) must fail:
/// zipping unequal grids must never read as agreement.
#[test]
fn o07_off_by_one_grid_fails_crossover_tolerance() {
    let payload = load_golden("o07_crossover_sum");
    let reference = complex_pairs(&payload);
    let worst = reference
        .iter()
        .zip(reference.iter().skip(1))
        .map(|(a, b)| complex_rel_error(*b, *a))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "shifted crossover grid must exceed the 1e-9 tolerance, got {worst:.3e}"
    );
}

/// O07: a 10*log10 (power) vs 20*log10 (pressure) mix-up must be visible
/// in the combined magnitudes.
#[test]
fn o07_power_vs_pressure_db_factor_fails_magnitudes() {
    let payload = load_golden("o07_crossover_sum");
    let reference = complex_pairs(&payload);
    let worst = reference
        .iter()
        .map(|z| {
            let mag = z.norm().max(1e-12);
            (20.0 * mag.log10() - 10.0 * mag.log10()).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-8,
        "10-vs-20 dB factor must be visible in crossover magnitudes, got {worst:.3e} dB"
    );
}

/// O08: summing seat SPL values as pressures (20log10 of the pressure
/// mean) instead of power-averaging must differ observably: the fixture
/// seats disagree, so the wrong average cannot coincide.
#[test]
fn o08_spl_as_pressure_sum_fails_joint_objective() {
    let payload = load_golden("o08_joint_multisub");
    let seats: Vec<Vec<f64>> = serde_json::from_value(payload["seat_levels_db"].clone()).unwrap();
    let worst = seats[0]
        .iter()
        .zip(seats[1].iter())
        .map(|(&a, &b)| {
            let power_mean =
                10.0 * ((10.0_f64.powf(a / 10.0) + 10.0_f64.powf(b / 10.0)) / 2.0).log10();
            let pressure_mean =
                20.0 * ((10.0_f64.powf(a / 20.0) + 10.0_f64.powf(b / 20.0)) / 2.0).log10();
            (power_mean - pressure_mean).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "SPL-as-pressure averaging must differ from power averaging, got {worst:.3e} dB"
    );
}

/// O08: a sign flip on the variation component must blow the total.
#[test]
fn o08_sign_flipped_variation_fails_total() {
    let payload = load_golden("o08_joint_multisub");
    let variation = payload["variation"].as_f64().unwrap();
    let total = payload["total"].as_f64().unwrap();
    let defected = total - 2.0 * variation;
    let gap = (defected - total).abs();
    assert!(
        gap > 1e-9,
        "sign-flipped variation must move the total, got {gap:.3e}"
    );
}

/// O09: skipping the unwrap (raw wrapped backward differences) must blow
/// the group-delay tolerance at the wrapped jump.
#[test]
fn o09_missing_unwrap_fails_group_delay() {
    let payload = load_golden("o09_group_delay");
    let freqs: Vec<f64> = serde_json::from_value(payload["freqs_hz"].clone()).unwrap();
    let wrapped: Vec<f64> = serde_json::from_value(payload["phase_wrapped_deg"].clone()).unwrap();
    let want: Vec<f64> = serde_json::from_value(payload["group_delay_ms"].clone()).unwrap();
    let mut worst = 0.0f64;
    for i in 1..freqs.len() {
        let raw = -(wrapped[i] - wrapped[i - 1]).to_radians()
            / (2.0 * std::f64::consts::PI * (freqs[i] - freqs[i - 1]))
            * 1000.0;
        worst = worst.max((raw - want[i]).abs());
    }
    assert!(
        worst > 1e-9,
        "un-unwrapped stencil must exceed the 1e-9 ms tolerance, got {worst:.3e} ms"
    );
}

/// O09: a flipped delay sign (tau = +dphi/domega) must fail everywhere
/// the true delay is nonzero.
#[test]
fn o09_flipped_delay_sign_fails_group_delay() {
    let payload = load_golden("o09_group_delay");
    let want: Vec<f64> = serde_json::from_value(payload["group_delay_ms"].clone()).unwrap();
    let worst = want.iter().map(|v| (2.0 * v).abs()).fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "sign-flipped group delay must exceed the tolerance, got {worst:.3e} ms"
    );
}

/// O10: a 0-based band index in the sharpness numerator (z-1 instead of z)
/// must move the two-tone acum value observably.
#[test]
fn o10_zero_based_band_index_fails_sharpness() {
    let payload = load_golden("o10_bark_sharpness");
    let specific: [f64; 24] = serde_json::from_value(payload["specific_loudness"].clone()).unwrap();
    let weights: Vec<f64> = serde_json::from_value(payload["sharpness_weights"].clone()).unwrap();
    let total: f64 = specific.iter().sum();
    let correct = 0.11
        * (0..24)
            .map(|z| specific[z] * weights[z] * (z + 1) as f64)
            .sum::<f64>()
        / total.max(0.001);
    let defected = 0.11
        * (0..24)
            .map(|z| specific[z] * weights[z] * z as f64)
            .sum::<f64>()
        / total.max(0.001);
    let gap = (correct - defected).abs();
    assert!(
        gap > 1e-9,
        "0-based band index must move sharpness, got {gap:.3e} acum"
    );
}

/// O10: a wrong Zwicker coefficient (13 -> 12 on the first arctan term)
/// must blow the Bark tolerance at the probe freqs.
#[test]
fn o10_wrong_zwicker_coefficient_fails_bark() {
    let payload = load_golden("o10_bark_sharpness");
    let probes: Vec<f64> = serde_json::from_value(payload["probe_freqs_hz"].clone()).unwrap();
    let want: Vec<f64> = serde_json::from_value(payload["bark_values"].clone()).unwrap();
    let worst = probes
        .iter()
        .zip(want.iter())
        .map(|(&f, &w)| {
            let defected = 12.0 * (0.00076 * f).atan() + 3.5 * ((f / 7500.0).powi(2)).atan();
            ((defected - w) / w).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "wrong Zwicker coefficient must exceed the Bark tolerance, got {worst:.3e}"
    );
}

/// O11: swapping pre/post audible energies must fail: the fixture rings
/// louder before the peak than after it.
#[test]
fn o11_pre_post_swap_fails_temporal_metrics() {
    let payload = load_golden("o11_temporal_ir");
    let pre = payload["pre_ringing_audible_db"].as_f64().unwrap();
    let post = payload["post_ringing_audible_db"].as_f64().unwrap();
    let gap = (pre - post).abs();
    assert!(
        gap > 1e-9,
        "pre/post audible energies must differ, got {gap:.3e} dB"
    );
}

/// O11: a wrong sample rate in the ms conversion (44100 instead of 48000)
/// must move the main-peak time observably.
#[test]
fn o11_wrong_sample_rate_fails_ir_timing() {
    let payload = load_golden("o11_temporal_ir");
    let main_time = payload["main_time_ms"].as_f64().unwrap();
    let defected = 48.0 * 1000.0 / 44100.0;
    let gap = (defected - main_time).abs();
    assert!(
        gap > 1e-9,
        "wrong-rate main time must differ, got {gap:.3e} ms"
    );
}

/// O12: dominance without the convergence tie-break keeps the failed twin
/// D(5,2,F) alongside B(5,2,T): the naive front {A,B,D} must differ from
/// the blessed {A,B}.
#[test]
fn o12_missing_tie_break_fails_pareto_front() {
    let payload = load_golden("o12_pareto_mixed");
    let want: Vec<usize> = serde_json::from_value(payload["front_indices"].clone()).unwrap();
    // Naive front without tie-break or non-finite filtering.
    let losses = [10.0, 5.0, 20.0, 5.0, f64::NAN, 8.0];
    let counts = [1, 2, 3, 2, 1, 2];
    let mut naive = Vec::new();
    for (i, (&l, &c)) in losses.iter().zip(counts.iter()).enumerate() {
        if !l.is_finite() {
            continue;
        }
        let dominated = losses
            .iter()
            .zip(counts.iter())
            .enumerate()
            .any(|(j, (&lj, &cj))| {
                j != i && lj.is_finite() && lj <= l && cj <= c && (lj < l || cj < c)
            });
        if !dominated {
            naive.push(i);
        }
    }
    assert!(
        naive != want,
        "naive front {naive:?} must differ from the tie-broken {want:?}"
    );
    assert!(
        naive.contains(&3),
        "naive front must wrongly retain the failed twin, got {naive:?}"
    );
}

/// O12: fitting the slope against log10 instead of log2 (dB per decade
/// instead of per octave) must fail the tilt fixture observably.
#[test]
fn o12_decade_instead_of_octave_fails_slope() {
    let payload = load_golden("o12_pareto_mixed");
    let want = payload["lw_slope_db_per_oct"].as_f64().unwrap();
    let freqs: Vec<f64> = serde_json::from_value(payload["freqs_hz"].clone()).unwrap();
    let lw: Vec<f64> = serde_json::from_value(payload["lw_db"].clone()).unwrap();
    let xs: Vec<f64> = freqs.iter().map(|f| f.log10()).collect();
    let n = xs.len() as f64;
    let (sx, sy): (f64, f64) = (xs.iter().sum(), lw.iter().sum());
    let sxy: f64 = xs.iter().zip(lw.iter()).map(|(x, y)| x * y).sum();
    let sx2: f64 = xs.iter().map(|x| x * x).sum();
    let decade_slope = (sxy - sx * sy / n) / (sx2 - sx * sx / n);
    assert!(
        (decade_slope - want).abs() > 1e-9,
        "decade slope {decade_slope:.6} must differ from {want:.6} dB/oct"
    );
}
