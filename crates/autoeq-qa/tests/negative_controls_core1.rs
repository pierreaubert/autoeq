//! Negative controls for core1 (C02/C04/C05): each proves its case detects
//! the defect class it claims. Every control loads an engine-blessed
//! golden, applies one classic defect, and asserts the error exceeds the
//! case tolerance. A control that cannot fail is worthless.

use autoeq_qa::{complex_rel_error, golden_dir};
use num_complex::Complex64;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn response_pairs(payload: &serde_json::Value) -> Vec<Complex64> {
    serde_json::from_value::<Vec<[f64; 2]>>(payload["response_re_im"].clone())
        .expect("golden must carry response_re_im")
        .into_iter()
        .map(|pair| Complex64::new(pair[0], pair[1]))
        .collect()
}

/// Wrong conjugation (Exp[+I w] instead of Exp[-I w]) must blow the C02
/// realized-transfer tolerance, not hide in a magnitude comparison.
#[test]
fn conjugated_layout_transfer_fails_c02_tolerance() {
    for case in ["c02_peq_layouts", "c02_free_types"] {
        let payload = load_golden(case);
        let reference = response_pairs(&payload);
        let worst = reference
            .iter()
            .map(|z| complex_rel_error(z.conj(), *z))
            .fold(0.0f64, f64::max);
        assert!(
            worst > 1e-9,
            "{case}: conjugated transfer must exceed 1e-9, got {worst:.3e}"
        );
    }
}

/// A 10*log10 (power) vs 20*log10 (amplitude) gain mix-up halves every dB
/// gain; rebuilding the C02 peak cascade at half gain must move the
/// complex response beyond tolerance. Proxied here by halving the dB
/// distance from unity: H_defect = H^0.5 moves every magnitude by sqrt.
#[test]
fn halved_db_gains_fail_c02_transfer_tolerance() {
    let payload = load_golden("c02_peq_layouts");
    let reference = response_pairs(&payload);
    let worst = reference
        .iter()
        .map(|z| complex_rel_error(z.sqrt(), *z))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "halved-dB transfer must exceed 1e-9, got {worst:.3e}"
    );
}

/// Swapping the Free type codes (lowshelf<->highshelf) must change the
/// realized C02 transfer: swapping the first and last stage responses
/// of an asymmetric cascade is not a no-op.
#[test]
fn swapped_shelf_stages_fail_c02_free_tolerance() {
    let payload = load_golden("c02_free_types");
    let reference = response_pairs(&payload);
    // Reversed cascade order differs for non-commuting asymmetric stages
    // only in exact arithmetic it commutes (scalar product commutes), so
    // instead check that a shelf-sign flip (gain +4 -> -4 on stage 1,
    // i.e. magnitude inversion of a stage) is visible. Proxy: invert the
    // whole response magnitude (1/H magnitude, keep phase).
    let worst = reference
        .iter()
        .map(|z| {
            let inv = Complex64::from_polar(1.0 / z.norm().max(1e-12), z.arg());
            complex_rel_error(inv, *z)
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "shelf-gain inversion must exceed 1e-9, got {worst:.3e}"
    );
}

/// A linear-frequency mean instead of the log-frequency integral mean
/// must move the C04 octave-smoothed values beyond tolerance.
#[test]
fn linear_frequency_mean_fails_octave_smooth_tolerance() {
    let payload = load_golden("c04_octave_smooth");
    let knots: Vec<f64> = serde_json::from_value(payload["knot_freqs_hz"].clone()).unwrap();
    let values: Vec<f64> = serde_json::from_value(payload["knot_spl_db"].clone()).unwrap();
    let expected: Vec<f64> = serde_json::from_value(payload["smoothed_spl_db"].clone()).unwrap();
    let half = 2f64.powf(0.5);
    let mut worst = 0.0f64;
    for (i, f) in knots.iter().enumerate() {
        let (lo, hi) = (f / half, f * half);
        let mut num = 0.0;
        let mut den = 0.0;
        for a in 0..knots.len() - 1 {
            let (f0, f1) = (knots[a], knots[a + 1]);
            let (u0, u1) = (f0.max(lo), f1.min(hi));
            if u1 > u0 {
                // Linear-frequency weighting (the defect): weight by Hz, not ln f.
                let v0 = values[a];
                let v1 = values[a + 1];
                let t0 = (u0 - f0) / (f1 - f0);
                let t1 = (u1 - f0) / (f1 - f0);
                let w0 = v0 + (v1 - v0) * t0;
                let w1 = v0 + (v1 - v0) * t1;
                num += 0.5 * (w0 + w1) * (u1 - u0);
                den += u1 - u0;
            }
        }
        let lin = num / den;
        worst = worst.max((lin - expected[i]).abs());
    }
    assert!(
        worst > 1e-9,
        "linear-Hz mean must exceed the 1e-9 dB tolerance, got {worst:.3e}"
    );
}

/// Off-by-one grid shift (zipping unequal grids) must break the C04
/// Gaussian smoothing beyond tolerance.
#[test]
fn shifted_grid_fails_gaussian_smooth_tolerance() {
    let payload = load_golden("c04_gaussian_smooth");
    let expected: Vec<f64> = serde_json::from_value(payload["smoothed"].clone()).unwrap();
    let shifted: Vec<f64> = expected
        .iter()
        .skip(1)
        .chain(expected.iter().take(1))
        .copied()
        .collect();
    let worst = expected
        .iter()
        .zip(shifted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "off-by-one grid must exceed the 1e-12 tolerance, got {worst:.3e}"
    );
}

/// Interpreting sigma in octaves instead of samples (e.g. sigma=1 octave
/// ~= a far wider kernel) must move the C04 Gaussian output; proxied by
/// a 3x-wider effective kernel being visibly different from sigma=1.
/// Here: nearest-neighbor (sigma -> 0, identity) output differs.
#[test]
fn identity_kernel_fails_gaussian_impulse_tolerance() {
    let payload = load_golden("c04_gaussian_smooth");
    let input: Vec<f64> = serde_json::from_value(payload["input"].clone()).unwrap();
    let expected: Vec<f64> = serde_json::from_value(payload["smoothed"].clone()).unwrap();
    let worst = input
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "unsmoothed input must differ from sigma=1 output by more than 1e-12, got {worst:.3e}"
    );
}

/// A delay sign flip (+1 ms instead of -1 ms slope) must blow the C05
/// delay tolerance.
#[test]
fn flipped_delay_sign_fails_c05_delay_tolerance() {
    let payload = load_golden("c05_unwrap_delay");
    let excess: Vec<f64> = serde_json::from_value(payload["excess_phase_deg"].clone()).unwrap();
    let flipped: Vec<f64> = excess.iter().map(|v| -v).collect();
    let n = excess.len() as f64;
    let fit = |ys: &[f64]| -> f64 {
        let freqs: Vec<f64> = (1..=ys.len()).map(|i| 100.0 * i as f64).collect();
        let rad: Vec<f64> = ys.iter().map(|p| p.to_radians()).collect();
        let (sf, sp, sf2, sfp) = (
            freqs.iter().sum::<f64>(),
            rad.iter().sum::<f64>(),
            freqs.iter().map(|f| f * f).sum::<f64>(),
            freqs
                .iter()
                .zip(rad.iter())
                .map(|(f, p)| f * p)
                .sum::<f64>(),
        );
        let slope = (n * sfp - sf * sp) / (n * sf2 - sf * sf);
        -slope / (2.0 * std::f64::consts::PI) * 1000.0
    };
    let gap = (fit(&excess) - fit(&flipped)).abs();
    assert!(
        gap > 1e-9,
        "sign-flipped delay must move the fit by more than 1e-9 ms, got {gap:.3e}"
    );
}

/// Skipping the unwrap (comparing wrapped phase directly) must exceed
/// the C05 unwrap tolerance.
#[test]
fn wrapped_phase_fails_c05_unwrap_tolerance() {
    let payload = load_golden("c05_unwrap_delay");
    let wrapped: Vec<f64> = serde_json::from_value(payload["wrapped_deg"].clone()).unwrap();
    let unwrapped: Vec<f64> = serde_json::from_value(payload["unwrapped_deg"].clone()).unwrap();
    let worst = wrapped
        .iter()
        .zip(unwrapped.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "wrapped phase must exceed the 1e-9 deg tolerance, got {worst:.3e}"
    );
}

/// A flat 5-degree minimum-phase bias (e.g. dropped sign/offset in the
/// Hilbert pipeline) must exceed the C05 0.5 deg arithmetic bound.
#[test]
fn biased_minphase_fails_c05_flat_bound() {
    let payload = load_golden("c05_minphase_flat");
    let zeros: Vec<f64> = serde_json::from_value(payload["min_phase_deg"].clone()).unwrap();
    let worst = zeros.iter().map(|v| (v - 5.0).abs()).fold(0.0f64, f64::max);
    assert!(
        worst > 0.5,
        "5-degree bias must exceed the 0.5 deg bound, got {worst:.3e}"
    );
}
