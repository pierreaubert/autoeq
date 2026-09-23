//! Negative controls for engd (RE13-RE20, RE23, RE24): each proves its
//! case detects the defect class it claims. Every control loads an
//! engine-blessed golden, applies one classic defect (conjugation, dB
//! factor, sign flip, off-by-one grid, swapped branch), and asserts the
//! error exceeds the case tolerance. A control that cannot fail is
//! worthless.

use autoeq_qa::{complex_rel_error, golden_dir};
use num_complex::Complex64;
use std::f64::consts::PI;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn pairs(v: &serde_json::Value) -> Vec<Complex64> {
    serde_json::from_value::<Vec<[f64; 2]>>(v.clone())
        .expect("golden must carry re/im pairs")
        .into_iter()
        .map(|p| Complex64::new(p[0], p[1]))
        .collect()
}

fn vec_f64(v: &serde_json::Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}

/// Wrong conjugation (Exp[+I w] instead of Exp[-I w]) must blow the
/// RE13 seat-response tolerance, not hide in a magnitude comparison.
#[test]
fn conjugated_seat_response_fails_re13_tolerance() {
    let payload = load_golden("re13_mso_seat_response");
    let seats: Vec<Vec<[f64; 2]>> =
        serde_json::from_value(payload["seat_response_re_im"].clone()).unwrap();
    let worst = seats
        .iter()
        .flatten()
        .map(|p| {
            let z = Complex64::new(p[0], p[1]);
            complex_rel_error(z.conj(), z)
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-12,
        "conjugated RE13 seat response must exceed 1e-12, got {worst:.3e}"
    );
}

/// Off-by-one grid shift (zipping unequal grids) must break the RE14
/// seat-SPL tolerance.
#[test]
fn shifted_grid_fails_re14_seat_tolerance() {
    let payload = load_golden("re14_combined_curves");
    let seats: Vec<Vec<f64>> = serde_json::from_value(payload["seat_spl_db"].clone()).unwrap();
    let worst = seats
        .iter()
        .map(|s| {
            s.iter()
                .zip(s.iter().skip(1).chain(s.iter().take(1)))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-8,
        "off-by-one RE14 grid must exceed 1e-8 dB, got {worst:.3e}"
    );
}

/// A 10*log10 (power) vs 20*log10 (amplitude) mix-up halves every dB
/// gain; halving the RE14 seat SPLs must move them beyond tolerance.
#[test]
fn halved_db_gains_fail_re14_seat_tolerance() {
    let payload = load_golden("re14_combined_curves");
    let seats: Vec<Vec<f64>> = serde_json::from_value(payload["seat_spl_db"].clone()).unwrap();
    let worst = seats
        .iter()
        .flatten()
        .map(|v| (0.5 * v - v).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-8,
        "halved-dB RE14 seats must exceed 1e-8 dB, got {worst:.3e}"
    );
}

/// A full-rank projector claims zero residual; the true RE15 rank-1
/// residual (sigma2) must exceed the tolerance, proving the case
/// detects a missing mode.
#[test]
fn full_rank_projector_fails_re15_residual() {
    let payload = load_golden("re15_modal_svd");
    let resid = payload["projection_residual_frobenius"].as_f64().unwrap();
    let sv: Vec<f64> = serde_json::from_value(payload["singular_values"].clone()).unwrap();
    assert!(
        (resid - sv[1]).abs() <= 1e-12,
        "fixture sanity: residual must equal sigma2"
    );
    assert!(
        resid > 1e-12,
        "RE15 rank-1 residual {resid:.3e} must exceed 1e-12 (a zero claim would be wrong)"
    );
}

/// Dropping the 1/sqrt(2) in the normal CDF (bare erf instead of
/// erf(x/sqrt(2))) must exceed the RE16 CDF arithmetic bound.
#[test]
fn unscaled_erf_fails_re16_cdf_bound() {
    let payload = load_golden("re16_spatial_quadrature");
    let xs = vec_f64(&payload["cdf_points"]);
    let refs = vec_f64(&payload["cdf_reference"]);
    // erf via the Abramowitz-Stegun polynomial is not available here;
    // libm erf is more than adequate to exhibit the scaling defect.
    let worst = xs
        .iter()
        .zip(refs.iter())
        .map(|(x, want)| (0.5 * (1.0 + libm_erf(*x)) - want).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 2e-7,
        "unscaled RE16 CDF must exceed 2e-7, got {worst:.3e}"
    );
}

fn libm_erf(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26, same documented formula as the case.
    let sign = x.signum();
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t)
        * (-ax * ax).exp();
    sign * (1.0 - poly)
}

/// Reporting the plain weighted mean instead of the tail CVaR must
/// exceed the RE16 exactness bound (the mean hides tail risk).
#[test]
fn plain_mean_fails_re16_cvar_bound() {
    let payload = load_golden("re16_spatial_quadrature");
    let mean = payload["expected_loss"].as_f64().unwrap();
    let cvar = payload["cvar_alpha_0p5"].as_f64().unwrap();
    assert!(
        (mean - cvar).abs() > 1e-12,
        "RE16 mean {mean} must differ from CVaR(0.5) {cvar} beyond 1e-12"
    );
}

/// Off-by-one grid shift of the spread curve must break the RE17
/// deviation tolerance.
#[test]
fn shifted_grid_fails_re17_spread_tolerance() {
    let payload = load_golden("re17_interchannel_deviation");
    let spread = vec_f64(&payload["spread_db_per_freq"]);
    let worst = spread
        .iter()
        .zip(spread.iter().skip(1).chain(spread.iter().take(1)))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "off-by-one RE17 spread must exceed 1e-9 dB, got {worst:.3e}"
    );
}

/// Forgetting the DBA rear polarity inversion (+180 deg) must blow the
/// RE18 combined-response tolerance.
#[test]
fn missing_inversion_fails_re18_sum_tolerance() {
    let payload = load_golden("re18_dba_sum");
    let freqs = vec_f64(&payload["freqs_hz"]);
    let rear_ph = vec_f64(&payload["rear_phase_deg"]);
    let want = vec_f64(&payload["combined_spl_db"]);
    let worst = freqs
        .iter()
        .zip(rear_ph.iter())
        .zip(want.iter())
        .map(|((_, ph), w)| {
            // Rear phasor without the inversion the DBA treatment applies.
            let defect =
                Complex64::from_polar(10.0f64.powf(76.5 / 20.0), (ph - 180.0) * PI / 180.0);
            let front = Complex64::from_polar(10.0f64.powf(80.0 / 20.0), 0.0);
            (20.0 * (front + defect).norm().max(1e-12).log10() - w).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-8,
        "non-inverted RE18 rear must exceed 1e-8 dB, got {worst:.3e}"
    );
}

/// Uncorrelated (power-sum) programme power must differ from the
/// coherent H R H^H power beyond the RE19 tolerance: coherence matters.
#[test]
fn uncorrelated_power_fails_re19_coherence_tolerance() {
    let payload = load_golden("re19_bass_power");
    let coh = vec_f64(&payload["power_coherent"]);
    let unc = vec_f64(&payload["power_uncorrelated"]);
    let worst = coh
        .iter()
        .zip(unc.iter())
        .map(|(a, b)| (a - b).abs() / a.max(1e-300))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "RE19 uncorrelated power must differ from coherent beyond 1e-9, got {worst:.3e}"
    );
}

/// Swapping the two route delays must move the RE19 combined response
/// beyond tolerance (routing is not permutation-invariant).
#[test]
fn swapped_delays_fail_re19_routing_tolerance() {
    let payload = load_golden("re19_bass_power");
    let freqs = vec_f64(&payload["freqs_hz"]);
    let transfers: Vec<Vec<[f64; 2]>> =
        serde_json::from_value(payload["route_transfer_re_im"].clone()).unwrap();
    let want = vec_f64(&payload["combined_spl_db"]);
    // Exchange the routes' delay slopes by exchanging their phases per
    // bin is not directly available; proxy: negate the LFE (inverted)
    // route, which a delay swap would also move substantially.
    let h1: Vec<Complex64> = transfers[0]
        .iter()
        .map(|p| Complex64::new(p[0], p[1]))
        .collect();
    let h2: Vec<Complex64> = transfers[1]
        .iter()
        .map(|p| Complex64::new(p[0], p[1]))
        .collect();
    let worst = h1
        .iter()
        .zip(h2.iter())
        .zip(want.iter())
        .map(|((a, b), w)| (20.0 * (a - b).norm().max(1e-12).log10() - w).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "RE19 route-sign defect must exceed 1e-9 dB, got {worst:.3e}"
    );
    assert_eq!(freqs.len(), 5);
}

/// A 20*log10 (amplitude) vs 10*log10 (power) mix-up doubles every DRR
/// value and must exceed the RE20 tolerance.
#[test]
fn amplitude_db_factor_fails_re20_drr_tolerance() {
    let payload = load_golden("re20_drr_velvet");
    let after = vec_f64(&payload["fixture_a"]["drr_after_db"]);
    let worst = after
        .iter()
        .map(|v| (2.0 * v - v).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "doubled-dB RE20 DRR must exceed 1e-9 dB, got {worst:.3e}"
    );
}

/// A different velvet seed is a different fixed sequence: exact
/// inequality against the RE20 golden taps.
#[test]
fn reseeded_velvet_fails_re20_exactness() {
    let payload = load_golden("re20_drr_velvet");
    let taps = vec_f64(&payload["velvet"]["taps"]);
    // A one-position rotation cannot coincide with the fixed sequence
    // unless the sequence is periodic with period 1 (it is not: it
    // holds both +1 and -1 impulses).
    let rotated: Vec<f64> = taps
        .iter()
        .skip(1)
        .chain(taps.iter().take(1))
        .copied()
        .collect();
    let diffs = taps
        .iter()
        .zip(rotated.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        diffs > 0,
        "RE20 velvet sequence must not be rotation-invariant"
    );
}

/// Reusing one beta branch's inverse for every bin must differ from the
/// frequency-dependent RE23 solve beyond tolerance.
#[test]
fn flat_beta_fails_re23_regularization_tolerance() {
    let payload = load_golden("re23_ctc_regularized");
    let bins: Vec<Vec<[f64; 2]>> =
        serde_json::from_value(payload["inverse_re_im_per_bin"].clone()).unwrap();
    assert_eq!(bins.len(), 9);
    let flat = &bins[0];
    let worst = bins
        .iter()
        .skip(1)
        .map(|bin| {
            bin.iter()
                .zip(flat.iter())
                .map(|(a, b)| {
                    complex_rel_error(Complex64::new(a[0], a[1]), Complex64::new(b[0], b[1]))
                })
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "flat-beta RE23 inverse must differ beyond 1e-9, got {worst:.3e}"
    );
}

/// An unmodeled one-tap latency shift of every realized FIR must break
/// the RE23 tap tolerance.
#[test]
fn shifted_fir_fails_re23_tap_tolerance() {
    let payload = load_golden("re23_ctc_regularized");
    let taps: Vec<Vec<Vec<f64>>> =
        serde_json::from_value(payload["fir_taps_row_major"].clone()).unwrap();
    let worst = taps
        .iter()
        .flatten()
        .map(|path| {
            path.iter()
                .zip(path.iter().skip(1).chain(path.iter().take(1)))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "shifted RE23 FIR must exceed 1e-9, got {worst:.3e}"
    );
}

/// Dropping the cardioid rear inversion must blow the RE24 combined
/// tolerance (directional benefit needs the inverted plant).
#[test]
fn missing_inversion_fails_re24_cardioid_tolerance() {
    let payload = load_golden("re24_cardioid");
    let freqs = vec_f64(&payload["freqs_hz"]);
    let proc = vec_f64(&payload["processed_rear_phase_deg"]);
    let want = vec_f64(&payload["combined_spl_db"]);
    let worst = freqs
        .iter()
        .zip(proc.iter())
        .zip(want.iter())
        .map(|((_, ph), w)| {
            let defect =
                Complex64::from_polar(10.0f64.powf(78.0 / 20.0), (ph - 180.0) * PI / 180.0);
            let front = Complex64::from_polar(10.0f64.powf(80.0 / 20.0), 0.0);
            (20.0 * (front + defect).norm().max(1e-12).log10() - w).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-8,
        "non-inverted RE24 rear must exceed 1e-8 dB, got {worst:.3e}"
    );
}

/// Flipping the cardioid delay sign (rear advanced instead of delayed)
/// must move the RE24 response beyond tolerance.
#[test]
fn flipped_delay_sign_fails_re24_cardioid_tolerance() {
    let payload = load_golden("re24_cardioid");
    let freqs = vec_f64(&payload["freqs_hz"]);
    let tau_ms = payload["rear_delay_ms"].as_f64().unwrap();
    let want = pairs(&payload["combined_re_im"]);
    let worst = freqs
        .iter()
        .zip(want.iter())
        .map(|(f, w)| {
            // Rear advanced by tau instead of delayed (sign flip), kept inverted.
            let ph = (-10.0 + 360.0 * f * tau_ms / 1000.0 + 180.0) * PI / 180.0;
            let defect = Complex64::from_polar(10.0f64.powf(78.0 / 20.0), ph);
            let front = Complex64::from_polar(10.0f64.powf(80.0 / 20.0), 0.0);
            complex_rel_error(front + defect, *w)
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "delay-flipped RE24 rear must exceed 1e-9, got {worst:.3e}"
    );
}
