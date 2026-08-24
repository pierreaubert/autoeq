use math_audio_iir_fir::Biquad;
use math_audio_optimisation::continuous_area::Prior;
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Find bracketing indices for interpolation
pub(super) fn find_bracket_indices(freqs: &Array1<f64>, target: f64) -> (usize, usize) {
    for i in 0..freqs.len().saturating_sub(1) {
        if freqs[i] <= target && freqs[i + 1] >= target {
            return (i, i + 1);
        }
    }

    if target <= freqs[0] {
        (0, 0)
    } else {
        let last = freqs.len().saturating_sub(1);
        (last, last)
    }
}

pub(super) fn allpass_complex_response(biquad: &Biquad, freq_hz: f64) -> Complex64 {
    let (a1, a2, b0, b1, b2) = biquad.constants();
    let omega = 2.0 * PI * freq_hz / biquad.srate;
    let z_inv = Complex64::from_polar(1.0, -omega);
    let z_inv2 = z_inv * z_inv;

    let numerator = b0 + b1 * z_inv + b2 * z_inv2;
    let denominator = 1.0 + a1 * z_inv + a2 * z_inv2;

    numerator / denominator
}

/// Seat-to-seat variance: mean of per-frequency std-dev across seats (dB).
pub(super) fn variance_from_responses(responses: &[Vec<f64>]) -> f64 {
    let num_freqs = responses[0].len();
    let mut total_std = 0.0;

    for freq_idx in 0..num_freqs {
        let mean: f64 = responses.iter().map(|s| s[freq_idx]).sum::<f64>() / responses.len() as f64;
        let variance = responses
            .iter()
            .map(|s| (s[freq_idx] - mean).powi(2))
            .sum::<f64>()
            / responses.len() as f64;
        total_std += variance.sqrt();
    }

    total_std / num_freqs as f64
}

pub(super) fn weighted_variance_from_responses(responses: &[Vec<f64>], weights: &[f64]) -> f64 {
    if responses.is_empty() || responses.iter().any(Vec::is_empty) {
        return f64::INFINITY;
    }
    let total_weight: f64 = weights.iter().copied().sum();
    if total_weight <= f64::EPSILON {
        return variance_from_responses(responses);
    }
    (0..responses[0].len())
        .map(|frequency| {
            let mean = responses
                .iter()
                .zip(weights.iter().copied().chain(std::iter::repeat(0.0)))
                .map(|(seat, weight)| weight * seat[frequency])
                .sum::<f64>()
                / total_weight;
            let variance = responses
                .iter()
                .zip(weights.iter().copied().chain(std::iter::repeat(0.0)))
                .map(|(seat, weight)| weight * (seat[frequency] - mean).powi(2))
                .sum::<f64>()
                / total_weight;
            variance.sqrt()
        })
        .sum::<f64>()
        / responses[0].len() as f64
}

pub(super) fn peak_response_curve(responses: &[Vec<f64>]) -> Vec<f64> {
    let num_freqs = responses[0].len();
    (0..num_freqs)
        .map(|fi| {
            responses
                .iter()
                .map(|seat| seat[fi])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect()
}

/// RMS of the positive entries of `violations`, normalised by the number of
/// *violating* bins. Returns 0 when nothing violates.
///
/// Why per-violation rather than per-bin: the same physical violation must
/// score the same on a coarse and a fine frequency grid. Dividing by the
/// total bin count would let a sharp peak look smaller simply because more
/// non-violating bins were averaged in.
pub(super) fn violation_rms_db<I: IntoIterator<Item = f64>>(violations: I) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for v in violations {
        if v > 0.0 {
            sum_sq += v * v;
            count += 1;
        }
    }
    if count > 0 {
        (sum_sq / count as f64).sqrt()
    } else {
        0.0
    }
}

pub(super) fn single_seat_flatness(combined: &[Vec<f64>]) -> f64 {
    // `combined` from `compute_combined_responses` is `[seat][freq]`; we
    // built it with seat-count = 1, so take seat 0 and compute the std of SPL.
    if combined.is_empty() || combined[0].is_empty() {
        return f64::INFINITY;
    }
    let row = &combined[0];
    let n = row.len() as f64;
    let mean: f64 = row.iter().sum::<f64>() / n;
    let variance: f64 = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

/// Seeded Sobol points in `[0, 1)^D` for the dimensions supported by the
/// continuous-listening-area dispatcher.
pub(super) fn sobol_unit<const D: usize>(num_points: usize, seed: u64) -> Vec<[f64; D]> {
    assert!(
        (1..=3).contains(&D),
        "Sobol implementation supports D=1..=3"
    );

    let mut directions = [[0_u64; 64]; D];
    for (bit, direction) in directions[0].iter_mut().enumerate() {
        *direction = 1_u64 << (63 - bit);
    }
    if D >= 2 {
        directions[1][0] = 1_u64 << 63;
        for bit in 1..64 {
            directions[1][bit] = directions[1][bit - 1] ^ (directions[1][bit - 1] >> 1);
        }
    }
    if D >= 3 {
        directions[2][0] = 1_u64 << 63;
        directions[2][1] = 3_u64 << 62;
        for bit in 2..64 {
            directions[2][bit] =
                directions[2][bit - 2] ^ (directions[2][bit - 2] >> 2) ^ directions[2][bit - 1];
        }
    }

    let digital_shift: [u64; D] = std::array::from_fn(|dimension| {
        if seed == 0 {
            0
        } else {
            splitmix64(seed ^ (dimension as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        }
    });
    let mut state = [0_u64; D];
    (1..=num_points)
        .map(|index| {
            let direction = (index - 1).trailing_ones() as usize;
            std::array::from_fn(|dimension| {
                state[dimension] ^= directions[dimension][direction];
                let shifted = state[dimension] ^ digital_shift[dimension];
                ((shifted >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
            })
        })
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Transform seeded Sobol points according to a supported continuous-area
/// prior. Custom-density priors fall back to the dependency implementation.
pub(super) fn sobol_quadrature_points<const D: usize>(
    prior: &Prior<D>,
    num_points: usize,
    seed: u64,
) -> Option<(Vec<[f64; D]>, Vec<f64>)> {
    if num_points == 0 {
        return None;
    }
    let unit = sobol_unit::<D>(num_points, seed);
    let points = match prior {
        Prior::Uniform { bounds } => unit
            .into_iter()
            .map(|sample| {
                std::array::from_fn(|dimension| {
                    let (lo, hi) = bounds[dimension];
                    lo + sample[dimension] * (hi - lo)
                })
            })
            .collect(),
        Prior::Gaussian {
            mean,
            cov_diag,
            truncation_sigmas,
        } => {
            let probability_lo = standard_normal_cdf(-*truncation_sigmas);
            let probability_hi = standard_normal_cdf(*truncation_sigmas);
            unit.into_iter()
                .map(|sample| {
                    std::array::from_fn(|dimension| {
                        let probability =
                            probability_lo + sample[dimension] * (probability_hi - probability_lo);
                        let sigma = cov_diag[dimension].sqrt();
                        let lo = mean[dimension] - truncation_sigmas * sigma;
                        let hi = mean[dimension] + truncation_sigmas * sigma;
                        (mean[dimension] + sigma * inv_standard_normal(probability)).clamp(lo, hi)
                    })
                })
                .collect()
        }
        Prior::Custom { .. } => return None,
    };
    Some((points, vec![1.0 / num_points as f64; num_points]))
}

fn standard_normal_cdf(value: f64) -> f64 {
    0.5 * (1.0 + erf(value / std::f64::consts::SQRT_2))
}

#[allow(clippy::excessive_precision)]
fn inv_standard_normal(probability: f64) -> f64 {
    let probability = probability.clamp(1e-12, 1.0 - 1e-12);
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const TAIL: f64 = 0.024_25;

    if probability < TAIL {
        let q = (-2.0 * probability.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if probability <= 1.0 - TAIL {
        let q = probability - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - probability).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

fn erf(value: f64) -> f64 {
    let sign = value.signum();
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial =
        (((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t)
            * (-x * x).exp();
    sign * (1.0 - polynomial)
}

#[cfg(test)]
mod sobol_tests {
    use super::*;

    #[test]
    fn unshifted_sobol_has_reference_first_points() {
        let points = sobol_unit::<2>(4, 0);
        assert_eq!(
            points,
            vec![[0.5, 0.5], [0.75, 0.25], [0.25, 0.75], [0.375, 0.375]]
        );
    }

    #[test]
    fn sobol_seed_is_deterministic_and_effective() {
        let first = sobol_unit::<3>(16, 41);
        assert_eq!(first, sobol_unit::<3>(16, 41));
        assert_ne!(first, sobol_unit::<3>(16, 42));
        assert!(
            first
                .iter()
                .flatten()
                .all(|value| (0.0..1.0).contains(value))
        );
    }

    #[test]
    fn gaussian_sobol_quadrature_is_centered_and_normalized() {
        let prior = Prior::Gaussian {
            mean: [2.0],
            cov_diag: [0.25],
            truncation_sigmas: 3.0,
        };
        let (points, weights) = sobol_quadrature_points(&prior, 1024, 0).expect("supported prior");
        let weighted_mean: f64 = points
            .iter()
            .zip(&weights)
            .map(|(point, weight)| point[0] * weight)
            .sum();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((weighted_mean - 2.0).abs() < 0.01);
    }
}
