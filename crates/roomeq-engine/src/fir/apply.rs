/// Apply a GD-alignment delay to existing FIR coefficients.
///
/// A pure delay is `H(f) * exp(-j 2 pi f tau)`: unity magnitude at every
/// frequency. To honour that contract the filter support is extended for
/// positive delays instead of silently discarding coefficients that move past
/// the end of the array. Fractional shifts use a Lanczos-windowed sinc
/// interpolator with zero padding and full-kernel normalization so edge taps
/// are interpolated rather than renormalized to unity (F02).
pub fn apply_gd_delay_to_fir_coefficients(
    coeffs: &[f64],
    delay_ms: f64,
    sample_rate: f64,
) -> Vec<f64> {
    if delay_ms.abs() <= 1e-6 {
        return coeffs.to_vec();
    }
    let delay_samples = delay_ms * 1e-3 * sample_rate;
    apply_fractional_sample_shift(coeffs, delay_samples)
}

/// Shift FIR coefficients by a given number of samples (positive = later).
///
/// Positive shifts extend the coefficient vector so no energy is discarded.
/// Negative shifts advance the sequence, dropping leading taps and zero
/// padding the tail to preserve length.
#[allow(dead_code)]
pub(super) fn apply_sample_shift(coeffs: &[f64], shift: isize) -> Vec<f64> {
    let n = coeffs.len();
    if shift >= 0 {
        let s = shift as usize;
        let mut shifted = vec![0.0; n + s];
        shifted[s..s + n].copy_from_slice(coeffs);
        shifted
    } else {
        let s = (-shift) as usize;
        let mut shifted = vec![0.0; n];
        let len = n.saturating_sub(s);
        if len > 0 {
            shifted[..len].copy_from_slice(&coeffs[s..s + len]);
        }
        shifted
    }
}

/// Shift FIR coefficients by a fractional number of samples using a
/// 16-tap Lanczos-windowed sinc interpolator. Positive shift = later.
///
/// The interpolator zero-pads outside the coefficient array and normalizes by
/// the full kernel weight (including out-of-bounds taps). Renormalizing by
/// only the in-bounds weights boosts edge energy: for a leading impulse that
/// produced +4.45 dB of spurious gain. Positive shifts extend support by
/// `ceil(shift)` samples so large delays remain unity-magnitude instead of
/// collapsing to silence.
pub(super) fn apply_fractional_sample_shift(coeffs: &[f64], shift: f64) -> Vec<f64> {
    let n = coeffs.len();
    if shift.abs() < 1e-9 {
        return coeffs.to_vec();
    }
    let integer_shift = shift.round();
    if (shift - integer_shift).abs() < 1e-9 {
        return apply_sample_shift(coeffs, integer_shift as isize);
    }
    const HALF_WIDTH: isize = 8;
    let output_len = if shift > 0.0 {
        n + shift.ceil() as usize
    } else {
        n
    };
    let mut shifted = vec![0.0; output_len];
    for (i, output) in shifted.iter_mut().enumerate() {
        let src = i as f64 - shift;
        let base = src.floor() as isize;
        let frac = src - base as f64;
        let mut full_normalization = 0.0;
        let mut value = 0.0;
        for offset in (-HALF_WIDTH + 1)..=HALF_WIDTH {
            let distance = frac - offset as f64;
            let weight = sinc(distance) * sinc(distance / HALF_WIDTH as f64);
            full_normalization += weight;
            let index = base + offset;
            if (0..n as isize).contains(&index) {
                value += weight * coeffs[index as usize];
            }
        }
        if full_normalization.abs() > 1e-12 {
            *output = value / full_normalization;
        }
    }
    // A causal finite-length positive delay discards the ideal pre-ringing
    // before sample zero, which raises LF gain for leading-edge energy
    // (about +1 dB for a leading impulse at half-sample delay). A pure
    // delay preserves DC gain and positive shifts extend support so no input
    // energy is legitimately removed; restore the coefficient sum with a
    // single global scale. This is distinct from per-tap renormalization,
    // which caused the +4.45 dB edge boost. Negative shifts legitimately drop
    // leading energy when advancing, so they are left unscaled.
    if shift > 0.0 {
        let input_sum: f64 = coeffs.iter().sum();
        let output_sum: f64 = shifted.iter().sum();
        if input_sum.abs() > 1e-9 && output_sum.abs() > 1e-12 {
            let scale = input_sum / output_sum;
            if scale.is_finite() && (scale - 1.0).abs() < 0.5 && scale > 0.0 {
                for output in shifted.iter_mut() {
                    *output *= scale;
                }
            }
        }
    }
    shifted
}

fn sinc(value: f64) -> f64 {
    if value.abs() < 1e-12 {
        1.0
    } else {
        let angle = std::f64::consts::PI * value;
        angle.sin() / angle
    }
}
