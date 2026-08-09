/// Apply a GD-alignment delay to existing FIR coefficients.
///
/// This is used when a production path has already generated FIRs before the
/// room-level GD target is known. It mirrors the delay handling in
/// `generate_fir_correction_with_gd_target` so both paths encode the same
/// sample-domain shift into the convolution IR.
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
/// Pads with zeros on the appropriate side and truncates to maintain length.
#[allow(dead_code)]
pub(super) fn apply_sample_shift(coeffs: &[f64], shift: isize) -> Vec<f64> {
    let n = coeffs.len();
    let mut shifted = vec![0.0; n];

    if shift >= 0 {
        let s = shift as usize;
        if s < n {
            shifted[s..n].copy_from_slice(&coeffs[..(n - s)]);
        }
    } else {
        let s = (-shift) as usize;
        let len = n.saturating_sub(s);
        if len > 0 {
            shifted[..len].copy_from_slice(&coeffs[s..s + len]);
        }
    }

    shifted
}

/// Shift FIR coefficients by a fractional number of samples using a
/// 16-tap Lanczos-windowed sinc interpolator. Positive shift = later.
///
/// The previous two-tap linear interpolator multiplied the complete FIR
/// response by a triangular-kernel response, causing severe high-frequency
/// droop. Windowed sinc approximates the ideal `e^-jΩD` fractional delay while
/// retaining a compact, deterministic kernel.
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
    let mut shifted = vec![0.0; n];
    for (i, output) in shifted.iter_mut().enumerate() {
        let src = i as f64 - shift;
        let base = src.floor() as isize;
        let frac = src - base as f64;
        let mut normalization = 0.0;
        let mut value = 0.0;
        for offset in (-HALF_WIDTH + 1)..=HALF_WIDTH {
            let distance = frac - offset as f64;
            let weight = sinc(distance) * sinc(distance / HALF_WIDTH as f64);
            normalization += weight;
            let index = base + offset;
            if (0..n as isize).contains(&index) {
                value += weight * coeffs[index as usize];
            }
        }
        if normalization.abs() > 1e-12 {
            *output = value / normalization;
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
