/// Half-support of the band-limited delay kernel. Its specified usable band is
/// 0..=0.46 * sample_rate, with <= 0.01 dB magnitude error. Nyquist itself is
/// deliberately excluded: a finite real FIR cannot implement an arbitrary
/// fractional delay there.
pub const GD_DELAY_KERNEL_HALF: usize = 64;
pub const GD_DELAY_MAX_NORMALIZED_FREQUENCY: f64 = 0.46;
pub const GD_DELAY_MAGNITUDE_TOLERANCE_DB: f64 = 0.01;

/// Common integer padding needed to realize a set of requested delays without
/// cropping any leading input coefficients. Use the maximum for the entire
/// alignment group, including its zero-delay reference.
pub fn gd_delay_padding_samples(delays_ms: &[f64], sample_rate: f64) -> usize {
    delays_ms
        .iter()
        .map(|delay| {
            let shift = delay * sample_rate / 1000.0;
            let fractional = (shift - shift.round()).abs() > 1e-9;
            let earliest = if fractional {
                shift.floor() - GD_DELAY_KERNEL_HALF as f64
            } else {
                shift.round()
            };
            (-earliest).max(0.0).ceil() as usize
        })
        .max()
        .unwrap_or(0)
}

/// Realized delay and the latency needed to keep its whole kernel causal.
#[derive(Debug, Clone)]
pub struct GdFirDelay {
    pub coefficients: Vec<f64>,
    pub effective_delay_ms: f64,
    pub common_padding_samples: usize,
}

/// Apply a delay with explicitly allocated common causal support. The common
/// padding must be identical for every aligned channel; it preserves relative
/// timing, and must appear in reported/exported absolute latency.
pub fn realize_gd_fir_delay(
    coeffs: &[f64],
    delay_ms: f64,
    sample_rate: f64,
    common_padding_samples: usize,
) -> Result<GdFirDelay, String> {
    if !sample_rate.is_finite()
        || sample_rate <= 0.0
        || !delay_ms.is_finite()
        || coeffs.is_empty()
        || coeffs.iter().any(|c| !c.is_finite())
    {
        return Err("invalid FIR delay input".into());
    }
    let required = gd_delay_padding_samples(&[delay_ms], sample_rate);
    if common_padding_samples < required {
        return Err(format!(
            "FIR delay requires at least {required} common padding samples"
        ));
    }
    let shift = delay_ms * sample_rate / 1000.0 + common_padding_samples as f64;
    let nearest = shift.round();
    let coefficients = if (shift - nearest).abs() <= 1e-9 {
        apply_sample_shift(coeffs, nearest as isize)
    } else {
        let base = shift.floor() as usize - GD_DELAY_KERNEL_HALF;
        let fraction = shift - shift.floor();
        let kernel: Vec<f64> = (0..=2 * GD_DELAY_KERNEL_HALF)
            .map(|index| {
                let x = index as f64 - GD_DELAY_KERNEL_HALF as f64 - fraction;
                // Symmetric Blackman window around the fractional kernel center.
                let radius = GD_DELAY_KERNEL_HALF as f64;
                let window = if x.abs() <= radius {
                    0.42 + 0.5 * (std::f64::consts::PI * x / radius).cos()
                        + 0.08 * (2.0 * std::f64::consts::PI * x / radius).cos()
                } else {
                    0.0
                };
                sinc(x) * window
            })
            .collect();
        let dc: f64 = kernel.iter().sum();
        let mut output = vec![0.0; base + coeffs.len() + kernel.len() - 1];
        for (index, coefficient) in coeffs.iter().enumerate() {
            if *coefficient == 0.0 {
                continue;
            }
            for (tap, weight) in kernel.iter().enumerate() {
                output[base + index + tap] += coefficient * weight / dc;
            }
        }
        output
    };
    Ok(GdFirDelay {
        coefficients,
        effective_delay_ms: shift * 1000.0 / sample_rate,
        common_padding_samples,
    })
}

/// Convenience realization for a single FIR. This adds the causal padding
/// returned by `gd_delay_padding_samples(&[delay_ms], sample_rate)`; the actual
/// delay is requested delay plus that padding. Group callers must use
/// `realize_gd_fir_delay` with shared padding and retain its latency metadata.
pub fn apply_gd_delay_to_fir_coefficients(
    coeffs: &[f64],
    delay_ms: f64,
    sample_rate: f64,
) -> Vec<f64> {
    let padding = gd_delay_padding_samples(&[delay_ms], sample_rate);
    realize_gd_fir_delay(coeffs, delay_ms, sample_rate, padding)
        .expect("valid FIR delay coefficients, rate and delay")
        .coefficients
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

fn sinc(value: f64) -> f64 {
    if value.abs() < 1e-12 {
        1.0
    } else {
        let angle = std::f64::consts::PI * value;
        angle.sin() / angle
    }
}
