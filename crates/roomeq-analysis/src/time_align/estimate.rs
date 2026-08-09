use super::error::PhaseArrivalError;

/// Estimate speaker propagation delay from a frequency-domain phase measurement.
///
/// Uses linear regression on the unwrapped phase in the [min_freq, max_freq] band:
///   φ(f) ≈ φ₀ - 2π·τ·f  →  τ = -slope / (2π)
///
/// Returns the estimated arrival time in milliseconds, or None if phase data
/// is absent, no points fall in the band, or the estimate is implausible.
#[allow(dead_code)]
pub fn estimate_arrival_from_phase(
    curve: &autoeq_core::Curve,
    min_freq: f64,
    max_freq: f64,
) -> Option<f64> {
    estimate_arrival_from_phase_detailed(curve, min_freq, max_freq).ok()
}

/// Estimate speaker propagation delay from phase data and report why rejected.
pub fn estimate_arrival_from_phase_detailed(
    curve: &autoeq_core::Curve,
    min_freq: f64,
    max_freq: f64,
) -> Result<f64, PhaseArrivalError> {
    use std::f64::consts::PI;

    let phase = curve
        .phase
        .as_ref()
        .ok_or(PhaseArrivalError::MissingPhase)?;

    // Unwrap phase to remove discontinuities
    let unwrapped = autoeq_core::phase_utils::unwrap_phase_degrees(phase);

    // Filter to the [min_freq, max_freq] band
    let points: Vec<(f64, f64)> = curve
        .freq
        .iter()
        .zip(unwrapped.iter())
        .filter(|&(&f, _)| f >= min_freq && f <= max_freq)
        .map(|(&f, &p)| (f, p))
        .collect();

    if points.len() < 5 {
        return Err(PhaseArrivalError::InsufficientBandPoints {
            min_freq,
            max_freq,
            points: points.len(),
        });
    }

    // Linear regression in radians: φ_rad = φ₀ - 2π·τ·f. Weight each bin by
    // its represented linear-frequency interval so a log-spaced grid does not
    // give the densely sampled LF end disproportionate influence.
    let weights: Vec<f64> = (0..points.len())
        .map(|index| match index {
            0 => (points[1].0 - points[0].0) * 0.5,
            index if index + 1 == points.len() => (points[index].0 - points[index - 1].0) * 0.5,
            index => (points[index + 1].0 - points[index - 1].0) * 0.5,
        })
        .collect();
    let sum_w: f64 = weights.iter().sum();
    let sum_f: f64 = points.iter().zip(&weights).map(|((f, _), w)| w * f).sum();
    let sum_phi: f64 = points
        .iter()
        .zip(&weights)
        .map(|((_, phase), weight)| weight * phase.to_radians())
        .sum();
    let sum_f2: f64 = points
        .iter()
        .zip(&weights)
        .map(|((frequency, _), weight)| weight * frequency * frequency)
        .sum();
    let sum_f_phi: f64 = points
        .iter()
        .zip(&weights)
        .map(|((frequency, phase), weight)| weight * frequency * phase.to_radians())
        .sum();

    let denom = sum_w * sum_f2 - sum_f * sum_f;
    if denom.abs() < 1e-12 {
        return Err(PhaseArrivalError::DegenerateRegression);
    }

    let slope = (sum_w * sum_f_phi - sum_f * sum_phi) / denom;

    // τ = -slope / (2π), convert seconds → milliseconds
    let delay_ms = -slope / (2.0 * PI) * 1000.0;

    // Sanity check: plausible acoustic propagation time (-50–500 ms).
    // Negative delays are valid when a speaker is closer than the reference
    // channel used for relative alignment.
    if delay_ms > -50.0 && delay_ms < 500.0 {
        Ok(delay_ms)
    } else {
        Err(PhaseArrivalError::ImplausibleDelay { delay_ms })
    }
}
