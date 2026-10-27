//! Qualification of actual matched-pole peak features, independent of FFT sampling.

use crate::iir::{Biquad, BiquadFilterType};

// These are design-consistency tolerances, not hearing or playback thresholds.
const NOMINAL_GAIN_TOLERANCE_DB: f64 = 1e-5;
const NUMERICAL_NEUTRAL_DB: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
pub struct MatchedFeature {
    pub center_hz: f64,
    pub width_octaves: f64,
}

// |c0+c1*z^-1+c2*z^-2|² in phi=sin²(omega/2). Unlike cos(omega),
// phi retains relative precision for bass frequencies close to DC.
fn power_polynomial(c0: f64, c1: f64, c2: f64) -> [f64; 3] {
    [
        (c0 + c1 + c2).powi(2),
        -4.0 * (c0 * c1 + 4.0 * c0 * c2 + c1 * c2),
        16.0 * c0 * c2,
    ]
}

fn roots(coefficients: [f64; 3]) -> [f64; 2] {
    let scale = coefficients
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return [f64::NAN; 2];
    }
    let [c, b, a] = coefficients.map(|value| value / scale);
    // A degree collapse uses the linear equation, avoiding division by noise.
    if a.abs() <= 16.0 * f64::EPSILON {
        return if b.abs() > 16.0 * f64::EPSILON {
            [-c / b, f64::NAN]
        } else {
            [f64::NAN; 2]
        };
    }
    let discriminant = b.mul_add(b, -4.0 * a * c);
    if !discriminant.is_finite() || discriminant < 0.0 {
        return [f64::NAN; 2];
    }
    let q = -0.5 * (b + discriminant.sqrt().copysign(b));
    if q == 0.0 {
        [-b / (2.0 * a), f64::NAN]
    } else {
        [q / a, c / q]
    }
}

fn frequency(phi: f64, sample_rate: f64) -> f64 {
    sample_rate / std::f64::consts::PI * phi.sqrt().asin()
}

pub fn feature(center: f64, sample_rate: f64, q: f64, gain: f64) -> Option<MatchedFeature> {
    if !center.is_finite()
        || center <= 0.0
        || center >= sample_rate / 2.0
        || !sample_rate.is_finite()
        || sample_rate <= 0.0
        || !q.is_finite()
        || q <= 0.0
        || !gain.is_finite()
        || gain == 0.0
    {
        return None;
    }
    let filter = Biquad::new(BiquadFilterType::PeakMatched, center, sample_rate, q, gain);
    let db = |frequency| 20.0 * filter.complex_response(frequency).norm().log10();
    let nominal = db(center);
    if !nominal.is_finite() || (nominal - gain).abs() > NOMINAL_GAIN_TOLERANCE_DB {
        // The matched numerator's symmetric fallback can fail to realize the
        // requested gain and introduce an unrelated notch. Never call it EQ.
        return None;
    }
    let coefficients = filter.coefficients();
    let n = power_polynomial(coefficients.b0, coefficients.b1, coefficients.b2);
    let d = power_polynomial(1.0, coefficients.a1, coefficients.a2);
    let extrema = roots([
        n[1] * d[0] - n[0] * d[1],
        2.0 * (n[2] * d[0] - n[0] * d[2]),
        n[2] * d[1] - n[1] * d[2],
    ]);
    let mut selected = None;
    for phi in extrema.into_iter().filter(|phi| *phi > 0.0 && *phi < 1.0) {
        let location = frequency(phi, sample_rate);
        let value = db(location);
        if !value.is_finite() {
            return None;
        }
        if value.abs() <= NUMERICAL_NEUTRAL_DB {
            continue;
        }
        // Gain is matched at the nominal frequency, not necessarily at the
        // actual extremum. Realized gain envelopes own the extremum's level.
        if value.signum() != gain.signum() {
            return None;
        }
        if selected.is_some() {
            return None;
        }
        selected = Some((phi, location, value));
    }
    let (peak_phi, center_hz, peak_db) = selected?;
    // Half the actual extremum's dB gain corresponds to this power ratio.
    let threshold = 10.0_f64.powf(peak_db / 20.0);
    let crossings = roots([
        n[0] - threshold * d[0],
        n[1] - threshold * d[1],
        n[2] - threshold * d[2],
    ]);
    let lower = crossings
        .iter()
        .copied()
        .filter(|phi| *phi > 0.0 && *phi < peak_phi)
        .max_by(f64::total_cmp)?;
    let upper = crossings
        .iter()
        .copied()
        .filter(|phi| *phi > peak_phi && *phi < 1.0)
        .min_by(f64::total_cmp)?;
    let width_octaves = (frequency(upper, sample_rate) / frequency(lower, sample_rate)).log2();
    (width_octaves.is_finite() && width_octaves > 0.0).then_some(MatchedFeature {
        center_hz,
        width_octaves,
    })
}

pub fn project_q(
    center: f64,
    sample_rate: f64,
    q: f64,
    gain: f64,
    minimum: f64,
    maximum: f64,
    mut required_width: impl FnMut(f64) -> Option<f64>,
) -> Option<f64> {
    let qualifies = |q, width: &mut dyn FnMut(f64) -> Option<f64>| {
        let measured = feature(center, sample_rate, q, gain)?;
        let required = width(measured.center_hz)?;
        (required.is_finite() && measured.width_octaves >= required).then_some(())
    };
    if q < minimum || minimum <= 0.0 || maximum < minimum {
        return None;
    }
    let mut candidate = q.min(maximum);
    if qualifies(candidate, &mut required_width).is_some() {
        return Some(candidate);
    }
    // Bounded logarithmic broadening preserves the original topology and gain.
    // Every accepted point is checked against the actual response; a bad
    // numerator or nonmonotone region cannot pass through interpolation.
    for _ in 0..96 {
        let rejected = candidate;
        candidate = (candidate * 0.95).max(minimum);
        if candidate == rejected {
            break;
        }
        if qualifies(candidate, &mut required_width).is_some() {
            let mut accepted = candidate;
            let mut upper = rejected;
            for _ in 0..18 {
                let middle = (accepted * upper).sqrt();
                if qualifies(middle, &mut required_width).is_some() {
                    accepted = middle;
                } else {
                    upper = middle;
                }
            }
            return Some(accepted);
        }
        if candidate == minimum {
            break;
        }
    }
    None
}
