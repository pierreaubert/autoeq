//! Versioned auditory-frequency integration helpers.

use ndarray::Array1;

/// Identifier for the frequency measure used by optimization and acceptance.
pub const AUDITORY_FREQUENCY_MEASURE_VERSION: &str = "glasberg-moore-erb-rate-1990-v1";

/// Glasberg-Moore ERB-rate number for a frequency in hertz.
pub fn erb_rate(frequency: f64) -> f64 {
    21.4 * (1.0 + 0.00437 * frequency).log10()
}

/// Trapezoidal cell widths on the ERB-rate axis.
pub fn erb_rate_cell_widths(frequencies: &Array1<f64>) -> Array1<f64> {
    let count = frequencies.len();
    if count == 0 {
        return Array1::zeros(0);
    }
    if count == 1
        || frequencies
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return Array1::from_elem(count, 1.0);
    }

    let rates: Vec<f64> = frequencies
        .iter()
        .map(|frequency| erb_rate(*frequency))
        .collect();
    let mut weights = Array1::zeros(count);
    weights[0] = 0.5 * (rates[1] - rates[0]);
    weights[count - 1] = 0.5 * (rates[count - 1] - rates[count - 2]);
    for index in 1..count - 1 {
        weights[index] = 0.5 * (rates[index + 1] - rates[index - 1]);
    }

    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        Array1::from_elem(count, 1.0)
    } else {
        weights
    }
}

/// RMS integrated on the ERB-rate axis.
pub fn erb_rate_weighted_rms(frequencies: &Array1<f64>, values: &[f64]) -> Option<f64> {
    if frequencies.len() != values.len() || values.is_empty() {
        return None;
    }
    let weights = erb_rate_cell_widths(frequencies);
    let total_weight: f64 = weights.iter().sum();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return None;
    }
    let weighted_square_sum: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| value * value * weight)
        .sum();
    weighted_square_sum
        .is_finite()
        .then(|| (weighted_square_sum / total_weight).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_error_is_grid_invariant() {
        for frequencies in [
            Array1::linspace(20.0, 20_000.0, 100),
            Array1::from_iter((0..100).map(|i| 20.0 * 1000.0_f64.powf(i as f64 / 99.0))),
            Array1::from(vec![20.0, 37.0, 91.0, 410.0, 2_300.0, 20_000.0]),
            Array1::linspace(20.0, 20_000.0, 10_000),
        ] {
            let values = vec![4.25; frequencies.len()];
            let rms = erb_rate_weighted_rms(&frequencies, &values).unwrap();
            assert!((rms - 4.25).abs() < 1e-12);
        }
    }
}
