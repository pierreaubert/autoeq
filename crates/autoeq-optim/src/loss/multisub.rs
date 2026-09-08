//! Multi-subwoofer flat-response loss.

use super::flat::flat_loss;
use super::{DriversLossData, compute_drivers_combined_response};

/// Preserve useful array output while fitting its shape. The reference is the
/// measured power sum, so an inverted DBA pair cannot use its cancelled identity
/// as an output reference. Three dB of mean loss and 12 dB of local cancellation
/// are allowed; deeper losses must pay a cost in the search itself.
pub fn array_output_penalty(candidate: &[f64], reference: &[f64]) -> f64 {
    if candidate.is_empty() || candidate.len() != reference.len() {
        return f64::INFINITY;
    }
    let n = candidate.len() as f64;
    let mean_loss = reference
        .iter()
        .zip(candidate)
        .map(|(r, c)| r - c)
        .sum::<f64>()
        / n;
    let null_loss = reference
        .iter()
        .zip(candidate)
        .map(|(r, c)| (r - c - 12.0).max(0.0).powi(2))
        .sum::<f64>()
        / n;
    4.0 * (mean_loss - 3.0).max(0.0).powi(2) + null_loss
}

pub fn multisub_output_penalty(
    data: &DriversLossData,
    combined: &ndarray::Array1<f64>,
    min_freq: f64,
    max_freq: f64,
) -> f64 {
    let mut candidate = Vec::new();
    let mut reference = Vec::new();
    for (i, &frequency) in data.freq_grid.iter().enumerate() {
        if frequency >= min_freq && frequency <= max_freq {
            candidate.push(combined[i]);
            reference.push(data.power_reference[i]);
        }
    }
    let low_count = data
        .freq_grid
        .iter()
        .filter(|&&frequency| frequency >= min_freq && frequency <= (min_freq * 2.0).min(max_freq))
        .count();
    let extension = if low_count > 0 {
        array_output_penalty(&candidate[..low_count], &reference[..low_count])
    } else {
        0.0
    };
    array_output_penalty(&candidate, &reference) + extension
}

/// Multi-subwoofer flat loss.
///
/// Computes the combined response of multiple subwoofers with configurable
/// gains and delays, normalizes it, and returns the flat loss (weighted MSE)
/// against a zero target over the evaluation range.
///
/// # Arguments
/// * `data` - Drivers loss data containing sub measurements and a frequency grid
/// * `gains` - Gain in dB for each sub
/// * `delays` - Delay in ms for each sub
/// * `sample_rate` - Sample rate
/// * `min_freq` - Min freq for evaluation
/// * `max_freq` - Max freq for evaluation
///
/// # Returns
/// * Loss value
pub fn multisub_flat_loss(
    data: &DriversLossData,
    gains: &[f64],
    delays: &[f64],
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
) -> f64 {
    // Pass empty crossover freqs (ignored because CrossoverType::None)
    let crossover_freqs = vec![];
    let combined_response =
        compute_drivers_combined_response(data, gains, &crossover_freqs, Some(delays), sample_rate);

    // Normalize the response (subtract the mean in the evaluation range)
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..data.freq_grid.len() {
        let freq = data.freq_grid[i];
        if freq >= min_freq && freq <= max_freq {
            sum += combined_response[i];
            count += 1;
        }
    }
    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    let normalized = &combined_response - mean;

    // Compute flatness loss (RMS deviation from zero)
    flat_loss(&data.freq_grid, &normalized, min_freq, max_freq)
        + multisub_output_penalty(data, &combined_response, min_freq, max_freq)
        + 0.01 * gains.iter().map(|gain| gain * gain).sum::<f64>()
}

#[cfg(test)]
mod output_tests {
    use super::*;

    #[test]
    fn output_guard_rejects_silence_and_common_attenuation() {
        let reference = [80.0, 85.0, 75.0];
        assert_eq!(array_output_penalty(&reference, &reference), 0.0);
        assert!(array_output_penalty(&[-240.0; 3], &reference) > 100_000.0);
        assert!(array_output_penalty(&[68.0, 73.0, 63.0], &reference) > 300.0);
    }
}
