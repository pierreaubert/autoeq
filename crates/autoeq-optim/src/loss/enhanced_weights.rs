//! Enhanced loss functions with configurable frequency band weights.
//!
//! Based on research:
//! - "Perceptually-Motivated Audio Equalization" (Kulkarni et al.)
//! - "Frequency-Dependent Weighting in Audio Equalization" (Zölzer et al.)

use ndarray::Array1;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Frequency band configuration for weighted loss
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FrequencyBandWeights {
    /// Bass band minimum frequency (Hz)
    pub bass_min: f64,
    /// Bass band maximum frequency (Hz)
    pub bass_max: f64,
    /// Midrange band minimum frequency (Hz)
    pub mid_min: f64,
    /// Midrange band maximum frequency (Hz)
    pub mid_max: f64,
    /// Treble band minimum frequency (Hz)
    pub treble_min: f64,
    /// Treble band maximum frequency (Hz)
    pub treble_max: f64,
    /// Weight for bass band (default: 2.0 - bass is more critical for room correction)
    pub bass_weight: f64,
    /// Weight for midrange band (default: 1.0)
    pub mid_weight: f64,
    /// Weight for treble band (default: 0.8 - less critical for room issues)
    pub treble_weight: f64,
}

impl Default for FrequencyBandWeights {
    fn default() -> Self {
        Self {
            bass_min: 20.0,
            bass_max: 200.0,
            mid_min: 200.0,
            mid_max: 4000.0,
            treble_min: 4000.0,
            treble_max: 20000.0,
            bass_weight: 2.0,
            mid_weight: 1.0,
            treble_weight: 0.8,
        }
    }
}

/// Compute ERB (Equivalent Rectangular Bandwidth) for a frequency
/// ERB formula: 24.7 * (1 + 4.37 * f / 1000)
pub fn erb(frequency: f64) -> f64 {
    24.7 * (1.0 + 4.37 * frequency / 1000.0)
}

/// Return trapezoidal quadrature weights on the logarithmic-frequency axis.
///
/// RoomEQ uses a denser grid below 1 kHz than above it. Weighting each point
/// equally would therefore make the low-frequency region dominate losses just
/// because it has more samples. These weights make point-based losses
/// approximate the same log-frequency integral on either grid.
fn log_frequency_quadrature_weights(freqs: &Array1<f64>) -> Array1<f64> {
    let n = freqs.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    if n == 1
        || freqs
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return Array1::from_elem(n, 1.0);
    }

    let log_frequencies: Vec<f64> = freqs.iter().map(|frequency| frequency.ln()).collect();
    let mut weights = Array1::zeros(n);
    weights[0] = 0.5 * (log_frequencies[1] - log_frequencies[0]);
    weights[n - 1] = 0.5 * (log_frequencies[n - 1] - log_frequencies[n - 2]);
    for index in 1..n - 1 {
        weights[index] = 0.5 * (log_frequencies[index + 1] - log_frequencies[index - 1]);
    }

    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        Array1::from_elem(n, 1.0)
    } else {
        weights
    }
}

/// Candidate-independent integration kernel for ERB and band-weighted losses.
///
/// Construction may allocate; [`Self::evaluate`] and
/// [`Self::evaluate_asymmetric`] do not.
#[derive(Debug, Clone)]
pub struct PreparedWeightedLoss {
    indices: Vec<usize>,
    erb_weights: Vec<f64>,
    quadrature_weights: Vec<f64>,
    band: Vec<u8>,
    erb_total: f64,
    band_totals: [f64; 3],
    bands: FrequencyBandWeights,
}

impl PreparedWeightedLoss {
    /// Prepare the fixed-grid weights for the inclusive active range.
    pub fn new(
        freqs: &Array1<f64>,
        min_freq: f64,
        max_freq: f64,
        bands: FrequencyBandWeights,
    ) -> Self {
        let indices: Vec<usize> = freqs
            .iter()
            .enumerate()
            .filter_map(|(index, &freq)| (freq >= min_freq && freq <= max_freq).then_some(index))
            .collect();
        let active_freqs = Array1::from_iter(indices.iter().map(|&index| freqs[index]));
        let quadrature = log_frequency_quadrature_weights(&active_freqs);
        let mut erb_weights = Vec::with_capacity(indices.len());
        let mut quadrature_weights = Vec::with_capacity(indices.len());
        let mut band = Vec::with_capacity(indices.len());
        let mut erb_total = 0.0;
        let mut band_totals = [0.0; 3];

        for (active_index, &source_index) in indices.iter().enumerate() {
            let freq = freqs[source_index];
            let quadrature_weight = quadrature[active_index];
            let erb_weight = quadrature_weight / erb(freq);
            let band_index = if freq >= bands.bass_min && freq <= bands.bass_max {
                0
            } else if freq >= bands.mid_min && freq <= bands.mid_max {
                1
            } else if freq >= bands.treble_min && freq <= bands.treble_max {
                2
            } else {
                3
            };
            erb_weights.push(erb_weight);
            quadrature_weights.push(quadrature_weight);
            band.push(band_index);
            erb_total += erb_weight;
            if band_index < 3 {
                band_totals[band_index as usize] += quadrature_weight;
            }
        }

        Self {
            indices,
            erb_weights,
            quadrature_weights,
            band,
            erb_total,
            band_totals,
            bands,
        }
    }

    /// Whether the configured range contains no frequency samples.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Active source indices, useful for preparing other per-frequency data.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Evaluate the standard combined loss without allocating.
    pub fn evaluate(&self, error: &Array1<f64>, erb_mix: f64, band_mix: f64) -> f64 {
        self.evaluate_weighted(error, erb_mix, band_mix, |_, _| 1.0)
    }

    /// Evaluate with sign-dependent, precomputed per-sample multipliers.
    pub fn evaluate_asymmetric(
        &self,
        error: &Array1<f64>,
        peak_weights: &[f64],
        dip_weights: &[f64],
        erb_mix: f64,
        band_mix: f64,
    ) -> f64 {
        if peak_weights.len() != self.indices.len() || dip_weights.len() != self.indices.len() {
            return f64::INFINITY;
        }
        self.evaluate_weighted(error, erb_mix, band_mix, |active_index, value| {
            if value > 0.0 {
                peak_weights[active_index]
            } else {
                dip_weights[active_index]
            }
        })
    }

    fn evaluate_weighted(
        &self,
        error: &Array1<f64>,
        erb_mix: f64,
        band_mix: f64,
        sample_weight: impl Fn(usize, f64) -> f64,
    ) -> f64 {
        if self.indices.is_empty() {
            return f64::INFINITY;
        }
        let mut erb_sum = 0.0;
        let mut band_sums = [0.0; 3];
        for active_index in 0..self.indices.len() {
            let value = error[self.indices[active_index]];
            let square = value * value * sample_weight(active_index, value).max(0.0);
            erb_sum += square * self.erb_weights[active_index];
            let band_index = self.band[active_index] as usize;
            if band_index < 3 {
                band_sums[band_index] += square * self.quadrature_weights[active_index];
            }
        }
        let erb_loss = if self.erb_total > 0.0 {
            (erb_sum / self.erb_total).sqrt()
        } else {
            0.0
        };
        let mut band_loss = 0.0;
        let output_weights = [
            self.bands.bass_weight,
            self.bands.mid_weight,
            self.bands.treble_weight,
        ];
        for band_index in 0..3 {
            if self.band_totals[band_index] > 0.0 {
                band_loss += output_weights[band_index]
                    * (band_sums[band_index] / self.band_totals[band_index]).sqrt();
            }
        }
        erb_mix * erb_loss + band_mix * band_loss
    }
}

/// Compute ERB-weighted error
///
/// The ERB scale provides better perceptual relevance than linear frequency.
/// Lower frequencies have smaller ERBs, meaning we get more resolution where
/// the human auditory system is more sensitive.
pub fn erb_weighted_loss(freqs: &Array1<f64>, error: &Array1<f64>) -> f64 {
    assert_eq!(freqs.len(), error.len());

    let erbs: Array1<f64> = freqs.mapv(erb);
    let quadrature_weights = log_frequency_quadrature_weights(freqs);

    // Weight inversely proportional to ERB (more weight at low frequencies)
    // while integrating over log frequency rather than over point count.
    let weights: Array1<f64> = erbs
        .iter()
        .zip(quadrature_weights.iter())
        .map(|(erb, quadrature)| quadrature / erb)
        .collect();

    // Normalize weights
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return 0.0;
    }

    // Compute weighted mean squared error
    let weighted_sum: f64 = error
        .iter()
        .zip(weights.iter())
        .map(|(e, w)| e * e * w)
        .sum();

    (weighted_sum / total_weight).sqrt()
}

/// Compute frequency band weighted error
pub fn band_weighted_loss(
    freqs: &Array1<f64>,
    error: &Array1<f64>,
    bands: &FrequencyBandWeights,
) -> f64 {
    assert_eq!(freqs.len(), error.len());

    let quadrature_weights = log_frequency_quadrature_weights(freqs);
    let mut bass_ss = 0.0;
    let mut bass_weight = 0.0;
    let mut mid_ss = 0.0;
    let mut mid_weight = 0.0;
    let mut treble_ss = 0.0;
    let mut treble_weight = 0.0;

    for ((&f, &e), &quadrature_weight) in freqs
        .iter()
        .zip(error.iter())
        .zip(quadrature_weights.iter())
    {
        if f >= bands.bass_min && f <= bands.bass_max {
            bass_ss += quadrature_weight * e * e;
            bass_weight += quadrature_weight;
        } else if f >= bands.mid_min && f <= bands.mid_max {
            mid_ss += quadrature_weight * e * e;
            mid_weight += quadrature_weight;
        } else if f >= bands.treble_min && f <= bands.treble_max {
            treble_ss += quadrature_weight * e * e;
            treble_weight += quadrature_weight;
        }
    }

    let bass_rms = if bass_weight > 0.0 {
        (bass_ss / bass_weight).sqrt()
    } else {
        0.0
    };
    let mid_rms = if mid_weight > 0.0 {
        (mid_ss / mid_weight).sqrt()
    } else {
        0.0
    };
    let treble_rms = if treble_weight > 0.0 {
        (treble_ss / treble_weight).sqrt()
    } else {
        0.0
    };

    bands.bass_weight * bass_rms + bands.mid_weight * mid_rms + bands.treble_weight * treble_rms
}

/// Combine ERB-weighted and band-weighted approaches
///
/// This provides both perceptual relevance (ERB) and user control (bands)
pub fn combined_weighted_loss(
    freqs: &Array1<f64>,
    error: &Array1<f64>,
    bands: &FrequencyBandWeights,
    erb_weight: f64,
    band_weight: f64,
) -> f64 {
    let erb_loss = erb_weighted_loss(freqs, error);
    let band_loss = band_weighted_loss(freqs, error, bands);

    erb_weight * erb_loss + band_weight * band_loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erb_at_1000hz() {
        let f = 1000.0;
        let expected = 24.7 * (1.0 + 4.37 * f / 1000.0);
        assert!((erb(f) - expected).abs() < 1e-12);
    }

    #[test]
    fn erb_at_zero() {
        let expected = 24.7;
        assert!((erb(0.0) - expected).abs() < 1e-12);
    }
}
