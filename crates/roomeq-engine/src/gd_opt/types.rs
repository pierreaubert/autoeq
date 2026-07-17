use math_audio_iir_fir::{Biquad, BiquadFilterType};
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::PI;

use super::gd_opt_config::GdOptConfig;

/// Per-channel result.
#[derive(Debug, Clone)]
pub struct ChannelGdResult {
    pub delay_ms: f64,
    pub polarity_inverted: bool,
    pub ap_filters: Vec<Biquad>,
    pub channel_gd_pre_rms_ms: f64,
    pub channel_gd_post_rms_ms: f64,
}

/// Overall optimisation result.
#[derive(Debug, Clone)]
pub struct GroupDelayOptResult {
    pub band: (f64, f64),
    pub per_channel: Vec<ChannelGdResult>,
    pub sum_gd_pre_rms_ms: f64,
    pub sum_gd_post_rms_ms: f64,
    pub mean_coherence: f64,
    pub improvement_db: f64,
}

/// Per-channel measurement input.
#[derive(Debug, Clone)]
pub struct ChannelMeasurementInput {
    /// Frequency grid (Hz), shared across spl/phase/coherence.
    pub freq: Array1<f64>,
    /// SPL in dB.
    pub spl: Array1<f64>,
    /// Unwrapped phase in radians.
    pub phase: Array1<f64>,
    /// Coherence (γ²) per bin, range [0, 1].
    pub coherence: Array1<f64>,
}

/// Alignment target for the PhaseLinear FIR path (§3.7, GD-3b).
///
/// When `PhaseLinear` mode is used, the FIR designer receives this struct
/// so it can incorporate inter-channel GD alignment into the filter design
/// via Kirkeby mixed-phase inversion. When absent, the FIR falls back to
/// pure magnitude correction.
#[derive(Debug, Clone)]
pub struct GdAlignmentTarget {
    /// Per-channel delay in ms (channel index → delay).
    pub per_channel_delay_ms: Vec<f64>,
    /// Reference sum GD curve (the target flat GD the FIR should approach).
    pub sum_gd_reference_ms: Vec<f64>,
    /// Frequency grid for `sum_gd_reference_ms`.
    pub freq: Array1<f64>,
}

/// Advisory reasons for GD-Opt outcomes (§3.5, GD-4).
#[derive(Debug, Clone, PartialEq)]
pub enum GdOptAdvisory {
    /// GD-Opt completed successfully with the given improvement.
    Success { improvement_db: f64 },
    /// GD-Opt skipped: no phase data available.
    NoPhaseData,
    /// GD-Opt skipped: coherence below threshold.
    CoherenceBelowThreshold { mean_coherence: f64 },
    /// GD-Opt skipped: PhaseLinear mode without FIR GD target.
    PhaseLinearNoTarget,
    /// GD-Opt skipped: insufficient channels (need ≥ 2).
    InsufficientChannels,
    /// GD-Opt skipped: band derivation produced empty range.
    EmptyBand,
    /// GD-Opt degraded: optimiser ran but improvement was minimal.
    MinimalImprovement { improvement_db: f64 },
    /// GD-Opt skipped: channels are sampled on different frequency grids.
    FrequencyGridMismatch,
    /// GD-Opt degraded: coherence was absent, so only delay was optimized.
    MissingCoherenceDelayOnly,
    /// GD-Opt degraded: all-pass was requested but bootstrap data was absent.
    AllPassDisabledNoBootstrapRealisations,
}

/// Decode parameters for a single channel from the flat parameter vector.
pub(super) struct ChannelParams {
    pub(super) delay_ms: f64,
    pub(super) ap_filters: Vec<(f64, f64)>, // (freq, q)
    pub(super) polarity_inverted: bool,
}

pub(super) fn normalize_per_channel_controls(results: &mut [ChannelGdResult]) {
    if results.is_empty() {
        return;
    }

    let min_delay = results
        .iter()
        .map(|ch| ch.delay_ms)
        .fold(f64::INFINITY, f64::min);
    if min_delay.is_finite() {
        for ch in results.iter_mut() {
            ch.delay_ms = (ch.delay_ms - min_delay).max(0.0);
            if ch.delay_ms < 1e-9 {
                ch.delay_ms = 0.0;
            }
        }
    }

    // Global polarity inversion is not identifiable in the summed response.
    // Use channel 0 as the deterministic reference and express all other
    // inversions relative to it.
    let reference_inverted = results[0].polarity_inverted;
    if reference_inverted {
        for ch in results.iter_mut() {
            ch.polarity_inverted = !ch.polarity_inverted;
        }
    }
    results[0].polarity_inverted = false;
}

/// Compute the complex response of a channel at frequency `f` with applied
/// delay, all-pass filters, and polarity.
pub(super) fn channel_complex_at(
    ch: &ChannelMeasurementInput,
    freq_idx: usize,
    ch_params: &ChannelParams,
    config: &GdOptConfig,
) -> Complex64 {
    let f = ch.freq[freq_idx];
    let omega = 2.0 * PI * f;

    // Original channel response as complex
    let mag = 10.0_f64.powf(ch.spl[freq_idx] / 20.0);
    let phase = ch.phase[freq_idx];
    let mut h = Complex64::from_polar(mag, phase);

    // Apply delay: e^(-jωτ)
    let delay_s = ch_params.delay_ms * 1e-3;
    h *= Complex64::from_polar(1.0, -omega * delay_s);

    // Apply all-pass filters
    for &(ap_freq, ap_q) in &ch_params.ap_filters {
        let ap = Biquad::new(
            BiquadFilterType::AllPass,
            ap_freq,
            config.sample_rate,
            ap_q,
            0.0,
        );
        h *= ap.complex_response(f);
    }

    // Apply polarity inversion
    if ch_params.polarity_inverted {
        h = -h;
    }

    h
}

/// Decode parameters for a single channel from the flat parameter vector.
pub(super) fn decode_channel_params(
    params: &[f64],
    ch: usize,
    config: &GdOptConfig,
) -> ChannelParams {
    let per_ch = 1 + config.ap_per_channel * 2 + if config.optimize_polarity { 1 } else { 0 };
    let offset = ch * per_ch;

    let delay_ms = params[offset];

    let mut ap_filters = Vec::with_capacity(config.ap_per_channel);
    for i in 0..config.ap_per_channel {
        let freq = params[offset + 1 + i * 2];
        let q = params[offset + 1 + i * 2 + 1];
        ap_filters.push((freq, q));
    }

    let polarity_inverted = if config.optimize_polarity {
        params[offset + 1 + config.ap_per_channel * 2] > 0.5
    } else {
        false
    };

    ChannelParams {
        delay_ms,
        ap_filters,
        polarity_inverted,
    }
}

/// Encode a `GroupDelayOptResult` back into a parameter vector for evaluation.
pub(super) fn encode_result_as_params(
    result: &GroupDelayOptResult,
    config: &GdOptConfig,
) -> Vec<f64> {
    let n_ch = result.per_channel.len();
    let per_ch = 1 + config.ap_per_channel * 2 + if config.optimize_polarity { 1 } else { 0 };
    let mut params = vec![0.0; n_ch * per_ch];

    for (ch_idx, ch_result) in result.per_channel.iter().enumerate() {
        let offset = ch_idx * per_ch;
        params[offset] = ch_result.delay_ms;

        for (i, ap) in ch_result.ap_filters.iter().enumerate() {
            if i < config.ap_per_channel {
                params[offset + 1 + i * 2] = ap.freq;
                params[offset + 1 + i * 2 + 1] = ap.q;
            }
        }

        if config.optimize_polarity {
            params[offset + 1 + config.ap_per_channel * 2] = if ch_result.polarity_inverted {
                1.0
            } else {
                0.0
            };
        }
    }

    params
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result_with(delay_ms: f64, polarity_inverted: bool) -> ChannelGdResult {
        ChannelGdResult {
            delay_ms,
            polarity_inverted,
            ap_filters: Vec::new(),
            channel_gd_pre_rms_ms: 0.0,
            channel_gd_post_rms_ms: 0.0,
        }
    }

    #[test]
    fn normalize_per_channel_controls_shifts_delays_to_minimum() {
        let mut results = vec![result_with(5.0, false), result_with(3.0, false)];
        normalize_per_channel_controls(&mut results);
        assert_eq!(results[0].delay_ms, 2.0);
        assert_eq!(results[1].delay_ms, 0.0);
    }

    #[test]
    fn normalize_per_channel_controls_makes_first_channel_reference_polarity() {
        let mut results = vec![result_with(0.0, true), result_with(0.0, true)];
        normalize_per_channel_controls(&mut results);
        assert!(!results[0].polarity_inverted);
        assert!(!results[1].polarity_inverted);
    }

    #[test]
    fn normalize_per_channel_controls_empty_is_noop() {
        let mut results: Vec<ChannelGdResult> = Vec::new();
        normalize_per_channel_controls(&mut results);
        assert!(results.is_empty());
    }
}
