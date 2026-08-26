use super::compute::compute_sum_gd;
use super::gd_opt_config::GdOptConfig;
use super::types::ChannelMeasurementInput;
use super::types::GdAlignmentTarget;
use super::types::GroupDelayOptResult;
use super::types::encode_result_as_params;
use ndarray::Array1;

/// Build a `GdAlignmentTarget` from a `GroupDelayOptResult`.
///
/// This extracts the per-channel delays and computes the reference GD
/// (post-optimisation sum GD) that the FIR designer should target.
/// Used by `PhaseLinear` mode to pass delay information to the FIR path.
pub fn build_gd_alignment_target(
    channels: &[ChannelMeasurementInput],
    result: &GroupDelayOptResult,
    config: &GdOptConfig,
) -> GdAlignmentTarget {
    let n_freq = channels[0].freq.len();
    let band_indices: Vec<usize> = (0..n_freq)
        .filter(|&i| channels[0].freq[i] >= result.band.0 && channels[0].freq[i] <= result.band.1)
        .collect();

    // Encode the optimised result as params to compute post-GD
    let params = encode_result_as_params(result, config);
    let sum_gd = compute_sum_gd(channels, &params, &band_indices, config);

    let per_channel_delay_ms = result.per_channel.iter().map(|ch| ch.delay_ms).collect();

    // Build frequency sub-grid for the band
    let valid_sum_gd: Vec<(f64, f64)> = band_indices
        .iter()
        .zip(sum_gd)
        .filter_map(|(&index, group_delay)| {
            group_delay
                .is_finite()
                .then_some((channels[0].freq[index], group_delay))
        })
        .collect();
    let freq = Array1::from_iter(valid_sum_gd.iter().map(|(frequency, _)| *frequency));

    GdAlignmentTarget {
        per_channel_delay_ms,
        per_channel_polarity_inverted: result
            .per_channel
            .iter()
            .map(|channel| channel.polarity_inverted)
            .collect(),
        sum_gd_reference_ms: valid_sum_gd
            .into_iter()
            .map(|(_, group_delay)| group_delay)
            .collect(),
        freq,
    }
}

/// Build DE bounds for all parameters.
pub(super) fn build_bounds(n_ch: usize, config: &GdOptConfig) -> Vec<(f64, f64)> {
    let mut bounds = Vec::new();
    for _ in 0..n_ch {
        // Delay is optimized as a relative control and normalized after DE so
        // the exported DSP never adds arbitrary common latency.
        bounds.push((-config.max_delay_ms, config.max_delay_ms));
        // AP filters: (freq, q) pairs
        for _ in 0..config.ap_per_channel {
            bounds.push((config.ap_min_freq, config.ap_max_freq));
            bounds.push((config.ap_min_q, config.ap_max_q));
        }
        // polarity: [0, 1] — decoded as inverted if > 0.5
        if config.optimize_polarity {
            bounds.push((0.0, 1.0));
        }
    }
    bounds
}
