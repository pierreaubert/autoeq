use super::compute::compute_sum_gd_rms;
use super::gd_opt_config::GdOptConfig;
use super::types::ChannelMeasurementInput;

/// The objective function for DE: coherence-weighted RMS GD of the sum.
pub(super) fn gd_loss(
    channels: &[ChannelMeasurementInput],
    params: &[f64],
    band_indices: &[usize],
    config: &GdOptConfig,
) -> f64 {
    compute_sum_gd_rms(channels, params, band_indices, config)
}
