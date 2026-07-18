//! Compatibility adapter for the extracted spectral-alignment engine.
//!
//! Filter fitting and response analysis live in `roomeq-engine`. This module
//! alone converts the engine's filter primitives into the root crate's legacy
//! output-plugin representation.

pub use roomeq_engine::spectral_align::{
    ChannelMatchingCorrectionProfile, ChannelMatchingResult, MIN_CORRECTION_DB,
    SpectralAlignmentResult, compute_inter_channel_deviation, compute_spectral_alignment,
    correct_inter_channel_deviation_with_profile, create_alignment_filters, log_spectral_alignment,
};

use super::types::PluginConfigWrapper;

/// Convert spectral-alignment filters to the legacy RoomEQ output plugins.
pub fn create_alignment_plugins(
    result: &SpectralAlignmentResult,
    sample_rate: f64,
) -> (Option<PluginConfigWrapper>, Option<PluginConfigWrapper>) {
    let shelf_filters = create_alignment_filters(result, sample_rate);
    let eq_plugin = (!shelf_filters.is_empty())
        .then(|| super::output::create_labeled_eq_plugin(&shelf_filters, "broadband"));
    let gain_plugin = (result.flat_gain_db.abs() >= MIN_CORRECTION_DB)
        .then(|| super::output::create_gain_plugin(result.flat_gain_db));
    (eq_plugin, gain_plugin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_preserves_broadband_output_contract() {
        let result = SpectralAlignmentResult {
            lowshelf_gain_db: -2.0,
            highshelf_gain_db: 1.5,
            flat_gain_db: -1.0,
            residual_rms_db: 0.5,
        };
        let (eq, gain) = create_alignment_plugins(&result, 48_000.0);
        assert_eq!(eq.expect("shelf plugin").plugin_type, "eq");
        assert_eq!(gain.expect("gain plugin").plugin_type, "gain");
    }
}
