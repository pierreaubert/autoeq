//! Compatibility adapter for engine-owned inter-channel timbre matching.

pub use roomeq_engine::inter_channel_timbre_matching::*;

use super::types::PluginConfigWrapper;

/// Convert engine correction primitives to legacy RoomEQ output plugins.
pub fn create_timbre_matching_plugins(
    result: &InterChannelTimbreMatchingResult,
    sample_rate: f64,
) -> Vec<PluginConfigWrapper> {
    result
        .alignment
        .as_ref()
        .map_or_else(Vec::new, |alignment| {
            let (eq_plugin, gain_plugin) =
                super::spectral_align::create_alignment_plugins(alignment, sample_rate);
            eq_plugin.into_iter().chain(gain_plugin).collect()
        })
}
