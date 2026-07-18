use super::types::{GdOptAdvisory, GroupDelayOptResult};

pub use roomeq_model::GroupDelayOptSummary;

/// Engine-side constructors for the neutral serialized summary contract.
pub trait GroupDelayOptSummaryExt {
    fn from_result_with_names(result: &GroupDelayOptResult, names: Vec<String>) -> Self;
    fn from_advisory(advisory: &GdOptAdvisory) -> Self;
}

impl GroupDelayOptSummaryExt for GroupDelayOptSummary {
    fn from_result_with_names(result: &GroupDelayOptResult, names: Vec<String>) -> Self {
        Self {
            band: result.band,
            channel_names: names,
            per_channel_delay_ms: result.per_channel.iter().map(|ch| ch.delay_ms).collect(),
            per_channel_polarity_inverted: result
                .per_channel
                .iter()
                .map(|ch| ch.polarity_inverted)
                .collect(),
            per_channel_ap_count: result
                .per_channel
                .iter()
                .map(|ch| ch.ap_filters.len())
                .collect(),
            sum_gd_pre_rms_ms: result.sum_gd_pre_rms_ms,
            sum_gd_post_rms_ms: result.sum_gd_post_rms_ms,
            mean_coherence: result.mean_coherence,
            improvement_db: result.improvement_db,
            advisory: "success".to_string(),
            applied: false,
        }
    }

    fn from_advisory(advisory: &GdOptAdvisory) -> Self {
        let reason = match advisory {
            GdOptAdvisory::Success { improvement_db } => {
                format!("success:{improvement_db:.1}dB")
            }
            GdOptAdvisory::NoPhaseData => "no_phase_data".to_string(),
            GdOptAdvisory::CoherenceBelowThreshold { mean_coherence } => {
                format!("coherence_below_threshold:{mean_coherence:.2}")
            }
            GdOptAdvisory::PhaseLinearNoTarget => "phase_linear_no_target".to_string(),
            GdOptAdvisory::InsufficientChannels => "insufficient_channels".to_string(),
            GdOptAdvisory::EmptyBand => "empty_band".to_string(),
            GdOptAdvisory::MinimalImprovement { improvement_db } => {
                format!("minimal_improvement:{improvement_db:.1}dB")
            }
            GdOptAdvisory::FrequencyGridMismatch => "frequency_grid_mismatch".to_string(),
            GdOptAdvisory::MissingCoherenceDelayOnly => "missing_coherence_delay_only".to_string(),
            GdOptAdvisory::AllPassDisabledNoBootstrapRealisations => {
                "allpass_disabled_no_bootstrap_realisations".to_string()
            }
        };
        Self {
            band: (0.0, 0.0),
            channel_names: vec![],
            per_channel_delay_ms: vec![],
            per_channel_polarity_inverted: vec![],
            per_channel_ap_count: vec![],
            sum_gd_pre_rms_ms: 0.0,
            sum_gd_post_rms_ms: 0.0,
            mean_coherence: 0.0,
            improvement_db: 0.0,
            advisory: reason,
            applied: false,
        }
    }
}
