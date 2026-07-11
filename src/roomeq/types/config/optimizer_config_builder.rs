//! Builder for [`OptimizerConfig`].
//!
//! The builder is a thin, additive wrapper around [`OptimizerConfig`]. It starts
//! from the default configuration and exposes a fluent setter for every public
//! field, so tests and binaries can construct focused variants without listing
//! dozens of unrelated defaults.

use super::OptimizerConfig;
use std::path::PathBuf;

/// Fluent builder for [`OptimizerConfig`].
#[derive(Debug, Clone, Default)]
pub struct OptimizerConfigBuilder(OptimizerConfig);

impl OptimizerConfigBuilder {
    /// Create a builder seeded with the default optimizer configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder seeded with an existing configuration.
    pub fn from_config(config: OptimizerConfig) -> Self {
        Self(config)
    }

    /// Consume the builder and return the configured [`OptimizerConfig`].
    pub fn build(self) -> OptimizerConfig {
        self.0
    }
}

macro_rules! setter {
    ($field:ident, $ty:ty) => {
        #[must_use]
        pub fn $field(mut self, value: $ty) -> OptimizerConfigBuilder {
            self.0.$field = value;
            self
        }
    };
}

impl OptimizerConfigBuilder {
    setter!(processing_mode, super::ProcessingMode);
    setter!(fir, Option<super::FirConfig>);
    setter!(mixed_config, Option<super::MixedModeConfig>);
    setter!(mixed_phase, Option<super::MixedPhaseSerdeConfig>);
    setter!(phase_correction, Option<super::MixedPhaseSerdeConfig>);
    setter!(loss_type, String);
    setter!(epa_config, Option<crate::loss::epa::score::EpaConfig>);
    setter!(algorithm, String);
    setter!(strategy, String);
    setter!(num_filters, usize);
    setter!(min_filter_improvement, f64);
    setter!(elimination_threshold, f64);
    setter!(min_q, f64);
    setter!(max_q, f64);
    setter!(min_db, f64);
    setter!(max_db, f64);
    setter!(min_freq, f64);
    setter!(max_freq, f64);
    setter!(max_iter, usize);
    setter!(population, usize);
    setter!(peq_model, String);
    setter!(seed, Option<u64>);
    setter!(refine, bool);
    setter!(local_algo, String);
    setter!(bo_initial_samples, Option<usize>);
    setter!(bo_batch_size, Option<usize>);
    setter!(bo_posterior_std_threshold, Option<f64>);
    setter!(bo_acquisition, Option<String>);
    setter!(bo_ehvi, Option<bool>);
    setter!(psychoacoustic, bool);
    setter!(
        psychoacoustic_smoothing,
        Option<crate::read::PsychoacousticSmoothingConfig>
    );
    setter!(smooth_n, usize);
    setter!(asymmetric_loss, bool);
    setter!(
        asymmetric_loss_config,
        Option<crate::loss::AsymmetricLossConfig>
    );
    setter!(perceptual_policy, Option<super::PerceptualPolicyConfig>);
    setter!(audibility_deadband, Option<super::AudibilityDeadbandConfig>);
    setter!(
        high_frequency_correction,
        Option<super::HighFrequencyCorrectionConfig>
    );
    setter!(
        early_late_correction,
        Option<super::EarlyLateCorrectionConfig>
    );
    setter!(validation_bundle, Option<super::ValidationBundleConfig>);
    setter!(tolerance, f64);
    setter!(atolerance, f64);
    setter!(allow_delay, Option<bool>);
    setter!(target_response, Option<super::TargetResponseConfig>);
    setter!(
        excursion_protection,
        Option<super::ExcursionProtectionConfig>
    );
    setter!(schroeder_split, Option<super::SchroederSplitConfig>);
    setter!(auto_optimizer, Option<super::AutoOptimizerConfig>);
    setter!(
        smoothness_penalty,
        Option<super::SmoothnessPenaltyConfigSerde>
    );
    setter!(phase_alignment, Option<super::PhaseAlignmentConfig>);
    setter!(group_delay, Option<super::GroupDelayOptimizationConfig>);
    setter!(multi_seat, Option<super::MultiSeatConfig>);
    setter!(
        inter_channel_timbre_matching,
        Option<super::InterChannelTimbreMatchingConfig>
    );
    setter!(
        height_channel_alignment,
        Option<super::HeightChannelAlignmentConfig>
    );
    setter!(vog, Option<super::InterChannelTimbreMatchingConfig>);
    setter!(multi_measurement, Option<super::MultiMeasurementConfig>);
    setter!(
        decomposed_correction,
        Option<super::DecomposedCorrectionSerdeConfig>
    );
    setter!(cea2034_correction, Option<super::Cea2034CorrectionConfig>);
    setter!(sub_config, Option<super::SubOptimizerConfig>);
    setter!(channel_matching, Option<super::ChannelMatchingConfig>);
    setter!(ssir_wav_path, Option<PathBuf>);
    setter!(max_boost_envelope, Option<Vec<(f64, f64)>>);
    setter!(min_cut_envelope, Option<Vec<(f64, f64)>>);
    setter!(from_measurement_slope_override, Option<f64>);
}

#[cfg(test)]
mod tests {
    use super::OptimizerConfigBuilder;
    use crate::roomeq::types::ProcessingMode;

    #[test]
    fn builder_starts_with_defaults() {
        let config = OptimizerConfigBuilder::new().build();
        assert_eq!(config.loss_type, "flat");
        assert!(config.refine);
    }

    #[test]
    fn builder_overrides_fields() {
        let config = OptimizerConfigBuilder::new()
            .loss_type("score".to_string())
            .num_filters(12)
            .processing_mode(ProcessingMode::PhaseLinear)
            .refine(false)
            .build();
        assert_eq!(config.loss_type, "score");
        assert_eq!(config.num_filters, 12);
        assert_eq!(config.processing_mode, ProcessingMode::PhaseLinear);
        assert!(!config.refine);
    }
}
