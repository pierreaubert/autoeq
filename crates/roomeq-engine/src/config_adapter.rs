//! Adapters from stable RoomEQ model settings to runtime backend settings.

use autoeq_optim::{LossType, OptimParams, PeqModel, SmoothnessPenaltyConfig};
use roomeq_model::{
    AsymmetricLossConfig, EpaConfig, FrequencyBandWeights, MultiMeasurementStrategy,
    OptimizerConfig, PsychoacousticSmoothingConfig, TemporalMaskingConfig, TemporalMaskingProfile,
};

/// Runtime projection of a neutral [`OptimizerConfig`].
pub trait OptimizerConfigExt {
    fn to_optim_params(&self, sample_rate: f64) -> OptimParams;
}

impl OptimizerConfigExt for OptimizerConfig {
    fn to_optim_params(&self, sample_rate: f64) -> OptimParams {
        let peq_model = self.peq_model.parse::<PeqModel>().unwrap_or(PeqModel::Pk);
        let loss = match self.loss_type.as_str() {
            "flat" if self.asymmetric_loss => LossType::SpeakerFlatAsymmetric,
            "flat" => LossType::SpeakerFlat,
            "score" => LossType::SpeakerScore,
            "epa" => LossType::Epa,
            _ => LossType::SpeakerFlat,
        };
        let smoothness_penalty = self.smoothness_penalty.as_ref().map(|value| {
            let mut runtime = SmoothnessPenaltyConfig {
                tv2_weight: value.tv2_weight,
                schroeder_hz: value.schroeder_hz,
                modal_weight_scale: value.modal_weight_scale,
                exponent: value.exponent,
            };
            if runtime.schroeder_hz.is_none()
                && let Some(split) = self.schroeder_split.as_ref().filter(|split| split.enabled)
            {
                runtime.schroeder_hz = Some(
                    split
                        .room_dimensions
                        .as_ref()
                        .map(|dimensions| dimensions.schroeder_frequency())
                        .unwrap_or(split.schroeder_freq),
                );
            }
            runtime
        });
        let audibility_deadband = self.audibility_deadband_config().map(|value| {
            autoeq_optim::roomeq::AudibilityDeadbandConfig {
                enabled: value.enabled,
                bass_db: value.bass_db,
                mid_db: value.mid_db,
                treble_db: value.treble_db,
                bass_mid_hz: value.bass_mid_hz,
                mid_treble_hz: value.mid_treble_hz,
                disable_below_schroeder: value.disable_below_schroeder,
                schroeder_hz: value.schroeder_hz,
            }
        });

        OptimParams {
            num_filters: self.num_filters,
            peq_model,
            sample_rate,
            min_freq: self.min_freq,
            max_freq: self.max_freq,
            min_q: self.min_q,
            max_q: self.max_q,
            min_db: self.min_db,
            max_db: self.max_db,
            loss,
            smooth: true,
            smooth_n: self.smooth_n,
            min_spacing_oct: 0.2,
            spacing_weight: 20.0,
            smoothness_penalty,
            audibility_deadband,
            algo: self.algorithm.clone(),
            population: self.population,
            maxeval: self.max_iter,
            refine: self.refine,
            local_algo: self.local_algo.clone(),
            bo_initial_samples: self.bo_initial_samples.unwrap_or(0),
            bo_batch_size: self.bo_batch_size.unwrap_or(0),
            bo_posterior_std_threshold: self.bo_posterior_std_threshold.unwrap_or(0.0),
            bo_acquisition: self
                .bo_acquisition
                .clone()
                .unwrap_or_else(|| "qei".to_string()),
            bo_ehvi: self.bo_ehvi.unwrap_or(false),
            strategy: self.strategy.clone(),
            tolerance: self.tolerance,
            atolerance: self.atolerance,
            recombination: 0.9,
            adaptive_weight_f: 0.9,
            adaptive_weight_cr: 0.9,
            no_parallel: false,
            parallel_threads: num_cpus::get(),
            seed: self.seed,
            quiet: false,
        }
    }
}

pub fn to_measurement_smoothing(
    value: PsychoacousticSmoothingConfig,
) -> autoeq_measurements::read::PsychoacousticSmoothingConfig {
    autoeq_measurements::read::PsychoacousticSmoothingConfig {
        low_freq_n: value.low_freq_n,
        high_freq_n: value.high_freq_n,
        low_freq: value.low_freq,
        high_freq: value.high_freq,
    }
}

pub fn to_optimizer_asymmetric_loss(
    value: AsymmetricLossConfig,
) -> autoeq_optim::loss::AsymmetricLossConfig {
    autoeq_optim::loss::AsymmetricLossConfig {
        peak_weight: value.peak_weight,
        dip_weight: value.dip_weight,
        bass_peak_weight: value.bass_peak_weight,
        bass_dip_weight: value.bass_dip_weight,
        transition_freq: value.transition_freq,
    }
}

fn to_optimizer_band_weights(
    value: &FrequencyBandWeights,
) -> autoeq_optim::loss::enhanced_weights::FrequencyBandWeights {
    autoeq_optim::loss::enhanced_weights::FrequencyBandWeights {
        bass_min: value.bass_min,
        bass_max: value.bass_max,
        mid_min: value.mid_min,
        mid_max: value.mid_max,
        treble_min: value.treble_min,
        treble_max: value.treble_max,
        bass_weight: value.bass_weight,
        mid_weight: value.mid_weight,
        treble_weight: value.treble_weight,
    }
}

fn to_optimizer_temporal_masking(
    value: &TemporalMaskingConfig,
) -> autoeq_optim::loss::epa::score::TemporalMaskingConfig {
    use autoeq_optim::loss::epa::score::TemporalMaskingProfile as RuntimeProfile;
    autoeq_optim::loss::epa::score::TemporalMaskingConfig {
        enabled: value.enabled,
        weight: value.weight,
        profile: match value.profile {
            TemporalMaskingProfile::Transient => RuntimeProfile::Transient,
            TemporalMaskingProfile::Mixed => RuntimeProfile::Mixed,
            TemporalMaskingProfile::Sustained => RuntimeProfile::Sustained,
        },
        ir_enabled: value.ir_enabled,
        ir_weight: value.ir_weight,
        pre_mask_ms: value.pre_mask_ms,
        post_mask_ms: value.post_mask_ms,
        pre_ringing_weight: value.pre_ringing_weight,
        post_ringing_weight: value.post_ringing_weight,
        ir_audibility_threshold_db: value.ir_audibility_threshold_db,
    }
}

pub fn to_optimizer_epa(value: &EpaConfig) -> autoeq_optim::loss::epa::score::EpaConfig {
    autoeq_optim::loss::epa::score::EpaConfig {
        listening_level_phon: value.listening_level_phon,
        target_sharpness: value.target_sharpness,
        max_roughness: value.max_roughness,
        evaluation_weight: value.evaluation_weight,
        potency_weight: value.potency_weight,
        activity_weight: value.activity_weight,
        flatness_band_weights: to_optimizer_band_weights(&value.flatness_band_weights),
        flatness_erb_weight: value.flatness_erb_weight,
        flatness_band_weight: value.flatness_band_weight,
        temporal_masking: to_optimizer_temporal_masking(&value.temporal_masking),
    }
}

pub fn to_optimizer_multi_measurement(
    value: MultiMeasurementStrategy,
) -> autoeq_optim::roomeq::MultiMeasurementStrategy {
    match value {
        MultiMeasurementStrategy::Average => {
            autoeq_optim::roomeq::MultiMeasurementStrategy::Average
        }
        MultiMeasurementStrategy::WeightedSum => {
            autoeq_optim::roomeq::MultiMeasurementStrategy::WeightedSum
        }
        MultiMeasurementStrategy::Minimax => {
            autoeq_optim::roomeq::MultiMeasurementStrategy::Minimax
        }
        MultiMeasurementStrategy::VariancePenalized => {
            autoeq_optim::roomeq::MultiMeasurementStrategy::VariancePenalized
        }
        MultiMeasurementStrategy::SpatialRobustness => {
            autoeq_optim::roomeq::MultiMeasurementStrategy::SpatialRobustness
        }
        MultiMeasurementStrategy::MinimaxUncertainty => {
            autoeq_optim::roomeq::MultiMeasurementStrategy::MinimaxUncertainty
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::{RoomDimensions, SchroederSplitConfig, SmoothnessPenaltyConfigSerde};

    fn smoothness_config(schroeder_hz: Option<f64>) -> SmoothnessPenaltyConfigSerde {
        SmoothnessPenaltyConfigSerde {
            tv2_weight: 0.05,
            schroeder_hz,
            modal_weight_scale: 0.1,
            exponent: 1.0,
        }
    }

    #[test]
    fn optimizer_adapter_uses_required_sample_rate() {
        let config = OptimizerConfig::default();
        for sample_rate in [44_100.0, 48_000.0, 96_000.0, 192_000.0] {
            assert_eq!(config.to_optim_params(sample_rate).sample_rate, sample_rate);
        }
    }

    #[test]
    fn optimizer_adapter_resolves_schroeder_frequency_and_bo_options() {
        let config = OptimizerConfig {
            smoothness_penalty: Some(smoothness_config(None)),
            schroeder_split: Some(SchroederSplitConfig {
                enabled: true,
                schroeder_freq: 280.0,
                room_dimensions: Some(RoomDimensions {
                    length: 4.0,
                    width: 3.0,
                    height: 2.5,
                }),
                ..Default::default()
            }),
            bo_initial_samples: Some(24),
            bo_batch_size: Some(4),
            bo_posterior_std_threshold: Some(0.02),
            bo_acquisition: Some("ei".to_string()),
            bo_ehvi: Some(true),
            ..Default::default()
        };

        let params = config.to_optim_params(48_000.0);
        let expected = config
            .schroeder_split
            .as_ref()
            .unwrap()
            .room_dimensions
            .as_ref()
            .unwrap()
            .schroeder_frequency();
        assert_eq!(
            params.smoothness_penalty.unwrap().schroeder_hz,
            Some(expected)
        );
        assert_eq!(params.bo_initial_samples, 24);
        assert_eq!(params.bo_batch_size, 4);
        assert_eq!(params.bo_posterior_std_threshold, 0.02);
        assert_eq!(params.bo_acquisition, "ei");
        assert!(params.bo_ehvi);
    }
}
