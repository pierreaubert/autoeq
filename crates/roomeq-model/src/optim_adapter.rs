use crate::config::OptimizerConfig;
use autoeq_optim::{LossType, OptimParams, PeqModel, RoomOptimizerConfig, SmoothnessPenaltyConfig};

impl RoomOptimizerConfig for OptimizerConfig {
    fn to_optim_params(&self) -> OptimParams {
        let peq_model = self.peq_model.parse::<PeqModel>().unwrap_or(PeqModel::Pk);
        let loss = match self.loss_type.as_str() {
            "flat" if self.asymmetric_loss => LossType::SpeakerFlatAsymmetric,
            "flat" => LossType::SpeakerFlat,
            "score" => LossType::SpeakerScore,
            "epa" => LossType::Epa,
            _ => LossType::SpeakerFlat,
        };
        let smoothness_penalty = self.smoothness_penalty.as_ref().map(|value| {
            let mut value = SmoothnessPenaltyConfig::from(value);
            if value.schroeder_hz.is_none()
                && let Some(split) = self.schroeder_split.as_ref().filter(|split| split.enabled)
            {
                value.schroeder_hz = Some(
                    split
                        .room_dimensions
                        .as_ref()
                        .map(|dimensions| dimensions.schroeder_frequency())
                        .unwrap_or(split.schroeder_freq),
                );
            }
            value
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
            sample_rate: 48000.0,
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
            bo_acquisition: self.bo_acquisition.clone().unwrap_or_else(|| "qei".to_string()),
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
