//! Builder for [`ObjectiveData`].
//!
//! The builder enforces per-loss invariants at construction time and caches the
//! matching [`Objective`] strategy so the caller does not have to remember to
//! populate the `objective` field manually.

use crate::Curve;
use crate::PeqModel;
use crate::error::AutoeqError;
use crate::loss::{
    AsymmetricLossConfig, DriversLossData, HeadphoneLossData, LossType, SpeakerLossData,
};
use crate::optim::{MultiObjectiveData, ObjectiveData, SmoothnessPenaltyConfig};
use crate::roomeq::AudibilityDeadbandConfig;
use ndarray::Array1;

/// Builder for constructing an [`ObjectiveData`].
///
/// Required fields are supplied to the constructor; optional fields have fluent
/// setters. The [`Self::build`] method validates loss-specific invariants and
/// caches the objective strategy.
#[derive(Debug, Clone)]
pub struct ObjectiveDataBuilder {
    freqs: Array1<f64>,
    target: Array1<f64>,
    deviation: Array1<f64>,
    srate: f64,
    loss_type: LossType,
    peq_model: PeqModel,

    min_spacing_oct: f64,
    spacing_weight: f64,
    max_db: f64,
    min_db: f64,
    min_freq: f64,
    max_freq: f64,

    speaker_score_data: Option<SpeakerLossData>,
    headphone_score_data: Option<HeadphoneLossData>,
    input_curve: Option<Curve>,
    drivers_data: Option<DriversLossData>,
    fixed_crossover_freqs: Option<Vec<f64>>,

    penalty_w_ceiling: f64,
    penalty_w_spacing: f64,
    penalty_w_mingain: f64,

    integrality: Option<Vec<bool>>,
    multi_objective: Option<MultiObjectiveData>,

    smooth: bool,
    smooth_n: usize,

    max_boost_envelope: Option<Vec<(f64, f64)>>,
    min_cut_envelope: Option<Vec<(f64, f64)>>,

    epa_config: Option<crate::loss::epa::score::EpaConfig>,
    temporal_masking_modes: Vec<crate::loss::epa::score::TemporalMaskingMode>,
    detected_problems: Vec<(f64, f64, f64)>,
    null_suppression: Option<Array1<f64>>,
    asymmetric_loss_config: AsymmetricLossConfig,
    smoothness_penalty: Option<SmoothnessPenaltyConfig>,
    audibility_deadband: Option<AudibilityDeadbandConfig>,
}

impl ObjectiveDataBuilder {
    /// Create a builder with the minimal fields required for every objective.
    ///
    /// Bounds and frequency range default to `0.0`; set them with the
    /// corresponding methods or use one of the loss-type convenience
    /// constructors.
    pub fn new(
        freqs: Array1<f64>,
        target: Array1<f64>,
        deviation: Array1<f64>,
        srate: f64,
        peq_model: PeqModel,
        loss_type: LossType,
    ) -> Self {
        Self {
            freqs,
            target,
            deviation,
            srate,
            loss_type,
            peq_model,
            min_spacing_oct: 0.0,
            spacing_weight: 0.0,
            max_db: 0.0,
            min_db: 0.0,
            min_freq: 0.0,
            max_freq: 0.0,
            speaker_score_data: None,
            headphone_score_data: None,
            input_curve: None,
            drivers_data: None,
            fixed_crossover_freqs: None,
            penalty_w_ceiling: 0.0,
            penalty_w_spacing: 0.0,
            penalty_w_mingain: 0.0,
            integrality: None,
            multi_objective: None,
            smooth: false,
            smooth_n: 1,
            max_boost_envelope: None,
            min_cut_envelope: None,
            epa_config: None,
            temporal_masking_modes: Vec::new(),
            detected_problems: Vec::new(),
            null_suppression: None,
            asymmetric_loss_config: AsymmetricLossConfig::default(),
            smoothness_penalty: None,
            audibility_deadband: None,
        }
    }

    /// Convenience constructor for `LossType::SpeakerFlat`.
    pub fn speaker_flat(
        freqs: Array1<f64>,
        target: Array1<f64>,
        deviation: Array1<f64>,
        srate: f64,
        peq_model: PeqModel,
    ) -> Self {
        Self::new(
            freqs,
            target,
            deviation,
            srate,
            peq_model,
            LossType::SpeakerFlat,
        )
    }

    /// Convenience constructor for `LossType::SpeakerFlatAsymmetric`.
    pub fn speaker_flat_asymmetric(
        freqs: Array1<f64>,
        target: Array1<f64>,
        deviation: Array1<f64>,
        srate: f64,
        peq_model: PeqModel,
    ) -> Self {
        Self::new(
            freqs,
            target,
            deviation,
            srate,
            peq_model,
            LossType::SpeakerFlatAsymmetric,
        )
    }

    /// Convenience constructor for `LossType::SpeakerScore`.
    pub fn speaker_score(
        freqs: Array1<f64>,
        target: Array1<f64>,
        deviation: Array1<f64>,
        srate: f64,
        peq_model: PeqModel,
        speaker_score_data: SpeakerLossData,
    ) -> Self {
        let mut b = Self::new(
            freqs,
            target,
            deviation,
            srate,
            peq_model,
            LossType::SpeakerScore,
        );
        b.speaker_score_data = Some(speaker_score_data);
        b
    }

    /// Convenience constructor for `LossType::HeadphoneFlat`.
    pub fn headphone_flat(
        freqs: Array1<f64>,
        target: Array1<f64>,
        deviation: Array1<f64>,
        srate: f64,
        peq_model: PeqModel,
    ) -> Self {
        Self::new(
            freqs,
            target,
            deviation,
            srate,
            peq_model,
            LossType::HeadphoneFlat,
        )
    }

    /// Convenience constructor for `LossType::HeadphoneScore`.
    pub fn headphone_score(
        freqs: Array1<f64>,
        target: Array1<f64>,
        deviation: Array1<f64>,
        srate: f64,
        peq_model: PeqModel,
        headphone_score_data: HeadphoneLossData,
    ) -> Self {
        let mut b = Self::new(
            freqs,
            target,
            deviation,
            srate,
            peq_model,
            LossType::HeadphoneScore,
        );
        b.headphone_score_data = Some(headphone_score_data);
        b
    }

    /// Convenience constructor for `LossType::DriversFlat`.
    pub fn drivers_flat(
        freq_grid: Array1<f64>,
        srate: f64,
        peq_model: PeqModel,
        drivers_data: DriversLossData,
    ) -> Self {
        let n = freq_grid.len();
        let mut b = Self::new(
            freq_grid,
            Array1::zeros(n),
            Array1::zeros(n),
            srate,
            peq_model,
            LossType::DriversFlat,
        );
        b.drivers_data = Some(drivers_data);
        b
    }

    /// Convenience constructor for `LossType::MultiSubFlat`.
    pub fn multi_sub_flat(
        freq_grid: Array1<f64>,
        srate: f64,
        peq_model: PeqModel,
        drivers_data: DriversLossData,
    ) -> Self {
        let n = freq_grid.len();
        let mut b = Self::new(
            freq_grid,
            Array1::zeros(n),
            Array1::zeros(n),
            srate,
            peq_model,
            LossType::MultiSubFlat,
        );
        b.drivers_data = Some(drivers_data);
        b
    }

    /// Convenience constructor for `LossType::Epa`.
    pub fn epa(
        freqs: Array1<f64>,
        target: Array1<f64>,
        deviation: Array1<f64>,
        srate: f64,
        peq_model: PeqModel,
    ) -> Self {
        Self::new(freqs, target, deviation, srate, peq_model, LossType::Epa)
    }

    /// Set the minimum spacing between filters in octaves.
    pub fn min_spacing_oct(mut self, v: f64) -> Self {
        self.min_spacing_oct = v;
        self
    }

    /// Set the spacing-penalty weight.
    pub fn spacing_weight(mut self, v: f64) -> Self {
        self.spacing_weight = v;
        self
    }

    /// Set the maximum absolute filter gain in dB.
    pub fn max_db(mut self, v: f64) -> Self {
        self.max_db = v;
        self
    }

    /// Set the minimum absolute filter gain in dB.
    pub fn min_db(mut self, v: f64) -> Self {
        self.min_db = v;
        self
    }

    /// Set the loss-evaluation frequency range.
    pub fn freq_range(mut self, min_freq: f64, max_freq: f64) -> Self {
        self.min_freq = min_freq;
        self.max_freq = max_freq;
        self
    }

    /// Set the score data required by `LossType::SpeakerScore`.
    pub fn speaker_score_data(mut self, v: SpeakerLossData) -> Self {
        self.speaker_score_data = Some(v);
        self
    }

    /// Set the score data required by `LossType::HeadphoneScore`.
    pub fn headphone_score_data(mut self, v: HeadphoneLossData) -> Self {
        self.headphone_score_data = Some(v);
        self
    }

    /// Set the input curve used by headphone loss calculations.
    pub fn input_curve(mut self, v: Curve) -> Self {
        self.input_curve = Some(v);
        self
    }

    /// Set the driver/sub data required by driver/sub losses.
    pub fn drivers_data(mut self, v: DriversLossData) -> Self {
        self.drivers_data = Some(v);
        self
    }

    /// Set fixed crossover frequencies for driver optimization.
    pub fn fixed_crossover_freqs(mut self, v: Vec<f64>) -> Self {
        self.fixed_crossover_freqs = Some(v);
        self
    }

    /// Set penalty weights.
    pub fn penalty_weights(mut self, ceiling: f64, spacing: f64, mingain: f64) -> Self {
        self.penalty_w_ceiling = ceiling;
        self.penalty_w_spacing = spacing;
        self.penalty_w_mingain = mingain;
        self
    }

    /// Set integrality constraints.
    pub fn integrality(mut self, v: Vec<bool>) -> Self {
        self.integrality = Some(v);
        self
    }

    /// Set multi-objective data.
    pub fn multi_objective(mut self, v: MultiObjectiveData) -> Self {
        self.multi_objective = Some(v);
        self
    }

    /// Set smoothing parameters.
    pub fn smoothing(mut self, enabled: bool, n: usize) -> Self {
        self.smooth = enabled;
        self.smooth_n = n;
        self
    }

    /// Set the maximum-boost envelope.
    pub fn max_boost_envelope(mut self, v: Vec<(f64, f64)>) -> Self {
        self.max_boost_envelope = Some(v);
        self
    }

    /// Set the minimum-cut envelope.
    pub fn min_cut_envelope(mut self, v: Vec<(f64, f64)>) -> Self {
        self.min_cut_envelope = Some(v);
        self
    }

    /// Set the EPA configuration.
    pub fn epa_config(mut self, v: crate::loss::epa::score::EpaConfig) -> Self {
        self.epa_config = Some(v);
        self
    }

    /// Set temporal-masking modes for EPA.
    pub fn temporal_masking_modes(
        mut self,
        v: Vec<crate::loss::epa::score::TemporalMaskingMode>,
    ) -> Self {
        self.temporal_masking_modes = v;
        self
    }

    /// Set pre-detected problems for smart initial guesses.
    pub fn detected_problems(mut self, v: Vec<(f64, f64, f64)>) -> Self {
        self.detected_problems = v;
        self
    }

    /// Set the null-suppression mask for asymmetric loss.
    pub fn null_suppression(mut self, v: Array1<f64>) -> Self {
        self.null_suppression = Some(v);
        self
    }

    /// Set the asymmetric-loss weights.
    pub fn asymmetric_loss_config(mut self, v: AsymmetricLossConfig) -> Self {
        self.asymmetric_loss_config = v;
        self
    }

    /// Set the smoothness-penalty configuration.
    pub fn smoothness_penalty(mut self, v: SmoothnessPenaltyConfig) -> Self {
        self.smoothness_penalty = Some(v);
        self
    }

    /// Set the smoothness-penalty configuration from an `Option`.
    pub fn smoothness_penalty_opt(mut self, v: Option<SmoothnessPenaltyConfig>) -> Self {
        self.smoothness_penalty = v;
        self
    }

    /// Set the audibility-deadband configuration.
    pub fn audibility_deadband(mut self, v: AudibilityDeadbandConfig) -> Self {
        self.audibility_deadband = Some(v);
        self
    }

    /// Set the audibility-deadband configuration from an `Option`.
    pub fn audibility_deadband_opt(mut self, v: Option<AudibilityDeadbandConfig>) -> Self {
        self.audibility_deadband = v;
        self
    }

    /// Build the [`ObjectiveData`], validating invariants and caching the
    /// objective strategy.
    ///
    /// # Errors
    /// Returns `AutoeqError::InvalidConfiguration` if:
    /// - `freqs`, `target`, and `deviation` have different lengths.
    /// - The chosen `loss_type` is missing required payload data.
    /// - `fixed_crossover_freqs` has the wrong length for the driver data.
    pub fn build(self) -> Result<ObjectiveData, AutoeqError> {
        let n = self.freqs.len();
        if self.target.len() != n || self.deviation.len() != n {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!(
                    "ObjectiveData length mismatch: freqs={}, target={}, deviation={}",
                    n,
                    self.target.len(),
                    self.deviation.len()
                ),
            });
        }

        match self.loss_type {
            LossType::SpeakerScore if self.speaker_score_data.is_none() => {
                return Err(AutoeqError::InvalidConfiguration {
                    message: "SpeakerScore loss requires speaker_score_data".to_string(),
                });
            }
            LossType::HeadphoneScore if self.headphone_score_data.is_none() => {
                return Err(AutoeqError::InvalidConfiguration {
                    message: "HeadphoneScore loss requires headphone_score_data".to_string(),
                });
            }
            LossType::DriversFlat | LossType::MultiSubFlat if self.drivers_data.is_none() => {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!("{:?} loss requires drivers_data", self.loss_type),
                });
            }
            _ => {}
        }

        if let (Some(drivers), Some(fixed)) = (&self.drivers_data, &self.fixed_crossover_freqs) {
            let expected = drivers.drivers.len().saturating_sub(1);
            if fixed.len() != expected {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "fixed_crossover_freqs length {} does not match {} drivers (expected {})",
                        fixed.len(),
                        drivers.drivers.len(),
                        expected
                    ),
                });
            }
        }

        let mut data = ObjectiveData {
            freqs: self.freqs,
            target: self.target,
            deviation: self.deviation,
            srate: self.srate,
            min_spacing_oct: self.min_spacing_oct,
            spacing_weight: self.spacing_weight,
            max_db: self.max_db,
            min_db: self.min_db,
            min_freq: self.min_freq,
            max_freq: self.max_freq,
            peq_model: self.peq_model,
            loss_type: self.loss_type,
            objective: None,
            speaker_score_data: self.speaker_score_data,
            headphone_score_data: self.headphone_score_data,
            input_curve: self.input_curve,
            drivers_data: self.drivers_data,
            fixed_crossover_freqs: self.fixed_crossover_freqs,
            penalty_w_ceiling: self.penalty_w_ceiling,
            penalty_w_spacing: self.penalty_w_spacing,
            penalty_w_mingain: self.penalty_w_mingain,
            integrality: self.integrality,
            multi_objective: self.multi_objective,
            smooth: self.smooth,
            smooth_n: self.smooth_n,
            max_boost_envelope: self.max_boost_envelope,
            min_cut_envelope: self.min_cut_envelope,
            epa_config: self.epa_config,
            temporal_masking_modes: self.temporal_masking_modes,
            detected_problems: self.detected_problems,
            null_suppression: self.null_suppression,
            asymmetric_loss_config: self.asymmetric_loss_config,
            smoothness_penalty: self.smoothness_penalty,
            audibility_deadband: self.audibility_deadband,
        };

        data.objective = Some(data.build_objective());
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PeqModel;
    use ndarray::array;

    #[test]
    fn builder_rejects_length_mismatch() {
        let result = ObjectiveDataBuilder::speaker_flat(
            array![100.0, 1000.0],
            array![0.0, 0.0],
            array![1.0], // wrong length
            48000.0,
            PeqModel::Pk,
        )
        .max_db(6.0)
        .min_db(0.0)
        .freq_range(20.0, 20000.0)
        .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_missing_speaker_score_data() {
        let result = ObjectiveDataBuilder::new(
            array![100.0],
            array![0.0],
            array![0.0],
            48000.0,
            PeqModel::Pk,
            LossType::SpeakerScore,
        )
        .build();
        assert!(result.is_err());
    }
}
