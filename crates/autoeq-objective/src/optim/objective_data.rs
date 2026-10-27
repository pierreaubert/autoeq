use super::super::loss::{
    AsymmetricLossConfig, DriversLossData, HeadphoneLossData, LossType, SpeakerLossData,
};
use super::loss::{
    AsymmetricStrategy, DriversMode, DriversStrategy, EpaStrategy, FlatStrategy,
    HeadphoneScoreStrategy, Objective, ObjectiveContext, SpeakerScoreStrategy,
};
use super::penalty_mode::PenaltyMode;
use super::smoothness_penalty_config::SmoothnessPenaltyConfig;
use super::types::MultiObjectiveData;
use crate::Curve;
use crate::PeqModel;
use ndarray::Array1;
use std::sync::{Arc, OnceLock};

/// Data structure for holding objective function parameters
///
/// This struct contains all the data needed to compute the objective function
/// for filter optimization.
#[derive(Clone)]
pub struct ObjectiveData {
    /// Lazily prepared fixed-grid loss kernel shared by all clones.
    ///
    /// This is public only to preserve construction through historical struct
    /// literals. Callers should initialize it with `Default::default()` and
    /// otherwise treat it as an implementation detail.
    #[doc(hidden)]
    pub prepared: Arc<OnceLock<super::prepared_objective::PreparedObjective>>,
    /// Frequency points for evaluation
    pub freqs: Arc<Array1<f64>>,
    /// Target spl
    pub target: Arc<Array1<f64>>,
    /// Target error values
    pub deviation: Arc<Array1<f64>>,
    /// Sample rate in Hz
    pub srate: f64,
    #[allow(dead_code)]
    /// Minimum spacing between filters in octaves
    pub min_spacing_oct: f64,
    /// Weight for spacing penalty term
    pub spacing_weight: f64,
    /// Maximum allowed dB level
    pub max_db: f64,
    /// Minimum absolute gain for filters
    pub min_db: f64,
    /// Minimum frequency in Hz for loss function evaluation
    pub min_freq: f64,
    /// Maximum frequency in Hz for loss function evaluation
    pub max_freq: f64,
    /// PEQ model that defines the filter structure
    pub peq_model: PeqModel,
    /// Type of loss function to use
    pub loss_type: LossType,
    /// Cached strategy object used to evaluate this objective.
    ///
    /// When `None`, [`ObjectiveData::build_objective`] is used as a fallback.
    /// Setup code should populate this field so the strategy is built once.
    #[allow(dead_code)]
    pub objective: Option<Arc<dyn Objective>>,
    /// Optional score data for SpeakerScore loss type
    pub speaker_score_data: Option<SpeakerLossData>,
    /// Optional score data for HeadphoneScore loss type
    pub headphone_score_data: Option<HeadphoneLossData>,
    /// Input curve for headphone loss (optional)
    pub input_curve: Option<Curve>,
    /// Optional data for multi-driver crossover optimization
    pub drivers_data: Option<DriversLossData>,
    /// Fixed crossover frequencies (when not optimizing frequencies)
    pub fixed_crossover_freqs: Option<Vec<f64>>,
    /// Penalty weights used when the optimizer does not support nonlinear constraints
    /// If zero, penalties are disabled and true constraints (if any) are used.
    /// Penalty for ceiling constraint
    pub penalty_w_ceiling: f64,
    /// Penalty for spacing constraint
    pub penalty_w_spacing: f64,
    /// Penalty for min gain constraint
    pub penalty_w_mingain: f64,
    /// Integrality constraints - true for integer parameters, false for continuous
    pub integrality: Option<Vec<bool>>,
    /// Multi-objective data for multi-measurement optimization.
    /// When `Some`, `compute_base_fitness` delegates to `compute_multi_objective_fitness`.
    pub multi_objective: Option<MultiObjectiveData>,
    /// Whether to smooth the error before computing the loss
    pub smooth: bool,
    /// Smoothing resolution as 1/N octave
    pub smooth_n: usize,
    /// Frequency-dependent maximum boost envelope for per-filter gain clamping.
    /// Each entry is (frequency_hz, max_boost_db). Interpolated in log-frequency.
    /// When Some, positive filter gains are clamped before loss evaluation.
    pub max_boost_envelope: Option<Vec<(f64, f64)>>,
    /// Optional minimum-cut safety envelope.
    /// Each entry is (frequency_hz, max_cut_db), where max_cut_db is negative.
    /// When Some, negative filter gains are clamped before loss evaluation.
    pub min_cut_envelope: Option<Vec<(f64, f64)>>,
    /// Experimental EPA diagnostics and active spectral-flatness blend.
    ///
    /// Used by the EPA objective and optional flat-objective spectral regression
    /// cost. When `None`, the optimizer uses `EpaConfig::default()`.
    pub epa_config: Option<crate::loss::epa::score::EpaConfig>,
    /// Detected modal ringing data retained for EPA diagnostic compatibility.
    ///
    /// It does not steer optimizer fitness.
    pub temporal_masking_modes: Vec<crate::loss::epa::score::TemporalMaskingMode>,
    /// Pre-detected frequency problems (usually SSIR / decomposed-correction
    /// room modes) to seed the DE optimizer's smart initial-guess
    /// generator with.
    ///
    /// Each entry is `(frequency_hz, q, suggested_gain_db)` — the sign of
    /// the gain says whether the seed should cut (negative) or boost
    /// (positive) the problem. When this list is non-empty,
    /// `initial_guess::create_smart_initial_guesses` will use it instead
    /// of running its own naive `find_peaks` over the smoothed deviation.
    /// Order should be "most important first" (i.e. sorted by
    /// `|gain_db|` descending) so that if there are fewer filters than
    /// problems the most prominent ones are the ones kept.
    ///
    /// Defaults to empty, which preserves the old behaviour
    /// (auto-detected peaks/dips).
    pub detected_problems: Vec<(f64, f64, f64)>,
    /// Measured modal evidence used by the audibility veto's optional
    /// proximity rule.  This is empty when no decomposition evidence exists.
    pub mode_proximity_evidence: Vec<crate::roomeq::ModeProximityEvidence>,
    /// Per-frequency dip-suppression mask for RoomEQ flat and asymmetric
    /// objectives.
    ///
    /// `Some(mask)` scales only the boost-seeking dip residual toward zero
    /// inside detected narrow nulls (see
    /// [`crate::roomeq::impulse_analysis::build_null_suppression_mask`]).
    /// `None` disables suppression; asymmetric dips use their configured
    /// `bass_dip_weight` / `dip_weight`, while flat loss remains symmetric.
    /// Must have the same length as `freqs` when provided.
    pub null_suppression: Option<Arc<Array1<f64>>>,
    /// Peak/dip weights for [`LossType::SpeakerFlatAsymmetric`].
    pub asymmetric_loss_config: AsymmetricLossConfig,
    /// Optional smoothness regularizer on the correction curve.
    /// `None` disables the penalty.
    pub smoothness_penalty: Option<SmoothnessPenaltyConfig>,
    /// Optional peak-filter Q-rail regularizer `(rail_q, relative_gain_weight)`.
    ///
    /// Adds weight times absolute gain times `(Q / rail_q)^4`, summed over
    /// active standard/matched peaks. Shelves, neutral filters, and non-PEQ losses are
    /// excluded. This is an engineering complexity cost, not audibility proof.
    pub q_rail_penalty: Option<(f64, f64)>,
    /// Optional measurement-derived peak bandwidth guard for RoomEQ.
    /// Search and final emission project peak Q using the same retained grids.
    pub peak_resolution_guard: Option<super::resolution_guard::PeakResolutionGuard>,
    /// Captured one-RMS perturbation bound for this prepared scalar loss.
    ///
    /// Runtime RoomEQ evidence for adaptive stopping, not statistical confidence.
    /// `None` means unavailable; spatial variation must not supply this value.
    pub captured_loss_perturbation_bound: Option<f64>,
    /// Weight for regression in EPA spectral error against the frozen baseline.
    ///
    /// Zero disables this cost. Uses the EPA flatness configuration without
    /// residual deadband or null suppression. It does not optimize transfer-only
    /// loudness, roughness, sharpness, or preference diagnostics.
    /// Only flat PEQ and flat realized-response objectives consult this weight.
    pub epa_flatness_regression_weight: f64,
    /// Optional audibility/JND deadband applied to residual errors before
    /// spectral objective scoring.
    pub audibility_deadband: Option<crate::roomeq::AudibilityDeadbandConfig>,
}

impl ObjectiveData {
    pub(crate) fn prepared(&self) -> &super::prepared_objective::PreparedObjective {
        self.prepared
            .get_or_init(|| super::prepared_objective::PreparedObjective::new(self))
    }

    pub fn with_run_control(&self, run_control: super::run_control::OptimizerRunControl) -> Self {
        self.with_evaluation_tracking(run_control, super::run_control::EvaluationStage::Search)
    }

    /// Count post-search scores separately from the shared search cap.
    ///
    /// Finalization and pipeline safety checks remain possible after the search
    /// cap is exhausted. Callers must check cancellation before starting them.
    pub fn with_validation_tracking(
        &self,
        run_control: super::run_control::OptimizerRunControl,
    ) -> Self {
        self.with_evaluation_tracking(run_control, super::run_control::EvaluationStage::Validation)
    }

    /// Preserve shared accounting while moving post-search scores to validation.
    pub fn post_search_validation_view(&self) -> Self {
        self.prepared().run_control.as_ref().map_or_else(
            || self.clone(),
            |control| self.with_validation_tracking(control.clone()),
        )
    }

    /// Check explicit terminal requests independently of search-cap exhaustion.
    pub fn terminal_stop_requested(&self) -> bool {
        self.prepared().run_control.as_ref().is_some_and(|control| {
            let snapshot = control.snapshot();
            snapshot.cancellation_requested || snapshot.deadline_reached
        })
    }

    fn with_evaluation_tracking(
        &self,
        run_control: super::run_control::OptimizerRunControl,
        stage: super::run_control::EvaluationStage,
    ) -> Self {
        // Do not share a source cache that may already have been initialized:
        // every controlled run gets candidate-independent data tied to this
        // ObjectiveData and its own accounting gate.
        let prepared = std::sync::OnceLock::new();
        let mut prepared_objective =
            super::prepared_objective::PreparedObjective::new(self).with_run_control(run_control);
        prepared_objective.evaluation_stage = stage;
        let _ = prepared.set(prepared_objective);
        let mut controlled = self.clone();
        controlled.prepared = Arc::new(prepared);
        controlled
    }

    /// Build the [`Objective`] strategy that corresponds to the configured
    /// [`LossType`] and payload fields.
    pub fn build_objective(&self) -> Arc<dyn Objective> {
        match self.loss_type {
            LossType::SpeakerFlat | LossType::HeadphoneFlat => Arc::new(FlatStrategy),
            LossType::SpeakerFlatAsymmetric => Arc::new(AsymmetricStrategy {
                config: self.asymmetric_loss_config,
                null_suppression: self.null_suppression.clone(),
            }),
            LossType::SpeakerScore => {
                if let Some(ref sd) = self.speaker_score_data {
                    Arc::new(SpeakerScoreStrategy {
                        score_data: sd.clone(),
                    })
                } else {
                    log::error!("speaker score loss requested but score data is missing");
                    Arc::new(MissingDataStrategy)
                }
            }
            LossType::HeadphoneScore => {
                if let Some(ref hd) = self.headphone_score_data {
                    Arc::new(HeadphoneScoreStrategy { data: hd.clone() })
                } else {
                    log::error!("headphone score loss requested but headphone data is missing");
                    Arc::new(MissingDataStrategy)
                }
            }
            LossType::DriversFlat => {
                if let Some(ref dd) = self.drivers_data {
                    Arc::new(DriversStrategy {
                        data: dd.clone(),
                        mode: DriversMode::Flat,
                        fixed_crossover_freqs: self.fixed_crossover_freqs.clone(),
                    })
                } else {
                    log::error!("drivers-flat loss requested but driver data is missing");
                    Arc::new(MissingDataStrategy)
                }
            }
            LossType::MultiSubFlat => {
                if let Some(ref dd) = self.drivers_data {
                    Arc::new(DriversStrategy {
                        data: dd.clone(),
                        mode: DriversMode::MultiSub,
                        fixed_crossover_freqs: None,
                    })
                } else {
                    log::error!("multi-sub-flat loss requested but driver data is missing");
                    Arc::new(MissingDataStrategy)
                }
            }
            LossType::Epa => Arc::new(EpaStrategy {
                config: self.epa_config.clone().unwrap_or_default(),
                temporal_masking_modes: self.temporal_masking_modes.clone(),
            }),
        }
    }

    /// Configure penalty weights based on the optimizer's requirements.
    ///
    /// Call this before optimization to set appropriate penalty weights.
    /// Use `PenaltyMode::Disabled` when the optimizer supports native constraints.
    pub fn configure_penalties(&mut self, mode: PenaltyMode) {
        self.penalty_w_ceiling = mode.ceiling_weight();
        self.penalty_w_mingain = mode.mingain_weight();
        // Spacing weight is computed from spacing_weight config, scaled by mode
        let spacing_scale = match mode {
            PenaltyMode::Disabled => 0.0,
            PenaltyMode::Standard => 1e3,
            PenaltyMode::Pso => 5e2,
        };
        self.penalty_w_spacing = self.spacing_weight.max(0.0) * spacing_scale;
    }
}

/// Placeholder strategy used when a loss type is selected but its required
/// payload data is missing.  Returning infinity keeps the optimizer from
/// selecting the invalid candidate.
struct MissingDataStrategy;

impl Objective for MissingDataStrategy {
    fn compute(&self, _x: &[f64], _ctx: &ObjectiveContext) -> f64 {
        f64::INFINITY
    }
}

impl std::fmt::Debug for ObjectiveData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectiveData")
            .field("freqs", &self.freqs)
            .field("target", &self.target)
            .field("deviation", &self.deviation)
            .field("srate", &self.srate)
            .field("min_spacing_oct", &self.min_spacing_oct)
            .field("spacing_weight", &self.spacing_weight)
            .field("max_db", &self.max_db)
            .field("min_db", &self.min_db)
            .field("min_freq", &self.min_freq)
            .field("max_freq", &self.max_freq)
            .field("peq_model", &self.peq_model)
            .field("loss_type", &self.loss_type)
            .field("objective", &self.objective.as_ref().map(|_| "<Objective>"))
            .field("speaker_score_data", &self.speaker_score_data)
            .field("headphone_score_data", &self.headphone_score_data)
            .field("input_curve", &self.input_curve)
            .field("drivers_data", &self.drivers_data)
            .field("fixed_crossover_freqs", &self.fixed_crossover_freqs)
            .field("penalty_w_ceiling", &self.penalty_w_ceiling)
            .field("penalty_w_spacing", &self.penalty_w_spacing)
            .field("penalty_w_mingain", &self.penalty_w_mingain)
            .field("integrality", &self.integrality)
            .field("multi_objective", &self.multi_objective)
            .field("smooth", &self.smooth)
            .field("smooth_n", &self.smooth_n)
            .field("max_boost_envelope", &self.max_boost_envelope)
            .field("min_cut_envelope", &self.min_cut_envelope)
            .field("epa_config", &self.epa_config)
            .field("temporal_masking_modes", &self.temporal_masking_modes)
            .field("detected_problems", &self.detected_problems)
            .field("null_suppression", &self.null_suppression)
            .field("asymmetric_loss_config", &self.asymmetric_loss_config)
            .field("smoothness_penalty", &self.smoothness_penalty)
            .field("q_rail_penalty", &self.q_rail_penalty)
            .field("peak_resolution_guard", &self.peak_resolution_guard)
            .field(
                "captured_loss_perturbation_bound",
                &self.captured_loss_perturbation_bound,
            )
            .field(
                "epa_flatness_regression_weight",
                &self.epa_flatness_regression_weight,
            )
            .field("audibility_deadband", &self.audibility_deadband)
            .finish()
    }
}
