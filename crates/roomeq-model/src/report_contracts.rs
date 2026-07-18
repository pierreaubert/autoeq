//! Neutral serialized result and diagnostic contracts.

use ndarray::Array1;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// True impulse-response temporal masking metrics for FIR / phase correction.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TemporalIrMaskingMetrics {
    /// Main impulse sample index used as the transient reference.
    pub main_index: usize,
    /// Main impulse time in milliseconds from the start of the FIR.
    pub main_time_ms: f64,
    /// Peak pre-ringing level before the main impulse, dB relative to main.
    pub pre_ringing_peak_db: f64,
    /// Peak post-ringing level after the main impulse, dB relative to main.
    pub post_ringing_peak_db: f64,
    /// Pre-masked audible pre-ringing energy, dB relative to main peak energy.
    pub pre_ringing_audible_db: f64,
    /// Post-masked audible post-ringing energy, dB relative to main peak energy.
    pub post_ringing_audible_db: f64,
    /// Scalar penalty using the configured material profile and IR weights.
    pub penalty: f64,
}

/// EPA dimensions computed from a frequency response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EpaScore {
    /// Evaluation: general quality (higher = better, 0-10 scale)
    pub evaluation: f64,
    /// Potency: perceived energy/strength (0-10 scale)
    pub potency: f64,
    /// Activity: temporal complexity (lower = calmer, 0-10 scale)
    pub activity: f64,
    /// Composite preference (weighted combination, higher = better)
    pub preference: f64,
    /// Individual metric values for diagnostics
    pub sharpness_acum: f64,
    pub roughness: f64,
    pub total_loudness_sone: f64,
    pub loudness_balance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerTermination {
    Converged,
    EvaluationLimit,
    NonConverged,
    UserStopped,
    BackendFailure,
    InvalidResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerConfidence {
    High,
    Low,
    Unusable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OptimizerRestartEvidence {
    pub attempt: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub termination: OptimizerTermination,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<f64>,
}

const fn default_true() -> bool {
    true
}

/// Structured evidence for one optimizer invocation.
///
/// Backends retain their historical tuple API, but callers should use this
/// type for production acceptance. In particular, an `Ok` tuple containing
/// "not converged" is classified as best-effort rather than success.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OptimizerRunEvidence {
    pub algorithm: String,
    pub termination: OptimizerTermination,
    pub converged: bool,
    pub best_effort: bool,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_count: Option<usize>,
    pub evaluation_limit: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub max_constraint_violation: f64,
    pub confidence: OptimizerConfidence,
    /// Whether this invocation supplied the parameters used in the emitted
    /// result. Attempts superseded by a better pass/refinement remain in the
    /// report but are not production-acceptance inputs.
    #[serde(default = "default_true")]
    pub selected_for_output: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub restart_history: Vec<OptimizerRestartEvidence>,
}

/// Serialisable summary of GD-Opt results for report plumbing (GD-4).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GroupDelayOptSummary {
    /// Optimisation band (Hz).
    pub band: (f64, f64),
    /// Channel names in the same order as the per-channel vectors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channel_names: Vec<String>,
    /// Per-channel delays applied (ms).
    pub per_channel_delay_ms: Vec<f64>,
    /// Per-channel polarity inversions.
    pub per_channel_polarity_inverted: Vec<bool>,
    /// Number of all-pass filters per channel.
    pub per_channel_ap_count: Vec<usize>,
    /// Sum GD RMS before optimisation (ms).
    pub sum_gd_pre_rms_ms: f64,
    /// Sum GD RMS after optimisation (ms).
    pub sum_gd_post_rms_ms: f64,
    /// Mean coherence in-band.
    pub mean_coherence: f64,
    /// Improvement in dB: 20*log10(pre/post).
    pub improvement_db: f64,
    /// Advisory outcome.
    pub advisory: String,
    /// Whether the reported GD controls were inserted into the exported DSP.
    #[serde(default)]
    pub applied: bool,
}

impl GroupDelayOptSummary {
    pub fn with_applied(mut self, applied: bool) -> Self {
        self.applied = applied;
        self
    }
}

/// Compact report for the excess-phase FIR generated by mixed-phase mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MixedPhaseCorrectionReport {
    /// Linear propagation delay removed from excess phase before FIR design.
    pub estimated_delay_ms: f64,
    /// Number of coefficients in the generated excess-phase FIR.
    pub fir_taps: usize,
    /// Minimum residual excess phase after delay removal.
    pub residual_excess_phase_min_deg: f64,
    /// Maximum residual excess phase after delay removal.
    pub residual_excess_phase_max_deg: f64,
    /// RMS residual excess phase after delay removal.
    pub residual_excess_phase_rms_deg: f64,
}

impl MixedPhaseCorrectionReport {
    pub fn from_residual(
        estimated_delay_ms: f64,
        fir_taps: usize,
        residual_excess_phase: &Array1<f64>,
    ) -> Self {
        let (minimum, maximum, sum_squares, count) = residual_excess_phase
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(
                (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0usize),
                |(minimum, maximum, sum_squares, count), value| {
                    (
                        minimum.min(value),
                        maximum.max(value),
                        sum_squares + value * value,
                        count + 1,
                    )
                },
            );
        let (minimum, maximum, rms) = if count == 0 {
            (0.0, 0.0, 0.0)
        } else {
            (minimum, maximum, (sum_squares / count as f64).sqrt())
        };
        Self {
            estimated_delay_ms,
            fir_taps,
            residual_excess_phase_min_deg: minimum,
            residual_excess_phase_max_deg: maximum,
            residual_excess_phase_rms_deg: rms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CtcReport {
    pub enabled: bool,
    pub source: String,
    pub artifact: String,
    pub speakers: Vec<String>,
    pub ears: Vec<String>,
    pub head_positions: usize,
    pub fir_taps: usize,
    pub latency_samples: usize,
    pub latency_ms: f64,
    pub max_filter_gain_db: f64,
    pub max_condition_number: f64,
    pub mean_reconstruction_error: f64,
    pub worst_position_error: f64,
    pub mean_crosstalk_residual_db: f64,
    pub max_electrical_sum_gain_db: f64,
    pub driver_headroom_limited: bool,
    pub room_eq_correction_applied: bool,
    pub room_eq_correction_channels: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivered_response: Option<CtcDeliveredResponseMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binaural_diagnostics: Option<CtcBinauralDiagnostics>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct CtcDeliveredResponseMetrics {
    pub mean_target_error: f64,
    pub worst_target_error: f64,
    pub mean_crosstalk_db: f64,
    pub worst_crosstalk_db: f64,
    pub mean_channel_balance_db: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct CtcBinauralDiagnostics {
    pub ild_error_db: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itd_error_proxy_us: Option<f64>,
    pub cue_deviation_score: f64,
    pub externalization_risk: String,
    pub imaging_risk: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hrtf_candidate_comparison: Option<CtcHrtfCandidateComparison>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct CtcHrtfCandidateComparison {
    pub candidate_count: usize,
    pub selected_source: String,
    pub advisory: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TemporalQualityEvidence {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_ringing_energy_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available_headroom_db: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QualityPartitionMetrics {
    pub curve_count: usize,
    pub pre_weighted_rms_median_db: f64,
    pub post_weighted_rms_median_db: f64,
    pub improvement_median_db: f64,
    /// Smallest per-position improvement. Negative values are regressions.
    #[serde(default)]
    pub worst_position_improvement_db: f64,
    pub pre_p95_abs_residual_db: f64,
    pub post_p95_abs_residual_db: f64,
    pub post_worst_abs_residual_db: f64,
    pub mean_normalized_seat_spread_db: f64,
    pub max_normalized_seat_spread_db: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_post_weighted_rms_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upper_post_weighted_rms_db: Option<f64>,
    /// Median RMS curvature of the residual below Schroeder frequency.
    ///
    /// This is measured in dB/octave² and distinguishes a response with
    /// narrow modal ripple from one with the same band RMS but a smooth tilt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_pre_modal_roughness_db_per_octave2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_post_modal_roughness_db_per_octave2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_modal_roughness_improvement_db_per_octave2: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AcousticQualityScorecard {
    pub training: QualityPartitionMetrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out: Option<QualityPartitionMetrics>,
    pub correction_rms_db: f64,
    pub max_boost_db: f64,
    pub max_cut_db: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub induced_group_delay_rms_ms: Option<f64>,
    pub temporal: TemporalQualityEvidence,
    pub evaluated_band_hz: [f64; 2],
    pub measurement_overlap_hz: [f64; 2],
    pub finite: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionAcceptancePolicy {
    RuntimeSafety,
    CorrectableFixture,
    AlreadyGoodFixture,
    PoorMeasurementFixture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionDecision {
    Accepted,
    RevertedStage,
    IdentityFallback,
}

pub const RUNTIME_ACCEPTANCE_POLICY_VERSION: &str = "1.0.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeOutputClass {
    LowLatencyIir,
    Fir,
    Hybrid,
}

/// Versioned limits applied to production RoomEQ output.
///
/// The output class changes only temporal limits. Spectral, spatial, boost,
/// headroom, and realization limits are invariant across filter classes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeAcceptancePolicy {
    pub version: String,
    pub output_class: RuntimeOutputClass,
    pub max_post_p95_abs_residual_db: f64,
    pub max_post_worst_abs_residual_db: f64,
    pub max_worst_position_regression_db: f64,
    pub max_boost_db: f64,
    pub min_available_headroom_db: f64,
    pub max_latency_ms: f64,
    pub max_pre_ringing_energy_db: f64,
    pub max_induced_group_delay_rms_ms: f64,
    pub max_realization_error_db: f64,
}

impl RuntimeAcceptancePolicy {
    pub fn for_output_class(output_class: RuntimeOutputClass) -> Self {
        let (max_latency_ms, max_induced_group_delay_rms_ms) = match output_class {
            RuntimeOutputClass::LowLatencyIir => (10.0, 5.0),
            RuntimeOutputClass::Fir => (250.0, 25.0),
            RuntimeOutputClass::Hybrid => (100.0, 10.0),
        };
        Self {
            version: RUNTIME_ACCEPTANCE_POLICY_VERSION.to_string(),
            output_class,
            max_post_p95_abs_residual_db: 6.0,
            max_post_worst_abs_residual_db: 12.0,
            max_worst_position_regression_db: 0.25,
            max_boost_db: 12.0,
            min_available_headroom_db: -12.0,
            max_latency_ms,
            max_pre_ringing_energy_db: -20.0,
            max_induced_group_delay_rms_ms,
            max_realization_error_db: 0.25,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != RUNTIME_ACCEPTANCE_POLICY_VERSION {
            return Err(format!(
                "unsupported runtime acceptance policy version '{}'; expected '{}'",
                self.version, RUNTIME_ACCEPTANCE_POLICY_VERSION
            ));
        }
        let finite = [
            self.max_post_p95_abs_residual_db,
            self.max_post_worst_abs_residual_db,
            self.max_worst_position_regression_db,
            self.max_boost_db,
            self.min_available_headroom_db,
            self.max_latency_ms,
            self.max_pre_ringing_energy_db,
            self.max_induced_group_delay_rms_ms,
            self.max_realization_error_db,
        ]
        .into_iter()
        .all(f64::is_finite);
        if !finite
            || self.max_post_p95_abs_residual_db < 0.0
            || self.max_post_worst_abs_residual_db < 0.0
            || self.max_worst_position_regression_db < 0.0
            || self.max_boost_db < 0.0
            || self.max_latency_ms < 0.0
            || self.max_induced_group_delay_rms_ms < 0.0
            || self.max_realization_error_db < 0.0
        {
            return Err("runtime acceptance policy contains invalid limits".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RealizationQualityEvidence {
    pub evaluated_channels: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_abs_error_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failed_channels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CorrectionMetricSummary {
    pub pre_target_weighted_rms_db: f64,
    pub post_target_weighted_rms_db: f64,
    pub improvement_db: f64,
    pub improvement_ratio: f64,
    pub post_p95_abs_residual_db: f64,
    pub post_worst_abs_residual_db: f64,
    pub correction_rms_db: f64,
    pub max_abs_correction_db: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CorrectionAcceptanceReport {
    pub policy: CorrectionAcceptancePolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_policy: Option<RuntimeAcceptancePolicy>,
    pub decision: CorrectionDecision,
    pub accepted: bool,
    pub metrics: CorrectionMetricSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub violations: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reverted_stages: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional multi-position quality evidence. Runtime callers without
    /// held-out measurements keep this absent for wire compatibility.
    pub acoustic_quality: Option<AcousticQualityScorecard>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realization_quality: Option<RealizationQualityEvidence>,
}
