//! Neutral serialized result and diagnostic contracts.

use ndarray::Array1;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Serializable summary of the audibility-veto adjudication against the
/// frozen full chain.  The engine retains removed filter coefficients for
/// rollback; reports/sidecars carry stable indices and measured deltas so the
/// decision remains inspectable without duplicating DSP state.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct VetoAdjudicationReport {
    pub f0_reference_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed_filter_indices: Vec<usize>,
    pub cumulative_loudness_delta_sones: f64,
    pub max_local_deviation_db: f64,
    pub enforced: bool,
}

#[cfg(test)]
mod pareto_dispatch_report_tests {
    use super::{
        ParetoCandidateEvidence, ParetoCrowdingDistance, ParetoDispatchReport,
        ParetoSelectionEvidence,
    };

    fn report() -> ParetoDispatchReport {
        ParetoDispatchReport {
            schema: "roomeq.pareto_dispatch/v1".to_string(),
            backend: "autoeq:nsga2".to_string(),
            submitted_count: 3,
            refused_source_indices: vec![1],
            candidates: vec![
                ParetoCandidateEvidence {
                    source_index: 0,
                    search_parameters: vec![0.2, 0.4],
                    search_objectives: vec![3.0, 1.5],
                    validated_parameters: vec![0.2, 0.4],
                    validated_objectives: vec![3.0, 1.5],
                    rank: Some(0),
                    crowding_distance: Some(ParetoCrowdingDistance::Finite(0.5)),
                    scalar_loss: Some(3.0),
                },
                ParetoCandidateEvidence {
                    source_index: 2,
                    search_parameters: vec![0.7, 0.1],
                    search_objectives: vec![1.0, 2.0],
                    validated_parameters: vec![0.7, 0.1],
                    validated_objectives: vec![1.0, 2.0],
                    rank: Some(0),
                    crowding_distance: Some(ParetoCrowdingDistance::Unbounded),
                    scalar_loss: Some(2.0),
                },
            ],
            selection: ParetoSelectionEvidence {
                rule: "normalized_compromise".to_string(),
                weights: vec![0.5, 0.5],
                ideal: vec![1.0, 1.5],
                nadir: vec![3.0, 2.0],
                selected_candidate_index: 1,
                selected_source_index: 2,
                scalar_baseline_rule: Some("configured_scalar:minimax".to_string()),
                scalar_best_source_index: Some(2),
                scalar_best_loss: Some(2.0),
            },
            returned_parameters: vec![0.7, 0.1],
            search_evaluations: Some(64),
            generations: Some(4),
        }
    }

    #[test]
    fn pareto_report_validates_and_roundtrips_tagged_unbounded_distance() {
        let report = report();
        report.validate().unwrap();
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains(r#""kind":"unbounded""#));
        assert!(!json.contains("Infinity"));
        let decoded: ParetoDispatchReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, report);
        decoded.validate().unwrap();
    }

    #[test]
    fn pareto_report_rejects_return_mismatch_and_non_finite_values() {
        let mut mismatched = report();
        mismatched.returned_parameters[0] = 0.8;
        assert!(
            mismatched
                .validate()
                .unwrap_err()
                .contains("does not match returned parameters")
        );

        let mut non_finite = report();
        non_finite.candidates[0].validated_objectives[0] = f64::NAN;
        assert!(
            non_finite
                .validate()
                .unwrap_err()
                .contains("malformed or non-finite")
        );
    }

    #[test]
    fn pareto_report_rejects_impossible_selection_policy_values() {
        let mut negative_crowding = report();
        negative_crowding.candidates[0].crowding_distance =
            Some(ParetoCrowdingDistance::Finite(-0.1));
        assert!(
            negative_crowding
                .validate()
                .unwrap_err()
                .contains("malformed or non-finite")
        );

        let mut negative_weight = report();
        negative_weight.selection.weights = vec![-0.5, 1.5];
        assert!(
            negative_weight
                .validate()
                .unwrap_err()
                .contains("selection frame")
        );

        let mut zero_weights = report();
        zero_weights.selection.weights = vec![0.0, 0.0];
        assert!(
            zero_weights
                .validate()
                .unwrap_err()
                .contains("selection frame")
        );

        let mut blank_rule = report();
        blank_rule.selection.rule = "  ".to_string();
        assert!(
            blank_rule
                .validate()
                .unwrap_err()
                .contains("selection frame")
        );

        let mut non_minimum_scalar = report();
        non_minimum_scalar.selection.scalar_best_source_index = Some(0);
        non_minimum_scalar.selection.scalar_best_loss = Some(3.0);
        assert!(
            non_minimum_scalar
                .validate()
                .unwrap_err()
                .contains("does not select the minimum")
        );
    }

    #[test]
    fn legacy_optimizer_evidence_defaults_to_no_pareto_report() {
        let value = serde_json::json!({
            "algorithm": "autoeq:cobyla",
            "termination": "converged",
            "converged": true,
            "best_effort": false,
            "status": "done",
            "evaluation_limit": 10,
            "max_constraint_violation": 0.0,
            "confidence": "high"
        });
        let evidence: super::OptimizerRunEvidence = serde_json::from_value(value).unwrap();
        assert!(evidence.pareto_report.is_none());
        assert!(
            serde_json::to_value(evidence)
                .unwrap()
                .get("pareto_report")
                .is_none()
        );
    }
}

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
    /// Physical pre-main energy (unmasked), dB relative to main peak energy.
    ///
    /// Peak, physical energy, and the masking-model prediction answer three
    /// different questions and must never be conflated.
    pub pre_energy_ratio_db: f64,
    /// Scalar penalty using the configured material profile and IR weights.
    pub penalty: f64,
    /// Analyzed impulse length in taps.
    pub taps: usize,
    /// Sample rate the impulse was analyzed at, in Hz.
    pub sample_rate_hz: f64,
    /// Masking-model programme profile assumed (`transient`/`mixed`/`sustained`).
    pub masking_profile: String,
    /// Pre-masking window retained from the analysis assumptions, in ms.
    pub pre_mask_ms: f64,
    /// Post-masking window retained from the analysis assumptions, in ms.
    pub post_mask_ms: f64,
    /// Audibility threshold retained from the analysis assumptions, in dB.
    pub audibility_threshold_db: f64,
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

/// Provenance for shipped EPA psychoacoustic scores.
///
/// EPA dimensions are predicted from frequency responses by a spectral
/// diagnostic model; they are not measured audibility and never substitute
/// for listening evidence. This block names the model and the calibration
/// it assumed so every score stays checkable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EpaProvenance {
    /// Scoring model identity.
    pub model: String,
    /// Always true: scores are model predictions, not measurements.
    pub predicted_not_measured: bool,
    /// Listening level the diagnostic assumed, in phon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub listening_level_phon: Option<f64>,
    /// Target sharpness the diagnostic assumed, in acum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_sharpness_acum: Option<f64>,
    /// Human-readable scope note.
    pub note: String,
}

/// Claim-level playback summary: what shipped, what benefit was
/// demonstrated, and which limits apply.
///
/// Rendered from (never a substitute for) the evidence blocks: every
/// number here repeats a value found elsewhere in the report. User-facing
/// reports and the mode matrix quote these headlines; auditors re-derive
/// them from the cited evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PlaybackSummary {
    /// Shipped outcome.
    pub outcome: RoomEqOutcome,
    /// Aggregate shape improvement in dB (echoes the acceptance metrics).
    pub improvement_db: f64,
    /// Training seats whose uncertainty-adjusted improvement clears zero.
    pub training_seats_improved: usize,
    /// Training seats evaluated.
    pub training_seats_total: usize,
    /// Lowest-benefit training seat as `input:index`, when evaluated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worst_seat: Option<String>,
    /// Modeled total playback latency in ms, when assessed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_latency_ms: Option<f64>,
    /// Available headroom in dB, when assessed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available_headroom_db: Option<f64>,
    /// Correction family actually present in the shipped graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realized_processing: Option<RealizedProcessing>,
    /// Requested-vs-realized divergence, when any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub processing_fallback: Option<String>,
    /// Applicable acceptance limits hit or in force, as `name=value`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub limits: Vec<String>,
    /// Templated human-readable claims, deterministic for a fixed report.
    pub headlines: Vec<String>,
}

/// Render the claim-level playback summary for an acceptance report.
///
/// Pure projection: seat counts come from uncertainty-adjusted training
/// bounds, latency/headroom echo the temporal block, and headlines are
/// fixed templates over the outcome. No new judgment is introduced here.
pub fn playback_summary(report: &CorrectionAcceptanceReport) -> PlaybackSummary {
    let seats: Vec<&FinalSeatEvaluation> = report
        .acoustic_quality
        .as_ref()
        .map(|score| {
            score
                .final_seats
                .iter()
                .filter(|seat| seat.partition == "training")
                .collect()
        })
        .unwrap_or_default();
    let improved = seats
        .iter()
        .filter(|seat| seat.improvement_lower_bound_db > 0.0)
        .count();
    let worst_seat = seats
        .iter()
        .min_by(|a, b| {
            a.improvement_lower_bound_db
                .total_cmp(&b.improvement_lower_bound_db)
        })
        .map(|seat| format!("{}:{}", seat.logical_input, seat.seat_index));
    let temporal = report
        .acoustic_quality
        .as_ref()
        .map(|score| &score.temporal);
    let total_latency_ms = temporal.and_then(|temporal| temporal.total_latency_ms);
    let available_headroom_db = temporal.and_then(|temporal| temporal.available_headroom_db);
    let mut limits = Vec::new();
    if let Some(policy) = report.runtime_policy.as_ref() {
        limits.push(format!("max_boost_db={:.1}", policy.max_boost_db));
        limits.push(format!("max_latency_ms={:.1}", policy.max_latency_ms));
        limits.push(format!(
            "max_pre_ringing_audible_db={:.1}",
            policy.max_pre_ringing_audible_db
        ));
    }
    let latency = total_latency_ms
        .map(|ms| format!("{ms:.1} ms"))
        .unwrap_or_else(|| "unassessed".to_string());
    let headroom = available_headroom_db
        .map(|db| format!("{db:.1} dB"))
        .unwrap_or_else(|| "unassessed".to_string());
    let mut violations = report.violations.clone();
    violations.sort();
    let headlines = match report.outcome {
        RoomEqOutcome::Accepted => vec![format!(
            "Accepted: aggregate improvement +{:.2} dB across {}/{} training seats; total latency {}; headroom {}.",
            report.metrics.improvement_db,
            improved,
            seats.len(),
            latency,
            headroom,
        )],
        RoomEqOutcome::Unchanged => vec![format!(
            "Unchanged: no candidate demonstrated benefit beyond uncertainty; protected baseline published ({}).",
            if violations.is_empty() {
                "no violations".to_string()
            } else {
                violations.join("; ")
            }
        )],
        RoomEqOutcome::Rejected => vec![format!(
            "Rejected: {}; protected baseline published.",
            if violations.is_empty() {
                "no stated violation".to_string()
            } else {
                violations.join("; ")
            }
        )],
        RoomEqOutcome::InsufficientEvidence => {
            vec!["Insufficient evidence: measurement gaps prevent an acceptance claim.".to_string()]
        }
    };
    PlaybackSummary {
        outcome: report.outcome,
        improvement_db: report.metrics.improvement_db,
        training_seats_improved: improved,
        training_seats_total: seats.len(),
        worst_seat,
        total_latency_ms,
        available_headroom_db,
        realized_processing: report.realized_processing,
        processing_fallback: report.processing_fallback.clone(),
        limits,
        headlines,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerTermination {
    Converged,
    EvaluationLimit,
    NonConverged,
    UserStopped,
    TimedOut,
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

/// Policy that supplied the actual EQ analysis normalization reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationReferencePolicy {
    /// A caller supplied a common reference; seats were not independently centered.
    ProvidedSharedReference,
    /// A target-relative reference outside a limited correction band was available.
    LimitedCorrectionTargetReference,
    /// Arithmetic mean of measured dB samples inside the active correction band.
    CorrectionBandArithmeticMean,
}

/// Actual analysis-level gain applied before EQ smoothing and objective construction.
///
/// Identities bind parsed numerical curves, not raw recordings. This gain is
/// not emitted playback gain, SPL calibration, or a claim about later conditioning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct InputNormalizationEvidence {
    pub input_curve_identity: String,
    pub normalized_curve_identity: String,
    /// Scalar added to input dB levels; subtracting the reference gives a negative gain.
    pub applied_gain_db: f64,
    pub reference_policy: NormalizationReferencePolicy,
    /// Active correction bounds, not necessarily the reference-estimation band.
    pub correction_band_hz: [f64; 2],
    /// Target used to derive a limited-band reference, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_target_identity: Option<String>,
}

/// Population whose actual objective curves were normalized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationPopulation {
    AlignedMeasurements,
    RirPrototype,
    BootstrapResamples,
    BootstrapOfRirPrototype,
}

/// Normalization records in the same order as the prepared multi-objective bank.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MultiInputNormalizationEvidence {
    pub population: NormalizationPopulation,
    pub objectives: Vec<InputNormalizationEvidence>,
}

/// Structured evidence for one optimizer invocation.
///
/// Backends retain their historical tuple API, but callers should use this
/// type for production acceptance. In particular, an `Ok` tuple containing
/// "not converged" is classified as best-effort rather than success.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OptimizerRunEvidence {
    /// Per-objective conditioning; objective indices are not authenticated seat identities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_input_normalization: Option<MultiInputNormalizationEvidence>,
    /// Absent means conditioning was not recorded by this producer, not zero gain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_normalization: Option<InputNormalizationEvidence>,
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
    /// Pareto selection evidence from this invocation, when the backend emitted it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pareto_report: Option<ParetoDispatchReport>,
}

/// A JSON-safe crowding distance from a Pareto backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ParetoCrowdingDistance {
    /// A finite crowding distance.
    Finite(f64),
    /// An unbounded boundary-point distance, represented without non-finite JSON numbers.
    Unbounded,
}

/// One source front member and its post-envelope validation result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ParetoCandidateEvidence {
    /// Position in the optimizer's submitted front.
    pub source_index: usize,
    /// Parameters nominated by the search backend.
    pub search_parameters: Vec<f64>,
    /// Search-stage objectives attached to the nominated member.
    pub search_objectives: Vec<f64>,
    /// Parameters retained after shared envelope validation and repair.
    pub validated_parameters: Vec<f64>,
    /// Validation-stage objectives evaluated at `validated_parameters`.
    pub validated_objectives: Vec<f64>,
    /// Backend non-dominated rank, when it reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    /// Backend crowding distance, when it reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crowding_distance: Option<ParetoCrowdingDistance>,
    /// Configured scalar loss already computed during this invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_loss: Option<f64>,
}

/// Policy and winner identity for one validated Pareto front.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ParetoSelectionEvidence {
    /// Selection rule, such as `normalized_compromise`.
    pub rule: String,
    /// Objective weights used by the selection rule.
    pub weights: Vec<f64>,
    /// Per-objective ideal point.
    pub ideal: Vec<f64>,
    /// Per-objective nadir point.
    pub nadir: Vec<f64>,
    /// Index into [`ParetoDispatchReport::candidates`] of the selected member.
    pub selected_candidate_index: usize,
    /// Original submitted-front index of the selected member.
    pub selected_source_index: usize,
    /// Scalar baseline policy, when the backend computed one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_baseline_rule: Option<String>,
    /// Original submitted-front index of the scalar-best member, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_best_source_index: Option<usize>,
    /// Scalar-best loss, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_best_loss: Option<f64>,
}

/// Invocation-local, post-validation evidence for a Pareto optimizer result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ParetoDispatchReport {
    /// Schema marker: `roomeq.pareto_dispatch/v1`.
    pub schema: String,
    /// Backend that nominated the source front.
    pub backend: String,
    /// Number of members submitted for shared validation.
    pub submitted_count: usize,
    /// Submitted-front indices refused by shared validation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refused_source_indices: Vec<usize>,
    /// Feasible members, in submitted-front order.
    pub candidates: Vec<ParetoCandidateEvidence>,
    /// Selection policy and winner identity.
    pub selection: ParetoSelectionEvidence,
    /// Exact parameters returned by the optimizer invocation.
    pub returned_parameters: Vec<f64>,
    /// Search objective evaluations, when reported by the backend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_evaluations: Option<usize>,
    /// Search generations completed, when reported by the backend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generations: Option<usize>,
}

impl ParetoDispatchReport {
    /// Check that the report is internally consistent and contains finite values.
    ///
    /// This method validates deserialized reports as well as reports created by
    /// an optimizer. It does not authenticate the optimizer or prove physical quality.
    ///
    /// # Errors
    ///
    /// Returns an explanation when identities, dimensions, selection, or numbers conflict.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != "roomeq.pareto_dispatch/v1" || self.backend.trim().is_empty() {
            return Err("Pareto report schema or backend identity is invalid".to_string());
        }
        let inventory_count = self
            .candidates
            .len()
            .checked_add(self.refused_source_indices.len())
            .ok_or_else(|| "Pareto report member inventory overflows usize".to_string())?;
        if self.submitted_count != inventory_count {
            return Err(
                "Pareto report submitted count does not match its member inventory".to_string(),
            );
        }
        let objective_count = self
            .candidates
            .first()
            .map(|candidate| candidate.validated_objectives.len())
            .filter(|count| *count > 0)
            .ok_or_else(|| "Pareto report has no validated candidates".to_string())?;
        let parameter_count = self.returned_parameters.len();
        if parameter_count == 0
            || !self
                .returned_parameters
                .iter()
                .all(|value| value.is_finite())
            || self.selection.weights.len() != objective_count
            || self.selection.ideal.len() != objective_count
            || self.selection.nadir.len() != objective_count
        {
            return Err("Pareto report has invalid selection dimensions or values".to_string());
        }
        let finite = |values: &[f64]| values.iter().all(|value| value.is_finite());
        if self.selection.rule.trim().is_empty()
            || !finite(&self.selection.weights)
            || self.selection.weights.iter().any(|weight| *weight < 0.0)
            || !self.selection.weights.iter().any(|weight| *weight > 0.0)
            || !finite(&self.selection.ideal)
            || !finite(&self.selection.nadir)
            || self
                .selection
                .ideal
                .iter()
                .zip(&self.selection.nadir)
                .any(|(ideal, nadir)| ideal > nadir)
        {
            return Err("Pareto report selection frame is invalid".to_string());
        }
        let mut source_indices = vec![false; self.submitted_count];
        for candidate in &self.candidates {
            if candidate.source_index >= self.submitted_count
                || source_indices[candidate.source_index]
                || candidate.search_parameters.len() != parameter_count
                || candidate.validated_parameters.len() != parameter_count
                || candidate.search_objectives.len() != objective_count
                || candidate.validated_objectives.len() != objective_count
                || !finite(&candidate.search_parameters)
                || !finite(&candidate.validated_parameters)
                || !finite(&candidate.search_objectives)
                || !finite(&candidate.validated_objectives)
                || candidate.scalar_loss.is_some_and(|loss| !loss.is_finite())
                || matches!(candidate.crowding_distance, Some(ParetoCrowdingDistance::Finite(value)) if !value.is_finite() || value < 0.0)
            {
                return Err("Pareto report candidate is malformed or non-finite".to_string());
            }
            source_indices[candidate.source_index] = true;
        }
        for &index in &self.refused_source_indices {
            if index >= self.submitted_count || source_indices[index] {
                return Err("Pareto report has a duplicate or invalid refused index".to_string());
            }
            source_indices[index] = true;
        }
        if source_indices.iter().any(|seen| !seen) {
            return Err("Pareto report omits a submitted-front index".to_string());
        }
        let selected = self
            .candidates
            .get(self.selection.selected_candidate_index)
            .ok_or_else(|| "Pareto report selected candidate index is out of range".to_string())?;
        if selected.source_index != self.selection.selected_source_index
            || selected.validated_parameters != self.returned_parameters
        {
            return Err("Pareto report selection does not match returned parameters".to_string());
        }
        match (
            self.selection.scalar_baseline_rule.as_ref(),
            self.selection.scalar_best_source_index,
            self.selection.scalar_best_loss,
        ) {
            (None, None, None) => {}
            (Some(rule), Some(source_index), Some(loss))
                if !rule.trim().is_empty() && loss.is_finite() =>
            {
                let Some(best_candidate) = self
                    .candidates
                    .iter()
                    .find(|candidate| candidate.source_index == source_index)
                else {
                    return Err(
                        "Pareto report scalar baseline source is not a candidate".to_string()
                    );
                };
                if best_candidate.scalar_loss != Some(loss) {
                    return Err(
                        "Pareto report scalar baseline source and loss disagree".to_string()
                    );
                }
                if rule.starts_with("configured_scalar:") {
                    let scalar_losses = self
                        .candidates
                        .iter()
                        .map(|candidate| candidate.scalar_loss)
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| {
                            "configured scalar baseline omits candidate scalar losses".to_string()
                        })?;
                    let minimum = scalar_losses
                        .iter()
                        .copied()
                        .min_by(f64::total_cmp)
                        .ok_or_else(|| {
                            "configured scalar baseline has no candidate scores".to_string()
                        })?;
                    if loss != minimum {
                        return Err(
                            "configured scalar baseline does not select the minimum loss"
                                .to_string(),
                        );
                    }
                }
            }
            _ => return Err("Pareto report scalar baseline evidence is inconsistent".to_string()),
        }
        Ok(())
    }
}

impl OptimizerRunEvidence {
    /// Whether the reported objective and recorded bound checks retain a valid candidate.
    ///
    /// This does not imply convergence, deployment acceptance, or that a stop
    /// request did not occur.
    pub fn has_valid_candidate(&self) -> bool {
        self.objective.is_some_and(f64::is_finite)
            && self.max_constraint_violation.is_finite()
            && self.max_constraint_violation <= 1e-9
    }
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
    ///
    /// Together with the sample rate this describes the finite record
    /// (duration and FFT bin spacing, e.g. 4096 taps at 48 kHz span
    /// 85.33 ms with 11.72 Hz spacing). That spacing is finite-record
    /// sampling, not a lower correction-frequency limit or proof of
    /// resolving power; zero padding interpolates but adds no information.
    pub fir_taps: usize,
    /// Causal FIR centering delay in milliseconds, when the realization is known.
    ///
    /// This is distinct from acoustic propagation and frequency-dependent
    /// excess group delay. It excludes backend buffering and other DSP stages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causal_center_delay_ms: Option<f64>,
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
            causal_center_delay_ms: None,
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
    /// Availability of time-referenced binaural/IR evidence. This records an
    /// evidence boundary only; it is not a claim that a precedence effect is
    /// audible or preferred for a listener.
    #[serde(default)]
    pub precedence_evidence: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hrtf_candidate_comparison: Option<CtcHrtfCandidateComparison>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct CtcHrtfCandidateComparison {
    pub candidate_count: usize,
    pub selected_source: String,
    pub advisory: String,
}

/// Whether evaluated inputs share one verified timing reference.
///
/// Phase arrays only prove values exist. Coherent summation additionally
/// needs every contributing capture to share one calibrated time zero.
/// Variants keep unknown inputs visibly limited instead of passing
/// finite-but-unsynchronized phase as verified evidence.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CoherentTimingEvidence {
    /// Coherent timing was not assessed at this boundary.
    #[default]
    Unassessed,
    /// Every evaluated input shares the cited reference across stationary captures.
    Verified {
        /// Shared timing-reference identity cited by all evaluated inputs.
        reference_id: String,
    },
    /// Verification failed; coherent claims stay unsupported.
    Refused {
        /// Precise reason verification failed.
        reason: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TemporalQualityEvidence {
    /// Worst masking-derived audible pre-ringing across channels, in dB
    /// relative to each FIR's main impulse peak.
    ///
    /// This is the maximum of the per-channel `pre_ringing_audible_db`
    /// values, not a physical pre/main energy ratio (that quantity is
    /// `TemporalIrMaskingMetrics.pre_energy_ratio_db`). Runtime acceptance compares it
    /// against `RuntimeAcceptancePolicy.max_pre_ringing_audible_db`
    /// (−20 dB), which is distinct from the −30 dB per-channel FIR design
    /// threshold: an aggregate between the two is expected, not a
    /// demonstrated violation. Neither value proves audibility without
    /// calibrated stimulus and listener context. IIR-only chains report
    /// the `NO_FIR_PRE_RINGING_FLOOR_DB` sentinel (−300 dB), a
    /// report floor rather than a measured room-noise floor.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "pre_ringing_energy_db"
    )]
    pub pre_ringing_audible_db: Option<f64>,
    /// FIR design delay: the maximum main-impulse time across channels.
    ///
    /// Kept under its legacy name; WP6 splits the playback total below.
    /// `None` means unresolved FIR evidence, never zero latency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    /// Common alignment delay every branch rose by at causal delay
    /// compilation (`delay_compile_causal`), in ms. Zero when compilation
    /// was a no-op; absent when the compile boundary never stamped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment_delay_ms: Option<f64>,
    /// Total modeled playback latency (design + alignment) in ms.
    ///
    /// Excludes host/block buffering and converter delay, which the engine
    /// cannot observe; the host adds those. Absent unless both summands
    /// are assessed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_latency_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available_headroom_db: Option<f64>,
    /// True only when both pre and post responses carried measured phase.
    /// A missing phase is an evidence limitation, never a zero-phase claim.
    /// Presence alone never verifies coherent timing; see `coherent_timing`.
    #[serde(default)]
    pub phase_evidence_available: bool,
    /// True when temporal measurements required by the output class exist.
    /// For an IIR-only chain this is true because no FIR precursor claim is
    /// being made; FIR and hybrid paths must provide measured masking data.
    #[serde(default)]
    pub temporal_evidence_available: bool,
    /// Shared-timing verification across evaluated inputs.
    /// Unassessed where replay did not evaluate; replay upgrades the verdict.
    #[serde(default)]
    pub coherent_timing: CoherentTimingEvidence,
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
    /// Median band RMS of the pre-correction residual above Schroeder frequency.
    ///
    /// Paired with `upper_post_weighted_rms_db`, this lets the quality gate
    /// enforce "do no harm" on timbre while the modal band is corrected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upper_pre_weighted_rms_db: Option<f64>,
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

/// Calibrated upper bound on an unmeasured acoustic transfer, before DSP.
/// An explicit declaration is a flat cap, not a fitted extrapolation. The
/// subwoofer stopband inference may instead record a measured rolloff, in
/// which case `max_spl_db` is the bound level at `band_hz[0]` and the bound
/// declines at `rolloff_db_per_oct` above it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct UpperBandAcousticBound {
    /// `training` or `held_out`, matching the immutable capture partition.
    pub partition: String,
    pub seat_index: usize,
    /// Bound must overlap the measurement endpoint and cover the assessed band.
    pub band_hz: [f64; 2],
    /// Same input reference and SPL calibration as the corresponding capture.
    /// Flat cap everywhere when `rolloff_db_per_oct` is absent, level at
    /// `band_hz[0]` when a rolloff is present.
    pub max_spl_db: f64,
    /// Measured decline in dB per octave above `band_hz[0]`. Absent means a
    /// flat cap. Only a non-positive (falling or flat) slope is admissible;
    /// a rising tail can never bound unmeasured output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rolloff_db_per_oct: Option<f64>,
    /// Stable reference to the calibration, specification, or measurement proof.
    pub evidence_id: String,
}

impl UpperBandAcousticBound {
    /// Bound level at one frequency. Flat below and at the band edge so the
    /// overlap with measured support stays a cap; declining above it only
    /// when a measured rolloff was recorded.
    pub fn level_at_hz(&self, frequency_hz: f64) -> f64 {
        match self.rolloff_db_per_oct {
            Some(slope)
                if slope.is_finite()
                    && slope <= 0.0
                    && self.band_hz[0] > 0.0
                    && frequency_hz > self.band_hz[0] =>
            {
                self.max_spl_db + slope * (frequency_hz / self.band_hz[0]).log2()
            }
            _ => self.max_spl_db,
        }
    }
}

/// Bound retained when a physical branch is omitted outside measured support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SummationSupportEvidence {
    pub physical_output: String,
    pub measured_band_hz: [f64; 2],
    pub omitted_band_hz: [f64; 2],
    pub acoustic_bound: UpperBandAcousticBound,
    /// Worst ratio of all omitted amplitude bounds to the retained complex sum.
    pub max_sum_omitted_amplitude_ratio: f64,
    /// Worst -20 log10(1-rho), including actual baseline/candidate DSP.
    pub max_magnitude_uncertainty_db: f64,
    /// Worst asin(rho) phase uncertainty of the retained nominal complex sum.
    pub max_phase_uncertainty_deg: f64,
}

/// One independently replayed final-chain listening-position assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FinalSeatEvaluation {
    pub partition: String,
    pub logical_input: String,
    /// Stable index in the input measurement array; never a channel aggregate.
    pub seat_index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seat_label: Option<String>,
    pub physical_outputs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pre_summation_support: Vec<SummationSupportEvidence>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub post_summation_support: Vec<SummationSupportEvidence>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unassessed_bands_hz: Vec<[f64; 2]>,
    /// Actual supported evaluation band for this input/position, not the
    /// optimizer's requested full band.
    pub evaluated_band_hz: [f64; 2],
    pub pre_weighted_rms_db: f64,
    pub post_weighted_rms_db: f64,
    pub improvement_db: f64,
    /// Improvement after subtracting pre/post summation uncertainty budgets.
    #[serde(default)]
    pub improvement_lower_bound_db: f64,
    /// Per-band raw improvements, when the seat band supports them.
    ///
    /// WP5 concealment evidence: a broadband gain must not hide a band
    /// regression. Raw (non-uncertainty-adjusted) by construction; the
    /// broadband lower bound stays the enforced quantity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_improvement_db: Option<BandImprovement>,
}

/// Raw improvement split into modal bass, crossover overlap, and upper band.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BandImprovement {
    /// Schroeder frequency splitting bass from upper, in Hz.
    pub schroeder_hz: f64,
    /// Modal-bass band actually evaluated, in Hz.
    pub bass_band_hz: [f64; 2],
    /// Raw improvement over the bass band; absent with fewer than two bins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_improvement_db: Option<f64>,
    /// Crossover-overlap band (XO ± one octave, clamped); absent when the
    /// input has no routed crossover frequency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossover_band_hz: Option<[f64; 2]>,
    /// Raw improvement over the crossover band; absent when unassessed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossover_improvement_db: Option<f64>,
    /// Remaining usable upper band actually evaluated, in Hz.
    pub upper_band_hz: [f64; 2],
    /// Raw improvement over the upper band; absent with fewer than two bins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upper_improvement_db: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OutputLossBand {
    /// Endpoints of consecutive evaluation samples exceeding the threshold.
    /// This does not establish a continuous-frequency bound between samples.
    pub sampled_band_hz: [f64; 2],
    pub sample_count: usize,
    pub worst_frequency_hz: f64,
    pub worst_loss_db: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct UsefulOutputEvidence {
    /// Present when evaluated by logical-input final playback replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_input: Option<String>,
    pub partition: String,
    pub seat_index: usize,
    /// Authorized broadband change, independent of fitted shape normalization.
    pub permitted_gain_db: f64,
    pub evaluated_band_hz: [f64; 2],
    pub mean_level_change_db: f64,
    /// Log-frequency-weighted RMS of losses beyond the authorized gain.
    pub unexplained_loss_rms_db: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worst_unexplained_loss_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worst_loss_frequency_hz: Option<f64>,
    /// Diagnostic threshold only; aggregate runtime policy remains separate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_band_threshold_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loss_bands: Vec<OutputLossBand>,
    /// Separate bass assessment so a broad main band cannot dilute lost bass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_evaluated_band_hz: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_unexplained_loss_rms_db: Option<f64>,
    /// Lowest reliable octave of the evaluated band, when at least two
    /// samples cover it. WP4 output distinction: loss here is lost low-end
    /// extension, while `worst_unexplained_loss_db`/`loss_bands` describe
    /// (usually narrow, benign) peak attenuation elsewhere.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extension_band_hz: Option<[f64; 2]>,
    /// Log-frequency-weighted mean unexplained loss over
    /// `extension_band_hz`. Stays near zero when cuts are peak-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extension_loss_db: Option<f64>,
    /// Peak absolute demand shift: peak post level minus peak pre level
    /// over the evaluated band. The acoustic drive proxy: how much less
    /// (or more) peak output the system is asked to produce. Electrical
    /// required-vs-available headroom stays in the bus simulation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_demand_change_db: Option<f64>,
    /// Calibrated target deficit, without fitting away broadband level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_shortfall_rms_db: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AcousticQualityScorecard {
    /// Unnormalized output evidence; absent in legacy scorecards, never inferred
    /// from level-independent shape metrics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub useful_output: Vec<UsefulOutputEvidence>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub final_seats: Vec<FinalSeatEvaluation>,
    pub training: QualityPartitionMetrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out: Option<QualityPartitionMetrics>,
    pub correction_rms_db: f64,
    /// Acoustic response-ratio improvement in dB (post minus pre level).
    ///
    /// This is measured level change at the seat, not filter gain: coherent
    /// summation across branches can move it without any filter boosting.
    /// The enforced backstop compares this against the runtime policy; see
    /// `max_electrical_boost_db` for the filter-side upper bound.
    pub max_boost_db: f64,
    /// Upper bound on electrical filter boost in dB, when evaluable.
    ///
    /// Sums positive IIR-element gains (EQ bands, shelves, gain trims, route
    /// gains) per branch; cascade overlap inflates it above the realized
    /// peak. Branches with convolution content are excluded (sidecar taps
    /// are unevaluated), and the field is absent when no branch is fully
    /// evaluable. Advisory: enforcement stays on the acoustic backstop
    /// until matrix evidence justifies switching the checked quantity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_electrical_boost_db: Option<f64>,
    pub max_cut_db: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub induced_group_delay_rms_ms: Option<f64>,
    pub temporal: TemporalQualityEvidence,
    /// Explicit active-correction support, when a policy was selected.
    /// `evaluated_band_hz` remains the fixed observation band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_band_hz: Option<[f64; 2]>,
    pub evaluated_band_hz: [f64; 2],
    /// Common measured band, absent when independently assessed inputs have
    /// disjoint support. Each final-seat record retains its assessed band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement_overlap_hz: Option<[f64; 2]>,
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
    Rejected,
    RevertedStage,
    IdentityFallback,
}

/// User-facing outcome of comparing a realized correction with its frozen
/// original baseline. This is deliberately separate from the implementation
/// decision so an identity fallback can be reported as unchanged and a
/// phase/seat evidence gap can be reported as insufficient evidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RoomEqOutcome {
    Accepted,
    Unchanged,
    Rejected,
    #[default]
    InsufficientEvidence,
}

/// Correction family actually present in the shipped graph.
///
/// Derived from serialized channel and per-driver plugins, never from the
/// requested mode label: a mixed-phase run whose excess-phase FIR never
/// materialized reports [`RealizedProcessing::IirOnly`], and a fully
/// reverted run reports [`RealizedProcessing::Identity`]. `eq`,
/// `warped_biquad`, and `kautz_filter` vote IIR; `convolution` votes FIR.
/// Gain, delay, and crossover plugins are alignment/routing: a graph with
/// only those reports [`RealizedProcessing::AlignmentOnly`], which still
/// corrects level and timing (audible in joint summation) but shapes no
/// frequency response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RealizedProcessing {
    /// No correction or alignment plugin on any channel or driver.
    #[default]
    Identity,
    /// Only gain/delay/crossover plugins: level and timing alignment
    /// without frequency-response shaping.
    AlignmentOnly,
    /// IIR correction plugins but no `convolution` anywhere.
    IirOnly,
    /// `convolution` plugins but no IIR correction anywhere.
    FirOnly,
    /// Both IIR correction and `convolution` present (band-split FIR or
    /// excess-phase correction).
    Hybrid,
}

/// Assess the realized correction family from shipped channel plugins.
///
/// Pure function over the serialized graph: channel-level and per-driver
/// IIR/FIR plugins vote their family (see [`RealizedProcessing`]).
pub fn assess_realized_processing(
    channels: &std::collections::HashMap<String, crate::output::ChannelDspChain>,
) -> RealizedProcessing {
    fn votes(plugins: &[crate::output::PluginConfigWrapper]) -> (bool, bool, bool) {
        let iir = plugins.iter().any(|p| {
            matches!(
                p.plugin_type.as_str(),
                "eq" | "warped_biquad" | "kautz_filter"
            )
        });
        let fir = plugins.iter().any(|p| p.plugin_type == "convolution");
        let alignment = plugins
            .iter()
            .any(|p| matches!(p.plugin_type.as_str(), "gain" | "delay" | "crossover"));
        (iir, fir, alignment)
    }
    let mut has_iir = false;
    let mut has_fir = false;
    let mut has_alignment = false;
    for chain in channels.values() {
        let (iir, fir, alignment) = votes(&chain.plugins);
        has_iir |= iir;
        has_fir |= fir;
        has_alignment |= alignment;
        if let Some(drivers) = &chain.drivers {
            for driver in drivers {
                let (iir, fir, alignment) = votes(&driver.plugins);
                has_iir |= iir;
                has_fir |= fir;
                has_alignment |= alignment;
            }
        }
    }
    match (has_iir, has_fir, has_alignment) {
        (true, true, _) => RealizedProcessing::Hybrid,
        (true, false, _) => RealizedProcessing::IirOnly,
        (false, true, _) => RealizedProcessing::FirOnly,
        (false, false, true) => RealizedProcessing::AlignmentOnly,
        (false, false, false) => RealizedProcessing::Identity,
    }
}

/// Explain a requested-mode versus realized-processing divergence.
///
/// Returns `None` when the shipped graph matches the requested family
/// (`LowLatency`/`WarpedIir`/`KautzModal`→IIR-only, `PhaseLinear`→FIR-only,
/// `Hybrid`/`MixedPhase`→hybrid IIR+FIR). Otherwise returns a stable
/// machine-readable reason
/// naming both sides, e.g. `mixed_phase_requested_iir_only_realized`.
/// A `None` requested mode (unknown configuration) never diverges.
pub fn processing_fallback_reason(
    requested: Option<&crate::config::ProcessingMode>,
    realized: &RealizedProcessing,
) -> Option<String> {
    let requested = requested?;
    let expected = match requested {
        crate::config::ProcessingMode::LowLatency
        | crate::config::ProcessingMode::WarpedIir
        | crate::config::ProcessingMode::KautzModal => RealizedProcessing::IirOnly,
        crate::config::ProcessingMode::PhaseLinear => RealizedProcessing::FirOnly,
        crate::config::ProcessingMode::Hybrid | crate::config::ProcessingMode::MixedPhase => {
            RealizedProcessing::Hybrid
        }
    };
    if *realized == expected {
        return None;
    }
    let requested_name = match requested {
        crate::config::ProcessingMode::LowLatency => "low_latency",
        crate::config::ProcessingMode::PhaseLinear => "phase_linear",
        crate::config::ProcessingMode::Hybrid => "hybrid",
        crate::config::ProcessingMode::MixedPhase => "mixed_phase",
        crate::config::ProcessingMode::WarpedIir => "warped_iir",
        crate::config::ProcessingMode::KautzModal => "kautz_modal",
    };
    let realized_name = match realized {
        RealizedProcessing::Identity => "identity",
        RealizedProcessing::AlignmentOnly => "alignment_only",
        RealizedProcessing::IirOnly => "iir_only",
        RealizedProcessing::FirOnly => "fir_only",
        RealizedProcessing::Hybrid => "hybrid",
    };
    Some(format!(
        "{requested_name}_requested_{realized_name}_realized"
    ))
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
    /// Ceiling for the gain backstop in dB.
    ///
    /// Compares against the electrical filter-gain bound
    /// (`max_electrical_boost_db`), not the acoustic response ratio:
    /// improved interference near a baseline null can legitimately move
    /// the ratio without any filter boosting. The acoustic
    /// `max_boost_db` stays reported alongside; an acoustic-only excess
    /// is recorded as an observation, while an unevaluable electrical
    /// bound fails closed to a violation.
    pub max_boost_db: f64,
    pub min_available_headroom_db: f64,
    pub max_latency_ms: f64,
    /// Runtime ceiling for the aggregate masking-derived audible
    /// pre-ringing (`TemporalQualityEvidence.pre_ringing_audible_db`).
    /// Distinct from the −30 dB per-channel FIR design threshold, which
    /// applies at design time, not to this aggregate.
    #[serde(alias = "max_pre_ringing_energy_db")]
    pub max_pre_ringing_audible_db: f64,
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
            max_pre_ringing_audible_db: -20.0,
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
            self.max_pre_ringing_audible_db,
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
    /// Versioned frequency measure used by target-weighted RMS fields.
    pub auditory_frequency_measure: String,
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
    /// Stable four-state summary kept alongside the detailed decision.
    #[serde(default)]
    pub outcome: RoomEqOutcome,
    pub metrics: CorrectionMetricSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub violations: Vec<String>,
    /// Correction family actually present in the shipped graph, derived
    /// from serialized plugins rather than the requested mode label.
    /// Set at graph conversion; reports built before the final graph
    /// exists leave this absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realized_processing: Option<RealizedProcessing>,
    /// Requested-mode versus realized-processing divergence, when any.
    /// Names both sides (see [`processing_fallback_reason`]); `None`
    /// means the shipped graph matches the requested family. A fallback
    /// is audit labeling, not a verdict: it never changes the outcome by
    /// itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub processing_fallback: Option<String>,
    /// Non-violating limit assessments recorded for audit agreement.
    ///
    /// Entries explain why an apparent limit/value divergence is not a
    /// violation under the defined quantity semantics (for example, an
    /// acoustic response ratio above the gain backstop while the
    /// electrical bound stays within it). Observations never change the
    /// outcome; only `violations` do.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observations: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reverted_stages: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional multi-position quality evidence. Runtime callers without
    /// held-out measurements keep this absent for wire compatibility.
    pub acoustic_quality: Option<AcousticQualityScorecard>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realization_quality: Option<RealizationQualityEvidence>,
}

impl CorrectionAcceptanceReport {
    /// Derive the stable four-state outcome vocabulary from the detailed
    /// acceptance record. An evidence limitation always wins over a generic
    /// rejection, so callers cannot mistake an unmeasured phase claim for a
    /// proven bad correction.
    pub fn derived_outcome(&self) -> RoomEqOutcome {
        if self.violations.iter().any(|violation| {
            violation.contains("insufficient")
                || violation.contains("evidence_missing")
                || violation.contains("evidence_unavailable")
        }) {
            return RoomEqOutcome::InsufficientEvidence;
        }
        if self.accepted && self.decision == CorrectionDecision::Accepted {
            RoomEqOutcome::Accepted
        } else if self.decision == CorrectionDecision::IdentityFallback {
            RoomEqOutcome::Unchanged
        } else {
            RoomEqOutcome::Rejected
        }
    }

    /// Refresh the serialized outcome after a caller changes violations or
    /// the detailed acceptance decision.
    pub fn refresh_outcome(&mut self) {
        self.outcome = self.derived_outcome();
    }
}

#[cfg(test)]
mod outcome_tests {
    use super::*;

    fn report(
        decision: CorrectionDecision,
        accepted: bool,
        violations: Vec<&str>,
    ) -> CorrectionAcceptanceReport {
        CorrectionAcceptanceReport {
            policy: CorrectionAcceptancePolicy::RuntimeSafety,
            runtime_policy: None,
            decision,
            accepted,
            outcome: RoomEqOutcome::Unchanged,
            metrics: CorrectionMetricSummary {
                auditory_frequency_measure: "erb_rate".into(),
                pre_target_weighted_rms_db: 1.0,
                post_target_weighted_rms_db: 1.0,
                improvement_db: 0.0,
                improvement_ratio: 0.0,
                post_p95_abs_residual_db: 1.0,
                post_worst_abs_residual_db: 1.0,
                correction_rms_db: 0.0,
                max_abs_correction_db: 0.0,
            },
            violations: violations.into_iter().map(str::to_string).collect(),
            realized_processing: None,
            processing_fallback: None,
            observations: Vec::new(),
            reverted_stages: Vec::new(),
            acoustic_quality: None,
            realization_quality: None,
        }
    }

    #[test]
    fn outcome_distinguishes_acceptance_identity_rejection_and_evidence_gap() {
        assert_eq!(
            report(CorrectionDecision::Accepted, true, vec![]).derived_outcome(),
            RoomEqOutcome::Accepted
        );
        assert_eq!(
            report(CorrectionDecision::IdentityFallback, false, vec![]).derived_outcome(),
            RoomEqOutcome::Unchanged
        );
        assert_eq!(
            report(
                CorrectionDecision::Rejected,
                false,
                vec!["no_safe_candidate"]
            )
            .derived_outcome(),
            RoomEqOutcome::Rejected
        );
        assert_eq!(
            report(
                CorrectionDecision::Rejected,
                false,
                vec!["final_seat_phase_evidence_insufficient"],
            )
            .derived_outcome(),
            RoomEqOutcome::InsufficientEvidence
        );
    }

    #[test]
    fn outcome_is_serialized_and_legacy_reports_default_to_insufficient_evidence() {
        let mut report = report(CorrectionDecision::Accepted, true, vec![]);
        report.refresh_outcome();
        let value = serde_json::to_value(&report).expect("serialize acceptance report");
        assert_eq!(value["outcome"], serde_json::json!("accepted"));

        let mut legacy = value;
        legacy
            .as_object_mut()
            .expect("report object")
            .remove("outcome");
        let decoded: CorrectionAcceptanceReport =
            serde_json::from_value(legacy).expect("legacy report remains readable");
        assert_eq!(decoded.outcome, RoomEqOutcome::InsufficientEvidence);
        assert_eq!(decoded.derived_outcome(), RoomEqOutcome::Accepted);
    }

    fn chain_with(plugin_types: &[&str]) -> crate::output::ChannelDspChain {
        let plugins: Vec<serde_json::Value> = plugin_types
            .iter()
            .map(|kind| serde_json::json!({"plugin_type": kind, "parameters": {}}))
            .collect();
        serde_json::from_value(serde_json::json!({
            "channel": "L",
            "plugins": plugins,
            "drivers": null,
            "initial_curve": null,
            "final_curve": null,
            "eq_response": null,
            "pre_ir": null,
            "post_ir": null,
        }))
        .expect("test chain deserializes")
    }

    #[test]
    fn realized_processing_votes_iir_and_fir_plugins() {
        use std::collections::HashMap;
        let graph = |kinds: &[&str]| {
            let mut channels = HashMap::new();
            channels.insert("L".to_string(), chain_with(kinds));
            channels
        };
        assert_eq!(
            super::assess_realized_processing(&graph(&["eq", "delay", "gain"])),
            super::RealizedProcessing::IirOnly
        );
        assert_eq!(
            super::assess_realized_processing(&graph(&["convolution", "delay"])),
            super::RealizedProcessing::FirOnly
        );
        assert_eq!(
            super::assess_realized_processing(&graph(&["eq", "convolution"])),
            super::RealizedProcessing::Hybrid
        );
        assert_eq!(
            super::assess_realized_processing(&graph(&["delay", "gain", "crossover"])),
            super::RealizedProcessing::AlignmentOnly
        );
        assert_eq!(
            super::assess_realized_processing(&graph(&[])),
            super::RealizedProcessing::Identity
        );
        assert_eq!(
            super::assess_realized_processing(&graph(&["warped_biquad"])),
            super::RealizedProcessing::IirOnly
        );
        assert_eq!(
            super::assess_realized_processing(&graph(&["kautz_filter", "convolution"])),
            super::RealizedProcessing::Hybrid
        );
    }

    #[test]
    fn fallback_reason_names_requested_and_realized() {
        use super::RealizedProcessing;
        use crate::config::ProcessingMode;
        assert_eq!(
            super::processing_fallback_reason(
                Some(&ProcessingMode::MixedPhase),
                &RealizedProcessing::IirOnly
            )
            .as_deref(),
            Some("mixed_phase_requested_iir_only_realized")
        );
        assert_eq!(
            super::processing_fallback_reason(
                Some(&ProcessingMode::LowLatency),
                &RealizedProcessing::IirOnly
            ),
            None
        );
        assert_eq!(
            super::processing_fallback_reason(
                Some(&ProcessingMode::PhaseLinear),
                &RealizedProcessing::Hybrid
            )
            .as_deref(),
            Some("phase_linear_requested_hybrid_realized")
        );
        assert_eq!(
            super::processing_fallback_reason(None, &RealizedProcessing::Identity),
            None
        );
    }

    #[test]
    fn legacy_pre_ringing_energy_name_still_reads() {
        let legacy: TemporalQualityEvidence =
            serde_json::from_value(serde_json::json!({"pre_ringing_energy_db": -27.0}))
                .expect("legacy aggregate name reads");
        assert_eq!(legacy.pre_ringing_audible_db, Some(-27.0));
        let mut policy_value = serde_json::to_value(RuntimeAcceptancePolicy::for_output_class(
            RuntimeOutputClass::Fir,
        ))
        .expect("serialize policy");
        let ceiling = policy_value
            .as_object_mut()
            .expect("policy object")
            .remove("max_pre_ringing_audible_db");
        policy_value
            .as_object_mut()
            .expect("policy object")
            .insert("max_pre_ringing_energy_db".into(), ceiling.unwrap());
        let policy: RuntimeAcceptancePolicy =
            serde_json::from_value(policy_value).expect("legacy policy name reads");
        assert_eq!(policy.max_pre_ringing_audible_db, -20.0);
        // New serializations use the corrected names.
        let value = serde_json::to_value(&legacy).expect("serialize evidence");
        assert!(value.get("pre_ringing_audible_db").is_some());
        assert!(value.get("pre_ringing_energy_db").is_none());
    }

    fn seat(
        partition: &str,
        input: &str,
        index: usize,
        lower_bound_db: f64,
    ) -> FinalSeatEvaluation {
        FinalSeatEvaluation {
            partition: partition.to_string(),
            logical_input: input.to_string(),
            seat_index: index,
            seat_label: None,
            physical_outputs: vec![input.to_string()],
            pre_summation_support: Vec::new(),
            post_summation_support: Vec::new(),
            unassessed_bands_hz: Vec::new(),
            evaluated_band_hz: [20.0, 20_000.0],
            pre_weighted_rms_db: 4.0,
            post_weighted_rms_db: 2.0,
            improvement_db: 2.0,
            improvement_lower_bound_db: lower_bound_db,
            band_improvement_db: None,
        }
    }

    fn scorecard_with_seats(seats: Vec<FinalSeatEvaluation>) -> AcousticQualityScorecard {
        AcousticQualityScorecard {
            useful_output: Vec::new(),
            final_seats: seats,
            training: QualityPartitionMetrics {
                curve_count: 2,
                pre_weighted_rms_median_db: 4.0,
                post_weighted_rms_median_db: 2.0,
                improvement_median_db: 2.0,
                worst_position_improvement_db: 1.0,
                pre_p95_abs_residual_db: 6.0,
                post_p95_abs_residual_db: 3.0,
                post_worst_abs_residual_db: 5.0,
                mean_normalized_seat_spread_db: 1.0,
                max_normalized_seat_spread_db: 2.0,
                bass_post_weighted_rms_db: None,
                upper_pre_weighted_rms_db: None,
                upper_post_weighted_rms_db: None,
                bass_pre_modal_roughness_db_per_octave2: None,
                bass_post_modal_roughness_db_per_octave2: None,
                bass_modal_roughness_improvement_db_per_octave2: None,
            },
            held_out: None,
            correction_rms_db: 2.0,
            max_boost_db: 4.0,
            max_electrical_boost_db: Some(3.5),
            max_cut_db: -6.0,
            induced_group_delay_rms_ms: Some(1.0),
            temporal: TemporalQualityEvidence {
                pre_ringing_audible_db: Some(-40.0),
                latency_ms: Some(5.0),
                alignment_delay_ms: Some(1.0),
                total_latency_ms: Some(6.0),
                available_headroom_db: Some(-3.0),
                phase_evidence_available: true,
                temporal_evidence_available: true,
                coherent_timing: CoherentTimingEvidence::Unassessed,
            },
            correction_band_hz: None,
            evaluated_band_hz: [20.0, 20_000.0],
            measurement_overlap_hz: Some([20.0, 20_000.0]),
            finite: true,
        }
    }

    #[test]
    fn playback_summary_counts_training_seats_and_echoes_temporal() {
        let mut accepted = report(CorrectionDecision::Accepted, true, vec![]);
        accepted.outcome = RoomEqOutcome::Accepted;
        accepted.metrics.improvement_db = 2.0;
        accepted.runtime_policy = Some(RuntimeAcceptancePolicy::for_output_class(
            RuntimeOutputClass::Fir,
        ));
        accepted.acoustic_quality = Some(scorecard_with_seats(vec![
            seat("training", "L", 0, 1.5),
            seat("training", "R", 1, -0.5),
            seat("held_out", "C", 2, 3.0),
        ]));
        let summary = super::playback_summary(&accepted);
        assert_eq!(summary.outcome, RoomEqOutcome::Accepted);
        assert_eq!(summary.training_seats_improved, 1);
        assert_eq!(summary.training_seats_total, 2);
        assert_eq!(summary.worst_seat.as_deref(), Some("R:1"));
        assert_eq!(summary.total_latency_ms, Some(6.0));
        assert_eq!(summary.available_headroom_db, Some(-3.0));
        assert_eq!(summary.limits.len(), 3);
        assert_eq!(summary.headlines.len(), 1);
        assert!(summary.headlines[0].contains("1/2 training seats"));
        assert!(summary.headlines[0].contains("6.0 ms"));
    }

    #[test]
    fn playback_summary_without_scorecard_reports_zero_seats() {
        let mut accepted = report(CorrectionDecision::Accepted, true, vec![]);
        accepted.outcome = RoomEqOutcome::Accepted;
        let summary = super::playback_summary(&accepted);
        assert_eq!(summary.training_seats_total, 0);
        assert_eq!(summary.training_seats_improved, 0);
        assert_eq!(summary.worst_seat, None);
        assert_eq!(summary.total_latency_ms, None);
        assert!(summary.limits.is_empty());
        assert!(summary.headlines[0].contains("0/0 training seats"));
        assert!(summary.headlines[0].contains("unassessed"));
    }
}
