//! Optimizer invocation evidence shared across AutoEQ and RoomEQ.
//!
//! These serialized types are produced by optimizer backends and
//! consumed by RoomEQ report assembly, so they live below both
//! `autoeq-optim` and `roomeq-model` (refactor-20261025 Phase 3c).
//! Serialized field names and schema markers are a wire contract and
//! must not change here.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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

// Preserve the original evidence import path after separating ownership.
pub use crate::q_adjustment::QAdjustment;
