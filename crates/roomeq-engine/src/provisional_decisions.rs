//! Provisional correction decision records emitted at the decision site (E2).
//!
//! Every accepted, limited, skipped, vetoed, or rejected candidate records a
//! provisional entry carrying evidence IDs, channel/source/seat context, and
//! the actual supported band. A per-filter veto stores its center frequency
//! separately; an affected interval is never inferred from Q. Optimizer
//! failure is recorded as an unresolved search, never as a physical
//! impossibility, and advisory nominations stay advisory: they never
//! authorize a removal.
//!
//! The K4 ledger types (model lane) are unlanded sibling work; these
//! engine-local records mirror that contract field-for-field so workflow
//! reconciliation can transcribe them once it freezes: stable decision ID,
//! logical input/physical output, measurement/seat references, optional
//! frequency interval, optional filter-center frequency, action, status,
//! reason codes, observed quantities/units, limits, evidence references,
//! confidence, and related/superseded links. Provisional records carry no
//! delivered-graph identity and can never read as final claims.

// Rust guideline compliant 2026-02-21

use std::collections::BTreeSet;

/// What a decision acts on: EQ, phase, gain, routing, or pruning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DecisionAction {
    /// Magnitude equalization.
    #[default]
    Equalize,
    /// Phase correction.
    PhaseCorrect,
    /// Gain trim or level adjustment.
    GainAdjust,
    /// Input/output routing change.
    Reroute,
    /// Filter removal or retention under a pruning budget.
    Prune,
}

/// Outcome vocabulary for a provisional decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DecisionStatus {
    /// Correction applied to the candidate chain.
    Applied,
    /// No correction needed; already within limits.
    AlreadyAcceptable,
    /// Evidence too weak to decide; filters are retained.
    InsufficientEvidence,
    /// Outside the configured correction scope.
    OutsideScope,
    /// Applied only partially under explicit limits; the remainder is a
    /// linked record, never dropped silently.
    Constrained,
    /// An attempted correction was rolled back.
    Reverted,
    /// No resolution reached; still open. The default.
    #[default]
    Unresolved,
    /// Advisory nomination; nothing was applied.
    Advisory,
}

/// One observed or limiting quantity with explicit units: a numerical
/// observation, never an acoustic diagnosis by itself.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedQuantity {
    /// Quantity name, e.g. `"post_p95_abs_residual_db"`.
    pub name: String,
    /// Observed value in `unit`.
    pub value: f64,
    /// Unit label, e.g. `"db"`, `"ms"`, `"hz"`.
    pub unit: String,
}

impl ObservedQuantity {
    /// Build a quantity; rejects empty names/units and nonfinite values.
    pub fn new(
        name: impl Into<String>,
        value: f64,
        unit: impl Into<String>,
    ) -> Result<Self, String> {
        let quantity = Self {
            name: name.into(),
            value,
            unit: unit.into(),
        };
        quantity.validate()?;
        Ok(quantity)
    }

    /// Reject empty names/units and nonfinite values.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err(String::from("observed quantity name must not be empty"));
        }
        if self.unit.trim().is_empty() {
            return Err(format!(
                "observed quantity '{}' must declare a unit",
                self.name
            ));
        }
        if !self.value.is_finite() {
            return Err(format!(
                "observed quantity '{}' must be finite (got {})",
                self.name, self.value
            ));
        }
        Ok(())
    }
}

/// Registry of stable evidence reference IDs backing emission.
///
/// Every evidence reference on an emitted record must exist here; unknown
/// references fail emission instead of shipping dangling citations.
#[derive(Debug, Clone, Default)]
pub struct EvidenceRegistry {
    ids: BTreeSet<String>,
}

impl EvidenceRegistry {
    /// Build a registry; rejects blank IDs.
    pub fn new(ids: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut registry = Self {
            ids: BTreeSet::new(),
        };
        for id in ids {
            if id.trim().is_empty() {
                return Err(String::from("evidence reference ID must not be empty"));
            }
            registry.ids.insert(id);
        }
        Ok(registry)
    }

    /// Whether an evidence reference is registered.
    pub fn contains(&self, id: &str) -> bool {
        self.ids.contains(id)
    }
}

/// Inputs for one provisional decision emission.
#[derive(Debug, Clone)]
pub struct DecisionParams {
    /// Stable decision identifier; must be nonempty.
    pub decision_id: String,
    /// Logical input identity, e.g. `"stereo"`.
    pub logical_input: String,
    /// Physical output identity, e.g. `"sub-1"`.
    pub physical_output: String,
    /// Stable measurement reference IDs.
    pub measurement_refs: Vec<String>,
    /// Seat reference IDs.
    pub seat_refs: Vec<String>,
    /// Explicit supported frequency interval in Hz, when band-scoped.
    /// Never inferred from `filter_center_hz`.
    pub frequency_band_hz: Option<[f64; 2]>,
    /// Filter center frequency in Hz, when the decision concerns one
    /// filter. A point, not an interval.
    pub filter_center_hz: Option<f64>,
    /// What the decision acts on.
    pub action: DecisionAction,
    /// Outcome vocabulary.
    pub status: DecisionStatus,
    /// Machine-readable reason codes.
    pub reason_codes: Vec<String>,
    /// Observed quantities with units.
    pub observed: Vec<ObservedQuantity>,
    /// Applied limits with units.
    pub limits: Vec<ObservedQuantity>,
    /// Stable evidence reference IDs; every one must be registered.
    pub evidence_refs: Vec<String>,
    /// Confidence in the acoustic diagnosis, if any.
    pub confidence: roomeq_model::AssessmentConfidence,
    /// Linked record IDs (e.g. an applied partial linked to its
    /// constrained remainder).
    pub related_decision_ids: Vec<String>,
    /// Superseded record IDs (e.g. a candidate superseded by rollback).
    pub supersedes_ids: Vec<String>,
}

/// One provisional correction decision record.
///
/// Engine stages record provisional entries only; only workflow
/// reconciliation binds final records to the delivered graph. This type has
/// no delivered-graph identity field, so it can never stand as a final
/// delivery claim.
#[derive(Debug, Clone, PartialEq)]
pub struct ProvisionalDecision {
    /// Stable decision identifier.
    pub decision_id: String,
    /// Logical input identity.
    pub logical_input: String,
    /// Physical output identity.
    pub physical_output: String,
    /// Stable measurement reference IDs.
    pub measurement_refs: Vec<String>,
    /// Seat reference IDs.
    pub seat_refs: Vec<String>,
    /// Explicit supported frequency interval in Hz, when band-scoped.
    pub frequency_band_hz: Option<[f64; 2]>,
    /// Filter center frequency in Hz. A point, not an interval.
    pub filter_center_hz: Option<f64>,
    /// What the decision acts on.
    pub action: DecisionAction,
    /// Outcome vocabulary.
    pub status: DecisionStatus,
    /// Machine-readable reason codes.
    pub reason_codes: Vec<String>,
    /// Observed quantities with units.
    pub observed: Vec<ObservedQuantity>,
    /// Applied limits with units.
    pub limits: Vec<ObservedQuantity>,
    /// Stable evidence reference IDs.
    pub evidence_refs: Vec<String>,
    /// Confidence in the acoustic diagnosis, if any.
    pub confidence: roomeq_model::AssessmentConfidence,
    /// Linked record IDs.
    pub related_decision_ids: Vec<String>,
    /// Superseded record IDs.
    pub supersedes_ids: Vec<String>,
}

impl ProvisionalDecision {
    /// Lifecycle stage of engine records: always provisional.
    pub fn stage(&self) -> &'static str {
        "provisional"
    }

    /// Provisional records are never final delivery claims.
    pub fn is_final_claim(&self) -> bool {
        false
    }

    /// The explicitly declared band only. A filter center never
    /// manufactures an interval: records with a center but no band return
    /// `None`.
    pub fn effective_band(&self) -> Option<[f64; 2]> {
        self.frequency_band_hz
    }

    /// Whether this record authorizes applying a removal: only applied or
    /// constrained pruning records do. Advisory nominations, unresolved
    /// searches, and insufficient-evidence records never authorize removal.
    pub fn authorizes_removal(&self) -> bool {
        self.action == DecisionAction::Prune
            && matches!(
                self.status,
                DecisionStatus::Applied | DecisionStatus::Constrained
            )
    }

    /// Link another record ID (e.g. an applied partial to its constrained
    /// remainder); rejects blank or duplicate links.
    pub fn link_related(&mut self, other_id: impl Into<String>) -> Result<(), EmitError> {
        let other_id = other_id.into();
        if other_id.trim().is_empty() {
            return Err(EmitError(String::from(
                "related decision ID must not be empty",
            )));
        }
        if !self.related_decision_ids.contains(&other_id) {
            self.related_decision_ids.push(other_id);
        }
        Ok(())
    }
}

/// Emission failure: unregistered evidence, invalid bands, missing units,
/// or blank identities. Never silently downgraded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitError(pub String);

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "decision emission failed: {}", self.0)
    }
}

impl std::error::Error for EmitError {}

/// Emit a provisional decision record at the decision site.
///
/// Validates identities, bands, quantities (units required), and evidence
/// references (every one must exist in `registry`). Records a veto's
/// filter-center frequency separately without inferring an interval.
pub fn emit_decision(
    registry: &EvidenceRegistry,
    params: DecisionParams,
) -> Result<ProvisionalDecision, EmitError> {
    let invalid = |message: String| EmitError(message);
    if params.decision_id.trim().is_empty() {
        return Err(invalid(String::from("decision_id must not be empty")));
    }
    if params.logical_input.trim().is_empty() {
        return Err(invalid(String::from(
            "decision logical_input must not be empty",
        )));
    }
    if params.physical_output.trim().is_empty() {
        return Err(invalid(String::from(
            "decision physical_output must not be empty",
        )));
    }
    if let Some(band_hz) = params.frequency_band_hz
        && (!band_hz[0].is_finite()
            || !band_hz[1].is_finite()
            || band_hz[0] <= 0.0
            || band_hz[1] <= band_hz[0])
    {
        return Err(invalid(format!(
            "frequency_band_hz must satisfy 0 < lo < hi with finite bounds (got [{}, {}])",
            band_hz[0], band_hz[1]
        )));
    }
    if let Some(center_hz) = params.filter_center_hz
        && (!center_hz.is_finite() || center_hz <= 0.0)
    {
        return Err(invalid(format!(
            "filter_center_hz must be finite and positive (got {center_hz})"
        )));
    }
    for quantity in params.observed.iter().chain(params.limits.iter()) {
        quantity.validate().map_err(invalid)?;
    }
    for reference in &params.evidence_refs {
        if !registry.contains(reference) {
            return Err(invalid(format!(
                "evidence reference '{reference}' is not in the supplied evidence registry"
            )));
        }
    }
    Ok(ProvisionalDecision {
        decision_id: params.decision_id,
        logical_input: params.logical_input,
        physical_output: params.physical_output,
        measurement_refs: params.measurement_refs,
        seat_refs: params.seat_refs,
        frequency_band_hz: params.frequency_band_hz,
        filter_center_hz: params.filter_center_hz,
        action: params.action,
        status: params.status,
        reason_codes: params.reason_codes,
        observed: params.observed,
        limits: params.limits,
        evidence_refs: params.evidence_refs,
        confidence: params.confidence,
        related_decision_ids: params.related_decision_ids,
        supersedes_ids: params.supersedes_ids,
    })
}

/// Record an optimizer failure as an unresolved search.
///
/// A failed search is not a physical impossibility: no diagnosis, no
/// uncorrectable verdict, and no removal authorization.
pub fn emit_optimizer_failure(
    registry: &EvidenceRegistry,
    decision_id: impl Into<String>,
    logical_input: impl Into<String>,
    physical_output: impl Into<String>,
    evidence_refs: Vec<String>,
    detail: impl Into<String>,
) -> Result<ProvisionalDecision, EmitError> {
    let detail = detail.into();
    emit_decision(
        registry,
        DecisionParams {
            decision_id: decision_id.into(),
            logical_input: logical_input.into(),
            physical_output: physical_output.into(),
            measurement_refs: Vec::new(),
            seat_refs: Vec::new(),
            frequency_band_hz: None,
            filter_center_hz: None,
            action: DecisionAction::Equalize,
            status: DecisionStatus::Unresolved,
            reason_codes: vec![String::from("optimizer_failed"), detail],
            observed: Vec::new(),
            limits: Vec::new(),
            evidence_refs,
            confidence: roomeq_model::AssessmentConfidence::Unknown,
            related_decision_ids: Vec::new(),
            supersedes_ids: Vec::new(),
        },
    )
}

/// Record an advisory nomination (e.g. a veto heuristic vote).
///
/// Advisory records never authorize removal; enforcement stays with the
/// cumulative adjudication and workflow reconciliation.
pub fn emit_advisory_nomination(
    registry: &EvidenceRegistry,
    decision_id: impl Into<String>,
    logical_input: impl Into<String>,
    physical_output: impl Into<String>,
    filter_center_hz: Option<f64>,
    evidence_refs: Vec<String>,
    reason: impl Into<String>,
) -> Result<ProvisionalDecision, EmitError> {
    emit_decision(
        registry,
        DecisionParams {
            decision_id: decision_id.into(),
            logical_input: logical_input.into(),
            physical_output: physical_output.into(),
            measurement_refs: Vec::new(),
            seat_refs: Vec::new(),
            frequency_band_hz: None,
            filter_center_hz,
            action: DecisionAction::Prune,
            status: DecisionStatus::Advisory,
            reason_codes: vec![String::from("advisory_nomination"), reason.into()],
            observed: Vec::new(),
            limits: Vec::new(),
            evidence_refs,
            confidence: roomeq_model::AssessmentConfidence::Low,
            related_decision_ids: Vec::new(),
            supersedes_ids: Vec::new(),
        },
    )
}

#[cfg(test)]
mod provisional_decision_tests {
    use super::*;

    fn registry() -> EvidenceRegistry {
        EvidenceRegistry::new([String::from("ev-1"), String::from("ev-2")]).expect("valid registry")
    }

    fn base_params() -> DecisionParams {
        DecisionParams {
            decision_id: String::from("dec-1"),
            logical_input: String::from("stereo"),
            physical_output: String::from("main-l"),
            measurement_refs: vec![String::from("meas-1")],
            seat_refs: vec![String::from("seat-a")],
            frequency_band_hz: Some([40.0, 400.0]),
            filter_center_hz: None,
            action: DecisionAction::Equalize,
            status: DecisionStatus::Applied,
            reason_codes: vec![String::from("within_limits")],
            observed: vec![
                ObservedQuantity::new("post_p95_abs_residual_db", 3.0, "db").expect("valid"),
            ],
            limits: vec![
                ObservedQuantity::new("max_post_p95_abs_residual_db", 6.0, "db").expect("valid"),
            ],
            evidence_refs: vec![String::from("ev-1")],
            confidence: roomeq_model::AssessmentConfidence::Moderate,
            related_decision_ids: Vec::new(),
            supersedes_ids: Vec::new(),
        }
    }

    #[test]
    fn engine_decision_outside_scope_vs_missing_evidence() {
        // Outside-scope and missing-evidence outcomes are distinct statuses
        // with distinct reasons; one never stands in for the other.
        let registry = registry();
        let mut outside = base_params();
        outside.decision_id = String::from("dec-scope");
        outside.status = DecisionStatus::OutsideScope;
        outside.frequency_band_hz = Some([4000.0, 8000.0]);
        outside.reason_codes = vec![String::from("outside_requested_scope")];
        let outside = emit_decision(&registry, outside).expect("valid scope record");

        let mut missing = base_params();
        missing.decision_id = String::from("dec-evidence");
        missing.status = DecisionStatus::InsufficientEvidence;
        missing.reason_codes = vec![String::from("no_supporting_evidence")];
        missing.evidence_refs = Vec::new();
        let missing = emit_decision(&registry, missing).expect("valid evidence record");

        assert_ne!(outside.status, missing.status);
        assert_eq!(outside.effective_band(), Some([4000.0, 8000.0]));
        assert!(!outside.authorizes_removal());
        assert!(!missing.authorizes_removal());
        // A per-filter veto records its center separately; no interval is
        // inferred from it.
        let mut veto = base_params();
        veto.decision_id = String::from("dec-veto");
        veto.frequency_band_hz = None;
        veto.filter_center_hz = Some(120.0);
        let veto = emit_decision(&registry, veto).expect("valid veto record");
        assert_eq!(veto.filter_center_hz, Some(120.0));
        assert_eq!(veto.effective_band(), None);
    }

    #[test]
    fn engine_decision_constraint_has_observed_limit_and_units() {
        // Constrained records carry observed quantities and limits with
        // units; unitless quantities fail emission.
        let registry = registry();
        let mut params = base_params();
        params.decision_id = String::from("dec-constrained");
        params.status = DecisionStatus::Constrained;
        params.reason_codes = vec![String::from("binding_gain_constraint")];
        let mut record = emit_decision(&registry, params).expect("valid constrained record");
        assert_eq!(record.observed[0].unit, "db");
        assert_eq!(record.limits[0].unit, "db");
        // A constrained partial links its remainder instead of dropping it.
        record.link_related("dec-remainder").expect("valid link");
        assert_eq!(
            record.related_decision_ids,
            vec![String::from("dec-remainder")]
        );
        assert!(record.link_related("  ").is_err());

        assert!(
            ObservedQuantity::new("post_p95_abs_residual_db", 3.0, "").is_err(),
            "unitless observations must fail"
        );
        assert!(
            ObservedQuantity::new("post_p95_abs_residual_db", f64::NAN, "db").is_err(),
            "nonfinite observations must fail"
        );
        let mut bad = base_params();
        bad.evidence_refs = vec![String::from("ev-unknown")];
        assert!(
            emit_decision(&registry, bad).is_err(),
            "unregistered evidence must fail emission"
        );
    }

    #[test]
    fn engine_advisory_nomination_not_applied_removal() {
        // Advisory nominations stay advisory: no removal authorization, no
        // final-claim binding, center recorded without an inferred band.
        let registry = registry();
        let advisory = emit_advisory_nomination(
            &registry,
            "dec-advisory",
            "stereo",
            "sub-1",
            Some(63.0),
            vec![String::from("ev-2")],
            "sub_jnd_nomination",
        )
        .expect("valid advisory");
        assert_eq!(advisory.status, DecisionStatus::Advisory);
        assert_eq!(advisory.stage(), "provisional");
        assert!(!advisory.is_final_claim());
        assert!(!advisory.authorizes_removal());
        assert_eq!(advisory.filter_center_hz, Some(63.0));
        assert_eq!(advisory.effective_band(), None);
        // Only applied/constrained pruning records authorize removal.
        let mut applied = base_params();
        applied.action = DecisionAction::Prune;
        applied.status = DecisionStatus::Applied;
        let applied = emit_decision(&registry, applied).expect("valid");
        assert!(applied.authorizes_removal());
    }

    #[test]
    fn engine_unresolved_search_not_uncorrectable_diagnosis() {
        // Optimizer failure records an unresolved search with the failure as
        // the reason: no physical diagnosis, no uncorrectable verdict.
        let registry = registry();
        let failure = emit_optimizer_failure(
            &registry,
            "dec-search",
            "stereo",
            "main-r",
            vec![String::from("ev-1")],
            "de_backend_did_not_converge",
        )
        .expect("valid failure record");
        assert_eq!(failure.status, DecisionStatus::Unresolved);
        assert!(
            failure
                .reason_codes
                .contains(&String::from("optimizer_failed"))
        );
        assert!(!failure.authorizes_removal());
        assert_eq!(
            failure.confidence,
            roomeq_model::AssessmentConfidence::Unknown
        );
        // Blank identities and bad bands fail instead of shipping.
        let mut bad = base_params();
        bad.logical_input = String::from("  ");
        assert!(emit_decision(&registry, bad).is_err());
        let mut bad_band = base_params();
        bad_band.frequency_band_hz = Some([400.0, 40.0]);
        assert!(emit_decision(&registry, bad_band).is_err());
    }
}
