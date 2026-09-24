//! K4 correction decision ledger and K5 playback/model-reference/listening
//! descriptors: additive versioned types only.
//!
//! Engine stages record [`DecisionStage::Provisional`] entries; only workflow
//! reconciliation binds [`DecisionStage::Final`] records to the delivered
//! graph. [`crate::CorrectionAcceptanceReport`] stays the authoritative
//! acceptance record: this ledger never renames or replaces its states.
//! A filter-center frequency is not a frequency interval
//! ([`DecisionRecord::effective_band`]).
//!
//! These types must not import `roomeq-quality`; that would create a
//! dependency cycle. Quality owns evaluation, this crate owns the record.

// Rust guideline compliant 2026-02-21

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::AssessmentConfidence;

/// Version pin for [`CorrectionDecisionLedger`] records.
pub const DECISION_LEDGER_VERSION: &str = "1.0.0";

/// Version pin for [`PlaybackComparison`] descriptors.
pub const PLAYBACK_POLICY_VERSION: &str = "1.0.0";

fn decision_ledger_version_default() -> String {
    DECISION_LEDGER_VERSION.to_string()
}

fn playback_policy_version_default() -> String {
    PLAYBACK_POLICY_VERSION.to_string()
}

/// Lifecycle stage of a decision record.
///
/// There are no aliases: a provisional record can never deserialize as a
/// final delivery claim merely because a stage field was present.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DecisionStage {
    /// Recorded by the engine before reconciliation; not a delivery claim.
    #[default]
    Provisional,
    /// Bound to the delivered graph by workflow reconciliation.
    Final,
}

/// What the decision acts on: EQ, phase, gain, routing, or pruning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
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

/// Pinned status vocabulary (global K4 contract).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DecisionStatus {
    /// Correction applied to the chain.
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
    /// No resolution reached; still open.
    #[default]
    Unresolved,
    /// Advisory nomination; nothing was applied.
    Advisory,
}

impl DecisionStatus {
    /// Every pinned status, for round-trip coverage.
    pub const ALL: [DecisionStatus; 8] = [
        DecisionStatus::Applied,
        DecisionStatus::AlreadyAcceptable,
        DecisionStatus::InsufficientEvidence,
        DecisionStatus::OutsideScope,
        DecisionStatus::Constrained,
        DecisionStatus::Reverted,
        DecisionStatus::Unresolved,
        DecisionStatus::Advisory,
    ];
}

/// One observed or limiting quantity with explicit units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ObservedQuantity {
    /// Quantity name, for example `"post_p95_abs_residual_db"`.
    pub name: String,
    /// Observed value in `unit`.
    pub value: f64,
    /// Unit label, for example `"db"`, `"ms"`, `"hz"`.
    pub unit: String,
}

impl ObservedQuantity {
    /// Reject nonfinite values and empty names or units.
    ///
    /// # Errors
    ///
    /// Returns a reason for an empty name/unit or a nonfinite value.
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

/// One versioned correction decision record (global K4 contract).
///
/// A constrained solution that still contains an applied partial correction
/// is represented as linked records via `related_decision_ids`, and a
/// rollback as a superseding record via `supersedes_ids`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DecisionRecord {
    /// Stable decision identifier; must be nonempty to validate.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub decision_id: String,
    /// Ledger version; must equal [`DECISION_LEDGER_VERSION`].
    #[serde(default = "decision_ledger_version_default")]
    pub ledger_version: String,
    /// Provisional (engine) or final (workflow-reconciled).
    #[serde(default)]
    pub stage: DecisionStage,
    /// Logical input identity, for example `"stereo"`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub logical_input: String,
    /// Physical output identity, for example `"sub-1"`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub physical_output: String,
    /// Stable measurement reference IDs behind the decision.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub measurement_refs: Vec<String>,
    /// Seat reference IDs behind the decision.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seat_refs: Vec<String>,
    /// Explicit overlapping frequency interval in Hz, when the decision is
    /// band-scoped. Never inferred from `filter_center_hz`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_band_hz: Option<[f64; 2]>,
    /// Filter center frequency in Hz, when the decision concerns one filter.
    /// This is a point, not an interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter_center_hz: Option<f64>,
    /// What the decision acts on.
    #[serde(default)]
    pub action: DecisionAction,
    /// Pinned status vocabulary.
    #[serde(default)]
    pub status: DecisionStatus,
    /// Machine-readable reason codes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reason_codes: Vec<String>,
    /// Observed quantities with units; a numerical observation, not a diagnosis.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observed: Vec<ObservedQuantity>,
    /// Applied limits with units.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub limits: Vec<ObservedQuantity>,
    /// Stable evidence reference IDs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_refs: Vec<String>,
    /// Confidence in the acoustic diagnosis, if any.
    #[serde(default)]
    pub confidence: AssessmentConfidence,
    /// Linked record IDs (for example an applied partial linked to its
    /// constrained remainder).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related_decision_ids: Vec<String>,
    /// Superseded record IDs (for example a candidate superseded by rollback).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supersedes_ids: Vec<String>,
    /// Immutable delivered-graph identity. Required for final delivery
    /// claims; provisional records leave it absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_graph_identity: Option<String>,
}

impl DecisionRecord {
    /// Minimal example record for one status, used by tests and documentation.
    pub fn example(status: DecisionStatus) -> Self {
        Self {
            decision_id: format!("example-{}", serde_json::to_value(status).unwrap()),
            ledger_version: decision_ledger_version_default(),
            stage: DecisionStage::Provisional,
            logical_input: String::from("stereo"),
            physical_output: String::from("main-l"),
            measurement_refs: vec![String::from("meas-1")],
            seat_refs: vec![String::from("seat-a")],
            frequency_band_hz: Some([40.0, 400.0]),
            filter_center_hz: None,
            action: DecisionAction::Equalize,
            status,
            reason_codes: vec![String::from("example")],
            observed: vec![ObservedQuantity {
                name: String::from("post_p95_abs_residual_db"),
                value: 3.0,
                unit: String::from("db"),
            }],
            limits: vec![ObservedQuantity {
                name: String::from("max_post_p95_abs_residual_db"),
                value: 6.0,
                unit: String::from("db"),
            }],
            evidence_refs: vec![String::from("ev-1")],
            confidence: AssessmentConfidence::Unknown,
            related_decision_ids: Vec::new(),
            supersedes_ids: Vec::new(),
            final_graph_identity: None,
        }
    }

    /// The explicitly declared band only. A filter center never manufactures
    /// an interval: records with a center but no band return `None`.
    pub fn effective_band(&self) -> Option<[f64; 2]> {
        self.frequency_band_hz
    }

    /// Whether this record is a verified final delivery claim: final stage
    /// bound to a stated delivered-graph identity.
    pub fn is_final_claim(&self) -> bool {
        self.stage == DecisionStage::Final
            && self
                .final_graph_identity
                .as_ref()
                .is_some_and(|identity| !identity.trim().is_empty())
    }

    /// Structural check: version, IDs, bands, finiteness, and finality
    /// binding. Rejected runs keep attempted records; only final delivery
    /// claims require a graph identity.
    ///
    /// # Errors
    ///
    /// Returns a reason for unknown versions, empty IDs, nonfinite or
    /// unordered bands, nonfinite quantities, or a final stage without a
    /// delivered-graph identity.
    pub fn validate(&self) -> Result<(), String> {
        if self.ledger_version != DECISION_LEDGER_VERSION {
            return Err(format!(
                "unsupported decision ledger version '{}'; expected '{}'",
                self.ledger_version, DECISION_LEDGER_VERSION
            ));
        }
        if self.decision_id.trim().is_empty() {
            return Err(String::from("decision_id must not be empty"));
        }
        if self.logical_input.trim().is_empty() {
            return Err(String::from("decision logical_input must not be empty"));
        }
        if self.physical_output.trim().is_empty() {
            return Err(String::from("decision physical_output must not be empty"));
        }
        if let Some(band_hz) = self.frequency_band_hz
            && (!band_hz[0].is_finite()
                || !band_hz[1].is_finite()
                || band_hz[0] <= 0.0
                || band_hz[1] <= band_hz[0])
        {
            return Err(format!(
                "frequency_band_hz must satisfy 0 < lo < hi with finite bounds (got [{}, {}])",
                band_hz[0], band_hz[1]
            ));
        }
        if let Some(center_hz) = self.filter_center_hz
            && (!center_hz.is_finite() || center_hz <= 0.0)
        {
            return Err(format!(
                "filter_center_hz must be finite and positive (got {center_hz})"
            ));
        }
        for quantity in self.observed.iter().chain(self.limits.iter()) {
            quantity.validate()?;
        }
        if self.stage == DecisionStage::Final && !self.is_final_claim() {
            return Err(String::from(
                "final-stage decisions require a delivered-graph identity; provisional records cannot stand as final claims",
            ));
        }
        Ok(())
    }
}

/// Ordered set of decision records with a pinned ledger version.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CorrectionDecisionLedger {
    /// Supplemental quality views, independently bound to this ledger's graph.
    /// Absent in legacy outputs; never promotes a correction or playback verdict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_evidence: Option<crate::acceptance_evidence::AcceptanceEvidence>,
    /// Recomputable payload binding for report consumers; absent in legacy files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_binding: Option<crate::payload_binding::PayloadBinding>,
    /// Ledger version; must equal [`DECISION_LEDGER_VERSION`].
    #[serde(default = "decision_ledger_version_default")]
    pub ledger_version: String,
    /// Decision records in insertion order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decisions: Vec<DecisionRecord>,
    /// Per-channel delivered-scope shares (req R6 `operational_room_response_pct`).
    /// Derived from the final records above at reconciliation time; absent in
    /// legacy outputs. A channel with no decided final equalization scope has
    /// no entry here, which viewers render as pending, never as 100%.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channel_summaries: Vec<ChannelOperationalSummary>,
}

/// Per-channel share of decided final equalization scope that was delivered.
///
/// Normative rule (req-roomeq-report R6), computed over the reconciled final
/// ledger only:
/// - scope: final-stage records (`is_final_claim`) with action `Equalize`,
///   grouped by `physical_output`;
/// - a record superseded by another record in the same ledger (its
///   `decision_id` appears in another record's `supersedes_ids`) is no longer
///   operative and is excluded from both counts;
/// - delivered: scope records with status `Applied`, `AlreadyAcceptable`, or
///   `Constrained` — the same delivery-claim set reconciliation binds to the
///   delivered graph. `Reverted`, `InsufficientEvidence`, `OutsideScope`, and
///   `Unresolved` scope lowers the share; `Advisory` records never enter it
///   because provisional history cannot become a final claim;
/// - `operational_response_pct` is `100 * delivered / decided`. It is a
///   ledger-delivery share with provenance-bound bands, not a measured
///   band-compliance percentage: an empty decided scope yields no percentage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ChannelOperationalSummary {
    /// Delivered physical output identity (ledger `physical_output`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub channel: String,
    /// Share of decided final equalization scope delivered, in percent.
    /// `None` only when the scope count is zero; summaries are then omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operational_response_pct: Option<f64>,
    /// Decided final equalization records for the channel.
    pub decided_equalize: u32,
    /// Thereof delivered (`Applied`/`AlreadyAcceptable`/`Constrained`).
    pub delivered_equalize: u32,
}

impl ChannelOperationalSummary {
    /// Reject empty channels, inconsistent counts, and nonfinite percentages.
    ///
    /// # Errors
    ///
    /// Returns a reason for an empty channel, delivered counts above decided
    /// counts, or a nonfinite percentage with nonzero scope.
    pub fn validate(&self) -> Result<(), String> {
        if self.channel.trim().is_empty() {
            return Err(String::from("channel summary channel must not be empty"));
        }
        if self.delivered_equalize > self.decided_equalize {
            return Err(format!(
                "channel summary '{}' delivers {} of {} decided records",
                self.channel, self.delivered_equalize, self.decided_equalize
            ));
        }
        match (self.decided_equalize, self.operational_response_pct) {
            (0, None) => Ok(()),
            (0, Some(_)) => Err(format!(
                "channel summary '{}' must not carry a percentage without decided scope",
                self.channel
            )),
            (_, None) => Err(format!(
                "channel summary '{}' must carry a percentage for decided scope",
                self.channel
            )),
            (_, Some(pct)) if !pct.is_finite() => Err(format!(
                "channel summary '{}' percentage must be finite (got {})",
                self.channel, pct
            )),
            (_, Some(_)) => Ok(()),
        }
    }
}

/// Derive per-channel operational summaries over reconciled final records.
///
/// See [`ChannelOperationalSummary`] for the normative rule. Output is sorted
/// by channel; channels without decided final equalization scope are absent.
#[must_use]
pub fn operational_summaries(decisions: &[DecisionRecord]) -> Vec<ChannelOperationalSummary> {
    let superseded: std::collections::BTreeSet<&str> = decisions
        .iter()
        .flat_map(|record| record.supersedes_ids.iter().map(String::as_str))
        .collect();
    let mut decided: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
    let mut delivered: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
    for record in decisions {
        if !record.is_final_claim() || record.action != DecisionAction::Equalize {
            continue;
        }
        if superseded.contains(record.decision_id.as_str()) {
            continue;
        }
        *decided.entry(record.physical_output.as_str()).or_default() += 1;
        if matches!(
            record.status,
            DecisionStatus::Applied
                | DecisionStatus::AlreadyAcceptable
                | DecisionStatus::Constrained
        ) {
            *delivered
                .entry(record.physical_output.as_str())
                .or_default() += 1;
        }
    }
    decided
        .into_iter()
        .map(|(channel, decided_equalize)| {
            let delivered_equalize = delivered.get(channel).copied().unwrap_or(0);
            ChannelOperationalSummary {
                channel: channel.to_owned(),
                operational_response_pct: Some(
                    100.0 * f64::from(delivered_equalize) / f64::from(decided_equalize),
                ),
                decided_equalize,
                delivered_equalize,
            }
        })
        .collect()
}

impl CorrectionDecisionLedger {
    /// Check version, per-record validity, and decision-ID uniqueness.
    ///
    /// # Errors
    ///
    /// Returns a reason for unknown versions, invalid records, or duplicate
    /// decision IDs. Seat/channel gaps stay visible: rows are never merged.
    pub fn validate(&self) -> Result<(), String> {
        if self.ledger_version != DECISION_LEDGER_VERSION {
            return Err(format!(
                "unsupported decision ledger version '{}'; expected '{}'",
                self.ledger_version, DECISION_LEDGER_VERSION
            ));
        }
        if let Some(evidence) = &self.acceptance_evidence {
            let graph = self
                .payload_binding
                .as_ref()
                .map_or(evidence.binding.graph_identity.as_str(), |binding| {
                    binding.graph_identity.as_str()
                });
            if !evidence.matches(graph) {
                return Err("acceptance evidence payload or graph binding is stale".into());
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for decision in &self.decisions {
            decision.validate()?;
            if !seen.insert(decision.decision_id.clone()) {
                return Err(format!("duplicate decision_id '{}'", decision.decision_id));
            }
        }
        let mut channels = std::collections::BTreeSet::new();
        for summary in &self.channel_summaries {
            summary.validate()?;
            if !channels.insert(summary.channel.clone()) {
                return Err(format!("duplicate channel summary '{}'", summary.channel));
            }
        }
        Ok(())
    }
}

/// What kind of capture backs a playback comparison (global K5 contract).
///
/// Acoustic recordings and simulated/backend renderings are distinct kinds:
/// no backend self-comparison or synthetic capture counts as real playback
/// validation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CaptureKind {
    /// Stationary impulse-response capture with a timing reference.
    StationaryIr,
    /// Spatial magnitude capture without a timing reference.
    SpatialMagnitude,
    /// Direct-sound capture.
    DirectSound,
    /// Exported-backend simulation or rendering; not an acoustic recording.
    SimulatedBackend,
    /// Capture kind not stated.
    #[default]
    Unknown,
}

/// Playback comparison descriptor binding a comparison to immutable graph,
/// stimulus, calibration, routing, and policy identities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PlaybackComparison {
    /// Descriptor policy version; must equal [`PLAYBACK_POLICY_VERSION`].
    #[serde(default = "playback_policy_version_default")]
    pub policy_version: String,
    /// Immutable baseline graph identity; absent means unassessed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_graph_identity: Option<String>,
    /// Immutable candidate graph identity; absent means unassessed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_graph_identity: Option<String>,
    /// Sample rate in Hz the comparison was evaluated at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    /// Calibration reference identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_ref: Option<String>,
    /// Source identifier the stimulus was rendered for.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_id: String,
    /// Seat identifiers the comparison covers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seat_ids: Vec<String>,
    /// Stimulus content hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stimulus_hash: Option<String>,
    /// Processing-state label, for example `"small_signal"`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub processing_state: String,
    /// Comparison policy version.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub comparison_policy_version: String,
    /// Capture kind; acoustic and simulated kinds never interchange.
    #[serde(default)]
    pub capture_kind: CaptureKind,
}

impl Default for PlaybackComparison {
    fn default() -> Self {
        Self {
            policy_version: playback_policy_version_default(),
            baseline_graph_identity: None,
            candidate_graph_identity: None,
            sample_rate_hz: None,
            calibration_ref: None,
            source_id: String::new(),
            seat_ids: Vec::new(),
            stimulus_hash: None,
            processing_state: String::new(),
            comparison_policy_version: String::new(),
            capture_kind: CaptureKind::default(),
        }
    }
}

impl PlaybackComparison {
    /// Reason the comparison is unassessed, or `None` when the identities
    /// required for assessment are present. Missing or mismatched graph,
    /// stimulus, or calibration identities yield insufficient evidence,
    /// never a promotion.
    pub fn unassessed_reason(&self) -> Option<String> {
        if self.policy_version != PLAYBACK_POLICY_VERSION {
            return Some(format!(
                "unsupported playback policy version '{}'",
                self.policy_version
            ));
        }
        let baseline_missing = self
            .baseline_graph_identity
            .as_ref()
            .is_none_or(|identity| identity.trim().is_empty());
        let candidate_missing = self
            .candidate_graph_identity
            .as_ref()
            .is_none_or(|identity| identity.trim().is_empty());
        if baseline_missing || candidate_missing {
            return Some(String::from(
                "baseline/candidate graph identities are missing; playback evidence is unassessed",
            ));
        }
        if self.baseline_graph_identity == self.candidate_graph_identity {
            return Some(String::from(
                "baseline and candidate share one graph identity; no backend self-comparison counts as validation",
            ));
        }
        if self
            .stimulus_hash
            .as_ref()
            .is_none_or(|hash| hash.trim().is_empty())
        {
            return Some(String::from(
                "stimulus hash is missing; playback evidence is unassessed",
            ));
        }
        None
    }

    /// Whether the comparison carries the identities needed for assessment.
    pub fn is_assessed(&self) -> bool {
        self.unassessed_reason().is_none()
    }
}

/// Reference to a model-reference protocol artifact, by identity.
///
/// A nonempty reference string alone never implies a validated result; only
/// an explicit `validated` flag counts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelReferenceDescriptor {
    /// Protocol artifact identifier, for example `"epa-model-v3"`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub protocol_id: String,
    /// Immutable artifact identity (hash or versioned ID).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_identity: Option<String>,
    /// Explicit validation flag; defaults to unvalidated.
    #[serde(default)]
    pub validated: bool,
}

impl ModelReferenceDescriptor {
    /// Whether the reference counts as a validated result.
    pub fn is_validated(&self) -> bool {
        self.validated
            && !self.protocol_id.trim().is_empty()
            && self
                .artifact_identity
                .as_ref()
                .is_some_and(|identity| !identity.trim().is_empty())
    }
}

/// Outcome vocabulary for listening evidence; inconclusive is not success.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ListeningResult {
    /// No trial ran or no result was recorded.
    #[default]
    Unassessed,
    /// Trials ran without meeting the preregistered benefit criteria.
    Inconclusive,
    /// Trials met the preregistered benefit criteria.
    BenefitDemonstrated,
    /// Trials met the preregistered no-benefit outcome.
    NoBenefit,
}

/// Reference to a listening-evidence protocol artifact, by identity.
///
/// No modeled perceptual score counts as a listener result here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ListeningEvidenceDescriptor {
    /// Listening protocol identifier.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub protocol_id: String,
    /// Stimulus content hash the trials used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stimulus_hash: Option<String>,
    /// Number of completed trials, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trial_count: Option<usize>,
    /// Recorded outcome; defaults to unassessed.
    #[serde(default)]
    pub result: ListeningResult,
}

/// Immutable delivered-payload identity: canonical JSON plus a compact
/// fingerprint.
///
/// Object keys are sorted before serialization, so two payloads with
/// identical content share one identity regardless of insertion order.
/// This is the workflow-local binding target until the export lane
/// publishes the canonical X1 graph hash (handoff); the comparison
/// semantics (exact canonical equality) already match that contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GraphIdentity {
    /// Canonical (key-order-stable) JSON of the delivered payload.
    pub canonical_json: String,
    /// FNV-1a 64-bit fingerprint of the canonical JSON, hex-encoded.
    pub fingerprint: String,
}

impl GraphIdentity {
    /// Whether this identity binds the given graph.
    pub fn binds(&self, graph: &crate::DspGraph) -> bool {
        canonical_graph_identity(graph) == *self
    }
}

fn sort_canonical(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let sorted: std::collections::BTreeMap<String, serde_json::Value> = map
                .into_iter()
                .map(|(key, value)| (key, sort_canonical(value)))
                .collect();
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(sort_canonical).collect())
        }
        scalar => scalar,
    }
}

fn fnv1a_hex(input: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Compute the immutable identity of any delivered JSON value.
///
/// The caller binds the DSP content, never the ledger itself: compute
/// over the payload with its ledger attachment cleared.
pub fn canonical_value_identity(value: &serde_json::Value) -> GraphIdentity {
    let canonical_json =
        serde_json::to_string(&sort_canonical(value.clone())).expect("canonical JSON");
    let fingerprint = fnv1a_hex(&canonical_json);
    GraphIdentity {
        canonical_json,
        fingerprint,
    }
}

/// Compute the immutable identity of a delivered graph.
pub fn canonical_graph_identity(graph: &crate::DspGraph) -> GraphIdentity {
    let value = serde_json::to_value(graph).expect("DspGraph serializes");
    canonical_value_identity(&value)
}

/// Rebind a carried ledger to repackaged bytes.
///
/// Reference rewriting (package-local sidecar names) changes the shipped
/// JSON without changing the processing the ledger accepted. Carried
/// Final rows are rebound to the repackaged payload instead of shipping
/// a stale fingerprint; rows without a binding (provisional history)
/// are untouched.
///
/// # Errors
///
/// Returns a reason when the rebound ledger does not validate.
pub fn rebind_ledger_to_repackaged_graph(
    ledger: &mut CorrectionDecisionLedger,
    repackaged_identity: &GraphIdentity,
) -> Result<(), String> {
    // The packaging caller must bind its exact rewritten payload again.
    ledger.payload_binding = None;
    for record in &mut ledger.decisions {
        if record.is_final_claim() {
            record.final_graph_identity = Some(repackaged_identity.fingerprint.clone());
        }
    }
    ledger.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn final_record(status: DecisionStatus) -> DecisionRecord {
        DecisionRecord {
            stage: DecisionStage::Final,
            final_graph_identity: Some(String::from("graph-final-1")),
            ..DecisionRecord::example(status)
        }
    }

    #[test]
    fn model_decision_ledger_all_states_roundtrip() {
        // Rust and JSON example per status; every status round-trips.
        for status in DecisionStatus::ALL {
            let record = final_record(status);
            assert!(record.validate().is_ok(), "validate {status:?}");
            let json = serde_json::to_value(&record).unwrap();
            let status_wire = serde_json::to_value(status).unwrap();
            assert_eq!(json["status"], status_wire, "JSON example {status:?}");
            let back: DecisionRecord = serde_json::from_value(json).unwrap();
            assert_eq!(back, record, "round-trip {status:?}");
            assert!(back.is_final_claim(), "final claim {status:?}");
        }
        let ledger = CorrectionDecisionLedger {
            acceptance_evidence: None,
            payload_binding: None,
            ledger_version: DECISION_LEDGER_VERSION.to_string(),
            decisions: DecisionStatus::ALL.map(final_record).into_iter().collect(),
            channel_summaries: Vec::new(),
        };
        assert!(ledger.validate().is_ok());
        let back: CorrectionDecisionLedger =
            serde_json::from_value(serde_json::to_value(&ledger).unwrap()).unwrap();
        assert_eq!(back, ledger);
    }

    #[test]
    fn model_decision_center_is_not_band() {
        // A filter center is a point: it never manufactures an interval.
        let record = DecisionRecord {
            frequency_band_hz: None,
            filter_center_hz: Some(120.0),
            ..DecisionRecord::example(DecisionStatus::Applied)
        };
        assert!(record.validate().is_ok());
        assert_eq!(record.effective_band(), None);

        let banded = DecisionRecord {
            frequency_band_hz: Some([80.0, 200.0]),
            filter_center_hz: Some(120.0),
            ..DecisionRecord::example(DecisionStatus::Applied)
        };
        assert_eq!(banded.effective_band(), Some([80.0, 200.0]));

        let bad_center = DecisionRecord {
            frequency_band_hz: None,
            filter_center_hz: Some(f64::NAN),
            ..DecisionRecord::example(DecisionStatus::Applied)
        };
        assert!(bad_center.validate().is_err());
    }

    #[test]
    fn model_decision_missing_final_identity_unverified() {
        // Provisional records stay provisional through serde: applying a
        // stage label elsewhere cannot promote them.
        let provisional = DecisionRecord::example(DecisionStatus::Applied);
        assert_eq!(provisional.stage, DecisionStage::Provisional);
        let back: DecisionRecord =
            serde_json::from_value(serde_json::to_value(&provisional).unwrap()).unwrap();
        assert_eq!(back.stage, DecisionStage::Provisional);
        assert!(!back.is_final_claim());

        // A final stage without a delivered-graph identity is unverified.
        let unbound = DecisionRecord {
            stage: DecisionStage::Final,
            final_graph_identity: None,
            ..DecisionRecord::example(DecisionStatus::Applied)
        };
        assert!(!unbound.is_final_claim());
        assert!(unbound.validate().is_err());
    }

    #[test]
    fn model_playback_capture_kind_preserved() {
        // Acoustic and simulated capture kinds round-trip distinctly.
        for kind in [
            CaptureKind::StationaryIr,
            CaptureKind::SpatialMagnitude,
            CaptureKind::DirectSound,
            CaptureKind::SimulatedBackend,
            CaptureKind::Unknown,
        ] {
            let comparison = PlaybackComparison {
                baseline_graph_identity: Some(String::from("graph-base")),
                candidate_graph_identity: Some(String::from("graph-cand")),
                stimulus_hash: Some(String::from("stim-1")),
                capture_kind: kind,
                ..PlaybackComparison::default()
            };
            let json = serde_json::to_value(&comparison).unwrap();
            let back: PlaybackComparison = serde_json::from_value(json).unwrap();
            assert_eq!(back.capture_kind, kind, "capture kind {kind:?}");
        }

        // Missing or mismatched identities stay unassessed, never promoted.
        let missing = PlaybackComparison::default();
        assert!(!missing.is_assessed());
        assert!(missing.unassessed_reason().is_some());

        let self_comparison = PlaybackComparison {
            baseline_graph_identity: Some(String::from("graph-same")),
            candidate_graph_identity: Some(String::from("graph-same")),
            stimulus_hash: Some(String::from("stim-1")),
            capture_kind: CaptureKind::SimulatedBackend,
            ..PlaybackComparison::default()
        };
        assert!(!self_comparison.is_assessed());
    }

    fn fixture_ledger(name: &str) -> CorrectionDecisionLedger {
        let path = format!("../test-data/decision_ledger/{name}");
        let contents = match name {
            "accepted.json" => include_str!("../test-data/decision_ledger/accepted.json"),
            "unchanged.json" => include_str!("../test-data/decision_ledger/unchanged.json"),
            "rejected_with_reversion.json" => {
                include_str!("../test-data/decision_ledger/rejected_with_reversion.json")
            }
            "insufficient_evidence.json" => {
                include_str!("../test-data/decision_ledger/insufficient_evidence.json")
            }
            "per_seat_gap.json" => include_str!("../test-data/decision_ledger/per_seat_gap.json"),
            "advisory.json" => include_str!("../test-data/decision_ledger/advisory.json"),
            "constrained_partial.json" => {
                include_str!("../test-data/decision_ledger/constrained_partial.json")
            }
            _ => panic!("unknown fixture {path}"),
        };
        let ledger: CorrectionDecisionLedger = serde_json::from_str(contents).unwrap();
        assert!(ledger.validate().is_ok(), "fixture {name} validates");
        let back: CorrectionDecisionLedger =
            serde_json::from_value(serde_json::to_value(&ledger).unwrap()).unwrap();
        assert_eq!(back, ledger, "fixture {name} round-trips");
        ledger
    }

    #[test]
    fn decision_ledger_canonical_fixtures_roundtrip() {
        // M3 canonical fixtures are specifications: they parse through the
        // actual Rust types, validate, and round-trip losslessly.
        let accepted = fixture_ledger("accepted.json");
        assert_eq!(accepted.decisions[0].status, DecisionStatus::Applied);
        assert!(accepted.decisions[0].is_final_claim());

        let unchanged = fixture_ledger("unchanged.json");
        assert_eq!(
            unchanged.decisions[0].status,
            DecisionStatus::AlreadyAcceptable
        );

        let reverted = fixture_ledger("rejected_with_reversion.json");
        assert_eq!(reverted.decisions.len(), 2);
        assert_eq!(reverted.decisions[0].stage, DecisionStage::Provisional);
        assert_eq!(reverted.decisions[1].status, DecisionStatus::Reverted);
        assert!(
            reverted.decisions[1]
                .supersedes_ids
                .contains(&reverted.decisions[0].decision_id.clone())
        );

        let gap = fixture_ledger("per_seat_gap.json");
        assert_eq!(gap.decisions.len(), 2);
        assert_ne!(
            gap.decisions[0].status, gap.decisions[1].status,
            "per-seat gap rows stay distinct, never merged"
        );

        let advisory = fixture_ledger("advisory.json");
        assert_eq!(advisory.decisions[0].status, DecisionStatus::Advisory);
        assert_eq!(advisory.decisions[0].effective_band(), None);

        // Constrained partial correction links to its remainder record.
        let partial = fixture_ledger("constrained_partial.json");
        assert_eq!(partial.decisions.len(), 2);
        let applied = &partial.decisions[0];
        let remainder = &partial.decisions[1];
        assert_eq!(applied.status, DecisionStatus::Applied);
        assert_eq!(remainder.status, DecisionStatus::Constrained);
        assert!(
            applied
                .related_decision_ids
                .contains(&remainder.decision_id)
        );
        assert!(
            remainder
                .related_decision_ids
                .contains(&applied.decision_id)
        );

        let _ = fixture_ledger("insufficient_evidence.json");
    }

    #[test]
    fn decision_ledger_malformed_fixtures_rejected() {
        for (name, contents) in [
            (
                "missing_ids",
                include_str!("../test-data/decision_ledger/malformed/missing_ids.json"),
            ),
            (
                "invalid_bounds",
                include_str!("../test-data/decision_ledger/malformed/invalid_bounds.json"),
            ),
            (
                "contradictory_final",
                include_str!("../test-data/decision_ledger/malformed/contradictory_final.json"),
            ),
        ] {
            let ledger: CorrectionDecisionLedger = serde_json::from_str(contents).unwrap();
            assert!(ledger.validate().is_err(), "malformed fixture {name} fails");
        }
    }

    #[test]
    fn canonical_identity_is_key_order_stable() {
        let left = serde_json::json!({"b": 1, "a": {"y": 2, "x": 1}});
        let right = serde_json::json!({"a": {"x": 1, "y": 2}, "b": 1});
        assert_eq!(
            canonical_value_identity(&left),
            canonical_value_identity(&right)
        );
        let changed = serde_json::json!({"a": {"x": 1, "y": 2}, "b": 2});
        assert_ne!(
            canonical_value_identity(&changed).fingerprint,
            canonical_value_identity(&left).fingerprint
        );
    }

    /// Rebinding refreshes Final rows to the repackaged fingerprint and
    /// leaves provisional history untouched.
    #[test]
    fn rebind_refreshes_final_rows_only() {
        let mut ledger = fixture_ledger("accepted.json");
        let flown: Vec<(String, DecisionStage, Option<String>)> = ledger
            .decisions
            .iter()
            .map(|record| {
                (
                    record.decision_id.clone(),
                    record.stage,
                    record.final_graph_identity.clone(),
                )
            })
            .collect();
        assert!(
            flown
                .iter()
                .any(|(_, stage, _)| *stage == DecisionStage::Final),
            "fixture needs a Final row"
        );
        let repackaged = GraphIdentity {
            canonical_json: String::from("{}"),
            fingerprint: String::from("abc123abc123abc1"),
        };
        rebind_ledger_to_repackaged_graph(&mut ledger, &repackaged).unwrap();
        assert!(ledger.validate().is_ok());
        for (id, stage, _) in &flown {
            let record = ledger
                .decisions
                .iter()
                .find(|record| &record.decision_id == id)
                .unwrap();
            assert_eq!(&record.stage, stage);
            if record.is_final_claim() {
                assert_eq!(
                    record.final_graph_identity.as_deref(),
                    Some("abc123abc123abc1")
                );
            }
        }
    }

    fn summary_record(
        id: &str,
        channel: &str,
        status: DecisionStatus,
        action: DecisionAction,
        stage: DecisionStage,
    ) -> DecisionRecord {
        let mut record = DecisionRecord::example(status);
        record.decision_id = id.to_owned();
        record.logical_input = channel.to_owned();
        record.physical_output = channel.to_owned();
        record.action = action;
        record.stage = stage;
        record.final_graph_identity = if stage == DecisionStage::Final {
            Some(String::from("graph-final-1"))
        } else {
            None
        };
        record
    }

    #[test]
    fn model_operational_summary_counts_only_final_equalize_scope() {
        // Delivered: Applied + AlreadyAcceptable + Constrained. Reverted,
        // InsufficientEvidence, OutsideScope, and Unresolved scope lower the
        // share; provisional history and non-EQ actions never enter it.
        let decisions = vec![
            summary_record(
                "applied",
                "L",
                DecisionStatus::Applied,
                DecisionAction::Equalize,
                DecisionStage::Final,
            ),
            summary_record(
                "acceptable",
                "L",
                DecisionStatus::AlreadyAcceptable,
                DecisionAction::Equalize,
                DecisionStage::Final,
            ),
            summary_record(
                "constrained",
                "L",
                DecisionStatus::Constrained,
                DecisionAction::Equalize,
                DecisionStage::Final,
            ),
            summary_record(
                "reverted",
                "L",
                DecisionStatus::Reverted,
                DecisionAction::Equalize,
                DecisionStage::Final,
            ),
            summary_record(
                "weak",
                "L",
                DecisionStatus::InsufficientEvidence,
                DecisionAction::Equalize,
                DecisionStage::Final,
            ),
            summary_record(
                "out",
                "L",
                DecisionStatus::OutsideScope,
                DecisionAction::Equalize,
                DecisionStage::Final,
            ),
            summary_record(
                "open",
                "L",
                DecisionStatus::Unresolved,
                DecisionAction::Equalize,
                DecisionStage::Final,
            ),
            summary_record(
                "provisional",
                "L",
                DecisionStatus::Applied,
                DecisionAction::Equalize,
                DecisionStage::Provisional,
            ),
            summary_record(
                "gain",
                "L",
                DecisionStatus::Applied,
                DecisionAction::GainAdjust,
                DecisionStage::Final,
            ),
            summary_record(
                "phase",
                "L",
                DecisionStatus::Applied,
                DecisionAction::PhaseCorrect,
                DecisionStage::Final,
            ),
            summary_record(
                "r-other",
                "R",
                DecisionStatus::Applied,
                DecisionAction::Equalize,
                DecisionStage::Final,
            ),
        ];
        let summaries = operational_summaries(&decisions);
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].channel, "L");
        assert_eq!(summaries[0].decided_equalize, 7);
        assert_eq!(summaries[0].delivered_equalize, 3);
        assert!((summaries[0].operational_response_pct.unwrap() - 100.0 * 3.0 / 7.0).abs() < 1e-12);
        assert_eq!(summaries[1].channel, "R");
        assert!((summaries[1].operational_response_pct.unwrap() - 100.0).abs() < 1e-12);
        for summary in &summaries {
            assert!(summary.validate().is_ok());
        }
    }

    #[test]
    fn model_operational_summary_excludes_superseded_scope() {
        // A rolled-back candidate stays provisional history; its final
        // Reverted successor supersedes it and counts as undecided scope.
        let mut candidate = summary_record(
            "candidate",
            "L",
            DecisionStatus::Applied,
            DecisionAction::Equalize,
            DecisionStage::Provisional,
        );
        candidate.final_graph_identity = None;
        let mut reverted = summary_record(
            "candidate-reverted",
            "L",
            DecisionStatus::Reverted,
            DecisionAction::Equalize,
            DecisionStage::Final,
        );
        reverted.supersedes_ids = vec![String::from("candidate")];
        let summaries = operational_summaries(&[candidate, reverted]);
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].decided_equalize, 1);
        assert_eq!(summaries[0].delivered_equalize, 0);
        assert_eq!(summaries[0].operational_response_pct, Some(0.0));
    }

    #[test]
    fn model_operational_summary_empty_scope_has_no_entry() {
        // No decided final equalization scope: no entry, so viewers render
        // pending instead of inventing 100%.
        let decisions = vec![
            summary_record(
                "gain",
                "L",
                DecisionStatus::Applied,
                DecisionAction::GainAdjust,
                DecisionStage::Final,
            ),
            summary_record(
                "provisional",
                "L",
                DecisionStatus::Applied,
                DecisionAction::Equalize,
                DecisionStage::Provisional,
            ),
        ];
        assert!(operational_summaries(&decisions).is_empty());
        assert!(operational_summaries(&[]).is_empty());
    }

    #[test]
    fn model_channel_summary_validation_rejects_incoherent_shares() {
        let mut summary = ChannelOperationalSummary {
            channel: String::from("L"),
            operational_response_pct: Some(50.0),
            decided_equalize: 2,
            delivered_equalize: 1,
        };
        assert!(summary.validate().is_ok());
        summary.channel = String::from("  ");
        assert!(summary.validate().is_err());
        summary.channel = String::from("L");
        summary.delivered_equalize = 3;
        assert!(summary.validate().is_err());
        summary.delivered_equalize = 1;
        summary.decided_equalize = 0;
        assert!(summary.validate().is_err());
        summary.operational_response_pct = None;
        summary.delivered_equalize = 0;
        assert!(summary.validate().is_ok());
        summary.decided_equalize = 2;
        assert!(summary.validate().is_err());
        summary.operational_response_pct = Some(f64::NAN);
        assert!(summary.validate().is_err());
    }

    #[test]
    fn model_ledger_with_summaries_roundtrips_and_validates() {
        let decisions = vec![summary_record(
            "applied",
            "L",
            DecisionStatus::Applied,
            DecisionAction::Equalize,
            DecisionStage::Final,
        )];
        let ledger = CorrectionDecisionLedger {
            acceptance_evidence: None,
            payload_binding: None,
            ledger_version: DECISION_LEDGER_VERSION.to_string(),
            decisions,
            channel_summaries: operational_summaries(&[]),
        };
        assert!(ledger.validate().is_ok());
        let mut with_summary = ledger.clone();
        with_summary.channel_summaries = operational_summaries(&ledger.decisions);
        assert!(with_summary.validate().is_ok());
        let back: CorrectionDecisionLedger =
            serde_json::from_value(serde_json::to_value(&with_summary).unwrap()).unwrap();
        assert_eq!(back, with_summary);
        assert_eq!(back.channel_summaries.len(), 1);
        // Legacy JSON without the additive field stays readable.
        let legacy = serde_json::to_value(&ledger).unwrap();
        assert!(
            legacy.get("channel_summaries").is_none()
                || legacy["channel_summaries"]
                    .as_array()
                    .is_some_and(Vec::is_empty)
        );
        let reread: CorrectionDecisionLedger = serde_json::from_value(legacy).unwrap();
        assert!(reread.channel_summaries.is_empty());
        // Duplicate channel entries fail closed.
        let mut dup = with_summary.clone();
        dup.channel_summaries.push(dup.channel_summaries[0].clone());
        assert!(dup.validate().is_err());
    }

    #[test]
    fn dsp_graph_legacy_output_fixture_reads_without_decisions() {
        // Legacy outputs carry no decision metadata: they deserialize with
        // `correction_decisions` absent and serialize without the field.
        let contents = include_str!("../test-data/decision_ledger/legacy_output.json");
        let graph: crate::DspGraph = serde_json::from_str(contents).unwrap();
        assert!(graph.correction_decisions.is_none());
        assert!(graph.validate().is_ok());
        let json = serde_json::to_value(&graph).unwrap();
        assert!(json.get("correction_decisions").is_none());

        // Attaching a ledger is additive and round-trips.
        let mut with_ledger = graph;
        with_ledger.correction_decisions = Some(fixture_ledger("accepted.json"));
        let back: crate::DspGraph =
            serde_json::from_value(serde_json::to_value(&with_ledger).unwrap()).unwrap();
        assert!(back.correction_decisions.is_some());
    }
}
