//! W2 — final decision reconciliation against the delivered graph.
//!
//! The engine records provisional K4 decisions; only workflow reconciliation
//! binds final records to the delivered graph:
//!
//! - [`canonical_graph_identity`] computes an immutable, key-order-stable
//!   identity for a [`DspGraph`](roomeq_model::DspGraph). Until the export
//!   lane publishes the canonical X1 graph hash, this workflow-local
//!   identity is the binding target; the handoff names the swap.
//! - [`reconcile_ledger`] gathers provisional records and, after rollback,
//!   structural/identity fallback, level trims, and final resource
//!   resolution, reconciles each record: retained, superseded, reverted, or
//!   unverified. Failed attempts stay in the ledger as history and can never
//!   read as delivered EQ.
//! - [`verify_final_binding`] re-checks every final delivery claim against
//!   the delivered graph. An export or package change affecting DSP
//!   invalidates the binding until verification is repeated.
//! - [`group_decision_rows`] merges adjacent rows only under the K4-exact
//!   identity/reason/evidence rules. Gaps and seat boundaries are never
//!   merged away.
//! - [`assess_output_regression`] keeps unnormalized useful-output loss
//!   visible when post-EQ display gain would hide it (F11), per measured
//!   band so unrelated-band damage is still caught (F14).
//!
//! The top-level final acceptance
//! ([`CorrectionAcceptanceReport`](roomeq_model::CorrectionAcceptanceReport))
//! stays authoritative: rejected or unchanged outputs cannot advertise
//! attempted EQ.

use roomeq_model::decision_ledger::{
    CorrectionDecisionLedger, DECISION_LEDGER_VERSION, DecisionAction, DecisionRecord,
    DecisionStage, DecisionStatus, ObservedQuantity,
};
use roomeq_model::{CorrectionAcceptanceReport, DspGraph};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::evidence_intake::WorkflowOutcome;

/// Workflow reconciliation policy version.
pub const RECONCILIATION_POLICY_VERSION: &str = "workflow-reconciliation-v1";

/// Immutable delivered-graph identity: canonical JSON plus a compact fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphIdentity {
    /// Canonical (key-order-stable) JSON of the delivered graph.
    pub canonical_json: String,
    /// FNV-1a 64-bit fingerprint of the canonical JSON, hex-encoded.
    pub fingerprint: String,
}

impl GraphIdentity {
    /// Whether this identity binds the given graph.
    pub fn binds(&self, graph: &DspGraph) -> bool {
        canonical_graph_identity(graph) == *self
    }
}

fn sort_canonical(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let sorted: BTreeMap<String, serde_json::Value> = map
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

/// Compute the immutable identity of a delivered graph.
///
/// Object keys are sorted before serialization, so two graphs with identical
/// content share one identity regardless of insertion order. This is the
/// workflow-local binding target until the export lane publishes the
/// canonical X1 graph hash (handoff); the comparison semantics (exact
/// canonical equality) already match that contract.
pub fn canonical_graph_identity(graph: &DspGraph) -> GraphIdentity {
    let value = serde_json::to_value(graph).expect("DspGraph serializes");
    let canonical_json = serde_json::to_string(&sort_canonical(value)).expect("canonical JSON");
    let fingerprint = fnv1a_hex(&canonical_json);
    GraphIdentity {
        canonical_json,
        fingerprint,
    }
}

/// Events consumed by reconciliation after the provisional stages ran.
#[derive(Debug, Clone, Default)]
pub struct ReconciliationEvents {
    /// Decision IDs rolled back: each gains a final `Reverted` record.
    pub rolled_back_ids: Vec<String>,
    /// Physical outputs that fell back to the identity chain.
    pub fallback_outputs: Vec<String>,
    /// Level trims applied after the provisional stages, in dB per output.
    pub post_stage_trims_db: Vec<(String, f64)>,
    /// Final resource revision (export/package build); bound into the ledger.
    pub resource_revision: Option<String>,
    /// Final authoritative acceptance for the delivered graph.
    pub final_acceptance: Option<CorrectionAcceptanceReport>,
}

fn observed(name: &str, value: f64, unit: &str) -> ObservedQuantity {
    ObservedQuantity {
        name: name.to_string(),
        value,
        unit: unit.to_string(),
    }
}

/// Reconcile provisional K4 records against the delivered graph.
///
/// Every provisional record is retained as history. Final records are bound
/// to `delivered_identity`:
/// - provisional records untouched by any event are retained as final claims
///   with their status preserved;
/// - rolled-back IDs gain a final `Reverted` record that supersedes the
///   attempted record (F10: the attempted benefit is never delivered);
/// - identity-fallback outputs gain a final `AlreadyAcceptable` record: the
///   output is unchanged, and any provisional benefit on that output is
///   superseded, never advertised;
/// - post-stage trims are recorded as linked final `GainAdjust` observations
///   so the delivered chain (with trims) matches the ledger.
///
/// Records that cannot be verified against the delivered graph stay
/// provisional (`Unresolved`): reconciled output never invents a final claim.
pub fn reconcile_ledger(
    provisional: &[DecisionRecord],
    delivered_identity: &GraphIdentity,
    events: &ReconciliationEvents,
) -> CorrectionDecisionLedger {
    let mut decisions: Vec<DecisionRecord> = Vec::new();
    let identity = delivered_identity.fingerprint.clone();

    for record in provisional {
        let mut retained = record.clone();
        // Only delivery claims become final: applied (or constrained
        // partial) corrections bound to the delivered graph. Attempts that
        // cannot stand as delivery (insufficient evidence, outside scope,
        // advisory, unresolved) stay provisional history — the model
        // contract forbids a final stage without a delivered-graph
        // identity, and history must never read as delivered EQ.
        if matches!(
            record.status,
            DecisionStatus::Applied
                | DecisionStatus::AlreadyAcceptable
                | DecisionStatus::Constrained
        ) {
            retained.stage = DecisionStage::Final;
            retained.final_graph_identity = Some(identity.clone());
        } else {
            retained.stage = DecisionStage::Provisional;
            retained.final_graph_identity = None;
        }
        decisions.push(retained);
    }

    // A rolled-back or fallback-superseded candidate must not remain a
    // final delivery claim: demote the retained copy to provisional
    // history so the reversion/fallback record alone speaks for delivery.
    for demoted in &events.rolled_back_ids {
        for retained in decisions.iter_mut() {
            if &retained.decision_id == demoted {
                retained.stage = DecisionStage::Provisional;
                retained.final_graph_identity = None;
            }
        }
    }
    for output in &events.fallback_outputs {
        for retained in decisions.iter_mut() {
            if retained.physical_output == *output
                && matches!(
                    retained.status,
                    DecisionStatus::Applied | DecisionStatus::Constrained
                )
            {
                // Superseded by the identity-fallback record below, which
                // lists this candidate in its own `supersedes_ids`.
                retained.stage = DecisionStage::Provisional;
                retained.final_graph_identity = None;
            }
        }
    }

    for rolled_back in &events.rolled_back_ids {
        let superseded = provisional
            .iter()
            .find(|record| &record.decision_id == rolled_back);
        let (logical_input, physical_output, measurement_refs, seat_refs, evidence_refs) =
            superseded.map_or_else(
                || {
                    (
                        String::from("unknown"),
                        String::from("unknown"),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    )
                },
                |record| {
                    (
                        record.logical_input.clone(),
                        record.physical_output.clone(),
                        record.measurement_refs.clone(),
                        record.seat_refs.clone(),
                        record.evidence_refs.clone(),
                    )
                },
            );
        decisions.push(DecisionRecord {
            decision_id: format!("{rolled_back}-reverted"),
            ledger_version: DECISION_LEDGER_VERSION.to_string(),
            stage: DecisionStage::Final,
            logical_input,
            physical_output,
            measurement_refs,
            seat_refs,
            frequency_band_hz: superseded.and_then(|record| record.frequency_band_hz),
            filter_center_hz: None,
            action: DecisionAction::Equalize,
            status: DecisionStatus::Reverted,
            reason_codes: vec![String::from("rollback_after_acceptance_failure")],
            observed: vec![observed("delivered_correction_db", 0.0, "db")],
            limits: Vec::new(),
            evidence_refs,
            confidence: roomeq_model::AssessmentConfidence::High,
            related_decision_ids: Vec::new(),
            supersedes_ids: vec![rolled_back.clone()],
            // A reversion is a final reconciliation fact about the
            // delivered graph (which carries no correction from the
            // rolled-back candidate), so it binds the delivery identity
            // without claiming delivered EQ.
            final_graph_identity: Some(identity.clone()),
        });
    }

    for output in &events.fallback_outputs {
        decisions.push(DecisionRecord {
            decision_id: format!("identity-fallback-{output}"),
            ledger_version: DECISION_LEDGER_VERSION.to_string(),
            stage: DecisionStage::Final,
            logical_input: output.clone(),
            physical_output: output.clone(),
            measurement_refs: Vec::new(),
            seat_refs: Vec::new(),
            frequency_band_hz: None,
            filter_center_hz: None,
            action: DecisionAction::Equalize,
            status: DecisionStatus::AlreadyAcceptable,
            reason_codes: vec![String::from("structural_identity_fallback")],
            observed: vec![observed("delivered_correction_db", 0.0, "db")],
            limits: Vec::new(),
            evidence_refs: Vec::new(),
            confidence: roomeq_model::AssessmentConfidence::High,
            related_decision_ids: Vec::new(),
            supersedes_ids: provisional
                .iter()
                .filter(|record| &record.physical_output == output)
                .map(|record| record.decision_id.clone())
                .collect(),
            final_graph_identity: Some(identity.clone()),
        });
    }

    for (output, trim_db) in &events.post_stage_trims_db {
        if !trim_db.is_finite() {
            decisions.push(DecisionRecord {
                decision_id: format!("trim-{output}-unverified"),
                ledger_version: DECISION_LEDGER_VERSION.to_string(),
                stage: DecisionStage::Provisional,
                logical_input: output.clone(),
                physical_output: output.clone(),
                measurement_refs: Vec::new(),
                seat_refs: Vec::new(),
                frequency_band_hz: None,
                filter_center_hz: None,
                action: DecisionAction::GainAdjust,
                status: DecisionStatus::Unresolved,
                reason_codes: vec![String::from("nonfinite_post_stage_trim")],
                observed: Vec::new(),
                limits: Vec::new(),
                evidence_refs: Vec::new(),
                confidence: roomeq_model::AssessmentConfidence::Unknown,
                related_decision_ids: Vec::new(),
                supersedes_ids: Vec::new(),
                final_graph_identity: None,
            });
            continue;
        }
        decisions.push(DecisionRecord {
            decision_id: format!("trim-{output}"),
            ledger_version: DECISION_LEDGER_VERSION.to_string(),
            stage: DecisionStage::Final,
            logical_input: output.clone(),
            physical_output: output.clone(),
            measurement_refs: Vec::new(),
            seat_refs: Vec::new(),
            frequency_band_hz: None,
            filter_center_hz: None,
            action: DecisionAction::GainAdjust,
            status: DecisionStatus::Applied,
            reason_codes: vec![String::from("post_stage_level_trim")],
            observed: vec![observed("post_stage_trim_db", *trim_db, "db")],
            limits: Vec::new(),
            evidence_refs: Vec::new(),
            confidence: roomeq_model::AssessmentConfidence::High,
            related_decision_ids: Vec::new(),
            supersedes_ids: Vec::new(),
            final_graph_identity: Some(identity.clone()),
        });
    }

    CorrectionDecisionLedger {
        ledger_version: DECISION_LEDGER_VERSION.to_string(),
        decisions,
    }
}

/// Re-check every final delivery claim against the delivered graph.
///
/// An export or package change affecting DSP produces a new identity, and
/// every stale final claim fails here until reconciliation and verification
/// are repeated against the new graph.
pub fn verify_final_binding(
    ledger: &CorrectionDecisionLedger,
    delivered_identity: &GraphIdentity,
) -> Result<(), String> {
    ledger.validate()?;
    for record in &ledger.decisions {
        if record.stage == DecisionStage::Final
            && matches!(
                record.status,
                DecisionStatus::Applied
                    | DecisionStatus::AlreadyAcceptable
                    | DecisionStatus::Constrained
            )
        {
            match &record.final_graph_identity {
                Some(bound) if bound == &delivered_identity.fingerprint => {}
                _ => {
                    return Err(format!(
                        "final record '{}' is not bound to the delivered graph '{}'; \
                         repeat reconciliation and verification",
                        record.decision_id, delivered_identity.fingerprint
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Merge adjacent rows only under the K4-exact grouping rules.
///
/// Two rows merge only when channel/seat scope, action, status, reasons,
/// constraints, evidence, stage, and graph binding are identical, and their
/// explicit bands overlap or touch. A gap between bands, different seats, or
/// any differing reason/evidence keeps rows separate. Merged rows take the
/// band hull and a deterministic joined decision ID; linkage (`related`,
/// `supersedes`) is unioned so no history is lost.
pub fn group_decision_rows(records: &[DecisionRecord]) -> Vec<DecisionRecord> {
    let mut grouped: Vec<DecisionRecord> = Vec::new();
    for record in records {
        let mut absorbed = false;
        for existing in grouped.iter_mut() {
            if group_key(existing) == group_key(record)
                && let Some(hull) =
                    mergeable_band(existing.frequency_band_hz, record.frequency_band_hz)
            {
                existing.frequency_band_hz = hull;
                existing.decision_id = format!("{}+{}", existing.decision_id, record.decision_id);
                for related in record
                    .related_decision_ids
                    .iter()
                    .chain(record.supersedes_ids.iter())
                {
                    if !existing.related_decision_ids.contains(related)
                        && !existing.supersedes_ids.contains(related)
                    {
                        existing.related_decision_ids.push(related.clone());
                    }
                }
                for observed in &record.observed {
                    if !existing.observed.contains(observed) {
                        existing.observed.push(observed.clone());
                    }
                }
                absorbed = true;
                break;
            }
        }
        if !absorbed {
            grouped.push(record.clone());
        }
    }
    grouped
}

type GroupKey = (
    String,
    String,
    Vec<String>,
    DecisionAction,
    DecisionStatus,
    Vec<String>,
    Vec<(String, f64, String)>,
    Vec<String>,
    DecisionStage,
    Option<String>,
    Option<f64>,
);

fn group_key(record: &DecisionRecord) -> GroupKey {
    let mut seats = record.seat_refs.clone();
    seats.sort();
    let mut reasons = record.reason_codes.clone();
    reasons.sort();
    let mut evidence = record.evidence_refs.clone();
    evidence.sort();
    let mut limits: Vec<(String, f64, String)> = record
        .limits
        .iter()
        .map(|limit| (limit.name.clone(), limit.value, limit.unit.clone()))
        .collect();
    limits.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then(left.1.total_cmp(&right.1))
            .then(left.2.cmp(&right.2))
    });
    (
        record.logical_input.clone(),
        record.physical_output.clone(),
        seats,
        record.action,
        record.status,
        reasons,
        limits,
        evidence,
        record.stage,
        record.final_graph_identity.clone(),
        record.filter_center_hz,
    )
}

/// Band hull when two explicit bands overlap or touch; `None` across a gap.
///
/// Bandless rows merge only with bandless rows; a banded row never merges
/// with a bandless one (different scope), and center-only rows merge only
/// through the shared group key (same center).
fn mergeable_band(first: Option<[f64; 2]>, second: Option<[f64; 2]>) -> Option<Option<[f64; 2]>> {
    match (first, second) {
        (None, None) => Some(None),
        (Some(_), None) | (None, Some(_)) => None,
        (Some([lo_a, hi_a]), Some([lo_b, hi_b])) => {
            let touches = lo_b <= hi_a + 1e-9 && lo_a <= hi_b + 1e-9;
            if touches {
                Some(Some([lo_a.min(lo_b), hi_a.max(hi_b)]))
            } else {
                None
            }
        }
    }
}

/// Useful-output regression assessment for one measured band.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputRegressionAssessment {
    /// Band label, e.g. `"bass"` or `"upper_measured"`.
    pub band: String,
    /// Unnormalized loss in dB (positive means the EQ lowered output).
    /// Always retained, even when display gain hides it.
    pub unnormalized_loss_db: f64,
    /// Display gain applied after EQ in dB.
    pub display_gain_db: f64,
    /// Violation reason when display gain hides a real regression.
    pub violation: Option<String>,
}

/// Keep unnormalized useful-output loss visible (F11/F14).
///
/// `bands` carries per-band `(label, pre_output_db, post_output_unnormalized_db)`
/// triples over the *measured* bands, so damage outside the corrected band
/// is still caught (F14). A band violates when its unnormalized loss exceeds
/// `max_loss_db` while the displayed level (post + display gain) looks
/// intact — the exact "normalization hides it" failure (F11).
pub fn assess_output_regression(
    bands: &[(&str, f64, f64)],
    display_gain_db: f64,
    max_loss_db: f64,
) -> Result<Vec<OutputRegressionAssessment>, String> {
    if !display_gain_db.is_finite() || !max_loss_db.is_finite() || max_loss_db < 0.0 {
        return Err(String::from(
            "display gain and max loss must be finite; max loss non-negative",
        ));
    }
    let mut assessments = Vec::new();
    for (band, pre_db, post_db) in bands {
        if !pre_db.is_finite() || !post_db.is_finite() {
            return Err(format!("band '{band}' outputs must be finite"));
        }
        let unnormalized_loss_db = pre_db - post_db;
        let displayed_loss_db = unnormalized_loss_db - display_gain_db;
        let violation = if unnormalized_loss_db > max_loss_db && displayed_loss_db <= max_loss_db {
            Some(format!(
                "band '{band}' lost {unnormalized_loss_db:.2} dB of useful output \
                 but {display_gain_db:.2} dB display gain hides it"
            ))
        } else if unnormalized_loss_db > max_loss_db {
            Some(format!(
                "band '{band}' lost {unnormalized_loss_db:.2} dB of useful output \
                 beyond the {max_loss_db:.2} dB limit"
            ))
        } else {
            None
        };
        assessments.push(OutputRegressionAssessment {
            band: (*band).to_string(),
            unnormalized_loss_db,
            display_gain_db,
            violation,
        });
    }
    Ok(assessments)
}

/// Final result document handed to root R3 and the CLI.
///
/// `outcome` uses the four stable states (accepted/unchanged/rejected/
/// unknown). Unknown covers insufficient evidence; failures stay in the
/// ledger as history but never read as delivered EQ.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowResultDocument {
    /// R3 vocabulary: accepted, unchanged, rejected, unknown.
    pub outcome: WorkflowOutcome,
    /// Fingerprint of the delivered graph this result describes.
    pub graph_fingerprint: String,
    /// Reconciliation policy version.
    pub reconciliation_policy: String,
    /// Final delivery-claim records (bound to the graph).
    pub applied_final_ids: Vec<String>,
    /// Historical records (provisional attempts, reversions, trims).
    pub history_ids: Vec<String>,
    /// Unassessed items with reasons (missing/mismatched evidence).
    pub unassessed: Vec<String>,
}

impl WorkflowResultDocument {
    /// Build the result from a reconciled ledger and its final acceptance.
    /// Rejected or unchanged acceptance can never yield an accepted result,
    /// no matter how many provisional records claim a benefit.
    pub fn from_ledger(
        ledger: &CorrectionDecisionLedger,
        graph: &GraphIdentity,
        acceptance: &CorrectionAcceptanceReport,
    ) -> Self {
        let mut report = acceptance.clone();
        report.refresh_outcome();
        let mut outcome = WorkflowOutcome::from_room_outcome(report.derived_outcome());
        let mut applied_final_ids = Vec::new();
        let mut history_ids = Vec::new();
        let mut unassessed = Vec::new();
        for record in &ledger.decisions {
            if record.is_final_claim()
                && matches!(
                    record.status,
                    DecisionStatus::Applied
                        | DecisionStatus::AlreadyAcceptable
                        | DecisionStatus::Constrained
                )
            {
                applied_final_ids.push(record.decision_id.clone());
            } else {
                history_ids.push(record.decision_id.clone());
            }
            if record.status == DecisionStatus::Unresolved
                || record.status == DecisionStatus::InsufficientEvidence
            {
                unassessed.push(format!(
                    "{}: {}",
                    record.decision_id,
                    record
                        .reason_codes
                        .first()
                        .cloned()
                        .unwrap_or_else(|| String::from("unassessed"))
                ));
            }
        }
        // The acceptance record is authoritative: without an accepted
        // decision, attempted EQ is history, never delivery.
        if outcome == WorkflowOutcome::Accepted && applied_final_ids.is_empty() {
            outcome = WorkflowOutcome::Unknown;
            unassessed.push(String::from(
                "acceptance claims delivery but no final record is bound to the graph",
            ));
        }
        WorkflowResultDocument {
            outcome,
            graph_fingerprint: graph.fingerprint.clone(),
            reconciliation_policy: RECONCILIATION_POLICY_VERSION.to_string(),
            applied_final_ids,
            history_ids,
            unassessed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::decision_ledger::{DecisionRecord, DecisionStatus};

    fn provisional_applied(id: &str, output: &str, band: Option<[f64; 2]>) -> DecisionRecord {
        DecisionRecord {
            decision_id: id.to_string(),
            ledger_version: DECISION_LEDGER_VERSION.to_string(),
            stage: DecisionStage::Provisional,
            logical_input: String::from("stereo"),
            physical_output: output.to_string(),
            measurement_refs: vec![String::from("meas-1")],
            seat_refs: vec![String::from("seat-a")],
            frequency_band_hz: band,
            filter_center_hz: None,
            action: DecisionAction::Equalize,
            status: DecisionStatus::Applied,
            reason_codes: vec![String::from("target_fit")],
            observed: vec![observed("improvement_db", 4.0, "db")],
            limits: Vec::new(),
            evidence_refs: vec![String::from("ev-1")],
            confidence: roomeq_model::AssessmentConfidence::Moderate,
            related_decision_ids: Vec::new(),
            supersedes_ids: Vec::new(),
            final_graph_identity: None,
        }
    }

    fn delivered_graph() -> (DspGraph, GraphIdentity) {
        let mut graph = DspGraph::new("test");
        graph.add_channel(
            "left",
            vec![roomeq_model::contracts::Plugin {
                kind: String::from("eq"),
                parameters: serde_json::json!({}),
            }],
        );
        let identity = canonical_graph_identity(&graph);
        (graph, identity)
    }

    fn flat_acceptance_curve() -> autoeq_core::Curve {
        autoeq_core::Curve {
            freq: ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 1000.0_f64.log10(), 32),
            spl: ndarray::Array1::from_elem(32, 80.0),
            phase: None,
            ..Default::default()
        }
    }

    fn accepted_report() -> CorrectionAcceptanceReport {
        let curve = flat_acceptance_curve();
        let mut report = roomeq_engine::quality::evaluate_correction_acceptance(
            &curve,
            &curve,
            &curve,
            None,
            roomeq_model::CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .unwrap();
        report.accepted = true;
        report.decision = roomeq_model::CorrectionDecision::Accepted;
        report.refresh_outcome();
        report
    }

    fn rejected_report() -> CorrectionAcceptanceReport {
        let curve = flat_acceptance_curve();
        let mut report = roomeq_engine::quality::evaluate_correction_acceptance(
            &curve,
            &curve,
            &curve,
            None,
            roomeq_model::CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .unwrap();
        report.accepted = false;
        report.decision = roomeq_model::CorrectionDecision::Rejected;
        report
            .violations
            .push(String::from("post_worst_abs_residual_db"));
        report.refresh_outcome();
        report
    }

    #[test]
    fn workflow_final_ledger_matches_delivered_graph() {
        let (graph, identity) = delivered_graph();
        // Identity is content-stable: key order and rebuilds do not move it.
        let rebuilt = delivered_graph().0;
        assert_eq!(canonical_graph_identity(&rebuilt), identity);
        assert!(identity.binds(&graph));
        assert_eq!(identity.fingerprint.len(), 16);

        let provisional = vec![
            provisional_applied("dec-eq-left", "left", Some([40.0, 400.0])),
            DecisionRecord {
                decision_id: String::from("dec-phase-left"),
                status: DecisionStatus::InsufficientEvidence,
                reason_codes: vec![String::from("no_timing_reference")],
                ..provisional_applied("dec-phase-left", "left", Some([400.0, 4000.0]))
            },
        ];
        let events = ReconciliationEvents {
            post_stage_trims_db: vec![(String::from("left"), -1.5)],
            ..Default::default()
        };
        let ledger = reconcile_ledger(&provisional, &identity, &events);
        assert!(ledger.validate().is_ok());
        // Every applied final record refers to the delivered graph.
        for record in &ledger.decisions {
            if record.is_final_claim()
                && matches!(
                    record.status,
                    DecisionStatus::Applied
                        | DecisionStatus::AlreadyAcceptable
                        | DecisionStatus::Constrained
                )
            {
                assert_eq!(
                    record.final_graph_identity.as_deref(),
                    Some(identity.fingerprint.as_str())
                );
            }
        }
        assert!(verify_final_binding(&ledger, &identity).is_ok());
        // JSON round trip retains final identity, evidence refs, history.
        let json = serde_json::to_value(&ledger).unwrap();
        let back: CorrectionDecisionLedger = serde_json::from_value(json).unwrap();
        assert_eq!(back, ledger);
        // An export change invalidates the binding until re-verification.
        let mut changed = graph.clone();
        changed.add_channel(
            "right",
            vec![roomeq_model::contracts::Plugin {
                kind: String::from("gain"),
                parameters: serde_json::json!({}),
            }],
        );
        let changed_identity = canonical_graph_identity(&changed);
        assert_ne!(changed_identity.fingerprint, identity.fingerprint);
        assert!(!identity.binds(&changed));
        assert!(verify_final_binding(&ledger, &changed_identity).is_err());
        // The result document binds the same fingerprint.
        let result = WorkflowResultDocument::from_ledger(&ledger, &identity, &accepted_report());
        assert_eq!(result.outcome, WorkflowOutcome::Accepted);
        assert_eq!(result.graph_fingerprint, identity.fingerprint);
        assert!(
            result
                .applied_final_ids
                .contains(&String::from("dec-eq-left"))
        );
        assert!(result.applied_final_ids.contains(&String::from("trim-left")));
        assert!(result.history_ids.contains(&String::from("dec-phase-left")));
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["outcome"], serde_json::json!("accepted"));
        let back: WorkflowResultDocument = serde_json::from_value(json).unwrap();
        assert_eq!(back, result);
    }

    #[test]
    fn workflow_reversion_supersedes_applied_candidate() {
        // F10: an applied candidate followed by rollback reconciles to a
        // reversion. The final ledger and result show the reversion, never
        // the attempted benefit as delivered.
        let (_, identity) = delivered_graph();
        let provisional = vec![provisional_applied(
            "dec-candidate",
            "left",
            Some([40.0, 400.0]),
        )];
        let events = ReconciliationEvents {
            rolled_back_ids: vec![String::from("dec-candidate")],
            final_acceptance: Some(rejected_report()),
            ..Default::default()
        };
        let ledger = reconcile_ledger(&provisional, &identity, &events);
        assert!(ledger.validate().is_ok());
        // History keeps the attempt as provisional, never as delivery ...
        assert!(
            ledger
                .decisions
                .iter()
                .any(|record| record.decision_id == "dec-candidate"
                    && record.stage == DecisionStage::Provisional
                    && record.status == DecisionStatus::Applied
                    && record.final_graph_identity.is_none())
        );
        // ... and the superseding reversion binds the delivered graph
        // without claiming delivered EQ.
        let reversion = ledger
            .decisions
            .iter()
            .find(|record| record.status == DecisionStatus::Reverted)
            .expect("rollback yields a reversion record");
        assert_eq!(
            reversion.supersedes_ids,
            vec![String::from("dec-candidate")]
        );
        assert_eq!(reversion.stage, DecisionStage::Final);
        // Rejected acceptance can never read as accepted delivery.
        let result = WorkflowResultDocument::from_ledger(&ledger, &identity, &rejected_report());
        assert_eq!(result.outcome, WorkflowOutcome::Rejected);
        assert!(
            !result
                .applied_final_ids
                .contains(&String::from("dec-candidate"))
        );
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["outcome"], serde_json::json!("rejected"));
    }

    #[test]
    fn workflow_identity_fallback_reports_unchanged() {
        let (_, identity) = delivered_graph();
        let provisional = vec![provisional_applied(
            "dec-candidate",
            "left",
            Some([40.0, 400.0]),
        )];
        let events = ReconciliationEvents {
            fallback_outputs: vec![String::from("left")],
            ..Default::default()
        };
        let ledger = reconcile_ledger(&provisional, &identity, &events);
        assert!(ledger.validate().is_ok());
        let fallback = ledger
            .decisions
            .iter()
            .find(|record| record.decision_id == "identity-fallback-left")
            .expect("fallback record exists");
        assert_eq!(fallback.status, DecisionStatus::AlreadyAcceptable);
        assert!(
            fallback
                .supersedes_ids
                .contains(&String::from("dec-candidate"))
        );
        assert!(verify_final_binding(&ledger, &identity).is_ok());
        // The committed fallback vocabulary reports unchanged, and so does R3.
        let mut acceptance = accepted_report();
        acceptance.decision = roomeq_model::CorrectionDecision::IdentityFallback;
        acceptance.accepted = false;
        acceptance.refresh_outcome();
        assert_eq!(
            acceptance.derived_outcome(),
            roomeq_model::RoomEqOutcome::Unchanged
        );
        let result = WorkflowResultDocument::from_ledger(&ledger, &identity, &acceptance);
        assert_eq!(result.outcome, WorkflowOutcome::Unchanged);
    }

    #[test]
    fn workflow_post_eq_gain_does_not_hide_output_regression() {
        // F11: EQ lowers useful output by 3 dB while 3 dB of display gain
        // hides it. The unnormalized loss stays in the assessment and the
        // violation is flagged.
        let hidden = assess_output_regression(&[("broadband", 100.0, 97.0)], 3.0, 0.25).unwrap();
        assert_eq!(hidden.len(), 1);
        assert!((hidden[0].unnormalized_loss_db - 3.0).abs() < 1e-9);
        assert!(hidden[0].violation.is_some());
        // Genuine improvement with no hidden loss passes cleanly.
        let improved = assess_output_regression(&[("broadband", 97.0, 100.0)], 0.0, 0.25).unwrap();
        assert!(improved[0].violation.is_none());
        // F14: bass-only correction improves bass but damages the unrelated
        // upper measured band. Evaluation retains the upper band and catches
        // the damage even though the corrected band improved.
        let bands = assess_output_regression(
            &[("bass", 100.0, 102.0), ("upper_measured", 100.0, 98.0)],
            0.0,
            0.25,
        )
        .unwrap();
        assert!(bands[0].violation.is_none());
        assert_eq!(bands[1].band, "upper_measured");
        assert!((bands[1].unnormalized_loss_db - 2.0).abs() < 1e-9);
        assert!(bands[1].violation.is_some());
        assert!(assess_output_regression(&[("x", f64::NAN, 0.0)], 0.0, 0.25).is_err());
    }

    #[test]
    fn workflow_ledger_grouping_preserves_gaps_and_seats() {
        // K4-exact grouping: identical channel/seat/status/reason/evidence
        // with touching bands merge; gaps and seat boundaries never merge.
        let base = provisional_applied("dec-a", "left", Some([40.0, 100.0]));
        let touching = DecisionRecord {
            decision_id: String::from("dec-b"),
            frequency_band_hz: Some([100.0, 400.0]),
            ..base.clone()
        };
        let gapped = DecisionRecord {
            decision_id: String::from("dec-c"),
            frequency_band_hz: Some([500.0, 800.0]),
            ..base.clone()
        };
        let other_seat = DecisionRecord {
            decision_id: String::from("dec-d"),
            seat_refs: vec![String::from("seat-b")],
            frequency_band_hz: Some([100.0, 400.0]),
            ..base.clone()
        };
        let other_reason = DecisionRecord {
            decision_id: String::from("dec-e"),
            reason_codes: vec![String::from("different_reason")],
            frequency_band_hz: Some([100.0, 400.0]),
            ..base.clone()
        };
        let center_only = DecisionRecord {
            decision_id: String::from("dec-f"),
            frequency_band_hz: None,
            filter_center_hz: Some(120.0),
            ..base.clone()
        };
        let grouped = group_decision_rows(&[base, touching, gapped, other_seat, other_reason]);
        // Touching pair merges to the hull; gap, seat, and reason rows stay.
        assert_eq!(grouped.len(), 4);
        let merged = grouped
            .iter()
            .find(|record| record.decision_id.contains('+'))
            .expect("touching rows merge");
        assert_eq!(merged.frequency_band_hz, Some([40.0, 400.0]));
        assert!(
            grouped.iter().any(|record| record.decision_id == "dec-c"),
            "unassessed gap band stays visible"
        );
        assert!(
            grouped
                .iter()
                .any(|record| record.seat_refs == vec![String::from("seat-b")]),
            "seat boundary never merges"
        );
        // A banded row never merges with a center-only (bandless) row.
        let mixed = group_decision_rows(&[
            provisional_applied("dec-g", "left", Some([40.0, 400.0])),
            center_only,
        ]);
        assert_eq!(mixed.len(), 2);
    }
}
