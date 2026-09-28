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
    DecisionStage, DecisionStatus, ObservedQuantity, operational_summaries,
};
use roomeq_model::{CorrectionAcceptanceReport, DspGraph};
use serde::{Deserialize, Serialize};

use crate::evidence_intake::WorkflowOutcome;

mod delivered_gains;

/// Workflow reconciliation policy version.
pub const RECONCILIATION_POLICY_VERSION: &str = "workflow-reconciliation-v1";

/// Capture serialized processing before final workflow mutation stages.
pub(crate) fn processing_snapshot(
    result: &roomeq_engine::room_result::RoomOptimizationResult,
) -> Result<serde_json::Value, String> {
    let output = result.to_dsp_chain_output();
    // Reuse the canonical channel-control projection (plugins and ordered
    // driver names/indices/plugins), excluding plots and diagnostic snapshots.
    let channels: std::collections::BTreeMap<_, _> = output
        .channels
        .iter()
        .map(|(name, chain)| {
            roomeq_model::joint_sub_report::joint_sub_processing_identity(chain)
                .map(|identity| (name, identity))
        })
        .collect::<Result<_, _>>()
        .map_err(|error| format!("channel controls do not serialize: {error}"))?;
    serde_json::to_value((output.global_plugins, channels))
        .map_err(|error| format!("processing snapshot does not serialize: {error}"))
}

/// Refuse a delivered graph that carries negative physical delays.
///
/// Amendment 4 ordering enforcement: delay compilation is the single graph
/// boundary that turns relative advances into causal delays, and every
/// production path compiles before this ledger runs. Any negative `delay_ms`
/// reaching finalization means compile never ran or a later stage regressed
/// it; either way the pre-compile splice verdict does not cover the shipped
/// graph, so fail closed. Mirrors the native-result audit's strict `< 0.0`
/// rule, and additionally rejects non-finite values.
fn refuse_negative_delays(
    result: &roomeq_engine::room_result::RoomOptimizationResult,
) -> Result<(), String> {
    let mut channel_names: Vec<&String> = result.channels.keys().collect();
    channel_names.sort();
    for name in channel_names {
        let chain = &result.channels[name];
        for (plugin_index, plugin) in chain.plugins.iter().enumerate() {
            if plugin.plugin_type != "delay" {
                continue;
            }
            let value = plugin
                .parameters
                .get("delay_ms")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    format!("channel '{name}' delay plugin {plugin_index} has no numeric delay_ms")
                })?;
            if !value.is_finite() {
                return Err(format!(
                    "channel '{name}' delay plugin {plugin_index} has non-finite delay {value}"
                ));
            }
            if value < 0.0 {
                return Err(format!(
                    "channel '{name}' delay plugin {plugin_index} ships unresolved causal delay {value} ms"
                ));
            }
        }
        if let Some(drivers) = chain.drivers.as_ref() {
            for (driver_index, driver) in drivers.iter().enumerate() {
                for (plugin_index, plugin) in driver.plugins.iter().enumerate() {
                    if plugin.plugin_type != "delay" {
                        continue;
                    }
                    let value = plugin
                        .parameters
                        .get("delay_ms")
                        .and_then(serde_json::Value::as_f64)
                        .ok_or_else(|| {
                            format!(
                                "channel '{name}' driver {driver_index} delay plugin {plugin_index} has no numeric delay_ms"
                            )
                        })?;
                    if !value.is_finite() {
                        return Err(format!(
                            "channel '{name}' driver {driver_index} delay plugin {plugin_index} has non-finite delay {value}"
                        ));
                    }
                    if value < 0.0 {
                        return Err(format!(
                            "channel '{name}' driver {driver_index} delay plugin {plugin_index} ships unresolved causal delay {value} ms"
                        ));
                    }
                }
            }
        }
    }
    if let Some(graph) = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
    {
        for (route_index, route) in graph.routes.iter().enumerate() {
            if !route.delay_ms.is_finite() {
                return Err(format!(
                    "routing_graph.routes[{route_index}] has non-finite delay {}",
                    route.delay_ms
                ));
            }
            if route.delay_ms < 0.0 {
                return Err(format!(
                    "routing_graph.routes[{route_index}] ships unresolved causal delay {} ms",
                    route.delay_ms
                ));
            }
        }
    }
    Ok(())
}

/// Finalize public workflow decisions after all processing and report mutations.
pub(crate) fn finalize_result_ledger(
    result: &mut roomeq_engine::room_result::RoomOptimizationResult,
    assessed_processing: &serde_json::Value,
    sample_rate_hz: f64,
) -> Result<(), String> {
    refuse_negative_delays(result)?;
    // Inner safety stages can remove phase processing before the outer processing
    // snapshot is taken. Reconcile the operation's actual emitted FIR reference,
    // not just the later snapshot or a generic acceptance status.
    let mut rolled_back_ids = Vec::new();
    for record in &mut result.metadata.provisional_decisions {
        if record.action != DecisionAction::PhaseCorrect || record.status != DecisionStatus::Applied
        {
            continue;
        }
        let expected: Vec<_> = record
            .evidence_refs
            .iter()
            .filter_map(|value| value.strip_prefix("phase-fir:"))
            .collect();
        if expected.is_empty() {
            continue;
        }
        let references: Vec<_> = result
            .channels
            .get(&record.physical_output)
            .into_iter()
            .flat_map(|channel| {
                channel.plugins.iter().chain(
                    channel
                        .drivers
                        .iter()
                        .flatten()
                        .flat_map(|driver| &driver.plugins),
                )
            })
            .filter(|plugin| plugin.plugin_type == "convolution")
            .map(|plugin| {
                plugin
                    .parameters
                    .get("ir_file")
                    .and_then(serde_json::Value::as_str)
            })
            .collect();
        let branches_match = record
            .evidence_refs
            .iter()
            .filter_map(|value| value.strip_prefix("phase-fir-driver:"))
            .all(|binding| {
                binding.split_once(':').is_some_and(|(index, filename)| {
                    index.parse::<usize>().ok().is_some_and(|index| {
                        result
                            .channels
                            .get(&record.physical_output)
                            .and_then(|channel| channel.drivers.as_ref())
                            .is_some_and(|drivers| {
                                drivers.iter().any(|driver| {
                                    driver.index == index
                                        && driver.plugins.iter().any(|plugin| {
                                            plugin.plugin_type == "convolution"
                                                && plugin
                                                    .parameters
                                                    .get("ir_file")
                                                    .and_then(serde_json::Value::as_str)
                                                    == Some(filename)
                                        })
                                })
                            })
                    })
                })
            });
        if expected
            .iter()
            .all(|reference| references.contains(&Some(*reference)))
            && branches_match
        {
            continue;
        }
        if references.is_empty() {
            // The emitted convolution stage is gone. Keep its candidate as history
            // and emit a separate bound reversion through the canonical reconciler.
            rolled_back_ids.push(record.decision_id.clone());
            record
                .reason_codes
                .push("phase_fir_removed_before_delivery".to_owned());
        } else {
            // A redistributed/replaced FIR may preserve the effect, but a path
            // substitution alone is not a replay witness of that equivalence.
            record.status = DecisionStatus::Unresolved;
            record
                .reason_codes
                .push("phase_fir_replaced_requires_reassessment".to_owned());
        }
    }
    // A candidate-stage claim cannot be promoted against changed processing.
    // Keep the original decision and its observations as unresolved history until
    // the owning operation supplies a final replay/reversion witness. A change
    // alone cannot establish whether a particular correction was removed.
    if processing_snapshot(result)? != *assessed_processing {
        for record in &mut result.metadata.provisional_decisions {
            if matches!(
                record.status,
                DecisionStatus::Applied
                    | DecisionStatus::Constrained
                    | DecisionStatus::AlreadyAcceptable
            ) {
                record.status = DecisionStatus::Unresolved;
                record.stage = DecisionStage::Provisional;
                record.final_graph_identity = None;
                record.reason_codes.push(
                    "processing_changed_during_finalization_requires_reassessment".to_owned(),
                );
            }
        }
    }
    result.finalized_decisions = None;
    let mut output = result.to_dsp_chain_output();
    finalize_output_ledger(
        &mut output,
        &result.metadata.provisional_decisions,
        &ReconciliationEvents {
            rolled_back_ids,
            final_acceptance: result.metadata.correction_acceptance.clone(),
            ..Default::default()
        },
    )?;
    let evidence = roomeq_quality::graph_acceptance_evidence(&output, sample_rate_hz)?;
    output
        .correction_decisions
        .as_mut()
        .ok_or("final ledger missing")?
        .acceptance_evidence = Some(evidence);
    result.finalized_decisions = Some(roomeq_engine::room_result::FinalizedDecisions::from_output(
        &output,
    )?);
    Ok(())
}

/// Canonical delivered-payload identity lives with the ledger contract in
/// the model so the export lane can rebind carried ledgers after
/// reference rewriting without depending on workflow.
pub use roomeq_model::decision_ledger::{
    GraphIdentity, canonical_graph_identity, canonical_value_identity,
};

/// Reconcile provisional records against the delivered output and attach
/// the finalized ledger.
///
/// The identity is computed over `output` with any previous ledger
/// cleared, so a stale or forged attachment cannot survive: it is
/// replaced by reconciliation against the exact bytes being shipped.
/// Final delivery claims bind the fresh fingerprint; history stays
/// provisional. Retained joint-stage refusals are added as history. Without
/// provisional records or such refusals, the ledger is explicitly empty
/// (valid) — the report layer, not this function, explains *why*
/// no decision applied. Every emission point (native save, export
/// package) finalizes its own object; any later mutation must
/// re-finalize, which [`verify_final_binding`] enforces.
///
/// # Errors
///
/// Returns a reason when the output does not serialize or a provisional
/// record is invalid: an invalid producer fails the emission
/// fail-closed instead of shipping an unbound ledger.
pub fn finalize_output_ledger(
    output: &mut DspGraph,
    provisional: &[DecisionRecord],
    events: &ReconciliationEvents,
) -> Result<GraphIdentity, String> {
    output.correction_decisions = None;
    let value = serde_json::to_value(&*output)
        .map_err(|error| format!("delivered output does not serialize: {error}"))?;
    let identity = canonical_value_identity(&value);
    let mut records = provisional.to_vec();
    let gains = delivered_gains::records(output)?;
    for previous in provisional
        .iter()
        .filter(|record| record.decision_id.starts_with(delivered_gains::ID_PREFIX))
    {
        if !gains
            .iter()
            .any(|gain| gain.decision_id == previous.decision_id)
        {
            return Err(format!(
                "stale delivered gain record: {}",
                previous.decision_id
            ));
        }
    }
    for record in joint_sub_rejection_history(output)?
        .into_iter()
        .chain(gains)
    {
        if let Some(existing) = records
            .iter()
            .find(|existing| existing.decision_id == record.decision_id)
        {
            let mut comparable = existing.clone();
            // Reconciliation may append final-acceptance context, but must
            // never replace the source refusal or promote it to applied EQ.
            comparable
                .reason_codes
                .retain(|code| !code.starts_with("final_acceptance_"));
            // A repeated finalization may carry the previous binding. The
            // processing facts must still match the freshly derived record.
            if record.decision_id.starts_with(delivered_gains::ID_PREFIX) {
                comparable.stage = record.stage;
                comparable.final_graph_identity = record.final_graph_identity.clone();
            }
            if comparable != record {
                return Err(format!(
                    "conflicting {} history: {}",
                    if record.decision_id.starts_with(delivered_gains::ID_PREFIX) {
                        "delivered-gain"
                    } else {
                        "joint-stage"
                    },
                    record.decision_id
                ));
            }
        } else {
            records.push(record);
        }
    }
    for record in &records {
        record
            .validate()
            .map_err(|reason| format!("provisional record refused: {reason}"))?;
    }
    let mut ledger = reconcile_ledger(&records, &identity, events);
    ledger.payload_binding = Some(roomeq_model::payload_binding::PayloadBinding::new(
        &value,
        &identity.fingerprint,
    ));
    verify_final_binding(&ledger, &identity)?;
    output.correction_decisions = Some(ledger);
    Ok(identity)
}

// Preserve actual stage refusals as history, not acceptance of the final route.
// The retained diagnostics are the producer; never infer a refusal from curves
// or promote a missing reason into "already acceptable".
fn joint_sub_rejection_history(output: &DspGraph) -> Result<Vec<DecisionRecord>, String> {
    let mut records = Vec::new();
    for (channel, chain) in &output.channels {
        let Some(report) = &chain.joint_sub else {
            continue;
        };
        let evidence = canonical_value_identity(
            &serde_json::to_value(report).map_err(|error| error.to_string())?,
        );
        let band = report.seats.first().and_then(|seat| {
            let frequencies = &seat.before.freq;
            (frequencies.len() >= 2
                && frequencies.iter().all(|f| f.is_finite() && *f > 0.0)
                && frequencies.windows(2).all(|pair| pair[0] < pair[1])
                && report
                    .seats
                    .iter()
                    .all(|seat| seat.before.freq == *frequencies))
            .then(|| [frequencies[0], frequencies[frequencies.len() - 1]])
        });
        for (kind, action, reason) in [
            (
                "joint_array_proposal_reverted",
                DecisionAction::GainAdjust,
                &report.array_rejection_reason,
            ),
            (
                "joint_shared_eq_proposal_reverted",
                DecisionAction::Equalize,
                &report.shared_eq_rejection_reason,
            ),
        ] {
            let Some(reason) = reason.as_ref().filter(|reason| !reason.trim().is_empty()) else {
                continue;
            };
            for physical_output in &report.physical_outputs {
                records.push(DecisionRecord {
                    decision_id: format!("{kind}:{channel}:{physical_output}:{}", evidence.fingerprint),
                    ledger_version: DECISION_LEDGER_VERSION.to_owned(),
                    stage: DecisionStage::Provisional,
                    logical_input: channel.clone(),
                    physical_output: physical_output.clone(),
                    // Capture identities are not retained in these stage diagnostics.
                    measurement_refs: Vec::new(),
                    seat_refs: report.seats.iter().map(|seat| format!(
                        "{}:seat-index-{}", seat.reference_scope, seat.seat_index,
                    )).collect(),
                    frequency_band_hz: band,
                    filter_center_hz: None,
                    action,
                    status: DecisionStatus::Reverted,
                    reason_codes: vec![kind.to_owned(), reason.clone(), "stage_history_not_final_route_acceptance".to_owned()],
                    observed: Vec::new(),
                    limits: vec![
                        ObservedQuantity { name: "stage_correction_band_min".into(), value: report.level_band_hz[0], unit: "hz".into() },
                        ObservedQuantity { name: "stage_correction_band_max".into(), value: report.level_band_hz[1], unit: "hz".into() },
                    ],
                    evidence_refs: vec![
                        format!("channels.{channel}.joint_sub:{}", evidence.fingerprint),
                        "predicted_stage_assessment_not_recorded_playback".into(),
                        "raw_capture_identity_unavailable; seats use recorded reference plus positional index".into(),
                    ],
                    confidence: roomeq_model::AssessmentConfidence::Unknown,
                    related_decision_ids: Vec::new(),
                    supersedes_ids: Vec::new(),
                    final_graph_identity: None,
                });
            }
        }
    }
    Ok(records)
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
/// - identity-fallback outputs gain a final `Reverted` record: the
///   candidate was not retained, which does not establish that the baseline
///   is acoustically acceptable;
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
        if let Some(acceptance) = &events.final_acceptance {
            let reason = match acceptance.derived_outcome() {
                roomeq_model::RoomEqOutcome::Accepted => "final_acceptance_accepted",
                roomeq_model::RoomEqOutcome::Unchanged => "final_acceptance_unchanged",
                roomeq_model::RoomEqOutcome::Rejected => "final_acceptance_rejected",
                roomeq_model::RoomEqOutcome::InsufficientEvidence => {
                    "final_acceptance_insufficient_evidence"
                }
            };
            // Applied describes processing, not a demonstrated benefit. An
            // acoustic failure alone does not prove retained protection was
            // removed. Only the explicit mutation events below claim reversion.
            if !retained.reason_codes.iter().any(|code| code == reason) {
                retained.reason_codes.push(reason.to_owned());
            }
        }
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
            action: superseded.map_or(DecisionAction::Equalize, |record| record.action.clone()),
            status: DecisionStatus::Reverted,
            reason_codes: vec![if superseded.is_some_and(|record| {
                record
                    .reason_codes
                    .iter()
                    .any(|reason| reason == "phase_fir_removed_before_delivery")
            }) {
                String::from("phase_fir_removed_before_delivery")
            } else {
                String::from("rollback_after_acceptance_failure")
            }],
            observed: if superseded
                .is_some_and(|record| record.action == DecisionAction::PhaseCorrect)
            {
                vec![observed("phase_fir_retained", 0.0, "ratio")]
            } else {
                vec![observed("delivered_correction_db", 0.0, "db")]
            },
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
            status: DecisionStatus::Reverted,
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

    // Per-channel delivered-scope shares (req R6) derive from the reconciled
    // final records: they inherit the delivery binding below without changing
    // the bound payload identity (ledgers never cover themselves).
    let channel_summaries = operational_summaries(&decisions);
    CorrectionDecisionLedger {
        acceptance_evidence: None,
        payload_binding: None,
        ledger_version: DECISION_LEDGER_VERSION.to_string(),
        decisions,
        channel_summaries,
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
        if record.stage == DecisionStage::Final {
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

    #[test]
    fn roadmap_correction_joint_phase_requires_every_resource() {
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        let mut candidate = provisional_applied("joint-phase", "left", None);
        candidate.action = DecisionAction::PhaseCorrect;
        candidate.evidence_refs.extend([
            "phase-fir:first.wav".to_owned(),
            "phase-fir:second.wav".to_owned(),
        ]);
        result.metadata.provisional_decisions.push(candidate);
        result.channels.get_mut("left").unwrap().plugins =
            vec![roomeq_engine::output::create_convolution_plugin(
                "first.wav",
            )];
        let processing = processing_snapshot(&result).unwrap();
        finalize_result_ledger(&mut result, &processing, 48000.0).unwrap();
        let output = result.to_dsp_chain_output();
        let ledger = output.correction_decisions.as_ref().unwrap();
        assert!(
            !ledger.decisions.iter().any(|record| {
                record.action == DecisionAction::PhaseCorrect
                    && record.stage == DecisionStage::Final
                    && record.status == DecisionStatus::Applied
            }),
            "one surviving FIR cannot prove a joint phase operation survived"
        );
    }

    #[test]
    fn amendment4_final_ledger_refuses_negative_shipped_delay() {
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        result
            .channels
            .get_mut("left")
            .unwrap()
            .plugins
            .push(roomeq_engine::output::create_delay_plugin(-1.5));
        let processing = processing_snapshot(&result).unwrap();
        let error = finalize_result_ledger(&mut result, &processing, 48_000.0)
            .expect_err("negative shipped delay must fail closed");
        assert!(
            error.contains("unresolved causal delay"),
            "unexpected refusal: {error}"
        );
    }

    #[test]
    fn roadmap_correction_phase_resource_replacement_is_not_delivery() {
        for replacement in [None, Some("replacement.wav")] {
            let mut result = crate::test_fixtures::single_channel_room_result("left");
            let mut candidate = provisional_applied("phase-candidate", "left", None);
            candidate.action = DecisionAction::PhaseCorrect;
            candidate
                .evidence_refs
                .push("phase-fir:original.wav".to_owned());
            result.metadata.provisional_decisions.push(candidate);
            let channel = result.channels.get_mut("left").unwrap();
            channel.plugins.clear();
            if let Some(reference) = replacement {
                channel
                    .plugins
                    .push(roomeq_engine::output::create_convolution_plugin(reference));
            }
            // The removal/replacement predates the outer finalization snapshot.
            let processing = processing_snapshot(&result).unwrap();
            finalize_result_ledger(&mut result, &processing, 48_000.0).unwrap();
            let output = result.to_dsp_chain_output();
            let ledger = output.correction_decisions.as_ref().unwrap();
            assert!(!ledger.decisions.iter().any(|record| {
                record.stage == DecisionStage::Final && record.status == DecisionStatus::Applied
            }));
            if replacement.is_some() {
                assert_eq!(ledger.decisions.len(), 1);
                assert_eq!(ledger.decisions[0].status, DecisionStatus::Unresolved);
            } else {
                let reversion = ledger
                    .decisions
                    .iter()
                    .find(|record| record.status == DecisionStatus::Reverted)
                    .unwrap();
                assert_eq!(reversion.action, DecisionAction::PhaseCorrect);
                assert_eq!(reversion.supersedes_ids, ["phase-candidate"]);
                assert_eq!(reversion.stage, DecisionStage::Final);
                assert!(reversion.final_graph_identity.is_some());
            }
            ledger.validate().unwrap();
        }
    }

    #[test]
    fn roadmap_correction_stale_reversion_binding_is_refused() {
        let (_, identity) = delivered_graph();
        let ledger = reconcile_ledger(
            &[provisional_applied("candidate", "left", None)],
            &identity,
            &ReconciliationEvents {
                rolled_back_ids: vec!["candidate".to_owned()],
                ..Default::default()
            },
        );
        assert!(verify_final_binding(&ledger, &identity).is_ok());
        let changed = canonical_value_identity(&serde_json::json!({"different": "processing"}));
        assert!(verify_final_binding(&ledger, &changed).is_err());
    }

    #[test]
    fn roadmap_correction_reconciled_ledger_carries_operational_summaries() {
        // Req R6: the reconciled ledger derives per-channel delivered-scope
        // shares from its own final records, inheriting the delivery binding.
        // The rolled-back band contributes a final Reverted successor (F10:
        // attempted benefit never delivered), which counts as undecided scope.
        let (_, identity) = delivered_graph();
        let mut gain = provisional_applied("trim", "left", None);
        gain.action = DecisionAction::GainAdjust;
        gain.decision_id = String::from("trim-left");
        let ledger = reconcile_ledger(
            &[
                provisional_applied("eq-band", "left", Some([40.0, 400.0])),
                provisional_applied("weak-band", "left", Some([100.0, 400.0])),
                gain.clone(),
            ],
            &identity,
            &ReconciliationEvents {
                rolled_back_ids: vec![String::from("weak-band")],
                ..Default::default()
            },
        );
        assert!(verify_final_binding(&ledger, &identity).is_ok());
        assert!(ledger.validate().is_ok());
        assert_eq!(ledger.channel_summaries.len(), 1);
        let summary = &ledger.channel_summaries[0];
        assert_eq!(summary.channel, "left");
        assert_eq!(summary.decided_equalize, 2);
        assert_eq!(summary.delivered_equalize, 1);
        assert_eq!(summary.operational_response_pct, Some(50.0));
        // A channel without decided equalization scope has no entry: the
        // viewer renders pending instead of inventing 100%.
        let scoped = reconcile_ledger(&[gain], &identity, &ReconciliationEvents::default());
        assert!(scoped.validate().is_ok());
        assert!(scoped.channel_summaries.is_empty());
    }

    #[test]
    fn roadmap_correction_public_snapshot_invalidates_later_mutation() {
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        result
            .metadata
            .provisional_decisions
            .push(provisional_applied("candidate", "left", None));
        let processing = processing_snapshot(&result).unwrap();
        finalize_result_ledger(&mut result, &processing, 48_000.0).unwrap();
        let output = result.to_dsp_chain_output();
        assert_eq!(
            output.correction_decisions.as_ref().unwrap().decisions[0].stage,
            DecisionStage::Final
        );
        assert!(roomeq_engine::room_result::FinalizedDecisions::from_output(&output).is_ok());

        result
            .channels
            .get_mut("left")
            .unwrap()
            .plugins
            .push(roomeq_engine::output::create_gain_plugin(1.0));
        let changed = result.to_dsp_chain_output();
        let record = &changed.correction_decisions.as_ref().unwrap().decisions[0];
        assert_eq!(record.status, DecisionStatus::Unresolved);
        assert_eq!(record.stage, DecisionStage::Provisional);
        assert!(record.final_graph_identity.is_none());
        assert!(
            record
                .reason_codes
                .iter()
                .any(|reason| reason == "payload_changed_after_reconciliation")
        );
        assert!(changed.correction_decisions.unwrap().validate().is_ok());
    }

    #[test]
    fn roadmap_correction_final_mutation_does_not_promote_candidate_claims() {
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        result
            .metadata
            .provisional_decisions
            .push(provisional_applied("candidate", "left", None));
        let processing = processing_snapshot(&result).unwrap();
        result
            .channels
            .get_mut("left")
            .unwrap()
            .plugins
            .push(roomeq_engine::output::create_gain_plugin(-1.0));
        finalize_result_ledger(&mut result, &processing, 48_000.0).unwrap();
        let output = result.to_dsp_chain_output();
        let record = &output.correction_decisions.as_ref().unwrap().decisions[0];
        assert_eq!(record.status, DecisionStatus::Unresolved);
        assert!(
            record
                .reason_codes
                .iter()
                .any(|reason| reason
                    == "processing_changed_during_finalization_requires_reassessment")
        );
        assert!(!record.is_final_claim());
    }

    #[test]
    fn roadmap_correction_report_only_updates_preserve_processing_claim() {
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        result
            .metadata
            .provisional_decisions
            .push(provisional_applied("candidate", "left", None));
        let processing = processing_snapshot(&result).unwrap();
        result.metadata.timestamp = "new report timestamp".to_owned();
        result.channels.get_mut("left").unwrap().final_curve = None;
        assert_eq!(processing_snapshot(&result).unwrap(), processing);
        finalize_result_ledger(&mut result, &processing, 48_000.0).unwrap();
        let output = result.to_dsp_chain_output();
        let record = &output.correction_decisions.as_ref().unwrap().decisions[0];
        assert_eq!(record.status, DecisionStatus::Applied);
        assert_eq!(record.stage, DecisionStage::Final);
    }

    #[test]
    fn roadmap_correction_acceptance_failure_does_not_invent_processing_reversion() {
        let (_, identity) = delivered_graph();
        let report = rejected_report();
        assert_eq!(
            report.derived_outcome(),
            roomeq_model::RoomEqOutcome::Rejected
        );
        let ledger = reconcile_ledger(
            &[provisional_applied("retained-protection", "left", None)],
            &identity,
            &ReconciliationEvents {
                final_acceptance: Some(report),
                ..Default::default()
            },
        );
        let record = &ledger.decisions[0];
        assert_eq!(record.status, DecisionStatus::Applied);
        assert!(
            record
                .reason_codes
                .iter()
                .any(|reason| reason == "final_acceptance_rejected")
        );
        assert!(
            !ledger
                .decisions
                .iter()
                .any(|record| record.status == DecisionStatus::Reverted)
        );
    }

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
        assert!(
            result
                .applied_final_ids
                .contains(&String::from("trim-left"))
        );
        assert!(result.history_ids.contains(&String::from("dec-phase-left")));
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["outcome"], serde_json::json!("accepted"));
        let back: WorkflowResultDocument = serde_json::from_value(json).unwrap();
        assert_eq!(back, result);
    }

    /// Emission attaches a bound ledger: provisional Applied rows become
    /// Final claims on the exact shipped bytes, and the returned identity
    /// matches the payload without its ledger.
    #[test]
    fn roadmap_correction_finalize_attaches_bound_ledger() {
        let (mut graph, _) = delivered_graph();
        assert!(graph.correction_decisions.is_none());
        let provisional = vec![provisional_applied(
            "dec-eq-left",
            "left",
            Some([40.0, 400.0]),
        )];
        let events = ReconciliationEvents::default();
        let identity =
            finalize_output_ledger(&mut graph, &provisional, &events).expect("finalize attaches");
        let ledger = graph
            .correction_decisions
            .as_ref()
            .expect("ledger attached to the shipped output");
        assert!(ledger.validate().is_ok());
        assert!(verify_final_binding(ledger, &identity).is_ok());
        let applied = ledger
            .decisions
            .iter()
            .find(|record| record.decision_id == "dec-eq-left")
            .expect("provisional row survives");
        assert_eq!(applied.stage, DecisionStage::Final);
        assert_eq!(
            applied.final_graph_identity.as_deref(),
            Some(identity.fingerprint.as_str())
        );
        // The identity binds the DSP content, never the ledger itself:
        // clearing the attachment reproduces the same fingerprint.
        let mut cleared = graph.clone();
        cleared.correction_decisions = None;
        assert_eq!(canonical_graph_identity(&cleared), identity);
    }

    /// A stale or forged attachment cannot survive finalization: it is
    /// replaced by reconciliation against the exact shipped bytes.
    #[test]
    fn roadmap_correction_finalize_replaces_stale_attachment() {
        let (mut graph, _) = delivered_graph();
        let mut stale = provisional_applied("dec-stale", "left", Some([40.0, 400.0]));
        stale.stage = DecisionStage::Final;
        stale.final_graph_identity = Some(String::from("deadbeefdeadbeef"));
        graph.correction_decisions = Some(CorrectionDecisionLedger {
            acceptance_evidence: None,
            payload_binding: None,
            ledger_version: DECISION_LEDGER_VERSION.to_string(),
            decisions: vec![stale],
            channel_summaries: Vec::new(),
        });
        let provisional = vec![provisional_applied(
            "dec-eq-left",
            "left",
            Some([40.0, 400.0]),
        )];
        finalize_output_ledger(&mut graph, &provisional, &ReconciliationEvents::default())
            .expect("finalize replaces stale ledger");
        let ledger = graph.correction_decisions.as_ref().unwrap();
        assert!(
            ledger
                .decisions
                .iter()
                .all(|record| record.decision_id != "dec-stale"),
            "stale rows must not survive"
        );
    }

    /// No applicable decisions ships an explicitly empty (valid) ledger,
    /// not a fabricated acceptance: reports explain the absence.
    #[test]
    fn roadmap_correction_finalize_empty_provisional_is_explicitly_empty() {
        let (mut graph, _) = delivered_graph();
        let identity = finalize_output_ledger(&mut graph, &[], &ReconciliationEvents::default())
            .expect("empty provisional finalizes");
        let ledger = graph.correction_decisions.as_ref().unwrap();
        assert!(ledger.validate().is_ok());
        assert!(ledger.decisions.is_empty());
        assert!(verify_final_binding(ledger, &identity).is_ok());
    }

    /// An invalid producer fails the emission fail-closed: nothing
    /// attaches and the caller must refuse to ship.
    #[test]
    fn roadmap_correction_finalize_refuses_invalid_provisional() {
        let (mut graph, _) = delivered_graph();
        let mut bad = provisional_applied("dec-bad", "left", Some([40.0, 400.0]));
        bad.logical_input.clear();
        let error = finalize_output_ledger(&mut graph, &[bad], &ReconciliationEvents::default())
            .expect_err("invalid provisional must refuse emission");
        assert!(error.contains("provisional record refused"), "{error}");
        assert!(graph.correction_decisions.is_none());
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
        assert_eq!(fallback.status, DecisionStatus::Reverted);
        assert!(!ledger.decisions.iter().any(|record| {
            record.stage == DecisionStage::Final
                && record.status == DecisionStatus::AlreadyAcceptable
        }));
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
