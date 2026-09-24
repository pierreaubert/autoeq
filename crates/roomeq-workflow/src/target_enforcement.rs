//! Target/transition reconciliation in the workflow (Wave 1, step 5).
//!
//! This module binds the engine [`TargetEnforcementReport`] to final K4
//! rows: blocked detail bands record `InsufficientEvidence` with the
//! guard reason codes, so the limit stays visible, while passing bands
//! record `Applied`. Bands are stated explicitly; rows are never merged.

// Rust guideline compliant 2026-02-21

use roomeq_engine::target_enforcement::TargetEnforcementReport;
use roomeq_model::AssessmentConfidence;
use roomeq_model::decision_ledger::{
    DECISION_LEDGER_VERSION, DecisionAction, DecisionRecord, DecisionStage, DecisionStatus,
    ObservedQuantity,
};
use roomeq_model::target_transition::ProposedDetailBand;
use serde::{Deserialize, Serialize};

/// Version pin for target-enforcement reconciliation.
pub const TARGET_RECONCILIATION_VERSION: &str = "workflow-target-v1";

/// Serializable reconciliation summary over one enforcement report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetReconciliationSummary {
    /// Reconciliation version.
    pub version: String,
    /// User target identity carried through enforcement.
    pub user_target_id: Option<String>,
    /// Decision identifiers bound by this reconciliation.
    pub decision_ids: Vec<String>,
    /// Bands where detail correction is limited.
    pub limited_bands: Vec<[f64; 2]>,
}

/// Reconcile target enforcement into K4 decision rows.
///
/// One row per proposal band, in proposal order. `graph_identity` binds
/// final claims; without it rows stay provisional and never read as
/// delivery claims. Every row carries the enforced user-target identity in
/// `evidence_refs` when the report names one, so the respected target stays
/// identifiable without inferring it from bands or reasons; callers must not
/// push it again.
#[allow(clippy::too_many_arguments)]
pub fn reconcile_target_decisions(
    report: &TargetEnforcementReport,
    proposals: &[ProposedDetailBand],
    logical_input: &str,
    physical_output: &str,
    measurement_refs: Vec<String>,
    seat_refs: Vec<String>,
    graph_identity: Option<String>,
) -> Vec<DecisionRecord> {
    report
        .outcomes
        .iter()
        .zip(proposals.iter())
        .enumerate()
        .map(|(index, (outcome, proposal))| {
            let (stage, final_graph_identity) = match graph_identity.clone() {
                Some(identity) => (DecisionStage::Final, Some(identity)),
                None => (DecisionStage::Provisional, None),
            };
            let (status, mut reasons) = if outcome.fires {
                (
                    DecisionStatus::InsufficientEvidence,
                    vec![String::from("direct_sound_damage_guard")],
                )
            } else {
                (DecisionStatus::Applied, vec![String::from("target_ok")])
            };
            reasons.extend(outcome.reason_codes.iter().cloned());
            let mut evidence_refs = proposal.evidence_refs.clone();
            if let Some(target_id) = &report.resolution.user_target_id
                && !evidence_refs.contains(target_id)
            {
                evidence_refs.push(target_id.clone());
            }
            DecisionRecord {
                decision_id: format!("target-band-{index}"),
                ledger_version: DECISION_LEDGER_VERSION.to_string(),
                stage,
                logical_input: logical_input.to_string(),
                physical_output: physical_output.to_string(),
                measurement_refs: measurement_refs.clone(),
                seat_refs: seat_refs.clone(),
                frequency_band_hz: Some(proposal.band_hz),
                filter_center_hz: None,
                action: DecisionAction::Equalize,
                status,
                reason_codes: reasons,
                observed: vec![ObservedQuantity {
                    name: String::from("anechoic_regime_weight"),
                    value: outcome.anechoic_weight,
                    unit: String::from("ratio"),
                }],
                limits: Vec::new(),
                evidence_refs,
                confidence: AssessmentConfidence::default(),
                related_decision_ids: Vec::new(),
                supersedes_ids: Vec::new(),
                final_graph_identity,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_engine::target_enforcement::enforce_target_chain;
    use roomeq_model::target_transition::{
        DirectEvidence, TARGET_TRANSITION_VERSION, TargetStage, TargetStageKind, TransitionConfig,
    };

    fn proposals() -> Vec<ProposedDetailBand> {
        vec![
            ProposedDetailBand {
                band_hz: [2000.0, 8000.0],
                evidence: DirectEvidence::RoomCurveOnly,
                evidence_refs: Vec::new(),
            },
            ProposedDetailBand {
                band_hz: [40.0, 80.0],
                evidence: DirectEvidence::RoomCurveOnly,
                evidence_refs: Vec::new(),
            },
        ]
    }

    #[test]
    fn guard_rows_record_insufficient_evidence() {
        let chain = roomeq_model::target_transition::TargetChain {
            version: TARGET_TRANSITION_VERSION.to_string(),
            stages: vec![TargetStage {
                kind: TargetStageKind::MeasuredCalibration,
                stage_id: String::from("cal"),
                label: String::from("calibration"),
                evidence_refs: Vec::new(),
            }],
            transition: TransitionConfig {
                version: TARGET_TRANSITION_VERSION.to_string(),
                center_hz: 300.0,
                width_oct: 1.0,
            },
            user_target_id: Some(String::from("user-flat")),
        };
        let proposals = proposals();
        let report = enforce_target_chain(&chain, &proposals).unwrap();
        let records = reconcile_target_decisions(
            &report,
            &proposals,
            "stereo",
            "main-l",
            vec![String::from("meas-1")],
            vec![String::from("mlp")],
            Some(String::from("graph-1")),
        );
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].status, DecisionStatus::InsufficientEvidence);
        assert_eq!(records[1].status, DecisionStatus::Applied);
        for record in &records {
            assert!(record.validate().is_ok());
            assert!(record.is_final_claim());
            assert_eq!(record.filter_center_hz, None);
            // The enforced user target stays identifiable on every row.
            assert_eq!(
                record
                    .evidence_refs
                    .iter()
                    .filter(|reference| *reference == "user-flat")
                    .count(),
                1,
                "row {} must carry the enforced target exactly once",
                record.decision_id
            );
        }
        assert!(
            records[0]
                .reason_codes
                .contains(&String::from("room_curve_only_detail_eq"))
        );
    }

    #[test]
    fn reconcile_rows_without_named_target_invent_no_identity() {
        let mut chain = roomeq_model::target_transition::TargetChain {
            version: TARGET_TRANSITION_VERSION.to_string(),
            stages: vec![TargetStage {
                kind: TargetStageKind::MeasuredCalibration,
                stage_id: String::from("cal"),
                label: String::from("calibration"),
                evidence_refs: Vec::new(),
            }],
            transition: TransitionConfig {
                version: TARGET_TRANSITION_VERSION.to_string(),
                center_hz: 300.0,
                width_oct: 1.0,
            },
            user_target_id: None,
        };
        // A chain without a user target still enforces the damage guard;
        // rows carry no invented identity.
        let proposals = proposals();
        let report = enforce_target_chain(&chain, &proposals).unwrap();
        assert!(report.resolution.user_target_id.is_none());
        let records = reconcile_target_decisions(
            &report,
            &proposals,
            "stereo",
            "main-l",
            vec![String::from("meas-1")],
            vec![String::from("mlp")],
            Some(String::from("graph-1")),
        );
        assert_eq!(records.len(), 2);
        for record in &records {
            assert!(record.validate().is_ok());
            assert!(!record.evidence_refs.iter().any(|reference| {
                reference == "user-flat" || reference.starts_with("user-target")
            }));
        }
        chain.user_target_id = Some(String::from("user-flat"));
        let report = enforce_target_chain(&chain, &proposals).unwrap();
        let records = reconcile_target_decisions(
            &report,
            &proposals,
            "stereo",
            "main-l",
            vec![String::from("meas-1")],
            vec![String::from("mlp")],
            Some(String::from("graph-1")),
        );
        for record in &records {
            assert!(record.evidence_refs.contains(&String::from("user-flat")));
        }
    }
}
