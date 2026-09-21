//! Target/transition enforcement in the engine (Wave 1, step 5).
//!
//! This module applies the model [`TargetChain`](roomeq_model::target_transition::TargetChain)
//! to proposed corrections: stage separation is checked, the
//! direct-sound damage guard judges every detail band, and the user
//! target resolves with evidence limits explained. Records produced
//! here are provisional; only workflow reconciliation binds final rows.

// Rust guideline compliant 2026-02-21

use roomeq_model::target_transition::{
    DamageGuardOutcome, ProposedDetailBand, TargetChain, TargetResolution, evaluate_damage_guard,
    resolve_user_target,
};

/// Provisional enforcement verdict over a target chain.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetEnforcementReport {
    /// Per-proposal damage-guard outcomes in proposal order.
    pub outcomes: Vec<DamageGuardOutcome>,
    /// User-target resolution with explained limits.
    pub resolution: TargetResolution,
}

/// Enforce the target chain against proposed detail bands.
///
/// Checks stage separation, evaluates the damage guard per band, and
/// resolves the user target with limits explained. The user target is
/// never substituted: limits attach to bands, not to a replacement
/// curve.
///
/// # Errors
///
/// Returns the chain validation reason or a proposal band-shape reason.
pub fn enforce_target_chain(
    chain: &TargetChain,
    proposals: &[ProposedDetailBand],
) -> Result<TargetEnforcementReport, String> {
    chain.validate()?;
    let outcomes: Vec<DamageGuardOutcome> = proposals
        .iter()
        .map(|proposal| evaluate_damage_guard(chain, proposal))
        .collect::<Result<_, _>>()?;
    let resolution = resolve_user_target(chain, proposals, &outcomes);
    Ok(TargetEnforcementReport {
        outcomes,
        resolution,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::target_transition::{
        DirectEvidence, TARGET_TRANSITION_VERSION, TargetStage, TargetStageKind, TransitionConfig,
    };

    fn chain() -> TargetChain {
        TargetChain {
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
        }
    }

    #[test]
    fn enforcement_blocks_room_curve_detail_eq() {
        let proposals = vec![ProposedDetailBand {
            band_hz: [2000.0, 8000.0],
            evidence: DirectEvidence::RoomCurveOnly,
            evidence_refs: Vec::new(),
        }];
        let report = enforce_target_chain(&chain(), &proposals).unwrap();
        assert!(report.outcomes[0].fires);
        assert_eq!(report.resolution.limited_bands, vec![[2000.0, 8000.0]]);
        assert_eq!(
            report.resolution.user_target_id,
            Some(String::from("user-flat"))
        );
    }

    #[test]
    fn enforcement_passes_validated_detail_eq() {
        let proposals = vec![ProposedDetailBand {
            band_hz: [2000.0, 8000.0],
            evidence: DirectEvidence::ValidatedDirectSound,
            evidence_refs: vec![String::from("ev-direct")],
        }];
        let report = enforce_target_chain(&chain(), &proposals).unwrap();
        assert!(!report.outcomes[0].fires);
        assert!(report.resolution.limited_bands.is_empty());
    }
}
