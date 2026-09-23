//! Target/transition architecture types (Wave 1, step 5).
//!
//! These types enforce the Toole decision rules as code: a good room
//! curve is not an inverse-design target. Below the modal transition,
//! in-situ source/seat data decides; above it, anechoic/angular data
//! decides. The handover is a smooth confidence-dependent transition,
//! never a hard Schroeder cutoff.
//!
//! Measured calibration, optional preference tilt, and playback-level
//! loudness compensation are separate identifiable stages. The
//! [`DamageGuardOutcome`] fires when room-curve-only detail EQ would
//! damage the direct sound, and [`TargetResolution`] respects the user
//! target with evidence limits explained, never silently replaced.

// Rust guideline compliant 2026-02-21

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Version pin for [`TransitionConfig`] and [`TargetChain`].
pub const TARGET_TRANSITION_VERSION: &str = "target-transition-v1";

fn target_transition_version_default() -> String {
    TARGET_TRANSITION_VERSION.to_string()
}

/// Smooth modal-transition handover between evidence regimes.
///
/// `center_hz` is the middle of the transition region and `width_oct`
/// its half-width in octaves. [`TransitionConfig::anechoic_weight`]
/// rises smoothly from 0 to 1 with frequency; there is no cutoff
/// frequency and no binary regime switch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TransitionConfig {
    /// Config version; must equal [`TARGET_TRANSITION_VERSION`].
    #[serde(default = "target_transition_version_default")]
    pub version: String,
    /// Middle of the transition region in Hz; finite and positive.
    pub center_hz: f64,
    /// Transition half-width in octaves; finite and positive.
    pub width_oct: f64,
}

impl TransitionConfig {
    /// Anechoic-regime weight at `freq_hz`: logistic in log frequency.
    ///
    /// The weight is 0.5 exactly at the center, strictly increasing, and
    /// smooth everywhere: adjacent frequencies never straddle a regime
    /// cliff. The in-situ weight is `1 - weight`.
    pub fn anechoic_weight(&self, freq_hz: f64) -> f64 {
        let log_distance = (freq_hz / self.center_hz).log2() / self.width_oct;
        1.0 / (1.0 + (-log_distance).exp())
    }

    /// Effective anechoic confidence: regime weight times direct-sound
    /// confidence. Low direct confidence attenuates the handover even
    /// high in frequency.
    pub fn effective_anechoic_confidence(&self, freq_hz: f64, direct_confidence: f64) -> f64 {
        self.anechoic_weight(freq_hz) * direct_confidence.clamp(0.0, 1.0)
    }

    /// Reject unknown versions and nonfinite/nonpositive parameters.
    ///
    /// # Errors
    ///
    /// Returns a reason for a version mismatch or an invalid parameter.
    pub fn validate(&self) -> Result<(), String> {
        if self.version != TARGET_TRANSITION_VERSION {
            return Err(format!(
                "unknown target transition version '{}'",
                self.version
            ));
        }
        if !self.center_hz.is_finite() || self.center_hz <= 0.0 {
            return Err(format!(
                "center_hz must be finite and positive (got {})",
                self.center_hz
            ));
        }
        if !self.width_oct.is_finite() || self.width_oct <= 0.0 {
            return Err(format!(
                "width_oct must be finite and positive (got {})",
                self.width_oct
            ));
        }
        Ok(())
    }
}

/// Separable target-chain stage: calibration, preference, and level
/// compensation stay identifiable and never merge into one curve.
///
/// Mirrors the engine-local `StageKind` vocabulary 1:1; this is the
/// serialized contract, the engine mirror stays local without serde.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TargetStageKind {
    /// Measured calibration baseline; always present.
    #[default]
    MeasuredCalibration,
    /// Optional user preference tilt layered over calibration.
    PreferenceTilt,
    /// Playback-level loudness compensation, separate from tuning.
    LevelCompensation,
}

/// One identifiable stage of the target chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TargetStage {
    /// Which separable stage this is.
    pub kind: TargetStageKind,
    /// Stable stage identifier; nonempty and unique within a chain.
    pub stage_id: String,
    /// Human-readable label; never a substitute for `kind`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
    /// Stable evidence reference IDs behind this stage.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_refs: Vec<String>,
}

/// Ordered target chain: calibration plus optional tilt and level.
///
/// The chain carries the user's target identity through every check so
/// enforcement explains limits against it instead of replacing it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TargetChain {
    /// Chain version; must equal [`TARGET_TRANSITION_VERSION`].
    #[serde(default = "target_transition_version_default")]
    pub version: String,
    /// Ordered stages: calibration first, tilt and level after.
    pub stages: Vec<TargetStage>,
    /// Smooth evidence-regime handover.
    pub transition: TransitionConfig,
    /// User target identity respected by every downstream check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_target_id: Option<String>,
}

impl TargetChain {
    /// Check version, stage separation, and transition parameters.
    ///
    /// Exactly one calibration stage is required, before optional tilt and
    /// level stages, which may each appear at most once. Identities stay unique.
    ///
    /// # Errors
    ///
    /// Returns a reason for a version mismatch, missing or duplicated
    /// stages, calibration after an optional stage, empty or duplicate
    /// identities, or an invalid transition.
    pub fn validate(&self) -> Result<(), String> {
        if self.version != TARGET_TRANSITION_VERSION {
            return Err(format!("unknown target chain version '{}'", self.version));
        }
        self.transition.validate()?;
        let calibrations = self
            .stages
            .iter()
            .filter(|stage| stage.kind == TargetStageKind::MeasuredCalibration)
            .count();
        if calibrations != 1 {
            return Err(format!(
                "target chain needs exactly one calibration stage (got {calibrations})"
            ));
        }
        if self.stages.first().map(|stage| stage.kind) != Some(TargetStageKind::MeasuredCalibration)
        {
            return Err(String::from(
                "target calibration must precede preference tilt and level compensation",
            ));
        }
        for kind in [
            TargetStageKind::PreferenceTilt,
            TargetStageKind::LevelCompensation,
        ] {
            let count = self
                .stages
                .iter()
                .filter(|stage| stage.kind == kind)
                .count();
            if count > 1 {
                return Err(format!("target chain holds {count} {kind:?} stages"));
            }
        }
        let mut ids: Vec<&str> = self
            .stages
            .iter()
            .map(|stage| {
                if stage.stage_id.trim().is_empty() {
                    return Err(String::from("target stage identity must not be empty"));
                }
                Ok(stage.stage_id.as_str())
            })
            .collect::<Result<_, String>>()?;
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != self.stages.len() {
            return Err(String::from("target stage identities must be unique"));
        }
        Ok(())
    }
}

/// Direct-sound evidence behind a proposed detail correction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DirectEvidence {
    /// Validated direct/angular evidence supports detail work.
    ValidatedDirectSound,
    /// Room-curve-only evidence: detail EQ would risk direct sound.
    RoomCurveOnly,
    /// Evidence state unknown; fails closed like room-curve-only.
    #[default]
    Unknown,
}

/// Proposed detail correction assessed by the damage guard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ProposedDetailBand {
    /// Proposed correction band in Hz.
    pub band_hz: [f64; 2],
    /// Direct-sound evidence behind the proposal.
    pub evidence: DirectEvidence,
    /// Stable evidence reference IDs cited by the proposal.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_refs: Vec<String>,
}

/// Direct-sound damage-guard verdict for one proposed band.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DamageGuardOutcome {
    /// Whether the proposal is blocked from detail correction.
    pub fires: bool,
    /// Anechoic-regime weight at the band top edge.
    pub anechoic_weight: f64,
    /// Machine-readable reason codes.
    pub reason_codes: Vec<String>,
    /// Human-readable explanation naming the user target and the limit.
    pub explanation: String,
}

/// Fire the damage guard on room-curve-only detail EQ.
///
/// The guard fires when the band reaches the anechoic-weighted region
/// (weight at least 0.5 at the band top edge) without validated
/// direct-sound evidence with nonempty reference IDs. Unknown or uncited
/// evidence fails closed. Reference validation is structural, not authentication.
/// Bass-region
/// proposals below the transition never fire regardless of evidence.
///
/// # Errors
///
/// Returns the chain [`TargetChain::validate`] reason or a band-shape
/// reason for nonfinite or unordered bounds.
pub fn evaluate_damage_guard(
    chain: &TargetChain,
    proposal: &ProposedDetailBand,
) -> Result<DamageGuardOutcome, String> {
    chain.validate()?;
    let [low_hz, high_hz] = proposal.band_hz;
    if !low_hz.is_finite() || !high_hz.is_finite() || high_hz <= low_hz || low_hz <= 0.0 {
        return Err(format!(
            "detail band must satisfy 0 < lo < hi with finite bounds (got [{low_hz}, {high_hz}])"
        ));
    }
    let weight = chain.transition.anechoic_weight(high_hz);
    let target = chain.user_target_id.as_deref().unwrap_or("user target");
    let mut reasons = Vec::new();
    let fires = if weight >= 0.5 {
        match proposal.evidence {
            DirectEvidence::ValidatedDirectSound => {
                let missing_references = proposal.evidence_refs.is_empty()
                    || proposal
                        .evidence_refs
                        .iter()
                        .any(|reference| reference.trim().is_empty());
                if missing_references {
                    reasons.push(String::from("missing_direct_evidence_references"));
                }
                missing_references
            }
            DirectEvidence::RoomCurveOnly => {
                reasons.push(String::from("room_curve_only_detail_eq"));
                true
            }
            DirectEvidence::Unknown => {
                reasons.push(String::from("unknown_direct_evidence"));
                true
            }
        }
    } else {
        reasons.push(String::from("in_situ_region"));
        false
    };
    let explanation = if fires {
        format!(
            "{target} keeps its shape; detail EQ on [{low_hz}, {high_hz}] Hz is limited \
             because anechoic-regime weight is {weight:.2} without validated direct-sound \
             evidence ({}). Broad restrained shaping stays available.",
            reasons.join(", ")
        )
    } else {
        format!(
            "{target} applies to [{low_hz}, {high_hz}] Hz with anechoic-regime weight \
             {weight:.2}; no direct-sound damage limit binds this band."
        )
    };
    Ok(DamageGuardOutcome {
        fires,
        anechoic_weight: weight,
        reason_codes: reasons,
        explanation,
    })
}

/// Whether the user target survived enforcement intact.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TargetResolutionStatus {
    /// Target applies unchanged.
    #[default]
    RespectedFully,
    /// Target applies with explained evidence limits on listed bands.
    RespectedWithLimits,
}

/// User-target resolution: respected with limits explained, never
/// silently replaced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TargetResolution {
    /// User target identity carried through enforcement.
    pub user_target_id: Option<String>,
    /// Resolution status.
    pub status: TargetResolutionStatus,
    /// Per-band explanations; nonempty whenever limits bind.
    pub explanations: Vec<String>,
    /// Bands where detail correction is limited.
    pub limited_bands: Vec<[f64; 2]>,
}

/// Resolve the user target against guard outcomes.
///
/// The user target identity is always carried through: limits are
/// explained against it, and the target itself is never substituted.
/// Every fired guard contributes its band and explanation.
pub fn resolve_user_target(
    chain: &TargetChain,
    proposals: &[ProposedDetailBand],
    outcomes: &[DamageGuardOutcome],
) -> TargetResolution {
    let mut limited_bands = Vec::new();
    let mut explanations = Vec::new();
    for (proposal, outcome) in proposals.iter().zip(outcomes.iter()) {
        if outcome.fires {
            limited_bands.push(proposal.band_hz);
            explanations.push(outcome.explanation.clone());
        }
    }
    TargetResolution {
        user_target_id: chain.user_target_id.clone(),
        status: if limited_bands.is_empty() {
            TargetResolutionStatus::RespectedFully
        } else {
            TargetResolutionStatus::RespectedWithLimits
        },
        explanations,
        limited_bands,
    }
}

/// Fallback transition center in Hz when no room-derived Schroeder
/// frequency is configured.
///
/// Mirrors the analysis-anchored handover used across the transition
/// tests: a neutral small-room value, documented as a fallback rather
/// than a measurement. Callers with a configured Schroeder frequency
/// must prefer it; the builder documents which source was used.
pub const FALLBACK_TRANSITION_CENTER_HZ: f64 = 300.0;
/// Fallback transition half-width in octaves; see
/// [`FALLBACK_TRANSITION_CENTER_HZ`].
pub const FALLBACK_TRANSITION_WIDTH_OCT: f64 = 1.0;

impl TargetChain {
    /// Build the chain from the configured target response.
    ///
    /// Calibration is always present. A preference-tilt stage is added
    /// exactly when the configuration shapes the target away from flat
    /// calibration: a non-flat `shape`, or nonzero preference shelves.
    /// No level-compensation stage is emitted: no playback-level
    /// compensation is currently configurable, and inventing one would
    /// fabricate a processing stage. The transition center prefers a
    /// room-derived `schroeder_hz` and falls back to
    /// [`FALLBACK_TRANSITION_CENTER_HZ`]; the user target keeps its
    /// configured identity throughout.
    pub fn from_target_config(
        target: Option<&crate::config::TargetResponseConfig>,
        schroeder_hz: Option<f64>,
        calibration_evidence_refs: Vec<String>,
    ) -> Self {
        let mut stages = vec![TargetStage {
            kind: TargetStageKind::MeasuredCalibration,
            stage_id: String::from("measured-calibration"),
            label: String::from("measured calibration baseline"),
            evidence_refs: calibration_evidence_refs,
        }];
        let mut user_target_id: Option<String> = None;
        if let Some(response) = target {
            let shape_tilt = !matches!(response.shape, crate::config::TargetShape::Flat);
            let shelf_tilt = response.preference.bass_shelf_db != 0.0
                || response.preference.treble_shelf_db != 0.0;
            if shape_tilt || shelf_tilt {
                let mut label = String::from("preference tilt");
                if shape_tilt {
                    label.push_str(&format!(" ({:?} target shape)", response.shape));
                }
                if shelf_tilt {
                    label.push_str(" + user shelves");
                }
                stages.push(TargetStage {
                    kind: TargetStageKind::PreferenceTilt,
                    stage_id: String::from("preference-tilt"),
                    label,
                    evidence_refs: Vec::new(),
                });
            }
            user_target_id = Some(format!("target-{:?}", response.shape).to_lowercase());
        }
        let center_hz = schroeder_hz
            .filter(|hz| hz.is_finite() && *hz > 0.0)
            .unwrap_or(FALLBACK_TRANSITION_CENTER_HZ);
        Self {
            version: TARGET_TRANSITION_VERSION.to_string(),
            stages,
            transition: TransitionConfig {
                version: TARGET_TRANSITION_VERSION.to_string(),
                center_hz,
                width_oct: FALLBACK_TRANSITION_WIDTH_OCT,
            },
            user_target_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> TargetChain {
        TargetChain {
            version: TARGET_TRANSITION_VERSION.to_string(),
            stages: vec![
                TargetStage {
                    kind: TargetStageKind::MeasuredCalibration,
                    stage_id: String::from("cal"),
                    label: String::from("measured calibration"),
                    evidence_refs: vec![String::from("ev-cal")],
                },
                TargetStage {
                    kind: TargetStageKind::PreferenceTilt,
                    stage_id: String::from("tilt"),
                    label: String::from("user tilt"),
                    evidence_refs: Vec::new(),
                },
                TargetStage {
                    kind: TargetStageKind::LevelCompensation,
                    stage_id: String::from("level"),
                    label: String::from("loudness compensation"),
                    evidence_refs: Vec::new(),
                },
            ],
            transition: TransitionConfig {
                version: TARGET_TRANSITION_VERSION.to_string(),
                center_hz: 300.0,
                width_oct: 1.0,
            },
            user_target_id: Some(String::from("user-harman")),
        }
    }

    #[test]
    fn roadmap_correction_target_calibration_precedes_optional_stages() {
        let original = chain();
        for order in [[1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            let mut reordered = original.clone();
            reordered.stages = order.map(|index| original.stages[index].clone()).to_vec();
            // Exercise the same contract after transport, not just a builder.
            let json = serde_json::to_string(&reordered).unwrap();
            let decoded: TargetChain = serde_json::from_str(&json).unwrap();
            assert!(decoded.validate().is_err(), "accepted order {order:?}");
        }
        // Either optional stage can follow calibration independently. No
        // additional ordering between the optional layers is imposed here.
        for optional in [None, Some(1), Some(2)] {
            let mut supported = original.clone();
            supported.stages.truncate(1);
            if let Some(index) = optional {
                supported.stages.push(original.stages[index].clone());
            }
            supported.validate().unwrap();
        }
        original.validate().unwrap();
    }

    #[test]
    fn transition_is_smooth_not_a_cutoff() {
        let transition = TransitionConfig {
            version: TARGET_TRANSITION_VERSION.to_string(),
            center_hz: 300.0,
            width_oct: 1.0,
        };
        assert!((transition.anechoic_weight(300.0) - 0.5).abs() < 1e-12);
        let freqs = [75.0, 150.0, 300.0, 600.0, 1200.0];
        let weights: Vec<f64> = freqs
            .iter()
            .map(|f| transition.anechoic_weight(*f))
            .collect();
        for pair in weights.windows(2) {
            assert!(pair[0] < pair[1], "weights must rise strictly");
        }
        assert!(weights[0] > 0.0 && weights[4] < 1.0);
    }

    #[test]
    fn chain_builder_keeps_stages_identifiable() {
        use crate::config::{TargetResponseConfig, TargetShape};
        // No configured response: calibration only, fallback handover.
        let bare = TargetChain::from_target_config(None, None, Vec::new());
        bare.validate().expect("built chain validates");
        assert_eq!(bare.stages.len(), 1);
        assert_eq!(bare.stages[0].kind, TargetStageKind::MeasuredCalibration);
        assert_eq!(bare.transition.center_hz, FALLBACK_TRANSITION_CENTER_HZ);
        assert_eq!(bare.user_target_id, None);
        // Flat shape without shelves: still calibration only.
        let flat = TargetResponseConfig {
            shape: TargetShape::Flat,
            ..Default::default()
        };
        let flat_chain = TargetChain::from_target_config(Some(&flat), None, Vec::new());
        assert_eq!(flat_chain.stages.len(), 1);
        assert_eq!(flat_chain.user_target_id, Some(String::from("target-flat")));
        // Non-flat shape adds exactly one tilt stage; the target keeps
        // its configured identity instead of being substituted.
        let harman = TargetResponseConfig {
            shape: TargetShape::Harman,
            ..Default::default()
        };
        let tilted = TargetChain::from_target_config(Some(&harman), None, Vec::new());
        tilted.validate().expect("tilted chain validates");
        assert_eq!(tilted.stages.len(), 2);
        assert_eq!(tilted.stages[1].kind, TargetStageKind::PreferenceTilt);
        assert_eq!(tilted.user_target_id, Some(String::from("target-harman")));
        // Shelves alone also declare tilt; room-derived Schroeder wins
        // over the fallback handover.
        let mut shelved = TargetResponseConfig {
            shape: TargetShape::Flat,
            ..Default::default()
        };
        shelved.preference.bass_shelf_db = 2.0;
        let room = TargetChain::from_target_config(
            Some(&shelved),
            Some(250.0),
            vec![String::from("meas-1")],
        );
        assert_eq!(room.stages.len(), 2);
        assert_eq!(room.transition.center_hz, 250.0);
        assert_eq!(room.stages[0].evidence_refs, vec![String::from("meas-1")]);
    }

    #[test]
    fn damage_guard_fires_on_room_curve_only_detail_eq() {
        let chain = chain();
        let proposal = ProposedDetailBand {
            band_hz: [2000.0, 8000.0],
            evidence: DirectEvidence::RoomCurveOnly,
            evidence_refs: Vec::new(),
        };
        let outcome = evaluate_damage_guard(&chain, &proposal).unwrap();
        assert!(outcome.fires);
        assert!(
            outcome
                .reason_codes
                .contains(&String::from("room_curve_only_detail_eq"))
        );
        assert!(outcome.explanation.contains("user-harman"));
    }

    #[test]
    fn validated_direct_sound_passes_guard() {
        let chain = chain();
        let proposal = ProposedDetailBand {
            band_hz: [2000.0, 8000.0],
            evidence: DirectEvidence::ValidatedDirectSound,
            evidence_refs: vec![String::from("ev-direct")],
        };
        let outcome = evaluate_damage_guard(&chain, &proposal).unwrap();
        assert!(!outcome.fires);
    }

    #[test]
    fn bass_region_never_fires_guard() {
        let chain = chain();
        let proposal = ProposedDetailBand {
            band_hz: [40.0, 80.0],
            evidence: DirectEvidence::RoomCurveOnly,
            evidence_refs: Vec::new(),
        };
        let outcome = evaluate_damage_guard(&chain, &proposal).unwrap();
        assert!(!outcome.fires);
        assert!(
            outcome
                .reason_codes
                .contains(&String::from("in_situ_region"))
        );
    }

    #[test]
    fn stage_separation_is_enforced() {
        let mut bad = chain();
        bad.stages.push(TargetStage {
            kind: TargetStageKind::MeasuredCalibration,
            stage_id: String::from("cal-2"),
            label: String::new(),
            evidence_refs: Vec::new(),
        });
        assert!(bad.validate().is_err());
        let mut bad_transition = chain();
        bad_transition.transition.center_hz = f64::NAN;
        assert!(bad_transition.validate().is_err());
    }

    #[test]
    fn user_target_respected_with_limits_explained() {
        let chain = chain();
        let proposals = vec![
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
        ];
        let outcomes: Vec<DamageGuardOutcome> = proposals
            .iter()
            .map(|proposal| evaluate_damage_guard(&chain, proposal).unwrap())
            .collect();
        let resolution = resolve_user_target(&chain, &proposals, &outcomes);
        assert_eq!(
            resolution.status,
            TargetResolutionStatus::RespectedWithLimits
        );
        assert_eq!(resolution.user_target_id, Some(String::from("user-harman")));
        assert_eq!(resolution.limited_bands, vec![[2000.0, 8000.0]]);
        assert_eq!(resolution.explanations.len(), 1);
    }
}
