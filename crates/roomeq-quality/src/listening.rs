//! Frozen listening protocol setup: arms, presentations and bindings.
//!
//! A [`ListeningSetup`] preregisters what a trial round is staged to show
//! before any listener data exists. Detectability, equivalence and
//! preference are separate intents: a preference result can never be
//! claimed as equivalence, and a nonsignificant detectability result is
//! not equivalence without the prespecified bound the protocol carries.
//!
//! Setups bind immutable chain/stimulus identities, keep single-speaker
//! mono, identical L+R summation and spatial conditions distinct, and hold
//! out programmes before model tuning. Nothing here records a listening
//! outcome: real trials are validated separately in `trial_import`, and
//! synthetic tables stay labelled synthetic there.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::protocol::{BlindedProtocol, ComparisonIntent};
use super::sha256_hex;

/// What the comparison is staged to claim. Preference outcomes never
/// support inaudibility claims; only the matching intent does.
pub fn check_claim_matches_intent(
    intent: ComparisonIntent,
    claimed: ComparisonIntent,
) -> Result<(), String> {
    if intent == claimed {
        return Ok(());
    }
    Err(format!(
        "claim {claimed:?} does not match preregistered intent {intent:?}: stage the claimed intent separately instead of reinterpreting this result"
    ))
}

/// Which correction pair a listening condition compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ListeningArm {
    /// Corrected chain against its own baseline.
    BaselineVsCorrected,
    /// Pruned chain against the full chain it was pruned from.
    PrunedVsFull,
}

/// Source presentation of a listening condition. Mono coloration,
/// identical L+R summation and spatial reproduction answer different
/// questions and are never interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SourcePresentation {
    /// One speaker in mono: coloration without spatial confounds.
    SingleSpeakerMono,
    /// Identical signal on left and right summed acoustically.
    IdenticalLrSum,
    /// Intended stereo or multichannel spatial reproduction.
    Spatial,
}

impl SourcePresentation {
    /// Routing label used in condition identities.
    pub fn as_str(self) -> &'static str {
        match self {
            SourcePresentation::SingleSpeakerMono => "mono",
            SourcePresentation::IdenticalLrSum => "lr-sum",
            SourcePresentation::Spatial => "spatial",
        }
    }
}

/// One preregistered listening condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ListeningCondition {
    /// Correction pair under test.
    pub arm: ListeningArm,
    /// Source presentation under test.
    pub presentation: SourcePresentation,
    /// Programme material identity (hash or manifest id).
    pub programme_id: String,
    /// Seat identity.
    pub seat_id: String,
}

impl ListeningCondition {
    /// Canonical condition identity. The presentation is part of the id,
    /// so a mono condition and an otherwise identical L+R-sum condition
    /// can never collapse into one trial cell.
    pub fn condition_id(&self) -> String {
        format!(
            "{:?}/{}/programme-{}/seat-{}",
            self.arm,
            self.presentation.as_str(),
            self.programme_id,
            self.seat_id,
        )
    }
}

/// Immutable chain/stimulus identities a setup is bound to. Any change to
/// the rendered chains or the stimulus voids the binding: re-render and
/// re-preregister instead of reusing the protocol.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ChainStimulusBinding {
    /// Immutable baseline graph identity.
    pub baseline_graph_id: String,
    /// Immutable corrected candidate graph identity.
    pub candidate_graph_id: String,
    /// Immutable full chain the pruned arm derives from.
    pub full_graph_id: String,
    /// Immutable pruned chain identity, when the pruned arm is staged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pruned_graph_id: Option<String>,
    /// Hash of the workflow-rendered stimulus actually played.
    pub stimulus_hash: String,
    /// Playback sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Absolute playback level calibration identity.
    pub calibration_id: String,
    /// Processing state, e.g. `"final-delivered"`.
    pub processing_state: String,
}

impl ChainStimulusBinding {
    /// Fail closed when any bound identity differs from the current
    /// rendering. There is no tolerance on identities.
    pub fn verify_unchanged(&self, current: &Self) -> Result<(), String> {
        if self.baseline_graph_id != current.baseline_graph_id {
            return Err(String::from(
                "baseline chain changed after preregistration: binding invalid",
            ));
        }
        if self.candidate_graph_id != current.candidate_graph_id {
            return Err(String::from(
                "candidate chain changed after preregistration: binding invalid",
            ));
        }
        if self.full_graph_id != current.full_graph_id {
            return Err(String::from(
                "full chain changed after preregistration: binding invalid",
            ));
        }
        if self.pruned_graph_id != current.pruned_graph_id {
            return Err(String::from(
                "pruned chain changed after preregistration: binding invalid",
            ));
        }
        if self.stimulus_hash != current.stimulus_hash {
            return Err(String::from(
                "stimulus changed after preregistration: binding invalid",
            ));
        }
        if self.sample_rate_hz != current.sample_rate_hz {
            return Err(String::from(
                "sample rate changed after preregistration: binding invalid",
            ));
        }
        if self.calibration_id != current.calibration_id {
            return Err(String::from(
                "playback calibration changed after preregistration: binding invalid",
            ));
        }
        if self.processing_state != current.processing_state {
            return Err(String::from(
                "processing state changed after preregistration: binding invalid",
            ));
        }
        Ok(())
    }
}

/// One participant allocation frozen before a scheduled trial is run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TrialParticipantAssignment {
    /// Trial identity from the protocol's frozen presentation schedule.
    pub trial_id: String,
    /// Pseudonymous participant assigned before trial collection.
    pub participant_id: String,
}

/// A frozen listening setup: preregistered protocol plus the arms,
/// conditions, holdout programmes and chain/stimulus binding it covers.
/// Carries no outcome: outcomes arrive only through validated trial import.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ListeningSetup {
    /// Preregistered blinded protocol (hash-verified on validation).
    pub protocol: BlindedProtocol,
    /// Listening conditions; their ids must equal the protocol conditions.
    pub conditions: Vec<ListeningCondition>,
    /// Programme identities held out before model tuning.
    pub holdout_programmes: Vec<String>,
    /// Immutable chain/stimulus binding.
    pub binding: ChainStimulusBinding,
    /// Model and edition/version used to prepare the comparison.
    pub model_id: String,
    pub model_version: String,
    /// Declared listener population and current room; participants and
    /// rooms reserved for holdout must be fixed before any tuning.
    pub listener_population: String,
    pub room_id: String,
    #[serde(default)]
    pub holdout_rooms: Vec<String>,
    #[serde(default)]
    pub holdout_participants: Vec<String>,
    /// Pseudonymous participants enrolled in this trial round.
    #[serde(default)]
    pub participant_ids: Vec<String>,
    /// Frozen allocation of each scheduled trial to one enrolled participant.
    #[serde(default)]
    pub trial_participants: Vec<TrialParticipantAssignment>,
    /// Calibrated absolute playback level and how arm levels were matched.
    pub absolute_playback_level_db_spl: f64,
    pub level_matching_method: String,
    pub maximum_level_mismatch_db: f64,
    /// Explicit programme coverage, with IDs drawn from staged conditions.
    #[serde(default)]
    pub sustained_programmes: Vec<String>,
    #[serde(default)]
    pub transient_programmes: Vec<String>,
    #[serde(default)]
    pub representative_programmes: Vec<String>,
    /// SHA-256 over this setup, including the protocol and binding, with
    /// this field blank. Trial imports bind to this as well as protocol hash.
    #[serde(default)]
    pub setup_hash: String,
}

impl ListeningSetup {
    /// Freeze the complete setup before collection.
    pub fn freeze(mut self) -> Result<Self, String> {
        self.validate_structure()?;
        self.setup_hash = self.canonical_hash()?;
        Ok(self)
    }

    fn canonical_hash(&self) -> Result<String, String> {
        let mut value = self.clone();
        value.setup_hash.clear();
        let bytes = serde_json::to_vec(&value)
            .map_err(|error| format!("listening setup serialize error: {error}"))?;
        Ok(sha256_hex(&bytes))
    }

    /// Freeze validation: proves preregistration, binds every condition id
    /// to the protocol, requires both arms, distinct presentations, held-out
    /// programmes and complete binding identities.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_structure()?;
        if self.setup_hash != self.canonical_hash()? {
            return Err(String::from("listening setup changed after registration"));
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), String> {
        self.protocol.verify_prereg()?;
        if self.conditions.is_empty() {
            return Err(String::from("listening setup needs at least one condition"));
        }
        let mut ids: Vec<String> = self.conditions.iter().map(|c| c.condition_id()).collect();
        ids.sort();
        ids.dedup();
        if ids.len() != self.conditions.len() {
            return Err(String::from(
                "listening conditions collapse to duplicate identities: mono, L+R-sum and spatial presentations must stay distinct",
            ));
        }
        let mut protocol_conditions = self.protocol.conditions.clone();
        protocol_conditions.sort();
        if ids != protocol_conditions {
            return Err(String::from(
                "listening conditions must equal the preregistered protocol conditions",
            ));
        }
        let has_baseline = self
            .conditions
            .iter()
            .any(|c| c.arm == ListeningArm::BaselineVsCorrected);
        let has_pruned = self
            .conditions
            .iter()
            .any(|c| c.arm == ListeningArm::PrunedVsFull);
        if !has_baseline || !has_pruned {
            return Err(String::from(
                "listening setup must stage both baseline-vs-corrected and pruned-vs-full arms",
            ));
        }
        for condition in &self.conditions {
            if condition.programme_id.trim().is_empty() {
                return Err(String::from(
                    "listening conditions need programme identities",
                ));
            }
            if condition.seat_id.trim().is_empty() {
                return Err(String::from("listening conditions need seat identities"));
            }
        }
        if self.holdout_programmes.is_empty()
            || self
                .holdout_programmes
                .iter()
                .any(|programme| programme.trim().is_empty())
            || self
                .holdout_programmes
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != self.holdout_programmes.len()
        {
            return Err(String::from(
                "listening setup needs non-blank holdout programmes fixed before tuning",
            ));
        }
        for programme in &self.holdout_programmes {
            if self
                .conditions
                .iter()
                .any(|condition| &condition.programme_id == programme)
            {
                return Err(format!(
                    "holdout programme {programme} is also a trial programme: holdouts must stay disjoint from staged material"
                ));
            }
        }
        for (name, value) in [
            ("model id", &self.model_id),
            ("model version", &self.model_version),
            ("listener population", &self.listener_population),
            ("room id", &self.room_id),
            ("level matching method", &self.level_matching_method),
        ] {
            if value.trim().is_empty() {
                return Err(format!("listening setup needs a {name}"));
            }
        }
        if !self.absolute_playback_level_db_spl.is_finite()
            || self.absolute_playback_level_db_spl <= 0.0
            || !self.maximum_level_mismatch_db.is_finite()
            || self.maximum_level_mismatch_db < 0.0
        {
            return Err(String::from(
                "listening setup needs finite positive absolute playback level and nonnegative matching tolerance",
            ));
        }
        for (name, ids) in [
            ("holdout rooms", &self.holdout_rooms),
            ("holdout participants", &self.holdout_participants),
        ] {
            let mut unique = std::collections::HashSet::new();
            if ids.is_empty()
                || ids
                    .iter()
                    .any(|id| id.trim().is_empty() || !unique.insert(id))
            {
                return Err(format!("listening setup needs unique non-blank {name}"));
            }
        }
        if self.holdout_rooms.iter().any(|room| room == &self.room_id) {
            return Err(String::from("trial room cannot also be a held-out room"));
        }
        let mut participants = std::collections::HashSet::new();
        if self.participant_ids.is_empty()
            || self
                .participant_ids
                .iter()
                .any(|id| id.trim().is_empty() || !participants.insert(id.as_str()))
        {
            return Err(String::from(
                "listening setup needs unique non-blank participant IDs",
            ));
        }
        if self
            .holdout_participants
            .iter()
            .any(|id| participants.contains(id.as_str()))
        {
            return Err(String::from(
                "trial participants cannot also be held-out participants",
            ));
        }
        if self.protocol.trial_assignments.is_empty() {
            if !self.trial_participants.is_empty() {
                return Err(String::from(
                    "participant allocation needs a frozen protocol trial schedule",
                ));
            }
        } else {
            let scheduled: std::collections::HashSet<_> = self
                .protocol
                .trial_assignments
                .iter()
                .map(|assignment| assignment.trial_id.as_str())
                .collect();
            let mut allocated = std::collections::HashSet::new();
            let mut participant_conditions = std::collections::HashSet::new();
            if self.trial_participants.len() != scheduled.len() {
                return Err(String::from(
                    "every preregistered trial needs a participant allocation",
                ));
            }
            for allocation in &self.trial_participants {
                if !scheduled.contains(allocation.trial_id.as_str())
                    || !allocated.insert(allocation.trial_id.as_str())
                    || !participants.contains(allocation.participant_id.as_str())
                {
                    return Err(String::from(
                        "participant allocation has an unknown or repeated trial or participant",
                    ));
                }
                let condition = &self
                    .protocol
                    .trial_assignments
                    .iter()
                    .find(|assignment| assignment.trial_id == allocation.trial_id)
                    .expect("scheduled trial was checked above")
                    .condition;
                if !participant_conditions
                    .insert((condition.as_str(), allocation.participant_id.as_str()))
                {
                    return Err(String::from(
                        "a participant cannot repeat a trial within one condition under the exact binomial rule",
                    ));
                }
            }
        }
        for (name, ids) in [
            ("sustained", &self.sustained_programmes),
            ("transient", &self.transient_programmes),
            ("representative", &self.representative_programmes),
        ] {
            let mut unique = std::collections::HashSet::new();
            if ids.is_empty()
                || ids.iter().any(|id| {
                    id.trim().is_empty()
                        || !unique.insert(id)
                        || !self
                            .conditions
                            .iter()
                            .any(|condition| &condition.programme_id == id)
                })
            {
                return Err(format!(
                    "listening setup needs unique staged {name} programme IDs"
                ));
            }
        }
        for (name, value) in [
            ("baseline graph", &self.binding.baseline_graph_id),
            ("candidate graph", &self.binding.candidate_graph_id),
            ("full chain graph", &self.binding.full_graph_id),
            ("stimulus hash", &self.binding.stimulus_hash),
            ("calibration", &self.binding.calibration_id),
            ("processing state", &self.binding.processing_state),
        ] {
            if value.trim().is_empty() {
                return Err(format!("listening binding needs a {name} identity"));
            }
        }
        if !self.binding.sample_rate_hz.is_finite() || self.binding.sample_rate_hz <= 0.0 {
            return Err(String::from(
                "listening binding needs a positive sample rate",
            ));
        }
        if self.binding.baseline_graph_id == self.binding.candidate_graph_id
            || self
                .binding
                .pruned_graph_id
                .as_deref()
                .is_none_or(|pruned| {
                    pruned.trim().is_empty() || pruned == self.binding.full_graph_id
                })
        {
            return Err(String::from(
                "listening setup needs distinct baseline/candidate and pruned/full chains",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod listening_tests {
    use super::super::protocol::{
        BlindedProtocol, ComparisonDesign, ComparisonIntent, ComparisonSpec, DecisionRule,
        ReferenceKind,
    };
    use super::*;

    fn comparison(intent: ComparisonIntent) -> ComparisonSpec {
        ComparisonSpec {
            name: String::from("correction-vs-baseline-listening"),
            intent,
            attributes: match intent {
                ComparisonIntent::Preference => vec![String::from("timbre-preference")],
                ComparisonIntent::Equivalence => vec![String::from("detectability")],
                ComparisonIntent::Detectability => vec![String::from("detectability")],
            },
            equivalence_bound: match intent {
                ComparisonIntent::Equivalence => {
                    Some(String::from("ABX correct-response rate below 0.75"))
                }
                _ => None,
            },
            equivalence_max_p_correct: (intent == ComparisonIntent::Equivalence).then_some(0.75),
            reference: ReferenceKind::IndependentImplementation {
                description: String::from(
                    "second trial-analysis implementation agrees within 1e-12",
                ),
            },
            validated_domain: String::from("unvalidated"),
        }
    }

    fn protocol_for(conditions: &[ListeningCondition]) -> BlindedProtocol {
        let ids: Vec<String> = conditions.iter().map(|c| c.condition_id()).collect();
        BlindedProtocol::preregister(
            ComparisonDesign::Abx,
            comparison(ComparisonIntent::Detectability),
            ids,
            30,
            0.05,
            0.8,
            0.75,
            DecisionRule::Abx {
                min_correct: 20,
                trials: 30,
                alpha: 0.05,
            },
            11,
        )
        .unwrap()
    }

    fn conditions() -> Vec<ListeningCondition> {
        vec![
            ListeningCondition {
                arm: ListeningArm::BaselineVsCorrected,
                presentation: SourcePresentation::SingleSpeakerMono,
                programme_id: String::from("resonance-strings-01"),
                seat_id: String::from("seat-1"),
            },
            ListeningCondition {
                arm: ListeningArm::BaselineVsCorrected,
                presentation: SourcePresentation::IdenticalLrSum,
                programme_id: String::from("resonance-strings-01"),
                seat_id: String::from("seat-1"),
            },
            ListeningCondition {
                arm: ListeningArm::PrunedVsFull,
                presentation: SourcePresentation::Spatial,
                programme_id: String::from("transient-drums-02"),
                seat_id: String::from("seat-2"),
            },
        ]
    }

    fn binding() -> ChainStimulusBinding {
        ChainStimulusBinding {
            baseline_graph_id: String::from("graph-baseline-immutable"),
            candidate_graph_id: String::from("graph-candidate-immutable"),
            full_graph_id: String::from("graph-full-immutable"),
            pruned_graph_id: Some(String::from("graph-pruned-immutable")),
            stimulus_hash: String::from("rendered-stimulus-hash"),
            sample_rate_hz: 48_000.0,
            calibration_id: String::from("spl-cal-94db"),
            processing_state: String::from("final-delivered"),
        }
    }

    fn setup() -> ListeningSetup {
        let staged = conditions();
        ListeningSetup {
            protocol: protocol_for(&staged),
            conditions: staged,
            holdout_programmes: vec![String::from("heldout-piano-09")],
            binding: binding(),
            model_id: String::from("paired-auditory-model"),
            model_version: String::from("fixture-v1"),
            listener_population: String::from("trained adult listeners"),
            room_id: String::from("room-a"),
            holdout_rooms: vec![String::from("room-b")],
            holdout_participants: vec![String::from("participant-heldout-1")],
            participant_ids: vec![String::from("participant-0000")],
            trial_participants: Vec::new(),
            absolute_playback_level_db_spl: 75.0,
            level_matching_method: String::from("calibrated programme-integrated level"),
            maximum_level_mismatch_db: 0.2,
            sustained_programmes: vec![String::from("resonance-strings-01")],
            transient_programmes: vec![String::from("transient-drums-02")],
            representative_programmes: vec![String::from("resonance-strings-01")],
            setup_hash: String::new(),
        }
        .freeze()
        .unwrap()
    }

    #[test]
    fn protocol_preference_cannot_claim_equivalence() {
        // A preference setup validates as preference, but its result can
        // never be claimed as equivalence or inaudibility evidence.
        let preference = comparison(ComparisonIntent::Preference);
        assert!(preference.validate().is_ok());
        assert!(
            check_claim_matches_intent(ComparisonIntent::Preference, ComparisonIntent::Preference)
                .is_ok()
        );
        assert!(
            check_claim_matches_intent(ComparisonIntent::Preference, ComparisonIntent::Equivalence)
                .is_err()
        );
        assert!(
            check_claim_matches_intent(
                ComparisonIntent::Preference,
                ComparisonIntent::Detectability
            )
            .is_err()
        );
        // A passed detectability arm is likewise not an equivalence claim.
        assert!(
            check_claim_matches_intent(
                ComparisonIntent::Detectability,
                ComparisonIntent::Equivalence
            )
            .is_err()
        );
    }

    #[test]
    fn protocol_equivalence_requires_prespecified_bound() {
        // Equivalence without a prespecified detection bound fails: a
        // nonsignificant result alone proves nothing.
        let mut unbound = comparison(ComparisonIntent::Equivalence);
        unbound.equivalence_bound = None;
        assert!(unbound.validate().is_err());
        unbound.equivalence_bound = Some(String::from("   "));
        assert!(unbound.validate().is_err());
        assert!(comparison(ComparisonIntent::Equivalence).validate().is_ok());
    }

    #[test]
    fn protocol_changed_chain_or_stimulus_invalidates_binding() {
        let frozen = binding();
        assert!(frozen.verify_unchanged(&frozen).is_ok());
        let mut changed = frozen.clone();
        changed.candidate_graph_id = String::from("graph-candidate-retuned");
        assert!(frozen.verify_unchanged(&changed).is_err());
        let mut changed = frozen.clone();
        changed.stimulus_hash = String::from("other-render-hash");
        assert!(frozen.verify_unchanged(&changed).is_err());
        let mut changed = frozen.clone();
        changed.processing_state = String::from("preview");
        assert!(frozen.verify_unchanged(&changed).is_err());
        // The setup validates while bound; rebinding invalidates its hash.
        assert!(setup().validate().is_ok());
        let mut rebound = setup();
        rebound.binding = changed;
        assert!(rebound.validate().is_err());
        let mut tampered = setup();
        tampered.protocol.trials_per_condition = 40;
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn protocol_mono_source_and_lr_sum_distinct() {
        // Mono and identical L+R-sum are different enum values with
        // different routing labels and different condition identities.
        assert_ne!(
            SourcePresentation::SingleSpeakerMono,
            SourcePresentation::IdenticalLrSum
        );
        assert_ne!(
            SourcePresentation::SingleSpeakerMono.as_str(),
            SourcePresentation::IdenticalLrSum.as_str()
        );
        let mono = ListeningCondition {
            arm: ListeningArm::BaselineVsCorrected,
            presentation: SourcePresentation::SingleSpeakerMono,
            programme_id: String::from("resonance-strings-01"),
            seat_id: String::from("seat-1"),
        };
        let lr_sum = ListeningCondition {
            presentation: SourcePresentation::IdenticalLrSum,
            ..mono.clone()
        };
        assert_ne!(mono.condition_id(), lr_sum.condition_id());
        // The frozen setup keeps them as separate trial cells.
        let frozen = setup();
        assert!(frozen.validate().is_ok());
        let ids: Vec<String> = frozen.conditions.iter().map(|c| c.condition_id()).collect();
        assert!(ids.iter().any(|id| id.contains("/mono/")));
        assert!(ids.iter().any(|id| id.contains("/lr-sum/")));
    }

    #[test]
    fn protocol_preregistration_changes_hash() {
        // Any post-registration protocol change voids the stored hash, and
        // the listening setup refuses to validate until re-preregistered.
        let frozen = setup();
        assert!(frozen.validate().is_ok());
        let mut edited = frozen.clone();
        edited
            .protocol
            .conditions
            .push(String::from("extra-condition"));
        assert!(edited.validate().is_err());
        let mut edited = frozen.clone();
        edited.protocol.randomization_seed = 999;
        assert!(edited.validate().is_err());
        // Re-preregistering identical content reproduces the hash.
        let staged = conditions();
        assert_eq!(
            protocol_for(&staged).prereg_hash,
            frozen.protocol.prereg_hash
        );
    }

    #[test]
    fn listening_setup_freezes_level_population_programmes_and_binding() {
        let frozen = setup();
        let json = serde_json::to_vec(&frozen).unwrap();
        let restored: ListeningSetup = serde_json::from_slice(&json).unwrap();
        assert!(restored.validate().is_ok());
        let mut changed = frozen.clone();
        changed.absolute_playback_level_db_spl = 80.0;
        assert!(changed.validate().is_err());
        let mut changed = frozen.clone();
        changed.listener_population = String::from("different population");
        assert!(changed.validate().is_err());
        let mut changed = frozen.clone();
        changed.binding.stimulus_hash = String::from("different stimulus");
        assert!(changed.validate().is_err());
        let mut changed = frozen.clone();
        changed.participant_ids[0] = String::from("different-participant");
        assert!(changed.validate().is_err());
        let mut changed = frozen.clone();
        changed.transient_programmes.clear();
        assert!(changed.freeze().is_err());
        let mut changed = frozen.clone();
        changed.holdout_rooms = vec![changed.room_id.clone()];
        assert!(changed.freeze().is_err());
        let mut changed = frozen;
        changed.maximum_level_mismatch_db = f64::NAN;
        assert!(changed.freeze().is_err());
    }
}
