//! Three-condition listening battery (Wave 3, step 7).
//!
//! A battery stages single-speaker mono coloration, intended spatial
//! reproduction, and identical L+R summation as separate trial cells,
//! across resonance-revealing, transient, and representative programme
//! material. Comparisons are level-matched at a recorded absolute
//! playback level, randomized under a concealed mapping, and sized by
//! preregistered bounds, power, and trial counts. A nonsignificant ABX
//! never proves equivalence, and synthetic verdicts never promote a
//! listening claim: only real successes under the preregistered rule
//! qualify.

// Rust guideline compliant 2026-02-21

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::listening::{ListeningArm, ListeningCondition, ListeningSetup, SourcePresentation};
use super::protocol::ComparisonIntent;
use super::sha256_hex;
use super::trial_import::{
    ClaimVerdict, TrialImport, qualifies_for_listening_claim, summarize_claims,
    validate_trial_import_with_setup,
};

/// Material class of one battery cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MaterialClass {
    /// Sustained resonance-revealing material (strings, piano, organ).
    ResonanceSustained,
    /// Transient material (drums, clicks, plosive speech).
    Transient,
    /// Representative speech/music programme.
    RepresentativeProgramme,
}

/// Level matching of one battery cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LevelMatch {
    /// Matching method, e.g. `"loudness-matched-at-1khz"`.
    pub matching_method: String,
    /// Residual mismatch bound in dB the match guarantees.
    pub matched_within_db: f64,
    /// Absolute playback level in dB SPL the match was verified at.
    pub absolute_level_db_spl: f64,
    /// Absolute playback calibration identity.
    pub calibration_id: String,
}

impl LevelMatch {
    /// Validate the match record.
    ///
    /// # Errors
    ///
    /// Returns an error on blank provenance, non-finite levels, or a
    /// negative mismatch bound.
    pub fn validate(&self) -> Result<(), String> {
        if self.matching_method.trim().is_empty() || self.calibration_id.trim().is_empty() {
            return Err(String::from(
                "battery cells need a matching method and a calibration identity",
            ));
        }
        if !self.matched_within_db.is_finite() || self.matched_within_db < 0.0 {
            return Err(String::from(
                "battery mismatch bound must be finite and non-negative",
            ));
        }
        if !self.absolute_level_db_spl.is_finite() {
            return Err(String::from(
                "battery cells need a finite absolute playback level",
            ));
        }
        Ok(())
    }
}

/// One staged battery cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BatteryCell {
    /// Correction pair under test; part of the frozen condition identity.
    pub arm: ListeningArm,
    /// Source presentation under test.
    pub presentation: SourcePresentation,
    /// Material class under test.
    pub material: MaterialClass,
    /// Programme material identity.
    pub programme_id: String,
    /// Seat identity.
    pub seat_id: String,
    /// What the cell is staged to show.
    pub intent: ComparisonIntent,
    /// Prespecified detection bound (required for equivalence cells).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub equivalence_bound: Option<String>,
    /// Level matching at a recorded absolute playback level.
    pub level_match: LevelMatch,
    /// Preregistration hash of the protocol scoring this cell.
    pub protocol_hash: String,
    /// Hash of the complete frozen listening setup for this trial round.
    pub setup_hash: String,
}

impl BatteryCell {
    /// Validate one cell.
    ///
    /// # Errors
    ///
    /// Returns an error on blank identities, an unvalidated level
    /// match, or an equivalence cell without its prespecified bound.
    pub fn validate(&self) -> Result<(), String> {
        if self.programme_id.trim().is_empty() || self.seat_id.trim().is_empty() {
            return Err(String::from(
                "battery cells need programme and seat identities",
            ));
        }
        if self.protocol_hash.trim().is_empty() || self.setup_hash.trim().is_empty() {
            return Err(String::from(
                "battery cells need protocol and frozen setup hashes",
            ));
        }
        self.level_match.validate()?;
        if self.intent == ComparisonIntent::Equivalence
            && self
                .equivalence_bound
                .as_deref()
                .is_none_or(|bound| bound.trim().is_empty())
        {
            return Err(String::from(
                "equivalence cells need a prespecified detection bound: a nonsignificant ABX alone proves nothing",
            ));
        }
        Ok(())
    }
}

/// A staged listening battery: protocol scaffolding, never an outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ListeningBattery {
    /// Battery cells in staging order.
    pub cells: Vec<BatteryCell>,
    /// Seed for trial randomization.
    pub randomization_seed: u64,
    /// Whether presentation order is concealed from listeners and runners.
    pub randomization_concealed: bool,
    /// One-sided significance level shared by the battery protocols.
    pub alpha: f64,
    /// Target statistical power shared by the battery protocols.
    pub target_power: f64,
    /// Trials per condition shared by the battery protocols.
    pub trials_per_condition: u32,
}

impl ListeningBattery {
    /// Validate battery coverage and preregistration.
    ///
    /// Requires all three presentations, all three material classes, a
    /// valid level match per cell, concealed randomization, and sane
    /// sizing (alpha, power, trial counts).
    ///
    /// # Errors
    ///
    /// Returns an error when coverage, matching, concealment, or sizing
    /// is incomplete.
    pub fn validate(&self) -> Result<(), String> {
        if self.cells.is_empty() {
            return Err(String::from("listening battery needs at least one cell"));
        }
        for cell in &self.cells {
            cell.validate()?;
        }
        for presentation in [
            SourcePresentation::SingleSpeakerMono,
            SourcePresentation::Spatial,
            SourcePresentation::IdenticalLrSum,
        ] {
            if !self
                .cells
                .iter()
                .any(|cell| cell.presentation == presentation)
            {
                return Err(format!(
                    "listening battery misses the {presentation:?} condition: mono, spatial, and L+R summation stay separate"
                ));
            }
        }
        for material in [
            MaterialClass::ResonanceSustained,
            MaterialClass::Transient,
            MaterialClass::RepresentativeProgramme,
        ] {
            if !self.cells.iter().any(|cell| cell.material == material) {
                return Err(format!(
                    "listening battery misses {material:?} material: resonance, transient, and programme material stay covered"
                ));
            }
        }
        if !self.randomization_concealed {
            return Err(String::from(
                "listening battery needs concealed randomization of conditions",
            ));
        }
        if !(0.0 < self.alpha && self.alpha < 1.0) {
            return Err(String::from("battery alpha must lie in (0, 1)"));
        }
        if !(0.0 < self.target_power && self.target_power < 1.0) {
            return Err(String::from("battery target power must lie in (0, 1)"));
        }
        if self.trials_per_condition == 0 || self.trials_per_condition > 5_000 {
            return Err(String::from(
                "battery trials per condition stay inside the 1-5000 exact range",
            ));
        }
        Ok(())
    }

    /// Bind every battery cell to exactly one frozen setup condition.
    ///
    /// A battery's own setup hashes and level declarations are only
    /// claims until checked against the preregistered setup records.
    pub fn validate_against_setups(&self, setups: &[ListeningSetup]) -> Result<(), String> {
        self.validate()?;
        if setups.is_empty() {
            return Err(String::from("listening battery needs frozen setups"));
        }
        let mut hashes = std::collections::HashSet::new();
        for setup in setups {
            setup.validate()?;
            if !hashes.insert(setup.setup_hash.as_str()) {
                return Err(String::from("listening battery repeats a frozen setup"));
            }
        }
        let mut covered = std::collections::HashSet::new();
        for cell in &self.cells {
            let setup = setups
                .iter()
                .find(|setup| setup.setup_hash == cell.setup_hash)
                .ok_or_else(|| {
                    format!(
                        "battery cell has no frozen setup for hash {}",
                        cell.setup_hash
                    )
                })?;
            if cell.protocol_hash != setup.protocol.prereg_hash
                || cell.intent != setup.protocol.comparison.intent
                || self.randomization_seed != setup.protocol.randomization_seed
                || self.alpha != setup.protocol.alpha
                || self.target_power != setup.protocol.target_power
                || self.trials_per_condition != setup.protocol.trials_per_condition
            {
                return Err(String::from(
                    "battery intent, randomization or trial sizing differs from its frozen protocol",
                ));
            }
            if cell.level_match.matching_method != setup.level_matching_method
                || cell.level_match.matched_within_db > setup.maximum_level_mismatch_db
                || cell.level_match.absolute_level_db_spl != setup.absolute_playback_level_db_spl
                || cell.level_match.calibration_id != setup.binding.calibration_id
            {
                return Err(String::from(
                    "battery level match differs from its frozen setup",
                ));
            }
            if cell.equivalence_bound != setup.protocol.comparison.equivalence_bound {
                return Err(String::from(
                    "battery equivalence bound differs from its frozen protocol",
                ));
            }
            let condition = setup
                .conditions
                .iter()
                .find(|condition| {
                    condition.arm == cell.arm
                        && condition.presentation == cell.presentation
                        && condition.programme_id == cell.programme_id
                        && condition.seat_id == cell.seat_id
                })
                .ok_or_else(|| String::from("battery cell is absent from its frozen setup"))?;
            let allowed_programmes = match cell.material {
                MaterialClass::ResonanceSustained => &setup.sustained_programmes,
                MaterialClass::Transient => &setup.transient_programmes,
                MaterialClass::RepresentativeProgramme => &setup.representative_programmes,
            };
            if !allowed_programmes.contains(&cell.programme_id) {
                return Err(String::from(
                    "battery material class differs from its frozen programme scope",
                ));
            }
            if !covered.insert((cell.setup_hash.as_str(), condition.condition_id())) {
                return Err(String::from("battery repeats a frozen setup condition"));
            }
        }
        for setup in setups {
            for condition in &setup.conditions {
                if !covered.contains(&(setup.setup_hash.as_str(), condition.condition_id())) {
                    return Err(String::from("battery misses a frozen setup condition"));
                }
            }
        }
        Ok(())
    }
}

/// Battery-level verdict over scored cells.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BatteryVerdict {
    /// Per-cell verdicts in battery order.
    pub cells: Vec<ClaimVerdict>,
    /// Whether every cell is a real success under its preregistered rule.
    pub all_success: bool,
    /// Whether an equivalence claim is supported (never on negatives).
    pub equivalence_supported: bool,
    /// Content hashes of validated typed imports, in setup order.
    /// Empty for verdict-only descriptive scoring.
    #[serde(default)]
    pub import_sha256: Vec<String>,
}

/// Score a battery from validated per-cell verdicts.
///
/// Cells that miss significance stay inconclusive or failed, never
/// equivalence evidence; synthetic verdicts never promote. Real data
/// collection itself is operator work outside this crate.
///
/// # Errors
///
/// Returns an error when no verdicts are scored.
pub fn score_battery(verdicts: &[ClaimVerdict]) -> Result<BatteryVerdict, String> {
    if verdicts.is_empty() {
        return Err(String::from("battery scoring needs at least one verdict"));
    }
    let all_success = qualifies_for_listening_claim(verdicts).is_ok();
    Ok(BatteryVerdict {
        cells: verdicts.to_vec(),
        all_success,
        equivalence_supported: check_battery_equivalence(verdicts).is_ok(),
        import_sha256: Vec::new(),
    })
}

/// Recompute a battery verdict from raw imports bound to frozen setups.
///
/// Exactly one import is required per setup. All cells are returned in
/// battery order, and the imported rows are validated before scoring.
/// An operator must still establish that rows marked real came from actual
/// listeners; this function cannot authenticate the person or session.
pub fn score_battery_imports(
    battery: &ListeningBattery,
    setups: &[ListeningSetup],
    imports: &[TrialImport],
) -> Result<BatteryVerdict, String> {
    battery.validate_against_setups(setups)?;
    if imports.len() != setups.len() {
        return Err(String::from(
            "battery needs exactly one trial import per setup",
        ));
    }
    let mut verdicts_by_condition = std::collections::HashMap::new();
    let mut import_sha256 = Vec::with_capacity(setups.len());
    for setup in setups {
        let matching: Vec<_> = imports
            .iter()
            .filter(|import| import.setup_hash.as_deref() == Some(setup.setup_hash.as_str()))
            .collect();
        if matching.len() != 1 {
            return Err(String::from(
                "battery needs one import for each frozen setup",
            ));
        }
        let validated = validate_trial_import_with_setup(matching[0], setup)?;
        let import_bytes = serde_json::to_vec(matching[0])
            .map_err(|error| format!("battery trial import serialization failed: {error}"))?;
        import_sha256.push(format!(
            "{}:{}",
            setup.setup_hash,
            sha256_hex(&import_bytes)
        ));
        for verdict in summarize_claims(&validated, &setup.protocol)? {
            if verdicts_by_condition
                .insert(
                    (setup.setup_hash.as_str(), verdict.condition.clone()),
                    verdict,
                )
                .is_some()
            {
                return Err(String::from("battery trial import repeats a condition"));
            }
        }
    }
    let mut ordered = Vec::with_capacity(battery.cells.len());
    for cell in &battery.cells {
        let condition = ListeningCondition {
            arm: cell.arm,
            presentation: cell.presentation,
            programme_id: cell.programme_id.clone(),
            seat_id: cell.seat_id.clone(),
        }
        .condition_id();
        let verdict = verdicts_by_condition
            .remove(&(cell.setup_hash.as_str(), condition))
            .ok_or_else(|| String::from("battery import misses a staged cell"))?;
        ordered.push(verdict);
    }
    if !verdicts_by_condition.is_empty() {
        return Err(String::from("battery import contains unstaged conditions"));
    }
    let mut scored = score_battery(&ordered)?;
    // The verdict-only scorer requires one common setup hash because it
    // cannot verify provenance. Here each round was checked against its own
    // frozen setup and raw import, so separate rooms may share a protocol.
    scored.all_success = ordered.iter().all(|verdict| {
        !verdict.synthetic
            && verdict
                .setup_hash
                .as_deref()
                .is_some_and(|hash| !hash.trim().is_empty())
            && verdict.decision == "success"
    });
    scored.equivalence_supported = scored.all_success
        && ordered.iter().all(|verdict| {
            verdict.intent == Some(ComparisonIntent::Equivalence)
                && verdict
                    .equivalence_bound_p_correct
                    .is_some_and(|bound| bound.is_finite() && (0.5..1.0).contains(&bound))
        });
    scored.import_sha256 = import_sha256;
    Ok(scored)
}

/// Check whether scored verdicts support an equivalence claim.
///
/// Equivalence needs every real cell to succeed under the numeric ABX
/// equivalence rule. A successful detectability table cannot satisfy it.
///
/// # Errors
///
/// Returns an error on empty, synthetic, non-success or non-equivalence cells.
pub fn check_battery_equivalence(verdicts: &[ClaimVerdict]) -> Result<(), String> {
    qualifies_for_listening_claim(verdicts).map_err(|error| {
        format!("battery equivalence unsupported: {error}: a nonsignificant ABX never proves equivalence")
    })?;
    if verdicts.iter().any(|verdict| {
        verdict.intent != Some(ComparisonIntent::Equivalence)
            || verdict
                .equivalence_bound_p_correct
                .is_none_or(|bound| !bound.is_finite() || !(0.5..1.0).contains(&bound))
    }) {
        return Err(String::from(
            "battery equivalence needs real successes from a numeric preregistered equivalence rule",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod battery_tests {
    use super::*;
    use crate::{
        AbxAnswer, BlindedProtocol, ChainStimulusBinding, ComparisonDesign, ComparisonSpec,
        DecisionRule, ImportedTrial, ListeningCondition, ReferenceKind, TrialParticipantAssignment,
    };

    fn matched() -> LevelMatch {
        LevelMatch {
            matching_method: String::from("loudness-matched-at-1khz"),
            matched_within_db: 0.2,
            absolute_level_db_spl: 76.0,
            calibration_id: String::from("spl-cal-94db"),
        }
    }

    fn cell(
        presentation: SourcePresentation,
        material: MaterialClass,
        intent: ComparisonIntent,
    ) -> BatteryCell {
        BatteryCell {
            arm: if presentation == SourcePresentation::Spatial {
                ListeningArm::PrunedVsFull
            } else {
                ListeningArm::BaselineVsCorrected
            },
            presentation,
            material,
            programme_id: String::from("programme-01"),
            seat_id: String::from("seat-1"),
            intent,
            equivalence_bound: match intent {
                ComparisonIntent::Equivalence => Some(String::from("d-prime below 0.5")),
                _ => None,
            },
            level_match: matched(),
            protocol_hash: String::from("prereg-hash"),
            setup_hash: String::from("frozen-setup-hash"),
        }
    }

    fn battery() -> ListeningBattery {
        ListeningBattery {
            cells: vec![
                cell(
                    SourcePresentation::SingleSpeakerMono,
                    MaterialClass::ResonanceSustained,
                    ComparisonIntent::Detectability,
                ),
                cell(
                    SourcePresentation::Spatial,
                    MaterialClass::Transient,
                    ComparisonIntent::Detectability,
                ),
                cell(
                    SourcePresentation::IdenticalLrSum,
                    MaterialClass::RepresentativeProgramme,
                    ComparisonIntent::Detectability,
                ),
            ],
            randomization_seed: 11,
            randomization_concealed: true,
            alpha: 0.05,
            target_power: 0.8,
            trials_per_condition: 30,
        }
    }

    fn bound_battery() -> (ListeningBattery, ListeningSetup) {
        let mut battery = battery();
        for (index, cell) in battery.cells.iter_mut().enumerate() {
            cell.programme_id = format!("programme-{:02}", index + 1);
        }
        let conditions: Vec<_> = battery
            .cells
            .iter()
            .map(|cell| ListeningCondition {
                arm: cell.arm,
                presentation: cell.presentation,
                programme_id: cell.programme_id.clone(),
                seat_id: cell.seat_id.clone(),
            })
            .collect();
        let protocol = BlindedProtocol::preregister(
            ComparisonDesign::Abx,
            ComparisonSpec {
                name: String::from("battery-detectability"),
                intent: ComparisonIntent::Detectability,
                attributes: vec![String::from("detectability")],
                equivalence_bound: None,
                equivalence_max_p_correct: None,
                reference: ReferenceKind::IndependentImplementation {
                    description: String::from("separate binomial reference within 1e-12"),
                },
                validated_domain: String::from("unvalidated"),
            },
            conditions
                .iter()
                .map(ListeningCondition::condition_id)
                .collect(),
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
        .unwrap();
        for cell in &mut battery.cells {
            cell.protocol_hash = protocol.prereg_hash.clone();
        }
        let setup = ListeningSetup {
            protocol,
            conditions,
            holdout_programmes: vec![String::from("heldout-02")],
            binding: ChainStimulusBinding {
                baseline_graph_id: String::from("baseline-graph"),
                candidate_graph_id: String::from("candidate-graph"),
                full_graph_id: String::from("full-graph"),
                pruned_graph_id: Some(String::from("pruned-graph")),
                stimulus_hash: String::from("rendered-stimulus"),
                sample_rate_hz: 48_000.0,
                calibration_id: String::from("spl-cal-94db"),
                processing_state: String::from("final-delivered"),
            },
            model_id: String::from("fixture-model"),
            model_version: String::from("fixture-v1"),
            listener_population: String::from("adult listeners"),
            room_id: String::from("room-a"),
            holdout_rooms: vec![String::from("room-b")],
            holdout_participants: vec![String::from("heldout-listener")],
            participant_ids: vec![String::from("listener-01")],
            trial_participants: Vec::new(),
            absolute_playback_level_db_spl: 76.0,
            level_matching_method: String::from("loudness-matched-at-1khz"),
            maximum_level_mismatch_db: 0.2,
            sustained_programmes: vec![String::from("programme-01")],
            transient_programmes: vec![String::from("programme-02")],
            representative_programmes: vec![String::from("programme-03")],
            setup_hash: String::new(),
        }
        .freeze()
        .unwrap();
        for cell in &mut battery.cells {
            cell.setup_hash = setup.setup_hash.clone();
        }
        (battery, setup)
    }

    #[test]
    fn battery_cells_bind_to_frozen_setup_conditions_and_levels() {
        let (battery, setup) = bound_battery();
        assert!(battery.validate_against_setups(&[setup.clone()]).is_ok());
        let mut stale = battery.clone();
        stale.cells[0].protocol_hash = String::from("stale-hash");
        assert!(stale.validate_against_setups(&[setup.clone()]).is_err());
        let mut wrong_arm = battery.clone();
        wrong_arm.cells[0].arm = ListeningArm::PrunedVsFull;
        assert!(wrong_arm.validate_against_setups(&[setup.clone()]).is_err());
        let mut wrong_level = battery.clone();
        wrong_level.cells[0].level_match.absolute_level_db_spl += 1.0;
        assert!(
            wrong_level
                .validate_against_setups(&[setup.clone()])
                .is_err()
        );
        let mut wrong_material = battery.clone();
        wrong_material.cells[0].material = MaterialClass::Transient;
        wrong_material.cells[1].material = MaterialClass::ResonanceSustained;
        assert!(
            wrong_material
                .validate_against_setups(&[setup.clone()])
                .is_err()
        );
        let mut repeated = battery.clone();
        repeated.cells.push(repeated.cells[0].clone());
        assert!(repeated.validate_against_setups(&[setup.clone()]).is_err());
        let mut tampered = setup;
        tampered.binding.stimulus_hash = String::from("changed-stimulus");
        assert!(battery.validate_against_setups(&[tampered]).is_err());
    }

    #[test]
    fn battery_scores_only_complete_setup_bound_trial_imports() {
        let (mut battery, mut setup) = bound_battery();
        let assignments = setup.protocol.abx_assignment_template().unwrap();
        setup.protocol = setup.protocol.with_trial_assignments(assignments).unwrap();
        setup.participant_ids = (0..30)
            .map(|order| format!("participant-{order:04}"))
            .collect();
        setup.trial_participants = setup
            .protocol
            .trial_assignments
            .iter()
            .map(|assignment| TrialParticipantAssignment {
                trial_id: assignment.trial_id.clone(),
                participant_id: format!("participant-{:04}", assignment.presentation_order),
            })
            .collect();
        setup = setup.freeze().unwrap();
        for cell in &mut battery.cells {
            cell.protocol_hash = setup.protocol.prereg_hash.clone();
            cell.setup_hash = setup.setup_hash.clone();
        }
        let rows = setup
            .protocol
            .trial_assignments
            .iter()
            .map(|assignment| {
                let correct = assignment.presentation_order < 20;
                ImportedTrial {
                    trial_id: assignment.trial_id.clone(),
                    participant_id: format!("participant-{:04}", assignment.presentation_order),
                    condition: assignment.condition.clone(),
                    presentation_order: assignment.presentation_order,
                    correct,
                    response: Some(if correct {
                        assignment.answer
                    } else if assignment.answer == AbxAnswer::A {
                        AbxAnswer::B
                    } else {
                        AbxAnswer::A
                    }),
                }
            })
            .collect();
        let mut import = TrialImport {
            protocol_hash: setup.protocol.prereg_hash.clone(),
            setup_hash: Some(setup.setup_hash.clone()),
            comparison: setup.protocol.comparison.name.clone(),
            claimed_intent: Some(ComparisonIntent::Detectability),
            binding: setup.binding.clone(),
            synthetic: false,
            rows,
        };
        let scored = score_battery_imports(&battery, &[setup.clone()], &[import.clone()]).unwrap();
        assert!(scored.all_success);
        assert_eq!(scored.cells.len(), battery.cells.len());
        assert_eq!(scored.import_sha256.len(), 1);
        assert!(scored.import_sha256[0].starts_with(&setup.setup_hash));
        assert!(
            scored
                .cells
                .iter()
                .zip(&battery.cells)
                .all(|(verdict, cell)| {
                    verdict.condition
                        == ListeningCondition {
                            arm: cell.arm,
                            presentation: cell.presentation,
                            programme_id: cell.programme_id.clone(),
                            seat_id: cell.seat_id.clone(),
                        }
                        .condition_id()
                })
        );
        let mut second_setup = setup.clone();
        second_setup.room_id = String::from("room-c");
        second_setup = second_setup.freeze().unwrap();
        let mut second_import = import.clone();
        second_import.setup_hash = Some(second_setup.setup_hash.clone());
        let mut two_rounds = battery.clone();
        let mut second_cells = battery.cells.clone();
        for cell in &mut second_cells {
            cell.setup_hash = second_setup.setup_hash.clone();
        }
        two_rounds.cells.extend(second_cells);
        let paired = score_battery_imports(
            &two_rounds,
            &[setup.clone(), second_setup],
            &[import.clone(), second_import],
        )
        .unwrap();
        assert!(paired.all_success);
        assert_eq!(paired.import_sha256.len(), 2);
        import.synthetic = true;
        assert!(
            !score_battery_imports(&battery, &[setup.clone()], &[import.clone()])
                .unwrap()
                .all_success
        );
        import.synthetic = false;
        import.rows.pop();
        assert!(score_battery_imports(&battery, &[setup], &[import]).is_err());
    }

    fn verdict(condition: &str, correct: u32, synthetic: bool) -> ClaimVerdict {
        ClaimVerdict {
            condition: String::from(condition),
            trials: 30,
            correct,
            effect_rate: f64::from(correct) / 30.0,
            ci95: [0.4, 0.8],
            p_value: 0.02,
            reference_p_value: None,
            setup_hash: Some(String::from("fixture-frozen-setup")),
            intent: Some(ComparisonIntent::Detectability),
            equivalence_bound_p_correct: None,
            decision: if correct >= 20 {
                String::from("success")
            } else {
                String::from("inconclusive")
            },
            synthetic,
        }
    }

    #[test]
    fn battery_requires_three_presentations() {
        assert!(battery().validate().is_ok());
        let mut missing = battery();
        missing
            .cells
            .retain(|cell| cell.presentation != SourcePresentation::IdenticalLrSum);
        let error = missing.validate().expect_err("L+R-sum missing");
        assert!(error.contains("IdenticalLrSum"), "{error}");
        let mut mono_only = battery();
        for cell in &mut mono_only.cells {
            cell.presentation = SourcePresentation::SingleSpeakerMono;
        }
        assert!(mono_only.validate().is_err());
    }

    #[test]
    fn battery_requires_material_classes() {
        assert!(battery().validate().is_ok());
        let mut missing = battery();
        for cell in &mut missing.cells {
            cell.material = MaterialClass::ResonanceSustained;
        }
        let error = missing.validate().expect_err("transients missing");
        assert!(error.contains("Transient"), "{error}");
    }

    #[test]
    fn battery_level_match_required() {
        assert!(battery().validate().is_ok());
        let mut unmatched = battery();
        unmatched.cells[0].level_match.matched_within_db = f64::NAN;
        assert!(unmatched.validate().is_err());
        let mut unconcealed = battery();
        unconcealed.randomization_concealed = false;
        assert!(unconcealed.validate().is_err());
        let mut unbound = battery();
        unbound.cells[0].intent = ComparisonIntent::Equivalence;
        unbound.cells[0].equivalence_bound = None;
        assert!(unbound.validate().is_err());
    }

    #[test]
    fn battery_nonsignificant_never_equivalence() {
        let verdicts = vec![
            verdict("mono", 22, false),
            verdict("spatial", 12, false),
            verdict("lr-sum", 21, false),
        ];
        let scored = score_battery(&verdicts).unwrap();
        assert!(!scored.all_success);
        assert!(!scored.equivalence_supported);
        assert!(check_battery_equivalence(&verdicts).is_err());
        let passing = vec![
            verdict("mono", 22, false),
            verdict("spatial", 23, false),
            verdict("lr-sum", 21, false),
        ];
        let scored = score_battery(&passing).unwrap();
        assert!(scored.all_success);
        assert!(!scored.equivalence_supported);
        assert!(check_battery_equivalence(&passing).is_err());
    }

    #[test]
    fn battery_synthetic_never_promotes() {
        let synthetic = vec![
            verdict("mono", 30, true),
            verdict("spatial", 30, true),
            verdict("lr-sum", 30, true),
        ];
        let scored = score_battery(&synthetic).unwrap();
        assert!(!scored.all_success);
        assert!(check_battery_equivalence(&synthetic).is_err());
    }
}
