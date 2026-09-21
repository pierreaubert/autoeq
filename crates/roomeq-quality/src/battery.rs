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

use super::listening::SourcePresentation;
use super::protocol::ComparisonIntent;
use super::trial_import::{ClaimVerdict, qualifies_for_listening_claim};

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
        if self.protocol_hash.trim().is_empty() {
            return Err(String::from(
                "battery cells need their protocol preregistration hash",
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
        equivalence_supported: false,
    })
}

/// Check whether scored verdicts support an equivalence claim.
///
/// Equivalence needs every equivalence cell to meet its prespecified
/// bound with real successes; this scaffolding records the verdicts
/// and refuses the claim until that evidence exists. Synthetic tables
/// fail closed here.
///
/// # Errors
///
/// Returns an error on empty input, any non-success, or any synthetic
/// verdict: a nonsignificant ABX never proves equivalence.
pub fn check_battery_equivalence(verdicts: &[ClaimVerdict]) -> Result<(), String> {
    qualifies_for_listening_claim(verdicts).map_err(|error| {
        format!("battery equivalence unsupported: {error}: a nonsignificant ABX never proves equivalence")
    })
}

#[cfg(test)]
mod battery_tests {
    use super::*;

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

    fn verdict(condition: &str, correct: u32, synthetic: bool) -> ClaimVerdict {
        ClaimVerdict {
            condition: String::from(condition),
            trials: 30,
            correct,
            effect_rate: f64::from(correct) / 30.0,
            ci95: [0.4, 0.8],
            p_value: 0.02,
            reference_p_value: None,
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
