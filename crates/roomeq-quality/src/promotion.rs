//! Perceptual enforcement readiness from the evidence side (Wave 3, step 6).
//!
//! The optimizer owns the fidelity-model pin; this module owns the
//! evidence the pin must reproduce before enforcement. A programme set
//! keeps tuning and held-out material disjoint, control outcomes record
//! all four validation families, and the readiness gate stays
//! `blocked_external` while independent references, calibration, domain,
//! or tolerances are missing. Equal total loudness never supports an
//! equivalence or preference claim.

// Rust guideline compliant 2026-02-21

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::protocol::ComparisonIntent;

/// Prefix marking an outcome blocked on external inputs.
pub const BLOCKED_EXTERNAL_PREFIX: &str = "blocked_external";

/// One programme in a validation set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ProgrammeEntry {
    /// Programme identity.
    pub id: String,
    /// Hash of the rendered stimulus bytes actually heard.
    pub stimulus_hash: String,
}

/// Tuning versus held-out programme split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ProgrammeSet {
    /// Programmes the model was tuned against.
    pub tuning: Vec<ProgrammeEntry>,
    /// Programmes held out before any tuning.
    pub held_out: Vec<ProgrammeEntry>,
}

impl ProgrammeSet {
    /// Verify entries and holdout discipline.
    ///
    /// # Errors
    ///
    /// Returns an error on blank identities or hashes, an empty holdout
    /// set, or any holdout that also ran in tuning.
    pub fn verify_holdout_disjoint(&self) -> Result<(), String> {
        if self.held_out.is_empty() {
            return Err(String::from(
                "programme set needs at least one held-out programme",
            ));
        }
        for entry in self.tuning.iter().chain(self.held_out.iter()) {
            if entry.id.trim().is_empty() || entry.stimulus_hash.trim().is_empty() {
                return Err(String::from(
                    "programme entries need identities and stimulus hashes",
                ));
            }
        }
        for held in &self.held_out {
            if self.tuning.iter().any(|tuned| tuned.id == held.id) {
                return Err(format!(
                    "held-out programme {} also ran in tuning: holdouts stay disjoint",
                    held.id
                ));
            }
        }
        Ok(())
    }
}

/// Validation-control family required before enforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ControlFamily {
    /// Level sweeps across the declared domain.
    LevelSweep,
    /// Bandwidth-change comparisons.
    Bandwidth,
    /// Equal-loudness, different-timbre controls.
    EqualLoudnessTimbre,
    /// Held-out programme checks.
    HeldOutProgramme,
}

/// Recorded outcome of one validation control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ControlOutcome {
    /// Control family exercised.
    pub family: ControlFamily,
    /// Condition identity the control ran under.
    pub condition: String,
    /// Whether the control met its predeclared tolerance.
    pub passed: bool,
    /// Observed quantities and the tolerance applied.
    pub detail: String,
}

/// Summary of one control family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ControlFamilySummary {
    /// Control family summarized.
    pub family: ControlFamily,
    /// Controls run in this family.
    pub controls: usize,
    /// Whether every control in the family passed.
    pub all_passed: bool,
}

/// Require every control family before enforcement evidence counts.
///
/// # Errors
///
/// Returns an error when any of the four families has no recorded
/// control, or when a record carries a blank condition or detail.
pub fn summarize_controls(
    outcomes: &[ControlOutcome],
) -> Result<Vec<ControlFamilySummary>, String> {
    for outcome in outcomes {
        if outcome.condition.trim().is_empty() || outcome.detail.trim().is_empty() {
            return Err(String::from(
                "control outcomes need conditions and tolerance details",
            ));
        }
    }
    let mut summary = Vec::with_capacity(4);
    for family in [
        ControlFamily::LevelSweep,
        ControlFamily::Bandwidth,
        ControlFamily::EqualLoudnessTimbre,
        ControlFamily::HeldOutProgramme,
    ] {
        let family_outcomes: Vec<&ControlOutcome> = outcomes
            .iter()
            .filter(|outcome| outcome.family == family)
            .collect();
        if family_outcomes.is_empty() {
            return Err(format!(
                "validation controls miss the {family:?} family: all four families stay recorded"
            ));
        }
        summary.push(ControlFamilySummary {
            family,
            controls: family_outcomes.len(),
            all_passed: family_outcomes.iter().all(|outcome| outcome.passed),
        });
    }
    Ok(summary)
}

/// Enforcement readiness from the evidence side.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EnforcementReadiness {
    /// Independent reference description with its predeclared tolerance.
    pub reference: String,
    /// Playback calibration identity the domain was validated under.
    pub calibration_id: String,
    /// Declared validated domain (`"unvalidated"` blocks enforcement).
    pub validated_domain: String,
    /// Predeclared per-metric tolerances.
    pub tolerances: Vec<String>,
}

impl EnforcementReadiness {
    /// Check enforcement readiness.
    ///
    /// # Errors
    ///
    /// Returns `blocked_external` while the reference, calibration,
    /// domain, or tolerances are missing.
    pub fn check_ready(&self) -> Result<(), String> {
        if self.reference.trim().is_empty() {
            return Err(format!(
                "{BLOCKED_EXTERNAL_PREFIX}: enforcement needs an independent reference"
            ));
        }
        if self.calibration_id.trim().is_empty() {
            return Err(format!(
                "{BLOCKED_EXTERNAL_PREFIX}: enforcement needs a declared calibration identity"
            ));
        }
        if self.validated_domain.trim().is_empty() || self.validated_domain == "unvalidated" {
            return Err(format!(
                "{BLOCKED_EXTERNAL_PREFIX}: enforcement needs a declared validated domain"
            ));
        }
        if self.tolerances.is_empty()
            || self
                .tolerances
                .iter()
                .any(|tolerance| tolerance.trim().is_empty())
        {
            return Err(format!(
                "{BLOCKED_EXTERNAL_PREFIX}: enforcement needs predeclared tolerances"
            ));
        }
        Ok(())
    }
}

/// Reject equivalence or preference claims from equal total loudness.
///
/// Loudness equality is a listening property of the control pair, not
/// evidence of equal timbre or an undetectable difference (F12).
///
/// # Errors
///
/// Returns an error when equal total loudness backs an equivalence or
/// preference intent.
pub fn check_loudness_claim(
    equal_total_loudness: bool,
    intent: ComparisonIntent,
) -> Result<(), String> {
    if !equal_total_loudness {
        return Ok(());
    }
    match intent {
        ComparisonIntent::Detectability => Ok(()),
        ComparisonIntent::Equivalence | ComparisonIntent::Preference => Err(format!(
            "equal total loudness never supports a {intent:?} claim: stage different-timbre controls instead"
        )),
    }
}

#[cfg(test)]
mod promotion_tests {
    use super::*;

    fn programme(id: &str) -> ProgrammeEntry {
        ProgrammeEntry {
            id: String::from(id),
            stimulus_hash: format!("hash-{id}"),
        }
    }

    fn outcomes() -> Vec<ControlOutcome> {
        vec![
            ControlOutcome {
                family: ControlFamily::LevelSweep,
                condition: String::from("sweep-50-80"),
                passed: true,
                detail: String::from("monotone within tolerance"),
            },
            ControlOutcome {
                family: ControlFamily::Bandwidth,
                condition: String::from("full-vs-limited"),
                passed: true,
                detail: String::from("within tolerance"),
            },
            ControlOutcome {
                family: ControlFamily::EqualLoudnessTimbre,
                condition: String::from("flat-vs-steep-tilt"),
                passed: true,
                detail: String::from("different timbre at equal loudness"),
            },
            ControlOutcome {
                family: ControlFamily::HeldOutProgramme,
                condition: String::from("heldout-piano-09"),
                passed: true,
                detail: String::from("within tolerance"),
            },
        ]
    }

    fn readiness() -> EnforcementReadiness {
        EnforcementReadiness {
            reference: String::from("second implementation agrees within 1e-9"),
            calibration_id: String::from("spl-cal-94db"),
            validated_domain: String::from("mono 50-80 dB SPL, 100 Hz-8 kHz"),
            tolerances: vec![String::from("level-sweep within tolerance")],
        }
    }

    #[test]
    fn promotion_controls_require_all_families() {
        let summary = summarize_controls(&outcomes()).unwrap();
        assert_eq!(summary.len(), 4);
        assert!(summary.iter().all(|family| family.all_passed));
        let mut missing = outcomes();
        missing.retain(|outcome| outcome.family != ControlFamily::Bandwidth);
        assert!(summarize_controls(&missing).is_err());
        assert!(summarize_controls(&[]).is_err());
    }

    #[test]
    fn promotion_holdout_disjoint() {
        let set = ProgrammeSet {
            tuning: vec![programme("tuning-speech-01")],
            held_out: vec![programme("heldout-piano-09")],
        };
        assert!(set.verify_holdout_disjoint().is_ok());
        let overlap = ProgrammeSet {
            tuning: vec![programme("shared-01")],
            held_out: vec![programme("shared-01")],
        };
        assert!(overlap.verify_holdout_disjoint().is_err());
        let empty = ProgrammeSet {
            tuning: vec![programme("tuning-speech-01")],
            held_out: Vec::new(),
        };
        assert!(empty.verify_holdout_disjoint().is_err());
    }

    #[test]
    fn promotion_equal_loudness_claim_rejected() {
        assert!(check_loudness_claim(false, ComparisonIntent::Equivalence).is_ok());
        assert!(check_loudness_claim(true, ComparisonIntent::Detectability).is_ok());
        assert!(check_loudness_claim(true, ComparisonIntent::Equivalence).is_err());
        assert!(check_loudness_claim(true, ComparisonIntent::Preference).is_err());
    }

    #[test]
    fn promotion_enforcement_blocked_without_reference() {
        assert!(readiness().check_ready().is_ok());
        let no_reference = EnforcementReadiness {
            reference: String::new(),
            ..readiness()
        };
        let error = no_reference.check_ready().expect_err("reference missing");
        assert!(error.starts_with(BLOCKED_EXTERNAL_PREFIX), "{error}");
        let unvalidated = EnforcementReadiness {
            validated_domain: String::from("unvalidated"),
            ..readiness()
        };
        let error = unvalidated.check_ready().expect_err("domain missing");
        assert!(error.starts_with(BLOCKED_EXTERNAL_PREFIX), "{error}");
        let no_tolerances = EnforcementReadiness {
            tolerances: Vec::new(),
            ..readiness()
        };
        assert!(no_tolerances.check_ready().is_err());
    }
}
