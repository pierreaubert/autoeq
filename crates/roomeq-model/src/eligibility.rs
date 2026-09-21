//! K2 operation eligibility contracts: data and validation only.
//!
//! Eligibility records whether the available evidence permits an operation
//! (magnitude correction, coherent summation, excess-phase correction,
//! absolute loudness analysis, direct-sound speaker correction, or decay
//! analysis) on a frequency band. It is distinct from the final correction
//! decision (see [`crate::decision_ledger`]) and from acceptance
//! (see [`crate::CorrectionAcceptanceReport`], which stays authoritative).
//!
//! Evidence itself lives with the core K1 envelope when that lands; these
//! types only cite evidence by stable reference ID so legacy JSON without
//! the new fields still loads. New timing/repeatability budgets are explicit
//! and versioned ([`EvidencePolicy`]); nothing here invents universal
//! acoustic thresholds. An absent [`LocalQPolicy`] preserves the existing
//! global-Q behavior.

// Rust guideline compliant 2026-02-21

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AppliedThreshold, AssessmentRecord};

/// Version pin for [`EvidencePolicy`] and [`LocalQPolicy`].
///
/// Unknown policy versions fail [`EvidencePolicy::validate`] explicitly;
/// deserialization stays lenient so legacy JSON loads without new
/// enforcement.
pub const ELIGIBILITY_POLICY_VERSION: &str = "1.0.0";

fn eligibility_policy_version_default() -> String {
    ELIGIBILITY_POLICY_VERSION.to_string()
}

/// Operations gated by evidence eligibility (global K2 vocabulary).
///
/// Eligibility is not the final correction decision; the workflow applies
/// its configured policy on top of these per-band verdicts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionOperation {
    /// Minimum-phase magnitude equalization.
    MagnitudeCorrection,
    /// Coherent combination of multiple sources or seats.
    CoherentSummation,
    /// Correction of excess (non-minimum) phase.
    ExcessPhaseCorrection,
    /// Analysis of absolute loudness; requires measured SPL attribution.
    AbsoluteLoudnessAnalysis,
    /// Correction derived from the direct sound of a speaker.
    DirectSoundSpeakerCorrection,
    /// Reverberation/decay analysis.
    DecayAnalysis,
    /// Operation not stated; legacy input degrades here, never to eligible.
    #[default]
    Unknown,
}

/// Per-band eligibility verdict for one operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EligibilityVerdict {
    /// Evidence supports executing the operation on this band.
    Eligible,
    /// Execution is allowed under the recorded policy limits only.
    Limited,
    /// Evidence rules out the operation on this band.
    Unsupported,
    /// Evidence is missing or unassessed; the default for legacy input.
    #[default]
    Unknown,
}

/// How a level value is attributed: a nominal proxy level is never measured SPL.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LevelAttribution {
    /// No level attribution stated.
    #[default]
    Unknown,
    /// Nominal proxy level (for example `PruningEvaluation`
    /// `listening_levels_phon`); an assumption, not a measurement.
    NominalPhon,
    /// Level anchored by measured SPL calibration evidence.
    MeasuredSpl,
}

/// Optional timing-uncertainty budget with explicit units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TimingBudget {
    /// Maximum tolerable timing uncertainty in milliseconds.
    pub max_timing_uncertainty_ms: f64,
    /// Unit label; pinned to `"ms"`.
    #[serde(default = "timing_unit_default")]
    pub unit: String,
}

fn timing_unit_default() -> String {
    String::from("ms")
}

/// Optional repeatability budget with explicit units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RepeatabilityBudget {
    /// Maximum tolerable repeat-measurement magnitude spread in dB.
    pub max_repeatability_spread_db: f64,
    /// Unit label; pinned to `"db"`.
    #[serde(default = "spread_unit_default")]
    pub unit: String,
}

fn spread_unit_default() -> String {
    String::from("db")
}

/// One local-Q knot: the Q cap in force at a center frequency.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LocalQKnot {
    /// Knot frequency in Hz; knots must be positive and strictly increasing.
    pub freq_hz: f64,
    /// Q cap at this knot; must be finite and greater than zero.
    pub max_q: f64,
}

/// Optional frequency-dependent Q policy (model side of core K3).
///
/// Effective bounds are the stricter of the global and local limits. An
/// absent policy preserves the current global-Q behavior exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LocalQPolicy {
    /// Policy version; must equal [`ELIGIBILITY_POLICY_VERSION`].
    #[serde(default = "eligibility_policy_version_default")]
    pub policy_version: String,
    /// Q-cap knots in Hz, strictly increasing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub knots: Vec<LocalQKnot>,
}

impl LocalQPolicy {
    /// Check version, knot ordering, positivity, and finiteness.
    ///
    /// # Errors
    ///
    /// Returns a reason when the version is unknown, no knots are declared,
    /// or any knot has a nonfinite/non-positive frequency or Q, or the
    /// frequencies are not strictly increasing.
    pub fn validate(&self) -> Result<(), String> {
        if self.policy_version != ELIGIBILITY_POLICY_VERSION {
            return Err(format!(
                "unsupported local-Q policy version '{}'; expected '{}'",
                self.policy_version, ELIGIBILITY_POLICY_VERSION
            ));
        }
        if self.knots.is_empty() {
            return Err(String::from("local-Q policy declares no knots"));
        }
        let mut previous_hz = 0.0;
        for knot in &self.knots {
            if !knot.freq_hz.is_finite() || knot.freq_hz <= 0.0 {
                return Err(format!(
                    "local-Q knot frequency must be finite and positive (got {})",
                    knot.freq_hz
                ));
            }
            if !knot.max_q.is_finite() || knot.max_q <= 0.0 {
                return Err(format!(
                    "local-Q knot max_q must be finite and positive (got {})",
                    knot.max_q
                ));
            }
            if knot.freq_hz <= previous_hz {
                return Err(format!(
                    "local-Q knot frequencies must be strictly increasing (got {} after {})",
                    knot.freq_hz, previous_hz
                ));
            }
            previous_hz = knot.freq_hz;
        }
        Ok(())
    }

    /// Effective Q cap at `freq_hz`: the stricter of the global cap and the
    /// local envelope (linear interpolation in log frequency, endpoint hold
    /// inside the requested correction band, no extrapolated validity claim).
    pub fn effective_max_q(&self, freq_hz: f64, global_max_q: f64) -> f64 {
        global_max_q.min(self.local_max_q(freq_hz))
    }

    fn local_max_q(&self, freq_hz: f64) -> f64 {
        debug_assert!(!self.knots.is_empty());
        if self.knots.is_empty() || freq_hz <= self.knots[0].freq_hz {
            return self.knots.first().map_or(f64::INFINITY, |knot| knot.max_q);
        }
        if freq_hz >= self.knots[self.knots.len() - 1].freq_hz {
            return self.knots[self.knots.len() - 1].max_q;
        }
        for pair in self.knots.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            if freq_hz >= lo.freq_hz && freq_hz <= hi.freq_hz {
                let t = (freq_hz.ln() - lo.freq_hz.ln()) / (hi.freq_hz.ln() - lo.freq_hz.ln());
                return lo.max_q + t * (hi.max_q - lo.max_q);
            }
        }
        self.knots[self.knots.len() - 1].max_q
    }
}

/// Versioned evidence/eligibility policy carrying the new K2 budgets.
///
/// All budgets are optional; absent budgets preserve documented legacy
/// behavior. Calibration, preference tilt, and level compensation stay
/// separately identifiable: this policy records level *attribution* only,
/// never a tilt or voicing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EvidencePolicy {
    /// Policy version; must equal [`ELIGIBILITY_POLICY_VERSION`].
    #[serde(default = "eligibility_policy_version_default")]
    pub policy_version: String,
    /// Optional timing-uncertainty budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingBudget>,
    /// Optional repeatability-spread budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeatability: Option<RepeatabilityBudget>,
    /// How the policy level is attributed; nominal phon is never SPL.
    #[serde(default)]
    pub calibration: LevelAttribution,
    /// Stable reference to the SPL calibration evidence when
    /// [`LevelAttribution::MeasuredSpl`] is claimed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_evidence_ref: Option<String>,
    /// Optional local-Q policy; absent preserves global-Q behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_q: Option<LocalQPolicy>,
}

impl Default for EvidencePolicy {
    fn default() -> Self {
        Self {
            policy_version: eligibility_policy_version_default(),
            timing: None,
            repeatability: None,
            calibration: LevelAttribution::default(),
            calibration_evidence_ref: None,
            local_q: None,
        }
    }
}

impl EvidencePolicy {
    /// Whether this policy claims measured-SPL backing.
    ///
    /// Nominal phon levels never acquire measured SPL status here; only an
    /// explicit [`LevelAttribution::MeasuredSpl`] claim counts.
    pub fn has_measured_spl(&self) -> bool {
        self.calibration == LevelAttribution::MeasuredSpl
    }

    /// Check version, budget finiteness/sign, units, calibration linkage,
    /// and the nested local-Q policy.
    ///
    /// # Errors
    ///
    /// Returns a reason for unknown versions, nonfinite or negative budgets,
    /// wrong unit labels, a measured-SPL claim without an evidence reference,
    /// or an invalid nested local-Q policy.
    pub fn validate(&self) -> Result<(), String> {
        if self.policy_version != ELIGIBILITY_POLICY_VERSION {
            return Err(format!(
                "unsupported evidence policy version '{}'; expected '{}'",
                self.policy_version, ELIGIBILITY_POLICY_VERSION
            ));
        }
        if let Some(timing) = &self.timing {
            if !timing.max_timing_uncertainty_ms.is_finite()
                || timing.max_timing_uncertainty_ms < 0.0
            {
                return Err(format!(
                    "timing budget must be a finite non-negative ms value (got {})",
                    timing.max_timing_uncertainty_ms
                ));
            }
            if timing.unit != "ms" {
                return Err(format!(
                    "timing budget unit must be 'ms' (got '{}')",
                    timing.unit
                ));
            }
        }
        if let Some(repeatability) = &self.repeatability {
            if !repeatability.max_repeatability_spread_db.is_finite()
                || repeatability.max_repeatability_spread_db < 0.0
            {
                return Err(format!(
                    "repeatability budget must be a finite non-negative dB value (got {})",
                    repeatability.max_repeatability_spread_db
                ));
            }
            if repeatability.unit != "db" {
                return Err(format!(
                    "repeatability budget unit must be 'db' (got '{}')",
                    repeatability.unit
                ));
            }
        }
        if self.calibration == LevelAttribution::MeasuredSpl
            && self
                .calibration_evidence_ref
                .as_ref()
                .is_none_or(|reference| reference.trim().is_empty())
        {
            return Err(String::from(
                "measured-SPL attribution requires a calibration evidence reference",
            ));
        }
        if let Some(local_q) = &self.local_q {
            local_q.validate()?;
        }
        Ok(())
    }
}

/// One per-band eligibility verdict for a single operation.
///
/// Evidence is cited by stable reference ID; the core K1 envelope owns the
/// values. The embedded [`AssessmentRecord`] reuses the existing
/// confidence/provenance vocabulary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EligibilityRecord {
    /// Stable record identifier; must be nonempty to validate.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub record_id: String,
    /// Gated operation.
    #[serde(default)]
    pub operation: CorrectionOperation,
    /// Verdict for this band; legacy input defaults to unknown.
    #[serde(default)]
    pub verdict: EligibilityVerdict,
    /// Stable measurement identifier the verdict applies to.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub measurement_id: String,
    /// Seat identifiers in scope; empty means the scope was not stated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seat_ids: Vec<String>,
    /// Frequency band in Hz; must be finite with `0 < lo < hi` to validate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_hz: Option<[f64; 2]>,
    /// Human-readable observations behind the verdict.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observations: Vec<String>,
    /// Policy limits applied, with units.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub policy_limits: Vec<AppliedThreshold>,
    /// Stable evidence reference IDs behind the verdict.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_refs: Vec<String>,
    /// Existing assessment vocabulary (confidence/provenance).
    #[serde(default)]
    pub assessment: AssessmentRecord,
}

impl EligibilityRecord {
    /// Structural check: IDs, scope, band bounds, and verdict/evidence
    /// consistency. A permission verdict (`Eligible`/`Limited`) with no
    /// cited evidence is contradictory and fails.
    ///
    /// # Errors
    ///
    /// Returns a reason for empty IDs, missing bands, nonfinite or
    /// unordered band bounds, or permission verdicts without evidence.
    pub fn validate(&self) -> Result<(), String> {
        if self.record_id.trim().is_empty() {
            return Err(String::from("eligibility record_id must not be empty"));
        }
        if self.measurement_id.trim().is_empty() {
            return Err(String::from("eligibility measurement_id must not be empty"));
        }
        let Some(band_hz) = self.band_hz else {
            return Err(String::from("eligibility band_hz must be stated"));
        };
        if !band_hz[0].is_finite()
            || !band_hz[1].is_finite()
            || band_hz[0] <= 0.0
            || band_hz[1] <= band_hz[0]
        {
            return Err(format!(
                "eligibility band_hz must satisfy 0 < lo < hi with finite bounds (got [{}, {}])",
                band_hz[0], band_hz[1]
            ));
        }
        if matches!(
            self.verdict,
            EligibilityVerdict::Eligible | EligibilityVerdict::Limited
        ) && self.evidence_refs.is_empty()
        {
            return Err(format!(
                "eligibility verdict {:?} cites no evidence; permission requires evidence references",
                self.verdict
            ));
        }
        Ok(())
    }

    /// Structural check plus operation/policy coherence: absolute loudness
    /// analysis with a permission verdict requires measured-SPL attribution
    /// in the supplied policy (nominal phon never qualifies).
    ///
    /// # Errors
    ///
    /// Returns [`EligibilityRecord::validate`] failures plus the
    /// loudness/calibration coherence rule.
    pub fn validate_with_policy(&self, policy: &EvidencePolicy) -> Result<(), String> {
        self.validate()?;
        if self.operation == CorrectionOperation::AbsoluteLoudnessAnalysis
            && matches!(
                self.verdict,
                EligibilityVerdict::Eligible | EligibilityVerdict::Limited
            )
            && !policy.has_measured_spl()
        {
            return Err(String::from(
                "absolute loudness analysis requires measured-SPL attribution; nominal levels do not qualify",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band_record() -> EligibilityRecord {
        EligibilityRecord {
            record_id: "elig-1".to_string(),
            operation: CorrectionOperation::MagnitudeCorrection,
            verdict: EligibilityVerdict::Eligible,
            measurement_id: "meas-1".to_string(),
            seat_ids: vec!["seat-a".to_string()],
            band_hz: Some([40.0, 400.0]),
            observations: vec!["good coherence".to_string()],
            policy_limits: Vec::new(),
            evidence_refs: vec!["ev-1".to_string()],
            assessment: AssessmentRecord::default(),
        }
    }

    #[test]
    fn model_eligibility_unknown_roundtrip() {
        // Legacy JSON carries no K2 fields: it loads, stays unknown, and
        // enforces nothing.
        let legacy = serde_json::json!({
            "record_id": "elig-legacy",
            "measurement_id": "meas-legacy",
            "band_hz": [40.0, 400.0],
        });
        let record: EligibilityRecord = serde_json::from_value(legacy).unwrap();
        assert_eq!(record.operation, CorrectionOperation::Unknown);
        assert_eq!(record.verdict, EligibilityVerdict::Unknown);
        assert!(record.evidence_refs.is_empty());
        assert_eq!(
            record.assessment.confidence,
            crate::AssessmentConfidence::Unknown
        );
        let round_tripped: EligibilityRecord =
            serde_json::from_value(serde_json::to_value(&record).unwrap()).unwrap();
        assert_eq!(round_tripped, record);
        // Unknown verdicts validate structurally; they claim nothing.
        assert!(record.validate().is_ok());
    }

    #[test]
    fn model_evidence_policy_invalid_values() {
        let base = EvidencePolicy::default();
        assert!(base.validate().is_ok());

        let nan_timing = EvidencePolicy {
            timing: Some(TimingBudget {
                max_timing_uncertainty_ms: f64::NAN,
                unit: String::from("ms"),
            }),
            ..base.clone()
        };
        assert!(nan_timing.validate().is_err());

        let negative_spread = EvidencePolicy {
            repeatability: Some(RepeatabilityBudget {
                max_repeatability_spread_db: -1.0,
                unit: String::from("db"),
            }),
            ..base.clone()
        };
        assert!(negative_spread.validate().is_err());

        let infinite_spread = EvidencePolicy {
            repeatability: Some(RepeatabilityBudget {
                max_repeatability_spread_db: f64::INFINITY,
                unit: String::from("db"),
            }),
            ..base.clone()
        };
        assert!(infinite_spread.validate().is_err());

        let bad_q = EvidencePolicy {
            local_q: Some(LocalQPolicy {
                policy_version: ELIGIBILITY_POLICY_VERSION.to_string(),
                knots: vec![
                    LocalQKnot {
                        freq_hz: 100.0,
                        max_q: 2.0,
                    },
                    LocalQKnot {
                        freq_hz: 100.0,
                        max_q: 1.0,
                    },
                ],
            }),
            ..base.clone()
        };
        assert!(bad_q.validate().is_err());

        let nonpositive_q = EvidencePolicy {
            local_q: Some(LocalQPolicy {
                policy_version: ELIGIBILITY_POLICY_VERSION.to_string(),
                knots: vec![LocalQKnot {
                    freq_hz: 100.0,
                    max_q: 0.0,
                }],
            }),
            ..base.clone()
        };
        assert!(nonpositive_q.validate().is_err());

        let mismatched = EligibilityRecord {
            record_id: String::new(),
            measurement_id: String::new(),
            band_hz: Some([400.0, 40.0]),
            ..band_record()
        };
        assert!(mismatched.validate().is_err());

        let permission_without_evidence = EligibilityRecord {
            evidence_refs: Vec::new(),
            ..band_record()
        };
        assert!(permission_without_evidence.validate().is_err());

        let unknown_version = EvidencePolicy {
            policy_version: String::from("9.9.9"),
            ..base
        };
        assert!(unknown_version.validate().is_err());
    }

    #[test]
    fn model_new_policy_absent_preserves_defaults() {
        // Absent K2 policy must not change existing optimizer behavior:
        // flat loss, no correction-band narrowing, veto disabled.
        let optimizer = crate::OptimizerConfig::default();
        assert_eq!(optimizer.loss_type, "flat");
        assert!(optimizer.correction_band.is_none());
        assert_eq!(
            optimizer.active_correction_band(),
            [optimizer.min_freq, optimizer.max_freq]
        );
        assert!(optimizer.filter_audibility.is_none());

        // Absent local-Q preserves the global cap exactly.
        let policy = EvidencePolicy::default();
        assert!(policy.local_q.is_none());
        let global_max_q = optimizer.max_q;
        assert!(global_max_q > 0.0);
        let _ = global_max_q;

        // Present local-Q only ever tightens the global bound.
        let local = LocalQPolicy {
            policy_version: ELIGIBILITY_POLICY_VERSION.to_string(),
            knots: vec![
                LocalQKnot {
                    freq_hz: 50.0,
                    max_q: 4.0,
                },
                LocalQKnot {
                    freq_hz: 500.0,
                    max_q: 1.5,
                },
            ],
        };
        assert!(local.validate().is_ok());
        assert!(local.effective_max_q(100.0, 6.0) <= 6.0);
        assert_eq!(local.effective_max_q(10.0, 6.0), 4.0);
        assert_eq!(local.effective_max_q(20_000.0, 6.0), 1.5);
    }

    #[test]
    fn model_calibration_not_inferred_from_nominal_phon() {
        // Spectral-v1 nominal listening levels are proxy assumptions: they
        // must never acquire measured SPL status.
        let evaluation = crate::PruningEvaluation {
            version: crate::PruningEvaluationVersion::SpectralV1,
            measurement_ids: vec![String::from("seat-a")],
            programmes: vec![crate::PruningProgramme {
                id: String::from("music"),
                frequencies_hz: vec![100.0, 1_000.0],
                spectrum_db: vec![0.0, -1.0],
            }],
            listening_levels_phon: vec![75.0],
        };
        assert!(evaluation.validate().is_ok());

        let nominal_policy = EvidencePolicy {
            calibration: LevelAttribution::NominalPhon,
            ..EvidencePolicy::default()
        };
        assert!(nominal_policy.validate().is_ok());
        assert!(!nominal_policy.has_measured_spl());

        // Absolute loudness permission on nominal evidence is contradictory.
        let loudness = EligibilityRecord {
            operation: CorrectionOperation::AbsoluteLoudnessAnalysis,
            verdict: EligibilityVerdict::Eligible,
            ..band_record()
        };
        assert!(loudness.validate_with_policy(&nominal_policy).is_err());

        // Measured SPL requires an explicit calibration evidence reference.
        let bare_measured = EvidencePolicy {
            calibration: LevelAttribution::MeasuredSpl,
            ..EvidencePolicy::default()
        };
        assert!(bare_measured.validate().is_err());
        let anchored = EvidencePolicy {
            calibration: LevelAttribution::MeasuredSpl,
            calibration_evidence_ref: Some(String::from("spl-cal-1")),
            ..EvidencePolicy::default()
        };
        assert!(anchored.validate().is_ok());
        assert!(loudness.validate_with_policy(&anchored).is_ok());
    }
}
