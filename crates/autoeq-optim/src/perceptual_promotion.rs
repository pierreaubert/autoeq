//! Pinned signal-pair fidelity promotion path (Wave 3, step 6).
//!
//! A perceptual model promotes from advisory diagnostics to bounded
//! shortlist reranking to enforcement in that order only. Exactly one
//! signal-pair fidelity family may be pinned (`pemo-q`); enforcement
//! additionally needs an independent reference, a declared
//! calibration/domain, and predeclared tolerances. Uncalibrated input
//! returns unsupported, and missing references or licenses leave the
//! step `blocked_external` instead of a passing gate.
//!
//! Scale discipline is structural: Bark, ERB, and mel stages never mix,
//! ISO 532-1 and ISO 532-2 constants never mix, and programme roughness
//! is never computed from transfer-peak spacing.

// Rust guideline compliant 2026-02-21

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The single pinnable signal-pair fidelity model family.
pub const PINNED_MODEL_FAMILY: &str = "pemo-q";

/// Prefix marking an outcome blocked on external inputs.
pub const BLOCKED_EXTERNAL_PREFIX: &str = "blocked_external";

/// Prefix marking input the path refuses to score.
pub const UNSUPPORTED_PREFIX: &str = "unsupported";

/// Pinned signal-pair fidelity model and edition.
///
/// The G7 lane pins the implementation commit, license, and reference
/// vectors before any adapter is built. An empty field means the lane
/// has not delivered that input yet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PinnedFidelityModel {
    /// Model family; only [`PINNED_MODEL_FAMILY`] is admitted.
    pub model_family: String,
    /// Model edition, e.g. the named PEMO-Q revision.
    pub edition: String,
    /// Implementation commit the pin was validated against.
    pub implementation_commit: String,
    /// License identifier covering the reference implementation.
    pub license_id: String,
    /// Reference-vector set identifier the adapter must reproduce.
    pub reference_vectors_id: String,
}

impl PinnedFidelityModel {
    /// Validate the pin.
    ///
    /// # Errors
    ///
    /// Returns `blocked_external` when any pin field is missing, and a
    /// hard error when the family is not the single pinned family.
    pub fn validate(&self) -> Result<(), String> {
        if self.model_family != PINNED_MODEL_FAMILY {
            return Err(format!(
                "only '{PINNED_MODEL_FAMILY}' may be pinned, not '{}'",
                self.model_family
            ));
        }
        let mut missing = Vec::new();
        for (field, value) in [
            ("edition", &self.edition),
            ("implementation_commit", &self.implementation_commit),
            ("license_id", &self.license_id),
            ("reference_vectors_id", &self.reference_vectors_id),
        ] {
            if value.trim().is_empty() {
                missing.push(field);
            }
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{BLOCKED_EXTERNAL_PREFIX}: fidelity-model pin missing {}",
                missing.join(", ")
            ))
        }
    }
}

/// Auditory frequency scale of one processing stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AuditoryScale {
    /// Critical-band (Zwicker) scale.
    Bark,
    /// Equivalent-rectangular-bandwidth scale.
    Erb,
    /// Mel scale.
    Mel,
}

/// ISO 532 loudness edition of one processing stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IsoLoudnessEdition {
    /// Zwicker method (stationary).
    Iso5321,
    /// Moore-Glasberg method.
    Iso5322,
}

/// Reject mixed auditory scales across stages.
///
/// # Errors
///
/// Returns an error when the two stages do not share one scale.
pub fn check_single_scale(first: AuditoryScale, second: AuditoryScale) -> Result<(), String> {
    if first == second {
        Ok(())
    } else {
        Err(format!(
            "auditory-scale mixing rejected: {first:?} stages never combine with {second:?} stages"
        ))
    }
}

/// Reject mixed ISO 532 editions across stages.
///
/// # Errors
///
/// Returns an error when the two stages use different editions.
pub fn check_iso_edition_purity(
    first: IsoLoudnessEdition,
    second: IsoLoudnessEdition,
) -> Result<(), String> {
    if first == second {
        Ok(())
    } else {
        Err(format!(
            "ISO 532 edition mixing rejected: {first:?} constants never combine with {second:?} constants"
        ))
    }
}

/// Claimed input to a roughness calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RoughnessInput {
    /// Time-domain programme waveform through auditory channels.
    ProgrammeWaveform,
    /// Spacing of transfer-response peaks. Never a roughness input.
    TransferPeakSpacing,
}

/// Reject roughness computed from transfer-peak spacing.
///
/// # Errors
///
/// Returns an error for anything but programme-waveform input.
pub fn check_roughness_input(input: RoughnessInput) -> Result<(), String> {
    match input {
        RoughnessInput::ProgrammeWaveform => Ok(()),
        RoughnessInput::TransferPeakSpacing => Err(String::from(
            "programme roughness from transfer-peak spacing is rejected: roughness needs the programme waveform",
        )),
    }
}

/// Calibration state of a perceptual input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InputCalibration {
    /// Calibrated ear-input levels with stated provenance.
    Calibrated {
        /// Absolute SPL reference identity.
        spl_reference_id: String,
        /// Playback sample rate in Hz.
        sample_rate_hz: f64,
        /// Free-/diffuse-field or headphone transfer identity.
        field_transfer: String,
    },
    /// No calibration: the path must refuse to score.
    Uncalibrated,
}

impl InputCalibration {
    /// Require calibration before any sone or fidelity estimate.
    ///
    /// # Errors
    ///
    /// Returns `unsupported` for uncalibrated input or blank provenance.
    pub fn require_calibrated(&self) -> Result<(), String> {
        match self {
            InputCalibration::Calibrated {
                spl_reference_id,
                sample_rate_hz,
                field_transfer,
            } => {
                if spl_reference_id.trim().is_empty()
                    || field_transfer.trim().is_empty()
                    || !sample_rate_hz.is_finite()
                    || *sample_rate_hz <= 0.0
                {
                    return Err(format!(
                        "{UNSUPPORTED_PREFIX}: calibrated input needs an SPL reference, a field transfer, and a positive sample rate"
                    ));
                }
                Ok(())
            }
            InputCalibration::Uncalibrated => Err(format!(
                "{UNSUPPORTED_PREFIX}: uncalibrated input returns unsupported, not a sone estimate"
            )),
        }
    }
}

/// One equal-total-loudness, different-spectrum control pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EqualLoudnessPair {
    /// First stimulus identity.
    pub first_id: String,
    /// Second stimulus identity with a different spectral envelope.
    pub second_id: String,
}

/// Validation controls the promotion path must run before reranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ValidationControls {
    /// Level-sweep points in dB SPL.
    pub level_sweep_db: Vec<f64>,
    /// Bandwidth-change condition identities.
    pub bandwidth_conditions: Vec<String>,
    /// Equal-loudness, different-timbre control pairs.
    pub equal_loudness_pairs: Vec<EqualLoudnessPair>,
    /// Programmes held out before any tuning.
    pub held_out_programmes: Vec<String>,
    /// Programmes used during tuning (must stay disjoint from holdouts).
    pub tuning_programmes: Vec<String>,
}

impl ValidationControls {
    /// Validate control coverage and holdout discipline.
    ///
    /// # Errors
    ///
    /// Returns an error when any control family is empty, levels are
    /// non-finite, identities are blank, or a holdout ran in tuning.
    pub fn validate(&self) -> Result<(), String> {
        if self.level_sweep_db.is_empty() {
            return Err(String::from(
                "validation controls need at least one level-sweep point",
            ));
        }
        if self.level_sweep_db.iter().any(|level| !level.is_finite()) {
            return Err(String::from("level-sweep points must be finite"));
        }
        if self.bandwidth_conditions.is_empty()
            || self
                .bandwidth_conditions
                .iter()
                .any(|condition| condition.trim().is_empty())
        {
            return Err(String::from(
                "validation controls need non-blank bandwidth conditions",
            ));
        }
        if self.equal_loudness_pairs.is_empty() {
            return Err(String::from(
                "validation controls need at least one equal-loudness, different-timbre pair",
            ));
        }
        for pair in &self.equal_loudness_pairs {
            if pair.first_id.trim().is_empty() || pair.second_id.trim().is_empty() {
                return Err(String::from(
                    "equal-loudness pairs need both stimulus identities",
                ));
            }
        }
        if self.held_out_programmes.is_empty()
            || self
                .held_out_programmes
                .iter()
                .any(|programme| programme.trim().is_empty())
        {
            return Err(String::from(
                "validation controls need non-blank held-out programmes",
            ));
        }
        for programme in &self.held_out_programmes {
            if self
                .tuning_programmes
                .iter()
                .any(|tuned| tuned == programme)
            {
                return Err(format!(
                    "held-out programme {programme} also ran in tuning: holdouts stay disjoint"
                ));
            }
        }
        Ok(())
    }
}

/// Independent reference backing an enforcement claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct IndependentReference {
    /// What was reimplemented and how agreement is measured.
    pub description: String,
    /// Predeclared agreement tolerance, e.g. `"within 1e-9 of the protocol p value"`.
    pub tolerance: String,
    /// Observed agreement against the reference.
    pub agreement: String,
}

impl IndependentReference {
    /// Validate the reference is stated before results exist.
    ///
    /// # Errors
    ///
    /// Returns `blocked_external` when description or tolerance is blank.
    pub fn validate(&self) -> Result<(), String> {
        if self.description.trim().is_empty() || self.tolerance.trim().is_empty() {
            Err(format!(
                "{BLOCKED_EXTERNAL_PREFIX}: enforcement needs an independent reference with a predeclared tolerance"
            ))
        } else {
            Ok(())
        }
    }
}

/// Declared calibration and domain of an enforcement claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DeclaredDomain {
    /// Playback calibration identity the domain was validated under.
    pub calibration_id: String,
    /// Validated domain, e.g. `"mono 50-80 dB SPL, 100 Hz-8 kHz"`.
    pub domain: String,
    /// Predeclared per-metric tolerances.
    pub tolerances: Vec<String>,
}

impl DeclaredDomain {
    /// Validate the declaration.
    ///
    /// # Errors
    ///
    /// Returns `blocked_external` while the domain stays `"unvalidated"`.
    pub fn validate(&self) -> Result<(), String> {
        if self.calibration_id.trim().is_empty() {
            return Err(format!(
                "{BLOCKED_EXTERNAL_PREFIX}: enforcement needs a declared calibration identity"
            ));
        }
        if self.domain.trim().is_empty() || self.domain == "unvalidated" {
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

/// Promotion stage of a perceptual model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PromotionStage {
    /// Advisory diagnostics only: no reranking, no enforcement.
    Advisory,
    /// Bounded shortlist reranker against the desired reference.
    Reranker,
    /// Enforcement with independent references and declared domain.
    Enforcement,
}

/// Admit a pinned model to bounded shortlist reranking.
///
/// # Errors
///
/// Returns an error (or `blocked_external`) when the pin, the
/// calibration, or any validation-control family is missing.
pub fn admit_to_reranker(
    model: &PinnedFidelityModel,
    calibration: &InputCalibration,
    controls: &ValidationControls,
) -> Result<PromotionStage, String> {
    model.validate()?;
    calibration.require_calibrated()?;
    controls.validate()?;
    Ok(PromotionStage::Reranker)
}

/// Admit a reranked model to enforcement.
///
/// Promotion order is advisory, then reranker, then enforcement: this
/// gate re-checks every reranker input plus the independent reference
/// and the declared domain. Skipping straight from advisory fails here.
///
/// # Errors
///
/// Returns `blocked_external` while references, domain, or tolerances
/// are missing; enforcement is never granted on advisory evidence alone.
pub fn admit_to_enforcement(
    model: &PinnedFidelityModel,
    calibration: &InputCalibration,
    controls: &ValidationControls,
    reference: &IndependentReference,
    domain: &DeclaredDomain,
) -> Result<PromotionStage, String> {
    admit_to_reranker(model, calibration, controls)?;
    reference.validate()?;
    domain.validate()?;
    Ok(PromotionStage::Enforcement)
}

/// Reject equivalence or preference claims from equal total loudness.
///
/// Equal total loudness never implies equal timbre or an undetectable
/// difference (F12).
///
/// # Errors
///
/// Returns an error when `equal_total_loudness` backs an equivalence,
/// preference, or inaudibility claim.
pub fn check_equal_loudness_claim(equal_total_loudness: bool, claimed: &str) -> Result<(), String> {
    if !equal_total_loudness {
        return Ok(());
    }
    let lowered = claimed.to_lowercase();
    for word in ["equivalen", "prefer", "inaudib", "undetect", "transparent"] {
        if lowered.contains(word) {
            return Err(format!(
                "equal total loudness never yields a '{claimed}' claim: loudness equality is not timbre equality"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod perceptual_promotion_tests {
    use super::*;

    fn pinned() -> PinnedFidelityModel {
        PinnedFidelityModel {
            model_family: String::from("pemo-q"),
            edition: String::from("pemo-q-2024-ed1"),
            implementation_commit: String::from("coordinator-pinned-commit"),
            license_id: String::from("g7-license-id"),
            reference_vectors_id: String::from("g7-reference-vectors-v1"),
        }
    }

    fn calibrated() -> InputCalibration {
        InputCalibration::Calibrated {
            spl_reference_id: String::from("spl-cal-94db"),
            sample_rate_hz: 48_000.0,
            field_transfer: String::from("diffuse-field-iso-262"),
        }
    }

    fn controls() -> ValidationControls {
        ValidationControls {
            level_sweep_db: vec![50.0, 65.0, 80.0],
            bandwidth_conditions: vec![String::from("full-band"), String::from("band-limited")],
            equal_loudness_pairs: vec![EqualLoudnessPair {
                first_id: String::from("complex-flat-tilt"),
                second_id: String::from("complex-steep-tilt"),
            }],
            held_out_programmes: vec![String::from("heldout-piano-09")],
            tuning_programmes: vec![String::from("tuning-speech-01")],
        }
    }

    fn reference() -> IndependentReference {
        IndependentReference {
            description: String::from("second implementation agrees on reference vectors"),
            tolerance: String::from("within pinned tolerance"),
            agreement: String::from("agrees"),
        }
    }

    fn domain() -> DeclaredDomain {
        DeclaredDomain {
            calibration_id: String::from("spl-cal-94db"),
            domain: String::from("mono 50-80 dB SPL, 100 Hz-8 kHz"),
            tolerances: vec![String::from("level-sweep within tolerance")],
        }
    }

    #[test]
    fn promotion_uncalibrated_input_returns_unsupported() {
        assert!(calibrated().require_calibrated().is_ok());
        let error = InputCalibration::Uncalibrated
            .require_calibrated()
            .expect_err("uncalibrated input must be refused");
        assert!(error.starts_with(UNSUPPORTED_PREFIX), "{error}");
        let blank = InputCalibration::Calibrated {
            spl_reference_id: String::new(),
            sample_rate_hz: 48_000.0,
            field_transfer: String::from("diffuse-field"),
        };
        assert!(blank.require_calibrated().is_err());
    }

    #[test]
    fn promotion_equal_loudness_never_equivalence() {
        assert!(check_equal_loudness_claim(false, "equivalent").is_ok());
        assert!(check_equal_loudness_claim(true, "level audit only").is_ok());
        for claimed in ["equivalent", "preferred", "inaudible", "transparent"] {
            let error = check_equal_loudness_claim(true, claimed).expect_err("F12");
            assert!(error.contains("never yields"), "{error}");
        }
    }

    #[test]
    fn promotion_missing_reference_license_blocked_external() {
        let mut unpinned = pinned();
        unpinned.license_id.clear();
        let error = unpinned.validate().expect_err("license missing");
        assert!(error.starts_with(BLOCKED_EXTERNAL_PREFIX), "{error}");
        let mut no_vectors = pinned();
        no_vectors.reference_vectors_id.clear();
        assert!(no_vectors.validate().is_err());
        let wrong_family = PinnedFidelityModel {
            model_family: String::from("peaq"),
            ..pinned()
        };
        let error = wrong_family.validate().expect_err("one family only");
        assert!(!error.starts_with(BLOCKED_EXTERNAL_PREFIX), "{error}");
    }

    #[test]
    fn promotion_scale_mixing_rejected() {
        assert!(check_single_scale(AuditoryScale::Bark, AuditoryScale::Bark).is_ok());
        assert!(check_single_scale(AuditoryScale::Bark, AuditoryScale::Erb).is_err());
        assert!(check_single_scale(AuditoryScale::Bark, AuditoryScale::Mel).is_err());
        assert!(
            check_iso_edition_purity(IsoLoudnessEdition::Iso5321, IsoLoudnessEdition::Iso5321)
                .is_ok()
        );
        assert!(
            check_iso_edition_purity(IsoLoudnessEdition::Iso5321, IsoLoudnessEdition::Iso5322)
                .is_err()
        );
    }

    #[test]
    fn promotion_transfer_peak_roughness_rejected() {
        assert!(check_roughness_input(RoughnessInput::ProgrammeWaveform).is_ok());
        assert!(check_roughness_input(RoughnessInput::TransferPeakSpacing).is_err());
    }

    #[test]
    fn promotion_enforcement_needs_independent_reference_and_tolerances() {
        assert_eq!(
            admit_to_reranker(&pinned(), &calibrated(), &controls()).unwrap(),
            PromotionStage::Reranker
        );
        assert_eq!(
            admit_to_enforcement(
                &pinned(),
                &calibrated(),
                &controls(),
                &reference(),
                &domain()
            )
            .unwrap(),
            PromotionStage::Enforcement
        );
        let blank_reference = IndependentReference {
            description: String::new(),
            tolerance: String::new(),
            agreement: String::new(),
        };
        let error = admit_to_enforcement(
            &pinned(),
            &calibrated(),
            &controls(),
            &blank_reference,
            &domain(),
        )
        .expect_err("reference missing");
        assert!(error.starts_with(BLOCKED_EXTERNAL_PREFIX), "{error}");
        let unvalidated = DeclaredDomain {
            domain: String::from("unvalidated"),
            ..domain()
        };
        let error = admit_to_enforcement(
            &pinned(),
            &calibrated(),
            &controls(),
            &reference(),
            &unvalidated,
        )
        .expect_err("domain unvalidated");
        assert!(error.starts_with(BLOCKED_EXTERNAL_PREFIX), "{error}");
    }

    #[test]
    fn promotion_advisory_to_reranker_order() {
        let mut no_controls = controls();
        no_controls.level_sweep_db.clear();
        assert!(admit_to_reranker(&pinned(), &calibrated(), &no_controls).is_err());
        assert!(
            admit_to_reranker(&pinned(), &InputCalibration::Uncalibrated, &controls()).is_err()
        );
        let mut overlap = controls();
        overlap.tuning_programmes = overlap.held_out_programmes.clone();
        assert!(admit_to_reranker(&pinned(), &calibrated(), &overlap).is_err());
    }
}
