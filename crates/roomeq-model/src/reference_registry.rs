//! Approved independent-reference registry for promotion claims.
//!
//! A perceptual enforcement claim needs verified passing evidence, not a
//! descriptive string. This module is the single canonical representation of
//! that evidence: [`VerifiedReferenceEvidence`] carries the observed numeric
//! agreement, and [`ApprovedReferenceRegistry`] lists the implementations the
//! programme accepts. With no licensed independent fixture in-tree, every
//! registry starts empty and verification fails closed with
//! `blocked_external`.
//!
//! Owners: the G7 lane curates entries; promotion gates only verify.

// Rust guideline compliant 2026-02-21

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Prefix marking an outcome blocked on external inputs.
pub const BLOCKED_EXTERNAL_PREFIX: &str = "blocked_external";

/// Numeric agreement observed against an independent reference.
///
/// Descriptive text lives with the claimant for explanation only; only these
/// fields decide whether agreement exists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VerifiedReferenceEvidence {
    /// Approved implementation identity that produced the observation.
    pub implementation_id: String,
    /// Implementation hash the observation was recorded against.
    pub implementation_hash: String,
    /// Independent vector set the observation ran over (nonempty).
    pub vectors_id: String,
    /// Model family the observation validates.
    pub model_family: String,
    /// Model edition the observation validates.
    pub edition: String,
    /// Calibration identity the observation ran under.
    pub calibration_id: String,
    /// Domain the observation covers.
    pub domain: String,
    /// Metric the observed error is expressed in (non-blank).
    pub error_metric: String,
    /// Observed error in `error_metric` (must be finite).
    pub observed_error: f64,
    /// Predeclared tolerance in `error_metric` (finite, non-negative).
    pub error_tolerance: f64,
    /// Applicable coverage the observation spans (nonempty).
    pub coverage: Vec<String>,
}

/// One implementation the programme accepts reference results from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ApprovedReferenceEntry {
    /// Approved implementation identity.
    pub implementation_id: String,
    /// Pinned implementation hash; observations on other hashes are stale.
    pub implementation_hash: String,
    /// Independent vector set the approval covers.
    pub vectors_id: String,
    /// Model family the approval covers.
    pub model_family: String,
    /// Model edition the approval covers.
    pub edition: String,
    /// Calibration identity the approval covers.
    pub calibration_id: String,
    /// Domain the approval covers.
    pub domain: String,
    /// Predeclared error metric; missing legacy criteria cannot authorize promotion.
    #[serde(default)]
    pub error_metric: String,
    /// Approved numerical tolerance; absence leaves the approval incomplete.
    #[serde(default)]
    pub error_tolerance: Option<f64>,
    /// Required reference cases, all of which the observation must cover.
    #[serde(default)]
    pub required_coverage: Vec<String>,
}

/// Registry of approved independent-reference implementations.
///
/// Empty until the G7 lane delivers a licensed implementation, reference
/// vectors, and tolerances; every lookup on an empty registry fails closed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ApprovedReferenceRegistry {
    /// Approved implementations.
    pub entries: Vec<ApprovedReferenceEntry>,
}

impl ApprovedReferenceRegistry {
    /// Find the entry backing an observation's implementation identity.
    ///
    /// # Errors
    ///
    /// Returns `blocked_external` when no entry carries the identity: either
    /// no implementation is approved yet or the claimant names an unknown one.
    pub fn find_entry(&self, implementation_id: &str) -> Result<&ApprovedReferenceEntry, String> {
        self.entries
            .iter()
            .find(|entry| entry.implementation_id == implementation_id)
            .ok_or_else(|| {
                format!(
                    "{BLOCKED_EXTERNAL_PREFIX}: no approved reference implementation '{implementation_id}'"
                )
            })
    }

    /// Verify observed agreement against the registry and the claim context.
    ///
    /// The evidence must name an approved implementation at its pinned hash
    /// and vectors, match the claimed model, edition, calibration, and
    /// domain, and show finite observed error within the predeclared
    /// tolerance over nonempty coverage. Stale hashes, wrong editions,
    /// and NaN errors fail; an empty registry blocks.
    ///
    /// # Errors
    ///
    /// Returns `blocked_external` when approval or evidence is missing, and
    /// a hard error when present evidence disagrees with its approval.
    pub fn verify_evidence(
        &self,
        evidence: &VerifiedReferenceEvidence,
        model_family: &str,
        edition: &str,
        calibration_id: &str,
        domain: &str,
    ) -> Result<(), String> {
        if evidence.vectors_id.trim().is_empty() {
            return Err(format!(
                "{BLOCKED_EXTERNAL_PREFIX}: reference evidence needs a nonempty independent vector set"
            ));
        }
        let entry = self.find_entry(&evidence.implementation_id)?;
        let tolerance = entry
            .error_tolerance
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| {
                format!("{BLOCKED_EXTERNAL_PREFIX}: approval needs a finite non-negative tolerance")
            })?;
        if entry.error_metric.trim().is_empty()
            || entry.required_coverage.is_empty()
            || entry
                .required_coverage
                .iter()
                .any(|case| case.trim().is_empty())
        {
            return Err(format!(
                "{BLOCKED_EXTERNAL_PREFIX}: approval needs a metric and required coverage"
            ));
        }
        if evidence.error_metric != entry.error_metric || evidence.error_tolerance != tolerance {
            return Err(String::from(
                "reference evidence metric or tolerance differs from approved criteria",
            ));
        }
        if !entry
            .required_coverage
            .iter()
            .all(|case| evidence.coverage.contains(case))
        {
            return Err(String::from(
                "reference evidence does not cover every approved reference case",
            ));
        }
        if entry.implementation_hash != evidence.implementation_hash {
            return Err(format!(
                "reference evidence for '{}' is stale: observed at '{}', approved '{}'",
                evidence.implementation_id, evidence.implementation_hash, entry.implementation_hash
            ));
        }
        for (field, observed, approved) in [
            ("vectors", &evidence.vectors_id, &entry.vectors_id),
            ("model family", &evidence.model_family, &entry.model_family),
            ("edition", &evidence.edition, &entry.edition),
            (
                "calibration",
                &evidence.calibration_id,
                &entry.calibration_id,
            ),
            ("domain", &evidence.domain, &entry.domain),
        ] {
            if observed != approved {
                return Err(format!(
                    "reference evidence {field} mismatch: observed '{observed}', approved '{approved}'"
                ));
            }
        }
        for (field, claimed, expected) in [
            ("model family", &evidence.model_family, model_family),
            ("edition", &evidence.edition, edition),
            ("calibration", &evidence.calibration_id, calibration_id),
            ("domain", &evidence.domain, domain),
        ] {
            if claimed != expected {
                return Err(format!(
                    "reference evidence {field} '{claimed}' does not match the '{expected}' claim"
                ));
            }
        }
        if evidence.error_metric.trim().is_empty() {
            return Err(format!(
                "{BLOCKED_EXTERNAL_PREFIX}: reference evidence needs a named error metric"
            ));
        }
        if !evidence.observed_error.is_finite() {
            return Err(String::from(
                "reference evidence observed error is not finite: no agreement demonstrated",
            ));
        }
        if !evidence.error_tolerance.is_finite() || evidence.error_tolerance < 0.0 {
            return Err(String::from(
                "reference evidence tolerance must be a finite non-negative number",
            ));
        }
        if evidence.coverage.is_empty()
            || evidence.coverage.iter().any(|item| item.trim().is_empty())
        {
            return Err(format!(
                "{BLOCKED_EXTERNAL_PREFIX}: reference evidence needs nonempty applicable coverage"
            ));
        }
        if evidence.observed_error > tolerance {
            return Err(format!(
                "reference agreement failed: observed {} {} exceeds tolerance {} {}",
                evidence.observed_error, evidence.error_metric, tolerance, evidence.error_metric
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod reference_registry_tests {
    use super::*;

    fn entry() -> ApprovedReferenceEntry {
        ApprovedReferenceEntry {
            implementation_id: "pemo-q-ref-impl".to_string(),
            implementation_hash: "abc123".to_string(),
            vectors_id: "g7-reference-vectors-v1".to_string(),
            model_family: "pemo-q".to_string(),
            edition: "pemo-q-2024-ed1".to_string(),
            calibration_id: "spl-cal-94db".to_string(),
            domain: "mono 50-80 dB SPL, 100 Hz-8 kHz".to_string(),
            error_metric: "max_abs_protocol_p_error".to_string(),
            error_tolerance: Some(1e-9),
            required_coverage: vec!["level-sweep".to_string(), "bandwidth".to_string()],
        }
    }

    fn evidence() -> VerifiedReferenceEvidence {
        VerifiedReferenceEvidence {
            implementation_id: "pemo-q-ref-impl".to_string(),
            implementation_hash: "abc123".to_string(),
            vectors_id: "g7-reference-vectors-v1".to_string(),
            model_family: "pemo-q".to_string(),
            edition: "pemo-q-2024-ed1".to_string(),
            calibration_id: "spl-cal-94db".to_string(),
            domain: "mono 50-80 dB SPL, 100 Hz-8 kHz".to_string(),
            error_metric: "max_abs_protocol_p_error".to_string(),
            observed_error: 1e-10,
            error_tolerance: 1e-9,
            coverage: vec!["level-sweep".to_string(), "bandwidth".to_string()],
        }
    }

    fn registry() -> ApprovedReferenceRegistry {
        ApprovedReferenceRegistry {
            entries: vec![entry()],
        }
    }

    #[test]
    fn empty_registry_blocks() {
        let error = ApprovedReferenceRegistry::default()
            .verify_evidence(
                &evidence(),
                "pemo-q",
                "pemo-q-2024-ed1",
                "spl-cal-94db",
                "mono 50-80 dB SPL, 100 Hz-8 kHz",
            )
            .expect_err("empty registry must block");
        assert!(error.starts_with(BLOCKED_EXTERNAL_PREFIX), "{error}");
    }

    #[test]
    fn matching_evidence_verifies() {
        registry()
            .verify_evidence(
                &evidence(),
                "pemo-q",
                "pemo-q-2024-ed1",
                "spl-cal-94db",
                "mono 50-80 dB SPL, 100 Hz-8 kHz",
            )
            .expect("matching evidence verifies");
    }

    #[test]
    fn roadmap_correction_promotion_cannot_choose_its_own_acceptance_criteria() {
        let mut inflated = evidence();
        inflated.observed_error = 1.0;
        inflated.error_tolerance = 2.0;
        let mut wrong_metric = evidence();
        wrong_metric.error_metric = "unapproved_metric".to_string();
        let mut incomplete = evidence();
        incomplete.coverage = vec!["level-sweep".to_string()];
        for claim in [inflated, wrong_metric, incomplete] {
            assert!(
                registry()
                    .verify_evidence(
                        &claim,
                        "pemo-q",
                        "pemo-q-2024-ed1",
                        "spl-cal-94db",
                        "mono 50-80 dB SPL, 100 Hz-8 kHz",
                    )
                    .is_err(),
                "claimant-selected acceptance criteria must be refused: {claim:?}",
            );
        }
    }

    #[test]
    fn roadmap_correction_promotion_legacy_approval_is_not_authorization() {
        let mut serialized = serde_json::to_value(entry()).unwrap();
        let fields = serialized.as_object_mut().unwrap();
        for key in ["error_metric", "error_tolerance", "required_coverage"] {
            fields.remove(key);
        }
        let legacy: ApprovedReferenceEntry = serde_json::from_value(serialized).unwrap();
        let registry = ApprovedReferenceRegistry {
            entries: vec![legacy],
        };
        let error = registry
            .verify_evidence(
                &evidence(),
                "pemo-q",
                "pemo-q-2024-ed1",
                "spl-cal-94db",
                "mono 50-80 dB SPL, 100 Hz-8 kHz",
            )
            .expect_err("legacy approval has no predeclared acceptance criteria");
        assert!(error.starts_with(BLOCKED_EXTERNAL_PREFIX), "{error}");
    }

    #[test]
    fn stale_hash_rejected() {
        let mut stale = evidence();
        stale.implementation_hash = "old-hash".to_string();
        assert!(
            registry()
                .verify_evidence(
                    &stale,
                    "pemo-q",
                    "pemo-q-2024-ed1",
                    "spl-cal-94db",
                    "mono 50-80 dB SPL, 100 Hz-8 kHz"
                )
                .is_err()
        );
    }

    #[test]
    fn failed_agreement_rejected() {
        let mut failed = evidence();
        failed.observed_error = 2e-9;
        assert!(
            registry()
                .verify_evidence(
                    &failed,
                    "pemo-q",
                    "pemo-q-2024-ed1",
                    "spl-cal-94db",
                    "mono 50-80 dB SPL, 100 Hz-8 kHz"
                )
                .is_err()
        );
    }

    #[test]
    fn nan_error_rejected() {
        let mut nan = evidence();
        nan.observed_error = f64::NAN;
        assert!(
            registry()
                .verify_evidence(
                    &nan,
                    "pemo-q",
                    "pemo-q-2024-ed1",
                    "spl-cal-94db",
                    "mono 50-80 dB SPL, 100 Hz-8 kHz"
                )
                .is_err()
        );
    }

    #[test]
    fn wrong_edition_rejected() {
        assert!(
            registry()
                .verify_evidence(
                    &evidence(),
                    "pemo-q",
                    "other-edition",
                    "spl-cal-94db",
                    "mono 50-80 dB SPL, 100 Hz-8 kHz"
                )
                .is_err()
        );
    }
}
