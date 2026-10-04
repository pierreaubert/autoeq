//! Evidence-gated rollout and release conformance (Stage 5).
//!
//! Four release gates stay independent: implementation correctness,
//! physical safety, perceptual-model validation, and demonstrated
//! listening benefit. Passing one never implies the others. A rule or
//! policy promotes only with the gates its claim needs: an advisory
//! release or an elapsed warning cycle is `NotAssessed`, never a pass.
//! The Stage 4 rerank objective is independently optional and never
//! blocks well-supported physical safeguards. Policy records carry an
//! explicit selection and version, and disabling a policy restores the
//! legacy behavior it replaced.

use roomeq_model::FilterAudibilityConfig;
use serde::{Deserialize, Serialize};

/// Independent release gates. Each assesses one kind of evidence;
/// absence of evidence is `NotAssessed`, never a quiet pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseGate {
    /// Implementation correctness (unit/integration/QA suites green).
    ImplementationCorrectness,
    /// Physical safety (headroom, stability, excursion, routing guards).
    PhysicalSafety,
    /// Perceptual-model validation (staged metrics against references).
    PerceptualValidation,
    /// Demonstrated listening benefit (recorded listening outcomes).
    ListeningBenefit,
}

/// Gate outcome with its evidence reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateAssessment {
    /// The gate assessed.
    pub gate: ReleaseGate,
    /// Whether the gate passed.
    pub passed: bool,
    /// Evidence reference (test suite id, report hash, protocol hash).
    /// Empty means unassessed: callers must not read a pass from it.
    #[serde(default)]
    pub evidence: String,
}

impl GateAssessment {
    /// A gate counts as passed only with non-blank evidence. An advisory
    /// release or an elapsed warning cycle carries no evidence and stays
    /// unassessed — it never promotes.
    pub fn effective_pass(&self) -> bool {
        self.passed && !self.evidence.trim().is_empty()
    }
}

/// What a policy release actually does when selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyBehavior {
    /// Policy disabled: legacy behavior, byte-for-byte where the owning
    /// path guarantees it. Always promotable — disabling never needs a
    /// gate, so rollback is never blocked.
    Legacy,
    /// Report-only/advisory: evaluates and records, changes nothing.
    /// Safe to ship, but promotes nothing by itself.
    Advisory,
    /// Enforcing: changes emitted output. Needs gates per its claim.
    Enforcing,
}

/// A versioned, explicitly selected policy release record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyRelease {
    /// Policy id, e.g. `"filter-audibility-veto"`.
    pub policy_id: String,
    /// Policy version.
    pub version: String,
    /// What selecting this record does.
    pub behavior: PolicyBehavior,
    /// Whether the policy makes a perceptual claim (needs the perceptual
    /// gate) or only a physical-safety one.
    pub perceptual_claim: bool,
}

impl PolicyRelease {
    /// Shape validation: identified, versioned, explicit.
    pub fn validate(&self) -> Result<(), String> {
        if self.policy_id.trim().is_empty() || self.version.trim().is_empty() {
            return Err(String::from("policy release needs an id and a version"));
        }
        Ok(())
    }

    /// Promotion decision against assessed gates.
    ///
    /// Legacy always promotes (rollback is never gated). Advisory never
    /// promotes (observation is not validation). Enforcing physical-only
    /// policies need correctness + physical safety; perceptual claims
    /// additionally need perceptual validation, and listening-benefit
    /// claims need demonstrated benefit. The Stage 4 objective rides the
    /// same rules as any other policy — it is optional and never a
    /// prerequisite for safeguards.
    pub fn promotion(
        &self,
        gates: &[GateAssessment],
        claims_listening_benefit: bool,
    ) -> Result<Promotion, String> {
        self.validate()?;
        if self.behavior == PolicyBehavior::Legacy {
            return Ok(Promotion::Promoted(String::from(
                "legacy behavior: no gates required",
            )));
        }
        if self.behavior == PolicyBehavior::Advisory {
            return Ok(Promotion::Held(String::from(
                "advisory releases observe but do not validate: not promoted",
            )));
        }
        let mut need = vec![
            ReleaseGate::ImplementationCorrectness,
            ReleaseGate::PhysicalSafety,
        ];
        if self.perceptual_claim {
            need.push(ReleaseGate::PerceptualValidation);
        }
        if claims_listening_benefit {
            need.push(ReleaseGate::ListeningBenefit);
        }
        let mut missing = Vec::new();
        for gate in need {
            let pass = gates
                .iter()
                .any(|assessment| assessment.gate == gate && assessment.effective_pass());
            if !pass {
                missing.push(format!("{gate:?}"));
            }
        }
        if missing.is_empty() {
            Ok(Promotion::Promoted(String::from(
                "required gates pass with evidence",
            )))
        } else {
            Ok(Promotion::Held(format!(
                "missing or unassessed gates: {}",
                missing.join(", ")
            )))
        }
    }
}

/// Promotion outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Promotion {
    /// Promoted with the reason.
    Promoted(String),
    /// Held with the reason.
    Held(String),
}

impl Promotion {
    /// True only for promotion.
    pub fn promoted(&self) -> bool {
        matches!(self, Self::Promoted(_))
    }
}

/// Map the real per-filter veto configuration to its release behavior.
///
/// `None` disables the veto entirely (legacy output); the default config
/// is report-only (advisory); only an explicit opt-in enforces. This is
/// the legacy-restore guarantee: deleting the selection returns the
/// legacy behavior the policy replaced.
pub fn veto_release_behavior(config: Option<&FilterAudibilityConfig>) -> PolicyBehavior {
    match config {
        None => PolicyBehavior::Legacy,
        Some(config) if !config.enabled => PolicyBehavior::Legacy,
        Some(config) if config.enforcement_authorized() => PolicyBehavior::Enforcing,
        // Report-only, or enforcement requested without the experimental
        // acknowledgment (which stays advisory): no applied change.
        Some(_) => PolicyBehavior::Advisory,
    }
}

/// Release record for the per-filter audibility veto at one selection.
pub fn veto_policy_release(
    config: Option<&FilterAudibilityConfig>,
    version: &str,
) -> PolicyRelease {
    PolicyRelease {
        policy_id: String::from("filter-audibility-veto"),
        version: String::from(version),
        behavior: veto_release_behavior(config),
        perceptual_claim: true,
    }
}

/// Outcome of one real listening trial leg.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialOutcome {
    /// Preregistered benefit criterion met on real trials.
    Benefit,
    /// Preregistered equivalence criterion met on real trials.
    Equivalence,
    /// Nonsignificant, underpowered, or otherwise undecided.
    Inconclusive,
}

/// Provenance of perceptual or listening evidence presented to a gate.
///
/// These descriptors explain the claimed source of evidence. They are not
/// validation receipts: a string saying "independent" or "real" cannot
/// promote either G7 gate on its own.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceProvenance {
    /// Pinned independent reference vector with declared domain/tolerance.
    IndependentReference {
        reference_id: String,
        calibration_domain: String,
        tolerance: String,
    },
    /// Approximate model metric (e.g. an audibility proxy): advisory only.
    ExperimentalProxy { metric_id: String },
    /// Simulated listeners or offline score deltas: never listening evidence.
    SyntheticTrial { seed: u64, cases: usize },
    /// Real listeners under a frozen protocol hash.
    RealTrial {
        protocol_hash: String,
        sufficient: bool,
        outcome: TrialOutcome,
    },
}

impl EvidenceProvenance {
    /// Descriptive provenance alone never validates numeric model agreement.
    pub fn accepts_perceptual_validation(&self) -> bool {
        false
    }

    /// A descriptor cannot prove trial import, binding, or statistical result.
    pub fn accepts_listening_benefit(&self, claims_equivalence: bool) -> bool {
        let _ = claims_equivalence;
        false
    }
}

/// Assess the perceptual-validation gate from evidence provenance.
///
/// Descriptive evidence yields an unassessed gate; numeric reference evidence
/// must pass the approved-registry path below.
pub fn assess_perceptual_gate(provenance: &EvidenceProvenance) -> GateAssessment {
    if provenance.accepts_perceptual_validation() {
        GateAssessment {
            gate: ReleaseGate::PerceptualValidation,
            passed: true,
            evidence: format!("independent-reference:{provenance:?}"),
        }
    } else {
        GateAssessment {
            gate: ReleaseGate::PerceptualValidation,
            passed: false,
            evidence: String::new(),
        }
    }
}

/// Assess model validation from a licensed model pin, hashed vector files,
/// and approved numeric reference evidence. The default registry is empty
/// until the G7 lane supplies independent vectors and approves their
/// implementation identity.
pub fn assess_verified_perceptual_gate(
    model: &autoeq_optim::perceptual_promotion::PinnedFidelityModel,
    readiness: &roomeq_quality::EnforcementReadiness,
    registry: &roomeq_model::reference_registry::ApprovedReferenceRegistry,
    independent_vectors: &[crate::corpus::ReferenceVector],
    generated_controls: &[crate::corpus::ReferenceVector],
) -> GateAssessment {
    let passed = model.validate().is_ok()
        && crate::corpus::validate_reference_separation(independent_vectors, generated_controls)
            .is_ok()
        && independent_vectors.iter().all(|vector| {
            vector.set_id == model.reference_vectors_id
                && vector.edition == model.edition
                && vector.license_id == model.license_id
        })
        && readiness.model_family == model.model_family
        && readiness.edition == model.edition
        && readiness
            .reference_evidence
            .as_ref()
            .is_some_and(|evidence| {
                evidence.vectors_id == model.reference_vectors_id
                    && evidence.implementation_hash == model.implementation_commit
            })
        && readiness.check_ready(registry).is_ok();
    GateAssessment {
        gate: ReleaseGate::PerceptualValidation,
        passed,
        evidence: if passed {
            let mut vector_hashes = independent_vectors
                .iter()
                .map(|vector| format!("{}={}", vector.id, vector.hash.to_ascii_lowercase()))
                .collect::<Vec<_>>();
            vector_hashes.sort();
            format!(
                "approved-reference:{}:{}:{}:{}:{}",
                model.edition,
                model.license_id,
                model.implementation_commit,
                model.reference_vectors_id,
                vector_hashes.join(",")
            )
        } else {
            String::new()
        },
    }
}

/// Assess the listening-benefit gate from evidence provenance.
///
/// A provenance description alone cannot certify a real, protocol-bound
/// listening result. The G7 importer must supply a verified verdict before
/// this gate can pass.
pub fn assess_listening_gate(
    provenance: &EvidenceProvenance,
    claims_equivalence: bool,
) -> GateAssessment {
    if provenance.accepts_listening_benefit(claims_equivalence) {
        GateAssessment {
            gate: ReleaseGate::ListeningBenefit,
            passed: true,
            evidence: format!("real-trial:{provenance:?}"),
        }
    } else {
        GateAssessment {
            gate: ReleaseGate::ListeningBenefit,
            passed: false,
            evidence: String::new(),
        }
    }
}

#[cfg(test)]
mod release_gates_tests {
    use super::*;

    fn gates_with(
        correctness: bool,
        safety: bool,
        perceptual: bool,
        benefit: bool,
    ) -> Vec<GateAssessment> {
        [
            (
                ReleaseGate::ImplementationCorrectness,
                correctness,
                "qa-suite-green",
            ),
            (
                ReleaseGate::PhysicalSafety,
                safety,
                "headroom-stability-report",
            ),
            (
                ReleaseGate::PerceptualValidation,
                perceptual,
                "staged-metric-report",
            ),
            (ReleaseGate::ListeningBenefit, benefit, "abx-protocol-hash"),
        ]
        .into_iter()
        .map(|(gate, passed, evidence)| GateAssessment {
            gate,
            passed,
            evidence: String::from(evidence),
        })
        .collect()
    }

    fn enforcing_physical() -> PolicyRelease {
        PolicyRelease {
            policy_id: String::from("headroom-guard"),
            version: String::from("1.0.0"),
            behavior: PolicyBehavior::Enforcing,
            perceptual_claim: false,
        }
    }

    #[test]
    fn gates_pass_separately_or_not_at_all() {
        // Physical safeguard promotes on correctness + safety alone:
        // Stage 4 (perceptual/listening) evidence is not required and its
        // absence must not block safeguards.
        let promotion = enforcing_physical()
            .promotion(&gates_with(true, true, false, false), false)
            .unwrap();
        assert!(promotion.promoted(), "{promotion:?}");
        // Either gate failing holds the release.
        let held = enforcing_physical()
            .promotion(&gates_with(true, false, true, true), false)
            .unwrap();
        assert!(!held.promoted());
        // A pass without evidence is unassessed, not a pass.
        let mut no_evidence = gates_with(true, true, false, false);
        no_evidence[1].evidence = String::from("");
        assert!(
            !enforcing_physical()
                .promotion(&no_evidence, false)
                .unwrap()
                .promoted()
        );
    }

    #[test]
    fn perceptual_and_benefit_claims_need_their_gates() {
        let perceptual = PolicyRelease {
            perceptual_claim: true,
            ..enforcing_physical()
        };
        // Staged metrics without listening: perceptual claims promote,
        // benefit claims hold.
        let gates = gates_with(true, true, true, false);
        assert!(perceptual.promotion(&gates, false).unwrap().promoted());
        assert!(!perceptual.promotion(&gates, true).unwrap().promoted());
        // Warning-cycle elapsed but no evidence recorded: held.
        let mut elapsed = gates.clone();
        elapsed[2].evidence = String::from("  ");
        assert!(!perceptual.promotion(&elapsed, false).unwrap().promoted());
    }

    #[test]
    fn advisory_never_promotes_and_legacy_never_blocks() {
        let advisory = PolicyRelease {
            behavior: PolicyBehavior::Advisory,
            ..enforcing_physical()
        };
        let full = gates_with(true, true, true, true);
        assert!(!advisory.promotion(&full, false).unwrap().promoted());
        let legacy = PolicyRelease {
            behavior: PolicyBehavior::Legacy,
            ..enforcing_physical()
        };
        let none = gates_with(false, false, false, false);
        assert!(legacy.promotion(&none, true).unwrap().promoted());
        // Releases must be identified and versioned.
        let mut unidentified = enforcing_physical();
        unidentified.policy_id = String::from("");
        assert!(unidentified.promotion(&full, false).is_err());
    }

    #[test]
    fn veto_selection_maps_to_legacy_advisory_enforcing() {
        // No selection: legacy.
        assert_eq!(veto_release_behavior(None), PolicyBehavior::Legacy);
        // Default config ships report-only: advisory, never promoted.
        let default = FilterAudibilityConfig::default();
        assert!(default.report_only);
        assert_eq!(
            veto_release_behavior(Some(&default)),
            PolicyBehavior::Advisory
        );
        let record = veto_policy_release(Some(&default), "phase-a-1.0");
        assert_eq!(record.behavior, PolicyBehavior::Advisory);
        let full = gates_with(true, true, true, true);
        assert!(!record.promotion(&full, false).unwrap().promoted());
        // Disabled flag: legacy even when a config is present.
        let off = FilterAudibilityConfig {
            enabled: false,
            ..FilterAudibilityConfig::default()
        };
        assert_eq!(veto_release_behavior(Some(&off)), PolicyBehavior::Legacy);
        // Explicit opt-in: enforcing, gated on correctness + safety +
        // perceptual (the veto makes a perceptual claim).
        let enforcing = FilterAudibilityConfig {
            report_only: false,
            ..FilterAudibilityConfig::default()
        };
        // Without the experimental acknowledgment, enforcement stays advisory.
        assert_eq!(
            veto_release_behavior(Some(&enforcing)),
            PolicyBehavior::Advisory
        );
        let enforcing = FilterAudibilityConfig {
            allow_enforcement_with_experimental_proxy: true,
            ..enforcing
        };
        assert_eq!(
            veto_release_behavior(Some(&enforcing)),
            PolicyBehavior::Enforcing
        );
        let record = veto_policy_release(Some(&enforcing), "phase-a-1.0");
        assert!(
            !record
                .promotion(&gates_with(true, true, false, false), false)
                .unwrap()
                .promoted()
        );
        assert!(record.promotion(&full, false).unwrap().promoted());
    }

    fn perceptual_release() -> PolicyRelease {
        PolicyRelease {
            policy_id: String::from("staged-perceptual-metric"),
            version: String::from("1.0.0"),
            behavior: PolicyBehavior::Enforcing,
            perceptual_claim: true,
        }
    }

    fn correctness_and_safety() -> Vec<GateAssessment> {
        vec![
            GateAssessment {
                gate: ReleaseGate::ImplementationCorrectness,
                passed: true,
                evidence: String::from("qa-suite-green"),
            },
            GateAssessment {
                gate: ReleaseGate::PhysicalSafety,
                passed: true,
                evidence: String::from("headroom-stability-report"),
            },
        ]
    }

    #[test]
    fn qa_proxy_does_not_promote_perceptual_validation() {
        // An experimental proxy is advisory evidence: it never validates
        // the perceptual model, so the perceptual gate stays unassessed
        // and the release is held even with correctness and safety green.
        let proxy = EvidenceProvenance::ExperimentalProxy {
            metric_id: String::from("audibility-proxy-v1"),
        };
        assert!(!proxy.accepts_perceptual_validation());
        let mut gates = correctness_and_safety();
        gates.push(assess_perceptual_gate(&proxy));
        assert!(
            !perceptual_release()
                .promotion(&gates, false)
                .unwrap()
                .promoted()
        );
        // Descriptive fields do not prove an independently reproduced
        // numeric comparison against an approved reference registry.
        let pinned = EvidenceProvenance::IndependentReference {
            reference_id: String::from("iso-226-2024"),
            calibration_domain: String::from("20Hz-500Hz small rooms"),
            tolerance: String::from("+-1dB RMS"),
        };
        assert!(!pinned.accepts_perceptual_validation());
        let mut unverified = correctness_and_safety();
        unverified.push(assess_perceptual_gate(&pinned));
        assert!(
            !perceptual_release()
                .promotion(&unverified, false)
                .unwrap()
                .promoted()
        );
        // An unpinned reference (missing domain) does not validate.
        let unpinned = EvidenceProvenance::IndependentReference {
            reference_id: String::from("iso-226-2024"),
            calibration_domain: String::new(),
            tolerance: String::from("+-1dB RMS"),
        };
        assert!(!unpinned.accepts_perceptual_validation());
    }

    #[test]
    fn qa_unapproved_reference_cannot_promote_model_validation() {
        use sha2::Digest;

        let model = autoeq_optim::perceptual_promotion::PinnedFidelityModel {
            model_family: "pemo-q".to_string(),
            edition: "test-edition".to_string(),
            implementation_commit: "test-commit".to_string(),
            license_id: "test-license".to_string(),
            reference_vectors_id: "test-vectors".to_string(),
        };
        let mut readiness = roomeq_quality::EnforcementReadiness {
            reference: "test reference".to_string(),
            calibration_id: "test-calibration".to_string(),
            validated_domain: "test-domain".to_string(),
            tolerances: vec!["test tolerance".to_string()],
            model_family: model.model_family.clone(),
            edition: model.edition.clone(),
            reference_evidence: None,
        };
        let temp = tempfile::tempdir().unwrap();
        let artifact_path = temp.path().join("test-vector.json");
        let fixture = br#"{"cases":[{"id":"test-case","expected":0.1}]}"#;
        std::fs::write(&artifact_path, fixture).unwrap();
        let hash = sha2::Sha256::digest(fixture)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let vector = crate::corpus::ReferenceVector {
            set_id: model.reference_vectors_id.clone(),
            id: "test-vector".to_string(),
            hash,
            artifact_path: artifact_path.clone(),
            edition: model.edition.clone(),
            license_id: model.license_id.clone(),
            independent: true,
        };
        let registry = roomeq_model::reference_registry::ApprovedReferenceRegistry::default();
        assert!(
            !assess_verified_perceptual_gate(
                &model,
                &readiness,
                &registry,
                std::slice::from_ref(&vector),
                &[]
            )
            .passed
        );

        // A test-local approved fixture exercises the positive software path;
        // it is not an independent model reference or a G7 result.
        let evidence = roomeq_model::reference_registry::VerifiedReferenceEvidence {
            implementation_id: "test-implementation".to_string(),
            implementation_hash: model.implementation_commit.clone(),
            vectors_id: model.reference_vectors_id.clone(),
            model_family: model.model_family.clone(),
            edition: model.edition.clone(),
            calibration_id: readiness.calibration_id.clone(),
            domain: readiness.validated_domain.clone(),
            error_metric: "test-error".to_string(),
            observed_error: 0.1,
            error_tolerance: 0.2,
            coverage: vec!["test-case".to_string()],
        };
        let registry = roomeq_model::reference_registry::ApprovedReferenceRegistry {
            entries: vec![roomeq_model::reference_registry::ApprovedReferenceEntry {
                implementation_id: evidence.implementation_id.clone(),
                implementation_hash: evidence.implementation_hash.clone(),
                vectors_id: evidence.vectors_id.clone(),
                model_family: evidence.model_family.clone(),
                edition: evidence.edition.clone(),
                calibration_id: evidence.calibration_id.clone(),
                domain: evidence.domain.clone(),
                error_metric: evidence.error_metric.clone(),
                error_tolerance: Some(evidence.error_tolerance),
                required_coverage: evidence.coverage.clone(),
            }],
        };
        readiness.reference_evidence = Some(evidence);
        let accepted = assess_verified_perceptual_gate(
            &model,
            &readiness,
            &registry,
            std::slice::from_ref(&vector),
            &[],
        );
        assert!(accepted.passed);
        assert!(accepted.evidence.contains(&vector.hash));
        assert!(accepted.evidence.contains(&model.implementation_commit));
        let mut mismatched = model.clone();
        mismatched.reference_vectors_id = "different-vectors".to_string();
        assert!(
            !assess_verified_perceptual_gate(
                &mismatched,
                &readiness,
                &registry,
                std::slice::from_ref(&vector),
                &[]
            )
            .passed
        );
        mismatched = model.clone();
        mismatched.license_id.clear();
        assert!(
            !assess_verified_perceptual_gate(
                &mismatched,
                &readiness,
                &registry,
                std::slice::from_ref(&vector),
                &[]
            )
            .passed
        );
        std::fs::remove_file(&artifact_path).unwrap();
        assert!(
            !assess_verified_perceptual_gate(&model, &readiness, &registry, &[vector], &[]).passed
        );
    }

    #[test]
    fn qa_synthetic_trials_do_not_promote_listening_benefit() {
        // Simulated listeners are programme output, not listener evidence.
        let synthetic = EvidenceProvenance::SyntheticTrial {
            seed: 42,
            cases: 1000,
        };
        assert!(!synthetic.accepts_listening_benefit(false));
        let mut gates = correctness_and_safety();
        gates.push(assess_listening_gate(&synthetic, false));
        assert!(
            !perceptual_release()
                .promotion(&gates, true)
                .unwrap()
                .promoted()
        );
        // A claimed real trial still needs a verified imported result.
        let real = EvidenceProvenance::RealTrial {
            protocol_hash: String::from("abx-protocol-v2:9f3a"),
            sufficient: true,
            outcome: TrialOutcome::Benefit,
        };
        assert!(!real.accepts_listening_benefit(false));
        // A real benefit trial does not satisfy an equivalence claim.
        assert!(!real.accepts_listening_benefit(true));
    }

    #[test]
    fn qa_inconclusive_trial_does_not_promote() {
        // Nonsignificant ABX, bare preference, and underpowered trials are
        // recorded as inconclusive and never promote either claim.
        let trial = EvidenceProvenance::RealTrial {
            protocol_hash: String::from("abx-protocol-v2:9f3a"),
            sufficient: true,
            outcome: TrialOutcome::Inconclusive,
        };
        assert!(!trial.accepts_listening_benefit(false));
        assert!(!trial.accepts_listening_benefit(true));
        // An insufficient trial with an apparent benefit is still held.
        let underpowered = EvidenceProvenance::RealTrial {
            protocol_hash: String::from("abx-protocol-v2:9f3a"),
            sufficient: false,
            outcome: TrialOutcome::Benefit,
        };
        assert!(!underpowered.accepts_listening_benefit(false));
        let mut gates = correctness_and_safety();
        gates.push(assess_listening_gate(&underpowered, false));
        assert!(
            !perceptual_release()
                .promotion(&gates, true)
                .unwrap()
                .promoted()
        );
    }

    #[test]
    fn qa_physical_safety_can_promote_without_listening_claim() {
        // A physical-only safeguard promotes on correctness + safety alone:
        // absent perceptual and listening evidence must not block it, and
        // the promotion carries no perceptual or listening claim.
        let safeguard = enforcing_physical();
        assert!(!safeguard.perceptual_claim);
        let gates = vec![
            GateAssessment {
                gate: ReleaseGate::ImplementationCorrectness,
                passed: true,
                evidence: String::from("qa-suite-green"),
            },
            GateAssessment {
                gate: ReleaseGate::PhysicalSafety,
                passed: true,
                evidence: String::from("headroom-stability-report"),
            },
            GateAssessment {
                gate: ReleaseGate::PerceptualValidation,
                passed: false,
                evidence: String::new(),
            },
            GateAssessment {
                gate: ReleaseGate::ListeningBenefit,
                passed: false,
                evidence: String::new(),
            },
        ];
        let promotion = safeguard.promotion(&gates, false).unwrap();
        assert!(promotion.promoted(), "{promotion:?}");
        // Claiming a listening benefit re-adds the listening gate.
        assert!(!safeguard.promotion(&gates, true).unwrap().promoted());
    }
}
