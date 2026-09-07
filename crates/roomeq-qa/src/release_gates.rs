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
            return Ok(Promotion::Promoted(String::from("legacy behavior: no gates required")));
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
            let pass = gates.iter().any(|assessment| {
                assessment.gate == gate && assessment.effective_pass()
            });
            if !pass {
                missing.push(format!("{gate:?}"));
            }
        }
        if missing.is_empty() {
            Ok(Promotion::Promoted(String::from("required gates pass with evidence")))
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
pub fn veto_policy_release(config: Option<&FilterAudibilityConfig>, version: &str) -> PolicyRelease {
    PolicyRelease {
        policy_id: String::from("filter-audibility-veto"),
        version: String::from(version),
        behavior: veto_release_behavior(config),
        perceptual_claim: true,
    }
}

#[cfg(test)]
mod release_gates_tests {
    use super::*;

    fn gates_with(correctness: bool, safety: bool, perceptual: bool, benefit: bool) -> Vec<GateAssessment> {
        [
            (ReleaseGate::ImplementationCorrectness, correctness, "qa-suite-green"),
            (ReleaseGate::PhysicalSafety, safety, "headroom-stability-report"),
            (ReleaseGate::PerceptualValidation, perceptual, "staged-metric-report"),
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
        assert!(!enforcing_physical().promotion(&no_evidence, false).unwrap().promoted());
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
        let mut off = FilterAudibilityConfig::default();
        off.enabled = false;
        assert_eq!(veto_release_behavior(Some(&off)), PolicyBehavior::Legacy);
        // Explicit opt-in: enforcing, gated on correctness + safety +
        // perceptual (the veto makes a perceptual claim).
        let mut enforcing = FilterAudibilityConfig::default();
        enforcing.report_only = false;
        // Without the experimental acknowledgment, enforcement stays advisory.
        assert_eq!(
            veto_release_behavior(Some(&enforcing)),
            PolicyBehavior::Advisory
        );
        enforcing.allow_enforcement_with_experimental_proxy = true;
        assert_eq!(
            veto_release_behavior(Some(&enforcing)),
            PolicyBehavior::Enforcing
        );
        let record = veto_policy_release(Some(&enforcing), "phase-a-1.0");
        assert!(!record.promotion(&gates_with(true, true, false, false), false).unwrap().promoted());
        assert!(record.promotion(&full, false).unwrap().promoted());
    }
}
