//! Confidence-aware inversion support for bounded correction (Stage 3).
//!
//! Minimum-phase classification alone never authorizes a boost into a
//! null: the boost must also clear measurement-depth, cross-seat support,
//! and confidence gates. Misclassified, missing, or noisy evidence fails
//! closed to cuts-only or refusal — never to an assumed-safe boost. An
//! uncertain classification is not a verified physical cause.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Minimum-phase classification of a null under a boost request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NullClassification {
    /// Verified minimum-phase (invertible in principle).
    MinimumPhase,
    /// Cancellation or otherwise non-invertible null: boosts stay refused.
    NonMinimumPhase,
    /// Classifier could not decide: treated as unverified, never boosted.
    Uncertain,
}

/// Evidence for one boost-into-a-null request.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct InversionEvidence {
    /// Null depth in dB (positive).
    pub null_depth_db: f64,
    /// Classification of the null.
    pub classification: NullClassification,
    /// Classifier confidence in `[0, 1]`.
    pub classification_confidence: f64,
    /// Measurement correction-depth scale in `[0, 1]` (`1` = full trust).
    pub measurement_depth_scale: f64,
    /// Seats agreeing on the null and its classification.
    pub supporting_seats: usize,
    /// Frequency bins supporting the null.
    pub supporting_bins: usize,
    /// Requested boost in dB (non-negative).
    pub requested_boost_db: f64,
}

/// Conservative bounds for boost authorization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BoundedInversionPolicy {
    /// Absolute boost ceiling in dB.
    pub max_boost_db: f64,
    /// Minimum classifier confidence for any boost.
    pub min_confidence: f64,
    /// Minimum measurement depth scale for any boost.
    pub min_depth_scale: f64,
    /// Minimum supporting bins for any boost.
    pub min_support_bins: usize,
    /// Minimum agreeing seats for any boost.
    pub min_seat_agreement: usize,
}

/// Verdict on the boost request. Cuts elsewhere are unaffected by
/// [`InversionVerdict::CutsOnly`]; [`InversionVerdict::Refuse`] means the
/// evidence itself is absent and nothing should be inferred from this null.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InversionVerdict {
    /// Boost authorized up to the stated dB.
    Allowed,
    /// Boost denied; cuts may proceed.
    CutsOnly,
    /// Evidence absent: no decision about this null.
    Refuse,
}

/// Assessed boost request with its provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct InversionDecision {
    /// The verdict.
    pub verdict: InversionVerdict,
    /// Authorized boost in dB (`0` unless `verdict` is `Allowed`).
    pub authorized_boost_db: f64,
    /// Gate reasons, in evaluation order.
    pub reasons: Vec<String>,
}

impl BoundedInversionPolicy {
    /// Validate the policy shape (ranges, non-negative caps).
    pub fn validate(&self) -> Result<(), String> {
        if !self.max_boost_db.is_finite() || self.max_boost_db < 0.0 {
            return Err(String::from("max_boost_db must be finite and non-negative"));
        }
        for (name, value) in [
            ("min_confidence", self.min_confidence),
            ("min_depth_scale", self.min_depth_scale),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must lie in [0, 1]"));
            }
        }
        Ok(())
    }
}

/// Assess one boost request against the policy.
///
/// False classifications fail closed: anything but a confident
/// minimum-phase verdict denies the boost. Missing evidence (no
/// supporting bins or seats) refuses outright; noisy evidence (low depth
/// scale) denies the boost while leaving cuts available.
pub fn assess_inversion(
    evidence: &InversionEvidence,
    policy: &BoundedInversionPolicy,
) -> Result<InversionDecision, String> {
    policy.validate()?;
    if !evidence.null_depth_db.is_finite() || evidence.null_depth_db < 0.0 {
        return Err(String::from("null_depth_db must be finite and non-negative"));
    }
    for (name, value) in [
        ("classification_confidence", evidence.classification_confidence),
        ("measurement_depth_scale", evidence.measurement_depth_scale),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(format!("{name} must lie in [0, 1]"));
        }
    }
    if !evidence.requested_boost_db.is_finite() || evidence.requested_boost_db < 0.0 {
        return Err(String::from(
            "requested_boost_db must be finite and non-negative",
        ));
    }
    let refuse = |reason: &str| {
        Ok(InversionDecision {
            verdict: InversionVerdict::Refuse,
            authorized_boost_db: 0.0,
            reasons: vec![String::from(reason)],
        })
    };
    if evidence.supporting_bins == 0 {
        return refuse("no supporting bins: null evidence absent");
    }
    if evidence.supporting_seats == 0 {
        return refuse("no supporting seats: null evidence absent");
    }
    if evidence.requested_boost_db == 0.0 {
        return Ok(InversionDecision {
            verdict: InversionVerdict::Allowed,
            authorized_boost_db: 0.0,
            reasons: vec![String::from("no boost requested")],
        });
    }
    let cuts_only = |reasons: Vec<String>| {
        Ok(InversionDecision {
            verdict: InversionVerdict::CutsOnly,
            authorized_boost_db: 0.0,
            reasons,
        })
    };
    match evidence.classification {
        NullClassification::NonMinimumPhase => {
            return cuts_only(vec![String::from(
                "non-minimum-phase null (cancellation): boosts refused",
            )]);
        }
        NullClassification::Uncertain => {
            return cuts_only(vec![String::from(
                "uncertain classification is not a verified physical cause: boosts refused",
            )]);
        }
        NullClassification::MinimumPhase => {}
    }
    let mut failed = Vec::new();
    if evidence.classification_confidence < policy.min_confidence {
        failed.push(format!(
            "confidence {} below minimum {}",
            evidence.classification_confidence, policy.min_confidence
        ));
    }
    if evidence.measurement_depth_scale < policy.min_depth_scale {
        failed.push(format!(
            "measurement depth scale {} below minimum {}",
            evidence.measurement_depth_scale, policy.min_depth_scale
        ));
    }
    if evidence.supporting_bins < policy.min_support_bins {
        failed.push(format!(
            "supporting bins {} below minimum {}",
            evidence.supporting_bins, policy.min_support_bins
        ));
    }
    if evidence.supporting_seats < policy.min_seat_agreement {
        failed.push(format!(
            "supporting seats {} below minimum {}",
            evidence.supporting_seats, policy.min_seat_agreement
        ));
    }
    if !failed.is_empty() {
        return cuts_only(failed);
    }
    let authorized = evidence.requested_boost_db.min(policy.max_boost_db);
    let mut reasons = vec![String::from("minimum-phase with full support")];
    if authorized < evidence.requested_boost_db {
        reasons.push(format!(
            "request {} dB capped at policy maximum {} dB",
            evidence.requested_boost_db, policy.max_boost_db
        ));
    }
    Ok(InversionDecision {
        verdict: InversionVerdict::Allowed,
        authorized_boost_db: authorized,
        reasons,
    })
}

#[cfg(test)]
mod inversion_support_tests {
    use super::*;

    fn policy() -> BoundedInversionPolicy {
        BoundedInversionPolicy {
            max_boost_db: 9.0,
            min_confidence: 0.8,
            min_depth_scale: 0.5,
            min_support_bins: 8,
            min_seat_agreement: 2,
        }
    }

    fn evidence() -> InversionEvidence {
        InversionEvidence {
            null_depth_db: 12.0,
            classification: NullClassification::MinimumPhase,
            classification_confidence: 0.95,
            measurement_depth_scale: 1.0,
            supporting_seats: 4,
            supporting_bins: 24,
            requested_boost_db: 6.0,
        }
    }

    #[test]
    fn ideal_synthetic_null_authorizes_bounded_boost() {
        let decision = assess_inversion(&evidence(), &policy()).unwrap();
        assert_eq!(decision.verdict, InversionVerdict::Allowed);
        assert_eq!(decision.authorized_boost_db, 6.0);
        // Requests past the ceiling are capped, not granted.
        let mut greedy = evidence();
        greedy.requested_boost_db = 20.0;
        let capped = assess_inversion(&greedy, &policy()).unwrap();
        assert_eq!(capped.verdict, InversionVerdict::Allowed);
        assert_eq!(capped.authorized_boost_db, 9.0);
    }

    #[test]
    fn false_classification_never_boosts() {
        // A cancellation mislabeled minimum-phase is still refused when the
        // classifier says so, however confident the rest looks.
        let mut misc = evidence();
        misc.classification = NullClassification::NonMinimumPhase;
        let decision = assess_inversion(&misc, &policy()).unwrap();
        assert_eq!(decision.verdict, InversionVerdict::CutsOnly);
        assert_eq!(decision.authorized_boost_db, 0.0);
        // Uncertainty is not a verified cause either.
        misc.classification = NullClassification::Uncertain;
        let decision = assess_inversion(&misc, &policy()).unwrap();
        assert_eq!(decision.verdict, InversionVerdict::CutsOnly);
    }

    #[test]
    fn weak_or_missing_evidence_fails_closed() {
        // Low classifier confidence denies the boost.
        let mut unsure = evidence();
        unsure.classification_confidence = 0.4;
        let decision = assess_inversion(&unsure, &policy()).unwrap();
        assert_eq!(decision.verdict, InversionVerdict::CutsOnly);
        // Noisy measurement (low depth scale) denies the boost.
        let mut noisy = evidence();
        noisy.measurement_depth_scale = 0.2;
        let decision = assess_inversion(&noisy, &policy()).unwrap();
        assert_eq!(decision.verdict, InversionVerdict::CutsOnly);
        // Thin support denies the boost.
        let mut thin = evidence();
        thin.supporting_bins = 3;
        thin.supporting_seats = 1;
        let decision = assess_inversion(&thin, &policy()).unwrap();
        assert_eq!(decision.verdict, InversionVerdict::CutsOnly);
        // Absent evidence refuses outright instead of guessing.
        let mut absent = evidence();
        absent.supporting_bins = 0;
        assert_eq!(
            assess_inversion(&absent, &policy()).unwrap().verdict,
            InversionVerdict::Refuse
        );
        absent.supporting_bins = 24;
        absent.supporting_seats = 0;
        assert_eq!(
            assess_inversion(&absent, &policy()).unwrap().verdict,
            InversionVerdict::Refuse
        );
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        let mut bad = evidence();
        bad.classification_confidence = f64::NAN;
        assert!(assess_inversion(&bad, &policy()).is_err());
        bad = evidence();
        bad.measurement_depth_scale = 1.5;
        assert!(assess_inversion(&bad, &policy()).is_err());
        bad = evidence();
        bad.requested_boost_db = -1.0;
        assert!(assess_inversion(&bad, &policy()).is_err());
        let mut policy_bad = policy();
        policy_bad.min_confidence = 2.0;
        assert!(assess_inversion(&evidence(), &policy_bad).is_err());
    }
}
