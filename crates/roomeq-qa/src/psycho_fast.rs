//! Fast psychoacoustic-principle coverage, section F (release gates).
//!
//! Production entry points under test:
//! - [`crate::release_gates::assess_perceptual_gate`] and
//!   [`crate::release_gates::assess_listening_gate`] (provenance-gated
//!   promotion: proxies, synthetic trials, and inconclusive outcomes never
//!   promote).
//! - [`roomeq_quality::qualifies_for_listening_claim`] (claim witness W5:
//!   only real successful verdicts qualify).

use roomeq_quality::{ClaimVerdict, qualifies_for_listening_claim};

use crate::release_gates::{
    EvidenceProvenance, TrialOutcome, assess_listening_gate, assess_perceptual_gate,
};

fn independent_reference() -> EvidenceProvenance {
    EvidenceProvenance::IndependentReference {
        reference_id: "iso532-1-pinned-v1".to_string(),
        calibration_domain: "free-field-48k".to_string(),
        tolerance: "reference-defined".to_string(),
    }
}

fn real_benefit() -> EvidenceProvenance {
    EvidenceProvenance::RealTrial {
        protocol_hash: "prereg-abc123".to_string(),
        sufficient: true,
        outcome: TrialOutcome::Benefit,
    }
}

/// F04: proxy metrics, synthetic trials, and inconclusive outcomes never
/// promote perceptual or listening claims. Only a pinned independent
/// reference validates a model, and only sufficient real trials with a
/// demonstrated outcome demonstrate benefit.
#[test]
fn psycho_fast_f04_proxy_and_synthetic_never_promote() {
    let proxy = EvidenceProvenance::ExperimentalProxy {
        metric_id: "heuristic-erb-proxy".to_string(),
    };
    assert!(!assess_perceptual_gate(&proxy).passed);
    assert!(!assess_listening_gate(&proxy, false).passed);

    let synthetic = EvidenceProvenance::SyntheticTrial { seed: 7, cases: 64 };
    assert!(!assess_perceptual_gate(&synthetic).passed);
    assert!(!assess_listening_gate(&synthetic, false).passed);

    let inconclusive = EvidenceProvenance::RealTrial {
        protocol_hash: "prereg-abc123".to_string(),
        sufficient: true,
        outcome: TrialOutcome::Inconclusive,
    };
    assert!(!assess_listening_gate(&inconclusive, false).passed);

    let underpowered = EvidenceProvenance::RealTrial {
        protocol_hash: "prereg-abc123".to_string(),
        sufficient: false,
        outcome: TrialOutcome::Benefit,
    };
    assert!(!assess_listening_gate(&underpowered, false).passed);

    assert!(assess_perceptual_gate(&independent_reference()).passed);
    assert!(assess_listening_gate(&real_benefit(), false).passed);
}

/// F04/E04: unsupported domains fail closed on every field. An empty
/// reference, an empty calibration domain, a missing tolerance, and a
/// blank protocol hash each independently withhold promotion.
#[test]
fn psycho_fast_f04_empty_provenance_fields_fail_closed() {
    for provenance in [
        EvidenceProvenance::IndependentReference {
            reference_id: String::new(),
            calibration_domain: "free-field-48k".to_string(),
            tolerance: "reference-defined".to_string(),
        },
        EvidenceProvenance::IndependentReference {
            reference_id: "iso532-1-pinned-v1".to_string(),
            calibration_domain: String::new(),
            tolerance: "reference-defined".to_string(),
        },
        EvidenceProvenance::IndependentReference {
            reference_id: "iso532-1-pinned-v1".to_string(),
            calibration_domain: "free-field-48k".to_string(),
            tolerance: String::new(),
        },
    ] {
        assert!(
            !assess_perceptual_gate(&provenance).passed,
            "empty provenance field must fail closed: {provenance:?}"
        );
    }
    let unbound = EvidenceProvenance::RealTrial {
        protocol_hash: String::new(),
        sufficient: true,
        outcome: TrialOutcome::Benefit,
    };
    assert!(!assess_listening_gate(&unbound, false).passed);
}

/// F03/F04: a benefit outcome never satisfies an equivalence claim, while
/// an equivalence outcome never satisfies a benefit claim. Preference-only
/// data arrives as `Inconclusive` and promotes nothing.
#[test]
fn psycho_fast_f04_intent_mismatch_never_promotes() {
    assert!(!assess_listening_gate(&real_benefit(), true).passed);
    let equivalence = EvidenceProvenance::RealTrial {
        protocol_hash: "prereg-abc123".to_string(),
        sufficient: true,
        outcome: TrialOutcome::Equivalence,
    };
    assert!(assess_listening_gate(&equivalence, true).passed);
    assert!(!assess_listening_gate(&equivalence, false).passed);
}

fn verdict(condition: &str, synthetic: bool, decision: &str) -> ClaimVerdict {
    ClaimVerdict {
        condition: condition.to_string(),
        trials: 10,
        correct: 10,
        effect_rate: 1.0,
        ci95: [0.7, 1.0],
        p_value: 1.0 / 1_024.0,
        reference_p_value: Some(1.0 / 1_024.0),
        decision: decision.to_string(),
        synthetic,
    }
}

/// W5 (claim-promotion witness): the production run's verdicts pass through
/// the real claim gate. Synthetic tables never qualify even with perfect
/// scores; inconclusive cells never qualify; only real successes do.
#[test]
fn psycho_fast_w5_claim_promotion_witness() {
    assert!(
        qualifies_for_listening_claim(&[verdict("mono", true, "success")]).is_err(),
        "synthetic verdicts never support a listening claim"
    );
    assert!(
        qualifies_for_listening_claim(&[verdict("mono", false, "inconclusive")]).is_err(),
        "inconclusive verdicts never support a listening claim"
    );
    assert!(
        qualifies_for_listening_claim(&[
            verdict("mono", false, "success"),
            verdict("stereo", false, "inconclusive"),
        ])
        .is_err(),
        "one inconclusive cell blocks the whole claim"
    );
    qualifies_for_listening_claim(&[
        verdict("mono", false, "success"),
        verdict("stereo", false, "success"),
    ])
    .expect("real successes qualify");
}
