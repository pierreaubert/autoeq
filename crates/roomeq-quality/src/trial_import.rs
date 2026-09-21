//! Real trial result import and per-claim conclusions.
//!
//! Validates imported listener rows against their preregistered protocol:
//! trial ids, per-condition counts, the randomization mapping and the
//! protocol hash must all match, and a post-hoc intent change fails
//! closed. Effects use Wilson 95% intervals; the protocol's exact binomial
//! test is cross-checked against an independent integer-combinatorics
//! calculation before any claim is scored.
//!
//! Each declared claim ends as success, failure or inconclusive.
//! Underpowered or missing trials stay inconclusive and never promote a
//! listening claim. Synthetic tables exercise these code paths only: they
//! stay labelled synthetic and [`qualifies_for_listening_claim`] rejects
//! them, so no real-outcome claim can rest on synthetic rows. Real data
//! collection itself is operator work outside this crate.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::listening::ChainStimulusBinding;
use super::protocol::{BlindedProtocol, ComparisonIntent, DecisionRule, abx_p_value};

/// One imported listener trial row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ImportedTrial {
    /// Stable trial identity; duplicates fail closed.
    pub trial_id: String,
    /// Preregistered condition identity.
    pub condition: String,
    /// Presentation position from the preregistered randomization mapping.
    pub presentation_order: u32,
    /// Whether the response was correct.
    pub correct: bool,
}

/// An imported trial table with its provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TrialImport {
    /// Preregistration hash of the protocol the trials ran under.
    pub protocol_hash: String,
    /// Comparison name the trials ran under.
    pub comparison: String,
    /// Intent the importer claims the trials ran under, when stated.
    /// A stated intent differing from the protocol is a post-hoc change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claimed_intent: Option<ComparisonIntent>,
    /// Chain/stimulus identities the trials ran under.
    pub binding: ChainStimulusBinding,
    /// True for synthetic code-path tables. Synthetic tables never
    /// support a listening claim.
    #[serde(default)]
    pub synthetic: bool,
    /// Trial rows.
    pub rows: Vec<ImportedTrial>,
}

/// One validated condition cell: exact trial count with correct count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ValidatedCondition {
    /// Preregistered condition identity.
    pub condition: String,
    /// Trials run (always equals the preregistered count).
    pub trials: u32,
    /// Correct responses.
    pub correct: u32,
}

/// Import validated against its protocol, ready for scoring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ValidatedImport {
    /// Preregistration hash validated.
    pub protocol_hash: String,
    /// Comparison name validated.
    pub comparison: String,
    /// Synthetic provenance carried through to every verdict.
    pub synthetic: bool,
    /// Per-condition cells in protocol condition order.
    pub conditions: Vec<ValidatedCondition>,
}

/// Validate trial ids, per-condition counts, the randomization mapping,
/// the protocol hash and the chain/stimulus binding. Missing mandatory
/// conditions, duplicates, count drift, a non-permutation order mapping
/// or a post-hoc intent change all fail closed.
pub fn validate_trial_import(
    import: &TrialImport,
    protocol: &BlindedProtocol,
    expected_binding: &ChainStimulusBinding,
) -> Result<ValidatedImport, String> {
    protocol.verify_prereg()?;
    if import.protocol_hash != protocol.prereg_hash {
        return Err(String::from(
            "trial import carries a different preregistration hash than the protocol",
        ));
    }
    if import.comparison != protocol.comparison.name {
        return Err(String::from(
            "trial import names a different comparison than the protocol",
        ));
    }
    if import
        .claimed_intent
        .is_some_and(|claimed| claimed != protocol.comparison.intent)
    {
        return Err(String::from(
            "post-hoc intent change: import claims a different intent than the preregistered protocol",
        ));
    }
    expected_binding.verify_unchanged(&import.binding)?;
    if import.rows.is_empty() {
        return Err(String::from("trial import has no rows"));
    }
    let mut seen = std::collections::HashSet::new();
    for row in &import.rows {
        if row.trial_id.trim().is_empty() {
            return Err(String::from("trial rows need non-blank trial ids"));
        }
        if !seen.insert(row.trial_id.clone()) {
            return Err(format!("duplicate trial id {}", row.trial_id));
        }
        if !protocol
            .conditions
            .iter()
            .any(|known| known == &row.condition)
        {
            return Err(format!(
                "trial {} names condition {} outside the preregistered protocol",
                row.trial_id, row.condition
            ));
        }
    }
    let mut cells = Vec::with_capacity(protocol.conditions.len());
    for condition in &protocol.conditions {
        let mut rows: Vec<&ImportedTrial> = import
            .rows
            .iter()
            .filter(|row| &row.condition == condition)
            .collect();
        if rows.len() as u32 != protocol.trials_per_condition {
            return Err(format!(
                "condition {condition}: imported {} trials under a protocol fixed for {}",
                rows.len(),
                protocol.trials_per_condition
            ));
        }
        rows.sort_by_key(|row| row.presentation_order);
        for (expected, row) in rows.iter().enumerate() {
            if row.presentation_order != expected as u32 {
                return Err(format!(
                    "condition {condition}: presentation orders are not the preregistered randomization mapping"
                ));
            }
        }
        cells.push(ValidatedCondition {
            condition: condition.clone(),
            trials: protocol.trials_per_condition,
            correct: rows.iter().filter(|row| row.correct).count() as u32,
        });
    }
    Ok(ValidatedImport {
        protocol_hash: import.protocol_hash.clone(),
        comparison: import.comparison.clone(),
        synthetic: import.synthetic,
        conditions: cells,
    })
}

/// Wilson score 95% interval for `correct` successes in `trials` trials.
/// Empty at the boundaries only when the data warrants it.
pub fn wilson_ci95(correct: u32, trials: u32) -> Result<[f64; 2], String> {
    if trials == 0 {
        return Err(String::from("effect interval needs at least one trial"));
    }
    if correct > trials {
        return Err(String::from("correct cannot exceed trials"));
    }
    // 95% two-sided normal quantile; fixed by the stated level, not tuned.
    const Z: f64 = 1.959_963_984_540_054;
    let n = trials as f64;
    let p = correct as f64 / n;
    let denominator = 1.0 + Z * Z / n;
    let center = (p + Z * Z / (2.0 * n)) / denominator;
    let half = Z * (p * (1.0 - p) / n + Z * Z / (4.0 * n * n)).sqrt() / denominator;
    Ok([
        (center - half).clamp(0.0, 1.0),
        (center + half).clamp(0.0, 1.0),
    ])
}

/// Exact one-sided binomial upper-tail P(X >= k) at p = 0.5 from integer
/// combinatorics, independent of the mode-anchored floating recurrence in
/// `protocol::abx_p_value`. `None` when coefficients would overflow u128.
fn binomial_tail_exact(k_correct: u32, n_trials: u32) -> Option<f64> {
    if k_correct > n_trials {
        return None;
    }
    if k_correct == 0 {
        return Some(1.0);
    }
    let n = n_trials as u128;
    // C(n, k) by multiplicative recurrence with exact division at each
    // step; checked arithmetic bails out instead of wrapping.
    let choose = |k: u128| -> Option<u128> {
        let k = k.min(n - k);
        let mut value: u128 = 1;
        for i in 1..=k {
            value = value.checked_mul(n - k + i)?.checked_div(i)?;
        }
        Some(value)
    };
    let mut numerator: u128 = 0;
    for k in k_correct as u128..=n {
        numerator = numerator.checked_add(choose(k)?)?;
    }
    let denominator = 1u128.checked_shl(n_trials)?;
    Some(numerator as f64 / denominator as f64)
}

/// Per-claim conclusion for one condition cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ClaimVerdict {
    /// Preregistered condition identity.
    pub condition: String,
    /// Trials run.
    pub trials: u32,
    /// Correct responses.
    pub correct: u32,
    /// Observed correct-response rate.
    pub effect_rate: f64,
    /// Wilson 95% interval for the rate.
    pub ci95: [f64; 2],
    /// Exact protocol p value.
    pub p_value: f64,
    /// Independent integer-combinatorics cross-check (`None` when the
    /// trial count exceeds exact u128 range; the protocol rule stays exact).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_p_value: Option<f64>,
    /// `success`, `failure` or `inconclusive` under the preregistered rule.
    pub decision: String,
    /// Synthetic provenance: synthetic verdicts never support a claim.
    pub synthetic: bool,
}

/// Score every validated cell under the preregistered decision rule.
///
/// The exact p value must agree with the independent calculation within
/// 1e-9 whenever the independent range covers the trial count; a
/// disagreement fails the whole summary rather than one cell. Cells that
/// miss significance but whose interval still covers the powered minimum
/// effect are inconclusive (underpowered negatives), not failures.
pub fn summarize_claims(
    validated: &ValidatedImport,
    protocol: &BlindedProtocol,
) -> Result<Vec<ClaimVerdict>, String> {
    protocol.verify_prereg()?;
    if validated.protocol_hash != protocol.prereg_hash {
        return Err(String::from(
            "validated import carries a different preregistration hash than the protocol",
        ));
    }
    let (min_correct, rule_trials, alpha) = match &protocol.decision {
        DecisionRule::Abx {
            min_correct,
            trials,
            alpha,
        } => (*min_correct, *trials, *alpha),
        DecisionRule::Mushra { .. } => {
            return Err(String::from(
                "MUSHRA free-text criteria need adjudication against the preregistered text: no automatic claim scoring",
            ));
        }
    };
    if protocol.trials_per_condition != rule_trials {
        return Err(String::from(
            "protocol trial count and embedded rule disagree: re-preregister",
        ));
    }
    if super::protocol::abx_p_value(min_correct, rule_trials)? > alpha {
        return Err(String::from(
            "preregistered rule cannot attain its own alpha: no valid significance decision exists",
        ));
    }
    let mut verdicts = Vec::with_capacity(validated.conditions.len());
    for cell in &validated.conditions {
        if cell.trials != rule_trials {
            return Err(format!(
                "condition {}: {} trials under a rule fixed for {rule_trials}",
                cell.condition, cell.trials
            ));
        }
        let p_value = abx_p_value(cell.correct, cell.trials)?;
        let reference_p_value = binomial_tail_exact(cell.correct, cell.trials);
        if let Some(reference) = reference_p_value
            && (p_value - reference).abs() > 1e-9
        {
            return Err(format!(
                "condition {}: protocol p value {p_value} disagrees with independent calculation {reference}",
                cell.condition
            ));
        }
        let ci95 = wilson_ci95(cell.correct, cell.trials)?;
        let decision = if cell.correct >= min_correct {
            String::from("success")
        } else if ci95[1] >= protocol.minimum_effect_p_correct {
            // Not significant, yet the powered minimum effect is still
            // inside the interval: the data cannot rule the effect out,
            // so the negative is underpowered, not a failure.
            String::from("inconclusive")
        } else {
            String::from("failure")
        };
        verdicts.push(ClaimVerdict {
            condition: cell.condition.clone(),
            trials: cell.trials,
            correct: cell.correct,
            effect_rate: cell.correct as f64 / cell.trials as f64,
            ci95,
            p_value,
            reference_p_value,
            decision,
            synthetic: validated.synthetic,
        });
    }
    Ok(verdicts)
}

/// Gate for real listening claims: every verdict must be real (not
/// synthetic) and a success. Inconclusive or failed cells, missing
/// bindings and synthetic tables all fail closed here while remaining
/// available as software-path evidence in their own verdicts.
pub fn qualifies_for_listening_claim(verdicts: &[ClaimVerdict]) -> Result<(), String> {
    if verdicts.is_empty() {
        return Err(String::from("no validated claims to qualify"));
    }
    for verdict in verdicts {
        if verdict.synthetic {
            return Err(format!(
                "condition {} ran on a synthetic table: synthetic verdicts never support a listening claim",
                verdict.condition
            ));
        }
        if verdict.decision != "success" {
            return Err(format!(
                "condition {} is {}: only successes meet the preregistered criterion",
                verdict.condition, verdict.decision
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod trial_import_tests {
    use super::super::listening::{
        ChainStimulusBinding, ListeningArm, ListeningCondition, SourcePresentation,
    };
    use super::super::protocol::{
        BlindedProtocol, ComparisonDesign, ComparisonIntent, ComparisonSpec, DecisionRule,
        ReferenceKind,
    };
    use super::*;

    fn conditions() -> Vec<ListeningCondition> {
        vec![
            ListeningCondition {
                arm: ListeningArm::BaselineVsCorrected,
                presentation: SourcePresentation::SingleSpeakerMono,
                programme_id: String::from("resonance-strings-01"),
                seat_id: String::from("seat-1"),
            },
            ListeningCondition {
                arm: ListeningArm::PrunedVsFull,
                presentation: SourcePresentation::Spatial,
                programme_id: String::from("transient-drums-02"),
                seat_id: String::from("seat-2"),
            },
        ]
    }

    fn protocol_for(staged: &[ListeningCondition]) -> BlindedProtocol {
        BlindedProtocol::preregister(
            ComparisonDesign::Abx,
            ComparisonSpec {
                name: String::from("correction-vs-baseline-listening"),
                intent: ComparisonIntent::Detectability,
                attributes: vec![String::from("detectability")],
                equivalence_bound: None,
                reference: ReferenceKind::IndependentImplementation {
                    description: String::from("second analysis implementation agrees within 1e-12"),
                },
                validated_domain: String::from("unvalidated"),
            },
            staged.iter().map(|c| c.condition_id()).collect(),
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
        .unwrap()
    }

    fn binding() -> ChainStimulusBinding {
        ChainStimulusBinding {
            baseline_graph_id: String::from("graph-baseline-immutable"),
            candidate_graph_id: String::from("graph-candidate-immutable"),
            full_graph_id: String::from("graph-full-immutable"),
            pruned_graph_id: Some(String::from("graph-pruned-immutable")),
            stimulus_hash: String::from("rendered-stimulus-hash"),
            sample_rate_hz: 48_000.0,
            calibration_id: String::from("spl-cal-94db"),
            processing_state: String::from("final-delivered"),
        }
    }

    fn rows_for(protocol: &BlindedProtocol, correct_each: u32, synthetic: bool) -> TrialImport {
        let mut rows = Vec::new();
        for condition in &protocol.conditions {
            for order in 0..protocol.trials_per_condition {
                rows.push(ImportedTrial {
                    trial_id: format!("{condition}-t{order:02}"),
                    condition: condition.clone(),
                    presentation_order: order,
                    correct: order < correct_each,
                });
            }
        }
        TrialImport {
            protocol_hash: protocol.prereg_hash.clone(),
            comparison: protocol.comparison.name.clone(),
            claimed_intent: Some(ComparisonIntent::Detectability),
            binding: binding(),
            synthetic,
            rows,
        }
    }

    #[test]
    fn trial_import_rejects_duplicates_hash_mismatch_and_count_drift() {
        let staged = conditions();
        let protocol = protocol_for(&staged);
        // Clean import validates with exact per-condition counts.
        let validated =
            validate_trial_import(&rows_for(&protocol, 20, true), &protocol, &binding()).unwrap();
        assert_eq!(validated.conditions.len(), 2);
        assert!(validated.conditions.iter().all(|cell| cell.trials == 30));

        // Duplicate trial ids fail closed.
        let mut import = rows_for(&protocol, 20, true);
        import.rows[1].trial_id = import.rows[0].trial_id.clone();
        assert!(validate_trial_import(&import, &protocol, &binding()).is_err());

        // Wrong protocol hash fails closed.
        let mut import = rows_for(&protocol, 20, true);
        import.protocol_hash = String::from("deadbeef");
        assert!(validate_trial_import(&import, &protocol, &binding()).is_err());

        // Missing rows (count drift) fail closed.
        let mut import = rows_for(&protocol, 20, true);
        import.rows.pop();
        assert!(validate_trial_import(&import, &protocol, &binding()).is_err());

        // Unknown condition fails closed.
        let mut import = rows_for(&protocol, 20, true);
        import.rows[0].condition = String::from("unregistered-condition");
        assert!(validate_trial_import(&import, &protocol, &binding()).is_err());

        // Scrambled randomization mapping fails closed.
        let mut import = rows_for(&protocol, 20, true);
        import.rows[0].presentation_order = 7;
        assert!(validate_trial_import(&import, &protocol, &binding()).is_err());

        // Post-hoc intent change fails closed.
        let mut import = rows_for(&protocol, 20, true);
        import.claimed_intent = Some(ComparisonIntent::Equivalence);
        assert!(validate_trial_import(&import, &protocol, &binding()).is_err());

        // Rebound chain fails closed.
        let mut import = rows_for(&protocol, 20, true);
        import.binding.candidate_graph_id = String::from("graph-candidate-retuned");
        assert!(validate_trial_import(&import, &protocol, &binding()).is_err());
    }

    #[test]
    fn trial_import_exact_binomial_agrees_with_protocol_scoring() {
        // The independent integer-combinatorics tail agrees with the
        // protocol's floating recurrence at textbook and extreme points.
        for (k, n) in [(20, 30), (30, 30), (0, 30), (15, 30), (25, 40)] {
            let exact = binomial_tail_exact(k, n).unwrap();
            let protocol = abx_p_value(k, n).unwrap();
            assert!(
                (exact - protocol).abs() <= 1e-9,
                "k={k} n={n}: exact={exact} protocol={protocol}"
            );
        }
        // A known 20/30 table scores success with matching p values and a
        // Wilson interval covering the observed rate.
        let staged = conditions();
        let protocol = protocol_for(&staged);
        let validated =
            validate_trial_import(&rows_for(&protocol, 20, true), &protocol, &binding()).unwrap();
        let verdicts = summarize_claims(&validated, &protocol).unwrap();
        assert_eq!(verdicts.len(), 2);
        for verdict in &verdicts {
            assert_eq!(verdict.decision, "success");
            assert!(verdict.reference_p_value.is_some());
            assert!(verdict.p_value < 0.05);
            assert!(
                verdict.ci95[0] <= verdict.effect_rate && verdict.effect_rate <= verdict.ci95[1]
            );
        }
        // An 18/30 negative misses the 20/30 rule yet its Wilson interval
        // still covers the powered 0.75 effect: inconclusive (underpowered
        // negative), not failure. 5/30 rules the effect out: failure.
        let validated =
            validate_trial_import(&rows_for(&protocol, 18, true), &protocol, &binding()).unwrap();
        let verdicts = summarize_claims(&validated, &protocol).unwrap();
        assert!(verdicts.iter().all(|v| v.decision == "inconclusive"));
        let validated =
            validate_trial_import(&rows_for(&protocol, 5, true), &protocol, &binding()).unwrap();
        let verdicts = summarize_claims(&validated, &protocol).unwrap();
        assert!(verdicts.iter().all(|v| v.decision == "failure"));
    }

    #[test]
    fn trial_import_synthetic_cannot_support_listening_claim() {
        let staged = conditions();
        let protocol = protocol_for(&staged);
        // Synthetic successes score as success but never qualify.
        let validated =
            validate_trial_import(&rows_for(&protocol, 20, true), &protocol, &binding()).unwrap();
        let verdicts = summarize_claims(&validated, &protocol).unwrap();
        assert!(verdicts.iter().all(|v| v.synthetic));
        assert!(qualifies_for_listening_claim(&verdicts).is_err());
        // Real successes with the same counts qualify structurally; real
        // data collection itself remains operator work.
        let validated =
            validate_trial_import(&rows_for(&protocol, 20, false), &protocol, &binding()).unwrap();
        let verdicts = summarize_claims(&validated, &protocol).unwrap();
        assert!(verdicts.iter().all(|v| !v.synthetic));
        assert!(qualifies_for_listening_claim(&verdicts).is_ok());
        // A real inconclusive table does not qualify either.
        let validated =
            validate_trial_import(&rows_for(&protocol, 18, false), &protocol, &binding()).unwrap();
        let verdicts = summarize_claims(&validated, &protocol).unwrap();
        assert!(qualifies_for_listening_claim(&verdicts).is_err());
    }
}
