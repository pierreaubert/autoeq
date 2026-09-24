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

use super::listening::{ChainStimulusBinding, ListeningSetup};
use super::protocol::{
    AbxAnswer, BlindedProtocol, ComparisonIntent, DecisionRule, abx_p_value, binomial_lower_tail,
};

/// One imported listener trial row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ImportedTrial {
    /// Stable trial identity; duplicates fail closed.
    pub trial_id: String,
    /// Pseudonymous participant identity from the frozen setup allocation.
    #[serde(default)]
    pub participant_id: String,
    /// Preregistered condition identity.
    pub condition: String,
    /// Presentation position from the preregistered randomization mapping.
    pub presentation_order: u32,
    /// Whether the response was correct.
    pub correct: bool,
    /// Recorded X response for real ABX rows. Correctness is recomputed
    /// against the protocol's frozen answer key; synthetic rows may omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<AbxAnswer>,
}

/// An imported trial table with its provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TrialImport {
    /// Preregistration hash of the protocol the trials ran under.
    pub protocol_hash: String,
    /// Hash of the complete frozen setup. A bare protocol hash omits the
    /// listener population, level, programme scope and chain binding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup_hash: Option<String>,
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
    /// Present only after validation against the frozen setup itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup_hash: Option<String>,
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
    if !import.synthetic && protocol.trial_assignments.is_empty() {
        return Err(String::from(
            "real trial import needs a preregistered answer key and presentation mapping",
        ));
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
        if !import.synthetic {
            if row.participant_id.trim().is_empty() {
                return Err(format!(
                    "real trial {} needs a participant ID",
                    row.trial_id
                ));
            }
            let assignment = protocol
                .trial_assignments
                .iter()
                .find(|assignment| assignment.trial_id == row.trial_id)
                .ok_or_else(|| {
                    format!(
                        "trial {} is absent from the preregistered mapping",
                        row.trial_id
                    )
                })?;
            if assignment.condition != row.condition
                || assignment.presentation_order != row.presentation_order
            {
                return Err(format!(
                    "trial {} differs from its preregistered condition or presentation order",
                    row.trial_id
                ));
            }
            let response = row.response.ok_or_else(|| {
                format!(
                    "real trial {} needs its recorded ABX response",
                    row.trial_id
                )
            })?;
            if row.correct != (response == assignment.answer) {
                return Err(format!(
                    "trial {} correctness disagrees with its preregistered answer key",
                    row.trial_id
                ));
            }
        }
    }
    let mut cells = Vec::with_capacity(protocol.conditions.len());
    let mut incomplete = Vec::new();
    for condition in &protocol.conditions {
        let mut rows: Vec<&ImportedTrial> = import
            .rows
            .iter()
            .filter(|row| &row.condition == condition)
            .collect();
        if rows.len() as u32 != protocol.trials_per_condition {
            let mut detail = format!(
                "condition {condition}: imported {} trials under a protocol fixed for {}",
                rows.len(),
                protocol.trials_per_condition
            );
            if !import.synthetic {
                let present: std::collections::HashSet<_> =
                    rows.iter().map(|row| row.trial_id.as_str()).collect();
                let missing: Vec<_> = protocol
                    .trial_assignments
                    .iter()
                    .filter(|assignment| assignment.condition == *condition)
                    .filter(|assignment| !present.contains(assignment.trial_id.as_str()))
                    .map(|assignment| assignment.trial_id.as_str())
                    .collect();
                if !missing.is_empty() {
                    let preview = missing
                        .iter()
                        .take(8)
                        .copied()
                        .collect::<Vec<_>>()
                        .join(", ");
                    detail.push_str(&format!(
                        "; missing {} preregistered trial ids: {preview}{}",
                        missing.len(),
                        if missing.len() > 8 { ", ..." } else { "" }
                    ));
                }
            }
            incomplete.push(detail);
            continue;
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
    if !incomplete.is_empty() {
        return Err(format!(
            "incomplete trial import: {}",
            incomplete.join("; ")
        ));
    }
    Ok(ValidatedImport {
        protocol_hash: import.protocol_hash.clone(),
        setup_hash: None,
        comparison: import.comparison.clone(),
        synthetic: import.synthetic,
        conditions: cells,
    })
}

/// Validate against the whole preregistered setup, not only its protocol and
/// a separately supplied binding. Real claim promotion must use this path.
pub fn validate_trial_import_with_setup(
    import: &TrialImport,
    setup: &ListeningSetup,
) -> Result<ValidatedImport, String> {
    setup.validate()?;
    if import.setup_hash.as_deref() != Some(setup.setup_hash.as_str()) {
        return Err(String::from(
            "trial import does not match the frozen listening setup hash",
        ));
    }
    if !import.synthetic {
        for row in &import.rows {
            let allocation = setup
                .trial_participants
                .iter()
                .find(|allocation| allocation.trial_id == row.trial_id)
                .ok_or_else(|| {
                    format!(
                        "trial {} has no frozen participant allocation",
                        row.trial_id
                    )
                })?;
            if row.participant_id != allocation.participant_id {
                return Err(format!(
                    "trial {} participant differs from the frozen allocation",
                    row.trial_id
                ));
            }
        }
    }
    let mut validated = validate_trial_import(import, &setup.protocol, &setup.binding)?;
    validated.setup_hash = Some(setup.setup_hash.clone());
    Ok(validated)
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

/// Independent direct-coefficient reference for the lower tail. The u128
/// range intentionally limits this cross-check; production scoring uses the
/// log-space recurrence for larger preregistered trials.
fn binomial_lower_reference(k: u32, n: u32, p: f64) -> Option<f64> {
    if n > 100 || k > n {
        return None;
    }
    let mut choose = 1u128;
    let mut sum = 0.0;
    for i in 0..=k {
        sum += choose as f64 * p.powi(i as i32) * (1.0 - p).powi((n - i) as i32);
        if i < k {
            choose = choose
                .checked_mul(u128::from(n - i))?
                .checked_div(u128::from(i + 1))?;
        }
    }
    Some(sum)
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
    /// Descriptive Wilson 95% interval for the rate. Exact preregistered
    /// binomial tails decide detectability and equivalence.
    pub ci95: [f64; 2],
    /// Exact protocol p value.
    pub p_value: f64,
    /// Independent integer-combinatorics cross-check (`None` when the
    /// trial count exceeds exact u128 range; the protocol rule stays exact).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_p_value: Option<f64>,
    /// Frozen setup hash checked by the setup-aware importer. Absence means
    /// the claim has only protocol-level software validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup_hash: Option<String>,
    /// Intent of the preregistered rule that produced this verdict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<ComparisonIntent>,
    /// Numeric maximum correct-response rate for equivalence, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub equivalence_bound_p_correct: Option<f64>,
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
    if validated.comparison != protocol.comparison.name {
        return Err(String::from(
            "validated import names a different comparison than the protocol",
        ));
    }
    if validated.conditions.len() != protocol.conditions.len()
        || validated
            .conditions
            .iter()
            .zip(&protocol.conditions)
            .any(|(cell, condition)| &cell.condition != condition)
    {
        return Err(String::from(
            "validated import must contain every preregistered condition exactly once in protocol order",
        ));
    }
    if validated
        .conditions
        .iter()
        .any(|cell| cell.correct > cell.trials)
    {
        return Err(String::from(
            "validated import has more correct responses than trials",
        ));
    }
    let rule_trials = match (&protocol.comparison.intent, &protocol.decision) {
        (ComparisonIntent::Detectability, DecisionRule::Abx { trials, .. })
        | (ComparisonIntent::Equivalence, DecisionRule::AbxEquivalence { trials, .. }) => *trials,
        (ComparisonIntent::Preference, _) => {
            return Err(String::from(
                "ABX correct-response counts cannot score preference; a preregistered choice-based rule is required",
            ));
        }
        (_, DecisionRule::Mushra { .. }) => {
            return Err(String::from(
                "MUSHRA free-text criteria need adjudication against the preregistered text: no automatic claim scoring",
            ));
        }
        _ => return Err(String::from("ABX decision rule and intent disagree")),
    };
    if protocol.trials_per_condition != rule_trials {
        return Err(String::from(
            "protocol trial count and embedded rule disagree: re-preregister",
        ));
    }
    match &protocol.decision {
        DecisionRule::Abx {
            min_correct, alpha, ..
        } if abx_p_value(*min_correct, rule_trials)? > *alpha => {
            return Err(String::from(
                "preregistered detection rule cannot attain alpha",
            ));
        }
        DecisionRule::AbxEquivalence {
            max_correct,
            alpha,
            max_p_correct,
            ..
        } if binomial_lower_tail(*max_correct, rule_trials, *max_p_correct)? > *alpha => {
            return Err(String::from(
                "preregistered equivalence rule cannot attain alpha",
            ));
        }
        _ => {}
    }
    let mut verdicts = Vec::with_capacity(validated.conditions.len());
    for cell in &validated.conditions {
        if cell.trials != rule_trials {
            return Err(format!(
                "condition {}: {} trials under a rule fixed for {rule_trials}",
                cell.condition, cell.trials
            ));
        }
        let (p_value, reference_p_value, decision, equivalence_bound_p_correct) =
            match &protocol.decision {
                DecisionRule::Abx {
                    min_correct,
                    alpha: _,
                    ..
                } => {
                    let p_value = abx_p_value(cell.correct, cell.trials)?;
                    let ci95 = wilson_ci95(cell.correct, cell.trials)?;
                    let decision = if cell.correct >= *min_correct {
                        "success"
                    } else if ci95[1] >= protocol.minimum_effect_p_correct {
                        "inconclusive"
                    } else {
                        "failure"
                    };
                    (
                        p_value,
                        binomial_tail_exact(cell.correct, cell.trials),
                        decision,
                        None,
                    )
                }
                DecisionRule::AbxEquivalence {
                    max_correct,
                    alpha,
                    max_p_correct,
                    ..
                } => {
                    let p_value = binomial_lower_tail(cell.correct, cell.trials, *max_p_correct)?;
                    let opposing = binomial_lower_tail(
                        cell.trials - cell.correct,
                        cell.trials,
                        1.0 - max_p_correct,
                    )?;
                    let decision = if cell.correct <= *max_correct && p_value <= *alpha {
                        "success"
                    } else if opposing <= *alpha {
                        "failure"
                    } else {
                        "inconclusive"
                    };
                    (
                        p_value,
                        binomial_lower_reference(cell.correct, cell.trials, *max_p_correct),
                        decision,
                        Some(*max_p_correct),
                    )
                }
                DecisionRule::Mushra { .. } => unreachable!("handled above"),
            };
        if let Some(reference) = reference_p_value
            && (p_value - reference).abs() > 1e-9
        {
            return Err(format!(
                "condition {}: protocol p value {p_value} disagrees with independent calculation {reference}",
                cell.condition
            ));
        }
        let ci95 = wilson_ci95(cell.correct, cell.trials)?;
        verdicts.push(ClaimVerdict {
            condition: cell.condition.clone(),
            trials: cell.trials,
            correct: cell.correct,
            effect_rate: cell.correct as f64 / cell.trials as f64,
            ci95,
            p_value,
            reference_p_value,
            setup_hash: validated.setup_hash.clone(),
            intent: Some(protocol.comparison.intent),
            equivalence_bound_p_correct,
            decision: String::from(decision),
            synthetic: validated.synthetic,
        });
    }
    Ok(verdicts)
}

/// Structural gate for listening claims: every verdict must come from a
/// setup-bound import, be marked nonsynthetic and succeed. This does not
/// authenticate the operator or prove that listeners completed the trials;
/// those external facts must be checked before release promotion.
pub fn qualifies_for_listening_claim(verdicts: &[ClaimVerdict]) -> Result<(), String> {
    if verdicts.is_empty() {
        return Err(String::from("no validated claims to qualify"));
    }
    let setup_hash = verdicts[0]
        .setup_hash
        .as_deref()
        .filter(|hash| !hash.trim().is_empty())
        .ok_or_else(|| String::from("listening claims need a frozen setup-bound import"))?;
    for verdict in verdicts {
        if verdict.setup_hash.as_deref() != Some(setup_hash) {
            return Err(String::from(
                "listening verdicts refer to different frozen setups",
            ));
        }
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
        TrialParticipantAssignment,
    };
    use super::super::protocol::{
        AbxAnswer, BlindedProtocol, ComparisonDesign, ComparisonIntent, ComparisonSpec,
        DecisionRule, ReferenceKind,
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
                equivalence_max_p_correct: None,
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

    fn with_real_assignments(protocol: BlindedProtocol) -> BlindedProtocol {
        let assignments = protocol.abx_assignment_template().unwrap();
        protocol.with_trial_assignments(assignments).unwrap()
    }

    fn setup_for(protocol: BlindedProtocol) -> ListeningSetup {
        let participant_ids = (0..protocol.trials_per_condition)
            .map(|order| format!("participant-{order:04}"))
            .collect();
        let trial_participants = protocol
            .trial_assignments
            .iter()
            .map(|assignment| TrialParticipantAssignment {
                trial_id: assignment.trial_id.clone(),
                participant_id: format!("participant-{:04}", assignment.presentation_order),
            })
            .collect();
        ListeningSetup {
            protocol,
            conditions: conditions(),
            holdout_programmes: vec![String::from("heldout-piano-09")],
            binding: binding(),
            model_id: String::from("paired-auditory-model"),
            model_version: String::from("fixture-v1"),
            listener_population: String::from("trained adult listeners"),
            room_id: String::from("room-a"),
            holdout_rooms: vec![String::from("room-b")],
            holdout_participants: vec![String::from("participant-heldout-1")],
            participant_ids,
            trial_participants,
            absolute_playback_level_db_spl: 75.0,
            level_matching_method: String::from("calibrated programme-integrated level"),
            maximum_level_mismatch_db: 0.2,
            sustained_programmes: vec![String::from("resonance-strings-01")],
            transient_programmes: vec![String::from("transient-drums-02")],
            representative_programmes: vec![String::from("resonance-strings-01")],
            setup_hash: String::new(),
        }
        .freeze()
        .unwrap()
    }

    fn rows_for(protocol: &BlindedProtocol, correct_each: u32, synthetic: bool) -> TrialImport {
        let mut rows = Vec::new();
        for condition in &protocol.conditions {
            for order in 0..protocol.trials_per_condition {
                rows.push(ImportedTrial {
                    trial_id: format!("{condition}-t{order:04}"),
                    participant_id: format!("participant-{order:04}"),
                    condition: condition.clone(),
                    presentation_order: order,
                    correct: order < correct_each,
                    response: (!synthetic).then(|| {
                        let answer = protocol
                            .trial_assignments
                            .iter()
                            .find(|assignment| {
                                assignment.condition == *condition
                                    && assignment.presentation_order == order
                            })
                            .expect("real fixture needs preregistered assignments")
                            .answer;
                        if order < correct_each {
                            answer
                        } else if answer == AbxAnswer::A {
                            AbxAnswer::B
                        } else {
                            AbxAnswer::A
                        }
                    }),
                });
            }
        }
        TrialImport {
            protocol_hash: protocol.prereg_hash.clone(),
            setup_hash: None,
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
    fn real_import_discloses_missing_ids_for_every_condition() {
        let protocol = with_real_assignments(protocol_for(&conditions()));
        let mut import = rows_for(&protocol, 20, false);
        import.rows.retain(|row| row.presentation_order != 0);
        let error = validate_trial_import(&import, &protocol, &binding()).unwrap_err();
        assert!(error.contains("incomplete trial import"), "{error}");
        for condition in &protocol.conditions {
            assert!(error.contains(condition), "{error}");
            assert!(error.contains(&format!("{condition}-t0000")), "{error}");
        }
    }

    #[test]
    fn real_trial_import_requires_frozen_mapping_and_raw_response() {
        let protocol = protocol_for(&conditions());
        let real = with_real_assignments(protocol.clone());
        let import = rows_for(&real, 20, false);
        assert!(validate_trial_import(&import, &real, &binding()).is_ok());

        let mut unregistered = import.clone();
        unregistered.protocol_hash = protocol.prereg_hash.clone();
        assert!(validate_trial_import(&unregistered, &protocol, &binding()).is_err());

        let mut missing_response = import.clone();
        missing_response.rows[0].response = None;
        assert!(validate_trial_import(&missing_response, &real, &binding()).is_err());

        let mut false_correctness = import.clone();
        false_correctness.rows[0].correct = !false_correctness.rows[0].correct;
        assert!(validate_trial_import(&false_correctness, &real, &binding()).is_err());

        let mut wrong_order = import.clone();
        wrong_order.rows.swap(0, 1);
        wrong_order.rows[0].presentation_order = 0;
        wrong_order.rows[1].presentation_order = 1;
        assert!(validate_trial_import(&wrong_order, &real, &binding()).is_err());

        let mut tampered_key = real.clone();
        tampered_key.trial_assignments[0].answer = AbxAnswer::B;
        assert!(tampered_key.verify_prereg().is_err());

        let mut unbalanced = real.trial_assignments.clone();
        for assignment in &mut unbalanced {
            assignment.answer = AbxAnswer::A;
        }
        assert!(real.with_trial_assignments(unbalanced).is_err());

        let real = with_real_assignments(protocol.clone());
        let mut off_seed = real.trial_assignments.clone();
        let a = off_seed
            .iter()
            .position(|assignment| assignment.answer == AbxAnswer::A)
            .unwrap();
        let b = off_seed
            .iter()
            .position(|assignment| {
                assignment.condition == off_seed[a].condition && assignment.answer == AbxAnswer::B
            })
            .unwrap();
        off_seed[a].answer = AbxAnswer::B;
        off_seed[b].answer = AbxAnswer::A;
        assert!(real.with_trial_assignments(off_seed).is_err());
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
    fn trial_import_scoring_rejects_incomplete_or_rebound_cells() {
        let staged = conditions();
        let protocol = protocol_for(&staged);
        let validated =
            validate_trial_import(&rows_for(&protocol, 20, true), &protocol, &binding()).unwrap();

        let mut missing = validated.clone();
        missing.conditions.pop();
        assert!(summarize_claims(&missing, &protocol).is_err());

        let mut repeated = validated.clone();
        repeated.conditions[1] = repeated.conditions[0].clone();
        assert!(summarize_claims(&repeated, &protocol).is_err());

        let mut renamed = validated.clone();
        renamed.comparison = String::from("other-comparison");
        assert!(summarize_claims(&renamed, &protocol).is_err());

        let mut impossible = validated;
        impossible.conditions[0].correct = impossible.conditions[0].trials + 1;
        assert!(summarize_claims(&impossible, &protocol).is_err());
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
        let real_protocol = with_real_assignments(protocol);
        let setup = setup_for(real_protocol.clone());
        let mut real_import = rows_for(&real_protocol, 20, false);
        let bare = validate_trial_import(&real_import, &real_protocol, &binding()).unwrap();
        assert!(
            qualifies_for_listening_claim(&summarize_claims(&bare, &real_protocol).unwrap())
                .is_err()
        );
        real_import.setup_hash = Some(setup.setup_hash.clone());
        let validated = validate_trial_import_with_setup(&real_import, &setup).unwrap();
        let verdicts = summarize_claims(&validated, &real_protocol).unwrap();
        assert!(verdicts.iter().all(|v| !v.synthetic));
        assert!(qualifies_for_listening_claim(&verdicts).is_ok());
        real_import.setup_hash = Some(String::from("wrong-setup"));
        assert!(validate_trial_import_with_setup(&real_import, &setup).is_err());
        // A real inconclusive table does not qualify either.
        let mut inconclusive = rows_for(&real_protocol, 18, false);
        inconclusive.setup_hash = Some(setup.setup_hash.clone());
        let validated = validate_trial_import_with_setup(&inconclusive, &setup).unwrap();
        let verdicts = summarize_claims(&validated, &real_protocol).unwrap();
        assert!(qualifies_for_listening_claim(&verdicts).is_err());
    }

    #[test]
    fn real_trial_participant_must_match_frozen_allocation() {
        let staged = conditions();
        let protocol = with_real_assignments(protocol_for(&staged));
        let setup = setup_for(protocol.clone());
        let mut import = rows_for(&protocol, 20, false);
        import.setup_hash = Some(setup.setup_hash.clone());
        assert!(validate_trial_import_with_setup(&import, &setup).is_ok());

        import.rows[0].participant_id.clear();
        assert!(
            validate_trial_import(&import, &protocol, &binding())
                .unwrap_err()
                .contains("participant ID")
        );

        import.rows[0].participant_id = String::from("participant-0001");
        assert!(
            validate_trial_import_with_setup(&import, &setup)
                .unwrap_err()
                .contains("frozen allocation")
        );

        import.rows[0].participant_id = String::from("participant-heldout-1");
        assert!(validate_trial_import_with_setup(&import, &setup).is_err());

        let mut repeated = setup.clone();
        repeated.trial_participants[0].participant_id =
            repeated.trial_participants[1].participant_id.clone();
        assert!(repeated.freeze().is_err());
    }

    #[test]
    fn abx_detection_rule_cannot_score_equivalence_or_preference() {
        let staged = conditions();
        let base = protocol_for(&staged);
        let mut equivalence = base.comparison.clone();
        equivalence.intent = ComparisonIntent::Equivalence;
        equivalence.equivalence_bound = Some(String::from("ABX correct-response rate < 0.75"));
        equivalence.equivalence_max_p_correct = Some(0.75);
        assert!(
            BlindedProtocol::preregister(
                base.design,
                equivalence,
                base.conditions.clone(),
                base.trials_per_condition,
                base.alpha,
                base.target_power,
                base.minimum_effect_p_correct,
                base.decision.clone(),
                base.randomization_seed,
            )
            .is_err()
        );
        let mut preference = base.comparison.clone();
        preference.intent = ComparisonIntent::Preference;
        preference.attributes = vec![String::from("timbre-preference")];
        assert!(
            BlindedProtocol::preregister(
                base.design,
                preference,
                base.conditions.clone(),
                base.trials_per_condition,
                base.alpha,
                base.target_power,
                base.minimum_effect_p_correct,
                base.decision.clone(),
                base.randomization_seed,
            )
            .is_err()
        );
    }

    #[test]
    fn numeric_equivalence_uses_exact_lower_tail_and_intent_bound_verdict() {
        let base = protocol_for(&conditions());
        let mut comparison = base.comparison.clone();
        comparison.intent = ComparisonIntent::Equivalence;
        comparison.equivalence_bound = Some(String::from("ABX correct-response rate < 0.75"));
        comparison.equivalence_max_p_correct = Some(0.75);
        let protocol = BlindedProtocol::preregister(
            base.design,
            comparison,
            base.conditions.clone(),
            30,
            0.05,
            base.target_power,
            base.minimum_effect_p_correct,
            DecisionRule::AbxEquivalence {
                max_correct: 17,
                trials: 30,
                alpha: 0.05,
                max_p_correct: 0.75,
                alternative_p_correct: 0.5,
            },
            base.randomization_seed,
        )
        .unwrap();
        let mut mismatched_bound = protocol.comparison.clone();
        mismatched_bound.equivalence_max_p_correct = Some(0.7);
        assert!(
            BlindedProtocol::preregister(
                protocol.design,
                mismatched_bound,
                protocol.conditions.clone(),
                30,
                0.05,
                protocol.target_power,
                protocol.minimum_effect_p_correct,
                protocol.decision.clone(),
                protocol.randomization_seed,
            )
            .is_err()
        );
        let protocol = with_real_assignments(protocol);
        let power = binomial_lower_tail(17, 30, 0.5).unwrap();
        assert!((power - 0.819_202_695_973_217_5).abs() < 1e-12);
        assert!((power - binomial_lower_reference(17, 30, 0.5).unwrap()).abs() < 1e-12);
        let large = binomial_lower_tail(2_500, 5_000, 0.5).unwrap();
        let complement = binomial_lower_tail(2_499, 5_000, 0.5).unwrap();
        assert!(large.is_finite() && (large + complement - 1.0).abs() < 1e-9);
        let mut synthetic = rows_for(&protocol, 17, true);
        synthetic.claimed_intent = Some(ComparisonIntent::Equivalence);
        let synthetic = validate_trial_import(&synthetic, &protocol, &binding()).unwrap();
        let synthetic_verdicts = summarize_claims(&synthetic, &protocol).unwrap();
        assert!(
            synthetic_verdicts
                .iter()
                .all(|verdict| verdict.decision == "success")
        );
        assert!(super::super::battery::check_battery_equivalence(&synthetic_verdicts).is_err());
        let setup = setup_for(protocol.clone());
        for (correct, expected) in [(17, "success"), (22, "inconclusive"), (30, "failure")] {
            let mut import = rows_for(&protocol, correct, false);
            import.claimed_intent = Some(ComparisonIntent::Equivalence);
            import.setup_hash = Some(setup.setup_hash.clone());
            let validated = validate_trial_import_with_setup(&import, &setup).unwrap();
            let verdicts = summarize_claims(&validated, &protocol).unwrap();
            assert!(verdicts.iter().all(|verdict| verdict.decision == expected));
            assert!(verdicts.iter().all(|verdict| {
                verdict.intent == Some(ComparisonIntent::Equivalence)
                    && verdict.equivalence_bound_p_correct == Some(0.75)
            }));
            assert_eq!(
                super::super::battery::check_battery_equivalence(&verdicts).is_ok(),
                expected == "success"
            );
            assert_eq!(
                super::super::battery::score_battery(&verdicts)
                    .unwrap()
                    .equivalence_supported,
                expected == "success"
            );
            if correct == 17 {
                assert!((verdicts[0].p_value - 0.021_593_640_879_855_47).abs() < 1e-12);
                assert!(
                    (verdicts[0].p_value - verdicts[0].reference_p_value.unwrap()).abs() < 1e-12
                );
            }
        }

        assert!(
            BlindedProtocol::preregister(
                protocol.design,
                protocol.comparison.clone(),
                protocol.conditions.clone(),
                30,
                0.05,
                protocol.target_power,
                protocol.minimum_effect_p_correct,
                DecisionRule::AbxEquivalence {
                    max_correct: 18,
                    trials: 30,
                    alpha: 0.05,
                    max_p_correct: 0.75,
                    alternative_p_correct: 0.5,
                },
                protocol.randomization_seed,
            )
            .is_err()
        );
        assert!(
            BlindedProtocol::preregister(
                protocol.design,
                protocol.comparison.clone(),
                protocol.conditions.clone(),
                30,
                0.05,
                0.9,
                protocol.minimum_effect_p_correct,
                protocol.decision.clone(),
                protocol.randomization_seed,
            )
            .is_err()
        );
    }
}
