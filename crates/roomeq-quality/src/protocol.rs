//! Blinded validation protocol staging for Stage 2.
//!
//! Names the auditory comparison, fixes the decision rules *before* data
//! collection (tamper-evident preregistration hash), sizes trials by power
//! analysis, and records results in sidecars. No listening happens here:
//! this module stages the validation so later stages can run it without
//! moving the goalposts.

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::sha256_hex;

/// Blinded comparison design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonDesign {
    /// Two/three-alternative forced choice.
    Abx,
    /// Multiple stimuli with hidden reference and anchor.
    Mushra,
}

/// What the auditory comparison is checked against. Exactly one must be
/// named before results count: an unnamed comparison has no validated
/// domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    /// A second, independently written implementation of the comparison
    /// metric that must agree within a stated tolerance.
    IndependentImplementation {
        /// What was reimplemented and the agreement tolerance.
        description: String,
    },
    /// A published algorithm with a citable reference implementation.
    PublishedAlgorithm {
        /// Citation plus version or commit.
        citation: String,
    },
    /// Published listening-test cases replayed verbatim.
    PublishedCases {
        /// Citation plus case identifiers.
        citation: String,
    },
}

/// What the comparison is staged to show. Detectability and equivalence
/// are staged separately from preference: a liked correction is not a
/// transparent one, and a nonsignificant ABX is not proof of equivalence
/// without a prespecified detection bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonIntent {
    /// Can listeners detect the difference.
    Detectability,
    /// Is the difference bounded below a prespecified detection bound.
    Equivalence,
    /// Which correction listeners prefer (never evidence of inaudibility).
    Preference,
}

fn default_intent() -> ComparisonIntent {
    ComparisonIntent::Detectability
}

/// Attribute words a preference-only comparison must not claim.
const PREFERENCE_FORBIDDEN: [&str; 4] =
    ["inaudib", "equivalen", "undetect", "transparent"];

/// The auditory comparison under validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ComparisonSpec {
    /// Comparison name, e.g. `"f0-vs-pruned-paired-comparison"`.
    pub name: String,
    /// What the comparison is staged to show.
    #[serde(default = "default_intent")]
    pub intent: ComparisonIntent,
    /// Tested attributes, e.g. `["detectability", "timbre-preference"]`.
    /// Preference is never a stand-in for inaudibility.
    #[serde(default)]
    pub attributes: Vec<String>,
    /// Prespecified effect/detection bound for [`ComparisonIntent::Equivalence`],
    /// e.g. `"d-prime below 0.5 at 80% power"`. Required for equivalence,
    /// ignored otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub equivalence_bound: Option<String>,
    /// The reference this comparison is checked against.
    pub reference: ReferenceKind,
    /// Domain where the comparison is validated. `"unvalidated"` until
    /// Stage 2 evidence lands — never assumed.
    #[serde(default = "unvalidated_domain")]
    pub validated_domain: String,
}

fn unvalidated_domain() -> String {
    String::from("unvalidated")
}

impl ComparisonSpec {
    /// Shape validation: a comparison needs a name, a reference, and an
    /// intent-consistent claim. Equivalence needs its bound up front;
    /// preference-only comparisons must not claim inaudibility — stage a
    /// detectability or equivalence arm separately instead.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err(String::from("comparison needs a name"));
        }
        if self.attributes.iter().any(|attribute| attribute.trim().is_empty()) {
            return Err(String::from("comparison attributes must be non-blank"));
        }
        if self.intent == ComparisonIntent::Equivalence
            && self.equivalence_bound.as_deref().is_none_or(|bound| bound.trim().is_empty())
        {
            return Err(String::from(
                "equivalence needs a prespecified detection bound: a nonsignificant result alone is not proof of equivalence",
            ));
        }
        if self.intent == ComparisonIntent::Preference {
            let lowered: Vec<String> =
                self.attributes.iter().map(|attribute| attribute.to_lowercase()).collect();
            if lowered.iter().any(|attribute| {
                PREFERENCE_FORBIDDEN.iter().any(|word| attribute.contains(word))
            }) {
                return Err(String::from(
                    "preference cannot claim inaudibility: stage detectability or equivalence separately",
                ));
            }
        }
        let description = match &self.reference {
            ReferenceKind::IndependentImplementation { description } => description,
            ReferenceKind::PublishedAlgorithm { citation } => citation,
            ReferenceKind::PublishedCases { citation } => citation,
        };
        if description.trim().is_empty() {
            return Err(String::from("comparison reference must be described"));
        }
        Ok(())
    }
}

/// Pre-registered decision rule. ABX rules are exact binomial; MUSHRA
/// rules are free-text criteria fixed before collection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DecisionRule {
    /// Pass when at least `min_correct` of `trials` are correct
    /// (one-sided level `alpha`).
    Abx {
        /// Minimum correct responses to pass.
        min_correct: u32,
        /// Trial count the rule was fixed for.
        trials: u32,
        /// One-sided significance level.
        alpha: f64,
    },
    /// Pass criterion written before data collection.
    Mushra {
        /// The criterion text, e.g. median thresholds and exclusion rules.
        criterion: String,
    },
}

/// A staged blinded protocol, preregistered by hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BlindedProtocol {
    /// Comparison design.
    pub design: ComparisonDesign,
    /// Named comparison under test.
    pub comparison: ComparisonSpec,
    /// Condition ids (F0-vs-chain pairs, seats, programmes, levels).
    #[serde(default)]
    pub conditions: Vec<String>,
    /// Trials per condition.
    pub trials_per_condition: u32,
    /// One-sided significance level.
    pub alpha: f64,
    /// Target statistical power for `minimum_effect_p_correct`.
    pub target_power: f64,
    /// Minimum correct-response rate the trial count is powered for.
    pub minimum_effect_p_correct: f64,
    /// Decision rule fixed before data collection.
    pub decision: DecisionRule,
    /// Seed for trial randomization.
    pub randomization_seed: u64,
    /// SHA-256 over the canonical protocol JSON (everything above).
    /// Recompute and compare before running or scoring: a mismatch means
    /// the rules moved after preregistration.
    pub prereg_hash: String,
}

impl BlindedProtocol {
    /// Build and preregister: validates, then hashes the canonical form.
    /// `prereg_hash` in the input is ignored and recomputed.
    #[allow(clippy::too_many_arguments)]
    pub fn preregister(
        design: ComparisonDesign,
        comparison: ComparisonSpec,
        conditions: Vec<String>,
        trials_per_condition: u32,
        alpha: f64,
        target_power: f64,
        minimum_effect_p_correct: f64,
        decision: DecisionRule,
        randomization_seed: u64,
    ) -> Result<Self, String> {
        comparison.validate()?;
        if conditions.is_empty() {
            return Err(String::from("protocol needs at least one condition"));
        }
        if trials_per_condition == 0 {
            return Err(String::from("trials_per_condition must be positive"));
        }
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(String::from("alpha must lie in (0, 1)"));
        }
        if !(0.0 < target_power && target_power < 1.0) {
            return Err(String::from("target_power must lie in (0, 1)"));
        }
        if !(0.5 < minimum_effect_p_correct && minimum_effect_p_correct <= 1.0) {
            return Err(String::from(
                "minimum_effect_p_correct must lie in (0.5, 1]",
            ));
        }
        if trials_per_condition > 5_000 {
            return Err(String::from("trials_per_condition above exact-computation range"));
        }
        let mut protocol = Self {
            design,
            comparison,
            conditions,
            trials_per_condition,
            alpha,
            target_power,
            minimum_effect_p_correct,
            decision,
            randomization_seed,
            prereg_hash: String::new(),
        };
        protocol.prereg_hash = protocol.canonical_hash()?;
        Ok(protocol)
    }

    /// Canonical JSON (prereg hash blanked) for hashing and comparison.
    fn canonical_json(&self) -> Result<String, String> {
        let mut clone = self.clone();
        clone.prereg_hash = String::new();
        serde_json::to_string(&clone)
            .map_err(|error| format!("protocol serialize error: {error}"))
    }

    /// SHA-256 over the canonical JSON.
    fn canonical_hash(&self) -> Result<String, String> {
        Ok(sha256_hex(self.canonical_json()?.as_bytes()))
    }

    /// Verify the stored preregistration hash against current content.
    pub fn verify_prereg(&self) -> Result<(), String> {
        if self.canonical_hash()? == self.prereg_hash {
            Ok(())
        } else {
            Err(String::from(
                "preregistration mismatch: protocol changed after registration",
            ))
        }
    }
}

/// Exact one-sided binomial upper-tail P(X >= k) at p = 0.5.
///
/// The sum is anchored at the distribution mode, whose term is always
/// representable, and recurrence steps outward from there (F13). Stepping up
/// from C(n,0) with a separate 2^-n factor instead underflows the factor
/// while the coefficient overflows, and the trailing `.min(1.0)` then turns
/// the resulting NaN into a plausible-looking 1.0 for large n.
pub fn abx_p_value(k_correct: u32, n_trials: u32) -> Result<f64, String> {
    if n_trials == 0 || n_trials > 5_000 {
        return Err(String::from("trials outside 1–5000 exact range"));
    }
    if k_correct > n_trials {
        return Err(String::from("correct cannot exceed trials"));
    }
    if k_correct == 0 {
        return Ok(1.0);
    }
    let n = n_trials as usize;
    let k = k_correct as usize;
    let mode = n / 2;
    // log P(X = mode): exact in log space, exp() is safe because the modal
    // term is >= ~0.005 for every supported n.
    let mut log_mode = -(n as f64) * std::f64::consts::LN_2;
    for i in 1..=mode {
        log_mode += ((n - mode + i) as f64 / i as f64).ln();
    }
    if !log_mode.is_finite() {
        return Err(String::from("binomial mode term is non-finite"));
    }
    let mut term = log_mode.exp();
    if k > mode {
        // Walk up from the mode, summing the k..=n tail. Terms shrink
        // monotonically past the mode; once they underflow, nothing further
        // can contribute.
        let mut cumulative = 0.0;
        for j in mode..=n {
            if j >= k {
                cumulative += term;
            }
            if j < n {
                term *= (n - j) as f64 / (j + 1) as f64;
                if term == 0.0 && j + 1 >= k {
                    break;
                }
            }
        }
        Ok(cumulative.min(1.0))
    } else {
        // Walk down from the mode, summing P(X <= k-1), and complement.
        // k <= mode keeps the lower sum below ~0.5, so no cancellation.
        let mut lower = 0.0;
        for j in (0..mode).rev() {
            term *= (j + 1) as f64 / (n - j) as f64;
            if j < k {
                lower += term;
            }
        }
        Ok((1.0 - lower).clamp(0.0, 1.0))
    }
}

/// Smallest k with `abx_p_value(k, n) <= alpha`.
///
/// Returns an error when no threshold attains `alpha` (e.g. one trial at
/// alpha 0.05, where even a perfect score has p = 0.5): there is no valid
/// significance rule, and callers must not score against a fabricated one.
pub fn abx_min_correct(n_trials: u32, alpha: f64) -> Result<u32, String> {
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(String::from("alpha must lie in (0, 1)"));
    }
    for k in 0..=n_trials {
        if abx_p_value(k, n_trials)? <= alpha {
            return Ok(k);
        }
    }
    Err(String::from(
        "no attainable significance threshold: even a perfect score cannot reach alpha at this trial count",
    ))
}

/// Normal-approximation trial count for detecting `p_alt` correct at
/// level `alpha` with `power`. Starting value for staging, not a
/// substitute for the exact rule actually applied.
pub fn abx_trials_needed(p_alt: f64, alpha: f64, power: f64) -> Result<u32, String> {
    if !(0.5 < p_alt && p_alt < 1.0) {
        return Err(String::from("p_alt must lie in (0.5, 1)"));
    }
    if !(0.0 < alpha && alpha < 1.0) || !(0.0 < power && power < 1.0) {
        return Err(String::from("alpha and power must lie in (0, 1)"));
    }
    let z_alpha = normal_quantile(1.0 - alpha);
    let z_power = normal_quantile(power);
    let numerator = z_alpha * 0.5 + z_power * (p_alt * (1.0 - p_alt)).sqrt();
    let trials = (numerator / (p_alt - 0.5)).powi(2).ceil() as u32;
    Ok(trials.clamp(8, 5_000))
}

/// Acklam-style inverse normal CDF (absolute error ~1e-9, plenty for
/// trial sizing; the applied rule stays exact).
fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let poly = |coeffs: &[f64], x: f64| coeffs.iter().fold(0.0, |acc, c| acc * x + c);
    if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        return poly(&C, q) / (poly(&D, q) * q + 1.0);
    }
    if p <= 0.97575 {
        let q = p - 0.5;
        let r = q * q;
        return poly(&A, r) * q / (poly(&B, r) * r + 1.0);
    }
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(poly(&C, q) / (poly(&D, q) * q + 1.0))
}

/// Worst-seat report with its full aggregation provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WorstSeatReport {
    /// Winning (worst) seat id.
    pub seat: String,
    /// Its value in dB.
    pub value_db: f64,
    /// Bins supporting the value.
    pub support_bins: usize,
    /// Per-bin weighting used.
    pub weighting: String,
    /// Aggregation order id (see seat-aggregation measure version).
    pub aggregation_order: String,
    /// 95% interval for the aggregate, same units.
    pub uncertainty_ci95_db: [f64; 2],
}

/// One scored condition outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ConditionOutcome {
    /// Condition id from the protocol.
    pub condition: String,
    /// Trials run.
    pub trials: u32,
    /// Correct responses.
    pub correct: u32,
    /// Exact one-sided p value (ABX) or NaN when not applicable.
    pub p_value: f64,
    /// `pass` or `fail` under the preregistered rule.
    pub decision: String,
    /// Minimum correct responses of the rule actually applied.
    #[serde(default)]
    pub rule_min_correct: u32,
    /// Trial count the applied rule was fixed for.
    #[serde(default)]
    pub rule_trials: u32,
    /// One-sided significance level of the applied rule (`0.0` when scored
    /// without a bound protocol, i.e. alpha unknown).
    #[serde(default)]
    pub rule_alpha: f64,
}

/// Results sidecar for a staged validation run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ValidationResult {
    /// Preregistration hash of the protocol actually scored.
    pub protocol_hash: String,
    /// Comparison name scored.
    pub comparison: String,
    /// Per-condition outcomes.
    pub outcomes: Vec<ConditionOutcome>,
    /// Aggregate gain in dB with its aggregation provenance.
    pub aggregate_gain_db: f64,
    /// How the aggregate was computed (order, weighting, support).
    pub aggregate_method: String,
    /// Worst supported seat.
    pub worst_seat: WorstSeatReport,
    /// Overall `pass`/`fail`/`inconclusive` under the preregistered rule.
    pub overall: String,
    /// Free notes (exclusions, deviations — deviations void preregistration).
    #[serde(default)]
    pub notes: String,
}

/// Score one ABX condition against explicit rule numbers.
///
/// The numbers are recorded into the outcome as the rule applied. Prefer
/// [`score_abx_condition_under_protocol`], which proves the numbers are the
/// preregistered rule instead of trusting the caller.
pub fn score_abx_condition(
    condition: &str,
    correct: u32,
    trials: u32,
    rule_min_correct: u32,
    rule_trials: u32,
) -> Result<ConditionOutcome, String> {
    // The loose scorer is not told alpha, so it records 0.0 (unknown) rather
    // than asserting one; the protocol-bound scorer records the real alpha.
    score_abx_condition_with_alpha(condition, correct, trials, rule_min_correct, rule_trials, 0.0)
}

/// Score one ABX condition, recording the applied rule's alpha as well.
pub fn score_abx_condition_with_alpha(
    condition: &str,
    correct: u32,
    trials: u32,
    rule_min_correct: u32,
    rule_trials: u32,
    rule_alpha: f64,
) -> Result<ConditionOutcome, String> {
    if trials != rule_trials {
        return Err(format!(
            "condition {condition}: ran {trials} trials under a rule fixed for {rule_trials}; re-preregister instead of reusing the rule"
        ));
    }
    let p_value = abx_p_value(correct, trials)?;
    Ok(ConditionOutcome {
        condition: String::from(condition),
        trials,
        correct,
        p_value,
        decision: String::from(if correct >= rule_min_correct { "pass" } else { "fail" }),
        rule_min_correct,
        rule_trials,
        rule_alpha,
    })
}

/// Score one ABX condition under a preregistered protocol.
///
/// Proves protocol identity before scoring: re-verifies the preregistration
/// hash against the canonical bytes, checks the condition against the
/// protocol's comparison spec, and scores only with the rule embedded in
/// that protocol — after validating the embedded threshold attains its own
/// alpha. Any substitution (edited protocol, foreign condition or rule)
/// fails closed.
pub fn score_abx_condition_under_protocol(
    protocol: &BlindedProtocol,
    condition: &str,
    correct: u32,
    trials: u32,
) -> Result<ConditionOutcome, String> {
    protocol.verify_prereg()?;
    let (min_correct, rule_trials, alpha) = match &protocol.decision {
        DecisionRule::Abx { min_correct, trials, alpha } => (*min_correct, *trials, *alpha),
        DecisionRule::Mushra { .. } => {
            return Err(String::from(
                "protocol embeds a MUSHRA rule: ABX counts cannot be scored against it",
            ));
        }
    };
    if !protocol.conditions.iter().any(|known| known == condition) {
        return Err(format!(
            "condition {condition} is not preregistered under comparison {}",
            protocol.comparison.name
        ));
    }
    if trials != protocol.trials_per_condition || trials != rule_trials {
        return Err(format!(
            "condition {condition}: ran {trials} trials under protocol rule fixed for {rule_trials}; re-preregister instead of reusing the rule"
        ));
    }
    if abx_p_value(min_correct, rule_trials)? > alpha {
        return Err(format!(
            "preregistered rule {min_correct}/{rule_trials} cannot attain alpha {alpha}: no valid significance decision exists"
        ));
    }
    score_abx_condition_with_alpha(condition, correct, trials, min_correct, rule_trials, alpha)
}

/// Write a results sidecar (`validation-result.json`) bound to the protocol
/// actually scored: re-verifies the preregistration hash, requires the
/// sidecar's hash to equal it, and requires the scored comparison id to
/// match the protocol's spec. Prefer this over [`write_results_sidecar`],
/// which can only check that *some* hash is present.
pub fn write_results_sidecar_under_protocol(
    dir: &Path,
    result: &ValidationResult,
    protocol: &BlindedProtocol,
) -> Result<PathBuf, String> {
    protocol.verify_prereg()?;
    if result.protocol_hash != protocol.prereg_hash {
        return Err(String::from(
            "results carry a different preregistration hash than the protocol scored; re-score under the right protocol instead of storing them under this one",
        ));
    }
    if result.comparison != protocol.comparison.name {
        return Err(String::from(
            "results name a different comparison than the protocol scored",
        ));
    }
    write_results_sidecar(dir, result)
}

/// Write a results sidecar (`validation-result.json`) next to staged
/// artifacts. Fails closed on a missing preregistration hash; prefer
/// [`write_results_sidecar_under_protocol`], which additionally proves the
/// hash and comparison id are the protocol actually scored.
pub fn write_results_sidecar(dir: &Path, result: &ValidationResult) -> Result<PathBuf, String> {
    if result.protocol_hash.trim().is_empty() {
        return Err(String::from(
            "results need the preregistration hash of the protocol scored",
        ));
    }
    std::fs::create_dir_all(dir)
        .map_err(|error| format!("cannot create results dir: {error}"))?;
    let json = serde_json::to_string_pretty(result)
        .map_err(|error| format!("results serialize error: {error}"))?;
    let path = dir.join("validation-result.json");
    std::fs::write(&path, json).map_err(|error| format!("cannot write sidecar: {error}"))?;
    Ok(path)
}

#[cfg(test)]
mod protocol_tests {
    use super::*;

    fn comparison() -> ComparisonSpec {
        ComparisonSpec {
            name: String::from("f0-vs-pruned-paired-comparison"),
            intent: ComparisonIntent::Detectability,
            attributes: vec![String::from("detectability")],
            equivalence_bound: None,
            reference: ReferenceKind::IndependentImplementation {
                description: String::from("log-domain reimplementation agrees within 1e-9 sones"),
            },
            validated_domain: String::from("unvalidated"),
        }
    }

    #[test]
    fn intent_rules_hold_equivalence_and_preference_apart() {
        // Equivalence without a prespecified bound fails: a
        // nonsignificant result alone proves nothing.
        let mut unbound = comparison();
        unbound.intent = ComparisonIntent::Equivalence;
        assert!(unbound.validate().is_err());
        unbound.equivalence_bound = Some(String::from("d-prime below 0.5 at 80% power"));
        assert!(unbound.validate().is_ok());
        // Blank bounds are bounds in name only.
        unbound.equivalence_bound = Some(String::from("  "));
        assert!(unbound.validate().is_err());
        // Preference claiming inaudibility fails; plain preference passes.
        let mut liked = comparison();
        liked.intent = ComparisonIntent::Preference;
        liked.attributes = vec![String::from("timbre-preference")];
        assert!(liked.validate().is_ok());
        liked.attributes = vec![String::from("proof of inaudibility")];
        assert!(liked.validate().is_err());
        // Blank attributes fail.
        let mut blank = comparison();
        blank.attributes = vec![String::from("  ")];
        assert!(blank.validate().is_err());
    }

    #[test]
    fn preregistration_is_stable_and_tamper_evident() {
        let protocol = BlindedProtocol::preregister(
            ComparisonDesign::Abx,
            comparison(),
            vec![String::from("seat-1-vs-f0")],
            30,
            0.05,
            0.8,
            0.75,
            DecisionRule::Abx { min_correct: 20, trials: 30, alpha: 0.05 },
            7,
        )
        .unwrap();
        assert!(protocol.verify_prereg().is_ok());
        // Re-registering the same content reproduces the hash.
        let twin = BlindedProtocol::preregister(
            ComparisonDesign::Abx,
            comparison(),
            vec![String::from("seat-1-vs-f0")],
            30,
            0.05,
            0.8,
            0.75,
            DecisionRule::Abx { min_correct: 20, trials: 30, alpha: 0.05 },
            7,
        )
        .unwrap();
        assert_eq!(protocol.prereg_hash, twin.prereg_hash);
        // Any post-registration edit voids it.
        let mut edited = protocol.clone();
        edited.trials_per_condition = 40;
        assert!(edited.verify_prereg().is_err());
    }

    #[test]
    fn abx_arithmetic_matches_textbook_values() {
        // 20/30 correct one-sided at p=0.5 is significant at 5%.
        let p = abx_p_value(20, 30).unwrap();
        assert!(p < 0.05, "p={p}");
        assert!(p > 0.01, "p={p}");
        // 15/30 is chance.
        let chance = abx_p_value(15, 30).unwrap();
        assert!((chance - 0.5).abs() < 0.08, "p={chance}");
        // Standard rule: 20/30 at alpha 0.05.
        assert_eq!(abx_min_correct(30, 0.05).unwrap(), 20);
        // Power sizing: normal approximation gives 23 for 75% vs chance
        // at alpha 0.05 / power 0.8 (exact binomial needs a few more;
        // the applied rule stays exact, this only stages trial counts).
        let sized = abx_trials_needed(0.75, 0.05, 0.8).unwrap();
        assert!((20..=28).contains(&sized), "sized={sized}");
    }

    #[test]
    fn abx_tail_is_exact_at_supported_extremes() {
        // F13: all-correct null probability is 2^-n, never 1.0.
        let p30 = abx_p_value(30, 30).unwrap();
        assert!(
            (p30 - 2.0_f64.powi(-30)).abs() / 2.0_f64.powi(-30) < 1e-12,
            "p30={p30:e}"
        );
        let p1000 = abx_p_value(1000, 1000).unwrap();
        assert!(
            (p1000 - 9.332636185032189e-302).abs() / 9.332636185032189e-302 < 1e-12,
            "p1000={p1000:e}"
        );
        let p1100 = abx_p_value(1100, 1100).unwrap();
        assert!(
            p1100 < 1e-300,
            "all-correct at 1100 trials must be ~2^-1100, got {p1100}"
        );
        // 2^-5000 is below f64 range: honest zero, not a plausible p value.
        assert_eq!(abx_p_value(5000, 5000).unwrap(), 0.0);
        // Sanity across the range: monotone in k, unity at k=0.
        let p0 = abx_p_value(0, 5000).unwrap();
        assert!((p0 - 1.0).abs() < 1e-12, "p0={p0}");
        let mut previous = 1.0;
        for k in [1, 7, 2500, 4999, 5000] {
            let p = abx_p_value(k, 5000).unwrap();
            assert!(p <= previous, "non-monotone at k={k}: {p} > {previous}");
            previous = p;
        }
    }

    #[test]
    fn abx_min_correct_reports_unattainable_threshold() {
        // F13: at one trial and alpha 0.05 no threshold meets alpha (best
        // achievable p is 0.5), so the rule must be reported unattainable
        // rather than returning 1 and scoring 1/1 as "pass".
        assert!(abx_min_correct(1, 0.05).is_err());
        assert!(abx_min_correct(2, 0.01).is_err());
        assert_eq!(abx_min_correct(30, 0.05).unwrap(), 20);
    }

    fn abx_protocol() -> BlindedProtocol {
        BlindedProtocol::preregister(
            ComparisonDesign::Abx,
            comparison(),
            vec![String::from("seat-1-vs-f0")],
            30,
            0.05,
            0.8,
            0.75,
            DecisionRule::Abx { min_correct: 20, trials: 30, alpha: 0.05 },
            7,
        )
        .unwrap()
    }

    #[test]
    fn scoring_is_bound_to_the_preregistered_protocol() {
        // F14: scoring must prove protocol identity (hash), comparison id,
        // and use the embedded rule — loose numbers must not suffice.
        let protocol = abx_protocol();
        let outcome =
            score_abx_condition_under_protocol(&protocol, "seat-1-vs-f0", 20, 30).unwrap();
        assert_eq!(outcome.decision, "pass");
        assert_eq!(outcome.rule_min_correct, 20);
        assert_eq!(outcome.rule_trials, 30);
        assert_eq!(outcome.rule_alpha, 0.05);
        // Unknown condition.
        assert!(
            score_abx_condition_under_protocol(&protocol, "seat-9-vs-f0", 20, 30).is_err()
        );
        // Trial-count drift.
        assert!(score_abx_condition_under_protocol(&protocol, "seat-1-vs-f0", 20, 31).is_err());
        // Tampered protocol (hash no longer matches canonical bytes).
        let mut tampered = protocol.clone();
        tampered.conditions.push(String::from("seat-9-vs-f0"));
        assert!(score_abx_condition_under_protocol(&tampered, "seat-1-vs-f0", 20, 30).is_err());
        // MUSHRA rule cannot score ABX counts.
        let mut mushra = protocol.clone();
        mushra.decision = DecisionRule::Mushra { criterion: String::from("median >= 80") };
        mushra.prereg_hash = mushra.canonical_hash().unwrap();
        assert!(score_abx_condition_under_protocol(&mushra, "seat-1-vs-f0", 20, 30).is_err());
        // Unattainable embedded rule (1/1 at alpha 0.05, best p = 0.5).
        let mut weak = abx_protocol();
        weak.decision = DecisionRule::Abx { min_correct: 1, trials: 1, alpha: 0.05 };
        weak.trials_per_condition = 1;
        weak.prereg_hash = weak.canonical_hash().unwrap();
        assert!(score_abx_condition_under_protocol(&weak, "seat-1-vs-f0", 1, 1).is_err());
    }

    #[test]
    fn sidecar_write_is_bound_to_the_preregistered_protocol() {
        // F14: the sidecar must record the protocol actually scored.
        let protocol = abx_protocol();
        let outcome =
            score_abx_condition_under_protocol(&protocol, "seat-1-vs-f0", 20, 30).unwrap();
        let result = ValidationResult {
            protocol_hash: protocol.prereg_hash.clone(),
            comparison: protocol.comparison.name.clone(),
            outcomes: vec![outcome],
            aggregate_gain_db: 0.0,
            aggregate_method: String::from("test"),
            worst_seat: WorstSeatReport {
                seat: String::from("seat-1-vs-f0"),
                value_db: 0.0,
                support_bins: 0,
                weighting: String::from("test"),
                aggregation_order: String::from("test"),
                uncertainty_ci95_db: [0.0, 0.0],
            },
            overall: String::from("pass"),
            notes: String::new(),
        };
        let dir = tempfile::tempdir().unwrap();
        write_results_sidecar_under_protocol(dir.path(), &result, &protocol).unwrap();
        // Hash mismatch and comparison mismatch fail closed.
        let mut wrong_hash = result.clone();
        wrong_hash.protocol_hash = String::from("deadbeef");
        assert!(write_results_sidecar_under_protocol(dir.path(), &wrong_hash, &protocol).is_err());
        let mut wrong_comparison = result.clone();
        wrong_comparison.comparison = String::from("other-comparison");
        assert!(
            write_results_sidecar_under_protocol(dir.path(), &wrong_comparison, &protocol)
                .is_err()
        );
    }

    #[test]
    fn scoring_rejects_trial_count_drift() {
        let outcome = score_abx_condition("seat-1", 20, 30, 20, 30).unwrap();
        assert_eq!(outcome.decision, "pass");
        let outcome = score_abx_condition("seat-1", 19, 30, 20, 30).unwrap();
        assert_eq!(outcome.decision, "fail");
        assert!(score_abx_condition("seat-1", 20, 31, 20, 30).is_err());
    }
}
