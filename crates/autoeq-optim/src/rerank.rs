//! Staged shortlist → rerank → refine pipeline for an optional auditory
//! objective (Stage 4).
//!
//! Fast existing objectives nominate a *bounded* shortlist (optimizer
//! results, Pareto-front members, and identity where appropriate). The
//! shortlist is reranked against the desired reference with a pinned
//! auditory evaluator — never by running an expensive auditory model
//! inside every optimizer evaluation. Shortlist, rerank, and refinement
//! are explicit stages with recorded loss pins: a loss definition that
//! changes mid-run aborts instead of silently switching. Evaluation,
//! wall-time, and memory budgets are enforced by a ledger, and fixed
//! measurement/probe transforms plus the full-chain reference are cached
//! by content hash so ablation reuses unaffected components.
//!
//! The evaluator slot accepts staged-metric evaluators today; a
//! listening-protocol basis validates only with recorded outcomes. A
//! steering/disagreement run through this pipeline is a plumbing test,
//! not proof of improvement — see [`compare_to_baselines`].

use std::collections::HashMap;
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Where a shortlist candidate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NominationSource {
    /// Winner (or shortlisted iterate) of an optimizer run.
    OptimizerRun,
    /// Member of a Pareto front (multi-objective tradeoffs stay visible).
    ParetoFront,
    /// Identity / zero-filter solution: the candidate to beat.
    Identity,
}

/// Pinned loss definition. Recorded at nomination and re-checked at every
/// later stage: a mismatch aborts the run instead of silently switching
/// loss definitions mid-run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LossPin {
    /// Loss id, e.g. `"speaker-flat"` or `"epa"`.
    pub loss: String,
    /// Loss implementation version.
    pub version: String,
}

impl LossPin {
    /// Both fields must be stated.
    pub fn validate(&self) -> Result<(), String> {
        if self.loss.trim().is_empty() || self.version.trim().is_empty() {
            return Err(String::from("loss pin needs a loss id and a version"));
        }
        Ok(())
    }
}

/// One nominated candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ShortlistCandidate {
    /// Stable candidate id.
    pub id: String,
    /// Parameter vector (opaque to this module).
    pub params: Vec<f64>,
    /// Fast-objective value at nomination (lower = better).
    pub fast_value: f64,
    /// Where the candidate came from.
    pub source: NominationSource,
    /// Loss definition the fast value was computed under.
    pub loss_pin: LossPin,
    /// Optimizer seed behind the nomination, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Bounded shortlist with its provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Shortlist {
    /// Candidates in nomination order.
    pub candidates: Vec<ShortlistCandidate>,
    /// Maximum size enforced at build time.
    pub max_size: usize,
    /// Whether identity inclusion was required.
    pub identity_required: bool,
}

/// Build a bounded shortlist.
///
/// Requires at least one candidate and at most `max_size`; values must be
/// finite; ids must be unique. When `identity_required`, an
/// [`NominationSource::Identity`] candidate must be present — the optimized
/// candidates are always compared against doing nothing.
pub fn build_shortlist(
    candidates: Vec<ShortlistCandidate>,
    max_size: usize,
    identity_required: bool,
) -> Result<Shortlist, String> {
    if candidates.is_empty() {
        return Err(String::from("shortlist needs at least one candidate"));
    }
    if max_size == 0 {
        return Err(String::from("shortlist max_size must be positive"));
    }
    if candidates.len() > max_size {
        return Err(format!(
            "shortlist holds {} candidates over the bound {max_size}",
            candidates.len()
        ));
    }
    let mut seen = std::collections::HashSet::new();
    for candidate in &candidates {
        if candidate.id.trim().is_empty() {
            return Err(String::from("shortlist candidate needs an id"));
        }
        if !seen.insert(candidate.id.clone()) {
            return Err(format!("duplicate shortlist candidate id '{}'", candidate.id));
        }
        if !candidate.fast_value.is_finite() {
            return Err(format!("candidate '{}' has a non-finite fast value", candidate.id));
        }
        candidate.loss_pin.validate().map_err(|error| {
            format!("candidate '{}': {error}", candidate.id)
        })?;
    }
    if identity_required
        && !candidates.iter().any(|candidate| candidate.source == NominationSource::Identity)
    {
        return Err(String::from(
            "shortlist requires the identity candidate and none was nominated",
        ));
    }
    Ok(Shortlist { candidates, max_size, identity_required })
}

/// Basis of the auditory evaluator used for reranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvaluatorBasis {
    /// Staged metric evaluator (e.g. a `roomeq-quality` final validation):
    /// plumbing-grade reranking, not listening evidence.
    StagedMetric {
        /// Metric name and version.
        metric: String,
    },
    /// Listening protocol with recorded outcomes. Validates only when the
    /// outcomes are present: naming a protocol stages nothing by itself.
    ListeningProtocol {
        /// Preregistration hash of the Stage 2 protocol.
        protocol_hash: String,
        /// Recorded outcomes (`true` = rerank order confirmed by listeners).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outcomes_recorded: Option<bool>,
    },
}

/// Pinned auditory evaluator descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AuditoryEvaluator {
    /// Evaluator name.
    pub name: String,
    /// Evaluator (model) version.
    pub model_version: String,
    /// What backs the scores.
    pub basis: EvaluatorBasis,
}

impl AuditoryEvaluator {
    /// The descriptor must be complete, and a listening basis without
    /// recorded outcomes fails closed: a protocol name is not evidence.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() || self.model_version.trim().is_empty() {
            return Err(String::from("evaluator needs a name and a model version"));
        }
        match &self.basis {
            EvaluatorBasis::StagedMetric { metric } if metric.trim().is_empty() => {
                Err(String::from("staged-metric basis needs a metric name"))
            }
            EvaluatorBasis::ListeningProtocol { protocol_hash, outcomes_recorded } => {
                if protocol_hash.trim().is_empty() {
                    return Err(String::from("listening basis needs the protocol hash"));
                }
                if outcomes_recorded != &Some(true) {
                    return Err(String::from(
                        "listening basis validates only with recorded outcomes",
                    ));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// Resource budgets for the rerank stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StageBudgets {
    /// Maximum evaluator calls (cache hits do not count).
    pub max_evaluations: usize,
    /// Wall-time budget in milliseconds.
    pub wall_time_ms: u64,
    /// Memory budget in bytes (tracked against caller-reported usage).
    pub max_memory_bytes: u64,
}

/// Ledger enforcing [`StageBudgets`]. Evaluation counts, cache hits, and
/// cancellation are recorded here so rerun reports stay comparable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BudgetLedger {
    /// Evaluator calls consumed.
    pub evaluations: usize,
    /// Cache hits served without consuming evaluations.
    pub cache_hits: usize,
    /// Caller-reported peak memory in bytes.
    pub peak_memory_bytes: u64,
    /// Cancellation requested (e.g. by the operator or a timeout owner).
    pub cancelled: bool,
    /// Budget the ledger enforces.
    pub budgets: Option<StageBudgets>,
}

impl BudgetLedger {
    /// Attach budgets (replaces any previous attachment).
    pub fn with_budgets(mut self, budgets: StageBudgets) -> Self {
        self.budgets = Some(budgets);
        self
    }

    /// Request cancellation: the next stage boundary aborts.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Consume one evaluation, enforcing count and cancellation.
    fn consume_evaluation(&mut self) -> Result<(), String> {
        if self.cancelled {
            return Err(String::from("rerank cancelled"));
        }
        self.evaluations += 1;
        if let Some(budgets) = &self.budgets
            && self.evaluations > budgets.max_evaluations
        {
            return Err(format!(
                "evaluation budget exhausted ({} > {})",
                self.evaluations, budgets.max_evaluations
            ));
        }
        Ok(())
    }

    /// Record caller-reported memory, enforcing the memory budget.
    pub fn record_memory(&mut self, bytes: u64) -> Result<(), String> {
        self.peak_memory_bytes = self.peak_memory_bytes.max(bytes);
        if let Some(budgets) = &self.budgets
            && self.peak_memory_bytes > budgets.max_memory_bytes
        {
            return Err(format!(
                "memory budget exhausted ({} > {})",
                self.peak_memory_bytes, budgets.max_memory_bytes
            ));
        }
        Ok(())
    }
}

/// Content cache for fixed measurement/probe transforms and the
/// full-chain reference. Keys are content hashes: identical inputs hit,
/// changed inputs miss — ablation reuses unaffected components without
/// any component-name bookkeeping at the call site.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RerankCache {
    /// Cached evaluator scores by `(transform_hash, candidate_id)`.
    entries: HashMap<String, f64>,
    /// Cache hits served (mirrored into the ledger by the caller).
    pub hits: usize,
    /// Cache misses computed.
    pub misses: usize,
}

impl RerankCache {
    /// Cache key for one candidate under one transform set.
    pub fn key(transform_hash: &str, candidate_id: &str) -> String {
        format!("{transform_hash}:{candidate_id}")
    }

    /// Hash opaque transform bytes (measurement/probe transforms,
    /// full-chain reference) to a cache namespace.
    pub fn hash_transforms(bytes: &[u8]) -> String {
        Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Look up a cached score.
    pub fn get(&mut self, key: &str) -> Option<f64> {
        let hit = self.entries.get(key).copied();
        if hit.is_some() {
            self.hits += 1;
        }
        hit
    }

    /// Store a computed score (miss accounting happens here).
    pub fn put(&mut self, key: String, score: f64) {
        self.misses += 1;
        self.entries.insert(key, score);
    }
}

/// One reranked candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RerankedCandidate {
    /// Shortlist candidate id.
    pub id: String,
    /// Evaluator score (lower = better).
    pub evaluator_score: f64,
    /// Rank (0 = winner). Ties break by candidate id for determinism.
    pub rank: usize,
    /// Score came from the cache rather than a fresh evaluation.
    pub cached: bool,
}

/// Rerank outcome with its full provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RerankReport {
    /// Candidates in rank order.
    pub ranked: Vec<RerankedCandidate>,
    /// Evaluator descriptor used.
    pub evaluator: AuditoryEvaluator,
    /// Loss pin the rerank ran under (must match the shortlist pins).
    pub loss_pin: LossPin,
    /// Wall time consumed in milliseconds.
    pub wall_time_ms: u64,
    /// Ledger at completion.
    pub ledger: BudgetLedger,
}

/// Rerank the shortlist with the pinned evaluator.
///
/// `scorer` runs at most once per candidate: cache hits reuse the stored
/// score without consuming the evaluation budget. Every candidate must be
/// scored exactly once (missing or non-finite scores abort); ranks are
/// deterministic under ties. The shortlist loss pins must all equal
/// `loss_pin` — a switched loss aborts the run. The wall-time budget is
/// enforced around scoring.
pub fn rerank(
    shortlist: &Shortlist,
    evaluator: &AuditoryEvaluator,
    loss_pin: &LossPin,
    transform_hash: &str,
    cache: &mut RerankCache,
    ledger: &mut BudgetLedger,
    scorer: impl Fn(&ShortlistCandidate) -> Result<f64, String>,
) -> Result<RerankReport, String> {
    evaluator.validate()?;
    loss_pin.validate()?;
    for candidate in &shortlist.candidates {
        if candidate.loss_pin != *loss_pin {
            return Err(format!(
                "candidate '{}' pins loss '{}@{}', rerank runs '{}@{}': loss switched mid-run",
                candidate.id,
                candidate.loss_pin.loss,
                candidate.loss_pin.version,
                loss_pin.loss,
                loss_pin.version
            ));
        }
    }
    let started = Instant::now();
    let wall_budget = ledger.budgets.as_ref().map(|budgets| budgets.wall_time_ms);
    let mut scored: Vec<(String, f64, bool)> = Vec::with_capacity(shortlist.candidates.len());
    for candidate in &shortlist.candidates {
        if ledger.cancelled {
            return Err(String::from("rerank cancelled"));
        }
        let key = RerankCache::key(transform_hash, &candidate.id);
        if let Some(score) = cache.get(&key) {
            ledger.cache_hits += 1;
            scored.push((candidate.id.clone(), score, true));
            continue;
        }
        ledger.consume_evaluation()?;
        let score = scorer(candidate)?;
        if !score.is_finite() {
            return Err(format!("evaluator score for '{}' is non-finite", candidate.id));
        }
        cache.put(key, score);
        scored.push((candidate.id.clone(), score, false));
        if let Some(budget_ms) = wall_budget
            && started.elapsed() > Duration::from_millis(budget_ms)
        {
            return Err(format!(
                "wall-time budget exhausted (>{budget_ms} ms) during rerank"
            ));
        }
    }
    scored.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    let ranked = scored
        .into_iter()
        .enumerate()
        .map(|(rank, (id, evaluator_score, cached))| RerankedCandidate {
            id,
            evaluator_score,
            rank,
            cached,
        })
        .collect();
    Ok(RerankReport {
        ranked,
        evaluator: evaluator.clone(),
        loss_pin: loss_pin.clone(),
        wall_time_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        ledger: ledger.clone(),
    })
}

/// Explicit refinement stage after reranking. Refinement never redefines
/// the loss: the result records the base candidate, the unchanged loss
/// pin, and the seed, so a silent loss switch would show up as a pin
/// mismatch at the next stage boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Refinement {
    /// Shortlist id the refinement started from.
    pub base_candidate_id: String,
    /// Loss pin carried over unchanged.
    pub loss_pin: LossPin,
    /// Refinement seed.
    pub seed: u64,
    /// Refined parameter vector.
    pub refined_params: Vec<f64>,
    /// Fast-objective value of the refinement under the pinned loss.
    pub refined_fast_value: f64,
}

/// Record a refinement, checking the pin still matches the base run.
pub fn record_refinement(
    base_candidate_id: &str,
    loss_pin: &LossPin,
    expected_pin: &LossPin,
    seed: u64,
    refined_params: Vec<f64>,
    refined_fast_value: f64,
) -> Result<Refinement, String> {
    loss_pin.validate()?;
    if loss_pin != expected_pin {
        return Err(String::from(
            "refinement loss pin differs from the rerank pin: loss switched between stages",
        ));
    }
    if base_candidate_id.trim().is_empty() {
        return Err(String::from("refinement needs its base candidate id"));
    }
    if !refined_fast_value.is_finite() {
        return Err(String::from("refinement fast value must be finite"));
    }
    Ok(Refinement {
        base_candidate_id: String::from(base_candidate_id),
        loss_pin: loss_pin.clone(),
        seed,
        refined_params,
        refined_fast_value,
    })
}

/// Held-out baseline entry (EPA/flat or any simpler objective).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BaselineEntry {
    /// Baseline name, e.g. `"epa"` or `"speaker-flat"`.
    pub name: String,
    /// Baseline value on the held-out set (lower = better).
    pub held_out_value: f64,
    /// Held-out set id the value was measured on.
    pub held_out_id: String,
}

/// Optional listening outcome attached to a baseline comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ListeningOutcome {
    /// Preregistration hash of the protocol that ran.
    pub protocol_hash: String,
    /// Whether listeners preferred the rerank winner over the baseline.
    pub winner_preferred: bool,
}

/// Verdict of the baseline comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonVerdict {
    /// The rerank winner beats the best baseline by the required margin
    /// on held-out data.
    AdoptCandidate,
    /// Benefit not demonstrated: keep the simpler objective.
    KeepSimpler,
}

/// Compare the rerank winner against EPA/flat baselines on held-out data.
///
/// Adoption needs a strictly better held-out value by at least
/// `min_margin`, measured on the same held-out set as the baselines.
/// Anything less keeps the simpler objective. Without listening outcomes
/// the verdict is held-out-only and says so; a steering/disagreement
/// fixture run is plumbing, never proof of improvement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BaselineComparison {
    /// Winner candidate id compared.
    pub winner_id: String,
    /// Winner held-out value (lower = better).
    pub winner_held_out_value: f64,
    /// Best baseline beaten (or the best baseline kept).
    pub baseline_name: String,
    /// The verdict.
    pub verdict: ComparisonVerdict,
    /// Margin of the winner over the best baseline in held-out units.
    pub margin: f64,
    /// Whether listening outcomes backed the verdict.
    pub listening_backed: bool,
}

pub fn compare_to_baselines(
    winner_id: &str,
    winner_held_out_value: f64,
    baselines: &[BaselineEntry],
    min_margin: f64,
    listening: Option<&ListeningOutcome>,
) -> Result<BaselineComparison, String> {
    if winner_id.trim().is_empty() {
        return Err(String::from("baseline comparison needs the winner id"));
    }
    if !winner_held_out_value.is_finite() {
        return Err(String::from("winner held-out value must be finite"));
    }
    if baselines.is_empty() {
        return Err(String::from("baseline comparison needs at least one baseline"));
    }
    if !min_margin.is_finite() || min_margin < 0.0 {
        return Err(String::from("min_margin must be finite and non-negative"));
    }
    let held_out_id = &baselines[0].held_out_id;
    let mut best: Option<&BaselineEntry> = None;
    for baseline in baselines {
        if baseline.name.trim().is_empty() {
            return Err(String::from("baseline entry needs a name"));
        }
        if !baseline.held_out_value.is_finite() {
            return Err(format!("baseline '{}' held-out value must be finite", baseline.name));
        }
        if baseline.held_out_id != *held_out_id {
            return Err(format!(
                "baseline '{}' ran on held-out '{}', expected '{held_out_id}': same-set comparison only",
                baseline.name, baseline.held_out_id
            ));
        }
        if best.is_none_or(|current| baseline.held_out_value < current.held_out_value) {
            best = Some(baseline);
        }
    }
    let best = best.expect("baselines non-empty");
    let margin = best.held_out_value - winner_held_out_value;
    let verdict = if margin >= min_margin {
        ComparisonVerdict::AdoptCandidate
    } else {
        ComparisonVerdict::KeepSimpler
    };
    Ok(BaselineComparison {
        winner_id: String::from(winner_id),
        winner_held_out_value,
        baseline_name: best.name.clone(),
        verdict,
        margin,
        listening_backed: listening.is_some_and(|outcome| {
            !outcome.protocol_hash.trim().is_empty() && outcome.winner_preferred
        }),
    })
}

/// Spectral resolution discipline: coarse grids may nominate, but final
/// comparisons run at validated resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GridResolution {
    /// Coarse screening grid (nominations only).
    Coarse,
    /// Validated final-comparison grid.
    Validated,
}

/// Check that a final comparison grid is valid: validated resolution at
/// or above the nomination grid, and never below the absolute minimum.
/// Coarse screening stays a nomination fast path, never the final word.
pub fn check_final_resolution(
    nomination_points: usize,
    final_points: usize,
    final_kind: GridResolution,
    min_validated_points: usize,
) -> Result<(), String> {
    if nomination_points == 0 || final_points == 0 || min_validated_points == 0 {
        return Err(String::from("grid point counts must be positive"));
    }
    if final_kind != GridResolution::Validated {
        return Err(String::from(
            "final comparisons require validated resolution, not coarse screening",
        ));
    }
    if final_points < nomination_points {
        return Err(format!(
            "final grid ({final_points}) is coarser than the nomination grid ({nomination_points}): temporal/spectral resolution needed by the evaluator would be lost"
        ));
    }
    if final_points < min_validated_points {
        return Err(format!(
            "final grid ({final_points}) is below the validated minimum ({min_validated_points})"
        ));
    }
    Ok(())
}

/// Supported temporal-masking model stage. Masking configuration must map
/// to one of these stages — never to an arbitrary extra penalty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MaskingStage {
    /// Forward-masking probe analysis.
    ForwardProbe,
    /// Backward-masking probe analysis.
    BackwardProbe,
    /// Simultaneous-masking analysis.
    Simultaneous,
}

/// Temporal-masking configuration for a loss option.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TemporalMaskingConfig {
    /// Model stage the configuration maps to.
    pub stage: MaskingStage,
    /// Probe/masker parameters for the stage.
    pub parameters: String,
}

impl TemporalMaskingConfig {
    /// The stage mapping must be a supported stage with stated parameters.
    pub fn validate(&self) -> Result<(), String> {
        let _ = self.stage;
        if self.parameters.trim().is_empty() {
            return Err(String::from(
                "masking configuration needs its stage parameters",
            ));
        }
        Ok(())
    }
}

/// Proposal for a new loss/schema option. Accepted only after its
/// semantics and numerical behavior are defined: a name plus a penalty
/// weight is not a loss definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LossOptionProposal {
    /// Proposed option name.
    pub name: String,
    /// What the option measures and why (semantics).
    pub semantics: String,
    /// Domain where the option stays finite and deterministic.
    pub finite_domain: String,
    /// Reference test pinning the numerical behavior.
    pub reference_test: String,
    /// Optional temporal-masking configuration (must map to a stage).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub masking: Option<TemporalMaskingConfig>,
}

impl LossOptionProposal {
    /// Every definition field must be stated; masking must map to a stage.
    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("name", &self.name),
            ("semantics", &self.semantics),
            ("finite_domain", &self.finite_domain),
            ("reference_test", &self.reference_test),
        ] {
            if value.trim().is_empty() {
                return Err(format!("loss proposal {field} must be stated"));
            }
        }
        if let Some(masking) = &self.masking {
            masking.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod rerank_tests {
    use super::*;

    fn pin() -> LossPin {
        LossPin { loss: String::from("speaker-flat"), version: String::from("v3") }
    }

    fn candidate(id: &str, value: f64, source: NominationSource) -> ShortlistCandidate {
        ShortlistCandidate {
            id: String::from(id),
            params: vec![value],
            fast_value: value,
            source,
            loss_pin: pin(),
            seed: Some(11),
        }
    }

    fn shortlist() -> Shortlist {
        build_shortlist(
            vec![
                candidate("opt-a", 2.0, NominationSource::OptimizerRun),
                candidate("pareto-b", 3.0, NominationSource::ParetoFront),
                candidate("identity", 9.0, NominationSource::Identity),
            ],
            8,
            true,
        )
        .unwrap()
    }

    fn evaluator() -> AuditoryEvaluator {
        AuditoryEvaluator {
            name: String::from("final-validation-metric"),
            model_version: String::from("roomeq-quality-final-v1"),
            basis: EvaluatorBasis::StagedMetric {
                metric: String::from("seat-aggregate-gain"),
            },
        }
    }

    #[test]
    fn shortlist_bounds_identity_and_shape() {
        assert_eq!(shortlist().candidates.len(), 3);
        // Over the bound fails.
        let many: Vec<ShortlistCandidate> = (0..9)
            .map(|index| candidate(&format!("c{index}"), index as f64, NominationSource::OptimizerRun))
            .collect();
        assert!(build_shortlist(many, 8, false).is_err());
        // Missing identity fails when required, passes when not.
        let no_identity = vec![candidate("opt-a", 1.0, NominationSource::OptimizerRun)];
        assert!(build_shortlist(no_identity.clone(), 8, true).is_err());
        assert!(build_shortlist(no_identity, 8, false).is_ok());
        // Duplicates, non-finite values, and empty ids fail.
        assert!(
            build_shortlist(
                vec![
                    candidate("dup", 1.0, NominationSource::OptimizerRun),
                    candidate("dup", 2.0, NominationSource::OptimizerRun),
                ],
                8,
                false,
            )
            .is_err()
        );
        assert!(
            build_shortlist(
                vec![candidate("nan", f64::NAN, NominationSource::OptimizerRun)],
                8,
                false,
            )
            .is_err()
        );
        assert!(build_shortlist(Vec::new(), 8, false).is_err());
    }

    #[test]
    fn rerank_orders_scores_and_counts_budgets() {
        let list = shortlist();
        let mut cache = RerankCache::default();
        let mut ledger = BudgetLedger::default().with_budgets(StageBudgets {
            max_evaluations: 10,
            wall_time_ms: 60_000,
            max_memory_bytes: u64::MAX,
        });
        let transform = RerankCache::hash_transforms(b"measurement-v1");
        // Identity scores best here: the rerank can prefer doing nothing.
        let report = rerank(&list, &evaluator(), &pin(), &transform, &mut cache, &mut ledger, |candidate| {
            Ok(match candidate.id.as_str() {
                "identity" => 1.0,
                "opt-a" => 2.5,
                _ => 4.0,
            })
        })
        .unwrap();
        assert_eq!(report.ranked[0].id, "identity");
        assert_eq!(report.ranked[0].rank, 0);
        assert_eq!(ledger.evaluations, 3);
        assert_eq!(ledger.cache_hits, 0);
        // A rerun under the same transforms hits the cache: no fresh
        // evaluations, same order.
        let rerun = rerank(&list, &evaluator(), &pin(), &transform, &mut cache, &mut ledger, |_| {
            panic!("cache hit must not rescore");
        })
        .unwrap();
        assert_eq!(rerun.ranked[0].id, "identity");
        assert_eq!(ledger.evaluations, 3);
        assert_eq!(ledger.cache_hits, 3);
        // Changed transforms miss: ablation recomputes only what changed.
        let other = RerankCache::hash_transforms(b"measurement-v2");
        let mut ledger2 = BudgetLedger::default();
        let report2 = rerank(&list, &evaluator(), &pin(), &other, &mut cache, &mut ledger2, |candidate| {
            Ok(candidate.fast_value)
        })
        .unwrap();
        assert_eq!(ledger2.evaluations, 3);
        assert!(report2.ranked.iter().all(|ranked| !ranked.cached));
    }

    #[test]
    fn rerank_refuses_switched_losses_and_blown_budgets() {
        let list = shortlist();
        let mut cache = RerankCache::default();
        let mut ledger = BudgetLedger::default();
        let transform = RerankCache::hash_transforms(b"measurement-v1");
        // A candidate pinned to another loss aborts the run.
        let mut switched = list.clone();
        switched.candidates[0].loss_pin.loss = String::from("epa");
        assert!(
            rerank(&switched, &evaluator(), &pin(), &transform, &mut cache, &mut ledger, |candidate| {
                Ok(candidate.fast_value)
            })
            .is_err()
        );
        // Evaluation budget of zero still scores nothing.
        let mut tight = BudgetLedger::default().with_budgets(StageBudgets {
            max_evaluations: 1,
            wall_time_ms: 60_000,
            max_memory_bytes: u64::MAX,
        });
        assert!(
            rerank(&list, &evaluator(), &pin(), &transform, &mut cache, &mut tight, |candidate| {
                Ok(candidate.fast_value)
            })
            .is_err()
        );
        // Non-finite scores abort instead of ranking garbage.
        let mut fresh = RerankCache::default();
        let mut ledger3 = BudgetLedger::default();
        assert!(
            rerank(&list, &evaluator(), &pin(), &transform, &mut fresh, &mut ledger3, |_| Ok(f64::NAN))
                .is_err()
        );
        // Cancellation aborts at the next candidate boundary.
        let mut cancelled = BudgetLedger::default();
        cancelled.cancel();
        assert!(
            rerank(&list, &evaluator(), &pin(), &transform, &mut fresh, &mut cancelled, |candidate| {
                Ok(candidate.fast_value)
            })
            .is_err()
        );
    }

    #[test]
    fn listening_basis_needs_recorded_outcomes() {
        let mut protocol = evaluator();
        protocol.basis = EvaluatorBasis::ListeningProtocol {
            protocol_hash: String::from("abc123"),
            outcomes_recorded: None,
        };
        assert!(protocol.validate().is_err());
        protocol.basis = EvaluatorBasis::ListeningProtocol {
            protocol_hash: String::from("abc123"),
            outcomes_recorded: Some(false),
        };
        assert!(protocol.validate().is_err());
        protocol.basis = EvaluatorBasis::ListeningProtocol {
            protocol_hash: String::from("abc123"),
            outcomes_recorded: Some(true),
        };
        assert!(protocol.validate().is_ok());
    }

    #[test]
    fn refinement_carries_the_pin_unchanged() {
        let refined = record_refinement("opt-a", &pin(), &pin(), 5, vec![1.5], 1.5).unwrap();
        assert_eq!(refined.base_candidate_id, "opt-a");
        let other = LossPin { loss: String::from("epa"), version: String::from("v1") };
        assert!(record_refinement("opt-a", &other, &pin(), 5, vec![1.5], 1.5).is_err());
    }

    #[test]
    fn baseline_comparison_keeps_the_simpler_objective() {
        let baselines = vec![
            BaselineEntry {
                name: String::from("epa"),
                held_out_value: 3.0,
                held_out_id: String::from("held-out-rooms"),
            },
            BaselineEntry {
                name: String::from("speaker-flat"),
                held_out_value: 4.0,
                held_out_id: String::from("held-out-rooms"),
            },
        ];
        // Winner beats EPA by the margin: adopt.
        let adopt = compare_to_baselines("opt-a", 2.0, &baselines, 0.5, None).unwrap();
        assert_eq!(adopt.verdict, ComparisonVerdict::AdoptCandidate);
        assert_eq!(adopt.baseline_name, "epa");
        assert!(!adopt.listening_backed);
        // Small gain: keep the simpler objective.
        let keep = compare_to_baselines("opt-a", 2.8, &baselines, 0.5, None).unwrap();
        assert_eq!(keep.verdict, ComparisonVerdict::KeepSimpler);
        // Listening backing is recorded when listeners preferred the winner.
        let backed = compare_to_baselines(
            "opt-a",
            2.0,
            &baselines,
            0.5,
            Some(&ListeningOutcome {
                protocol_hash: String::from("abc123"),
                winner_preferred: true,
            }),
        )
        .unwrap();
        assert!(backed.listening_backed);
        // Mixed held-out sets are rejected, not averaged.
        let mut mixed = baselines.clone();
        mixed[1].held_out_id = String::from("other-rooms");
        assert!(compare_to_baselines("opt-a", 2.0, &mixed, 0.5, None).is_err());
        assert!(compare_to_baselines("opt-a", 2.0, &[], 0.5, None).is_err());
    }

    #[test]
    fn final_resolution_discipline_holds() {
        assert!(
            check_final_resolution(200, 400, GridResolution::Validated, 256).is_ok()
        );
        // Coarse finals are refused, however fine.
        assert!(
            check_final_resolution(200, 800, GridResolution::Coarse, 256).is_err()
        );
        // Finals coarser than nominations lose evaluator resolution.
        assert!(
            check_final_resolution(400, 200, GridResolution::Validated, 128).is_err()
        );
        // Below the validated minimum is refused.
        assert!(
            check_final_resolution(100, 200, GridResolution::Validated, 256).is_err()
        );
    }

    #[test]
    fn loss_proposals_need_semantics_not_just_a_weight() {
        let proposal = LossOptionProposal {
            name: String::from("temporal-masking-probe"),
            semantics: String::from("forward-masking probe residual under staged probes"),
            finite_domain: String::from("finite-complex transfer with trusted timing"),
            reference_test: String::from("rerank_tests::masking_proposal_shape"),
            masking: Some(TemporalMaskingConfig {
                stage: MaskingStage::ForwardProbe,
                parameters: String::from("masker 1 kHz/200 ms, gap 20 ms"),
            }),
        };
        assert!(proposal.validate().is_ok());
        let mut vague = proposal.clone();
        vague.semantics = String::from("  ");
        assert!(vague.validate().is_err());
        let mut unmapped = proposal.clone();
        unmapped.masking = Some(TemporalMaskingConfig {
            stage: MaskingStage::ForwardProbe,
            parameters: String::from(""),
        });
        assert!(unmapped.validate().is_err());
    }
}
