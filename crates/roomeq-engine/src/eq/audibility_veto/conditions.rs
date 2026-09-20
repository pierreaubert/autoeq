//! Frozen-reference evaluation of declared pruning conditions using the experimental proxy.
//!
//! A condition supplies the measured seat/programme spectrum before EQ and a
//! nominal playback level. This remains a magnitude-only heuristic, not a
//! calibrated auditory comparison model. Missing or invalid conditions prevent
//! removal; neither averaging nor re-normalizing each candidate can hide drift.

use ndarray::Array1;
use roomeq_model::BudgetAggregation;

/// One declared seat, programme, and playback-level combination.
#[derive(Debug, Clone)]
pub struct VetoCondition {
    /// Stable identifier binding this evidence to the declared condition set.
    pub id: String,
    /// Seat plus programme magnitude spectrum, in dB on the adjudication grid.
    ///
    /// This must describe the uncorrected background. The evaluator adds each
    /// candidate's filter response and freezes the level anchor at full-chain F0.
    pub background_db: Array1<f64>,
    /// Nominal full-chain playback level, not a physical SPL calibration.
    pub listening_phon: f64,
}

/// Complete condition evidence and its predeclared cumulative aggregation policy.
#[derive(Debug, Clone)]
pub struct VetoConditionSet<'a> {
    /// Every declared identifier must occur exactly once in `conditions`.
    pub declared_ids: &'a [String],
    /// Conditions evaluated independently for every proposed removal.
    pub conditions: &'a [VetoCondition],
    /// Sum or worst-condition cumulative distance; never an average.
    pub aggregation: BudgetAggregation,
}

/// Bind the frozen response to its grid, ordered filters, and condition evidence.
pub(super) fn reference_id(
    response_id: &str,
    freqs: &Array1<f64>,
    responses: &[Array1<f64>],
    set: &VetoConditionSet<'_>,
) -> String {
    // FNV-1a is a reproducibility identifier, not an authenticity guarantee.
    let mut hash = 0xcbf29ce484222325_u64;
    let mut append = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    for value in freqs
        .iter()
        .chain(responses.iter().flat_map(|response| response.iter()))
    {
        append(&value.to_bits().to_le_bytes());
    }
    for id in set.declared_ids {
        append(&(id.len() as u64).to_le_bytes());
        append(id.as_bytes());
    }
    for condition in set.conditions {
        append(&(condition.id.len() as u64).to_le_bytes());
        append(condition.id.as_bytes());
        append(&condition.listening_phon.to_bits().to_le_bytes());
        for value in &condition.background_db {
            append(&value.to_bits().to_le_bytes());
        }
    }
    format!("{response_id}:conditions-v1:{hash:016x}")
}

/// Frozen experimental loudness references for a complete declared condition set.
///
/// Different conditions may have different delivered responses, as in a routed
/// graph. Anchors remain fixed at construction; nominal phons are not physical
/// calibration and the resulting differences do not prove perceptual equivalence.
#[derive(Debug)]
pub struct ConditionEvaluator<'a> {
    set: &'a VetoConditionSet<'a>,
    erb: Vec<f64>,
    weights: Array1<f64>,
    anchors: Vec<f64>,
    f0_loudness: Vec<f64>,
}

impl<'a> ConditionEvaluator<'a> {
    /// Freezes the original response and level anchor of every condition.
    ///
    /// For condition-specific original responses, include each in its condition's
    /// background and pass a zero common response.
    ///
    /// # Errors
    /// Returns an error for incomplete, malformed, or nonfinite condition evidence.
    pub fn new(
        set: &'a VetoConditionSet<'a>,
        freqs: &Array1<f64>,
        f0: &Array1<f64>,
    ) -> Result<Self, String> {
        if freqs
            .windows(2)
            .into_iter()
            .any(|pair| pair[1] <= pair[0] || super::erb_rate(pair[1]) <= super::erb_rate(pair[0]))
        {
            return Err(String::from(
                "frequency grid must increase strictly in Hz and ERB rate",
            ));
        }
        let weights = super::erb_weights(freqs).ok_or_else(|| {
            let pair = freqs.windows(2).into_iter().find_map(|pair| {
                (pair[1] <= pair[0] || super::erb_rate(pair[1]) <= super::erb_rate(pair[0]))
                    .then_some((pair[0], pair[1]))
            });
            format!(
                "invalid frequency grid: {} points, invalid adjacent pair {pair:?}",
                freqs.len()
            )
        })?;
        if f0.len() != freqs.len() || f0.iter().any(|value| !value.is_finite()) {
            return Err(String::from("invalid frozen-chain response"));
        }
        let declared: std::collections::BTreeSet<_> = set.declared_ids.iter().collect();
        let supplied: std::collections::BTreeSet<_> = set
            .conditions
            .iter()
            .map(|condition| &condition.id)
            .collect();
        if declared.is_empty()
            || declared.len() != set.declared_ids.len()
            || supplied.len() != set.conditions.len()
            || declared != supplied
            || declared.iter().any(|id| id.trim().is_empty())
        {
            return Err(String::from(
                "missing, duplicate, or undeclared condition evidence",
            ));
        }
        let erb = super::erb_positions(freqs);
        let mut anchors = Vec::with_capacity(set.conditions.len());
        let mut f0_loudness = Vec::with_capacity(set.conditions.len());
        for condition in set.conditions {
            if condition.background_db.len() != freqs.len()
                || condition
                    .background_db
                    .iter()
                    .any(|value| !value.is_finite())
                || !condition.listening_phon.is_finite()
                || condition.listening_phon <= 0.0
            {
                return Err(format!("invalid evidence for condition {}", condition.id));
            }
            let full = &condition.background_db + f0;
            let anchor = weighted_mean(&full, &weights);
            let loudness = super::approximate_loudness_sones(
                &full,
                &erb,
                Some(&weights),
                condition.listening_phon,
            );
            if !anchor.is_finite() || !loudness.is_finite() {
                return Err(format!("non-finite F0 for condition {}", condition.id));
            }
            anchors.push(anchor);
            f0_loudness.push(loudness);
        }
        Ok(Self {
            set,
            erb,
            weights,
            anchors,
            f0_loudness,
        })
    }

    fn loudness(&self, index: usize, response: &Array1<f64>) -> f64 {
        let condition = &self.set.conditions[index];
        let shape = &condition.background_db + response;
        // The proxy normally subtracts each input's mean. Compensate for that
        // normalization so every candidate retains the original F0 level anchor.
        let level =
            condition.listening_phon + weighted_mean(&shape, &self.weights) - self.anchors[index];
        super::approximate_loudness_sones(&shape, &self.erb, Some(&self.weights), level)
    }

    /// Worst-condition incremental impact and aggregated distance from frozen F0.
    pub(super) fn differences(
        &self,
        current: &Array1<f64>,
        candidate: &Array1<f64>,
    ) -> Option<(f64, f64)> {
        self.compare((0..self.set.conditions.len()).map(|_| (current, candidate)))
    }

    /// Compares condition-specific responses against the same frozen originals.
    ///
    /// Arrays follow `VetoConditionSet::conditions` order and share its grid. Returns
    /// `None` for missing, malformed, or nonfinite evidence. The first difference
    /// is the worst incremental condition; the second uses the declared cumulative
    /// sum/max aggregation. Response arrays are relative to each condition background.
    pub fn differences_for_responses(
        &self,
        current: &[Array1<f64>],
        candidate: &[Array1<f64>],
    ) -> Option<(f64, f64)> {
        if current.len() != self.set.conditions.len() || candidate.len() != current.len() {
            return None;
        }
        self.compare(current.iter().zip(candidate))
    }

    fn compare<'b>(
        &self,
        responses: impl IntoIterator<Item = (&'b Array1<f64>, &'b Array1<f64>)>,
    ) -> Option<(f64, f64)> {
        let mut incremental = 0.0_f64;
        let mut cumulative = 0.0_f64;
        for (index, (current, candidate)) in responses.into_iter().enumerate() {
            if current.len() != self.weights.len()
                || candidate.len() != self.weights.len()
                || current
                    .iter()
                    .chain(candidate)
                    .any(|value| !value.is_finite())
            {
                return None;
            }
            let before = self.loudness(index, current);
            let after = self.loudness(index, candidate);
            let step = (after - before).abs();
            let total = (after - self.f0_loudness[index]).abs();
            if !step.is_finite() || !total.is_finite() {
                return None;
            }
            incremental = incremental.max(step);
            cumulative = match self.set.aggregation {
                BudgetAggregation::Sum => cumulative + total,
                BudgetAggregation::Max => cumulative.max(total),
            };
        }
        cumulative.is_finite().then_some((incremental, cumulative))
    }

    pub(super) fn description(&self) -> String {
        format!(
            "conditions={:?}; aggregation={:?}",
            self.set.declared_ids, self.set.aggregation
        )
    }

    pub(super) fn calibration(&self) -> String {
        let levels: Vec<_> = self
            .set
            .conditions
            .iter()
            .map(|condition| format!("{}={}phon", condition.id, condition.listening_phon))
            .collect();
        format!(
            "nominal-condition-levels:[{}];unknown-spl",
            levels.join(",")
        )
    }
}

fn weighted_mean(shape: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    shape
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / weights.sum()
}

#[cfg(test)]
mod tests {
    use super::super::{
        AdjudicationConfig, VetoAdjudication, VetoEvaluation, adjudicate_veto_removals_for_budget,
        adjudicate_veto_removals_for_conditions, evaluate_audibility_veto,
    };
    use super::*;
    use math_audio_iir_fir::{Biquad, BiquadFilterType};
    use roomeq_model::{FilterAudibilityConfig, FilterVetoVerdict, ReportOutcome, VetoDecision};

    fn grid() -> Array1<f64> {
        Array1::from_iter((0..240).map(|index| 20.0 * 1000_f64.powf(index as f64 / 239.0)))
    }

    #[test]
    fn qa_roomeq_pruning_conditions_different_delivered_responses_do_not_cancel() {
        let freqs = grid();
        let conditions = condition_matrix(&freqs);
        let ids: Vec<_> = conditions
            .iter()
            .map(|condition| condition.id.clone())
            .collect();
        let zero = Array1::zeros(freqs.len());
        let current = vec![zero.clone(); conditions.len()];
        let candidate: Vec<_> = (0..conditions.len())
            .map(|index| Array1::from_elem(freqs.len(), if index % 2 == 0 { 0.25 } else { -0.25 }))
            .collect();
        let mut independent = Vec::new();
        for (index, condition) in conditions.iter().enumerate() {
            let set = VetoConditionSet {
                declared_ids: std::slice::from_ref(&ids[index]),
                conditions: std::slice::from_ref(condition),
                aggregation: BudgetAggregation::Max,
            };
            let evaluator = ConditionEvaluator::new(&set, &freqs, &zero).unwrap();
            independent.push(evaluator.differences(&zero, &candidate[index]).unwrap().1);
        }
        for aggregation in [BudgetAggregation::Sum, BudgetAggregation::Max] {
            let set = VetoConditionSet {
                declared_ids: &ids,
                conditions: &conditions,
                aggregation,
            };
            let evaluator = ConditionEvaluator::new(&set, &freqs, &zero).unwrap();
            let (step, cumulative) = evaluator
                .differences_for_responses(&current, &candidate)
                .unwrap();
            let worst = independent.iter().copied().fold(0.0_f64, f64::max);
            let expected = match aggregation {
                BudgetAggregation::Sum => independent.iter().sum(),
                BudgetAggregation::Max => worst,
            };
            assert!(worst > 0.0, "opposing responses must not average to zero");
            assert_eq!(step, worst);
            assert_eq!(cumulative, expected);
            assert!(
                evaluator
                    .differences_for_responses(&current[..1], &candidate)
                    .is_none()
            );
            let mut invalid = candidate.clone();
            invalid[0] = Array1::zeros(1);
            assert!(
                evaluator
                    .differences_for_responses(&current, &invalid)
                    .is_none()
            );
            invalid[0] = Array1::from_elem(freqs.len(), f64::NAN);
            assert!(
                evaluator
                    .differences_for_responses(&current, &invalid)
                    .is_none()
            );
        }
    }

    fn peak(gain: f64, frequency: f64, q: f64) -> Biquad {
        Biquad::new(BiquadFilterType::Peak, frequency, 48_000.0, q, gain)
    }

    fn config(enforce: bool) -> AdjudicationConfig {
        AdjudicationConfig {
            listening_phon: 75.0,
            per_step_quantum_sones: 0.05,
            cumulative_cap_sones: None,
            local_deviation_cap_db: 1.0,
            enforce,
            model_version: String::from("condition-matrix-v1"),
        }
    }

    fn nominations(filters: &[Biquad], freqs: &Array1<f64>) -> Vec<FilterVetoVerdict> {
        let mut verdicts = evaluate_audibility_veto(&VetoEvaluation {
            filters,
            freqs,
            listening_phon: 75.0,
            config: FilterAudibilityConfig::default(),
            hf_guard_start_hz: 1600.0,
            mode_proximity_evidence: &[],
        });
        // Deliberately nominate every filter: acceptance must independently
        // guard against a heuristic that proposes cancelling or narrow filters.
        for verdict in &mut verdicts {
            verdict.decision = VetoDecision::Remove;
        }
        verdicts
    }

    fn condition_matrix(freqs: &Array1<f64>) -> Vec<VetoCondition> {
        let mut conditions = Vec::new();
        for seat in 0..2 {
            for programme in 0..2 {
                for level in [55.0, 85.0] {
                    conditions.push(VetoCondition {
                        id: format!("seat-{seat}/programme-{programme}/{level}phon"),
                        background_db: freqs.mapv(|frequency| {
                            let seat_peak = if seat == 1 && (200.0..800.0).contains(&frequency) {
                                8.0
                            } else {
                                0.0
                            };
                            seat_peak - programme as f64 * 3.0 * (frequency / 1000.0).log2()
                        }),
                        listening_phon: level,
                    });
                }
            }
        }
        conditions
    }

    fn run(
        filters: Vec<Biquad>,
        conditions: &[VetoCondition],
        config: &AdjudicationConfig,
    ) -> (VetoAdjudication, Vec<FilterVetoVerdict>) {
        let freqs = grid();
        let ids: Vec<_> = conditions
            .iter()
            .map(|condition| condition.id.clone())
            .collect();
        let mut verdicts = nominations(&filters, &freqs);
        let result = adjudicate_veto_removals_for_conditions(
            filters,
            &mut verdicts,
            &freqs,
            config,
            &VetoConditionSet {
                declared_ids: &ids,
                conditions,
                aggregation: BudgetAggregation::Max,
            },
        );
        (result, verdicts)
    }

    #[test]
    fn qa_roomeq_pruning_conditions_frozen_level_does_not_normalize_away_gain() {
        let freqs = grid();
        let ids = vec![String::from("seat/programme/75phon")];
        let conditions = vec![VetoCondition {
            id: ids[0].clone(),
            background_db: Array1::zeros(freqs.len()),
            listening_phon: 75.0,
        }];
        let set = VetoConditionSet {
            declared_ids: &ids,
            conditions: &conditions,
            aggregation: BudgetAggregation::Max,
        };
        let f0 = Array1::zeros(freqs.len());
        let evaluator = ConditionEvaluator::new(&set, &freqs, &f0).unwrap();
        let candidate = Array1::from_elem(freqs.len(), -1.0);
        let (step, cumulative) = evaluator.differences(&f0, &candidate).unwrap();
        // For a flat spectrum there is no masking spread contribution. A 1 dB
        // reduction changes every band's excitation from 65 to 64 dB above the
        // proxy's threshold; the known power law gives an independent oracle.
        let expected = 0.1
            * (65_f64.powf(0.23) - 64_f64.powf(0.23))
            * super::super::erb_weights(&freqs).unwrap().sum();
        assert!(expected > 0.0);
        assert!((step - expected).abs() < 1e-12);
        assert!((cumulative - expected).abs() < 1e-12);
    }

    #[test]
    fn qa_roomeq_pruning_conditions_sum_and_max_cover_every_condition() {
        let freqs = grid();
        let conditions = condition_matrix(&freqs);
        let ids: Vec<_> = conditions
            .iter()
            .map(|condition| condition.id.clone())
            .collect();
        let f0 = Array1::zeros(freqs.len());
        let candidate = Array1::from_elem(freqs.len(), -0.2);
        let mut individual = Vec::new();
        for condition in &conditions {
            let set = VetoConditionSet {
                declared_ids: std::slice::from_ref(&condition.id),
                conditions: std::slice::from_ref(condition),
                aggregation: BudgetAggregation::Max,
            };
            individual.push(
                ConditionEvaluator::new(&set, &freqs, &f0)
                    .unwrap()
                    .differences(&f0, &candidate)
                    .unwrap()
                    .1,
            );
        }
        let worst = individual.iter().copied().fold(0.0, f64::max);
        let total: f64 = individual.iter().sum();
        for aggregation in [BudgetAggregation::Sum, BudgetAggregation::Max] {
            let set = VetoConditionSet {
                declared_ids: &ids,
                conditions: &conditions,
                aggregation,
            };
            let (step, cumulative) = ConditionEvaluator::new(&set, &freqs, &f0)
                .unwrap()
                .differences(&f0, &candidate)
                .unwrap();
            assert!((step - worst).abs() < 1e-12);
            let expected = if aggregation == BudgetAggregation::Sum {
                total
            } else {
                worst
            };
            assert!((cumulative - expected).abs() < 1e-12);
        }
        assert!(total > worst);
    }

    #[test]
    fn qa_roomeq_pruning_conditions_recompute_cumulative_overlap_and_rollback() {
        let conditions = condition_matrix(&grid());
        let filters = vec![peak(0.4, 500.0, 1.0), peak(0.4, 500.0, 1.0)];
        let mut policy = config(true);
        policy.local_deviation_cap_db = 0.6;
        let (applied, verdicts) = run(filters.clone(), &conditions, &policy);
        assert_eq!(applied.removed.len(), 1);
        assert_eq!(applied.kept.len(), 1);
        assert!(
            verdicts
                .iter()
                .any(|verdict| verdict.acceptance.reason.contains("local deviation"))
        );
        let mut restored = applied.kept.clone();
        for removal in &applied.removed {
            restored.insert(removal.index, removal.filter.clone());
        }
        let response = |filters: &[Biquad]| {
            filters
                .iter()
                .fold(Array1::<f64>::zeros(grid().len()), |sum, filter| {
                    sum + filter.np_log_result(&grid())
                })
        };
        assert!(
            (&response(&restored) - &response(&filters))
                .iter()
                .all(|value| value.abs() < 1e-12)
        );
        policy.enforce = false;
        let (advisory, verdicts) = run(filters, &conditions, &policy);
        assert_eq!(advisory.kept.len(), 2);
        assert!(advisory.removed.is_empty());
        assert_eq!(advisory.f0_reference_id, applied.f0_reference_id);
        assert_eq!(
            verdicts
                .iter()
                .filter(|verdict| verdict.acceptance.outcome == ReportOutcome::CandidateRemoval)
                .count(),
            1
        );
    }

    #[test]
    fn qa_roomeq_pruning_conditions_cancellation_narrow_peak_and_identity() {
        let conditions = condition_matrix(&grid());
        for filters in [
            vec![peak(6.0, 500.0, 1.0), peak(-6.0, 500.0, 1.0)],
            vec![peak(6.0, grid()[150], 100.0)],
        ] {
            let (result, _) = run(filters.clone(), &conditions, &config(true));
            assert_eq!(result.kept.len(), filters.len());
            assert!(result.removed.is_empty());
        }
        let (result, _) = run(vec![peak(0.1, 500.0, 1.0)], &conditions, &config(true));
        assert!(result.kept.is_empty());
        assert_eq!(result.removed.len(), 1);
    }

    #[test]
    fn qa_roomeq_pruning_conditions_reject_nonmonotonic_and_duplicate_grids() {
        // Positive central-difference cell widths alone do not prove that every
        // adjacent point is ordered (the second case is a counterexample).
        for frequencies in [
            vec![100.0, 100.0, 300.0, 400.0],
            vec![100.0, 500.0, 200.0, 600.0],
            vec![20.0 * (1.0 - f64::EPSILON), 20.0, 100.0, 1000.0],
            vec![100.0, f64::NAN, 300.0, 400.0],
        ] {
            let freqs = Array1::from_vec(frequencies);
            let ids = vec![String::from("condition")];
            let conditions = vec![VetoCondition {
                id: ids[0].clone(),
                background_db: Array1::zeros(freqs.len()),
                listening_phon: 75.0,
            }];
            let set = VetoConditionSet {
                declared_ids: &ids,
                conditions: &conditions,
                aggregation: BudgetAggregation::Max,
            };
            assert!(ConditionEvaluator::new(&set, &freqs, &Array1::zeros(freqs.len())).is_err());
        }
    }

    #[test]
    fn qa_roomeq_pruning_conditions_unknown_evidence_retains_filters() {
        let freqs = grid();
        let conditions = condition_matrix(&freqs);
        let ids: Vec<_> = conditions
            .iter()
            .map(|condition| condition.id.clone())
            .collect();
        let mut variants = vec![Vec::new(), conditions[..7].to_vec()];
        let mut duplicate = conditions.clone();
        duplicate[7].id = duplicate[0].id.clone();
        variants.push(duplicate);
        let mut invalid = conditions.clone();
        invalid[7].background_db[3] = f64::NAN;
        variants.push(invalid);
        let mut short = conditions.clone();
        short[7].background_db = Array1::zeros(1);
        variants.push(short);
        for supplied in variants {
            let filters = vec![peak(0.1, 500.0, 1.0)];
            let mut verdicts = nominations(&filters, &freqs);
            verdicts[0].enforced = true; // A reused record must be cleared.
            let result = adjudicate_veto_removals_for_conditions(
                filters,
                &mut verdicts,
                &freqs,
                &config(true),
                &VetoConditionSet {
                    declared_ids: &ids,
                    conditions: &supplied,
                    aggregation: BudgetAggregation::Max,
                },
            );
            assert_eq!(result.kept.len(), 1);
            assert!(!result.enforced);
            assert!(!verdicts[0].enforced);
            assert_eq!(
                verdicts[0].acceptance.outcome,
                ReportOutcome::InsufficientEvidence
            );
        }
    }

    #[test]
    fn qa_roomeq_pruning_conditions_unresolved_budget_cannot_claim_enforcement() {
        let freqs = grid();
        let filters = vec![peak(0.1, 500.0, 1.0)];
        let mut verdicts = nominations(&filters, &freqs);
        let budget = roomeq_model::PruningBudget {
            conditions: vec![String::from("unmeasured-seat/programme/75phon")],
            ..Default::default()
        };
        let result = adjudicate_veto_removals_for_budget(
            filters,
            &mut verdicts,
            &freqs,
            &config(true),
            Some(&budget),
        );
        assert_eq!(result.kept.len(), 1);
        assert!(!result.enforced);
        assert_eq!(
            verdicts[0].acceptance.outcome,
            ReportOutcome::InsufficientEvidence
        );
    }

    #[test]
    fn qa_roomeq_pruning_conditions_reference_binds_condition_evidence() {
        let mut conditions = condition_matrix(&grid());
        let filters = vec![peak(0.1, 500.0, 1.0)];
        let (first, _) = run(filters.clone(), &conditions, &config(false));
        conditions[7].listening_phon += 1.0;
        let (second, _) = run(filters, &conditions, &config(false));
        assert_ne!(first.f0_reference_id, second.f0_reference_id);
    }

    #[test]
    fn qa_roomeq_pruning_conditions_nominations_do_not_authorize_batch_removal() {
        let filters = vec![peak(0.1, 500.0, 1.0), peak(0.1, 1000.0, 1.0)];
        let verdicts = nominations(&filters, &grid());
        let (kept, verdicts) = super::super::enforce_veto_verdicts(filters.clone(), verdicts, true);
        assert_eq!(kept.len(), 2);
        assert!(verdicts.iter().all(|verdict| !verdict.enforced));
        let (kept, verdicts) =
            super::super::enforce_veto_verdicts(filters, verdicts[..1].to_vec(), true);
        assert_eq!(
            kept.len(),
            2,
            "a verdict mismatch must not truncate the chain"
        );
        assert_eq!(
            verdicts[0].acceptance.outcome,
            ReportOutcome::InsufficientEvidence
        );
    }
}
