//! Seat-wise quality definition and final validation (Stage 3).
//!
//! Percentiles are declared before use: the definition states the
//! frequency support band, the weighting measure, whether the percentile
//! runs over bins or seats, the aggregation order, the uncertainty
//! method, and the permitted degradation budget. Final validation then
//! compares the candidate against the declared baseline/target and — for
//! pruned candidates — against the accepted full chain, checks held-out
//! seats and the final sub/main sum, and reports aggregate gains with the
//! worst supported seat instead of a mean curve alone. Rejection is
//! bounded (per-seat budgets, distribution regression) rather than
//! zero-tolerance on a noisy single-bin maximum.

use autoeq_core::Curve;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::metrics::{
    AggregationOrder, SeatScore, bootstrap_mean_ci95, weighted_percentile, worst_supported_seat,
};
use super::protocol::WorstSeatReport;

/// Declared domain of the residual percentile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PercentileDomain {
    /// Percentile over frequency bins within each seat (worst seat rules).
    Bins,
    /// Percentile over per-seat mean residuals across seats.
    Seats,
}

/// Seat-wise quality metric definition. Every field is fixed before the
/// numbers are computed; two reports under different definitions are not
/// comparable even where field names match.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SeatMetricDefinition {
    /// Frequency support band in Hz. Only bins inside the band count.
    pub support_band_hz: [f64; 2],
    /// Weighting measure name, e.g. `"erb-rate"`.
    pub weighting: String,
    /// Weighting measure version, e.g. the auditory frequency measure.
    pub measure_version: String,
    /// Residual percentile level in `(0, 1)`.
    pub percentile: f64,
    /// Whether the percentile runs over bins or seats.
    pub percentile_domain: PercentileDomain,
    /// Order id for seat/bin aggregation (see [`AggregationOrder`]).
    pub aggregation_order: AggregationOrder,
    /// Uncertainty method id, e.g. `"seeded-bootstrap-ci95"`.
    pub uncertainty: String,
    /// Bootstrap resamples for the aggregate-gain interval.
    pub uncertainty_resamples: usize,
    /// Seed for the uncertainty resampling.
    pub uncertainty_seed: u64,
    /// Ceiling for the declared residual percentile in dB.
    pub max_residual_percentile_db: f64,
    /// Per-seat regression budget vs the baseline improvement in dB.
    /// A seat may lose up to this much and still pass: bounded tradeoff,
    /// not zero-tolerance.
    pub permitted_degradation_db: f64,
    /// Aggregate-gain floor vs baseline in dB.
    pub min_aggregate_gain_db: f64,
    /// Max per-seat improvement loss of a pruned candidate vs the
    /// accepted full chain in dB.
    pub max_pruning_loss_db: f64,
    /// Minimum in-band bins for a seat to count as supported.
    pub min_support_bins: usize,
}

impl SeatMetricDefinition {
    /// Validate the definition shape before any number is computed.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0 < self.support_band_hz[0] && self.support_band_hz[0] < self.support_band_hz[1])
            || !self.support_band_hz[1].is_finite()
        {
            return Err(String::from(
                "support_band_hz must be a finite ascending pair",
            ));
        }
        for (name, text) in [
            ("weighting", &self.weighting),
            ("measure_version", &self.measure_version),
            ("uncertainty", &self.uncertainty),
        ] {
            if text.trim().is_empty() {
                return Err(format!("seat metric {name} must be stated"));
            }
        }
        if !(0.0 < self.percentile && self.percentile < 1.0) {
            return Err(String::from("percentile must lie in (0, 1)"));
        }
        if self.uncertainty_resamples == 0 {
            return Err(String::from("uncertainty_resamples must be positive"));
        }
        if !self.max_residual_percentile_db.is_finite() {
            return Err(String::from("max_residual_percentile_db must be finite"));
        }
        for (name, value) in [
            ("permitted_degradation_db", self.permitted_degradation_db),
            ("max_pruning_loss_db", self.max_pruning_loss_db),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and non-negative"));
            }
        }
        if !self.min_aggregate_gain_db.is_finite() {
            return Err(String::from("min_aggregate_gain_db must be finite"));
        }
        Ok(())
    }
}

/// Declared baseline and accepted-chain references for final validation.
#[derive(Debug, Clone)]
pub struct FinalReferences {
    /// Baseline post-correction curves (declared comparator: previous
    /// correction, or pre-correction curves for an identity baseline).
    pub baseline_post: Vec<Curve>,
    /// Accepted full-chain post curves, required when the candidate is a
    /// pruned chain: pruning is compared against the accepted full chain.
    pub full_chain_post: Option<Vec<Curve>>,
}

/// Final sub/main summation evidence. Magnitude-level staging of the
/// routed sum: relative-phase damage shows up as in-band cancellation
/// even when isolated sub/main magnitudes look fine.
#[derive(Debug, Clone)]
pub struct SubMainSumEvidence {
    /// Sub-only response.
    pub sub: Curve,
    /// Main-only response.
    pub main: Curve,
    /// Final routed sum.
    pub summed: Curve,
    /// Crossover band in Hz where summation is checked.
    pub crossover_band_hz: [f64; 2],
    /// Largest tolerated in-band cancellation in dB.
    pub max_cancellation_db: f64,
}

/// Per-seat final outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FinalSeatOutcome {
    /// Seat index.
    pub seat_index: usize,
    /// Candidate improvement vs pre in dB (ERB-weighted RMS).
    pub candidate_improvement_db: f64,
    /// Baseline improvement vs pre in dB.
    pub baseline_improvement_db: f64,
    /// Candidate shortfall vs baseline in dB (positive = worse).
    pub degradation_vs_baseline_db: f64,
    /// Improvement loss vs the accepted full chain in dB, when staged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pruning_loss_db: Option<f64>,
    /// In-band supporting bins.
    pub support_bins: usize,
    /// Seat passed its bounded checks.
    pub accepted: bool,
    /// Seat violations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub violations: Vec<String>,
}

/// Final validation report: aggregate gain plus worst supported seat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FinalValidationReport {
    /// Metric definition the numbers were computed under.
    pub definition: SeatMetricDefinition,
    /// Mean baseline-post minus candidate-post RMS over seats in dB
    /// (positive = candidate better than baseline).
    pub aggregate_gain_db: f64,
    /// 95% interval for the aggregate gain, same units.
    pub aggregate_gain_ci95_db: [f64; 2],
    /// Worst supported seat by post-correction residual.
    pub worst_seat: WorstSeatReport,
    /// Declared residual percentile value in dB.
    pub residual_percentile_db: f64,
    /// Per-seat outcomes.
    pub seats: Vec<FinalSeatOutcome>,
    /// Sub/main summation verdict detail.
    pub sub_main_detail: String,
    /// Overall violations (empty = pass).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub violations: Vec<String>,
}

impl FinalValidationReport {
    /// True when no violation was recorded.
    pub fn accepted(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Shared-grid structural check: length and frequency alignment.
fn check_grid(curve: &Curve, reference: &Curve, label: &str) -> Result<(), String> {
    for grid in [curve, reference] {
        if grid.freq.len() != grid.spl.len()
            || grid.freq.iter().any(|f| !f.is_finite() || *f <= 0.0)
            || grid
                .freq
                .iter()
                .zip(grid.freq.iter().skip(1))
                .any(|(a, b)| a >= b)
        {
            return Err(format!(
                "{label} requires finite ascending frequencies and matching SPL lengths"
            ));
        }
    }
    if curve.freq.len() < 2
        || curve.freq.len() != reference.freq.len()
        || curve
            .freq
            .iter()
            .zip(reference.freq.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9)
    {
        return Err(format!("{label} requires the shared frequency grid"));
    }
    if curve.spl.len() != curve.freq.len() {
        return Err(format!("{label} has mismatched spl/freq lengths"));
    }
    Ok(())
}

/// Indices of grid bins inside `band`.
fn support_indices(freq: &ndarray::Array1<f64>, band: [f64; 2]) -> Vec<usize> {
    freq.iter()
        .enumerate()
        .filter(|(_, frequency)| **frequency >= band[0] && **frequency <= band[1])
        .map(|(index, _)| index)
        .collect()
}

/// ERB-rate-weighted RMS of `values` on the `freq` axis (`None` when the
/// axis is invalid — callers turn that into a violation, never a number).
fn band_rms(freq: &ndarray::Array1<f64>, values: &[f64]) -> Option<f64> {
    if values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    autoeq_core::erb_rate_weighted_rms(freq, values)
}

/// ERB cell weights for the axis (`None` when invalid).
fn band_weights(freq: &ndarray::Array1<f64>) -> Option<Vec<f64>> {
    autoeq_core::try_erb_rate_cell_widths(freq).map(|weights| weights.to_vec())
}

/// Validate the final candidate.
///
/// Compares per-seat candidate post curves against the declared baseline
/// (and the accepted full chain when staged), checks the sub/main sum,
/// and reports aggregate gain with the worst supported seat. Structural
/// grid problems abort with `Err`; content problems become violations so
/// the report stays complete.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_final_validation(
    definition: &SeatMetricDefinition,
    pre: &[Curve],
    candidate_post: &[Curve],
    target: &Curve,
    references: &FinalReferences,
    sub_main: &SubMainSumEvidence,
) -> Result<FinalValidationReport, String> {
    definition.validate()?;
    check_grid(target, target, "target")?;
    if definition.weighting != "erb-rate"
        || definition.measure_version != "auditory-frequency-measure-v1"
    {
        return Err("unsupported final-validation weighting or measure version".to_string());
    }
    if pre.is_empty() {
        return Err(String::from("final validation needs at least one seat"));
    }
    if pre.len() != candidate_post.len() || pre.len() != references.baseline_post.len() {
        return Err(String::from(
            "pre/candidate/baseline seat counts must match",
        ));
    }
    if let Some(full_chain) = &references.full_chain_post
        && full_chain.len() != pre.len()
    {
        return Err(String::from("full-chain seat count must match"));
    }
    for (index, curve) in pre
        .iter()
        .chain(candidate_post.iter())
        .chain(references.baseline_post.iter())
        .chain(references.full_chain_post.iter().flatten())
        .enumerate()
    {
        check_grid(curve, target, &format!("seat curve {index}"))?;
    }
    check_grid(&sub_main.sub, &sub_main.main, "sub/main")?;
    check_grid(&sub_main.summed, &sub_main.main, "summed")?;
    if !(0.0 < sub_main.crossover_band_hz[0]
        && sub_main.crossover_band_hz[0] < sub_main.crossover_band_hz[1])
        || !sub_main.crossover_band_hz[1].is_finite()
    {
        return Err(String::from(
            "crossover_band_hz must be a finite ascending pair",
        ));
    }
    if !sub_main.max_cancellation_db.is_finite() || sub_main.max_cancellation_db < 0.0 {
        return Err(String::from(
            "max_cancellation_db must be finite and non-negative",
        ));
    }

    let support = support_indices(&target.freq, definition.support_band_hz);
    let support_freq: ndarray::Array1<f64> =
        support.iter().map(|index| target.freq[*index]).collect();
    let weights = band_weights(&support_freq);
    let mut violations = Vec::new();
    let mut seats = Vec::with_capacity(pre.len());
    let mut gains = Vec::with_capacity(pre.len());
    let mut seat_scores = Vec::with_capacity(pre.len());
    let mut seat_mean_residuals = Vec::with_capacity(pre.len());

    for (seat_index, ((pre_curve, cand_curve), base_curve)) in pre
        .iter()
        .zip(candidate_post.iter())
        .zip(references.baseline_post.iter())
        .enumerate()
    {
        let mut outcome = FinalSeatOutcome {
            seat_index,
            candidate_improvement_db: 0.0,
            baseline_improvement_db: 0.0,
            degradation_vs_baseline_db: 0.0,
            pruning_loss_db: None,
            support_bins: support.len(),
            accepted: false,
            violations: Vec::new(),
        };
        let gather = |curve: &Curve| -> Vec<f64> {
            support.iter().map(|index| curve.spl[*index]).collect()
        };
        let pre_spl = gather(pre_curve);
        let cand_spl = gather(cand_curve);
        let base_spl = gather(base_curve);
        let target_spl: Vec<f64> = support.iter().map(|index| target.spl[*index]).collect();
        let residual = |spl: &[f64]| {
            spl.iter()
                .zip(target_spl.iter())
                .map(|(value, target)| value - target)
                .collect::<Vec<f64>>()
        };
        let (Some(pre_rms), Some(cand_rms), Some(base_rms)) = (
            band_rms(&support_freq, &residual(&pre_spl)),
            band_rms(&support_freq, &residual(&cand_spl)),
            band_rms(&support_freq, &residual(&base_spl)),
        ) else {
            outcome
                .violations
                .push(format!("seat_{seat_index}_metric_unavailable"));
            seats.push(outcome);
            continue;
        };
        outcome.candidate_improvement_db = pre_rms - cand_rms;
        outcome.baseline_improvement_db = pre_rms - base_rms;
        outcome.degradation_vs_baseline_db = outcome.baseline_improvement_db
            - outcome.candidate_improvement_db;
        gains.push(base_rms - cand_rms);
        if let Some(weights) = &weights {
            let abs_residual: Vec<f64> =
                residual(&cand_spl).iter().map(|value| value.abs()).collect();
            let weighted_mean: f64 = abs_residual
                .iter()
                .zip(weights.iter())
                .map(|(value, weight)| value * weight)
                .sum::<f64>()
                / weights.iter().sum::<f64>();
            seat_mean_residuals.push(weighted_mean);
        }
        seat_scores.push(SeatScore {
            seat: format!("seat-{seat_index}"),
            value_db: cand_rms,
            support_bins: support.len(),
        });
        if outcome.degradation_vs_baseline_db > definition.permitted_degradation_db {
            outcome.violations.push(format!(
                "seat_{seat_index}_degraded_vs_baseline ({:.2} dB beyond {:.2} dB budget)",
                outcome.degradation_vs_baseline_db, definition.permitted_degradation_db
            ));
        }
        if let Some(full_chain) = &references.full_chain_post {
            let chain_spl = gather(&full_chain[seat_index]);
            if let Some(chain_rms) = band_rms(&support_freq, &residual(&chain_spl)) {
                let loss = (pre_rms - chain_rms) - outcome.candidate_improvement_db;
                outcome.pruning_loss_db = Some(loss);
                if loss > definition.max_pruning_loss_db {
                    outcome.violations.push(format!(
                        "seat_{seat_index}_pruning_loss ({loss:.2} dB beyond {:.2} dB budget)",
                        definition.max_pruning_loss_db
                    ));
                }
            } else {
                outcome
                    .violations
                    .push(format!("seat_{seat_index}_full_chain_metric_unavailable"));
            }
        }
        outcome.accepted = outcome.violations.is_empty();
        seats.push(outcome);
    }

    // Aggregate gain with its uncertainty interval.
    let aggregate_gain_db = if gains.len() == pre.len() && !gains.is_empty() {
        gains.iter().sum::<f64>() / gains.len() as f64
    } else {
        violations.push(String::from("aggregate_gain_unavailable"));
        0.0
    };
    let aggregate_gain_ci95_db = bootstrap_mean_ci95(
        &gains,
        definition.uncertainty_resamples,
        definition.uncertainty_seed,
    )
    .map(|(lower, upper)| [lower, upper])
    .unwrap_or_else(|| {
        violations.push(String::from("aggregate_gain_uncertainty_unavailable"));
        [aggregate_gain_db, aggregate_gain_db]
    });
    if aggregate_gain_db < definition.min_aggregate_gain_db {
        violations.push(format!(
            "aggregate_gain {aggregate_gain_db:.2} dB below floor {:.2} dB",
            definition.min_aggregate_gain_db
        ));
    }

    // Declared residual percentile on its declared domain.
    let residual_percentile_db = match definition.percentile_domain {
        PercentileDomain::Bins => {
            let mut worst = 0.0_f64;
            for (seat_index, cand_curve) in candidate_post.iter().enumerate() {
                let cand_spl: Vec<f64> =
                    support.iter().map(|index| cand_curve.spl[*index]).collect();
                let target_spl: Vec<f64> =
                    support.iter().map(|index| target.spl[*index]).collect();
                let abs_residual: Vec<f64> = cand_spl
                    .iter()
                    .zip(target_spl.iter())
                    .map(|(value, target)| (value - target).abs())
                    .collect();
                if let Some(weights) = &weights {
                    worst = worst.max(weighted_percentile(
                        &abs_residual,
                        weights,
                        definition.percentile,
                    ));
                } else {
                    violations.push(format!(
                        "seat_{seat_index}_percentile_weights_unavailable"
                    ));
                }
            }
            worst
        }
        PercentileDomain::Seats => {
            if seat_mean_residuals.len() == pre.len() && !seat_mean_residuals.is_empty() {
                let unit = vec![1.0; seat_mean_residuals.len()];
                weighted_percentile(&seat_mean_residuals, &unit, definition.percentile)
            } else {
                violations.push(String::from("seats_percentile_unavailable"));
                0.0
            }
        }
    };
    if residual_percentile_db > definition.max_residual_percentile_db {
        violations.push(format!(
            "residual percentile {residual_percentile_db:.2} dB exceeds ceiling {:.2} dB",
            definition.max_residual_percentile_db
        ));
    }

    // Worst supported seat by post-correction residual.
    let worst = worst_supported_seat(&seat_scores, definition.min_support_bins);
    let worst_seat = match worst {
        Some(score) => WorstSeatReport {
            seat: score.seat.clone(),
            value_db: score.value_db,
            support_bins: score.support_bins,
            weighting: format!(
                "{} {}",
                definition.weighting, definition.measure_version
            ),
            aggregation_order: definition.aggregation_order.as_str().to_string(),
            uncertainty_ci95_db: aggregate_gain_ci95_db,
        },
        None => {
            violations.push(String::from("worst_supported_seat_unavailable"));
            WorstSeatReport {
                seat: String::from("unsupported"),
                value_db: f64::NAN,
                support_bins: 0,
                weighting: format!(
                    "{} {}",
                    definition.weighting, definition.measure_version
                ),
                aggregation_order: definition.aggregation_order.as_str().to_string(),
                uncertainty_ci95_db: aggregate_gain_ci95_db,
            }
        }
    };

    // Final sub/main sum: worst in-band dip of the sum below the louder
    // branch. Structural grid problems already aborted above.
    let crossover: Vec<usize> = support_indices(&sub_main.main.freq, sub_main.crossover_band_hz);
    let sub_main_detail;
    // This argument declares a required check, not an optional topology. A
    // missing/partial band must not certify an unassessed sum. Three bins are
    // the minimum for an interior dip; the grid must bracket both band edges.
    let full_band = sub_main.main.freq[0] <= sub_main.crossover_band_hz[0]
        && sub_main.main.freq[sub_main.main.freq.len() - 1] >= sub_main.crossover_band_hz[1];
    if crossover.len() < 3 || !full_band {
        violations.push("sub_main_sum_insufficient_support".to_string());
        sub_main_detail = format!(
            "sub/main sum unassessed (insufficient crossover support: {} bins, full band: {full_band})",
            crossover.len()
        );
    } else {
        let mut dip = f64::NEG_INFINITY;
        let mut non_finite = false;
        for index in &crossover {
            let (sub, main, summed) = (
                sub_main.sub.spl[*index],
                sub_main.main.spl[*index],
                sub_main.summed.spl[*index],
            );
            if !sub.is_finite() || !main.is_finite() || !summed.is_finite() {
                non_finite = true;
                break;
            }
            dip = dip.max(sub.max(main) - summed);
        }
        if non_finite {
            violations.push(String::from("sub_main_sum_non_finite"));
            sub_main_detail = String::from("sub/main sum non-finite in crossover band");
        } else {
            sub_main_detail = format!("worst in-band summation dip {dip:.2} dB");
            if dip > sub_main.max_cancellation_db {
                violations.push(format!(
                    "sub_main_sum_cancellation ({dip:.2} dB beyond {:.2} dB tolerance)",
                    sub_main.max_cancellation_db
                ));
            }
        }
    }

    for seat in &seats {
        violations.extend(seat.violations.iter().cloned());
    }
    violations.sort();
    violations.dedup();
    Ok(FinalValidationReport {
        definition: definition.clone(),
        aggregate_gain_db,
        aggregate_gain_ci95_db,
        worst_seat,
        residual_percentile_db,
        seats,
        sub_main_detail,
        violations,
    })
}

#[cfg(test)]
mod final_check_tests {
    use super::*;
    use ndarray::Array1;

    fn grid() -> Array1<f64> {
        // 20 Hz–20 kHz log grid, 64 bins.
        Array1::from(
            (0..64)
                .map(|index| 20.0 * (1000.0_f64).powf(index as f64 / 63.0))
                .collect::<Vec<f64>>(),
        )
    }

    fn curve(grid: &Array1<f64>, offset_db: f64) -> Curve {
        Curve {
            freq: grid.clone(),
            spl: grid.iter().map(|_| offset_db).collect(),
            ..Default::default()
        }
    }

    fn definition() -> SeatMetricDefinition {
        SeatMetricDefinition {
            support_band_hz: [50.0, 16_000.0],
            weighting: String::from("erb-rate"),
            measure_version: String::from("auditory-frequency-measure-v1"),
            percentile: 0.95,
            percentile_domain: PercentileDomain::Bins,
            aggregation_order: AggregationOrder::BinsThenSeats,
            uncertainty: String::from("seeded-bootstrap-ci95"),
            uncertainty_resamples: 200,
            uncertainty_seed: 7,
            max_residual_percentile_db: 6.0,
            permitted_degradation_db: 0.5,
            min_aggregate_gain_db: 1.0,
            max_pruning_loss_db: 0.5,
            min_support_bins: 8,
        }
    }

    fn sub_main_ok(grid: &Array1<f64>) -> SubMainSumEvidence {
        // Coherent sum: +6 dB over equal branches everywhere.
        SubMainSumEvidence {
            sub: curve(grid, 80.0),
            main: curve(grid, 80.0),
            summed: curve(grid, 86.0),
            crossover_band_hz: [60.0, 120.0],
            max_cancellation_db: 3.0,
        }
    }

    #[test]
    fn required_crossover_rejects_missing_sparse_and_partial_support() {
        let grid = grid();
        let pre = vec![curve(&grid, 84.0)];
        let post = vec![curve(&grid, 81.0)];
        let target = curve(&grid, 80.0);
        let refs = FinalReferences {
            baseline_post: pre.clone(),
            full_chain_post: None,
        };
        for band in [
            [30_000.0, 40_000.0],
            [grid[10], grid[10] + 0.01],
            [10.0, 100.0],
            [100.0, 30_000.0],
        ] {
            let mut sum = sub_main_ok(&grid);
            sum.crossover_band_hz = band;
            let report =
                evaluate_final_validation(&definition(), &pre, &post, &target, &refs, &sum)
                    .unwrap();
            assert!(!report.accepted(), "band {band:?}");
            assert!(
                report
                    .violations
                    .iter()
                    .any(|v| v == "sub_main_sum_insufficient_support")
            );
        }
    }

    #[test]
    fn final_validation_rejects_malformed_target_and_unknown_weighting() {
        let grid = grid();
        let pre = vec![curve(&grid, 84.0)];
        let post = vec![curve(&grid, 81.0)];
        let refs = FinalReferences {
            baseline_post: pre.clone(),
            full_chain_post: None,
        };
        let mut target = curve(&grid, 80.0);
        target.spl = vec![80.0].into();
        assert!(
            evaluate_final_validation(
                &definition(),
                &pre,
                &post,
                &target,
                &refs,
                &sub_main_ok(&grid)
            )
            .is_err()
        );
        let target = curve(&grid, 80.0);
        let mut def = definition();
        def.weighting = "unknown".into();
        assert!(
            evaluate_final_validation(&def, &pre, &post, &target, &refs, &sub_main_ok(&grid))
                .is_err()
        );
    }

    #[test]
    fn better_candidate_passes_with_gain_and_worst_seat() {
        let grid = grid();
        let target = curve(&grid, 80.0);
        // Pre is +4 dB off target everywhere; candidate halves it; the
        // baseline (identity) leaves pre untouched.
        let pre = vec![curve(&grid, 84.0), curve(&grid, 84.0)];
        let candidate = vec![curve(&grid, 82.0), curve(&grid, 83.0)];
        let references = FinalReferences {
            baseline_post: pre.clone(),
            full_chain_post: None,
        };
        let report = evaluate_final_validation(
            &definition(),
            &pre,
            &candidate,
            &target,
            &references,
            &sub_main_ok(&grid),
        )
        .unwrap();
        assert!(report.accepted(), "{:?}", report.violations);
        assert!((report.aggregate_gain_db - 1.5).abs() < 1e-9, "{}", report.aggregate_gain_db);
        // Seat 1 improved less: worst supported seat by residual.
        assert_eq!(report.worst_seat.seat, "seat-1");
        assert!(report.aggregate_gain_ci95_db[0] <= report.aggregate_gain_db);
        assert!(report.aggregate_gain_db <= report.aggregate_gain_ci95_db[1]);
        assert_eq!(report.worst_seat.aggregation_order, "bins-then-seats-v1");
    }

    #[test]
    fn regressed_seat_fails_within_budget_only() {
        let grid = grid();
        let target = curve(&grid, 80.0);
        let pre = vec![curve(&grid, 84.0), curve(&grid, 84.0)];
        // Seat 0 regresses 0.25 dB vs baseline: inside the 0.5 budget.
        // Seat 1 regresses 2 dB: outside.
        let candidate = vec![curve(&grid, 82.25), curve(&grid, 86.0)];
        let references = FinalReferences {
            baseline_post: vec![curve(&grid, 82.0), curve(&grid, 84.0)],
            full_chain_post: None,
        };
        let report = evaluate_final_validation(
            &definition(),
            &pre,
            &candidate,
            &target,
            &references,
            &sub_main_ok(&grid),
        )
        .unwrap();
        assert!(!report.accepted());
        assert!(report.seats[0].accepted);
        assert!(!report.seats[1].accepted);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.contains("seat_1_degraded_vs_baseline")),
            "{:?}",
            report.violations
        );
    }

    #[test]
    fn pruning_is_compared_against_the_accepted_full_chain() {
        let grid = grid();
        let target = curve(&grid, 80.0);
        let pre = vec![curve(&grid, 84.0)];
        // Pruned candidate keeps half the full chain's improvement.
        let candidate = vec![curve(&grid, 83.0)];
        let references = FinalReferences {
            baseline_post: pre.clone(),
            full_chain_post: Some(vec![curve(&grid, 82.0)]),
        };
        let report = evaluate_final_validation(
            &definition(),
            &pre,
            &candidate,
            &target,
            &references,
            &sub_main_ok(&grid),
        )
        .unwrap();
        assert!(!report.accepted());
        let loss = report.seats[0].pruning_loss_db.unwrap();
        assert!((loss - 1.0).abs() < 1e-9, "loss {loss}");
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.contains("pruning_loss")),
            "{:?}",
            report.violations
        );
    }

    #[test]
    fn sub_main_cancellation_is_caught() {
        let grid = grid();
        let target = curve(&grid, 80.0);
        let pre = vec![curve(&grid, 84.0)];
        let candidate = vec![curve(&grid, 82.0)];
        let references = FinalReferences {
            baseline_post: pre.clone(),
            full_chain_post: None,
        };
        // Sum collapses 10 dB below the branches in the crossover band:
        // relative-phase damage the isolated magnitudes hide.
        let damaged = SubMainSumEvidence {
            sub: curve(&grid, 80.0),
            main: curve(&grid, 80.0),
            summed: curve(&grid, 70.0),
            crossover_band_hz: [60.0, 120.0],
            max_cancellation_db: 3.0,
        };
        let report = evaluate_final_validation(
            &definition(),
            &pre,
            &candidate,
            &target,
            &references,
            &damaged,
        )
        .unwrap();
        assert!(!report.accepted());
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.contains("sub_main_sum_cancellation")),
            "{:?}",
            report.violations
        );
    }

    #[test]
    fn seats_domain_percentile_and_bad_shapes() {
        let grid = grid();
        let target = curve(&grid, 80.0);
        let pre = vec![curve(&grid, 84.0), curve(&grid, 90.0)];
        let candidate = vec![curve(&grid, 82.0), curve(&grid, 82.0)];
        let references = FinalReferences {
            baseline_post: pre.clone(),
            full_chain_post: None,
        };
        let mut seats_domain = definition();
        seats_domain.percentile_domain = PercentileDomain::Seats;
        let report = evaluate_final_validation(
            &seats_domain,
            &pre,
            &candidate,
            &target,
            &references,
            &sub_main_ok(&grid),
        )
        .unwrap();
        assert!(report.accepted(), "{:?}", report.violations);
        // Structural problems abort instead of guessing.
        let mut short = candidate.clone();
        short.pop();
        assert!(
            evaluate_final_validation(
                &definition(),
                &pre,
                &short,
                &target,
                &references,
                &sub_main_ok(&grid)
            )
            .is_err()
        );
        // Definitions validate before numbers run.
        let mut bad = definition();
        bad.percentile = 1.0;
        assert!(
            evaluate_final_validation(
                &bad,
                &pre,
                &candidate,
                &target,
                &references,
                &sub_main_ok(&grid)
            )
            .is_err()
        );
        bad = definition();
        bad.support_band_hz = [16_000.0, 50.0];
        assert!(
            evaluate_final_validation(
                &bad,
                &pre,
                &candidate,
                &target,
                &references,
                &sub_main_ok(&grid)
            )
            .is_err()
        );
    }
}
