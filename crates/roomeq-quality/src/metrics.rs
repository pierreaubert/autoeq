// Transfer-function metric evaluation.
use super::types::{AcousticOracle, CandidateTransfer, ImpulseEvidence, ProhibitedBehavior};
use num_complex::Complex64;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

const MAGNITUDE_FLOOR: f64 = 1e-12;

/// Versioned identifier for the frequency measure used by every integrated
/// oracle-path metric in this module.
///
/// `target_weighted_rms_db`, `p95_abs_residual_db`, and
/// `correction_energy_db2` are all integrated against the same normalized
/// log-frequency (trapezoid-cell) weights from [`log_frequency_weights`].
/// `worst_abs_residual_db` is intentionally NOT measure-integrated: it is a
/// bin maximum and stays grid-dependent by construction. `AcousticMetrics`
/// values must never be compared with ERB-rate-weighted acceptance metrics
/// (see `autoeq_core::AUDITORY_FREQUENCY_MEASURE_VERSION`) as if they were
/// the same metric; the measures differ even when the field names look alike.
pub const ORACLE_FREQUENCY_MEASURE_VERSION: &str = "log-frequency-trapezoid-v1";

#[derive(Debug, Clone, PartialEq)]
pub struct AcousticMetrics {
    /// RMS integrated with [`ORACLE_FREQUENCY_MEASURE_VERSION`] weights.
    pub target_weighted_rms_db: f64,
    /// Weighted (measure-integrated) 0.95 quantile under
    /// [`ORACLE_FREQUENCY_MEASURE_VERSION`]. Not comparable with the
    /// ERB-rate-weighted p95 reported by correction acceptance.
    pub p95_abs_residual_db: f64,
    /// Bin maximum, deliberately NOT measure-integrated. Grid-dependent by
    /// construction; compare only across identical grids.
    pub worst_abs_residual_db: f64,
    /// Mean square residual weighted by [`ORACLE_FREQUENCY_MEASURE_VERSION`].
    pub correction_energy_db2: f64,
    /// RMS over inter-bin intervals weighted by normalized log-frequency
    /// interval widths, so densifying a band does not move the metric.
    pub group_delay_residual_rms_ms: f64,
    pub max_boost_db: f64,
    pub pre_ringing_energy_db: Option<f64>,
    pub latency_ms: Option<f64>,
    pub finite: bool,
}

impl AcousticMetrics {
    /// Frequency measure every integrated field of this struct is expressed in.
    pub fn frequency_measure(&self) -> &'static str {
        ORACLE_FREQUENCY_MEASURE_VERSION
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceThresholds {
    pub max_weighted_rms_db: f64,
    pub max_p95_residual_db: f64,
    pub max_worst_residual_db: f64,
    pub max_correction_energy_db2: f64,
    pub max_group_delay_residual_rms_ms: f64,
}

impl Default for AcceptanceThresholds {
    /// Engineering-policy defaults, NOT listening-calibrated limits. They pin
    /// currently useful QA behavior; changing them changes what QA accepts.
    fn default() -> Self {
        Self {
            max_weighted_rms_db: 0.25,
            max_p95_residual_db: 0.5,
            max_worst_residual_db: 1.0,
            max_correction_energy_db2: 144.0,
            max_group_delay_residual_rms_ms: 0.1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceViolation {
    pub metric: String,
    pub observed: f64,
    pub limit: f64,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceReport {
    pub oracle_name: String,
    pub accepted: bool,
    pub metrics: AcousticMetrics,
    pub violations: Vec<AcceptanceViolation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DistributionSummary {
    pub count: usize,
    pub accepted_fraction: f64,
    pub median_weighted_rms_db: f64,
    pub p95_weighted_rms_db: f64,
    pub worst_tail_cvar_db: f64,
}

fn magnitude_db(value: Complex64) -> f64 {
    20.0 * value.norm().max(MAGNITUDE_FLOOR).log10()
}

/// Lower weighted quantile: the first value (in ascending order) whose
/// cumulative weight reaches `quantile` times the total positive, finite
/// weight. Returns `0.0` when no usable value/weight pair exists, mirroring
/// [`percentile`]. Unlike [`percentile`], duplicating or densifying bins in a
/// narrow band only adds that band's measure weight, so the result is stable
/// under grid refinement for a fixed underlying response.
pub(crate) fn weighted_percentile(values: &[f64], weights: &[f64], quantile: f64) -> f64 {
    let mut pairs: Vec<(f64, f64)> = values
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .filter(|(value, weight)| value.is_finite() && weight.is_finite() && *weight > 0.0)
        .collect();
    if pairs.is_empty() {
        return 0.0;
    }
    pairs.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let total: f64 = pairs.iter().map(|(_, weight)| *weight).sum();
    if !total.is_finite() || total <= 0.0 {
        return 0.0;
    }
    let target = quantile.clamp(0.0, 1.0) * total;
    let mut accumulated = 0.0;
    for (value, weight) in &pairs {
        accumulated += *weight;
        if accumulated >= target {
            return *value;
        }
    }
    pairs.last().map(|(value, _)| *value).unwrap_or(0.0)
}

pub(super) fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if (quantile - 0.5).abs() <= f64::EPSILON && values.len().is_multiple_of(2) {
        let upper = values.len() / 2;
        return (values[upper - 1] + values[upper]) * 0.5;
    }
    let index = ((values.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).ceil() as usize;
    values[index.min(values.len() - 1)]
}

pub(super) fn log_frequency_weights(frequencies_hz: &[f64]) -> Vec<f64> {
    let count = frequencies_hz.len();
    if count < 2 {
        return vec![1.0; count];
    }
    let mut weights = vec![0.0; count];
    for index in 0..count {
        let left = if index == 0 {
            frequencies_hz[0]
        } else {
            frequencies_hz[index - 1]
        };
        let right = if index + 1 == count {
            frequencies_hz[count - 1]
        } else {
            frequencies_hz[index + 1]
        };
        weights[index] = (right / left).ln().max(0.0);
    }
    let total = weights.iter().sum::<f64>();
    if total > 0.0 {
        weights.iter_mut().for_each(|weight| *weight /= total);
    }
    weights
}

fn unwrap_phase(values: &[Complex64]) -> Vec<f64> {
    let mut phases = values.iter().map(|value| value.arg()).collect::<Vec<_>>();
    for index in 1..phases.len() {
        let mut delta = phases[index] - phases[index - 1];
        while delta > PI {
            phases[index] -= 2.0 * PI;
            delta -= 2.0 * PI;
        }
        while delta < -PI {
            phases[index] += 2.0 * PI;
            delta += 2.0 * PI;
        }
    }
    phases
}

/// Group delay in milliseconds for each interior frequency interval.
pub fn group_delay_ms(frequencies_hz: &[f64], transfer: &[Complex64]) -> Vec<f64> {
    if frequencies_hz.len() != transfer.len() || transfer.len() < 2 {
        return Vec::new();
    }
    let phase = unwrap_phase(transfer);
    frequencies_hz
        .windows(2)
        .zip(phase.windows(2))
        .filter_map(|(frequency, phase)| {
            let delta_hz = frequency[1] - frequency[0];
            (delta_hz > 0.0).then(|| -(phase[1] - phase[0]) / (2.0 * PI * delta_hz) * 1000.0)
        })
        .collect()
}

fn impulse_metrics(evidence: ImpulseEvidence<'_>) -> (Option<f64>, Option<f64>) {
    if evidence.samples.is_empty()
        || !evidence.sample_rate.is_finite()
        || evidence.sample_rate <= 0.0
        || evidence.samples.iter().any(|sample| !sample.is_finite())
    {
        return (None, None);
    }
    let peak_index = evidence
        .samples
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.abs()
                .partial_cmp(&right.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0);
    let total_energy = evidence
        .samples
        .iter()
        .map(|sample| sample * sample)
        .sum::<f64>();
    let pre_energy = evidence.samples[..peak_index]
        .iter()
        .map(|sample| sample * sample)
        .sum::<f64>();
    let pre_ringing_db = if total_energy > 0.0 {
        Some(10.0 * (pre_energy / total_energy).max(1e-30).log10())
    } else {
        None
    };
    let latency_ms = Some(peak_index as f64 / evidence.sample_rate * 1000.0);
    (pre_ringing_db, latency_ms)
}

fn violation(
    metric: impl Into<String>,
    observed: f64,
    limit: f64,
    detail: impl Into<String>,
) -> AcceptanceViolation {
    AcceptanceViolation {
        metric: metric.into(),
        observed,
        limit,
        detail: detail.into(),
    }
}

/// Evaluate generated DSP against analytic complex ground truth.
///
/// Spectral metrics are integrated with [`ORACLE_FREQUENCY_MEASURE_VERSION`]
/// weights; the complex shape/transfer checks below (exact transfer
/// comparison, correction-region, null-boost, group-delay, latency, and
/// pre-ringing prohibitions) are per-bin or time-domain and unchanged.
/// Reported values must not be compared with ERB-rate-weighted acceptance
/// metrics as if they were the same metric.
pub fn evaluate_oracle(
    oracle: &AcousticOracle,
    candidate: CandidateTransfer<'_>,
    thresholds: &AcceptanceThresholds,
) -> Result<AcceptanceReport, String> {
    oracle.validate()?;
    if candidate.transfer.len() != oracle.expected_transfer.len() {
        return Err(format!(
            "candidate length {} does not match oracle length {}",
            candidate.transfer.len(),
            oracle.expected_transfer.len()
        ));
    }

    let finite = candidate
        .transfer
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite());
    let residual_transfer = candidate
        .transfer
        .iter()
        .zip(oracle.expected_transfer.iter())
        .map(|(&candidate, &expected)| candidate / expected)
        .collect::<Vec<_>>();
    let residual_db = residual_transfer
        .iter()
        .map(|&value| magnitude_db(value))
        .collect::<Vec<_>>();
    let absolute_residual_db = residual_db
        .iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    let frequencies = oracle.frequencies_hz.as_slice().unwrap_or(&[]);
    let weights = log_frequency_weights(frequencies);
    let target_weighted_rms_db = residual_db
        .iter()
        .zip(weights.iter())
        .map(|(residual, weight)| residual * residual * weight)
        .sum::<f64>()
        .sqrt();
    // Measure-integrated quantile under ORACLE_FREQUENCY_MEASURE_VERSION: same
    // weights as the RMS above, so densifying a narrow band cannot move p95
    // without changing the physical response.
    let p95_abs_residual_db = weighted_percentile(&absolute_residual_db, &weights, 0.95);
    let worst_abs_residual_db = absolute_residual_db.iter().copied().fold(0.0_f64, f64::max);
    let correction_db = residual_db.clone();
    // Weighted by the same frequency measure as the RMS: energy reductions
    // below used to be unweighted bin means, a different metric presented
    // under the same name.
    let correction_energy_db2 = correction_db
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| value * value * weight)
        .sum::<f64>();
    let max_boost_db = correction_db
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let group_delay = group_delay_ms(frequencies, &residual_transfer);
    // Weight each inter-bin interval by its normalized log-frequency width so
    // the group-delay RMS is grid-invariant like the spectral metrics.
    let group_delay_residual_rms_ms = if group_delay.is_empty() {
        0.0
    } else {
        let mut weight_total = 0.0;
        let mut weighted_squares = 0.0;
        for (index, value) in group_delay.iter().enumerate() {
            let width = frequencies
                .get(index)
                .zip(frequencies.get(index + 1))
                .map(|(&low, &high)| {
                    if low > 0.0 && high > low {
                        (high / low).ln().max(0.0)
                    } else {
                        0.0
                    }
                })
                .unwrap_or(0.0);
            weight_total += width;
            weighted_squares += value * value * width;
        }
        if weight_total > 0.0 {
            (weighted_squares / weight_total).sqrt()
        } else {
            (group_delay.iter().map(|value| value * value).sum::<f64>()
                / group_delay.len() as f64)
                .sqrt()
        }
    };
    let (pre_ringing_energy_db, latency_ms) = candidate
        .impulse
        .map(impulse_metrics)
        .unwrap_or((None, None));

    let metrics = AcousticMetrics {
        target_weighted_rms_db,
        p95_abs_residual_db,
        worst_abs_residual_db,
        correction_energy_db2,
        group_delay_residual_rms_ms,
        max_boost_db,
        pre_ringing_energy_db,
        latency_ms,
        finite,
    };
    let mut violations = Vec::new();
    if !finite {
        violations.push(violation(
            "finite_transfer",
            0.0,
            1.0,
            "candidate contains NaN or infinity",
        ));
    }
    if target_weighted_rms_db > thresholds.max_weighted_rms_db {
        violations.push(violation(
            "target_weighted_rms_db",
            target_weighted_rms_db,
            thresholds.max_weighted_rms_db,
            "target-weighted magnitude RMS exceeded",
        ));
    }
    if p95_abs_residual_db > thresholds.max_p95_residual_db {
        violations.push(violation(
            "p95_abs_residual_db",
            p95_abs_residual_db,
            thresholds.max_p95_residual_db,
            "95th-percentile magnitude residual exceeded",
        ));
    }
    if worst_abs_residual_db > thresholds.max_worst_residual_db {
        violations.push(violation(
            "worst_abs_residual_db",
            worst_abs_residual_db,
            thresholds.max_worst_residual_db,
            "worst magnitude residual exceeded",
        ));
    }
    if correction_energy_db2 > thresholds.max_correction_energy_db2 {
        violations.push(violation(
            "correction_energy_db2",
            correction_energy_db2,
            thresholds.max_correction_energy_db2,
            "correction energy exceeded",
        ));
    }
    if group_delay_residual_rms_ms > thresholds.max_group_delay_residual_rms_ms {
        violations.push(violation(
            "group_delay_residual_rms_ms",
            group_delay_residual_rms_ms,
            thresholds.max_group_delay_residual_rms_ms,
            "group-delay residual exceeded",
        ));
    }

    for prohibited in &oracle.prohibited_behaviors {
        match *prohibited {
            ProhibitedBehavior::NonFiniteTransfer if !finite => {}
            ProhibitedBehavior::NonFiniteTransfer => {}
            ProhibitedBehavior::CorrectionOutsideRegion { max_abs_db } => {
                let observed = oracle
                    .frequencies_hz
                    .iter()
                    .zip(correction_db.iter())
                    .filter(|(frequency, _)| {
                        **frequency < oracle.valid_correction_region_hz.0
                            || **frequency > oracle.valid_correction_region_hz.1
                    })
                    .map(|(_, value)| value.abs())
                    .fold(0.0_f64, f64::max);
                if observed > max_abs_db {
                    violations.push(violation(
                        "correction_outside_region_db",
                        observed,
                        max_abs_db,
                        "candidate changed bins outside the valid correction region",
                    ));
                }
            }
            ProhibitedBehavior::BoostIntoNull {
                center_hz,
                half_width_octaves,
                max_boost_db,
            } => {
                let low = center_hz / 2.0_f64.powf(half_width_octaves);
                let high = center_hz * 2.0_f64.powf(half_width_octaves);
                let observed = oracle
                    .frequencies_hz
                    .iter()
                    .zip(correction_db.iter())
                    .filter(|(frequency, _)| **frequency >= low && **frequency <= high)
                    .map(|(_, value)| *value)
                    .fold(f64::NEG_INFINITY, f64::max);
                if observed > max_boost_db {
                    violations.push(violation(
                        "boost_into_null_db",
                        observed,
                        max_boost_db,
                        format!("candidate boosted the null around {center_hz:.1} Hz"),
                    ));
                }
            }
            ProhibitedBehavior::GroupDelayResidual { max_rms_ms } => {
                if group_delay_residual_rms_ms > max_rms_ms {
                    violations.push(violation(
                        "fixture_group_delay_residual_rms_ms",
                        group_delay_residual_rms_ms,
                        max_rms_ms,
                        "fixture-specific group-delay limit exceeded",
                    ));
                }
            }
            ProhibitedBehavior::Latency { max_ms } => {
                if let Some(observed) = latency_ms
                    && observed > max_ms
                {
                    violations.push(violation(
                        "latency_ms",
                        observed,
                        max_ms,
                        "latency limit exceeded",
                    ));
                }
            }
            ProhibitedBehavior::PreRinging { max_energy_db } => {
                if let Some(observed) = pre_ringing_energy_db
                    && observed > max_energy_db
                {
                    violations.push(violation(
                        "pre_ringing_energy_db",
                        observed,
                        max_energy_db,
                        "pre-ringing energy limit exceeded",
                    ));
                }
            }
        }
    }

    Ok(AcceptanceReport {
        oracle_name: oracle.name.clone(),
        accepted: violations.is_empty(),
        metrics,
        violations,
    })
}

/// Exact transfer comparison used for export/runtime equivalence checks.
pub fn compare_complex_transfers(
    frequencies_hz: &[f64],
    expected: &[Complex64],
    actual: &[Complex64],
    max_magnitude_error_db: f64,
    max_phase_error_deg: f64,
) -> Result<(), String> {
    if frequencies_hz.len() != expected.len() || expected.len() != actual.len() {
        return Err("transfer comparison length mismatch".to_string());
    }
    for (index, ((&expected, &actual), &frequency)) in expected
        .iter()
        .zip(actual.iter())
        .zip(frequencies_hz.iter())
        .enumerate()
    {
        let ratio = actual / expected;
        let magnitude_error_db = magnitude_db(ratio).abs();
        let phase_error_deg = ratio.arg().to_degrees().abs();
        if magnitude_error_db > max_magnitude_error_db || phase_error_deg > max_phase_error_deg {
            return Err(format!(
                "transfer mismatch at bin {index} ({frequency:.2} Hz): magnitude {magnitude_error_db:.4} dB, phase {phase_error_deg:.4} deg"
            ));
        }
    }
    Ok(())
}

/// Mean of the worst `tail_fraction` values (CVaR-style error metric).
pub fn worst_tail_mean(values: &[f64], tail_fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let count = ((sorted.len() as f64 * tail_fraction.clamp(0.0, 1.0)).ceil() as usize)
        .max(1)
        .min(sorted.len());
    sorted[..count].iter().sum::<f64>() / count as f64
}

/// Cross-oracle summary over per-oracle weighted-RMS values. The median/p95
/// here are unweighted quantiles across oracles (each oracle already carries
/// its own frequency-measure integration), not frequency bins: do not compare
/// them with within-response p95 metrics.
pub fn summarize_distribution(reports: &[AcceptanceReport]) -> DistributionSummary {
    let values = reports
        .iter()
        .map(|report| report.metrics.target_weighted_rms_db)
        .collect::<Vec<_>>();
    DistributionSummary {
        count: reports.len(),
        accepted_fraction: if reports.is_empty() {
            0.0
        } else {
            reports.iter().filter(|report| report.accepted).count() as f64 / reports.len() as f64
        },
        median_weighted_rms_db: percentile(values.clone(), 0.5),
        p95_weighted_rms_db: percentile(values.clone(), 0.95),
        worst_tail_cvar_db: worst_tail_mean(&values, 0.05),
    }
}

/// Normalized max-min timbre spread across channels, averaged over bins.
pub fn normalized_timbre_spread_db(channels_db: &[Vec<f64>]) -> Option<f64> {
    let bins = channels_db.first()?.len();
    if channels_db.len() < 2
        || bins == 0
        || channels_db
            .iter()
            .any(|channel| channel.len() != bins || channel.iter().any(|value| !value.is_finite()))
    {
        return None;
    }
    let normalized = channels_db
        .iter()
        .map(|channel| {
            let mean = channel.iter().sum::<f64>() / bins as f64;
            channel.iter().map(|value| value - mean).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mean_spread = (0..bins)
        .map(|index| {
            let minimum = normalized
                .iter()
                .map(|channel| channel[index])
                .fold(f64::INFINITY, f64::min);
            let maximum = normalized
                .iter()
                .map(|channel| channel[index])
                .fold(f64::NEG_INFINITY, f64::max);
            maximum - minimum
        })
        .sum::<f64>()
        / bins as f64;
    Some(mean_spread)
}

/// Versioned identifier for seat/bin aggregation of per-seat per-bin
/// values. Linear means coincide for any weights, so the two orders
/// below always agree here — the order is still recorded on every
/// aggregate because it binds the moment a nonlinear reduction is used,
/// and because unnamed orders invite cross-measure confusion.
pub const SEAT_AGGREGATION_MEASURE_VERSION: &str = "seat-aggregation-v1";

/// Order of the two averaging stages in [`aggregate_seat_bin`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AggregationOrder {
    /// Weighted mean over bins first, then mean over seats.
    BinsThenSeats,
    /// Mean over seats per bin first, then weighted mean over bins.
    SeatsThenBins,
}

impl AggregationOrder {
    /// Stable string id recorded in reports and sidecars.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BinsThenSeats => "bins-then-seats-v1",
            Self::SeatsThenBins => "seats-then-bins-v1",
        }
    }
}

/// Aggregate per-seat per-bin values with explicit order.
///
/// `values` holds one row per seat, each `bins` long; `bin_weights` holds
/// the per-bin weights (ERB/log-frequency, matching the frequency
/// measure in use). Returns `None` on empty input, ragged rows, weight
/// mismatch, non-finite entries, or a non-positive weight sum — an
/// aggregate over undefined support is `None`, never zero.
pub fn aggregate_seat_bin(
    values: &[Vec<f64>],
    bin_weights: &[f64],
    order: AggregationOrder,
) -> Option<f64> {
    let bins = bin_weights.len();
    if values.is_empty()
        || bins == 0
        || values.iter().any(|row| row.len() != bins)
        || values.iter().any(|row| row.iter().any(|value| !value.is_finite()))
        || bin_weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return None;
    }
    let weight_sum: f64 = bin_weights.iter().sum();
    if !matches!(
        weight_sum.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    ) {
        return None;
    }
    let weighted_bin_mean = |row: &[f64]| {
        row.iter()
            .zip(bin_weights.iter())
            .map(|(value, weight)| value * weight)
            .sum::<f64>()
            / weight_sum
    };
    match order {
        AggregationOrder::BinsThenSeats => {
            let per_seat: Vec<f64> = values.iter().map(|row| weighted_bin_mean(row)).collect();
            Some(per_seat.iter().sum::<f64>() / per_seat.len() as f64)
        }
        AggregationOrder::SeatsThenBins => {
            let per_bin: Vec<f64> = (0..bins)
                .map(|bin| {
                    values.iter().map(|row| row[bin]).sum::<f64>() / values.len() as f64
                })
                .collect();
            Some(weighted_bin_mean(&per_bin))
        }
    }
}

/// One seat's scalar score with its frequency support.
#[derive(Debug, Clone, PartialEq)]
pub struct SeatScore {
    /// Seat id.
    pub seat: String,
    /// Score in dB (higher = worse).
    pub value_db: f64,
    /// Bins supporting the value.
    pub support_bins: usize,
}

/// Worst seat with at least `min_support_bins` bins of support.
///
/// Returns `None` when no seat qualifies — an unsupported worst seat is
/// absent from the report, never a zero or a guess. Ties resolve to the
/// last maximum in slice order (`max_by` semantics; deterministic, so
/// record the input order).
pub fn worst_supported_seat(
    scores: &[SeatScore],
    min_support_bins: usize,
) -> Option<&SeatScore> {
    scores
        .iter()
        .filter(|score| score.support_bins >= min_support_bins && score.value_db.is_finite())
        .max_by(|a, b| {
            a.value_db
                .partial_cmp(&b.value_db)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Deterministic bootstrap 95% interval for the mean of `values`.
///
/// Resamples with replacement `resamples` times under `seed` and returns
/// the 2.5/97.5 percentiles of resampled means. Same seed, same interval:
/// uncertainty itself is reproducible. Returns `None` on empty input or
/// zero resamples.
pub fn bootstrap_mean_ci95(
    values: &[f64],
    resamples: usize,
    seed: u64,
) -> Option<(f64, f64)> {
    if values.is_empty() || resamples == 0 || values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut rng = crate::SeededRng::new(seed);
    let mut means = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut sum = 0.0;
        for _ in 0..values.len() {
            sum += values[rng.next_below(values.len())];
        }
        means.push(sum / values.len() as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lower_index = ((0.025 * resamples as f64).floor() as usize) % resamples;
    let upper_index = (((0.975 * resamples as f64).ceil() as usize).saturating_sub(1)) % resamples;
    Some((means[lower_index], means[upper_index]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{identity_oracle, log_frequency_grid};
    use ndarray::Array1;

    fn permissive_thresholds() -> AcceptanceThresholds {
        AcceptanceThresholds {
            max_weighted_rms_db: 100.0,
            max_p95_residual_db: 100.0,
            max_worst_residual_db: 100.0,
            max_correction_energy_db2: 10_000.0,
            max_group_delay_residual_rms_ms: 100.0,
        }
    }

    #[test]
    fn weighted_rms_uses_normalized_log_frequency_weights() {
        let oracle = identity_oracle(Array1::from(vec![100.0, 200.0, 400.0, 800.0]));
        let residual_db = [0.0, 1.0, 2.0, 3.0];
        let candidate = residual_db
            .iter()
            .map(|value| Complex64::new(10.0_f64.powf(*value / 20.0), 0.0))
            .collect::<Vec<_>>();
        let report = evaluate_oracle(
            &oracle,
            CandidateTransfer {
                transfer: &candidate,
                impulse: None,
            },
            &permissive_thresholds(),
        )
        .expect("oracle report");
        let expected = (19.0_f64 / 6.0).sqrt();
        assert!((report.metrics.target_weighted_rms_db - expected).abs() < 1e-12);
    }

    /// Same physical response (1 dB floor, 8 dB narrow band around 1 kHz)
    /// sampled on a base log grid and on a grid densified inside the band.
    /// Weighted RMS/p95 must be (near-)invariant while the legacy unweighted
    /// bin percentile moves: that movement is a grid artifact, not acoustics.
    #[test]
    fn weighted_metrics_are_invariant_under_narrow_band_densification() {
        fn residual_db(frequency: f64) -> f64 {
            if (950.0..=1050.0).contains(&frequency) {
                8.0
            } else {
                1.0
            }
        }
        fn candidate_for(frequencies: &[f64]) -> Vec<Complex64> {
            frequencies
                .iter()
                .map(|frequency| {
                    Complex64::new(10.0_f64.powf(residual_db(*frequency) / 20.0), 0.0)
                })
                .collect()
        }
        // Fine base grid so cell-width edge effects at the band boundary are
        // small; densification then only redistributes the band's own measure.
        let base_grid: Vec<f64> = log_frequency_grid(2000, 20.0, 20_000.0).to_vec();
        let mut dense_grid: Vec<f64> = base_grid
            .iter()
            .copied()
            .filter(|frequency| *frequency < 950.0 || *frequency > 1050.0)
            .collect();
        for index in 0..200 {
            let fraction = index as f64 / 199.0;
            dense_grid.push(950.0 * (1050.0_f64 / 950.0).powf(fraction));
        }
        dense_grid.sort_by(f64::total_cmp);

        let report_for = |grid: Vec<f64>| {
            let oracle = identity_oracle(Array1::from(grid.clone()));
            let candidate = candidate_for(&grid);
            evaluate_oracle(
                &oracle,
                CandidateTransfer {
                    transfer: &candidate,
                    impulse: None,
                },
                &permissive_thresholds(),
            )
            .expect("oracle report")
        };
        let base = report_for(base_grid.clone());
        let dense = report_for(dense_grid.clone());
        assert_eq!(base.metrics.frequency_measure(), ORACLE_FREQUENCY_MEASURE_VERSION);
        assert!(
            (dense.metrics.target_weighted_rms_db - base.metrics.target_weighted_rms_db).abs()
                < 0.1,
            "weighted RMS moved under densification: {} -> {}",
            base.metrics.target_weighted_rms_db,
            dense.metrics.target_weighted_rms_db
        );
        assert!(
            (dense.metrics.p95_abs_residual_db - base.metrics.p95_abs_residual_db).abs() < 1e-9,
            "weighted p95 moved under densification: {} -> {}",
            base.metrics.p95_abs_residual_db,
            dense.metrics.p95_abs_residual_db
        );
        assert!(
            (dense.metrics.correction_energy_db2 - base.metrics.correction_energy_db2).abs()
                < 0.5,
            "weighted correction energy moved under densification"
        );
        // Documented grid artifact: the unweighted bin percentile jumps from
        // the 1 dB floor to the 8 dB band purely because the band now owns
        // more than 5 % of the bins.
        let unweighted = |grid: &[f64]| {
            percentile(
                grid.iter().map(|frequency| residual_db(*frequency)).collect(),
                0.95,
            )
        };
        assert!((unweighted(&base_grid) - 1.0).abs() < 1e-12);
        assert!((unweighted(&dense_grid) - 8.0).abs() < 1e-12);
    }

    /// Flatness bought with a deep narrow cut: lenient residual limits stay
    /// quiet but the measure-weighted correction energy fires.
    #[test]
    fn excessive_attenuation_is_flagged_by_correction_energy() {
        let grid = log_frequency_grid(25, 20.0, 20_000.0);
        let oracle = identity_oracle(grid.clone());
        let candidate = grid
            .iter()
            .map(|frequency| {
                let residual = if (400.0..=600.0).contains(frequency) {
                    -18.0
                } else {
                    0.1
                };
                Complex64::new(10.0_f64.powf(residual / 20.0), 0.0)
            })
            .collect::<Vec<_>>();
        let thresholds = AcceptanceThresholds {
            max_weighted_rms_db: 100.0,
            max_p95_residual_db: 100.0,
            max_worst_residual_db: 100.0,
            max_correction_energy_db2: 1.0,
            max_group_delay_residual_rms_ms: 100.0,
        };
        let report = evaluate_oracle(
            &oracle,
            CandidateTransfer {
                transfer: &candidate,
                impulse: None,
            },
            &thresholds,
        )
        .expect("oracle report");
        assert!(!report.accepted);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.metric == "correction_energy_db2")
        );
    }

    /// Adversarial final realization: spectral residual is zero yet the
    /// impulse rings before the main peak. Spectral metrics must stay quiet
    /// while the pre-ringing prohibition fires.
    #[test]
    fn low_spectral_error_with_ringing_is_rejected() {
        let grid = log_frequency_grid(17, 20.0, 20_000.0);
        let mut oracle = identity_oracle(grid);
        oracle.prohibited_behaviors = vec![ProhibitedBehavior::PreRinging {
            max_energy_db: -20.0,
        }];
        let candidate = vec![Complex64::new(1.0, 0.0); oracle.frequencies_hz.len()];
        let samples = vec![0.4, 0.4, 0.4, 0.4, 1.0, 0.1, 0.05];
        let report = evaluate_oracle(
            &oracle,
            CandidateTransfer {
                transfer: &candidate,
                impulse: Some(ImpulseEvidence {
                    samples: &samples,
                    sample_rate: 48_000.0,
                }),
            },
            &permissive_thresholds(),
        )
        .expect("oracle report");
        assert!(!report.accepted);
        assert!(report.metrics.target_weighted_rms_db.abs() < 1e-9);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.metric == "pre_ringing_energy_db")
        );
    }

    /// Adversarial final realization: magnitude is exactly right but a 2 ms
    /// pure delay hides in the phase. Spectral metrics must stay quiet while
    /// the group-delay residual fires.
    #[test]
    fn correct_magnitude_with_wrong_delay_is_rejected() {
        let grid = log_frequency_grid(17, 20.0, 20_000.0);
        let oracle = identity_oracle(grid.clone());
        let delay_seconds = 0.002;
        let candidate = grid
            .iter()
            .map(|frequency| {
                Complex64::from_polar(
                    1.0,
                    -2.0 * std::f64::consts::PI * frequency * delay_seconds,
                )
            })
            .collect::<Vec<_>>();
        let thresholds = AcceptanceThresholds {
            max_group_delay_residual_rms_ms: 0.1,
            ..permissive_thresholds()
        };
        let report = evaluate_oracle(
            &oracle,
            CandidateTransfer {
                transfer: &candidate,
                impulse: None,
            },
            &thresholds,
        )
        .expect("oracle report");
        assert!(!report.accepted);
        assert!(report.metrics.target_weighted_rms_db.abs() < 1e-9);
        assert!(report.metrics.p95_abs_residual_db.abs() < 1e-9);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.metric == "group_delay_residual_rms_ms")
        );
    }

    #[test]
    fn aggregation_orders_coincide_for_uniform_weights() {
        // Two seats, three bins, uniform weights: both orders reduce to
        // the grand mean (hand-computed: 40/6 = 20/3).
        let values = [vec![1.0, 2.0, 3.0], vec![4.0, 12.0, 18.0]];
        let weights = [1.0, 1.0, 1.0];
        let bins_first =
            aggregate_seat_bin(&values, &weights, AggregationOrder::BinsThenSeats).unwrap();
        let seats_first =
            aggregate_seat_bin(&values, &weights, AggregationOrder::SeatsThenBins).unwrap();
        assert!((bins_first - 20.0 / 3.0).abs() < 1e-12);
        assert!((seats_first - 20.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn aggregation_order_is_recorded_but_mean_is_order_free() {
        // Linear means coincide for any weights — both orders must agree
        // here (hand-computed 5.0). The order is still recorded on every
        // aggregate: it binds the moment a nonlinear reduction (median,
        // worst-bin) is introduced, and it keeps cross-measure
        // comparisons honest.
        let values = [vec![1.0, 2.0, 3.0], vec![4.0, 12.0, 18.0]];
        let weights = [3.0, 1.0, 1.0];
        let bins_first =
            aggregate_seat_bin(&values, &weights, AggregationOrder::BinsThenSeats).unwrap();
        let seats_first =
            aggregate_seat_bin(&values, &weights, AggregationOrder::SeatsThenBins).unwrap();
        // Bins-first: seat means (1*3+2+3)/5=1.6 and (12+12+18)/5=8.4, then (1.6+8.4)/2=5.0.
        assert!((bins_first - 5.0).abs() < 1e-12);
        // Seats-first: per-bin seat means (2.5, 7.0, 10.5), then (2.5*3+7+10.5)/5=5.0.
        assert!((seats_first - 5.0).abs() < 1e-12);
        assert_eq!(AggregationOrder::BinsThenSeats.as_str(), "bins-then-seats-v1");
        assert_eq!(AggregationOrder::SeatsThenBins.as_str(), "seats-then-bins-v1");
    }

    #[test]
    fn aggregation_rejects_undefined_support() {
        let values = [vec![1.0, 2.0]];
        assert!(aggregate_seat_bin(&[], &[1.0], AggregationOrder::BinsThenSeats).is_none());
        assert!(aggregate_seat_bin(&values, &[], AggregationOrder::BinsThenSeats).is_none());
        assert!(
            aggregate_seat_bin(&[vec![1.0]], &[1.0, 2.0], AggregationOrder::BinsThenSeats).is_none()
        );
        assert!(
            aggregate_seat_bin(&[vec![f64::NAN, 1.0]], &[1.0, 1.0], AggregationOrder::BinsThenSeats)
                .is_none()
        );
        assert!(
            aggregate_seat_bin(&values, &[0.0, 0.0], AggregationOrder::BinsThenSeats).is_none()
        );
    }

    #[test]
    fn worst_seat_needs_support() {
        let scores = [
            SeatScore { seat: String::from("a"), value_db: 3.0, support_bins: 50 },
            SeatScore { seat: String::from("b"), value_db: 5.0, support_bins: 4 },
            SeatScore { seat: String::from("c"), value_db: 4.0, support_bins: 60 },
        ];
        // "b" is worst but unsupported at min 10: "c" wins.
        assert_eq!(worst_supported_seat(&scores, 10).unwrap().seat, "c");
        // No minimum: "b" wins.
        assert_eq!(worst_supported_seat(&scores, 0).unwrap().seat, "b");
        // Impossible minimum: absent, not zero.
        assert!(worst_supported_seat(&scores, 1000).is_none());
        assert!(worst_supported_seat(&[], 0).is_none());
    }

    #[test]
    fn bootstrap_interval_is_deterministic_and_honest() {
        let values = [1.0, 2.0, 3.0, 4.0, 10.0];
        let first = bootstrap_mean_ci95(&values, 2000, 42).unwrap();
        let second = bootstrap_mean_ci95(&values, 2000, 42).unwrap();
        assert_eq!(first, second);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!(first.0 <= mean && mean <= first.1, "{first:?} vs {mean}");
        assert!(first.0 < first.1);
        assert!(bootstrap_mean_ci95(&[], 100, 1).is_none());
        assert!(bootstrap_mean_ci95(&values, 0, 1).is_none());
    }

    #[test]
    fn fixture_group_delay_limit_is_enforced_independently() {
        let frequencies = Array1::from(vec![100.0, 200.0, 300.0, 400.0]);
        let mut oracle = identity_oracle(frequencies.clone());
        oracle.prohibited_behaviors =
            vec![ProhibitedBehavior::GroupDelayResidual { max_rms_ms: 0.5 }];
        let candidate = frequencies
            .iter()
            .map(|frequency| {
                Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * frequency * 0.001)
            })
            .collect::<Vec<_>>();
        let report = evaluate_oracle(
            &oracle,
            CandidateTransfer {
                transfer: &candidate,
                impulse: None,
            },
            &permissive_thresholds(),
        )
        .expect("oracle report");
        assert!(
            report
                .violations
                .iter()
                .any(|violation| { violation.metric == "fixture_group_delay_residual_rms_ms" })
        );
    }
}
