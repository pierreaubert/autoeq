use super::types::{AcousticOracle, CandidateTransfer, ImpulseEvidence, ProhibitedBehavior};
use num_complex::Complex64;
use std::f64::consts::PI;

const MAGNITUDE_FLOOR: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub struct AcousticMetrics {
    pub target_weighted_rms_db: f64,
    pub p95_abs_residual_db: f64,
    pub worst_abs_residual_db: f64,
    pub correction_energy_db2: f64,
    pub group_delay_residual_rms_ms: f64,
    pub max_boost_db: f64,
    pub pre_ringing_energy_db: Option<f64>,
    pub latency_ms: Option<f64>,
    pub finite: bool,
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

pub(super) fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
    let weights = log_frequency_weights(oracle.frequencies_hz.as_slice().unwrap_or(&[]));
    let target_weighted_rms_db = residual_db
        .iter()
        .zip(weights.iter())
        .map(|(residual, weight)| residual * residual * weight)
        .sum::<f64>()
        .sqrt();
    let p95_abs_residual_db = percentile(absolute_residual_db.clone(), 0.95);
    let worst_abs_residual_db = absolute_residual_db.iter().copied().fold(0.0_f64, f64::max);
    let correction_db = residual_db.clone();
    let correction_energy_db2 = correction_db.iter().map(|value| value * value).sum::<f64>()
        / correction_db.len().max(1) as f64;
    let max_boost_db = correction_db
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let group_delay = group_delay_ms(
        oracle.frequencies_hz.as_slice().unwrap_or(&[]),
        &residual_transfer,
    );
    let group_delay_residual_rms_ms = if group_delay.is_empty() {
        0.0
    } else {
        (group_delay.iter().map(|value| value * value).sum::<f64>() / group_delay.len() as f64)
            .sqrt()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomeq::acoustic_qa::identity_oracle;
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
