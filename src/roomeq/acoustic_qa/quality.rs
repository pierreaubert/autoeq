use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::metrics::{log_frequency_weights, percentile};
use crate::Curve;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QualityEvaluationConfig {
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schroeder_hz: Option<f64>,
    /// Remove each measurement's broadband level before scoring its shape.
    pub normalize_level: bool,
}

impl QualityEvaluationConfig {
    fn validate(self) -> Result<(), String> {
        if !self.min_freq_hz.is_finite()
            || !self.max_freq_hz.is_finite()
            || self.min_freq_hz <= 0.0
            || self.max_freq_hz <= self.min_freq_hz
        {
            return Err("invalid acoustic quality evaluation band".to_string());
        }
        if self
            .schroeder_hz
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err("invalid Schroeder frequency".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TemporalQualityEvidence {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_ringing_energy_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available_headroom_db: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TemporalChannelEvidence {
    pub pre_ringing_audible_db: Option<f64>,
    pub main_time_ms: Option<f64>,
    pub fir_taps: Option<usize>,
}

pub fn derive_temporal_quality_evidence(
    channels: &[TemporalChannelEvidence],
    pre: &[Curve],
    post: &[Curve],
    sample_rate: f64,
) -> TemporalQualityEvidence {
    let has_fir = channels.iter().any(|channel| {
        channel.pre_ringing_audible_db.is_some()
            || channel.main_time_ms.is_some()
            || channel.fir_taps.is_some()
    });
    let pre_ringing_energy_db = channels
        .iter()
        .filter_map(|channel| channel.pre_ringing_audible_db)
        .reduce(f64::max)
        .unwrap_or(if has_fir { -120.0 } else { -300.0 });
    let latency_ms = channels
        .iter()
        .map(|channel| {
            let explicit = channel.main_time_ms.unwrap_or(0.0);
            let linear_phase = channel
                .fir_taps
                .map(|taps| taps.saturating_sub(1) as f64 * 500.0 / sample_rate)
                .unwrap_or(0.0);
            explicit.max(linear_phase)
        })
        .fold(0.0_f64, f64::max);
    let max_boost_db = pre
        .iter()
        .zip(post)
        .flat_map(|(pre, post)| post.spl.iter().zip(&pre.spl))
        .map(|(post, pre)| post - pre)
        .fold(0.0_f64, f64::max);
    TemporalQualityEvidence {
        pre_ringing_energy_db: Some(pre_ringing_energy_db),
        latency_ms: Some(latency_ms),
        available_headroom_db: Some(-max_boost_db.max(0.0)),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QualityPartitionMetrics {
    pub curve_count: usize,
    pub pre_weighted_rms_median_db: f64,
    pub post_weighted_rms_median_db: f64,
    pub improvement_median_db: f64,
    pub pre_p95_abs_residual_db: f64,
    pub post_p95_abs_residual_db: f64,
    pub post_worst_abs_residual_db: f64,
    pub mean_normalized_seat_spread_db: f64,
    pub max_normalized_seat_spread_db: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_post_weighted_rms_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upper_post_weighted_rms_db: Option<f64>,
    /// Median RMS curvature of the residual below Schroeder frequency.
    ///
    /// This is measured in dB/octave² and distinguishes a response with
    /// narrow modal ripple from one with the same band RMS but a smooth tilt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_pre_modal_roughness_db_per_octave2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_post_modal_roughness_db_per_octave2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_modal_roughness_improvement_db_per_octave2: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AcousticQualityScorecard {
    pub training: QualityPartitionMetrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out: Option<QualityPartitionMetrics>,
    pub correction_rms_db: f64,
    pub max_boost_db: f64,
    pub max_cut_db: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub induced_group_delay_rms_ms: Option<f64>,
    pub temporal: TemporalQualityEvidence,
    pub evaluated_band_hz: [f64; 2],
    pub measurement_overlap_hz: [f64; 2],
    pub finite: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QualityGatePolicy {
    pub min_held_out_improvement_db: f64,
    pub max_p95_regression_db: f64,
    pub max_boost_db: f64,
}

impl Default for QualityGatePolicy {
    fn default() -> Self {
        Self {
            min_held_out_improvement_db: 0.1,
            max_p95_regression_db: 0.25,
            max_boost_db: 12.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QualityGateReport {
    pub passed: bool,
    pub enforced: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub violations: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub advisories: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QualityBaselinePartition {
    Training,
    HeldOut,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QualityBaselineMetrics {
    pub partition: QualityBaselinePartition,
    pub post_weighted_rms_median_db: f64,
    pub post_p95_abs_residual_db: f64,
    pub improvement_median_db: f64,
    pub max_boost_db: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub induced_group_delay_rms_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_modal_roughness_db_per_octave2: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QualityRegressionPolicy {
    pub max_weighted_rms_regression_db: f64,
    pub max_p95_regression_db: f64,
    pub max_improvement_loss_db: f64,
    pub max_group_delay_regression_ms: f64,
    pub max_modal_roughness_regression_db_per_octave2: f64,
    pub max_modal_roughness_regression_fraction: f64,
}

impl Default for QualityRegressionPolicy {
    fn default() -> Self {
        Self {
            max_weighted_rms_regression_db: 0.1,
            max_p95_regression_db: 0.25,
            max_improvement_loss_db: 0.1,
            max_group_delay_regression_ms: 0.1,
            max_modal_roughness_regression_db_per_octave2: 0.5,
            max_modal_roughness_regression_fraction: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QualityBaselineComparison {
    pub partition: QualityBaselinePartition,
    pub weighted_rms_delta_db: f64,
    pub p95_delta_db: f64,
    pub improvement_delta_db: f64,
    pub max_boost_delta_db: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_delay_delta_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_modal_roughness_delta_db_per_octave2: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub violations: Vec<String>,
}

pub fn evaluate_acoustic_quality(
    training_pre: &[Curve],
    training_post: &[Curve],
    held_out_pre: &[Curve],
    held_out_post: &[Curve],
    target: Option<&Curve>,
    config: QualityEvaluationConfig,
    temporal: TemporalQualityEvidence,
) -> Result<AcousticQualityScorecard, String> {
    config.validate()?;
    validate_pairs("training", training_pre, training_post)?;
    if !held_out_pre.is_empty() || !held_out_post.is_empty() {
        validate_pairs("held-out", held_out_pre, held_out_post)?;
    }
    if let Some(target) = target {
        target
            .validate("quality target")
            .map_err(|error| error.to_string())?;
    }

    let training = evaluate_partition(training_pre, training_post, target, config)?;
    let held_out = (!held_out_pre.is_empty())
        .then(|| evaluate_partition(held_out_pre, held_out_post, target, config))
        .transpose()?;

    let mut corrections = Vec::new();
    let mut group_delays = Vec::new();
    let all_pairs = training_pre
        .iter()
        .zip(training_post)
        .chain(held_out_pre.iter().zip(held_out_post));
    let mut overlap_low = config.min_freq_hz;
    let mut overlap_high = config.max_freq_hz;
    for (pre, post) in all_pairs {
        let samples = aligned_samples(pre, post, target, config)?;
        corrections.extend(samples.iter().map(|sample| sample.post - sample.pre));
        if let Some(value) = induced_group_delay_rms_ms(pre, post, config)? {
            group_delays.push(value);
        }
        overlap_low = overlap_low.max(pre.freq[0]).max(post.freq[0]);
        overlap_high = overlap_high
            .min(curve_max_freq(pre)?)
            .min(curve_max_freq(post)?);
    }
    let correction_rms_db = rms(&corrections);
    let max_boost_db = corrections
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let max_cut_db = corrections.iter().copied().fold(f64::INFINITY, f64::min);
    let induced_group_delay_rms_ms = (!group_delays.is_empty()).then(|| rms(&group_delays));
    let finite = [
        correction_rms_db,
        max_boost_db,
        max_cut_db,
        overlap_low,
        overlap_high,
    ]
    .into_iter()
    .all(f64::is_finite)
        && induced_group_delay_rms_ms.is_none_or(f64::is_finite);

    Ok(AcousticQualityScorecard {
        training,
        held_out,
        correction_rms_db,
        max_boost_db,
        max_cut_db,
        induced_group_delay_rms_ms,
        temporal,
        evaluated_band_hz: [config.min_freq_hz, config.max_freq_hz],
        measurement_overlap_hz: [overlap_low, overlap_high],
        finite,
    })
}

pub fn evaluate_quality_gate(
    scorecard: &AcousticQualityScorecard,
    policy: QualityGatePolicy,
    enforce: bool,
) -> QualityGateReport {
    let mut violations = Vec::new();
    let mut advisories = Vec::new();
    if !scorecard.finite {
        violations.push("non_finite_quality_metrics".to_string());
    }
    if scorecard.max_boost_db > policy.max_boost_db {
        violations.push("maximum_boost_exceeded".to_string());
    }
    if let Some(held_out) = &scorecard.held_out {
        if held_out.improvement_median_db < policy.min_held_out_improvement_db {
            violations.push("held_out_improvement_below_threshold".to_string());
        }
        if held_out.post_p95_abs_residual_db
            > held_out.pre_p95_abs_residual_db + policy.max_p95_regression_db
        {
            violations.push("held_out_p95_residual_regressed".to_string());
        }
    } else {
        advisories.push("held_out_measurements_unavailable".to_string());
    }
    QualityGateReport {
        passed: !enforce || violations.is_empty(),
        enforced: enforce,
        violations,
        advisories,
    }
}

pub fn compare_quality_to_baseline(
    scorecard: &AcousticQualityScorecard,
    baseline: &QualityBaselineMetrics,
    policy: QualityRegressionPolicy,
) -> Result<QualityBaselineComparison, String> {
    let current = match baseline.partition {
        QualityBaselinePartition::Training => &scorecard.training,
        QualityBaselinePartition::HeldOut => scorecard
            .held_out
            .as_ref()
            .ok_or_else(|| "held-out baseline requires held-out candidate metrics".to_string())?,
    };
    let weighted_rms_delta_db =
        current.post_weighted_rms_median_db - baseline.post_weighted_rms_median_db;
    let p95_delta_db = current.post_p95_abs_residual_db - baseline.post_p95_abs_residual_db;
    let improvement_delta_db = current.improvement_median_db - baseline.improvement_median_db;
    let max_boost_delta_db = scorecard.max_boost_db - baseline.max_boost_db;
    let group_delay_delta_ms = scorecard
        .induced_group_delay_rms_ms
        .zip(baseline.induced_group_delay_rms_ms)
        .map(|(current, baseline)| current - baseline);
    let bass_modal_roughness_delta_db_per_octave2 = current
        .bass_post_modal_roughness_db_per_octave2
        .zip(baseline.bass_modal_roughness_db_per_octave2)
        .map(|(current, baseline)| current - baseline);
    let mut violations = Vec::new();
    if weighted_rms_delta_db > policy.max_weighted_rms_regression_db {
        violations.push("baseline_weighted_rms_regressed".to_string());
    }
    if p95_delta_db > policy.max_p95_regression_db {
        violations.push("baseline_p95_residual_regressed".to_string());
    }
    if improvement_delta_db < -policy.max_improvement_loss_db {
        violations.push("baseline_improvement_margin_regressed".to_string());
    }
    if group_delay_delta_ms.is_some_and(|delta| delta > policy.max_group_delay_regression_ms) {
        violations.push("baseline_group_delay_regressed".to_string());
    }
    if bass_modal_roughness_delta_db_per_octave2.is_some_and(|delta| {
        delta > policy.max_modal_roughness_regression_db_per_octave2
            && delta
                > baseline
                    .bass_modal_roughness_db_per_octave2
                    .unwrap_or(0.0)
                    .abs()
                    * policy.max_modal_roughness_regression_fraction
    }) {
        violations.push("baseline_bass_modal_roughness_regressed".to_string());
    }
    Ok(QualityBaselineComparison {
        partition: baseline.partition,
        weighted_rms_delta_db,
        p95_delta_db,
        improvement_delta_db,
        max_boost_delta_db,
        group_delay_delta_ms,
        bass_modal_roughness_delta_db_per_octave2,
        violations,
    })
}

#[derive(Clone, Copy)]
struct Sample {
    frequency: f64,
    pre: f64,
    post: f64,
    target: f64,
}

fn validate_pairs(label: &str, pre: &[Curve], post: &[Curve]) -> Result<(), String> {
    if pre.is_empty() || pre.len() != post.len() {
        return Err(format!(
            "{label} quality evaluation needs equal, non-empty pre/post curve sets"
        ));
    }
    for (index, curve) in pre.iter().enumerate() {
        curve
            .validate(&format!("{label} pre curve {index}"))
            .map_err(|error| error.to_string())?;
    }
    for (index, curve) in post.iter().enumerate() {
        curve
            .validate(&format!("{label} post curve {index}"))
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn evaluate_partition(
    pre: &[Curve],
    post: &[Curve],
    target: Option<&Curve>,
    config: QualityEvaluationConfig,
) -> Result<QualityPartitionMetrics, String> {
    let mut pre_rms_values = Vec::new();
    let mut post_rms_values = Vec::new();
    let mut pre_abs = Vec::new();
    let mut post_abs = Vec::new();
    let mut bass_post = Vec::new();
    let mut upper_post = Vec::new();
    let mut bass_pre_modal_roughness = Vec::new();
    let mut bass_post_modal_roughness = Vec::new();
    for (pre, post) in pre.iter().zip(post) {
        let samples = aligned_samples(pre, post, target, config)?;
        let mut pre_residual: Vec<f64> = samples.iter().map(|s| s.pre - s.target).collect();
        let mut post_residual: Vec<f64> = samples.iter().map(|s| s.post - s.target).collect();
        if config.normalize_level {
            center(&mut pre_residual);
            center(&mut post_residual);
        }
        let frequencies: Vec<f64> = samples.iter().map(|s| s.frequency).collect();
        pre_rms_values.push(weighted_rms(&frequencies, &pre_residual));
        post_rms_values.push(weighted_rms(&frequencies, &post_residual));
        pre_abs.extend(pre_residual.iter().map(|value| value.abs()));
        post_abs.extend(post_residual.iter().map(|value| value.abs()));
        if let Some(split) = config.schroeder_hz {
            if let Some(value) = band_rms(&frequencies, &post_residual, config.min_freq_hz, split) {
                bass_post.push(value);
            }
            if let Some(value) = band_rms(&frequencies, &post_residual, split, config.max_freq_hz) {
                upper_post.push(value);
            }
            if let Some(value) =
                modal_roughness(&frequencies, &pre_residual, config.min_freq_hz, split)
            {
                bass_pre_modal_roughness.push(value);
            }
            if let Some(value) =
                modal_roughness(&frequencies, &post_residual, config.min_freq_hz, split)
            {
                bass_post_modal_roughness.push(value);
            }
        }
    }
    let (mean_spread, max_spread) = normalized_seat_spread(post, target, config)?;
    let pre_median = median(pre_rms_values);
    let post_median = median(post_rms_values);
    let bass_pre_modal_roughness_db_per_octave2 =
        (!bass_pre_modal_roughness.is_empty()).then(|| median(bass_pre_modal_roughness));
    let bass_post_modal_roughness_db_per_octave2 =
        (!bass_post_modal_roughness.is_empty()).then(|| median(bass_post_modal_roughness));
    let bass_modal_roughness_improvement_db_per_octave2 = bass_pre_modal_roughness_db_per_octave2
        .zip(bass_post_modal_roughness_db_per_octave2)
        .map(|(pre, post)| pre - post);
    Ok(QualityPartitionMetrics {
        curve_count: pre.len(),
        pre_weighted_rms_median_db: pre_median,
        post_weighted_rms_median_db: post_median,
        improvement_median_db: pre_median - post_median,
        pre_p95_abs_residual_db: percentile(pre_abs, 0.95),
        post_p95_abs_residual_db: percentile(post_abs.clone(), 0.95),
        post_worst_abs_residual_db: post_abs.into_iter().fold(0.0, f64::max),
        mean_normalized_seat_spread_db: mean_spread,
        max_normalized_seat_spread_db: max_spread,
        bass_post_weighted_rms_db: (!bass_post.is_empty()).then(|| median(bass_post)),
        upper_post_weighted_rms_db: (!upper_post.is_empty()).then(|| median(upper_post)),
        bass_pre_modal_roughness_db_per_octave2,
        bass_post_modal_roughness_db_per_octave2,
        bass_modal_roughness_improvement_db_per_octave2,
    })
}

fn modal_roughness(
    frequencies: &[f64],
    residual: &[f64],
    min_freq_hz: f64,
    max_freq_hz: f64,
) -> Option<f64> {
    let samples: Vec<(f64, f64)> = frequencies
        .iter()
        .copied()
        .zip(residual.iter().copied())
        .filter(|(frequency, value)| {
            frequency.is_finite()
                && value.is_finite()
                && *frequency >= min_freq_hz
                && *frequency <= max_freq_hz
                && *frequency > 0.0
        })
        .map(|(frequency, value)| (frequency.log2(), value))
        .collect();
    if samples.len() < 3 {
        return None;
    }

    let curvatures: Vec<f64> = samples
        .windows(3)
        .filter_map(|window| {
            let [(x0, y0), (x1, y1), (x2, y2)] = window else {
                return None;
            };
            let left_dx = x1 - x0;
            let right_dx = x2 - x1;
            if left_dx <= 0.0 || right_dx <= 0.0 {
                return None;
            }
            let left_slope = (y1 - y0) / left_dx;
            let right_slope = (y2 - y1) / right_dx;
            Some(2.0 * (right_slope - left_slope) / (left_dx + right_dx))
        })
        .collect();
    (!curvatures.is_empty()).then(|| rms(&curvatures))
}

fn aligned_samples(
    pre: &Curve,
    post: &Curve,
    target: Option<&Curve>,
    config: QualityEvaluationConfig,
) -> Result<Vec<Sample>, String> {
    let low = config
        .min_freq_hz
        .max(pre.freq[0])
        .max(post.freq[0])
        .max(target.map_or(0.0, |curve| curve.freq[0]));
    let target_high = target
        .map(curve_max_freq)
        .transpose()?
        .unwrap_or(f64::INFINITY);
    let high = config
        .max_freq_hz
        .min(curve_max_freq(pre)?)
        .min(curve_max_freq(post)?)
        .min(target_high);
    if high <= low {
        return Err("quality curves have no overlap in the evaluation band".to_string());
    }
    let samples: Vec<_> = pre
        .freq
        .iter()
        .copied()
        .zip(pre.spl.iter().copied())
        .filter(|(frequency, _)| *frequency >= low && *frequency <= high)
        .map(|(frequency, pre_value)| Sample {
            frequency,
            pre: pre_value,
            post: sample_log(post, frequency, &post.spl),
            target: target.map_or(0.0, |curve| sample_log(curve, frequency, &curve.spl)),
        })
        .collect();
    if samples.len() < 2 {
        return Err("quality evaluation overlap needs at least two bins".to_string());
    }
    Ok(samples)
}

fn normalized_seat_spread(
    curves: &[Curve],
    target: Option<&Curve>,
    config: QualityEvaluationConfig,
) -> Result<(f64, f64), String> {
    if curves.len() < 2 {
        return Ok((0.0, 0.0));
    }
    let reference = target.unwrap_or(&curves[0]);
    let low = curves
        .iter()
        .fold(config.min_freq_hz.max(reference.freq[0]), |value, curve| {
            value.max(curve.freq[0])
        });
    let high = curves.iter().try_fold(
        config.max_freq_hz.min(curve_max_freq(reference)?),
        |value, curve| Ok::<_, String>(value.min(curve_max_freq(curve)?)),
    )?;
    let frequencies: Vec<f64> = reference
        .freq
        .iter()
        .copied()
        .filter(|frequency| *frequency >= low && *frequency <= high)
        .collect();
    if frequencies.len() < 2 {
        return Err("seat curves have insufficient shared frequency overlap".to_string());
    }
    let mut seats: Vec<Vec<f64>> = curves
        .iter()
        .map(|curve| {
            frequencies
                .iter()
                .map(|frequency| sample_log(curve, *frequency, &curve.spl))
                .collect()
        })
        .collect();
    if config.normalize_level {
        seats.iter_mut().for_each(|values| center(values));
    }
    let spreads: Vec<f64> = (0..frequencies.len())
        .map(|index| {
            let values: Vec<f64> = seats.iter().map(|seat| seat[index]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            (values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64)
                .sqrt()
        })
        .collect();
    Ok((
        spreads.iter().sum::<f64>() / spreads.len() as f64,
        spreads.into_iter().fold(0.0, f64::max),
    ))
}

fn induced_group_delay_rms_ms(
    pre: &Curve,
    post: &Curve,
    config: QualityEvaluationConfig,
) -> Result<Option<f64>, String> {
    let (Some(pre_phase), Some(post_phase)) = (&pre.phase, &post.phase) else {
        return Ok(None);
    };
    let low = config.min_freq_hz.max(pre.freq[0]).max(post.freq[0]);
    let high = config
        .max_freq_hz
        .min(curve_max_freq(pre)?)
        .min(curve_max_freq(post)?);
    let frequencies: Vec<f64> = pre
        .freq
        .iter()
        .copied()
        .filter(|frequency| *frequency >= low && *frequency <= high)
        .collect();
    if frequencies.len() < 3 {
        return Ok(None);
    }
    let mut phase: Vec<f64> = frequencies
        .iter()
        .map(|frequency| {
            (sample_log(post, *frequency, post_phase) - sample_log(pre, *frequency, pre_phase))
                .to_radians()
        })
        .collect();
    for index in 1..phase.len() {
        while phase[index] - phase[index - 1] > std::f64::consts::PI {
            phase[index] -= 2.0 * std::f64::consts::PI;
        }
        while phase[index] - phase[index - 1] < -std::f64::consts::PI {
            phase[index] += 2.0 * std::f64::consts::PI;
        }
    }
    let delay_ms: Vec<f64> = frequencies
        .windows(2)
        .zip(phase.windows(2))
        .filter_map(|(frequency, phase)| {
            let delta_omega = 2.0 * std::f64::consts::PI * (frequency[1] - frequency[0]);
            (delta_omega > 0.0).then_some(-(phase[1] - phase[0]) / delta_omega * 1000.0)
        })
        .collect();
    Ok((!delay_ms.is_empty()).then(|| rms(&delay_ms)))
}

fn sample_log(curve: &Curve, frequency: f64, values: &ndarray::Array1<f64>) -> f64 {
    let mut low = 0usize;
    let mut high = curve.freq.len();
    while low < high {
        let middle = low + (high - low) / 2;
        match curve.freq[middle].total_cmp(&frequency) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return values[middle],
        }
    }
    let upper = low.min(curve.freq.len() - 1);
    let lower = upper.saturating_sub(1);
    if lower == upper {
        return values[lower];
    }
    let low_log = curve.freq[lower].ln();
    let high_log = curve.freq[upper].ln();
    let t = (frequency.ln() - low_log) / (high_log - low_log);
    values[lower] + t * (values[upper] - values[lower])
}

fn center(values: &mut [f64]) {
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    values.iter_mut().for_each(|value| *value -= mean);
}

fn weighted_rms(frequencies: &[f64], values: &[f64]) -> f64 {
    let weights = log_frequency_weights(frequencies);
    values
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * value * weight)
        .sum::<f64>()
        .sqrt()
}

fn band_rms(frequencies: &[f64], values: &[f64], low: f64, high: f64) -> Option<f64> {
    let selected: Vec<(f64, f64)> = frequencies
        .iter()
        .copied()
        .zip(values.iter().copied())
        .filter(|(frequency, _)| *frequency >= low && *frequency <= high)
        .collect();
    (selected.len() >= 2).then(|| {
        let (frequencies, values): (Vec<_>, Vec<_>) = selected.into_iter().unzip();
        weighted_rms(&frequencies, &values)
    })
}

fn median(values: Vec<f64>) -> f64 {
    percentile(values, 0.5)
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|value| value * value).sum::<f64>() / values.len().max(1) as f64).sqrt()
}

fn curve_max_freq(curve: &Curve) -> Result<f64, String> {
    curve
        .freq
        .last()
        .copied()
        .ok_or_else(|| "quality curve has no frequency bins".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn curve(freq: &[f64], spl: &[f64]) -> Curve {
        Curve {
            freq: Array1::from(freq.to_vec()),
            spl: Array1::from(spl.to_vec()),
            ..Default::default()
        }
    }

    fn config() -> QualityEvaluationConfig {
        QualityEvaluationConfig {
            min_freq_hz: 20.0,
            max_freq_hz: 20_000.0,
            schroeder_hz: Some(200.0),
            normalize_level: true,
        }
    }

    #[test]
    fn held_out_scorecard_detects_generalized_improvement() {
        let pre = curve(&[20.0, 100.0, 1000.0, 10_000.0], &[6.0, -4.0, 2.0, -2.0]);
        let post = curve(&[20.0, 100.0, 1000.0, 10_000.0], &[2.0, -1.0, 0.5, -0.5]);
        let held_pre = curve(&[20.0, 80.0, 800.0, 10_000.0], &[5.0, -3.0, 2.0, -1.0]);
        let held_post = curve(&[20.0, 80.0, 800.0, 10_000.0], &[2.0, -1.0, 0.5, -0.2]);
        let scorecard = evaluate_acoustic_quality(
            &[pre],
            &[post],
            &[held_pre],
            &[held_post],
            None,
            config(),
            TemporalQualityEvidence::default(),
        )
        .expect("scorecard");
        assert!(scorecard.training.improvement_median_db > 0.1);
        assert!(scorecard.held_out.unwrap().improvement_median_db > 0.1);
    }

    #[test]
    fn modal_roughness_distinguishes_ripple_from_log_frequency_tilt() {
        let frequencies = [20.0, 28.284, 40.0, 56.569, 80.0, 113.137, 160.0, 200.0];
        let tilt: Vec<f64> = frequencies
            .iter()
            .map(|frequency| 2.0 * (*frequency / 20.0_f64).log2())
            .collect();
        let ripple = [0.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 0.0];

        let smooth = modal_roughness(&frequencies, &tilt, 20.0, 200.0).expect("smooth metric");
        let rough = modal_roughness(&frequencies, &ripple, 20.0, 200.0).expect("rough metric");
        assert!(
            smooth < 1e-6,
            "linear log-frequency tilt should have zero curvature"
        );
        assert!(rough > 10.0, "modal ripple should have material curvature");
    }

    #[test]
    fn temporal_evidence_uses_worst_pre_ringing_and_fir_latency() {
        let pre = curve(&[20.0, 100.0, 1_000.0], &[0.0, 0.0, 0.0]);
        let post = curve(&[20.0, 100.0, 1_000.0], &[1.0, 2.0, 3.0]);
        let evidence = derive_temporal_quality_evidence(
            &[
                TemporalChannelEvidence {
                    pre_ringing_audible_db: Some(-42.0),
                    main_time_ms: Some(2.0),
                    fir_taps: Some(481),
                },
                TemporalChannelEvidence {
                    pre_ringing_audible_db: Some(-30.0),
                    main_time_ms: Some(1.0),
                    fir_taps: None,
                },
            ],
            &[pre],
            &[post],
            48_000.0,
        );
        assert_eq!(evidence.pre_ringing_energy_db, Some(-30.0));
        assert_eq!(evidence.latency_ms, Some(5.0));
        assert_eq!(evidence.available_headroom_db, Some(-3.0));
    }

    #[test]
    fn scorecard_handles_mismatched_sparse_grids_over_overlap() {
        let pre = curve(&[20.0, 100.0, 1000.0, 20_000.0], &[3.0, -2.0, 1.0, 0.0]);
        let post = curve(&[30.0, 300.0, 3000.0, 10_000.0], &[1.0, -0.5, 0.2, 0.0]);
        let target = curve(&[25.0, 250.0, 2500.0, 15_000.0], &[0.0; 4]);
        let scorecard = evaluate_acoustic_quality(
            &[pre],
            &[post],
            &[],
            &[],
            Some(&target),
            config(),
            TemporalQualityEvidence::default(),
        )
        .expect("overlap is explicitly aligned");
        assert_eq!(scorecard.measurement_overlap_hz, [30.0, 10_000.0]);
        assert!(scorecard.finite);
    }

    #[test]
    fn scorecard_is_invariant_to_global_gain_and_seat_order() {
        let left = curve(&[20.0, 50.0, 100.0, 200.0], &[4.0, -2.0, 3.0, -1.0]);
        let right = curve(&[20.0, 50.0, 100.0, 200.0], &[-3.0, 2.0, -1.0, 4.0]);
        let left_post = curve(&[20.0, 50.0, 100.0, 200.0], &[2.0, -1.0, 1.5, -0.5]);
        let right_post = curve(&[20.0, 50.0, 100.0, 200.0], &[-1.5, 1.0, -0.5, 2.0]);
        let baseline = evaluate_acoustic_quality(
            &[left.clone(), right.clone()],
            &[left_post.clone(), right_post.clone()],
            &[],
            &[],
            None,
            config(),
            TemporalQualityEvidence::default(),
        )
        .expect("baseline");
        let add_gain = |mut input: Curve| {
            input.spl += 12.0;
            input
        };
        let transformed = evaluate_acoustic_quality(
            &[add_gain(right), add_gain(left)],
            &[add_gain(right_post), add_gain(left_post)],
            &[],
            &[],
            None,
            config(),
            TemporalQualityEvidence::default(),
        )
        .expect("transformed");
        assert_eq!(baseline.training, transformed.training);
    }

    #[test]
    fn duplicated_measurements_do_not_change_partition_metrics() {
        let pre = curve(&[20.0, 50.0, 100.0, 200.0], &[4.0, -2.0, 3.0, -1.0]);
        let post = curve(&[20.0, 50.0, 100.0, 200.0], &[2.0, -1.0, 1.5, -0.5]);
        let single = evaluate_partition(
            std::slice::from_ref(&pre),
            std::slice::from_ref(&post),
            None,
            config(),
        )
        .expect("single");
        let duplicated =
            evaluate_partition(&[pre.clone(), pre], &[post.clone(), post], None, config())
                .expect("duplicated");
        assert_eq!(
            single.post_weighted_rms_median_db,
            duplicated.post_weighted_rms_median_db
        );
        assert_eq!(
            single.post_p95_abs_residual_db,
            duplicated.post_p95_abs_residual_db
        );
        assert_eq!(
            single.bass_post_modal_roughness_db_per_octave2,
            duplicated.bass_post_modal_roughness_db_per_octave2
        );
    }

    #[test]
    fn missing_phase_and_controlled_noise_are_handled_deterministically() {
        let mut pre = curve(
            &[20.0, 35.0, 50.0, 80.0, 120.0, 200.0],
            &[5.0, -3.0, 4.0, -2.0, 2.0, -1.0],
        );
        let mut post = curve(
            &[20.0, 35.0, 50.0, 80.0, 120.0, 200.0],
            &[2.0, -1.5, 2.0, -1.0, 1.0, -0.5],
        );
        pre.phase = Some(Array1::zeros(pre.freq.len()));
        post.phase = Some(Array1::zeros(post.freq.len()));
        let with_phase = evaluate_partition(
            std::slice::from_ref(&pre),
            std::slice::from_ref(&post),
            None,
            config(),
        )
        .expect("phase");
        pre.phase = None;
        post.phase = None;
        for (index, value) in post.spl.iter_mut().enumerate() {
            *value += (index as f64 * 1.618_033_988_75).sin() * 0.1;
        }
        let noisy = evaluate_partition(&[pre], &[post], None, config()).expect("noisy");
        assert!(noisy.post_weighted_rms_median_db.is_finite());
        assert!(noisy.improvement_median_db > 0.5);
        assert!(
            (noisy.post_weighted_rms_median_db - with_phase.post_weighted_rms_median_db).abs()
                < 0.15
        );
    }

    #[test]
    fn report_only_gate_records_but_does_not_fail_regression() {
        let curve = curve(&[20.0, 100.0, 1000.0], &[0.0, 1.0, 0.0]);
        let scorecard = evaluate_acoustic_quality(
            std::slice::from_ref(&curve),
            std::slice::from_ref(&curve),
            std::slice::from_ref(&curve),
            std::slice::from_ref(&curve),
            None,
            config(),
            TemporalQualityEvidence::default(),
        )
        .expect("scorecard");
        let report = evaluate_quality_gate(&scorecard, QualityGatePolicy::default(), false);
        assert!(report.passed);
        assert!(!report.violations.is_empty());
    }

    #[test]
    fn paired_baseline_comparison_detects_material_regression() {
        let pre = curve(&[20.0, 100.0, 1000.0], &[3.0, -2.0, 1.0]);
        let post = curve(&[20.0, 100.0, 1000.0], &[2.0, -1.5, 0.8]);
        let scorecard = evaluate_acoustic_quality(
            std::slice::from_ref(&pre),
            std::slice::from_ref(&post),
            std::slice::from_ref(&pre),
            std::slice::from_ref(&post),
            None,
            config(),
            TemporalQualityEvidence::default(),
        )
        .expect("scorecard");
        let current = scorecard.held_out.as_ref().expect("held-out metrics");
        let baseline = QualityBaselineMetrics {
            partition: QualityBaselinePartition::HeldOut,
            post_weighted_rms_median_db: current.post_weighted_rms_median_db - 0.2,
            post_p95_abs_residual_db: current.post_p95_abs_residual_db - 0.5,
            improvement_median_db: current.improvement_median_db + 0.2,
            max_boost_db: scorecard.max_boost_db,
            induced_group_delay_rms_ms: scorecard.induced_group_delay_rms_ms,
            bass_modal_roughness_db_per_octave2: current.bass_post_modal_roughness_db_per_octave2,
        };
        let comparison =
            compare_quality_to_baseline(&scorecard, &baseline, QualityRegressionPolicy::default())
                .expect("comparison");
        assert!(comparison.violations.len() >= 3);
    }
}
