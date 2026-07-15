use autoeq_measurements::{MeasurementQuality, MeasurementQualityReport};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::Curve;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionAcceptancePolicy {
    RuntimeSafety,
    CorrectableFixture,
    AlreadyGoodFixture,
    PoorMeasurementFixture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionDecision {
    Accepted,
    RevertedStage,
    IdentityFallback,
}

pub const RUNTIME_ACCEPTANCE_POLICY_VERSION: &str = "1.0.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeOutputClass {
    LowLatencyIir,
    Fir,
    Hybrid,
}

/// Versioned limits applied to production RoomEQ output.
///
/// The output class changes only temporal limits. Spectral, spatial, boost,
/// headroom, and realization limits are invariant across filter classes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeAcceptancePolicy {
    pub version: String,
    pub output_class: RuntimeOutputClass,
    pub max_post_p95_abs_residual_db: f64,
    pub max_post_worst_abs_residual_db: f64,
    pub max_worst_position_regression_db: f64,
    pub max_boost_db: f64,
    pub min_available_headroom_db: f64,
    pub max_latency_ms: f64,
    pub max_pre_ringing_energy_db: f64,
    pub max_induced_group_delay_rms_ms: f64,
    pub max_realization_error_db: f64,
}

impl RuntimeAcceptancePolicy {
    pub fn for_output_class(output_class: RuntimeOutputClass) -> Self {
        let (max_latency_ms, max_induced_group_delay_rms_ms) = match output_class {
            RuntimeOutputClass::LowLatencyIir => (10.0, 5.0),
            RuntimeOutputClass::Fir => (250.0, 25.0),
            RuntimeOutputClass::Hybrid => (100.0, 10.0),
        };
        Self {
            version: RUNTIME_ACCEPTANCE_POLICY_VERSION.to_string(),
            output_class,
            max_post_p95_abs_residual_db: 6.0,
            max_post_worst_abs_residual_db: 12.0,
            max_worst_position_regression_db: 0.25,
            max_boost_db: 12.0,
            min_available_headroom_db: -12.0,
            max_latency_ms,
            max_pre_ringing_energy_db: -20.0,
            max_induced_group_delay_rms_ms,
            max_realization_error_db: 0.25,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.version != RUNTIME_ACCEPTANCE_POLICY_VERSION {
            return Err(format!(
                "unsupported runtime acceptance policy version '{}'; expected '{}'",
                self.version, RUNTIME_ACCEPTANCE_POLICY_VERSION
            ));
        }
        let finite = [
            self.max_post_p95_abs_residual_db,
            self.max_post_worst_abs_residual_db,
            self.max_worst_position_regression_db,
            self.max_boost_db,
            self.min_available_headroom_db,
            self.max_latency_ms,
            self.max_pre_ringing_energy_db,
            self.max_induced_group_delay_rms_ms,
            self.max_realization_error_db,
        ]
        .into_iter()
        .all(f64::is_finite);
        if !finite
            || self.max_post_p95_abs_residual_db < 0.0
            || self.max_post_worst_abs_residual_db < 0.0
            || self.max_worst_position_regression_db < 0.0
            || self.max_boost_db < 0.0
            || self.max_latency_ms < 0.0
            || self.max_induced_group_delay_rms_ms < 0.0
            || self.max_realization_error_db < 0.0
        {
            return Err("runtime acceptance policy contains invalid limits".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RealizationQualityEvidence {
    pub evaluated_channels: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_abs_error_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failed_channels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CorrectionMetricSummary {
    pub pre_target_weighted_rms_db: f64,
    pub post_target_weighted_rms_db: f64,
    pub improvement_db: f64,
    pub improvement_ratio: f64,
    pub post_p95_abs_residual_db: f64,
    pub post_worst_abs_residual_db: f64,
    pub correction_rms_db: f64,
    pub max_abs_correction_db: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CorrectionAcceptanceReport {
    pub policy: CorrectionAcceptancePolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_policy: Option<RuntimeAcceptancePolicy>,
    pub decision: CorrectionDecision,
    pub accepted: bool,
    pub metrics: CorrectionMetricSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub violations: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reverted_stages: Vec<String>,
    /// Optional multi-position quality evidence. Runtime callers without
    /// held-out measurements keep this absent for wire compatibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acoustic_quality: Option<super::AcousticQualityScorecard>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realization_quality: Option<RealizationQualityEvidence>,
}

pub fn evaluate_correction_acceptance(
    pre: &Curve,
    post: &Curve,
    target: &Curve,
    measurement_quality: Option<&MeasurementQualityReport>,
    policy: CorrectionAcceptancePolicy,
) -> Result<CorrectionAcceptanceReport, String> {
    validate_shared_grid(pre, post, target)?;
    let pre_residual: Vec<f64> = pre
        .spl
        .iter()
        .zip(&target.spl)
        .map(|(value, target)| value - target)
        .collect();
    let post_residual: Vec<f64> = post
        .spl
        .iter()
        .zip(&target.spl)
        .map(|(value, target)| value - target)
        .collect();
    let correction: Vec<f64> = post
        .spl
        .iter()
        .zip(&pre.spl)
        .map(|(post, pre)| post - pre)
        .collect();
    if pre_residual
        .iter()
        .chain(&post_residual)
        .chain(&correction)
        .any(|value| !value.is_finite())
    {
        return Err("correction acceptance received non-finite curve data".to_string());
    }

    let pre_rms = rms(&pre_residual);
    let post_rms = rms(&post_residual);
    let improvement = pre_rms - post_rms;
    let improvement_ratio = if pre_rms > 1e-9 {
        improvement / pre_rms
    } else {
        0.0
    };
    let mut absolute_residual: Vec<f64> = post_residual.iter().map(|value| value.abs()).collect();
    absolute_residual.sort_by(f64::total_cmp);
    let p95_index = ((absolute_residual.len() - 1) as f64 * 0.95).ceil() as usize;
    let metrics = CorrectionMetricSummary {
        pre_target_weighted_rms_db: pre_rms,
        post_target_weighted_rms_db: post_rms,
        improvement_db: improvement,
        improvement_ratio,
        post_p95_abs_residual_db: absolute_residual[p95_index],
        post_worst_abs_residual_db: absolute_residual.last().copied().unwrap_or(0.0),
        correction_rms_db: rms(&correction),
        max_abs_correction_db: correction
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max),
    };

    let mut violations = Vec::new();
    match policy {
        CorrectionAcceptancePolicy::RuntimeSafety => {
            if post_rms > pre_rms + runtime_epsilon(pre_rms) {
                violations.push("target_weighted_rms_regressed".to_string());
            }
        }
        CorrectionAcceptancePolicy::CorrectableFixture => {
            if improvement < 0.25 || improvement_ratio < 0.10 {
                violations.push("meaningful_audible_improvement_not_reached".to_string());
            }
        }
        CorrectionAcceptancePolicy::AlreadyGoodFixture => {
            if metrics.correction_rms_db > 0.5 || metrics.max_abs_correction_db > 1.0 {
                violations.push("already_good_input_overcorrected".to_string());
            }
            if post_rms > pre_rms + runtime_epsilon(pre_rms) {
                violations.push("already_good_input_regressed".to_string());
            }
        }
        CorrectionAcceptancePolicy::PoorMeasurementFixture => {
            let scale = measurement_quality
                .map(|report| report.correction_depth_scale)
                .unwrap_or(0.35);
            if scale > 0.35 || metrics.max_abs_correction_db > 3.0 {
                violations.push("poor_measurement_correction_not_restrained".to_string());
            }
        }
    }
    if measurement_quality.is_some_and(|report| report.quality == MeasurementQuality::Unusable) {
        violations.push("measurement_unusable".to_string());
    }

    Ok(CorrectionAcceptanceReport {
        policy,
        runtime_policy: None,
        decision: if violations.is_empty() {
            CorrectionDecision::Accepted
        } else {
            CorrectionDecision::IdentityFallback
        },
        accepted: violations.is_empty(),
        metrics,
        violations,
        reverted_stages: Vec::new(),
        acoustic_quality: None,
        realization_quality: None,
    })
}

/// Apply the production acceptance policy to evidence derived from the final
/// canonical DSP graph. This is deliberately separate from the curve-only
/// fixture policies so runtime decisions cannot silently omit evidence that
/// was computed later in the pipeline.
pub fn enforce_runtime_acceptance_evidence(
    report: &mut CorrectionAcceptanceReport,
    acoustic_quality: super::AcousticQualityScorecard,
    realization_quality: RealizationQualityEvidence,
    policy: RuntimeAcceptancePolicy,
) -> Result<(), String> {
    policy.validate()?;
    if report.policy != CorrectionAcceptancePolicy::RuntimeSafety {
        return Err("runtime evidence requires the runtime_safety policy".to_string());
    }

    let mut violations = Vec::new();
    let partitions =
        std::iter::once(&acoustic_quality.training).chain(acoustic_quality.held_out.as_ref());
    let mut max_p95 = 0.0_f64;
    let mut max_worst = 0.0_f64;
    let mut worst_position_improvement = f64::INFINITY;
    for partition in partitions {
        max_p95 = max_p95.max(partition.post_p95_abs_residual_db);
        max_worst = max_worst.max(partition.post_worst_abs_residual_db);
        worst_position_improvement =
            worst_position_improvement.min(partition.worst_position_improvement_db);
    }
    if !acoustic_quality.finite {
        violations.push("acoustic_quality_non_finite".to_string());
    }
    if max_p95 > policy.max_post_p95_abs_residual_db {
        violations.push("post_p95_residual_limit_exceeded".to_string());
    }
    if max_worst > policy.max_post_worst_abs_residual_db {
        violations.push("post_worst_residual_limit_exceeded".to_string());
    }
    if worst_position_improvement < -policy.max_worst_position_regression_db {
        violations.push("worst_position_regressed".to_string());
    }
    if acoustic_quality.max_boost_db > policy.max_boost_db {
        violations.push("max_boost_limit_exceeded".to_string());
    }
    if acoustic_quality
        .temporal
        .available_headroom_db
        .is_some_and(|value| value < policy.min_available_headroom_db)
    {
        violations.push("headroom_limit_exceeded".to_string());
    }
    if acoustic_quality
        .temporal
        .latency_ms
        .is_some_and(|value| value > policy.max_latency_ms)
    {
        violations.push("latency_limit_exceeded".to_string());
    }
    if acoustic_quality
        .temporal
        .pre_ringing_energy_db
        .is_some_and(|value| value > policy.max_pre_ringing_energy_db)
    {
        violations.push("pre_ringing_limit_exceeded".to_string());
    }
    if acoustic_quality
        .induced_group_delay_rms_ms
        .is_some_and(|value| value > policy.max_induced_group_delay_rms_ms)
    {
        violations.push("induced_group_delay_limit_exceeded".to_string());
    }
    if realization_quality
        .max_abs_error_db
        .is_some_and(|value| value > policy.max_realization_error_db)
    {
        violations.push("realization_error_limit_exceeded".to_string());
    }
    if realization_quality.evaluated_channels == 0
        || realization_quality.max_abs_error_db.is_none()
        || !realization_quality.failed_channels.is_empty()
    {
        violations.push("realization_incomplete".to_string());
    }

    let runtime_violated = !violations.is_empty();
    report.runtime_policy = Some(policy);
    report.acoustic_quality = Some(acoustic_quality);
    report.realization_quality = Some(realization_quality);
    report.violations.extend(violations);
    report.violations.sort();
    report.violations.dedup();
    if runtime_violated {
        report.accepted = false;
        if report.decision == CorrectionDecision::Accepted {
            report.decision = CorrectionDecision::IdentityFallback;
        }
    }
    Ok(())
}

fn validate_shared_grid(pre: &Curve, post: &Curve, target: &Curve) -> Result<(), String> {
    for (name, curve) in [("pre", pre), ("post", post), ("target", target)] {
        curve.validate(name).map_err(|error| error.to_string())?;
    }
    if pre.freq.len() != post.freq.len()
        || pre.freq.len() != target.freq.len()
        || pre
            .freq
            .iter()
            .zip(&post.freq)
            .any(|(a, b)| (a - b).abs() > 1e-9)
        || pre
            .freq
            .iter()
            .zip(&target.freq)
            .any(|(a, b)| (a - b).abs() > 1e-9)
    {
        return Err(
            "correction acceptance requires explicitly aligned frequency grids".to_string(),
        );
    }
    Ok(())
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|value| value * value).sum::<f64>() / values.len().max(1) as f64).sqrt()
}

fn runtime_epsilon(pre_rms: f64) -> f64 {
    (pre_rms.abs() * 1e-4).max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn curve(spl: &[f64]) -> Curve {
        Curve {
            freq: Array1::from(vec![20.0, 100.0, 1000.0, 10_000.0]),
            spl: Array1::from(spl.to_vec()),
            ..Default::default()
        }
    }

    #[test]
    fn correction_acceptance_correctable_fixture_requires_meaningful_margin() {
        let target = curve(&[0.0; 4]);
        let pre = curve(&[4.0, -4.0, 3.0, -3.0]);
        let improved = curve(&[1.0, -1.0, 0.5, -0.5]);
        let unchanged = pre.clone();
        assert!(
            evaluate_correction_acceptance(
                &pre,
                &improved,
                &target,
                None,
                CorrectionAcceptancePolicy::CorrectableFixture,
            )
            .unwrap()
            .accepted
        );
        assert!(
            !evaluate_correction_acceptance(
                &pre,
                &unchanged,
                &target,
                None,
                CorrectionAcceptancePolicy::CorrectableFixture,
            )
            .unwrap()
            .accepted
        );
    }

    #[test]
    fn correction_acceptance_already_good_fixture_rejects_excess_correction() {
        let target = curve(&[0.0; 4]);
        let pre = curve(&[0.1, -0.1, 0.0, 0.0]);
        let post = curve(&[2.0, -2.0, 2.0, -2.0]);
        let report = evaluate_correction_acceptance(
            &pre,
            &post,
            &target,
            None,
            CorrectionAcceptancePolicy::AlreadyGoodFixture,
        )
        .unwrap();
        assert!(!report.accepted);
        assert!(
            report
                .violations
                .iter()
                .any(|value| value.contains("overcorrected"))
        );
    }

    #[test]
    fn correction_acceptance_runtime_safety_rejects_regression_and_mismatched_grids() {
        let target = curve(&[0.0; 4]);
        let pre = curve(&[1.0, -1.0, 1.0, -1.0]);
        let post = curve(&[2.0, -2.0, 2.0, -2.0]);
        assert!(
            !evaluate_correction_acceptance(
                &pre,
                &post,
                &target,
                None,
                CorrectionAcceptancePolicy::RuntimeSafety,
            )
            .unwrap()
            .accepted
        );
        let mut mismatched = post;
        mismatched.freq[1] = 110.0;
        assert!(
            evaluate_correction_acceptance(
                &pre,
                &mismatched,
                &target,
                None,
                CorrectionAcceptancePolicy::RuntimeSafety,
            )
            .is_err()
        );
    }

    #[test]
    fn correction_metrics_are_computed_relative_to_the_target() {
        let target = curve(&[10.0, 10.0, 10.0, 10.0]);
        let pre = curve(&[11.0, 8.0, 13.0, 6.0]);
        let post = curve(&[10.5, 9.0, 11.0, 8.0]);
        let report = evaluate_correction_acceptance(
            &pre,
            &post,
            &target,
            None,
            CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .expect("acceptance report");

        let expected_pre_rms = 7.5_f64.sqrt();
        let expected_post_rms = 1.25;
        assert!((report.metrics.pre_target_weighted_rms_db - expected_pre_rms).abs() < 1e-12);
        assert!((report.metrics.post_target_weighted_rms_db - expected_post_rms).abs() < 1e-12);
        assert!(
            (report.metrics.improvement_db - (expected_pre_rms - expected_post_rms)).abs() < 1e-12
        );
    }

    fn runtime_scorecard() -> super::super::AcousticQualityScorecard {
        let partition = super::super::QualityPartitionMetrics {
            curve_count: 2,
            pre_weighted_rms_median_db: 4.0,
            post_weighted_rms_median_db: 2.0,
            improvement_median_db: 2.0,
            worst_position_improvement_db: 1.0,
            pre_p95_abs_residual_db: 6.0,
            post_p95_abs_residual_db: 3.0,
            post_worst_abs_residual_db: 5.0,
            mean_normalized_seat_spread_db: 1.0,
            max_normalized_seat_spread_db: 2.0,
            bass_post_weighted_rms_db: None,
            upper_post_weighted_rms_db: None,
            bass_pre_modal_roughness_db_per_octave2: None,
            bass_post_modal_roughness_db_per_octave2: None,
            bass_modal_roughness_improvement_db_per_octave2: None,
        };
        super::super::AcousticQualityScorecard {
            training: partition,
            held_out: None,
            correction_rms_db: 2.0,
            max_boost_db: 4.0,
            max_cut_db: -6.0,
            induced_group_delay_rms_ms: Some(1.0),
            temporal: super::super::TemporalQualityEvidence {
                pre_ringing_energy_db: Some(-40.0),
                latency_ms: Some(5.0),
                available_headroom_db: Some(-4.0),
            },
            evaluated_band_hz: [20.0, 20_000.0],
            measurement_overlap_hz: [20.0, 20_000.0],
            finite: true,
        }
    }

    #[test]
    fn runtime_policy_accepts_complete_safe_evidence_and_records_version() {
        let target = curve(&[0.0; 4]);
        let pre = curve(&[4.0, -4.0, 3.0, -3.0]);
        let post = curve(&[1.0, -1.0, 0.5, -0.5]);
        let mut report = evaluate_correction_acceptance(
            &pre,
            &post,
            &target,
            None,
            CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .unwrap();
        let policy = RuntimeAcceptancePolicy::for_output_class(RuntimeOutputClass::Hybrid);
        let realization = RealizationQualityEvidence {
            evaluated_channels: 2,
            max_abs_error_db: Some(0.01),
            failed_channels: Vec::new(),
        };

        enforce_runtime_acceptance_evidence(
            &mut report,
            runtime_scorecard(),
            realization,
            policy.clone(),
        )
        .unwrap();

        assert!(report.accepted);
        assert_eq!(report.runtime_policy, Some(policy));
        assert!(report.realization_quality.is_some());
    }

    #[test]
    fn runtime_policy_rejects_each_unsafe_quality_dimension() {
        let target = curve(&[0.0; 4]);
        let pre = curve(&[4.0, -4.0, 3.0, -3.0]);
        let post = curve(&[1.0, -1.0, 0.5, -0.5]);
        let mut report = evaluate_correction_acceptance(
            &pre,
            &post,
            &target,
            None,
            CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .unwrap();
        let mut scorecard = runtime_scorecard();
        scorecard.training.post_p95_abs_residual_db = 20.0;
        scorecard.training.post_worst_abs_residual_db = 30.0;
        scorecard.training.worst_position_improvement_db = -2.0;
        scorecard.max_boost_db = 20.0;
        scorecard.induced_group_delay_rms_ms = Some(50.0);
        scorecard.temporal = super::super::TemporalQualityEvidence {
            pre_ringing_energy_db: Some(-5.0),
            latency_ms: Some(500.0),
            available_headroom_db: Some(-20.0),
        };
        let realization = RealizationQualityEvidence {
            evaluated_channels: 1,
            max_abs_error_db: Some(2.0),
            failed_channels: vec!["right".to_string()],
        };

        enforce_runtime_acceptance_evidence(
            &mut report,
            scorecard,
            realization,
            RuntimeAcceptancePolicy::for_output_class(RuntimeOutputClass::Hybrid),
        )
        .unwrap();

        assert!(!report.accepted);
        for expected in [
            "post_p95_residual_limit_exceeded",
            "post_worst_residual_limit_exceeded",
            "worst_position_regressed",
            "max_boost_limit_exceeded",
            "headroom_limit_exceeded",
            "latency_limit_exceeded",
            "pre_ringing_limit_exceeded",
            "induced_group_delay_limit_exceeded",
            "realization_error_limit_exceeded",
            "realization_incomplete",
        ] {
            assert!(
                report.violations.iter().any(|value| value == expected),
                "missing violation {expected}: {:?}",
                report.violations
            );
        }
    }

    #[test]
    fn runtime_policy_rejects_unknown_policy_versions() {
        let target = curve(&[0.0; 4]);
        let pre = curve(&[4.0, -4.0, 3.0, -3.0]);
        let post = curve(&[1.0, -1.0, 0.5, -0.5]);
        let mut report = evaluate_correction_acceptance(
            &pre,
            &post,
            &target,
            None,
            CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .unwrap();
        let mut policy =
            RuntimeAcceptancePolicy::for_output_class(RuntimeOutputClass::LowLatencyIir);
        policy.version = "2.0.0".to_string();

        let error = enforce_runtime_acceptance_evidence(
            &mut report,
            runtime_scorecard(),
            RealizationQualityEvidence {
                evaluated_channels: 2,
                max_abs_error_db: Some(0.0),
                failed_channels: Vec::new(),
            },
            policy,
        )
        .expect_err("unknown policy versions must fail closed");

        assert!(error.contains("unsupported runtime acceptance policy version"));
    }
}
