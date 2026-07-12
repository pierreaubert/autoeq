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
    })
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
}
