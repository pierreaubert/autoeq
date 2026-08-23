use autoeq_core::{Curve, MeasurementQuality, MeasurementQualityReport};
pub use roomeq_model::{
    CorrectionAcceptancePolicy, CorrectionAcceptanceReport, CorrectionDecision,
    CorrectionMetricSummary, RUNTIME_ACCEPTANCE_POLICY_VERSION, RealizationQualityEvidence,
    RuntimeAcceptancePolicy, RuntimeOutputClass,
};

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

    let pre_rms = autoeq_core::erb_rate_weighted_rms(&pre.freq, &pre_residual)
        .ok_or_else(|| "could not compute ERB-rate-weighted pre-correction RMS".to_string())?;
    let post_rms = autoeq_core::erb_rate_weighted_rms(&post.freq, &post_residual)
        .ok_or_else(|| "could not compute ERB-rate-weighted post-correction RMS".to_string())?;
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
        auditory_frequency_measure: autoeq_core::AUDITORY_FREQUENCY_MEASURE_VERSION.to_string(),
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

/// Minimum p95 residual increase (dB) treated as a real distribution
/// regression by [`enforce_runtime_acceptance_evidence`]. Matches the
/// worst-position regression guard already present in the runtime policy.
const RESIDUAL_REGRESSION_TOLERANCE_DB: f64 = 0.25;

/// Relative p95 residual increase treated as a real distribution regression,
/// so the tolerance scales with the room's residual level (5 %, mirroring the
/// per-channel safety gate's regression-ratio band).
const RESIDUAL_REGRESSION_RELATIVE: f64 = 0.05;

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
    let mut max_pre_p95 = 0.0_f64;
    let mut max_worst = 0.0_f64;
    let mut worst_position_improvement = f64::INFINITY;
    for partition in partitions {
        max_pre_p95 = max_pre_p95.max(partition.pre_p95_abs_residual_db);
        max_p95 = max_p95.max(partition.post_p95_abs_residual_db);
        max_worst = max_worst.max(partition.post_worst_abs_residual_db);
        worst_position_improvement =
            worst_position_improvement.min(partition.worst_position_improvement_db);
    }
    if !acoustic_quality.finite {
        violations.push("acoustic_quality_non_finite".to_string());
    }
    // Absolute residual limits must not turn a beneficial correction into a
    // worse identity fallback merely because the uncorrected room already
    // exceeds the limit. The limit becomes a hard violation when the
    // correction also regresses the residual distribution; independently
    // measured worst-position regression remains guarded below. The p95/worst
    // statistics are computed on smoothed curves and wobble at the sub-dB
    // level between runs, and an absolute dB tolerance is meaningless across
    // rooms with wildly different residual scales: a 0.4 dB change matters on
    // a 4 dB residual but is noise on a 27 dB one. Require both a floor and a
    // relative (5 %) regression before calling the distribution degraded —
    // the same relative band the per-channel safety gate uses.
    let regression_tolerance =
        RESIDUAL_REGRESSION_TOLERANCE_DB.max(RESIDUAL_REGRESSION_RELATIVE * max_pre_p95);
    let residual_regressed = max_p95 > max_pre_p95 + regression_tolerance;
    if max_p95 > policy.max_post_p95_abs_residual_db && residual_regressed {
        violations.push("post_p95_residual_limit_exceeded".to_string());
    }
    if max_worst > policy.max_post_worst_abs_residual_db && residual_regressed {
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
    match acoustic_quality.temporal.pre_ringing_energy_db {
        Some(value) if value > policy.max_pre_ringing_energy_db => {
            violations.push("pre_ringing_limit_exceeded".to_string());
        }
        None => violations.push("pre_ringing_evidence_missing".to_string()),
        _ => {}
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

        let expected_pre_rms =
            autoeq_core::erb_rate_weighted_rms(&pre.freq, &[1.0, -2.0, 3.0, -4.0]).unwrap();
        let expected_post_rms =
            autoeq_core::erb_rate_weighted_rms(&post.freq, &[0.5, -1.0, 1.0, -2.0]).unwrap();
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
    fn runtime_policy_keeps_non_regressing_correction_when_room_exceeds_residual_limit() {
        let target = curve(&[0.0; 4]);
        let pre = curve(&[8.0, -8.0, 7.0, -7.0]);
        let post = curve(&[8.0, -8.0, 7.0, -7.0]);
        let mut report = evaluate_correction_acceptance(
            &pre,
            &post,
            &target,
            None,
            CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .unwrap();
        let mut scorecard = runtime_scorecard();
        scorecard.training.pre_p95_abs_residual_db = 20.0;
        scorecard.training.post_p95_abs_residual_db = 20.0;
        scorecard.training.post_worst_abs_residual_db = 30.0;

        enforce_runtime_acceptance_evidence(
            &mut report,
            scorecard,
            RealizationQualityEvidence {
                evaluated_channels: 2,
                max_abs_error_db: Some(0.01),
                failed_channels: Vec::new(),
            },
            RuntimeAcceptancePolicy::for_output_class(RuntimeOutputClass::LowLatencyIir),
        )
        .unwrap();

        assert!(report.accepted);
        assert!(
            !report
                .violations
                .iter()
                .any(|violation| violation.contains("residual_limit_exceeded"))
        );
    }

    #[test]
    fn runtime_policy_tolerates_small_relative_p95_wobble_on_bad_rooms() {
        let target = curve(&[0.0; 4]);
        let pre = curve(&[8.0, -8.0, 7.0, -7.0]);
        let post = curve(&[8.0, -8.0, 7.0, -7.0]);
        let mut report = evaluate_correction_acceptance(
            &pre,
            &post,
            &target,
            None,
            CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .unwrap();
        let mut scorecard = runtime_scorecard();
        // On an already-poor room (p95 residual ~27 dB), a sub-half-dB p95
        // wobble is measurement noise, not a distribution regression.
        scorecard.training.pre_p95_abs_residual_db = 27.2;
        scorecard.training.post_p95_abs_residual_db = 27.65;
        scorecard.training.post_worst_abs_residual_db = 30.0;

        enforce_runtime_acceptance_evidence(
            &mut report,
            scorecard,
            RealizationQualityEvidence {
                evaluated_channels: 2,
                max_abs_error_db: Some(0.01),
                failed_channels: Vec::new(),
            },
            RuntimeAcceptancePolicy::for_output_class(RuntimeOutputClass::LowLatencyIir),
        )
        .unwrap();

        assert!(
            !report
                .violations
                .iter()
                .any(|violation| violation.contains("residual_limit_exceeded")),
            "relative p95 wobble should not trip residual limits: {:?}",
            report.violations
        );
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
    fn runtime_policy_rejects_missing_pre_ringing_evidence() {
        let mut report = evaluate_correction_acceptance(
            &curve(&[0.0; 4]),
            &curve(&[0.0; 4]),
            &curve(&[0.0; 4]),
            None,
            CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .unwrap();
        let mut scorecard = runtime_scorecard();
        scorecard.temporal.pre_ringing_energy_db = None;
        enforce_runtime_acceptance_evidence(
            &mut report,
            scorecard,
            RealizationQualityEvidence {
                evaluated_channels: 1,
                max_abs_error_db: Some(0.0),
                failed_channels: Vec::new(),
            },
            RuntimeAcceptancePolicy::for_output_class(RuntimeOutputClass::Hybrid),
        )
        .expect("runtime evidence should validate");
        assert!(
            report
                .violations
                .iter()
                .any(|v| v == "pre_ringing_evidence_missing")
        );
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
