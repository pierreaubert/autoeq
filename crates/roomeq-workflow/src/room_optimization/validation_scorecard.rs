use std::collections::HashMap;

use roomeq_model::{
    AcousticQualityScorecard, AutoeqError, CorrectionAcceptanceReport, Curve, Result,
};

use super::RoomOptimizationResult;

pub(super) fn attach_validation_scorecard(
    result: &mut RoomOptimizationResult,
    validation: &HashMap<String, Vec<Curve>>,
    sample_rate: f64,
    schroeder_hz: Option<f64>,
    processing_mode: roomeq_model::ProcessingMode,
) -> Result<()> {
    use roomeq_engine::quality::{
        CorrectionAcceptancePolicy, QualityEvaluationConfig, RuntimeAcceptancePolicy,
        TemporalChannelEvidence, derive_temporal_quality_evidence, evaluate_acoustic_quality,
        evaluate_correction_acceptance,
    };

    let mut names: Vec<_> = result.channel_results.keys().cloned().collect();
    names.sort();
    if names.is_empty() {
        return Err(AutoeqError::InvalidConfiguration {
            message: "cannot evaluate RoomEQ quality without output channels".to_string(),
        });
    }
    let training_pre: Vec<_> = names
        .iter()
        .map(|name| result.channel_results[name].initial_curve.clone())
        .collect();
    let training_post: Vec<_> = names
        .iter()
        .map(|name| result.channel_results[name].final_curve.clone())
        .collect();
    let mut held_out_pre = Vec::new();
    let mut held_out_post = Vec::new();
    for name in &names {
        let Some(curves) = validation.get(name) else {
            continue;
        };
        let chain = result
            .channels
            .get(name)
            .ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: format!("validation channel '{name}' is absent from pipeline output"),
            })?;
        for curve in curves {
            held_out_pre.push(curve.clone());
            held_out_post.push(crate::ctc::apply_channel_dsp_chain_to_curve(
                chain,
                curve,
                sample_rate,
            )?);
        }
    }
    let min_freq_hz = training_pre
        .iter()
        .chain(&training_post)
        .chain(&held_out_pre)
        .chain(&held_out_post)
        .map(|curve| curve.freq[0])
        .fold(0.0_f64, f64::max);
    // The evaluator aligns every curve pair on its own measured support.  Use
    // the widest available upper bound here so a short-band subwoofer cannot
    // hide a main speaker's measured upper band from the shared scorecard.
    // The common-overlap diagnostic remains conservative, while each
    // `useful_output` entry records the actual per-channel band.
    let max_freq_hz = training_pre
        .iter()
        .chain(&training_post)
        .chain(&held_out_pre)
        .chain(&held_out_post)
        .filter_map(|curve| curve.freq.last().copied())
        .fold(0.0_f64, f64::max);
    let temporal_channels: Vec<_> = names
        .iter()
        .map(|name| {
            let masking = result.channels[name].fir_temporal_masking.as_ref();
            TemporalChannelEvidence {
                pre_ringing_audible_db: masking.map(|metrics| metrics.pre_ringing_audible_db),
                main_time_ms: masking.map(|metrics| metrics.main_time_ms),
                fir_taps: result.channel_results[name]
                    .fir_coeffs
                    .as_ref()
                    .map(Vec::len),
            }
        })
        .collect();
    let temporal = derive_temporal_quality_evidence(
        &temporal_channels,
        &training_pre,
        &training_post,
        sample_rate,
    );
    let scorecard = evaluate_acoustic_quality(
        &training_pre,
        &training_post,
        &held_out_pre,
        &held_out_post,
        None,
        QualityEvaluationConfig {
            min_freq_hz,
            max_freq_hz,
            schroeder_hz,
            normalize_level: true,
        },
        temporal,
    )
    .map_err(|message| AutoeqError::InvalidConfiguration { message })?;

    if result.metadata.correction_acceptance.is_none() {
        let first = &names[0];
        let channel = &result.channel_results[first];
        let mean = channel.initial_curve.spl.mean().unwrap_or(0.0);
        let mut target = channel.initial_curve.clone();
        target.spl.fill(mean);
        target.phase = None;
        result.metadata.correction_acceptance = Some(
            evaluate_correction_acceptance(
                &channel.initial_curve,
                &channel.final_curve,
                &target,
                None,
                CorrectionAcceptancePolicy::RuntimeSafety,
            )
            .map_err(|message| AutoeqError::InvalidConfiguration { message })?,
        );
    }
    if let Some(report) = &mut result.metadata.correction_acceptance {
        if report.runtime_policy.is_none() {
            report.runtime_policy = Some(RuntimeAcceptancePolicy::for_output_class(
                processing_mode.runtime_output_class(),
            ));
        }
        align_acceptance_metrics_with_scorecard(report, &scorecard);
        let training_improvement = scorecard.training.improvement_median_db;
        let training_epsilon =
            (scorecard.training.pre_weighted_rms_median_db.abs() * 1e-4).max(1e-6);
        if !training_improvement.is_finite() || training_improvement < -training_epsilon {
            report
                .violations
                .push("target_weighted_rms_regressed".to_string());
            report.accepted = false;
        }
        // F05: the runtime gate enforces training + held-out partitions
        // together, but it runs before this scorecard is attached. A held-out
        // seat that regresses under the final chain must therefore be gated
        // here with the same worst-position budget, or the final decision
        // would certify a seat it never evaluated.
        if let Some(held_out) = scorecard.held_out.as_ref()
            && let Some(budget) = report
                .runtime_policy
                .as_ref()
                .map(|policy| policy.max_worst_position_regression_db)
            && held_out.worst_position_improvement_db < -budget
        {
            report
                .violations
                .push("worst_position_regressed".to_string());
            report.violations.sort();
            report.violations.dedup();
            report.accepted = false;
        }
        report.acoustic_quality = Some(scorecard);
        report.refresh_outcome();
    }
    Ok(())
}

/// Map the requested processing mode to the temporal budget that applies to
/// the realized graph.  The scorecard is shared by all modes, but a full FIR
/// or a mixed/excess-phase chain has a different latency/pre-ringing envelope
/// than a low-latency IIR chain.  Keep this mapping at the validation boundary
/// so a missing policy can never silently inherit the IIR defaults.
/// Keep the serialized scalar summary on the same realized-graph metric as
/// the detailed scorecard. A legacy one-channel summary is useful for older
/// callers, but cannot be allowed to contradict the multi-channel decision.
fn align_acceptance_metrics_with_scorecard(
    report: &mut CorrectionAcceptanceReport,
    scorecard: &AcousticQualityScorecard,
) {
    let training = &scorecard.training;
    report.metrics.pre_target_weighted_rms_db = training.pre_weighted_rms_median_db;
    report.metrics.post_target_weighted_rms_db = training.post_weighted_rms_median_db;
    report.metrics.improvement_db = training.improvement_median_db;
    report.metrics.improvement_ratio = if training.pre_weighted_rms_median_db.abs() > 1e-9 {
        training.improvement_median_db / training.pre_weighted_rms_median_db.abs()
    } else {
        0.0
    };
    report.metrics.post_p95_abs_residual_db = training.post_p95_abs_residual_db;
    report.metrics.post_worst_abs_residual_db = training.post_worst_abs_residual_db;
    report.metrics.correction_rms_db = scorecard.correction_rms_db;
    report.metrics.max_abs_correction_db =
        scorecard.max_boost_db.abs().max(scorecard.max_cut_db.abs());
}

/// Reconcile the legacy scalar summary after later final-seat replay has
/// enriched or replaced the detailed scorecard. This intentionally preserves
/// the scorecard (including `final_seats`) instead of re-running validation
/// and losing the replay evidence.
pub(super) fn align_report_metrics_to_scorecard(report: &mut CorrectionAcceptanceReport) {
    if let Some(scorecard) = report.acoustic_quality.clone() {
        align_acceptance_metrics_with_scorecard(report, &scorecard);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_policy_follows_processing_mode() {
        use roomeq_engine::quality::RuntimeOutputClass;
        use roomeq_model::ProcessingMode;

        assert_eq!(
            ProcessingMode::LowLatency.runtime_output_class(),
            RuntimeOutputClass::LowLatencyIir
        );
        assert_eq!(
            ProcessingMode::WarpedIir.runtime_output_class(),
            RuntimeOutputClass::LowLatencyIir
        );
        assert_eq!(
            ProcessingMode::KautzModal.runtime_output_class(),
            RuntimeOutputClass::LowLatencyIir
        );
        assert_eq!(
            ProcessingMode::PhaseLinear.runtime_output_class(),
            RuntimeOutputClass::Fir
        );
        assert_eq!(
            ProcessingMode::Hybrid.runtime_output_class(),
            RuntimeOutputClass::Hybrid
        );
        assert_eq!(
            ProcessingMode::MixedPhase.runtime_output_class(),
            RuntimeOutputClass::Hybrid
        );
    }

    #[test]
    fn held_out_seat_regression_is_exposed_in_final_decision() {
        // F05: the representative improves but a held-out seat worsens under
        // the final chain. The attached decision must expose that regression.
        use roomeq_model::{
            CorrectionAcceptancePolicy, CorrectionAcceptanceReport, CorrectionDecision,
            CorrectionMetricSummary, RuntimeAcceptancePolicy, RuntimeOutputClass,
        };
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        let grid = result.channel_results["left"].initial_curve.freq.clone();
        let rep_pre = roomeq_model::Curve {
            freq: grid.clone(),
            spl: grid
                .iter()
                .map(|f| 80.0 + 6.0 * (-((f - 120.0) / 25.0).powi(2)).exp())
                .collect(),
            ..Default::default()
        };
        let rep_post = roomeq_model::Curve {
            freq: grid.clone(),
            spl: ndarray::Array1::from_elem(grid.len(), 80.0),
            ..Default::default()
        };
        result
            .channel_results
            .get_mut("left")
            .unwrap()
            .initial_curve = rep_pre.clone();
        result.channel_results.get_mut("left").unwrap().final_curve = rep_post;
        let cut = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            120.0,
            48_000.0,
            1.0,
            -6.0,
        );
        result
            .channels
            .get_mut("left")
            .unwrap()
            .plugins
            .push(roomeq_engine::output::create_eq_plugin(&[cut]));
        // Flat seat: the deployed cut carves a null here, regressing it.
        let seat = roomeq_model::Curve {
            freq: grid.clone(),
            spl: ndarray::Array1::from_elem(grid.len(), 80.0),
            ..Default::default()
        };
        result.metadata.correction_acceptance = Some(CorrectionAcceptanceReport {
            policy: CorrectionAcceptancePolicy::RuntimeSafety,
            runtime_policy: Some(RuntimeAcceptancePolicy::for_output_class(
                RuntimeOutputClass::LowLatencyIir,
            )),
            decision: CorrectionDecision::Accepted,
            accepted: true,
            outcome: roomeq_model::RoomEqOutcome::Accepted,
            metrics: CorrectionMetricSummary {
                auditory_frequency_measure: "erb".to_string(),
                pre_target_weighted_rms_db: 3.0,
                post_target_weighted_rms_db: 0.5,
                improvement_db: 2.5,
                improvement_ratio: 6.0,
                post_p95_abs_residual_db: 1.0,
                post_worst_abs_residual_db: 2.0,
                correction_rms_db: 2.0,
                max_abs_correction_db: 6.0,
            },
            violations: Vec::new(),
            reverted_stages: Vec::new(),
            acoustic_quality: None,
            realization_quality: None,
        });
        let validation = HashMap::from([("left".to_string(), vec![seat])]);
        attach_validation_scorecard(
            &mut result,
            &validation,
            48_000.0,
            None,
            roomeq_model::ProcessingMode::LowLatency,
        )
        .expect("runtime validation");

        let report = result
            .metadata
            .correction_acceptance
            .as_ref()
            .expect("acceptance report");
        assert!(
            report
                .acoustic_quality
                .as_ref()
                .and_then(|quality| quality.held_out.as_ref())
                .is_some(),
            "held-out seat evidence must be attached"
        );
        assert!(
            report
                .violations
                .iter()
                .any(|v| v == "worst_position_regressed"),
            "seat regression must be exposed, violations={:?}, accepted={}",
            report.violations,
            report.accepted
        );
        assert!(
            !report.accepted,
            "final decision must not accept a regressed held-out seat"
        );
    }

    #[test]
    fn runtime_validation_populates_held_out_scorecard() {
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        let validation_curve = result.channel_results["left"].initial_curve.clone();
        let validation = HashMap::from([("left".to_string(), vec![validation_curve])]);

        attach_validation_scorecard(
            &mut result,
            &validation,
            48_000.0,
            Some(200.0),
            roomeq_model::ProcessingMode::LowLatency,
        )
        .expect("runtime validation");

        let quality = result
            .metadata
            .correction_acceptance
            .as_ref()
            .and_then(|report| report.acoustic_quality.as_ref())
            .expect("quality scorecard");
        assert_eq!(
            quality
                .held_out
                .as_ref()
                .expect("held-out metrics")
                .curve_count,
            1
        );
        let training = &quality.training;
        assert!(
            training.upper_pre_weighted_rms_db.is_some()
                && training.upper_post_weighted_rms_db.is_some(),
            "resolved Schroeder split must populate upper-band diagnostics"
        );
        let report = result
            .metadata
            .correction_acceptance
            .as_ref()
            .expect("acceptance report");
        assert_eq!(
            report.metrics.pre_target_weighted_rms_db,
            training.pre_weighted_rms_median_db
        );
        assert_eq!(
            report.metrics.post_target_weighted_rms_db,
            training.post_weighted_rms_median_db
        );
    }
}
