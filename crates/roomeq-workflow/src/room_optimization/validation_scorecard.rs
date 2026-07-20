use std::collections::HashMap;

use roomeq_model::{AutoeqError, Curve, Result};

use super::RoomOptimizationResult;

pub(super) fn attach_validation_scorecard(
    result: &mut RoomOptimizationResult,
    validation: &HashMap<String, Vec<Curve>>,
    sample_rate: f64,
) -> Result<()> {
    use roomeq_engine::quality::{
        CorrectionAcceptancePolicy, QualityEvaluationConfig, TemporalChannelEvidence,
        derive_temporal_quality_evidence, evaluate_acoustic_quality,
        evaluate_correction_acceptance,
    };

    let mut names: Vec<_> = result.channel_results.keys().cloned().collect();
    names.sort();
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
    if held_out_pre.is_empty() {
        return Ok(());
    }

    let min_freq_hz = training_pre
        .iter()
        .chain(&training_post)
        .chain(&held_out_pre)
        .chain(&held_out_post)
        .map(|curve| curve.freq[0])
        .fold(0.0_f64, f64::max);
    let max_freq_hz = training_pre
        .iter()
        .chain(&training_post)
        .chain(&held_out_pre)
        .chain(&held_out_post)
        .filter_map(|curve| curve.freq.last().copied())
        .fold(f64::INFINITY, f64::min);
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
            schroeder_hz: None,
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
        report.acoustic_quality = Some(scorecard);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_validation_populates_held_out_scorecard() {
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        let validation_curve = result.channel_results["left"].initial_curve.clone();
        let validation = HashMap::from([("left".to_string(), vec![validation_curve])]);

        attach_validation_scorecard(&mut result, &validation, 48_000.0)
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
    }
}
