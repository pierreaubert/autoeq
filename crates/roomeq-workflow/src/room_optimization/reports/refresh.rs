use super::super::room_optimization_result::RoomOptimizationResult;
use super::super::room_optimization_result::routed_baseline_curve;
use super::super::*;
use super::build::build_bootstrap_uncertainty_report;
use super::build::build_perceptual_policy_report;
use super::misc::direct_early_late_correction_metrics;
use super::misc::final_score_band_for_channel;
use super::misc::recompute_curve_flatness_score;
use super::role::update_perceptual_metrics;

pub(in super::super) fn refresh_final_reports(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
    sample_rate: f64,
) {
    for ch_result in result.channel_results.values_mut() {
        let (score_min_freq, score_max_freq) =
            final_score_band_for_channel(config, &ch_result.name);
        let (topology_pre, topology_post) = (ch_result.pre_score, ch_result.post_score);
        if let Some(baseline) = result
            .channels
            .get(&ch_result.name)
            .and_then(|chain| routed_baseline_curve(chain, &ch_result.initial_curve, sample_rate))
        {
            ch_result.pre_score =
                recompute_curve_flatness_score(&baseline, score_min_freq, score_max_freq);
        }
        ch_result.post_score =
            recompute_curve_flatness_score(&ch_result.final_curve, score_min_freq, score_max_freq);
        log::debug!(
            "refresh_final_reports '{}': band=[{:.0},{:.0}] topology {:.4}->{:.4}, refreshed {:.4}->{:.4}",
            ch_result.name,
            score_min_freq,
            score_max_freq,
            topology_pre,
            topology_post,
            ch_result.pre_score,
            ch_result.post_score,
        );
        if let Some(chain) = result.channels.get_mut(&ch_result.name) {
            chain.final_curve = Some((&ch_result.final_curve).into());
        }
    }

    let count = result.channel_results.len().max(1) as f64;
    let avg_pre = result
        .channel_results
        .values()
        .map(|ch| ch.pre_score)
        .sum::<f64>()
        / count;
    let avg_post = result
        .channel_results
        .values()
        .map(|ch| ch.post_score)
        .sum::<f64>()
        / count;
    result.combined_pre_score = avg_pre;
    result.combined_post_score = avg_post;
    result.metadata.pre_score = avg_pre;
    result.metadata.post_score = avg_post;
    result.metadata.home_cinema_layout = Some(roomeq_engine::home_cinema::analyze_layout(config));
    result.metadata.multi_seat_coverage = Some(crate::home_cinema::multi_seat_coverage(config));
    let existing_bass_management = result.metadata.bass_management.clone();
    result.metadata.bass_management = if let Some(existing) = existing_bass_management {
        roomeq_engine::home_cinema::bass_management_report_with_optimization_and_sample_rate(
            config,
            existing.applied_sub_gain_db,
            existing.gain_limited,
            existing.optimization,
            sample_rate,
        )
    } else {
        roomeq_engine::home_cinema::bass_management_report(config, None, false)
    };

    let epa_cfg = config.optimizer.epa_config.clone().unwrap_or_default();
    let runtime_epa_cfg = roomeq_engine::config_adapter::to_optimizer_epa(&epa_cfg);
    result.metadata.epa_per_channel =
        roomeq_engine::output::compute_epa_per_channel(&result.channels, &epa_cfg);
    result.metadata.epa_multichannel =
        roomeq_engine::output::compute_epa_multichannel(&result.channels, &epa_cfg);

    let ir_inputs: Vec<_> = result
        .channel_results
        .iter()
        .map(|(name, ch)| {
            let delay_ms = result
                .channels
                .get(name)
                .map(total_chain_delay_ms)
                .unwrap_or(0.0);
            (
                name.clone(),
                ch.initial_curve.clone(),
                ch.biquads.clone(),
                ch.fir_coeffs.clone(),
                delay_ms,
            )
        })
        .collect();

    for (channel_name, initial_curve, biquads, fir_coeffs, delay_ms) in ir_inputs {
        if let Some((pre_ir, post_ir)) =
            roomeq_engine::analysis::ir_waveform::compute_channel_ir_waveforms(
                &initial_curve,
                &biquads,
                fir_coeffs.as_deref(),
                delay_ms,
                sample_rate,
            )
            && let Some(chain) = result.channels.get_mut(&channel_name)
        {
            chain.pre_ir = Some(pre_ir);
            chain.post_ir = Some(post_ir);
        }

        if let Some(coeffs) = fir_coeffs.as_deref()
            && let Some(metrics) = roomeq_engine::loss::epa::score::temporal_ir_masking_metrics(
                coeffs,
                sample_rate,
                &runtime_epa_cfg.temporal_masking,
            )
            && let Some(chain) = result.channels.get_mut(&channel_name)
        {
            chain.fir_temporal_masking = Some(
                roomeq_engine::report_adapter::to_temporal_ir_masking(metrics),
            );
        }
    }

    refresh_direct_early_late_reports(result, config);
    refresh_perceptual_policy_reports(result, config);

    update_perceptual_metrics(&mut result.metadata, Some(&result.channels), Some(config));
}

pub(in super::super) fn refresh_direct_early_late_reports(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
) {
    let Some(early_late_cfg) = config.optimizer.early_late_correction_config() else {
        return;
    };
    for chain in result.channels.values_mut() {
        chain.direct_early_late_correction = match (&chain.pre_ir, &chain.post_ir) {
            (Some(pre), Some(post)) => {
                direct_early_late_correction_metrics(pre, post, &early_late_cfg)
            }
            _ => None,
        };
    }
}

pub(in super::super) fn refresh_perceptual_policy_reports(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
) {
    result.metadata.perceptual_policy = build_perceptual_policy_report(config);
    result.metadata.bootstrap_uncertainty = build_bootstrap_uncertainty_report(config);
}
