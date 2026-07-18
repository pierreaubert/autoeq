use super::super::output;
use super::super::types::{MeasurementSource, RoomConfig};
use super::build::build_clamped_optimizer;
use super::misc::create_kautz_filter_config;
use super::types::ChannelDspChain;
use super::types::ChannelOptimizationInput;
use super::types::ChannelReport;
use super::types::MixedModeResult;
use super::types::OptimizerOutput;
use super::types::PreparedMeasurement;
use super::types::PreprocessedFeatures;
use super::types::TargetContext;
use crate::Curve;
use crate::error::{AutoeqError, Result};
use crate::response;
use log::{debug, info};
use math_audio_dsp::analysis::compute_average_response;
use std::path::Path;

pub(super) fn prepare_measurement(
    input: &ChannelOptimizationInput<'_>,
) -> Result<PreparedMeasurement> {
    let curve = input.prepared.measurements().representative().clone();
    debug!(
        "  Loaded measurement: {:.1} Hz - {:.1} Hz",
        curve.freq[0],
        curve.freq[curve.freq.len() - 1]
    );
    super::super::optimize::warn_if_optimizer_bounds_exceed_data(
        input.channel_name,
        &curve,
        &input.room_config.optimizer,
    );
    let arrival_time_ms = input.prepared.arrival_time_ms();
    let curve_raw = curve.clone();

    Ok(PreparedMeasurement {
        curve,
        curve_raw,
        arrival_time_ms,
    })
}

fn compute_flat_score(curve: &Curve, min_freq: f64, max_freq: f64) -> f64 {
    let freqs_f32: Vec<f32> = curve.freq.iter().map(|&f| f as f32).collect();
    let spl_f32: Vec<f32> = curve.spl.iter().map(|&s| s as f32).collect();
    let mean = compute_average_response(
        &freqs_f32,
        &spl_f32,
        Some((min_freq as f32, max_freq as f32)),
    ) as f64;
    let normalized = &curve.spl - mean;
    crate::loss::flat_loss(&curve.freq, &normalized, min_freq, max_freq)
}

/// Assemble the decomposed DSP-chain parts for a single channel.
///
/// This function only builds plugin configurations and the filter set used for
/// response simulation; it does not compute curves or scores.
pub(super) fn assemble_dsp_chain(
    _input: &ChannelOptimizationInput<'_>,
    preprocessed: &PreprocessedFeatures,
    optim_output: &OptimizerOutput,
) -> Result<ChannelDspChain> {
    let mut pre_eq_plugins = Vec::new();
    let mut eq_plugins = Vec::new();
    let mut post_eq_plugins = Vec::new();
    let mut filters = Vec::new();

    match optim_output {
        OptimizerOutput::PhaseLinear { wav_filename, .. } => {
            pre_eq_plugins.extend(preprocessed.broadband_plugins.iter().cloned());
            post_eq_plugins.push(output::create_convolution_plugin(wav_filename));
        }
        OptimizerOutput::Hybrid {
            eq_filters,
            wav_filename,
            ..
        } => {
            pre_eq_plugins.extend(preprocessed.broadband_plugins.iter().cloned());
            if !eq_filters.is_empty() {
                eq_plugins.push(output::create_labeled_eq_plugin(
                    eq_filters,
                    "room_eq_correction",
                ));
            }
            post_eq_plugins.push(output::create_convolution_plugin(wav_filename));
            filters.extend(eq_filters.iter().cloned());
        }
        OptimizerOutput::MixedPhase {
            eq_filters,
            fir_filename,
            report,
            ..
        } => {
            pre_eq_plugins.extend(preprocessed.broadband_plugins.iter().cloned());
            if !eq_filters.is_empty() {
                eq_plugins.push(output::create_labeled_eq_plugin(
                    eq_filters,
                    "room_eq_correction",
                ));
            }
            if let Some(filename) = fir_filename {
                post_eq_plugins.push(if let Some(report) = report {
                    output::create_mixed_phase_convolution_plugin(filename, report)
                } else {
                    output::create_convolution_plugin(filename)
                });
            }
            filters.extend(eq_filters.iter().cloned());
        }
        OptimizerOutput::LowLatency {
            eq_filters,
            preference_filters,
        } => {
            pre_eq_plugins.extend(preprocessed.cea2034_plugins.iter().cloned());
            pre_eq_plugins.extend(preprocessed.broadband_plugins.iter().cloned());

            let mut main_eq_filters = preprocessed.excursion_filters.clone();
            main_eq_filters.extend(eq_filters.iter().cloned());
            if !main_eq_filters.is_empty() {
                eq_plugins.push(output::create_labeled_eq_plugin(
                    &main_eq_filters,
                    "room_eq_correction",
                ));
            }

            if !preference_filters.is_empty() {
                post_eq_plugins.push(output::create_labeled_eq_plugin(
                    preference_filters,
                    "user_preference",
                ));
            }

            filters.extend(preprocessed.excursion_filters.iter().cloned());
            filters.extend(preprocessed.cea2034_filters.iter().cloned());
            filters.extend(preprocessed.broadband_biquads.iter().cloned());
            filters.extend(eq_filters.iter().cloned());
            filters.extend(preference_filters.iter().cloned());
        }
        OptimizerOutput::WarpedIir {
            eq_filters,
            preference_filters,
            warped_lambda,
        } => {
            pre_eq_plugins.extend(preprocessed.cea2034_plugins.iter().cloned());
            pre_eq_plugins.extend(preprocessed.broadband_plugins.iter().cloned());

            if !eq_filters.is_empty() || !preprocessed.excursion_filters.is_empty() {
                eq_plugins.push(output::create_warped_eq_plugin(
                    &preprocessed.excursion_filters,
                    eq_filters,
                    Some(*warped_lambda),
                ));
            }

            if !preference_filters.is_empty() {
                post_eq_plugins.push(output::create_labeled_eq_plugin(
                    preference_filters,
                    "user_preference",
                ));
            }

            filters.extend(preprocessed.excursion_filters.iter().cloned());
            filters.extend(preprocessed.cea2034_filters.iter().cloned());
            filters.extend(preprocessed.broadband_biquads.iter().cloned());
            filters.extend(eq_filters.iter().cloned());
            filters.extend(preference_filters.iter().cloned());
        }
        OptimizerOutput::KautzModal {
            eq_filters,
            kautz_sections,
            preference_filters,
        } => {
            pre_eq_plugins.extend(preprocessed.cea2034_plugins.iter().cloned());
            pre_eq_plugins.extend(preprocessed.broadband_plugins.iter().cloned());

            let mut main_filter_configs: Vec<serde_json::Value> = preprocessed
                .excursion_filters
                .iter()
                .map(output::biquad_to_json)
                .collect();
            main_filter_configs.push(create_kautz_filter_config(kautz_sections));
            eq_plugins.push(output::create_labeled_eq_plugin_from_filter_configs(
                main_filter_configs,
                "kautz_modal",
            ));

            if !preference_filters.is_empty() {
                post_eq_plugins.push(output::create_labeled_eq_plugin(
                    preference_filters,
                    "user_preference",
                ));
            }

            filters.extend(preprocessed.excursion_filters.iter().cloned());
            filters.extend(preprocessed.cea2034_filters.iter().cloned());
            filters.extend(preprocessed.broadband_biquads.iter().cloned());
            filters.extend(eq_filters.iter().cloned());
            filters.extend(preference_filters.iter().cloned());
        }
    }

    let mut plugin_order =
        Vec::with_capacity(pre_eq_plugins.len() + eq_plugins.len() + post_eq_plugins.len());
    plugin_order.extend(pre_eq_plugins.iter().cloned());
    plugin_order.extend(eq_plugins.iter().cloned());
    plugin_order.extend(post_eq_plugins.iter().cloned());

    Ok(ChannelDspChain {
        pre_eq_plugins,
        eq_plugins,
        post_eq_plugins,
        plugin_order,
        delays: Vec::new(),
        gains: Vec::new(),
        filters,
    })
}

/// Assemble the report curves, scores and metadata for a single channel.
///
/// This does not construct the final serialized [`ChannelDspChain`]; it returns
/// the per-channel report data that [`process_single_speaker`] combines with
/// the DSP-chain parts to build the final result.
pub(super) fn assemble_channel_report(
    input: &ChannelOptimizationInput<'_>,
    prepared: &PreparedMeasurement,
    target: &TargetContext,
    preprocessed: &PreprocessedFeatures,
    dsp_chain: &ChannelDspChain,
    optim_output: &OptimizerOutput,
) -> Result<ChannelReport> {
    let curve_raw = &prepared.curve_raw;
    let min_freq = target.min_freq;
    let max_freq = target.max_freq;
    let pre_score = target.pre_score;
    let mean_spl = target.mean_spl;
    let sample_rate = input.sample_rate;
    let norm_range = preprocessed.norm_range;
    let target_tilt_curve = &target.target_tilt_curve;
    let bb_mean_shift = preprocessed.broadband_mean_shift;

    let display_initial = output::extend_curve_to_full_range(curve_raw);
    let mut initial_data: super::super::types::CurveData = (&display_initial).into();
    initial_data.norm_range = norm_range;

    let (final_curve, display_final) = match optim_output {
        OptimizerOutput::PhaseLinear { coeffs, .. } => {
            let complex_resp = response::compute_fir_complex_response(
                coeffs,
                &preprocessed.curve.freq,
                sample_rate,
            );
            let final_curve =
                response::apply_complex_response(&preprocessed.curve_for_optim, &complex_resp);

            let display_fir_resp =
                response::compute_fir_complex_response(coeffs, &display_initial.freq, sample_rate);
            let display_final =
                response::apply_complex_response(&display_initial, &display_fir_resp);

            (final_curve, display_final)
        }
        OptimizerOutput::Hybrid {
            eq_filters, coeffs, ..
        } => {
            let iir_resp = response::compute_peq_complex_response(
                eq_filters,
                &preprocessed.curve.freq,
                sample_rate,
            );
            let final_curve_iir = response::apply_complex_response(&preprocessed.curve, &iir_resp);
            let fir_resp = response::compute_fir_complex_response(
                coeffs,
                &preprocessed.curve.freq,
                sample_rate,
            );
            let final_curve = response::apply_complex_response(&final_curve_iir, &fir_resp);

            let display_iir_resp = response::compute_peq_complex_response(
                eq_filters,
                &display_initial.freq,
                sample_rate,
            );
            let display_iir_corrected =
                response::apply_complex_response(&display_initial, &display_iir_resp);
            let display_fir_resp =
                response::compute_fir_complex_response(coeffs, &display_initial.freq, sample_rate);
            let display_final =
                response::apply_complex_response(&display_iir_corrected, &display_fir_resp);

            (final_curve, display_final)
        }
        OptimizerOutput::MixedPhase {
            eq_filters,
            fir_coeffs,
            ..
        } => {
            let eq_resp = response::compute_peq_complex_response(
                eq_filters,
                &preprocessed.curve.freq,
                sample_rate,
            );
            let after_eq =
                response::apply_complex_response(&preprocessed.curve_for_optim, &eq_resp);
            let final_curve = if let Some(coeffs) = fir_coeffs {
                let fir_resp =
                    response::compute_fir_complex_response(coeffs, &after_eq.freq, sample_rate);
                response::apply_complex_response(&after_eq, &fir_resp)
            } else {
                after_eq
            };

            let display_eq_resp = response::compute_peq_complex_response(
                eq_filters,
                &display_initial.freq,
                sample_rate,
            );
            let display_after_eq =
                response::apply_complex_response(&display_initial, &display_eq_resp);
            let display_final = if let Some(coeffs) = fir_coeffs {
                let fir_resp = response::compute_fir_complex_response(
                    coeffs,
                    &display_after_eq.freq,
                    sample_rate,
                );
                response::apply_complex_response(&display_after_eq, &fir_resp)
            } else {
                display_after_eq
            };

            (final_curve, display_final)
        }
        OptimizerOutput::LowLatency { .. }
        | OptimizerOutput::WarpedIir { .. }
        | OptimizerOutput::KautzModal { .. } => {
            let mut score_raw = curve_raw.clone();
            score_raw.spl += bb_mean_shift;
            let all_resp = response::compute_peq_complex_response(
                &dsp_chain.filters,
                &score_raw.freq,
                sample_rate,
            );
            let final_curve = response::apply_complex_response(&score_raw, &all_resp);

            let mut display_raw_with_bb = display_initial.clone();
            display_raw_with_bb.spl += bb_mean_shift;
            let display_resp = response::compute_peq_complex_response(
                &dsp_chain.filters,
                &display_raw_with_bb.freq,
                sample_rate,
            );
            let display_final =
                response::apply_complex_response(&display_raw_with_bb, &display_resp);

            (final_curve, display_final)
        }
    };

    let post_score = match optim_output {
        OptimizerOutput::LowLatency { .. }
        | OptimizerOutput::WarpedIir { .. }
        | OptimizerOutput::KautzModal { .. } => {
            let score_curve = if let Some(tilt_curve) = target_tilt_curve {
                Curve {
                    freq: final_curve.freq.clone(),
                    spl: &final_curve.spl - &tilt_curve.spl,
                    phase: final_curve.phase.clone(),
                    ..Default::default()
                }
            } else {
                final_curve.clone()
            };
            compute_flat_score(&score_curve, min_freq, max_freq)
        }
        _ => compute_flat_score(&final_curve, min_freq, max_freq),
    };

    info!(
        "  Pre-score: {:.6}, Post-score: {:.6}",
        pre_score, post_score
    );

    let mut final_data: super::super::types::CurveData = (&display_final).into();
    final_data.norm_range = norm_range;

    let eq_response = output::compute_eq_response(&initial_data, &final_data);

    let target_curve = match optim_output {
        OptimizerOutput::LowLatency { .. }
        | OptimizerOutput::WarpedIir { .. }
        | OptimizerOutput::KautzModal { .. } => {
            let display_target_spl = if let Some(tilt_curve) = target_tilt_curve {
                let tilt_at_display = crate::read::normalize_and_interpolate_response(
                    &display_initial.freq,
                    tilt_curve,
                );
                &tilt_at_display.spl + mean_spl
            } else {
                ndarray::Array1::from_elem(display_initial.freq.len(), mean_spl)
            };
            Some(super::super::types::CurveData {
                freq: display_initial.freq.to_vec(),
                spl: display_target_spl.to_vec(),
                phase: None,
                norm_range,
            })
        }
        _ => None,
    };

    let report_filters = match optim_output {
        OptimizerOutput::PhaseLinear { .. } => Vec::new(),
        OptimizerOutput::Hybrid { eq_filters, .. } => eq_filters.clone(),
        OptimizerOutput::MixedPhase { eq_filters, .. } => eq_filters.clone(),
        OptimizerOutput::LowLatency { eq_filters, .. } => eq_filters.clone(),
        OptimizerOutput::WarpedIir { eq_filters, .. } => eq_filters.clone(),
        OptimizerOutput::KautzModal { eq_filters, .. } => eq_filters.clone(),
    };

    Ok(ChannelReport {
        channel_name: input.channel_name.to_string(),
        pre_score,
        post_score,
        raw_pre_eq_curve: curve_raw.clone(),
        raw_post_eq_curve: final_curve.clone(),
        pre_eq_curve: display_initial,
        post_eq_curve: display_final,
        eq_curve: eq_response,
        target_curve,
        filters: report_filters,
        mean_spl,
        arrival_time_ms: prepared.arrival_time_ms,
    })
}

pub(super) fn build_mixed_mode_result(
    dsp_chain: ChannelDspChain,
    report: ChannelReport,
    optim_output: OptimizerOutput,
    optimizer_evidence: Vec<crate::optim::OptimizerRunEvidence>,
) -> MixedModeResult {
    let public_chain = super::super::types::ChannelDspChain {
        channel: report.channel_name,
        plugins: dsp_chain.plugin_order,
        drivers: None,
        initial_curve: Some((&report.pre_eq_curve).into()),
        final_curve: Some((&report.post_eq_curve).into()),
        eq_response: Some(report.eq_curve),
        pre_ir: None,
        post_ir: None,
        fir_temporal_masking: None,
        direct_early_late_correction: None,
        target_curve: report.target_curve,
    };

    let fir_coeffs = match optim_output {
        OptimizerOutput::PhaseLinear { coeffs, .. } => Some(coeffs),
        OptimizerOutput::Hybrid { coeffs, .. } => Some(coeffs),
        OptimizerOutput::MixedPhase { fir_coeffs, .. } => fir_coeffs,
        _ => None,
    };

    (
        public_chain,
        report.pre_score,
        report.post_score,
        report.raw_pre_eq_curve,
        report.raw_post_eq_curve,
        report.filters,
        report.mean_spl,
        report.arrival_time_ms,
        fir_coeffs,
        optimizer_evidence,
    )
}
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn process_single_speaker(
    channel_name: &str,
    source: &MeasurementSource,
    room_config: &RoomConfig,
    sample_rate: f64,
    output_dir: &Path,
    callback: Option<crate::optim::OptimProgressCallback>,
    probe_arrival_ms: Option<f64>,
    shared_mean_spl: Option<f64>,
) -> Result<MixedModeResult> {
    let prepared_input = roomeq_workflow::prepare_channel_input(
        channel_name,
        source,
        room_config,
        sample_rate,
        probe_arrival_ms,
    )
    .map_err(|e| AutoeqError::InvalidMeasurement {
        message: format!(
            "Failed to load measurement for channel {}: {}",
            channel_name, e
        ),
    })?;
    let mut input = ChannelOptimizationInput {
        channel_name,
        prepared: &prepared_input,
        room_config,
        sample_rate,
        output_dir,
        callback,
        shared_mean_spl,
    };

    let prepared = prepare_measurement(&input)?;
    let mut target = roomeq_engine::channel_target::build_target_context(
        input.channel_name,
        input.room_config,
        &prepared.curve,
        input.shared_mean_spl,
    );
    let preprocessed = roomeq_engine::channel_preprocessing::preprocess_channel(
        input.channel_name,
        input.prepared,
        input.room_config,
        input.sample_rate,
        input.shared_mean_spl,
        &mut target,
    );

    let clamped_optimizer = build_clamped_optimizer(
        channel_name,
        room_config,
        &prepared.curve_raw,
        &preprocessed.curve_for_optim,
        target.min_freq,
        target.max_freq,
        target.target_tilt_curve.as_ref(),
        preprocessed.broadband_enabled,
    );

    let mut eq_resources = prepared_input.eq_resources().clone();
    eq_resources.target = roomeq_workflow::prepare_eq_target(target.effective_target(room_config))
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!(
                "Failed to prepare EQ resources for channel {}: {}",
                channel_name, e
            ),
        })?;

    super::strategies::strategy_for_mode(room_config.optimizer.processing_mode.clone()).process(
        &mut input,
        &prepared,
        &target,
        &preprocessed,
        &clamped_optimizer,
        &eq_resources,
    )
}
