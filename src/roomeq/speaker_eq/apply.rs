use super::super::excursion;
use super::super::optimize::detect_passband_and_mean;
use super::super::output;
use super::super::spectral_align;
use super::super::types::{MeasurementSource, PluginConfigWrapper, RoomConfig};
use super::build::build_clamped_optimizer;
use super::build::build_target_tilt_curve;
use super::misc::broadband_correction_rejected;
use super::misc::cea2034_correction_active;
use super::misc::create_kautz_filter_config;
use super::misc::detect_channel_arrival_time;
use super::misc::flatness_score_in_range;
use super::misc::generate_excursion_filters;
use super::misc::load_channel_measurement;
use super::misc::maybe_clamp_min_freq_for_target_tilt;
use super::misc::mean_response_in_range;
use super::misc::target_mean_spl;
#[allow(unused_imports)]
pub(super) use super::schroeder::optimize_with_schroeder_split;
use super::types::BroadbandPreCorrection;
use super::types::ChannelDspChain;
use super::types::ChannelOptimizationInput;
use super::types::ChannelReport;
use super::types::MixedModeResult;
use super::types::OptimizerOutput;
use super::types::PreparedMeasurement;
use super::types::PreprocessedFeatures;
use super::types::TargetContext;
use crate::Curve;
use crate::error::Result;
use crate::response;
use log::{debug, info, warn};
use math_audio_dsp::analysis::compute_average_response;
use math_audio_iir_fir::Biquad;
use ndarray::Array1;
use std::path::Path;

pub(super) fn apply_excursion_filters_to_curve(
    curve: Curve,
    excursion_filters: &[Biquad],
    sample_rate: f64,
) -> Curve {
    if excursion_filters.is_empty() {
        return curve;
    }

    let hpf_resp =
        response::compute_peq_complex_response(excursion_filters, &curve.freq, sample_rate);
    let adjusted = response::apply_complex_response(&curve, &hpf_resp);
    info!(
        "  Simulating excursion HPF on optimization curve ({} filters)",
        excursion_filters.len()
    );
    adjusted
}

pub(super) fn apply_cea2034_speaker_correction(
    channel_name: &str,
    source: &MeasurementSource,
    room_config: &RoomConfig,
    curve: Curve,
    arrival_time_ms: Option<f64>,
    sample_rate: f64,
) -> (Curve, Vec<Biquad>, Vec<PluginConfigWrapper>) {
    let Some(cea_config) = &room_config.optimizer.cea2034_correction else {
        return (curve, vec![], vec![]);
    };
    if !cea_config.enabled {
        return (curve, vec![], vec![]);
    }

    // Resolve speaker name: config override > MeasurementSource
    let speaker_name = cea_config
        .speaker_name
        .as_deref()
        .or_else(|| source.speaker_name());

    let Some(name) = speaker_name else {
        debug!(
            "  No speaker_name configured for '{}'. Skipping CEA2034 correction.",
            channel_name
        );
        return (curve, vec![], vec![]);
    };

    let cea_data = room_config
        .cea2034_cache
        .as_ref()
        .and_then(|cache| cache.get(name));

    let Some(data) = cea_data else {
        warn!(
            "  No CEA2034 data in cache for speaker '{}'. Skipping Pass 1.",
            name
        );
        return (curve, vec![], vec![]);
    };

    let schroeder_freq = cea_config.min_freq.unwrap_or_else(|| {
        room_config
            .optimizer
            .schroeder_split
            .as_ref()
            .filter(|s| s.enabled)
            .map(|s| s.schroeder_freq)
            .unwrap_or(300.0)
    });

    match super::super::cea2034_correction::compute_speaker_correction(
        data,
        cea_config,
        &curve,
        schroeder_freq,
        arrival_time_ms,
        sample_rate,
    ) {
        Ok((filters, corrected_curve)) => {
            info!(
                "  Pass 1 CEA2034 correction: {} filters above {:.0} Hz for '{}'",
                filters.len(),
                schroeder_freq,
                name
            );
            let plugin = output::create_labeled_eq_plugin(&filters, "cea2034_speaker_correction");
            (corrected_curve, filters, vec![plugin])
        }
        Err(e) => {
            warn!(
                "  CEA2034 correction failed for '{}': {}. Skipping Pass 1.",
                name, e
            );
            (curve, vec![], vec![])
        }
    }
}

pub(super) fn apply_broadband_precorrection(
    room_config: &RoomConfig,
    curve: &Curve,
    _target_tilt_curve: Option<&Curve>,
    mean_spl: f64,
    min_freq: f64,
    max_freq: f64,
    sample_rate: f64,
) -> BroadbandPreCorrection {
    let broadband_enabled = room_config
        .optimizer
        .target_response
        .as_ref()
        .is_some_and(|tr| tr.broadband_precorrection);

    if !broadband_enabled {
        return BroadbandPreCorrection {
            curve_for_optim: curve.clone(),
            plugins: Vec::new(),
            biquads: Vec::new(),
            mean_shift: 0.0,
        };
    }

    info!("  Broadband pre-correction enabled...");

    // Detect F3 to avoid shelf-correcting below the speaker's rolloff.
    let detected_f3 = match excursion::detect_f3_with_config(
        curve,
        None,
        room_config.optimizer.excursion_protection.as_ref(),
    ) {
        Ok(f3_result) if f3_result.f3_hz > min_freq && f3_result.f3_hz < max_freq * 0.5 => {
            info!("  Broadband: detected speaker F3={:.1}Hz", f3_result.f3_hz);
            Some(f3_result.f3_hz)
        }
        _ => None,
    };
    let bb_min_freq = detected_f3.unwrap_or(min_freq);

    // Broadband pre-correction removes only coarse measurement shape. Target
    // tilt and preference shaping are handled by the following optimizer; if
    // included here they are applied twice and the fine EQ has to fight the
    // shelf approximation.
    let target = Curve {
        freq: curve.freq.clone(),
        spl: Array1::from_elem(curve.freq.len(), mean_spl),
        phase: None,
        ..Default::default()
    };

    let Some(mut result) =
        spectral_align::compute_target_alignment(curve, &target, bb_min_freq, 20000.0, sample_rate)
    else {
        return BroadbandPreCorrection {
            curve_for_optim: curve.clone(),
            plugins: Vec::new(),
            biquads: Vec::new(),
            mean_shift: 0.0,
        };
    };

    // Suppress the low-shelf when a rolloff is detected below the
    // shelf frequency: the shelf response extends to DC and would
    // partially boost the rolloff region, creating a worse shape
    // than leaving it uncorrected.
    if let Some(f3) = detected_f3
        && f3 < spectral_align::LOWSHELF_FREQ
    {
        info!(
            "  Broadband: suppressing low-shelf (F3={:.1}Hz < shelf={:.1}Hz)",
            f3,
            spectral_align::LOWSHELF_FREQ
        );
        result.lowshelf_gain_db = 0.0;
    }
    info!(
        "  Broadband correction: LS={:+.2}dB, HS={:+.2}dB, Gain={:+.2}dB",
        result.lowshelf_gain_db, result.highshelf_gain_db, result.flat_gain_db
    );

    let (eq_plugin, gain_plugin) = spectral_align::create_alignment_plugins(&result, sample_rate);

    let mut plugins = Vec::new();
    if let Some(g) = gain_plugin {
        plugins.push(g);
    }
    if let Some(mut eq) = eq_plugin {
        // Label the broadband EQ so the UI can distinguish it
        // from the main room-correction EQ.
        eq.parameters["label"] = serde_json::json!("broadband");
        plugins.push(eq);
    }

    use math_audio_iir_fir::{BiquadFilterType, DEFAULT_Q_HIGH_LOW_SHELF};
    let mut filters = Vec::new();
    if result.lowshelf_gain_db.abs() > 1e-3 {
        filters.push(Biquad::new(
            BiquadFilterType::Lowshelf,
            spectral_align::LOWSHELF_FREQ,
            sample_rate,
            DEFAULT_Q_HIGH_LOW_SHELF,
            result.lowshelf_gain_db,
        ));
    }
    if result.highshelf_gain_db.abs() > 1e-3 {
        filters.push(Biquad::new(
            BiquadFilterType::Highshelf,
            spectral_align::HIGHSHELF_FREQ,
            sample_rate,
            DEFAULT_Q_HIGH_LOW_SHELF,
            result.highshelf_gain_db,
        ));
    }

    let mut temp_curve = curve.clone();
    temp_curve.spl += result.flat_gain_db;

    let corrected_curve = if !filters.is_empty() {
        let resp = response::compute_peq_complex_response(&filters, &curve.freq, sample_rate);
        response::apply_complex_response(&temp_curve, &resp)
    } else {
        temp_curve
    };

    // Validate: reject broadband correction if it makes things worse.
    // Measure deviation from the tilted target — broadband should move
    // us CLOSER to the target, not further away.
    let target_spl = &target.spl;
    let pre_bb_dev = &curve.spl - target_spl;
    let pre_bb_score = crate::loss::flat_loss(&curve.freq, &pre_bb_dev, min_freq, max_freq);
    let post_bb_dev = &corrected_curve.spl - target_spl;
    let post_bb_score =
        crate::loss::flat_loss(&corrected_curve.freq, &post_bb_dev, min_freq, max_freq);

    if broadband_correction_rejected(pre_bb_score, post_bb_score) {
        warn!(
            "  Broadband correction rejected: deviation from target {:.4} -> {:.4} \
                 (worse by {:.0}%). Shelf fit likely confused by room modes or HPF rolloff.",
            pre_bb_score,
            post_bb_score,
            (post_bb_score / pre_bb_score - 1.0) * 100.0,
        );
        BroadbandPreCorrection {
            curve_for_optim: curve.clone(),
            plugins: Vec::new(),
            biquads: Vec::new(),
            mean_shift: 0.0,
        }
    } else {
        BroadbandPreCorrection {
            curve_for_optim: corrected_curve,
            plugins,
            biquads: filters,
            mean_shift: result.flat_gain_db,
        }
    }
}

/// Process a simple speaker with a single measurement
///
/// Returns: (DSP chain, pre_score, post_score, initial_curve, final_curve, biquads, mean_spl, arrival_time_ms)
///
/// `shared_mean_spl` — when `Some`, the target level is this shared average
/// instead of the channel's own mean. Reduces inter-channel deviation at the
/// source by making all channels optimize toward the same reference level.
pub(super) fn prepare_measurement(
    input: &ChannelOptimizationInput<'_>,
) -> Result<PreparedMeasurement> {
    let curve = load_channel_measurement(input.channel_name, input.source, input.room_config)?;
    let arrival_time_ms = detect_channel_arrival_time(
        input.channel_name,
        input.source,
        input.room_config,
        input.sample_rate,
        input.probe_arrival_ms,
    );
    let curve_raw = curve.clone();

    Ok(PreparedMeasurement {
        curve,
        curve_raw,
        arrival_time_ms,
    })
}

pub(super) fn build_target_context(
    input: &ChannelOptimizationInput<'_>,
    prepared: &PreparedMeasurement,
) -> Result<TargetContext> {
    let cea2034_active = cea2034_correction_active(input.room_config);
    let target_tilt_curve = build_target_tilt_curve(
        input.channel_name,
        input.room_config,
        &prepared.curve,
        cea2034_active,
    );

    if target_tilt_curve.is_some() && input.room_config.target_curve.is_some() {
        warn!(
            "  Both target_curve and target_response are configured for '{}'. \
             target_response is baked into the measurement; target_curve will be \
             ignored to avoid double-application.",
            input.channel_name
        );
    }

    let min_freq = input.room_config.optimizer.min_freq;
    let max_freq = input.room_config.optimizer.max_freq;

    let pre_score = flatness_score_in_range(&prepared.curve, min_freq, max_freq);
    let channel_mean_spl = mean_response_in_range(&prepared.curve, min_freq, max_freq);
    let mean_spl = input.shared_mean_spl.unwrap_or(channel_mean_spl);

    Ok(TargetContext {
        target_tilt_curve,
        min_freq,
        max_freq,
        pre_score,
        mean_spl,
        cea2034_active,
    })
}

pub(super) fn preprocess_features(
    input: &ChannelOptimizationInput<'_>,
    prepared: &PreparedMeasurement,
    target: &mut TargetContext,
) -> Result<PreprocessedFeatures> {
    let excursion_filters =
        generate_excursion_filters(input.room_config, &prepared.curve, input.sample_rate);

    let curve = apply_excursion_filters_to_curve(
        prepared.curve.clone(),
        &excursion_filters,
        input.sample_rate,
    );
    let (curve, cea2034_filters, cea2034_plugins) = apply_cea2034_speaker_correction(
        input.channel_name,
        input.source,
        input.room_config,
        curve,
        prepared.arrival_time_ms,
        input.sample_rate,
    );

    let (norm_range, _passband_mean) = detect_passband_and_mean(&curve);

    if let Some((f_low, f_high)) = norm_range {
        info!(
            "  Detected passband for '{}': {:.1} Hz - {:.1} Hz",
            input.channel_name, f_low, f_high
        );
    }

    target.min_freq = maybe_clamp_min_freq_for_target_tilt(
        input.channel_name,
        input.room_config,
        &curve,
        target.target_tilt_curve.as_ref(),
        target.min_freq,
        target.max_freq,
    );

    let pre_score = flatness_score_in_range(&curve, target.min_freq, target.max_freq);
    let channel_mean_spl = mean_response_in_range(&curve, target.min_freq, target.max_freq);
    let mean_spl = target_mean_spl(input.channel_name, channel_mean_spl, input.shared_mean_spl);

    let broadband_enabled = input
        .room_config
        .optimizer
        .target_response
        .as_ref()
        .is_some_and(|tr| tr.broadband_precorrection);

    let broadband = apply_broadband_precorrection(
        input.room_config,
        &curve,
        target.target_tilt_curve.as_ref(),
        mean_spl,
        target.min_freq,
        target.max_freq,
        input.sample_rate,
    );
    let BroadbandPreCorrection {
        curve_for_optim,
        plugins: broadband_plugins,
        biquads: broadband_biquads,
        mean_shift: broadband_mean_shift,
    } = broadband;

    let mean_spl = mean_spl + broadband_mean_shift;

    target.pre_score = pre_score;
    target.mean_spl = mean_spl;

    Ok(PreprocessedFeatures {
        curve,
        curve_for_optim,
        excursion_filters,
        cea2034_filters,
        cea2034_plugins,
        broadband_plugins,
        broadband_biquads,
        broadband_mean_shift,
        broadband_enabled,
        norm_range,
    })
}

/// Compute a flatness score for `curve` over `[min_freq, max_freq]` using the
/// same range-based mean normalization as the rest of the speaker-EQ pipeline.
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
                post_eq_plugins.push(output::create_convolution_plugin(filename));
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
    let mut input = ChannelOptimizationInput {
        channel_name,
        source,
        room_config,
        sample_rate,
        output_dir,
        callback,
        probe_arrival_ms,
        shared_mean_spl,
    };

    let prepared = prepare_measurement(&input)?;
    let mut target = build_target_context(&input, &prepared)?;
    let preprocessed = preprocess_features(&input, &prepared, &mut target)?;

    let clamped_optimizer = build_clamped_optimizer(
        channel_name,
        source,
        room_config,
        &prepared.curve_raw,
        &preprocessed.curve_for_optim,
        target.min_freq,
        target.max_freq,
        target.target_tilt_curve.as_ref(),
        preprocessed.broadband_enabled,
    );

    super::strategies::strategy_for_mode(room_config.optimizer.processing_mode.clone()).process(
        &mut input,
        &prepared,
        &target,
        &preprocessed,
        &clamped_optimizer,
    )
}
