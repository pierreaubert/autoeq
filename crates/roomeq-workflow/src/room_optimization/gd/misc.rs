use super::super::types::ChannelOptimizationResult;
use super::super::*;

pub(super) fn existing_fir_convolution_filename(chain: &ChannelDspChain) -> Option<String> {
    chain.plugins.iter().find_map(|plugin| {
        if plugin.plugin_type != "convolution" {
            return None;
        }
        let ir_file = plugin
            .parameters
            .get("ir_file")
            .and_then(|value| value.as_str())?;
        let file_name = Path::new(ir_file).file_name()?.to_str()?;
        let is_full_fir = (file_name.contains("_fir_") || file_name.ends_with("_fir.wav"))
            && !file_name.contains("residual_fir")
            && !file_name.contains("excess_phase_fir")
            && !file_name.contains("band_fir");
        is_full_fir.then(|| ir_file.to_string())
    })
}

pub(in crate::room_optimization) fn source_for_output_channel<'a>(
    config: &'a RoomConfig,
    channel_name: &str,
) -> Option<&'a MeasurementSource> {
    let speaker_config = config
        .speakers
        .get(channel_name)
        .or_else(|| {
            config
                .speakers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(channel_name))
                .map(|(_, speaker)| speaker)
        })
        .or_else(|| {
            let system = config.system.as_ref()?;
            let measurement_key = system.speakers.get(channel_name).or_else(|| {
                system
                    .speakers
                    .iter()
                    .find(|(role, _)| role.eq_ignore_ascii_case(channel_name))
                    .map(|(_, measurement_key)| measurement_key)
            })?;
            config.speakers.get(measurement_key)
        })?;

    match speaker_config {
        SpeakerConfig::Single(source) => Some(source),
        _ => None,
    }
}

pub(super) fn interpolate_optional_array_log(
    freq_out: &ndarray::Array1<f64>,
    freq_in: &ndarray::Array1<f64>,
    values: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let curve = Curve {
        freq: freq_in.clone(),
        spl: values.clone(),
        phase: None,
        ..Default::default()
    };
    autoeq_measurements::read::interpolate_log_space(freq_out, &curve).spl
}

pub(super) fn corrected_realisation_to_gd_input(
    raw_curve: &Curve,
    initial_curve: &Curve,
    final_curve: &Curve,
) -> Option<roomeq_engine::gd_opt::ChannelMeasurementInput> {
    let raw_on_grid =
        autoeq_measurements::read::interpolate_log_space(&final_curve.freq, raw_curve);
    let initial_on_grid =
        autoeq_measurements::read::interpolate_log_space(&final_curve.freq, initial_curve);

    let raw_phase = raw_on_grid.phase.as_ref()?;
    let final_phase = final_curve.phase.as_ref()?;
    let raw_coherence = raw_curve.coherence.as_ref()?;

    let coherence =
        interpolate_optional_array_log(&final_curve.freq, &raw_curve.freq, raw_coherence);

    let spl = &raw_on_grid.spl + &(&final_curve.spl - &initial_on_grid.spl);
    // Multi-position magnitude aggregation may omit representative phase.
    // In that case `apply_complex_response` makes `final_phase` the applied DSP
    // phase, so it can be composed with each raw sweep directly.
    let phase_delta = match initial_on_grid.phase.as_ref() {
        Some(initial_phase) => final_phase - initial_phase,
        None => final_phase.clone(),
    };
    let phase = (raw_phase + &phase_delta).mapv(|deg| deg.to_radians());

    Some(roomeq_engine::gd_opt::ChannelMeasurementInput {
        freq: final_curve.freq.clone(),
        spl,
        phase,
        coherence,
    })
}

pub(super) fn build_gd_sweep_realisations(
    config: &RoomConfig,
    channel_results: &HashMap<String, ChannelOptimizationResult>,
    channel_names: &[String],
) -> Option<Vec<Vec<roomeq_engine::gd_opt::ChannelMeasurementInput>>> {
    let mut per_channel: Vec<Vec<roomeq_engine::gd_opt::ChannelMeasurementInput>> = Vec::new();

    for name in channel_names {
        let Some(source) = source_for_output_channel(config, name) else {
            log::debug!("GD-Opt adaptive bootstrap: no measurement source for '{name}'");
            return None;
        };
        let raw_curves = match autoeq_measurements::read::load_source_individual(source) {
            Ok(curves) => curves,
            Err(error) => {
                log::debug!(
                    "GD-Opt adaptive bootstrap: failed to load sweeps for '{name}': {error}"
                );
                return None;
            }
        };
        if raw_curves.len() < 2 {
            log::debug!(
                "GD-Opt adaptive bootstrap: '{}' has {} sweep(s), need at least 2",
                name,
                raw_curves.len()
            );
            return None;
        }

        let Some(ch) = channel_results.get(name.as_str()) else {
            log::debug!("GD-Opt adaptive bootstrap: no channel result for '{name}'");
            return None;
        };
        let mut realisations = Vec::with_capacity(raw_curves.len());
        for raw_curve in &raw_curves {
            let Some(input) =
                corrected_realisation_to_gd_input(raw_curve, &ch.initial_curve, &ch.final_curve)
            else {
                log::debug!(
                    "GD-Opt adaptive bootstrap: unusable sweep for '{}': raw_phase={} \
                     initial_phase={} final_phase={} coherence={}",
                    name,
                    raw_curve.phase.is_some(),
                    ch.initial_curve.phase.is_some(),
                    ch.final_curve.phase.is_some(),
                    raw_curve.coherence.is_some()
                );
                return None;
            };
            realisations.push(input);
        }
        per_channel.push(realisations);
    }

    let n_realisations = per_channel.first()?.len();
    if n_realisations < 2
        || per_channel
            .iter()
            .any(|inputs| inputs.len() != n_realisations)
    {
        return None;
    }

    let mut by_sweep = Vec::with_capacity(n_realisations);
    for sweep_idx in 0..n_realisations {
        by_sweep.push(
            per_channel
                .iter()
                .map(|inputs| inputs[sweep_idx].clone())
                .collect(),
        );
    }

    Some(by_sweep)
}

pub(super) fn coherence_average_gd_realisations(
    realisations: &[Vec<roomeq_engine::gd_opt::ChannelMeasurementInput>],
) -> Option<Vec<roomeq_engine::gd_opt::ChannelMeasurementInput>> {
    let channel_count = realisations.first()?.len();
    if channel_count == 0
        || realisations
            .iter()
            .any(|sweep| sweep.len() != channel_count)
    {
        return None;
    }

    let mut averaged = Vec::with_capacity(channel_count);
    for channel_index in 0..channel_count {
        let reference = &realisations[0][channel_index];
        let bin_count = reference.freq.len();
        if bin_count == 0
            || realisations.iter().any(|sweep| {
                let channel = &sweep[channel_index];
                channel.spl.len() != bin_count
                    || channel.phase.len() != bin_count
                    || channel.coherence.len() != bin_count
                    || !roomeq_engine::analysis::frequency_grid::same_frequency_grid(
                        &reference.freq,
                        &channel.freq,
                    )
            })
        {
            return None;
        }

        let mut spl = ndarray::Array1::zeros(bin_count);
        let mut phase = ndarray::Array1::zeros(bin_count);
        let mut coherence = ndarray::Array1::zeros(bin_count);
        for bin in 0..bin_count {
            let coherence_sum: f64 = realisations
                .iter()
                .map(|sweep| sweep[channel_index].coherence[bin].clamp(0.0, 1.0))
                .sum();
            let use_equal_weights = coherence_sum <= f64::EPSILON;
            let mut weight_sum = 0.0;
            let mut power_sum = 0.0;
            let mut phase_sin_sum = 0.0;
            let mut phase_cos_sum = 0.0;
            let mut raw_coherence_sum = 0.0;

            for sweep in realisations {
                let channel = &sweep[channel_index];
                let raw_coherence = channel.coherence[bin].clamp(0.0, 1.0);
                let weight = if use_equal_weights {
                    1.0
                } else {
                    raw_coherence
                };
                weight_sum += weight;
                power_sum += weight * 10.0_f64.powf(channel.spl[bin] / 10.0);
                phase_sin_sum += weight * channel.phase[bin].sin();
                phase_cos_sum += weight * channel.phase[bin].cos();
                raw_coherence_sum += raw_coherence;
            }

            spl[bin] = 10.0 * (power_sum / weight_sum).max(1e-30).log10();
            phase[bin] = phase_sin_sum.atan2(phase_cos_sum);
            coherence[bin] = raw_coherence_sum / realisations.len() as f64;
        }

        averaged.push(roomeq_engine::gd_opt::ChannelMeasurementInput {
            freq: reference.freq.clone(),
            spl,
            phase,
            coherence,
        });
    }
    Some(averaged)
}

pub(in super::super) fn apply_gd_opt_result(
    result: &roomeq_engine::gd_opt::GroupDelayOptResult,
    channel_names: &[String],
    channel_results: &mut HashMap<String, ChannelOptimizationResult>,
    channel_chains: &mut HashMap<String, ChannelDspChain>,
    sample_rate: f64,
) -> bool {
    let mut applied_any = false;

    for (name, ch_result) in channel_names.iter().zip(result.per_channel.iter()) {
        let mut inserted_for_channel = false;
        if let Some(chain) = channel_chains.get_mut(name.as_str()) {
            if ch_result.polarity_inverted {
                chain
                    .plugins
                    .push(output::create_gain_plugin_with_invert(0.0, true));
                inserted_for_channel = true;
            }
            if ch_result.delay_ms > 0.01 {
                chain
                    .plugins
                    .push(output::create_delay_plugin(ch_result.delay_ms));
                inserted_for_channel = true;
            }
            if !ch_result.ap_filters.is_empty() {
                chain
                    .plugins
                    .push(output::create_eq_plugin(&ch_result.ap_filters));
                inserted_for_channel = true;
            }
        }

        if let Some(ch) = channel_results.get_mut(name.as_str()) {
            let response = gd_phase_response_for_curve(
                &ch.final_curve.freq,
                ch_result.delay_ms,
                ch_result.polarity_inverted,
                &ch_result.ap_filters,
                sample_rate,
            );
            ch.final_curve =
                roomeq_engine::response::apply_complex_response(&ch.final_curve, &response);
            if let Some(chain) = channel_chains.get_mut(name.as_str()) {
                chain.final_curve = Some((&ch.final_curve).into());
            }
        }

        applied_any |= inserted_for_channel;
    }

    applied_any
}

pub(in super::super) fn gd_phase_response_for_curve(
    freqs: &ndarray::Array1<f64>,
    delay_ms: f64,
    polarity_inverted: bool,
    ap_filters: &[Biquad],
    sample_rate: f64,
) -> Vec<Complex64> {
    freqs
        .iter()
        .map(|&f| {
            let mut h = Complex64::new(1.0, 0.0);
            if delay_ms.abs() > 1e-12 {
                h *= Complex64::from_polar(1.0, -2.0 * PI * f * delay_ms * 1e-3);
            }
            for ap in ap_filters {
                // Rebuild with the active sample rate so persisted filter
                // metadata and phase-curve reporting stay aligned.
                let ap = Biquad::new(ap.filter_type, ap.freq, sample_rate, ap.q, ap.db_gain);
                h *= ap.complex_response(f);
            }
            if polarity_inverted {
                h = -h;
            }
            h
        })
        .collect()
}
