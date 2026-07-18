use autoeq_core::{Curve, Result, normalize_and_interpolate_response, response};
use autoeq_optim::optim::OptimizerRunEvidence;
use log::info;
use math_audio_iir_fir::Biquad;
use ndarray::Array1;
use roomeq_model::{ChannelDspChain, CurveData, PluginConfigWrapper};

use super::{IirChannelRequest, IirChannelResult, IirOptimizerOutput};
use crate::{channel_target, output};

struct DspAssembly {
    plugins: Vec<PluginConfigWrapper>,
    filters: Vec<Biquad>,
}

pub(super) fn assemble_iir_result(
    request: &IirChannelRequest<'_>,
    optimizer_output: IirOptimizerOutput,
    optimizer_evidence: Vec<OptimizerRunEvidence>,
) -> Result<IirChannelResult> {
    let dsp = assemble_dsp_chain(request, &optimizer_output);
    let raw_curve = request.prepared.measurements().representative();
    let mut score_input = raw_curve.clone();
    score_input.spl += request.preprocessed.broadband_mean_shift;
    let response = response::compute_peq_complex_response(
        &dsp.filters,
        &score_input.freq,
        request.sample_rate,
    );
    let final_curve = response::apply_complex_response(&score_input, &response);

    let display_initial = output::extend_curve_to_full_range(raw_curve);
    let mut display_input = display_initial.clone();
    display_input.spl += request.preprocessed.broadband_mean_shift;
    let display_response = response::compute_peq_complex_response(
        &dsp.filters,
        &display_input.freq,
        request.sample_rate,
    );
    let display_final = response::apply_complex_response(&display_input, &display_response);

    let score_curve = if let Some(tilt_curve) = &request.target.target_tilt_curve {
        Curve {
            freq: final_curve.freq.clone(),
            spl: &final_curve.spl - &tilt_curve.spl,
            phase: final_curve.phase.clone(),
            ..Curve::default()
        }
    } else {
        final_curve.clone()
    };
    let post_score = channel_target::flatness_score_in_range(
        &score_curve,
        request.target.min_freq,
        request.target.max_freq,
    );
    info!(
        "  Pre-score: {:.6}, Post-score: {:.6}",
        request.target.pre_score, post_score
    );

    let norm_range = request.preprocessed.norm_range;
    let mut initial_data: CurveData = (&display_initial).into();
    initial_data.norm_range = norm_range;
    let mut final_data: CurveData = (&display_final).into();
    final_data.norm_range = norm_range;
    let eq_response = output::compute_eq_response(&initial_data, &final_data);
    let target_curve = Some(display_target_curve(request, &display_initial, norm_range));
    let report_filters = optimizer_output.eq_filters().to_vec();
    let channel = ChannelDspChain {
        channel: request.channel_name.to_string(),
        plugins: dsp.plugins,
        drivers: None,
        initial_curve: Some(initial_data),
        final_curve: Some(final_data),
        eq_response: Some(eq_response),
        pre_ir: None,
        post_ir: None,
        fir_temporal_masking: None,
        direct_early_late_correction: None,
        target_curve,
    };

    Ok(IirChannelResult {
        channel,
        pre_score: request.target.pre_score,
        post_score,
        raw_pre_eq_curve: raw_curve.clone(),
        raw_post_eq_curve: final_curve,
        filters: report_filters,
        mean_spl: request.target.mean_spl,
        arrival_time_ms: request.prepared.arrival_time_ms(),
        optimizer_evidence,
    })
}

fn assemble_dsp_chain(
    request: &IirChannelRequest<'_>,
    optimizer_output: &IirOptimizerOutput,
) -> DspAssembly {
    let mut pre_eq_plugins = request.preprocessed.cea2034_plugins.clone();
    pre_eq_plugins.extend(request.preprocessed.broadband_plugins.iter().cloned());
    let mut eq_plugins = Vec::new();
    let mut post_eq_plugins = Vec::new();

    match optimizer_output {
        IirOptimizerOutput::LowLatency {
            eq_filters,
            preference_filters,
        } => {
            let mut room_filters = request.preprocessed.excursion_filters.clone();
            room_filters.extend(eq_filters.iter().cloned());
            if !room_filters.is_empty() {
                eq_plugins.push(output::create_labeled_eq_plugin(
                    &room_filters,
                    "room_eq_correction",
                ));
            }
            append_preference_plugin(&mut post_eq_plugins, preference_filters);
        }
        IirOptimizerOutput::WarpedIir {
            eq_filters,
            preference_filters,
            warped_lambda,
        } => {
            if !eq_filters.is_empty() || !request.preprocessed.excursion_filters.is_empty() {
                eq_plugins.push(output::create_warped_eq_plugin(
                    &request.preprocessed.excursion_filters,
                    eq_filters,
                    Some(*warped_lambda),
                ));
            }
            append_preference_plugin(&mut post_eq_plugins, preference_filters);
        }
        IirOptimizerOutput::KautzModal {
            kautz_sections,
            preference_filters,
            ..
        } => {
            let mut filter_configs: Vec<serde_json::Value> = request
                .preprocessed
                .excursion_filters
                .iter()
                .map(output::biquad_to_json)
                .collect();
            filter_configs.push(create_kautz_filter_config(kautz_sections));
            eq_plugins.push(output::create_labeled_eq_plugin_from_filter_configs(
                filter_configs,
                "kautz_modal",
            ));
            append_preference_plugin(&mut post_eq_plugins, preference_filters);
        }
    }

    let mut plugins =
        Vec::with_capacity(pre_eq_plugins.len() + eq_plugins.len() + post_eq_plugins.len());
    plugins.extend(pre_eq_plugins);
    plugins.extend(eq_plugins);
    plugins.extend(post_eq_plugins);

    let mut filters = request.preprocessed.excursion_filters.clone();
    filters.extend(request.preprocessed.cea2034_filters.iter().cloned());
    filters.extend(request.preprocessed.broadband_biquads.iter().cloned());
    filters.extend(optimizer_output.eq_filters().iter().cloned());
    match optimizer_output {
        IirOptimizerOutput::LowLatency {
            preference_filters, ..
        }
        | IirOptimizerOutput::WarpedIir {
            preference_filters, ..
        }
        | IirOptimizerOutput::KautzModal {
            preference_filters, ..
        } => filters.extend(preference_filters.iter().cloned()),
    }

    DspAssembly { plugins, filters }
}

fn append_preference_plugin(plugins: &mut Vec<PluginConfigWrapper>, preference_filters: &[Biquad]) {
    if !preference_filters.is_empty() {
        plugins.push(output::create_labeled_eq_plugin(
            preference_filters,
            "user_preference",
        ));
    }
}

fn display_target_curve(
    request: &IirChannelRequest<'_>,
    display_initial: &Curve,
    norm_range: Option<(f64, f64)>,
) -> CurveData {
    let spl = if let Some(tilt_curve) = &request.target.target_tilt_curve {
        let display_tilt = normalize_and_interpolate_response(&display_initial.freq, tilt_curve);
        &display_tilt.spl + request.target.mean_spl
    } else {
        Array1::from_elem(display_initial.freq.len(), request.target.mean_spl)
    };
    CurveData {
        freq: display_initial.freq.to_vec(),
        spl: spl.to_vec(),
        phase: None,
        norm_range,
    }
}

pub(super) fn create_kautz_filter_config(sections: &[(f64, f64, f64)]) -> serde_json::Value {
    let kautz_sections: Vec<serde_json::Value> = sections
        .iter()
        .map(|(pole_frequency, q, gain)| {
            serde_json::json!({
                "pole_freq": pole_frequency,
                "q": q,
                "gain": gain,
            })
        })
        .collect();
    let (frequency, q, _) = sections.first().copied().unwrap_or((100.0, 1.0, 0.0));
    serde_json::json!({
        "topology": "kautz_filter",
        "filter_type": "peak",
        "freq": frequency,
        "q": q,
        "db_gain": 0.0,
        "kautz_sections": kautz_sections,
    })
}
