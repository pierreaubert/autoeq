use autoeq_core::{Curve, Result, response};
use autoeq_optim::optim::OptimizerRunEvidence;
use log::info;
use roomeq_model::{ChannelDspChain, CurveData, PluginConfigWrapper};

use super::{FirChannelRequest, FirOptimizerOutput};
use crate::channel_result::{ChannelProcessingResult, GeneratedConvolutionSidecar};
use crate::{channel_target, output};

struct FirDspAssembly {
    plugins: Vec<PluginConfigWrapper>,
}

pub(super) fn assemble_fir_result(
    request: &FirChannelRequest<'_>,
    optimizer_output: FirOptimizerOutput,
    optimizer_evidence: Vec<OptimizerRunEvidence>,
) -> Result<ChannelProcessingResult> {
    let dsp = assemble_dsp_chain(request, &optimizer_output);
    let raw_curve = request.prepared.measurements().representative();
    let display_initial = output::extend_curve_to_full_range(raw_curve);
    let (final_curve, display_final) =
        corrected_curves(request, &optimizer_output, &display_initial);
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
        request.preprocessed.score_min_freq,
        request.target.max_freq,
    );
    info!(
        "  Pre-score: {:.6}, Post-score: {:.6}",
        request.target.pre_score, post_score
    );

    let mut initial_data: CurveData = (&display_initial).into();
    initial_data.norm_range = request.preprocessed.norm_range;
    let mut final_data: CurveData = (&display_final).into();
    final_data.norm_range = request.preprocessed.norm_range;
    let eq_response = output::compute_eq_response(&initial_data, &final_data);
    let filters = optimizer_output.eq_filters().to_vec();
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
        target_curve: None,
    };
    let (fir_coeffs, convolution_sidecar) = sidecar_output(optimizer_output);

    Ok(ChannelProcessingResult {
        channel,
        pre_score: request.target.pre_score,
        post_score,
        raw_pre_eq_curve: raw_curve.clone(),
        raw_post_eq_curve: final_curve,
        filters,
        mean_spl: request.target.mean_spl,
        arrival_time_ms: request.prepared.arrival_time_ms(),
        fir_coeffs,
        convolution_sidecar,
        optimizer_evidence,
    })
}

fn assemble_dsp_chain(
    request: &FirChannelRequest<'_>,
    optimizer_output: &FirOptimizerOutput,
) -> FirDspAssembly {
    let mut plugins = request.preprocessed.cea2034_plugins.clone();
    plugins.extend(request.preprocessed.broadband_plugins.iter().cloned());
    if !request.preprocessed.excursion_filters.is_empty() {
        plugins.push(output::create_labeled_eq_plugin(
            &request.preprocessed.excursion_filters,
            "excursion_protection",
        ));
    }
    match optimizer_output {
        FirOptimizerOutput::PhaseLinear {
            sidecar_reference, ..
        } => plugins.push(output::create_convolution_plugin(
            sidecar_reference.filename(),
        )),
        FirOptimizerOutput::Hybrid {
            eq_filters,
            sidecar_reference,
            ..
        } => {
            if !eq_filters.is_empty() {
                plugins.push(output::create_labeled_eq_plugin(
                    eq_filters,
                    "room_eq_correction",
                ));
            }
            plugins.push(output::create_convolution_plugin(
                sidecar_reference.filename(),
            ));
        }
        FirOptimizerOutput::MixedPhase {
            eq_filters,
            fir_coefficients,
            sidecar_reference,
            report,
        } => {
            if !eq_filters.is_empty() {
                plugins.push(output::create_labeled_eq_plugin(
                    eq_filters,
                    "room_eq_correction",
                ));
            }
            if fir_coefficients.is_some() {
                plugins.push(if let Some(report) = report {
                    output::create_mixed_phase_convolution_plugin(
                        sidecar_reference.filename(),
                        report,
                    )
                } else {
                    output::create_convolution_plugin(sidecar_reference.filename())
                });
            }
        }
    }
    let preference_filters = crate::channel_iir::preference_filters(
        request.room_config,
        request.target,
        request.sample_rate,
    );
    if !preference_filters.is_empty() {
        plugins.push(output::create_labeled_eq_plugin(
            &preference_filters,
            "user_preference",
        ));
    }
    FirDspAssembly { plugins }
}

fn corrected_curves(
    request: &FirChannelRequest<'_>,
    optimizer_output: &FirOptimizerOutput,
    display_initial: &Curve,
) -> (Curve, Curve) {
    let display_preprocessed = display_preprocessed_curve(request, display_initial);
    let preference_filters = crate::channel_iir::preference_filters(
        request.room_config,
        request.target,
        request.sample_rate,
    );
    match optimizer_output {
        FirOptimizerOutput::PhaseLinear { coefficients, .. } => {
            let response = response::compute_fir_complex_response(
                coefficients,
                &request.preprocessed.curve.freq,
                request.sample_rate,
            );
            let final_curve =
                response::apply_complex_response(&request.preprocessed.curve_for_optim, &response);
            let display_response = response::compute_fir_complex_response(
                coefficients,
                &display_preprocessed.freq,
                request.sample_rate,
            );
            let display_final =
                response::apply_complex_response(&display_preprocessed, &display_response);
            apply_preference_filters(
                final_curve,
                display_final,
                &preference_filters,
                request.sample_rate,
            )
        }
        FirOptimizerOutput::Hybrid {
            eq_filters,
            coefficients,
            ..
        } => {
            let iir_response = response::compute_peq_complex_response(
                eq_filters,
                &request.preprocessed.curve.freq,
                request.sample_rate,
            );
            let after_iir = response::apply_complex_response(
                &request.preprocessed.curve_for_optim,
                &iir_response,
            );
            let fir_response = response::compute_fir_complex_response(
                coefficients,
                &request.preprocessed.curve.freq,
                request.sample_rate,
            );
            let final_curve = response::apply_complex_response(&after_iir, &fir_response);

            let display_iir_response = response::compute_peq_complex_response(
                eq_filters,
                &display_preprocessed.freq,
                request.sample_rate,
            );
            let display_after_iir =
                response::apply_complex_response(&display_preprocessed, &display_iir_response);
            let display_fir_response = response::compute_fir_complex_response(
                coefficients,
                &display_preprocessed.freq,
                request.sample_rate,
            );
            let display_final =
                response::apply_complex_response(&display_after_iir, &display_fir_response);
            apply_preference_filters(
                final_curve,
                display_final,
                &preference_filters,
                request.sample_rate,
            )
        }
        FirOptimizerOutput::MixedPhase {
            eq_filters,
            fir_coefficients,
            ..
        } => {
            let eq_response = response::compute_peq_complex_response(
                eq_filters,
                &request.preprocessed.curve.freq,
                request.sample_rate,
            );
            let after_eq = response::apply_complex_response(
                &request.preprocessed.curve_for_optim,
                &eq_response,
            );
            let final_curve =
                apply_optional_fir(after_eq, fir_coefficients.as_deref(), request.sample_rate);

            let display_eq_response = response::compute_peq_complex_response(
                eq_filters,
                &display_preprocessed.freq,
                request.sample_rate,
            );
            let display_after_eq =
                response::apply_complex_response(&display_preprocessed, &display_eq_response);
            let display_final = apply_optional_fir(
                display_after_eq,
                fir_coefficients.as_deref(),
                request.sample_rate,
            );
            apply_preference_filters(
                final_curve,
                display_final,
                &preference_filters,
                request.sample_rate,
            )
        }
    }
}

fn display_preprocessed_curve(request: &FirChannelRequest<'_>, display_initial: &Curve) -> Curve {
    let mut curve = display_initial.clone();
    curve.spl += request.preprocessed.broadband_mean_shift;
    let mut filters = request.preprocessed.excursion_filters.clone();
    filters.extend(request.preprocessed.cea2034_filters.iter().cloned());
    filters.extend(request.preprocessed.broadband_biquads.iter().cloned());
    if filters.is_empty() {
        curve
    } else {
        let response =
            response::compute_peq_complex_response(&filters, &curve.freq, request.sample_rate);
        response::apply_complex_response(&curve, &response)
    }
}

fn apply_preference_filters(
    final_curve: Curve,
    display_final: Curve,
    filters: &[math_audio_iir_fir::Biquad],
    sample_rate: f64,
) -> (Curve, Curve) {
    if filters.is_empty() {
        return (final_curve, display_final);
    }
    let final_response =
        response::compute_peq_complex_response(filters, &final_curve.freq, sample_rate);
    let display_response =
        response::compute_peq_complex_response(filters, &display_final.freq, sample_rate);
    (
        response::apply_complex_response(&final_curve, &final_response),
        response::apply_complex_response(&display_final, &display_response),
    )
}

fn apply_optional_fir(curve: Curve, coefficients: Option<&[f64]>, sample_rate: f64) -> Curve {
    let Some(coefficients) = coefficients else {
        return curve;
    };
    let fir_response =
        response::compute_fir_complex_response(coefficients, &curve.freq, sample_rate);
    response::apply_complex_response(&curve, &fir_response)
}

fn sidecar_output(
    optimizer_output: FirOptimizerOutput,
) -> (Option<Vec<f64>>, Option<GeneratedConvolutionSidecar>) {
    match optimizer_output {
        FirOptimizerOutput::PhaseLinear {
            coefficients,
            sidecar_reference,
        }
        | FirOptimizerOutput::Hybrid {
            coefficients,
            sidecar_reference,
            ..
        } => (
            Some(coefficients),
            Some(GeneratedConvolutionSidecar {
                reference: sidecar_reference,
                required: true,
            }),
        ),
        FirOptimizerOutput::MixedPhase {
            fir_coefficients,
            sidecar_reference,
            ..
        } => {
            let sidecar = fir_coefficients
                .as_ref()
                .map(|_| GeneratedConvolutionSidecar {
                    reference: sidecar_reference,
                    required: false,
                });
            (fir_coefficients, sidecar)
        }
    }
}
