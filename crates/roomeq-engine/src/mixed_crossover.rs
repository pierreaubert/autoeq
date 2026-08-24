//! Path-free frequency-split FIR/IIR processing for one prepared channel.

use autoeq_core::{AutoeqError, Curve, Result, response};
use autoeq_optim::optim::OptimProgressCallback;
use log::info;
use math_audio_iir_fir::Biquad;
use num_complex::Complex64;
use roomeq_analysis::crossover_utils::{
    compute_lr24_crossover_responses, split_curve_at_frequency,
};
use roomeq_model::{CurveData, MixedModeConfig, OptimizerConfig};

use crate::channel_result::subtract_target_tilt;
use crate::channel_result::{
    ChannelProcessingResult, ConvolutionSidecarReference, GeneratedConvolutionSidecar,
};
use crate::channel_target::TargetContext;
use crate::eq::{self, EqResources};
use crate::output;

/// Complete in-memory request for the legacy frequency-split Hybrid topology.
pub struct MixedCrossoverRequest<'a> {
    pub channel_name: &'a str,
    pub curve: &'a Curve,
    pub target: &'a TargetContext,
    pub preference_filters: &'a [Biquad],
    pub mixed_config: &'a MixedModeConfig,
    pub optimizer: &'a OptimizerConfig,
    pub eq_resources: &'a EqResources,
    pub sample_rate: f64,
    pub min_freq: f64,
    pub max_freq: f64,
    pub mean_spl: f64,
    pub pre_score: f64,
    pub arrival_time_ms: Option<f64>,
    pub sidecar_reference: ConvolutionSidecarReference,
    pub callback: Option<OptimProgressCallback>,
}

/// Process the group-specific Hybrid crossover without filesystem access.
pub fn process_mixed_crossover(
    request: MixedCrossoverRequest<'_>,
) -> Result<ChannelProcessingResult> {
    let crossover_freq = request.mixed_config.crossover_freq;
    let fir_uses_low = request.mixed_config.fir_band.to_lowercase() == "low";
    info!(
        "  Mixed mode crossover at {} Hz (FIR on {} band, IIR on {} band)",
        crossover_freq,
        if fir_uses_low { "low" } else { "high" },
        if fir_uses_low { "high" } else { "low" }
    );

    let optimization_curve = subtract_target_tilt(request.curve, request.target);
    let (low_curve, high_curve) = split_curve_at_frequency(&optimization_curve, crossover_freq);
    let (fir_curve, iir_curve) = if fir_uses_low {
        (&low_curve, &high_curve)
    } else {
        (&high_curve, &low_curve)
    };
    let fir_min_freq = fir_curve.freq.first().copied().unwrap_or(request.min_freq);
    let fir_max_freq = fir_curve.freq.last().copied().unwrap_or(crossover_freq);
    let iir_min_freq = iir_curve.freq.first().copied().unwrap_or(crossover_freq);
    let iir_max_freq = iir_curve.freq.last().copied().unwrap_or(request.max_freq);
    info!(
        "  FIR band: {:.1}-{:.1} Hz, IIR band: {:.1}-{:.1} Hz",
        fir_min_freq, fir_max_freq, iir_min_freq, iir_max_freq
    );

    let iir_config = OptimizerConfig {
        min_freq: iir_min_freq,
        max_freq: iir_max_freq,
        ..request.optimizer.clone()
    };
    let eq_result = if let Some(callback) = request.callback {
        eq::optimize_channel_eq_with_callback_detailed(
            iir_curve,
            &iir_config,
            Some(request.eq_resources),
            request.sample_rate,
            callback,
        )
    } else {
        eq::optimize_channel_eq_detailed(
            iir_curve,
            &iir_config,
            Some(request.eq_resources),
            request.sample_rate,
        )
    }
    .map_err(|error| AutoeqError::OptimizationFailed {
        message: format!(
            "IIR optimization failed for {} band: {error}",
            if fir_uses_low { "high" } else { "low" }
        ),
    })?;
    info!(
        "  IIR stage: {} filters for {} band",
        eq_result.filters.len(),
        if fir_uses_low { "high" } else { "low" }
    );

    let fir_config = OptimizerConfig {
        min_freq: fir_min_freq,
        max_freq: fir_max_freq,
        ..request.optimizer.clone()
    };
    let fir_coefficients = crate::fir::generate_fir_correction_with_resources(
        fir_curve,
        &fir_config,
        request.eq_resources,
        request.sample_rate,
    )
    .map_err(|error| AutoeqError::OptimizationFailed {
        message: format!(
            "FIR generation failed for {} band: {error}",
            if fir_uses_low { "low" } else { "high" }
        ),
    })?;
    let fir_bulk_delay_ms = fir_bulk_delay_ms(&fir_coefficients, request.sample_rate);

    let eq_filters = eq_result.filters;
    let mut reported_filters = eq_filters.clone();
    reported_filters.extend_from_slice(request.preference_filters);
    let mut channel = output::build_mixed_mode_crossover_chain_with_post_merge_eq(
        request.channel_name,
        request.mixed_config,
        &eq_filters,
        request.preference_filters,
        request.sidecar_reference.filename(),
        fir_uses_low,
        fir_bulk_delay_ms,
        None,
    );
    let neutral_output_curve = apply_crossover_response(
        request.curve,
        &eq_filters,
        &[],
        &fir_coefficients,
        crossover_freq,
        fir_uses_low,
        fir_bulk_delay_ms,
        request.sample_rate,
    );
    let final_curve = apply_crossover_response(
        request.curve,
        &eq_filters,
        request.preference_filters,
        &fir_coefficients,
        crossover_freq,
        fir_uses_low,
        fir_bulk_delay_ms,
        request.sample_rate,
    );
    let neutral_final = subtract_target_tilt(&neutral_output_curve, request.target);
    let (norm_range, mean_final) =
        roomeq_analysis::response_metrics::detect_passband_and_mean(&neutral_final);
    let normalized_final_spl = &neutral_final.spl - mean_final;
    let post_score = autoeq_optim::loss::flat_loss(
        &final_curve.freq,
        &normalized_final_spl,
        request.min_freq,
        request.max_freq,
    );
    info!(
        "  Pre-score: {:.6}, Post-score: {:.6}",
        request.pre_score, post_score
    );

    let display_initial = output::extend_curve_to_full_range(request.curve);
    let display_final = apply_crossover_response(
        &display_initial,
        &eq_filters,
        request.preference_filters,
        &fir_coefficients,
        crossover_freq,
        fir_uses_low,
        fir_bulk_delay_ms,
        request.sample_rate,
    );
    let mut initial_data: CurveData = (&display_initial).into();
    initial_data.norm_range = norm_range;
    let mut final_data: CurveData = (&display_final).into();
    final_data.norm_range = norm_range;
    channel.initial_curve = Some(initial_data.clone());
    channel.final_curve = Some(final_data.clone());
    channel.eq_response = Some(output::compute_eq_response(&initial_data, &final_data));

    Ok(ChannelProcessingResult {
        channel,
        pre_score: request.pre_score,
        post_score,
        raw_pre_eq_curve: request.curve.clone(),
        raw_post_eq_curve: final_curve,
        filters: reported_filters,
        mean_spl: request.mean_spl,
        arrival_time_ms: request.arrival_time_ms,
        fir_coeffs: Some(fir_coefficients),
        convolution_sidecar: Some(GeneratedConvolutionSidecar {
            reference: request.sidecar_reference,
            required: true,
        }),
        optimizer_evidence: eq_result.optimizer_evidence,
    })
}

#[allow(clippy::too_many_arguments)]
fn apply_crossover_response(
    curve: &Curve,
    eq_filters: &[math_audio_iir_fir::Biquad],
    post_merge_filters: &[math_audio_iir_fir::Biquad],
    fir_coefficients: &[f64],
    crossover_freq: f64,
    fir_uses_low: bool,
    fir_bulk_delay_ms: f64,
    sample_rate: f64,
) -> Curve {
    let iir_response = response::compute_peq_complex_response(eq_filters, &curve.freq, sample_rate);
    let post_merge_response =
        response::compute_peq_complex_response(post_merge_filters, &curve.freq, sample_rate);
    let fir_response =
        response::compute_fir_complex_response(fir_coefficients, &curve.freq, sample_rate);
    let (lowpass, highpass) =
        compute_lr24_crossover_responses(&curve.freq, crossover_freq, sample_rate);
    let combined = lowpass
        .iter()
        .zip(highpass.iter())
        .zip(fir_response.iter().zip(iir_response.iter()))
        .zip(curve.freq.iter())
        .map(|(((lowpass, highpass), (fir, iir)), frequency)| {
            let iir_delay = Complex64::from_polar(
                1.0,
                -2.0 * std::f64::consts::PI * frequency * fir_bulk_delay_ms / 1000.0,
            );
            if fir_uses_low {
                lowpass * fir + highpass * iir * iir_delay
            } else {
                lowpass * iir * iir_delay + highpass * fir
            }
        })
        .collect::<Vec<_>>();
    let combined = combined
        .into_iter()
        .zip(post_merge_response)
        .map(|(mixed, post_merge)| mixed * post_merge)
        .collect::<Vec<_>>();
    response::apply_complex_response(curve, &combined)
}

fn fir_bulk_delay_ms(coefficients: &[f64], sample_rate: f64) -> f64 {
    if sample_rate <= 0.0 {
        return 0.0;
    }
    coefficients
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index as f64 * 1000.0 / sample_rate)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use roomeq_model::{FirConfig, MixedModeConfig};

    use super::*;

    fn curve() -> Curve {
        let freq = Array1::logspace(10.0, f64::log10(100.0), f64::log10(1_600.0), 64);
        Curve {
            freq: freq.clone(),
            spl: Array1::from_elem(freq.len(), 80.0),
            ..Curve::default()
        }
    }

    #[test]
    fn mixed_crossover_returns_required_path_free_sidecar() {
        let curve = curve();
        let mixed_config = MixedModeConfig {
            crossover_freq: 500.0,
            fir_band: "low".to_string(),
            ..MixedModeConfig::default()
        };
        let optimizer = OptimizerConfig {
            min_freq: 100.0,
            max_freq: 1_600.0,
            num_filters: 1,
            max_iter: 3,
            population: 4,
            seed: Some(2),
            refine: false,
            fir: Some(FirConfig {
                taps: 64,
                ..FirConfig::default()
            }),
            ..OptimizerConfig::default()
        };
        let target = TargetContext {
            target_tilt_curve: None,
            min_freq: 100.0,
            max_freq: 1_600.0,
            pre_score: 1.0,
            mean_spl: 80.0,
            cea2034_active: false,
        };
        let result = process_mixed_crossover(MixedCrossoverRequest {
            channel_name: "left",
            curve: &curve,
            target: &target,
            preference_filters: &[],
            mixed_config: &mixed_config,
            optimizer: &optimizer,
            eq_resources: &EqResources::default(),
            sample_rate: 48_000.0,
            min_freq: 100.0,
            max_freq: 1_600.0,
            mean_spl: 80.0,
            pre_score: 1.0,
            arrival_time_ms: Some(2.5),
            sidecar_reference: ConvolutionSidecarReference::new("left_band_fir_48000hz.wav")
                .unwrap(),
            callback: None,
        })
        .unwrap();

        assert_eq!(result.arrival_time_ms, Some(2.5));
        assert_eq!(result.fir_coeffs.as_ref().unwrap().len(), 64);
        let generated = result.convolution_sidecar.unwrap();
        assert!(generated.required);
        assert_eq!(generated.reference.filename(), "left_band_fir_48000hz.wav");
        assert_eq!(result.channel.plugins[0].plugin_type, "band_split");
        assert!(result.channel.plugins.iter().any(|plugin| {
            plugin.plugin_type == "convolution"
                && plugin.parameters["ir_file"] == "left_band_fir_48000hz.wav"
        }));
        let alignment = result
            .channel
            .plugins
            .iter()
            .find(|plugin| plugin.parameters["label"] == "hybrid_fir_latency_alignment")
            .expect("hybrid branch alignment delay");
        assert_eq!(alignment.plugin_type, "delay");
        assert_eq!(alignment.parameters["channels"], serde_json::json!([2, 3]));
        assert_eq!(
            alignment.parameters["delay_ms"].as_f64().unwrap(),
            fir_bulk_delay_ms(result.fir_coeffs.as_ref().unwrap(), 48_000.0)
        );
        assert_eq!(
            result.channel.plugins.last().unwrap().plugin_type,
            "band_merge"
        );
    }

    #[test]
    fn mixed_crossover_applies_preference_eq_after_full_band_merge() {
        let curve = curve();
        let preference = [Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Lowshelf,
            250.0,
            48_000.0,
            0.707,
            4.0,
        )];
        let neutral =
            apply_crossover_response(&curve, &[], &[], &[1.0], 500.0, true, 0.0, 48_000.0);
        let voiced =
            apply_crossover_response(&curve, &[], &preference, &[1.0], 500.0, true, 0.0, 48_000.0);
        let preference_response =
            response::compute_peq_complex_response(&preference, &curve.freq, 48_000.0);
        for ((voiced_level, neutral_level), expected) in voiced
            .spl
            .iter()
            .zip(neutral.spl.iter())
            .zip(preference_response.iter())
        {
            let expected_db = 20.0 * expected.norm().max(1e-12).log10();
            assert!((voiced_level - neutral_level - expected_db).abs() < 1e-9);
        }

        let chain = output::build_mixed_mode_crossover_chain_with_post_merge_eq(
            "left",
            &MixedModeConfig {
                crossover_freq: 500.0,
                ..MixedModeConfig::default()
            },
            &[],
            &preference,
            "left_fir.wav",
            true,
            0.0,
            None,
        );
        let merge_index = chain
            .plugins
            .iter()
            .position(|plugin| plugin.plugin_type == "band_merge")
            .expect("band merge");
        let preference_index = chain
            .plugins
            .iter()
            .rposition(|plugin| plugin.plugin_type == "eq")
            .expect("preference EQ");
        assert!(preference_index > merge_index);
    }

    #[test]
    fn fir_bulk_delay_tracks_the_impulse_peak() {
        assert_eq!(fir_bulk_delay_ms(&[0.1, -0.2, 1.0, 0.4], 1_000.0), 2.0);
        assert_eq!(fir_bulk_delay_ms(&[], 48_000.0), 0.0);
        assert_eq!(fir_bulk_delay_ms(&[1.0], 0.0), 0.0);
    }
}
