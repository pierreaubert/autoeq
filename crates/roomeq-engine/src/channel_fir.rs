//! Path-free FIR and mixed-phase processing for one prepared RoomEQ channel.

mod assemble;
#[cfg(test)]
mod tests;

use autoeq_core::{AutoeqError, Result, response};
use autoeq_optim::optim::{OptimProgressCallback, OptimizerRunEvidence};
use log::{info, warn};
use math_audio_iir_fir::Biquad;
use ndarray::Array1;
use roomeq_model::{OptimizerConfig, RoomConfig};

use crate::PreparedChannelInput;
use crate::channel_preprocessing::PreprocessedFeatures;
use crate::channel_result::{
    ChannelProcessingResult, ConvolutionSidecarReference, subtract_target_tilt,
};
use crate::channel_target::TargetContext;
use crate::eq::{self, EqResources};

/// Artifact-producing generic channel modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FirChannelMode {
    PhaseLinear,
    Hybrid,
    MixedPhase,
}

/// Complete path-free request for one FIR-capable channel run.
pub struct FirChannelRequest<'a> {
    pub mode: FirChannelMode,
    pub channel_name: &'a str,
    pub prepared: &'a PreparedChannelInput,
    pub room_config: &'a RoomConfig,
    pub sample_rate: f64,
    pub target: &'a TargetContext,
    pub preprocessed: &'a PreprocessedFeatures,
    pub optimizer: &'a OptimizerConfig,
    pub eq_resources: &'a EqResources,
    pub sidecar_reference: ConvolutionSidecarReference,
    pub callback: Option<OptimProgressCallback>,
}

pub(super) enum FirOptimizerOutput {
    PhaseLinear {
        coefficients: Vec<f64>,
        sidecar_reference: ConvolutionSidecarReference,
    },
    Hybrid {
        eq_filters: Vec<Biquad>,
        coefficients: Vec<f64>,
        sidecar_reference: ConvolutionSidecarReference,
    },
    MixedPhase {
        eq_filters: Vec<Biquad>,
        fir_coefficients: Option<Vec<f64>>,
        sidecar_reference: ConvolutionSidecarReference,
        report: Option<crate::mixed_phase::MixedPhaseCorrectionReport>,
    },
}

impl FirOptimizerOutput {
    pub(super) fn eq_filters(&self) -> &[Biquad] {
        match self {
            Self::PhaseLinear { .. } => &[],
            Self::Hybrid { eq_filters, .. } | Self::MixedPhase { eq_filters, .. } => eq_filters,
        }
    }
}

/// Optimize and assemble one generic artifact-producing channel.
///
/// The returned coefficients are still in memory. The workflow owns sidecar
/// persistence and matches them to the returned logical reference.
pub fn process_fir_channel(request: FirChannelRequest<'_>) -> Result<ChannelProcessingResult> {
    match request.mode {
        FirChannelMode::PhaseLinear => process_phase_linear(request),
        FirChannelMode::Hybrid => process_hybrid(request),
        FirChannelMode::MixedPhase => process_mixed_phase(request),
    }
}

fn process_phase_linear(request: FirChannelRequest<'_>) -> Result<ChannelProcessingResult> {
    info!("  Generating FIR filter...");
    let input_curve = subtract_target_tilt(&request.preprocessed.curve_for_optim, request.target);
    let coefficients = crate::fir::generate_fir_correction_with_resources(
        &input_curve,
        request.optimizer,
        request.eq_resources,
        request.sample_rate,
    )
    .map_err(|error| AutoeqError::OptimizationFailed {
        message: format!("FIR generation failed: {error}"),
    })?;
    assemble::assemble_fir_result(
        &request,
        FirOptimizerOutput::PhaseLinear {
            coefficients,
            sidecar_reference: request.sidecar_reference.clone(),
        },
        request.preprocessed.optimizer_evidence.clone(),
    )
}

fn process_hybrid(mut request: FirChannelRequest<'_>) -> Result<ChannelProcessingResult> {
    let optimization_curve =
        subtract_target_tilt(&request.preprocessed.curve_for_optim, request.target);
    let eq_result = if let Some(callback) = request.callback.take() {
        eq::optimize_channel_eq_with_callback_detailed(
            &optimization_curve,
            request.optimizer,
            Some(request.eq_resources),
            request.sample_rate,
            callback,
        )
    } else {
        eq::optimize_channel_eq_detailed(
            &optimization_curve,
            request.optimizer,
            Some(request.eq_resources),
            request.sample_rate,
        )
    }
    .map_err(|error| AutoeqError::OptimizationFailed {
        message: format!(
            "IIR optimization failed for channel {}: {error}",
            request.channel_name
        ),
    })?;
    info!("  IIR stage: {} filters", eq_result.filters.len());

    let iir_response = response::compute_peq_complex_response(
        &eq_result.filters,
        &request.preprocessed.curve.freq,
        request.sample_rate,
    );
    let residual_curve =
        response::apply_complex_response(&request.preprocessed.curve, &iir_response);
    let coefficients = crate::fir::generate_fir_correction_with_resources(
        &residual_curve,
        request.optimizer,
        request.eq_resources,
        request.sample_rate,
    )
    .map_err(|error| AutoeqError::OptimizationFailed {
        message: format!("FIR generation failed: {error}"),
    })?;
    assemble::assemble_fir_result(
        &request,
        FirOptimizerOutput::Hybrid {
            eq_filters: eq_result.filters,
            coefficients,
            sidecar_reference: request.sidecar_reference.clone(),
        },
        with_preprocessing_evidence(request.preprocessed, eq_result.optimizer_evidence),
    )
}

fn process_mixed_phase(mut request: FirChannelRequest<'_>) -> Result<ChannelProcessingResult> {
    let optimization_curve =
        subtract_target_tilt(&request.preprocessed.curve_for_optim, request.target);
    let eq_result = crate::channel_optimizer::optimize_maybe_multi(
        request.channel_name,
        request.prepared,
        &optimization_curve,
        request.optimizer,
        request.eq_resources,
        request.sample_rate,
        request.callback.take(),
        request.target.target_tilt_curve.as_ref(),
    )?;
    info!("  IIR stage: {} filters", eq_result.filters.len());

    let mixed_config = request
        .room_config
        .optimizer
        .mixed_phase
        .as_ref()
        .map(|config| crate::mixed_phase::MixedPhaseConfig {
            max_fir_length_ms: config.max_fir_length_ms,
            pre_ringing_threshold_db: config.pre_ringing_threshold_db,
            min_spatial_depth: config.min_spatial_depth,
            phase_smoothing_octaves: config.phase_smoothing_octaves,
        })
        .unwrap_or_default();
    let spatial_depth = spatial_depth(&request);
    let generated = if request.preprocessed.curve_for_optim.phase.is_some() {
        match crate::mixed_phase::decompose_phase(
            &request.preprocessed.curve_for_optim,
            &mixed_config,
        ) {
            Ok((_minimum_phase, _excess_phase, delay_ms, residual)) => {
                info!(
                    "  Mixed-phase: delay={:.2} ms, generating excess phase FIR...",
                    delay_ms
                );
                let coefficients = crate::mixed_phase::generate_excess_phase_fir_with_depth(
                    &request.preprocessed.curve_for_optim.freq,
                    &residual,
                    &mixed_config,
                    request.sample_rate,
                    spatial_depth.as_ref(),
                );
                let report = crate::mixed_phase::MixedPhaseCorrectionReport::from_residual(
                    delay_ms,
                    coefficients.len(),
                    &residual,
                );
                Some((coefficients, report))
            }
            Err(error) => {
                warn!(
                    "  Mixed-phase decomposition failed for '{}': {}. Using IIR only.",
                    request.channel_name, error
                );
                None
            }
        }
    } else {
        info!(
            "  No phase data for '{}', using IIR only (skipping excess phase FIR).",
            request.channel_name
        );
        None
    };
    let (fir_coefficients, report) = generated
        .map(|(coefficients, report)| (Some(coefficients), Some(report)))
        .unwrap_or((None, None));
    let result = assemble::assemble_fir_result(
        &request,
        FirOptimizerOutput::MixedPhase {
            eq_filters: eq_result.filters,
            fir_coefficients,
            sidecar_reference: request.sidecar_reference.clone(),
            report,
        },
        with_preprocessing_evidence(request.preprocessed, eq_result.optimizer_evidence),
    )?;
    info!(
        "  Mixed-phase result: pre={:.6}, post={:.6}",
        result.pre_score, result.post_score
    );
    Ok(result)
}

fn spatial_depth(request: &FirChannelRequest<'_>) -> Option<Array1<f64>> {
    if !request
        .prepared
        .measurements()
        .is_multi_measurement_source()
    {
        return None;
    }
    let curves = request.prepared.measurements().individual();
    if curves.len() <= 1 {
        return None;
    }
    let config = request
        .room_config
        .optimizer
        .multi_measurement
        .as_ref()
        .and_then(|config| config.spatial_robustness.as_ref())
        .map(
            |config| roomeq_analysis::spatial_robustness::SpatialRobustnessConfig {
                variance_threshold_db: config.variance_threshold_db,
                transition_width_db: config.transition_width_db,
                min_correction_depth: config.min_correction_depth,
                mask_smoothing_octaves: config.mask_smoothing_octaves,
            },
        )
        .unwrap_or_default();
    let weights = request
        .room_config
        .optimizer
        .multi_measurement
        .as_ref()
        .and_then(|config| config.weights.as_deref());
    match roomeq_analysis::spatial_robustness::analyze_spatial_robustness_weighted(
        curves, &config, weights,
    ) {
        Ok(analysis) => {
            info!(
                "  Spatial depth for mixed-phase: mean={:.2}",
                analysis.correction_depth.iter().sum::<f64>()
                    / analysis.correction_depth.len() as f64
            );
            Some(analysis.correction_depth)
        }
        Err(error) => {
            warn!("  Spatial robustness analysis skipped: {error}");
            None
        }
    }
}

fn with_preprocessing_evidence(
    preprocessed: &PreprocessedFeatures,
    mut optimizer_evidence: Vec<OptimizerRunEvidence>,
) -> Vec<OptimizerRunEvidence> {
    let mut combined = preprocessed.optimizer_evidence.clone();
    combined.append(&mut optimizer_evidence);
    combined
}
