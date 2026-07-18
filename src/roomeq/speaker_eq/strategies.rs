//! Processing-mode strategies for single-speaker room EQ.
//!
//! Artifact-producing modes remain local strategies until their sidecar
//! boundary moves. Artifact-free IIR modes delegate through one engine adapter.

use super::apply::{assemble_channel_report, assemble_dsp_chain, build_mixed_mode_result};
use super::misc::optimize_eq_maybe_multi;
use super::types::{
    ChannelOptimizationInput, MixedModeResult, OptimizerOutput, PreparedMeasurement,
    PreprocessedFeatures, TargetContext,
};
use crate::Curve;
use crate::error::{AutoeqError, Result};
use crate::response;
use crate::roomeq::types::{OptimizerConfig, ProcessingMode};
use crate::roomeq::{artifacts, fir, group_processing};
use log::{info, warn};
use roomeq_engine::channel_iir::{
    IirChannelMode, IirChannelRequest, IirChannelResult, process_iir_channel,
};
use roomeq_engine::eq::{self as engine_eq, EqResources};

fn with_preprocessing_evidence(
    preprocessed: &PreprocessedFeatures,
    mut optimizer_evidence: Vec<crate::optim::OptimizerRunEvidence>,
) -> Vec<crate::optim::OptimizerRunEvidence> {
    let mut combined = preprocessed.optimizer_evidence.clone();
    combined.append(&mut optimizer_evidence);
    combined
}

/// Strategy trait for processing a single speaker according to a processing mode.
pub trait ChannelProcessingStrategy {
    /// Run the processing pipeline for this mode and return the assembled result.
    fn process(
        &self,
        input: &mut ChannelOptimizationInput,
        prepared: &PreparedMeasurement,
        target: &TargetContext,
        preprocessed: &PreprocessedFeatures,
        clamped_optimizer: &OptimizerConfig,
        eq_resources: &EqResources,
    ) -> Result<MixedModeResult>;
}

/// Factory that maps a [`ProcessingMode`] to its strategy implementation.
pub fn strategy_for_mode(mode: ProcessingMode) -> Box<dyn ChannelProcessingStrategy> {
    match mode {
        ProcessingMode::PhaseLinear => Box::new(PhaseLinearStrategy),
        ProcessingMode::Hybrid => Box::new(HybridStrategy),
        ProcessingMode::MixedPhase => Box::new(MixedPhaseStrategy),
        ProcessingMode::LowLatency => Box::new(EngineIirStrategy {
            mode: IirChannelMode::LowLatency,
        }),
        ProcessingMode::WarpedIir => Box::new(EngineIirStrategy {
            mode: IirChannelMode::WarpedIir,
        }),
        ProcessingMode::KautzModal => Box::new(EngineIirStrategy {
            mode: IirChannelMode::KautzModal,
        }),
    }
}

/// Phase-linear mode: a single FIR correction filter.
pub struct PhaseLinearStrategy;

impl ChannelProcessingStrategy for PhaseLinearStrategy {
    fn process(
        &self,
        input: &mut ChannelOptimizationInput,
        prepared: &PreparedMeasurement,
        target: &TargetContext,
        preprocessed: &PreprocessedFeatures,
        clamped_optimizer: &OptimizerConfig,
        _eq_resources: &EqResources,
    ) -> Result<MixedModeResult> {
        info!("  Generating FIR filter...");

        if let Some(ref mut cb) = input.callback {
            cb(1, target.pre_score, None);
        }

        let opt_config = clamped_optimizer.clone();

        let fir_input_curve = if let Some(ref tilt_curve) = target.target_tilt_curve {
            Curve {
                freq: preprocessed.curve_for_optim.freq.clone(),
                spl: &preprocessed.curve_for_optim.spl - &tilt_curve.spl,
                phase: preprocessed.curve_for_optim.phase.clone(),
                ..Default::default()
            }
        } else {
            preprocessed.curve_for_optim.clone()
        };

        let coeffs = fir::generate_fir_correction(
            &fir_input_curve,
            &opt_config,
            target.effective_target(input.room_config),
            input.sample_rate,
        )
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!("FIR generation failed: {}", e),
        })?;

        let (filename, wav_path) = artifacts::reserve_convolution_artifact_path(
            input.output_dir,
            input.channel_name,
            artifacts::ConvolutionArtifactKind::Fir,
            input.sample_rate,
        );
        crate::fir::save_fir_to_wav(&coeffs, input.sample_rate as u32, &wav_path).map_err(|e| {
            AutoeqError::OptimizationFailed {
                message: format!("Failed to save FIR WAV: {}", e),
            }
        })?;

        info!("  Saved FIR filter to {}", wav_path.display());

        let optim_output = OptimizerOutput::PhaseLinear {
            coeffs,
            wav_filename: filename,
        };
        let dsp_chain = assemble_dsp_chain(input, preprocessed, &optim_output)?;
        let report = assemble_channel_report(input, prepared, target, preprocessed, &optim_output)?;

        if let Some(ref mut cb) = input.callback {
            cb(2, report.post_score, None);
        }

        Ok(build_mixed_mode_result(
            dsp_chain,
            report,
            optim_output,
            preprocessed.optimizer_evidence.clone(),
        ))
    }
}

/// Hybrid mode: IIR correction for the low end, FIR for the residual.
pub struct HybridStrategy;

impl ChannelProcessingStrategy for HybridStrategy {
    fn process(
        &self,
        input: &mut ChannelOptimizationInput,
        prepared: &PreparedMeasurement,
        target: &TargetContext,
        preprocessed: &PreprocessedFeatures,
        clamped_optimizer: &OptimizerConfig,
        eq_resources: &EqResources,
    ) -> Result<MixedModeResult> {
        if let Some(mixed_config) = &input.room_config.optimizer.mixed_config {
            return group_processing::process_mixed_mode_crossover(
                input.channel_name,
                &preprocessed.curve_for_optim,
                input.room_config,
                mixed_config,
                input.sample_rate,
                input.output_dir,
                target.min_freq,
                target.max_freq,
                target.mean_spl,
                target.pre_score,
                prepared.arrival_time_ms,
                input.callback.take(),
            );
        }

        let opt_config = clamped_optimizer.clone();

        let hybrid_optim_curve = if let Some(ref tilt_curve) = target.target_tilt_curve {
            Curve {
                freq: preprocessed.curve_for_optim.freq.clone(),
                spl: &preprocessed.curve_for_optim.spl - &tilt_curve.spl,
                phase: preprocessed.curve_for_optim.phase.clone(),
                ..Default::default()
            }
        } else {
            preprocessed.curve_for_optim.clone()
        };

        let eq_result = if let Some(cb) = input.callback.take() {
            engine_eq::optimize_channel_eq_with_callback_detailed(
                &hybrid_optim_curve,
                &opt_config,
                Some(eq_resources),
                input.sample_rate,
                cb,
            )
        } else {
            engine_eq::optimize_channel_eq_detailed(
                &hybrid_optim_curve,
                &opt_config,
                Some(eq_resources),
                input.sample_rate,
            )
        }
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!(
                "IIR optimization failed for channel {}: {}",
                input.channel_name, e
            ),
        })?;
        let engine_eq::EqOptimizationResult {
            filters: eq_filters,
            optimizer_evidence,
            ..
        } = eq_result;

        info!("  IIR stage: {} filters", eq_filters.len());

        let iir_resp = response::compute_peq_complex_response(
            &eq_filters,
            &preprocessed.curve.freq,
            input.sample_rate,
        );
        let input_plus_iir = response::apply_complex_response(&preprocessed.curve, &iir_resp);

        let coeffs = fir::generate_fir_correction(
            &input_plus_iir,
            &opt_config,
            target.effective_target(input.room_config),
            input.sample_rate,
        )
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!("FIR generation failed: {}", e),
        })?;

        let (filename, wav_path) = artifacts::reserve_convolution_artifact_path(
            input.output_dir,
            input.channel_name,
            artifacts::ConvolutionArtifactKind::ResidualFir,
            input.sample_rate,
        );
        crate::fir::save_fir_to_wav(&coeffs, input.sample_rate as u32, &wav_path).map_err(|e| {
            AutoeqError::OptimizationFailed {
                message: format!("Failed to save FIR WAV: {}", e),
            }
        })?;

        info!("  Saved FIR filter to {}", wav_path.display());

        let optim_output = OptimizerOutput::Hybrid {
            eq_filters,
            coeffs,
            wav_filename: filename,
        };
        let dsp_chain = assemble_dsp_chain(input, preprocessed, &optim_output)?;
        let report = assemble_channel_report(input, prepared, target, preprocessed, &optim_output)?;

        Ok(build_mixed_mode_result(
            dsp_chain,
            report,
            optim_output,
            with_preprocessing_evidence(preprocessed, optimizer_evidence),
        ))
    }
}

/// Mixed-phase mode: minimum-phase IIR plus optional excess-phase FIR.
pub struct MixedPhaseStrategy;

impl ChannelProcessingStrategy for MixedPhaseStrategy {
    fn process(
        &self,
        input: &mut ChannelOptimizationInput,
        prepared: &PreparedMeasurement,
        target: &TargetContext,
        preprocessed: &PreprocessedFeatures,
        clamped_optimizer: &OptimizerConfig,
        eq_resources: &EqResources,
    ) -> Result<MixedModeResult> {
        let optimization_curve = if let Some(ref tilt_curve) = target.target_tilt_curve {
            Curve {
                freq: preprocessed.curve_for_optim.freq.clone(),
                spl: &preprocessed.curve_for_optim.spl - &tilt_curve.spl,
                phase: preprocessed.curve_for_optim.phase.clone(),
                ..Default::default()
            }
        } else {
            preprocessed.curve_for_optim.clone()
        };

        let engine_eq::EqOptimizationResult {
            filters: eq_filters,
            optimizer_evidence,
            ..
        } = optimize_eq_maybe_multi(
            input.prepared.measurements(),
            &optimization_curve,
            clamped_optimizer,
            eq_resources,
            input.sample_rate,
            input.channel_name,
            input.callback.take(),
            target.target_tilt_curve.as_ref(),
        )?;

        info!("  IIR stage: {} filters", eq_filters.len());

        let mp_config = match &input.room_config.optimizer.mixed_phase {
            Some(sc) => super::super::mixed_phase::MixedPhaseConfig {
                max_fir_length_ms: sc.max_fir_length_ms,
                pre_ringing_threshold_db: sc.pre_ringing_threshold_db,
                min_spatial_depth: sc.min_spatial_depth,
                phase_smoothing_octaves: sc.phase_smoothing_octaves,
            },
            None => super::super::mixed_phase::MixedPhaseConfig::default(),
        };

        let spatial_depth = if input.prepared.measurements().is_multi_measurement_source() {
            let curves = input.prepared.measurements().individual();
            if curves.len() > 1 {
                let sr_config = input
                    .room_config
                    .optimizer
                    .multi_measurement
                    .as_ref()
                    .and_then(|mc| mc.spatial_robustness.as_ref())
                    .map(
                        |sc| super::super::spatial_robustness::SpatialRobustnessConfig {
                            variance_threshold_db: sc.variance_threshold_db,
                            transition_width_db: sc.transition_width_db,
                            min_correction_depth: sc.min_correction_depth,
                            mask_smoothing_octaves: sc.mask_smoothing_octaves,
                        },
                    )
                    .unwrap_or_default();
                let weights = input
                    .room_config
                    .optimizer
                    .multi_measurement
                    .as_ref()
                    .and_then(|mc| mc.weights.as_deref());
                match super::super::spatial_robustness::analyze_spatial_robustness_weighted(
                    curves, &sr_config, weights,
                ) {
                    Ok(analysis) => {
                        info!(
                            "  Spatial depth for mixed-phase: mean={:.2}",
                            analysis.correction_depth.iter().sum::<f64>()
                                / analysis.correction_depth.len() as f64,
                        );
                        Some(analysis.correction_depth)
                    }
                    Err(e) => {
                        warn!("  Spatial robustness analysis skipped: {e}");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        let fir_coeffs = if preprocessed.curve_for_optim.phase.is_some() {
            match super::super::mixed_phase::decompose_phase(
                &preprocessed.curve_for_optim,
                &mp_config,
            ) {
                Ok((_min_phase, _excess, delay_ms, residual)) => {
                    info!(
                        "  Mixed-phase: delay={:.2} ms, generating excess phase FIR...",
                        delay_ms
                    );
                    let coeffs = super::super::mixed_phase::generate_excess_phase_fir_with_depth(
                        &preprocessed.curve_for_optim.freq,
                        &residual,
                        &mp_config,
                        input.sample_rate,
                        spatial_depth.as_ref(),
                    );

                    let (filename, wav_path) = artifacts::reserve_convolution_artifact_path(
                        input.output_dir,
                        input.channel_name,
                        artifacts::ConvolutionArtifactKind::ExcessPhaseFir,
                        input.sample_rate,
                    );
                    if let Err(e) =
                        crate::fir::save_fir_to_wav(&coeffs, input.sample_rate as u32, &wav_path)
                    {
                        warn!("Failed to save excess phase FIR WAV: {}", e);
                    } else {
                        info!("  Saved excess phase FIR to {}", wav_path.display());
                    }

                    let report =
                        super::super::mixed_phase::MixedPhaseCorrectionReport::from_residual(
                            delay_ms,
                            coeffs.len(),
                            &residual,
                        );
                    Some((coeffs, filename, report))
                }
                Err(e) => {
                    warn!(
                        "  Mixed-phase decomposition failed for '{}': {}. Using IIR only.",
                        input.channel_name, e
                    );
                    None
                }
            }
        } else {
            info!(
                "  No phase data for '{}', using IIR only (skipping excess phase FIR).",
                input.channel_name
            );
            None
        };

        let optim_output = OptimizerOutput::MixedPhase {
            eq_filters,
            fir_coeffs: fir_coeffs.as_ref().map(|(coeffs, _, _)| coeffs.clone()),
            fir_filename: fir_coeffs.as_ref().map(|(_, filename, _)| filename.clone()),
            report: fir_coeffs.as_ref().map(|(_, _, report)| report.clone()),
        };
        let dsp_chain = assemble_dsp_chain(input, preprocessed, &optim_output)?;
        let report = assemble_channel_report(input, prepared, target, preprocessed, &optim_output)?;

        info!(
            "  Mixed-phase result: pre={:.6}, post={:.6}",
            report.pre_score, report.post_score
        );

        Ok(build_mixed_mode_result(
            dsp_chain,
            report,
            optim_output,
            with_preprocessing_evidence(preprocessed, optimizer_evidence),
        ))
    }
}

/// Thin compatibility adapter for the engine-owned artifact-free modes.
pub struct EngineIirStrategy {
    mode: IirChannelMode,
}

impl ChannelProcessingStrategy for EngineIirStrategy {
    fn process(
        &self,
        input: &mut ChannelOptimizationInput,
        _prepared: &PreparedMeasurement,
        target: &TargetContext,
        preprocessed: &PreprocessedFeatures,
        clamped_optimizer: &OptimizerConfig,
        eq_resources: &EqResources,
    ) -> Result<MixedModeResult> {
        let IirChannelResult {
            channel,
            pre_score,
            post_score,
            raw_pre_eq_curve,
            raw_post_eq_curve,
            filters,
            mean_spl,
            arrival_time_ms,
            optimizer_evidence,
        } = process_iir_channel(IirChannelRequest {
            mode: self.mode,
            channel_name: input.channel_name,
            prepared: input.prepared,
            room_config: input.room_config,
            sample_rate: input.sample_rate,
            target,
            preprocessed,
            optimizer: clamped_optimizer,
            eq_resources,
            callback: input.callback.take(),
        })?;

        Ok((
            channel,
            pre_score,
            post_score,
            raw_pre_eq_curve,
            raw_post_eq_curve,
            filters,
            mean_spl,
            arrival_time_ms,
            None,
            optimizer_evidence,
        ))
    }
}
