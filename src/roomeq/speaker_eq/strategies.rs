//! Processing-mode strategies for single-speaker room EQ.
//!
//! Each [`ProcessingMode`] is implemented as a separate strategy so the main
//! `process_single_speaker` dispatch is a simple lookup instead of a large match.

use super::apply::{assemble_channel_report, assemble_dsp_chain, build_mixed_mode_result};
use super::misc::optimize_eq_maybe_multi;
use super::schroeder::optimize_with_schroeder_split_detailed;
use super::types::{
    ChannelOptimizationInput, MixedModeResult, OptimizerOutput, PreparedMeasurement,
    PreprocessedFeatures, TargetContext,
};
use crate::Curve;
use crate::error::{AutoeqError, Result};
use crate::response;
use crate::roomeq::types::{MeasurementSource, OptimizerConfig, ProcessingMode};
use crate::roomeq::{artifacts, eq, fir, group_processing};
use log::{info, warn};
use math_audio_iir_fir::Biquad;

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
    ) -> Result<MixedModeResult>;
}

/// Factory that maps a [`ProcessingMode`] to its strategy implementation.
pub fn strategy_for_mode(mode: ProcessingMode) -> Box<dyn ChannelProcessingStrategy> {
    match mode {
        ProcessingMode::PhaseLinear => Box::new(PhaseLinearStrategy),
        ProcessingMode::Hybrid => Box::new(HybridStrategy),
        ProcessingMode::MixedPhase => Box::new(MixedPhaseStrategy),
        ProcessingMode::LowLatency => Box::new(LowLatencyStrategy { warped: false }),
        ProcessingMode::WarpedIir => Box::new(LowLatencyStrategy { warped: true }),
        ProcessingMode::KautzModal => Box::new(KautzModalStrategy),
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
        let report = assemble_channel_report(
            input,
            prepared,
            target,
            preprocessed,
            &dsp_chain,
            &optim_output,
        )?;

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
            eq::optimize_channel_eq_with_callback_detailed(
                &hybrid_optim_curve,
                &opt_config,
                target.effective_target(input.room_config),
                input.sample_rate,
                cb,
            )
        } else {
            eq::optimize_channel_eq_detailed(
                &hybrid_optim_curve,
                &opt_config,
                target.effective_target(input.room_config),
                input.sample_rate,
            )
        }
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!(
                "IIR optimization failed for channel {}: {}",
                input.channel_name, e
            ),
        })?;
        let eq::EqOptimizationResult {
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
        let report = assemble_channel_report(
            input,
            prepared,
            target,
            preprocessed,
            &dsp_chain,
            &optim_output,
        )?;

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

        let eq::EqOptimizationResult {
            filters: eq_filters,
            optimizer_evidence,
            ..
        } = optimize_eq_maybe_multi(
            input.source,
            &optimization_curve,
            clamped_optimizer,
            target.effective_target(input.room_config),
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

        let spatial_depth = if matches!(
            input.source,
            MeasurementSource::Multiple(_) | MeasurementSource::InMemoryMultiple(_)
        ) {
            match crate::read::load_source_individual(input.source) {
                Ok(curves) if curves.len() > 1 => {
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
                        &curves, &sr_config, weights,
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
                }
                _ => None,
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

                    Some((coeffs, filename))
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
            fir_coeffs: fir_coeffs.as_ref().map(|(coeffs, _)| coeffs.clone()),
            fir_filename: fir_coeffs.as_ref().map(|(_, filename)| filename.clone()),
        };
        let dsp_chain = assemble_dsp_chain(input, preprocessed, &optim_output)?;
        let report = assemble_channel_report(
            input,
            prepared,
            target,
            preprocessed,
            &dsp_chain,
            &optim_output,
        )?;

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

/// Low-latency / warped-IIR mode: IIR-only correction.
pub struct LowLatencyStrategy {
    warped: bool,
}

impl ChannelProcessingStrategy for LowLatencyStrategy {
    fn process(
        &self,
        input: &mut ChannelOptimizationInput,
        prepared: &PreparedMeasurement,
        target: &TargetContext,
        preprocessed: &PreprocessedFeatures,
        clamped_optimizer: &OptimizerConfig,
    ) -> Result<MixedModeResult> {
        let warped_iir = self.warped;
        let warped_lambda = warped_iir.then(|| math_audio_iir_fir::bark_lambda(input.sample_rate));

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

        let (eq_filters, optimizer_evidence) = if let Some(schroeder_config) =
            &clamped_optimizer.schroeder_split
        {
            if schroeder_config.enabled {
                let schroeder_freq = if let Some(ref dims) = schroeder_config.room_dimensions {
                    let calculated = dims.schroeder_frequency();
                    info!(
                        "  Schroeder split: calculated frequency {:.1} Hz from room dimensions",
                        calculated
                    );
                    calculated
                } else {
                    schroeder_config.schroeder_freq
                };
                info!(
                    "  Schroeder split: optimizing below {:.1} Hz with max_q={:.1}, above with max_q={:.1}",
                    schroeder_freq,
                    schroeder_config.low_freq_config.max_q,
                    schroeder_config.high_freq_config.max_q
                );

                let result = optimize_with_schroeder_split_detailed(
                    &optimization_curve,
                    clamped_optimizer,
                    schroeder_config,
                    input.sample_rate,
                )?;

                let mut combined_filters = result.low_filters;
                combined_filters.extend(result.high_filters);
                info!(
                    "  Schroeder split: {} low-freq filters + {} high-freq filters",
                    combined_filters
                        .iter()
                        .filter(|f| f.freq < schroeder_freq)
                        .count(),
                    combined_filters
                        .iter()
                        .filter(|f| f.freq >= schroeder_freq)
                        .count()
                );
                (combined_filters, result.optimizer_evidence)
            } else {
                let result = optimize_eq_maybe_multi(
                    input.source,
                    &optimization_curve,
                    clamped_optimizer,
                    target.effective_target(input.room_config),
                    input.sample_rate,
                    input.channel_name,
                    input.callback.take(),
                    target.target_tilt_curve.as_ref(),
                )?;
                (result.filters, result.optimizer_evidence)
            }
        } else {
            let result = optimize_eq_maybe_multi(
                input.source,
                &optimization_curve,
                clamped_optimizer,
                target.effective_target(input.room_config),
                input.sample_rate,
                input.channel_name,
                input.callback.take(),
                target.target_tilt_curve.as_ref(),
            )?;
            (result.filters, result.optimizer_evidence)
        };

        info!("  Optimized {} EQ filters", eq_filters.len());

        let preference_filters = if target.cea2034_active {
            if let Some(ref target_resp) = input.room_config.optimizer.target_response {
                super::super::cea2034_correction::generate_preference_filters(
                    &target_resp.preference,
                    input.sample_rate,
                )
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let optim_output = if warped_iir {
            OptimizerOutput::WarpedIir {
                eq_filters,
                preference_filters,
                warped_lambda: warped_lambda.unwrap_or(0.0),
            }
        } else {
            OptimizerOutput::LowLatency {
                eq_filters,
                preference_filters,
            }
        };
        let dsp_chain = assemble_dsp_chain(input, preprocessed, &optim_output)?;
        let report = assemble_channel_report(
            input,
            prepared,
            target,
            preprocessed,
            &dsp_chain,
            &optim_output,
        )?;

        Ok(build_mixed_mode_result(
            dsp_chain,
            report,
            optim_output,
            with_preprocessing_evidence(preprocessed, optimizer_evidence),
        ))
    }
}

/// Kautz-modal mode: pole-tuned filters targeted at detected room modes.
pub struct KautzModalStrategy;

impl ChannelProcessingStrategy for KautzModalStrategy {
    fn process(
        &self,
        input: &mut ChannelOptimizationInput,
        prepared: &PreparedMeasurement,
        target: &TargetContext,
        preprocessed: &PreprocessedFeatures,
        _clamped_optimizer: &OptimizerConfig,
    ) -> Result<MixedModeResult> {
        info!("  KautzModal mode: starting optimization...");

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

        let decomposed_config =
            super::super::impulse_analysis::DecomposedCorrectionConfig::default();
        let room_modes = super::super::impulse_analysis::detect_room_modes(
            &optimization_curve.freq,
            &optimization_curve.spl,
            &decomposed_config,
        );

        if room_modes.is_empty() {
            return Err(AutoeqError::OptimizationFailed {
                message: format!(
                    "KautzModal found no room modes for channel '{}'; use low_latency or \
                     provide a measurement with clear modal peaks",
                    input.channel_name
                ),
            });
        }

        info!(
            "  Detected {} room modes, building Kautz filter",
            room_modes.len()
        );

        let mode_tuples: Vec<(f64, f64)> = room_modes.iter().map(|m| (m.frequency, m.q)).collect();

        let mut kautz =
            math_audio_iir_fir::KautzFilter::from_room_modes(&mode_tuples, input.sample_rate);

        let freqs_f64: Vec<f64> = optimization_curve.freq.iter().copied().collect();
        let measured_f64: Vec<f64> = optimization_curve.spl.iter().copied().collect();
        let target_f64: Vec<f64> = vec![0.0; freqs_f64.len()];

        kautz.optimize_gains(&freqs_f64, &measured_f64, &target_f64);

        let kautz_sections: Vec<(f64, f64, f64)> = room_modes
            .iter()
            .zip(kautz.sections.iter())
            .filter(|(_, s)| s.gain.abs() > 0.1)
            .map(|(mode, section)| (mode.frequency, mode.q.max(0.5), section.gain))
            .collect();

        let eq_filters: Vec<Biquad> = kautz_sections
            .iter()
            .map(|(freq, q, gain)| {
                use math_audio_iir_fir::BiquadFilterType;
                Biquad::new(BiquadFilterType::Peak, *freq, input.sample_rate, *q, *gain)
            })
            .collect();

        if kautz_sections.is_empty() {
            return Err(AutoeqError::OptimizationFailed {
                message: format!(
                    "KautzModal optimized zero usable filters for channel '{}'; use low_latency \
                     or adjust the measurement/optimizer range",
                    input.channel_name
                ),
            });
        }

        info!(
            "  KautzModal: {} Kautz sections from {} modes",
            kautz_sections.len(),
            room_modes.len()
        );

        let preference_filters = if target.cea2034_active {
            if let Some(ref target_resp) = input.room_config.optimizer.target_response {
                super::super::cea2034_correction::generate_preference_filters(
                    &target_resp.preference,
                    input.sample_rate,
                )
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let optim_output = OptimizerOutput::KautzModal {
            eq_filters,
            kautz_sections,
            preference_filters,
        };
        let dsp_chain = assemble_dsp_chain(input, preprocessed, &optim_output)?;
        let report = assemble_channel_report(
            input,
            prepared,
            target,
            preprocessed,
            &dsp_chain,
            &optim_output,
        )?;

        Ok(build_mixed_mode_result(
            dsp_chain,
            report,
            optim_output,
            preprocessed.optimizer_evidence.clone(),
        ))
    }
}
