//! Processing-mode adapters for single-speaker room EQ.
//!
//! All processing modes delegate to roomeq-engine; this module only adapts the
//! remaining root workflow contract and persists workflow-owned sidecars.

use super::types::{
    ChannelOptimizationInput, MixedModeResult, PreparedMeasurement, PreprocessedFeatures,
    TargetContext,
};
use crate::error::{AutoeqError, Result};
use crate::roomeq::types::{OptimizerConfig, ProcessingMode};
use log::{info, warn};
use roomeq_engine::channel_fir::{FirChannelMode, FirChannelRequest, process_fir_channel};
use roomeq_engine::channel_iir::{IirChannelMode, IirChannelRequest, process_iir_channel};
use roomeq_engine::channel_result::ChannelProcessingResult;
use roomeq_engine::eq::EqResources;
use roomeq_engine::mixed_crossover::{MixedCrossoverRequest, process_mixed_crossover};

/// Strategy trait retained by the root compatibility facade.
pub trait ChannelProcessingStrategy {
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

pub fn strategy_for_mode(mode: ProcessingMode) -> Box<dyn ChannelProcessingStrategy> {
    match mode {
        ProcessingMode::PhaseLinear => Box::new(EngineFirStrategy {
            mode: FirChannelMode::PhaseLinear,
        }),
        ProcessingMode::Hybrid => Box::new(EngineFirStrategy {
            mode: FirChannelMode::Hybrid,
        }),
        ProcessingMode::MixedPhase => Box::new(EngineFirStrategy {
            mode: FirChannelMode::MixedPhase,
        }),
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

/// Thin compatibility adapter for engine-owned FIR-capable modes.
pub struct EngineFirStrategy {
    mode: FirChannelMode,
}

impl ChannelProcessingStrategy for EngineFirStrategy {
    fn process(
        &self,
        input: &mut ChannelOptimizationInput,
        prepared: &PreparedMeasurement,
        target: &TargetContext,
        preprocessed: &PreprocessedFeatures,
        clamped_optimizer: &OptimizerConfig,
        eq_resources: &EqResources,
    ) -> Result<MixedModeResult> {
        if self.mode == FirChannelMode::PhaseLinear
            && let Some(callback) = input.callback.as_mut()
        {
            callback(1, target.pre_score, None);
        }
        let (reservation, result) = if self.mode == FirChannelMode::Hybrid
            && let Some(mixed_config) = &input.room_config.optimizer.mixed_config
        {
            let reservation = roomeq_workflow::reserve_mixed_crossover_sidecar(
                input.output_dir,
                input.channel_name,
                input.sample_rate,
            )
            .map_err(|error| AutoeqError::OptimizationFailed {
                message: format!(
                    "Failed to reserve convolution artifact for channel {}: {error}",
                    input.channel_name
                ),
            })?;
            let result = process_mixed_crossover(MixedCrossoverRequest {
                channel_name: input.channel_name,
                curve: &preprocessed.curve_for_optim,
                mixed_config,
                optimizer: &input.room_config.optimizer,
                eq_resources,
                sample_rate: input.sample_rate,
                min_freq: target.min_freq,
                max_freq: target.max_freq,
                mean_spl: target.mean_spl,
                pre_score: target.pre_score,
                arrival_time_ms: prepared.arrival_time_ms,
                sidecar_reference: reservation.reference().clone(),
                callback: input.callback.take(),
            })?;
            (reservation, result)
        } else {
            let mode = match self.mode {
                FirChannelMode::PhaseLinear => ProcessingMode::PhaseLinear,
                FirChannelMode::Hybrid => ProcessingMode::Hybrid,
                FirChannelMode::MixedPhase => ProcessingMode::MixedPhase,
            };
            let reservation = roomeq_workflow::reserve_channel_convolution_sidecar(
                input.output_dir,
                input.channel_name,
                mode,
                input.sample_rate,
            )
            .map_err(|error| AutoeqError::OptimizationFailed {
                message: format!(
                    "Failed to reserve convolution artifact for channel {}: {error}",
                    input.channel_name
                ),
            })?
            .expect("FIR-capable modes always reserve a convolution sidecar");
            let result = process_fir_channel(FirChannelRequest {
                mode: self.mode,
                channel_name: input.channel_name,
                prepared: input.prepared,
                room_config: input.room_config,
                sample_rate: input.sample_rate,
                target,
                preprocessed,
                optimizer: clamped_optimizer,
                eq_resources,
                sidecar_reference: reservation.reference().clone(),
                callback: if self.mode == FirChannelMode::PhaseLinear {
                    None
                } else {
                    input.callback.take()
                },
            })?;
            (reservation, result)
        };

        if let Some(generated) = result.convolution_sidecar.as_ref() {
            let coefficients = result
                .fir_coeffs
                .as_deref()
                .expect("generated sidecar always has FIR coefficients");
            if let Err(error) = roomeq_workflow::persist_convolution_sidecar(
                &reservation,
                generated,
                coefficients,
                input.sample_rate as u32,
            ) {
                if generated.required {
                    return Err(AutoeqError::OptimizationFailed {
                        message: format!("Failed to save FIR WAV: {error}"),
                    });
                }
                warn!("Failed to save excess phase FIR WAV: {error}");
            } else {
                info!("  Saved FIR filter to {}", reservation.path().display());
            }
        }
        if self.mode == FirChannelMode::PhaseLinear
            && let Some(callback) = input.callback.as_mut()
        {
            callback(2, result.post_score, None);
        }

        Ok(result_tuple(result))
    }
}

/// Thin compatibility adapter for engine-owned artifact-free modes.
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
        let result = process_iir_channel(IirChannelRequest {
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
        Ok(result_tuple(result))
    }
}

fn result_tuple(result: ChannelProcessingResult) -> MixedModeResult {
    (
        result.channel,
        result.pre_score,
        result.post_score,
        result.raw_pre_eq_curve,
        result.raw_post_eq_curve,
        result.filters,
        result.mean_spl,
        result.arrival_time_ms,
        result.fir_coeffs,
        result.optimizer_evidence,
    )
}
