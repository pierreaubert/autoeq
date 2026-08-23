//! Path-free IIR processing for one prepared RoomEQ channel.

mod assemble;
mod optimize;
#[cfg(test)]
mod tests;

use autoeq_core::{AutoeqError, Curve, Result};
use autoeq_optim::optim::{OptimProgressCallback, OptimizerRunEvidence};
use log::info;
use math_audio_iir_fir::{Biquad, BiquadFilterType, KautzFilter};
use roomeq_model::{OptimizerConfig, RoomConfig};

use crate::PreparedChannelInput;
use crate::channel_preprocessing::PreprocessedFeatures;
pub use crate::channel_result::ChannelProcessingResult as IirChannelResult;
use crate::channel_target::TargetContext;
use crate::eq::EqResources;

/// The artifact-free processing modes owned by this module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IirChannelMode {
    LowLatency,
    WarpedIir,
    KautzModal,
}

/// Complete path-free request for one IIR channel run.
pub struct IirChannelRequest<'a> {
    pub mode: IirChannelMode,
    pub channel_name: &'a str,
    pub prepared: &'a PreparedChannelInput,
    pub room_config: &'a RoomConfig,
    pub sample_rate: f64,
    pub target: &'a TargetContext,
    pub preprocessed: &'a PreprocessedFeatures,
    pub optimizer: &'a OptimizerConfig,
    pub eq_resources: &'a EqResources,
    pub callback: Option<OptimProgressCallback>,
}

pub(super) enum IirOptimizerOutput {
    LowLatency {
        eq_filters: Vec<Biquad>,
        preference_filters: Vec<Biquad>,
    },
    WarpedIir {
        eq_filters: Vec<Biquad>,
        preference_filters: Vec<Biquad>,
        warped_lambda: f64,
    },
    KautzModal {
        eq_filters: Vec<Biquad>,
        kautz_sections: Vec<(f64, f64, f64)>,
        preference_filters: Vec<Biquad>,
    },
}

impl IirOptimizerOutput {
    pub(super) fn eq_filters(&self) -> &[Biquad] {
        match self {
            Self::LowLatency { eq_filters, .. }
            | Self::WarpedIir { eq_filters, .. }
            | Self::KautzModal { eq_filters, .. } => eq_filters,
        }
    }
}

/// Optimize, assemble, and score one artifact-free channel.
///
/// This function performs no measurement, network, filesystem, or artifact
/// I/O. All source-backed resources must already be present in the prepared
/// channel input.
pub fn process_iir_channel(mut request: IirChannelRequest<'_>) -> Result<IirChannelResult> {
    let optimization_curve = crate::channel_result::subtract_target_tilt(
        &request.preprocessed.curve_for_optim,
        request.target,
    );

    match request.mode {
        IirChannelMode::LowLatency | IirChannelMode::WarpedIir => {
            let (eq_filters, optimizer_evidence) = optimize::optimize_iir_eq(
                request.channel_name,
                request.prepared,
                &optimization_curve,
                request.optimizer,
                request.eq_resources,
                request.sample_rate,
                request.callback.take(),
                request.target.target_tilt_curve.as_ref(),
            )?;
            info!("  Optimized {} EQ filters", eq_filters.len());

            let preference_filters = preference_filters(
                request.channel_name,
                request.room_config,
                request.target,
                request.sample_rate,
            );
            let output = match request.mode {
                IirChannelMode::LowLatency => IirOptimizerOutput::LowLatency {
                    eq_filters,
                    preference_filters,
                },
                IirChannelMode::WarpedIir => IirOptimizerOutput::WarpedIir {
                    eq_filters,
                    preference_filters,
                    warped_lambda: math_audio_iir_fir::bark_lambda(request.sample_rate),
                },
                IirChannelMode::KautzModal => unreachable!(),
            };
            assemble::assemble_iir_result(
                &request,
                output,
                with_preprocessing_evidence(request.preprocessed, optimizer_evidence),
            )
        }
        IirChannelMode::KautzModal => {
            let output = optimize_kautz_modal(&request, &optimization_curve)?;
            assemble::assemble_iir_result(
                &request,
                output,
                request.preprocessed.optimizer_evidence.clone(),
            )
        }
    }
}

pub(crate) fn preference_filters(
    channel_name: &str,
    room_config: &RoomConfig,
    _target: &TargetContext,
    sample_rate: f64,
) -> Vec<Biquad> {
    crate::channel_preference::build_preference_filters(channel_name, room_config, sample_rate)
}

fn with_preprocessing_evidence(
    preprocessed: &PreprocessedFeatures,
    mut optimizer_evidence: Vec<OptimizerRunEvidence>,
) -> Vec<OptimizerRunEvidence> {
    let mut combined = preprocessed.optimizer_evidence.clone();
    combined.append(&mut optimizer_evidence);
    combined
}

fn optimize_kautz_modal(
    request: &IirChannelRequest<'_>,
    optimization_curve: &Curve,
) -> Result<IirOptimizerOutput> {
    info!("  KautzModal mode: starting optimization...");

    let room_modes = roomeq_analysis::impulse_analysis::detect_room_modes(
        &optimization_curve.freq,
        &optimization_curve.spl,
        &roomeq_analysis::impulse_analysis::DecomposedCorrectionConfig::default(),
    );
    if room_modes.is_empty() {
        return Err(AutoeqError::OptimizationFailed {
            message: format!(
                "KautzModal found no room modes for channel '{}'; use low_latency or provide a measurement with clear modal peaks",
                request.channel_name
            ),
        });
    }

    info!(
        "  Detected {} room modes, building Kautz filter",
        room_modes.len()
    );
    let mode_tuples: Vec<(f64, f64)> = room_modes
        .iter()
        .map(|mode| (mode.frequency, mode.q))
        .collect();
    let mut kautz = KautzFilter::from_room_modes(&mode_tuples, request.sample_rate);
    let frequencies: Vec<f64> = optimization_curve.freq.iter().copied().collect();
    let measured: Vec<f64> = optimization_curve.spl.iter().copied().collect();
    let target = vec![0.0; frequencies.len()];
    kautz.optimize_gains(&frequencies, &measured, &target);

    let kautz_sections: Vec<(f64, f64, f64)> = room_modes
        .iter()
        .zip(kautz.sections.iter())
        .filter(|(_, section)| section.gain.abs() > 0.1)
        .map(|(mode, section)| (mode.frequency, mode.q.max(0.5), section.gain))
        .collect();
    if kautz_sections.is_empty() {
        return Err(AutoeqError::OptimizationFailed {
            message: format!(
                "KautzModal optimized zero usable filters for channel '{}'; use low_latency or adjust the measurement/optimizer range",
                request.channel_name
            ),
        });
    }

    info!(
        "  KautzModal: {} Kautz sections from {} modes",
        kautz_sections.len(),
        room_modes.len()
    );
    let eq_filters = kautz_sections
        .iter()
        .map(|(frequency, q, gain)| {
            Biquad::new(
                BiquadFilterType::Peak,
                *frequency,
                request.sample_rate,
                *q,
                *gain,
            )
        })
        .collect();
    Ok(IirOptimizerOutput::KautzModal {
        eq_filters,
        kautz_sections,
        preference_filters: preference_filters(
            request.channel_name,
            request.room_config,
            request.target,
            request.sample_rate,
        ),
    })
}
