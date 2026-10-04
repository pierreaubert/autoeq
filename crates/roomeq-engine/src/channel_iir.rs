//! Path-free IIR processing for one prepared RoomEQ channel.

mod assemble;
mod kautz_fit;
mod optimize;
#[cfg(test)]
mod tests;

use autoeq_core::{AutoeqError, Curve, Result};
use autoeq_optim::optim::{OptimProgressCallback, OptimizerRunEvidence};
use log::info;
use math_audio_iir_fir::{Biquad, KautzFilter};
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
        kautz_sections: Vec<(f64, f64, f64)>,
        preference_filters: Vec<Biquad>,
    },
}

impl IirOptimizerOutput {
    pub(super) fn eq_filters(&self) -> &[Biquad] {
        match self {
            Self::LowLatency { eq_filters, .. } | Self::WarpedIir { eq_filters, .. } => eq_filters,
            // Kautz linear weights are not PEQ gains or realizable biquads.
            Self::KautzModal { .. } => &[],
        }
    }
}

/// Optimize, assemble, and score one artifact-free channel.
///
/// This function performs no measurement, network, filesystem, or artifact
/// I/O. All source-backed resources must already be present in the prepared
/// channel input.
pub fn process_iir_channel(mut request: IirChannelRequest<'_>) -> Result<IirChannelResult> {
    process_iir_channel_inner(&mut request, None)
}

/// Execute one LowLatency channel using exact seeded AutoEQ DE recovery.
pub fn process_iir_channel_with_exact_checkpoint(
    mut request: IirChannelRequest<'_>,
    exact: crate::eq::exact_recovery::ExactDERecoveryOptions,
) -> Result<IirChannelResult> {
    process_iir_channel_inner(&mut request, Some(exact))
}

fn process_iir_channel_inner(
    request: &mut IirChannelRequest<'_>,
    exact: Option<crate::eq::exact_recovery::ExactDERecoveryOptions>,
) -> Result<IirChannelResult> {
    if exact.is_some() && request.mode != IirChannelMode::LowLatency {
        return Err(AutoeqError::InvalidConfiguration {
            message: "exact DE recovery supports LowLatency IIR only".into(),
        });
    }
    let usable_curve = request
        .prepared
        .usable_curve(&request.preprocessed.curve_for_optim)?;
    let optimization_curve =
        crate::channel_result::subtract_target_tilt(&usable_curve, request.target);

    match request.mode {
        IirChannelMode::LowLatency | IirChannelMode::WarpedIir => {
            let (eq_filters, optimizer_evidence, audibility_veto, veto_adjudication) = match exact {
                Some(exact) => optimize::optimize_iir_eq_with_exact_checkpoint(
                    request.channel_name,
                    request.prepared,
                    &optimization_curve,
                    request.optimizer,
                    request.eq_resources,
                    request.sample_rate,
                    request.callback.take(),
                    request.target.target_tilt_curve.as_ref(),
                    Some(exact),
                )?,
                None => optimize::optimize_iir_eq(
                    request.channel_name,
                    request.prepared,
                    &optimization_curve,
                    request.optimizer,
                    request.eq_resources,
                    request.sample_rate,
                    request.callback.take(),
                    request.target.target_tilt_curve.as_ref(),
                )?,
            };
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
                audibility_veto,
                veto_adjudication,
            )
        }
        IirChannelMode::KautzModal => {
            let output = optimize_kautz_modal(&request, &optimization_curve)?;
            assemble::assemble_iir_result(
                &request,
                output,
                request.preprocessed.optimizer_evidence.clone(),
                Vec::new(),
                None,
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
    let optimizer = request.optimizer;
    let invalid = |message: String| AutoeqError::OptimizationFailed {
        message: format!("KautzModal channel '{}': {message}", request.channel_name),
    };
    if !request.sample_rate.is_finite()
        || request.sample_rate <= 0.0
        || !optimizer.min_freq.is_finite()
        || !optimizer.max_freq.is_finite()
        || optimizer.min_freq <= 0.0
        || optimizer.min_freq >= optimizer.max_freq
        || !optimizer.min_q.is_finite()
        || !optimizer.max_q.is_finite()
        || optimizer.min_q <= 0.0
        || optimizer.min_q > optimizer.max_q
        // The playback section implementation has a minimum supported Q of 0.1.
        || optimizer.max_q < 0.1
        || optimizer.num_filters == 0
    {
        return Err(invalid(
            "invalid frequency/Q bounds, sample rate, or zero section budget".into(),
        ));
    }
    let (min_frequency, max_frequency) = if let Some(band) = &optimizer.correction_band {
        band.validate_against(optimizer.min_freq, optimizer.max_freq)
            .map_err(invalid)?;
        (band.min_hz, band.max_hz)
    } else {
        (optimizer.min_freq, optimizer.max_freq)
    };

    let detection_config = request
        .optimizer
        .decomposed_correction
        .as_ref()
        .map(
            |config| roomeq_analysis::impulse_analysis::DecomposedCorrectionConfig {
                schroeder_freq: config.schroeder_freq,
                transition_width_oct: config.transition_width_oct,
                min_mode_q: config.min_mode_q,
                min_mode_prominence_db: config.min_mode_prominence_db,
                mode_correction_weight: config.mode_correction_weight,
                early_reflection_weight: config.early_reflection_weight,
                steady_state_weight: config.steady_state_weight,
                fdw_enabled: config.fdw_enabled,
                fdw_cycles: config.fdw_cycles,
                fdw_min_window_ms: config.fdw_min_window_ms,
                fdw_max_window_ms: config.fdw_max_window_ms,
                fdw_smoothing_octaves: config.fdw_smoothing_octaves,
            },
        )
        .unwrap_or_default();
    let mut room_modes = roomeq_analysis::impulse_analysis::detect_room_modes(
        &optimization_curve.freq,
        &optimization_curve.spl,
        &detection_config,
    );
    if room_modes.is_empty() {
        return Err(AutoeqError::OptimizationFailed {
            message: format!(
                "KautzModal found no room modes for channel '{}'; use low_latency or provide a measurement with clear modal peaks",
                request.channel_name
            ),
        });
    }
    // Constrain the pole inventory before constructing the allpass chain.
    // Removing sections after fitting changes the basis of later sections.
    room_modes.retain(|mode| {
        mode.frequency.is_finite()
            && mode.q.is_finite()
            && mode.prominence_db.is_finite()
            && mode.frequency >= min_frequency
            && mode.frequency <= max_frequency
            && mode.frequency < request.sample_rate / 2.0
    });
    room_modes.sort_by(|a, b| {
        b.prominence_db
            .total_cmp(&a.prominence_db)
            .then_with(|| a.frequency.total_cmp(&b.frequency))
    });
    room_modes.truncate(optimizer.num_filters);
    room_modes.sort_by(|a, b| a.frequency.total_cmp(&b.frequency));
    for mode in &mut room_modes {
        mode.q = mode.q.clamp(optimizer.min_q.max(0.1), optimizer.max_q);
    }
    if room_modes.is_empty() {
        return Err(invalid(format!(
            "no room modes within the permitted pole band [{min_frequency}, {max_frequency}] Hz below Nyquist"
        )));
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
    let measured_mean = roomeq_analysis::response_metrics::mean_response_in_range(
        optimization_curve,
        request.optimizer.min_freq,
        request.optimizer.max_freq,
    );
    let mut normalized = optimization_curve.clone();
    normalized.spl -= measured_mean;
    let target = crate::eq::resources::target_curve(&normalized, Some(request.eq_resources));
    let correction = Curve {
        freq: normalized.freq.clone(),
        spl: &target.spl - &normalized.spl,
        ..Curve::default()
    };
    kautz_fit::fit_playback_gains(
        &mut kautz,
        &correction,
        min_frequency..=max_frequency,
        optimizer,
    )?;

    let kautz_sections: Vec<(f64, f64, f64)> = room_modes
        .iter()
        .zip(kautz.sections.iter())
        .map(|(mode, section)| (mode.frequency, mode.q, section.gain))
        .collect();
    // Unity is a valid best feasible response, including an already-met target.
    // Do not force a nonzero correction merely because modes were detected.

    info!(
        "  KautzModal: {} Kautz sections from {} modes",
        kautz_sections.len(),
        room_modes.len()
    );
    Ok(IirOptimizerOutput::KautzModal {
        kautz_sections,
        preference_filters: preference_filters(
            request.channel_name,
            request.room_config,
            request.target,
            request.sample_rate,
        ),
    })
}
