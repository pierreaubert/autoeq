use super::super::types::{MeasurementSource, RoomConfig};
use super::build::build_clamped_optimizer;
use super::types::{ChannelOptimizationInput, MixedModeResult, PreparedMeasurement};
use crate::error::{AutoeqError, Result};
use log::debug;
use std::path::Path;

pub(super) fn prepare_measurement(
    input: &ChannelOptimizationInput<'_>,
) -> Result<PreparedMeasurement> {
    let curve = input.prepared.measurements().representative().clone();
    debug!(
        "  Loaded measurement: {:.1} Hz - {:.1} Hz",
        curve.freq[0],
        curve.freq[curve.freq.len() - 1]
    );
    super::super::optimize::warn_if_optimizer_bounds_exceed_data(
        input.channel_name,
        &curve,
        &input.room_config.optimizer,
    );
    let arrival_time_ms = input.prepared.arrival_time_ms();
    let curve_raw = curve.clone();
    Ok(PreparedMeasurement {
        curve,
        curve_raw,
        arrival_time_ms,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn process_single_speaker(
    channel_name: &str,
    source: &MeasurementSource,
    room_config: &RoomConfig,
    sample_rate: f64,
    output_dir: &Path,
    callback: Option<crate::optim::OptimProgressCallback>,
    probe_arrival_ms: Option<f64>,
    shared_mean_spl: Option<f64>,
) -> Result<MixedModeResult> {
    let prepared_input = roomeq_workflow::prepare_channel_input(
        channel_name,
        source,
        room_config,
        sample_rate,
        probe_arrival_ms,
    )
    .map_err(|error| AutoeqError::InvalidMeasurement {
        message: format!("Failed to load measurement for channel {channel_name}: {error}"),
    })?;
    let mut input = ChannelOptimizationInput {
        channel_name,
        prepared: &prepared_input,
        room_config,
        sample_rate,
        output_dir,
        callback,
        shared_mean_spl,
    };

    let prepared = prepare_measurement(&input)?;
    let mut target = roomeq_engine::channel_target::build_target_context(
        input.channel_name,
        input.room_config,
        &prepared.curve,
        input.shared_mean_spl,
    );
    let preprocessed = roomeq_engine::channel_preprocessing::preprocess_channel(
        input.channel_name,
        input.prepared,
        input.room_config,
        input.sample_rate,
        input.shared_mean_spl,
        &mut target,
    );
    let clamped_optimizer = build_clamped_optimizer(
        channel_name,
        room_config,
        &prepared.curve_raw,
        &preprocessed.curve_for_optim,
        target.min_freq,
        target.max_freq,
        target.target_tilt_curve.as_ref(),
        preprocessed.broadband_enabled,
    );

    let mut eq_resources = prepared_input.eq_resources().clone();
    eq_resources.target = roomeq_workflow::prepare_eq_target(target.effective_target(room_config))
        .map_err(|error| AutoeqError::OptimizationFailed {
            message: format!("Failed to prepare EQ resources for channel {channel_name}: {error}"),
        })?;

    super::strategies::strategy_for_mode(room_config.optimizer.processing_mode.clone()).process(
        &mut input,
        &prepared,
        &target,
        &preprocessed,
        &clamped_optimizer,
        &eq_resources,
    )
}
