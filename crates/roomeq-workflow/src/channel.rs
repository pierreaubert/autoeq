//! Resource-owning workflow for one independently processed RoomEQ channel.

use std::path::Path;

use log::{info, warn};
use math_audio_iir_fir::Biquad;
use roomeq_engine::channel_execution::{execute_prepared_channel, prepare_channel_execution};
use roomeq_engine::channel_result::ChannelProcessingResult;
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_engine::{Curve, OptimProgressCallback, OptimizerRunEvidence};
use roomeq_model::{ChannelDspChain, MeasurementSource, ProcessingMode, RoomConfig};

use crate::arrival::prepare_channel_input_with_frequency_samples;
use crate::{
    persist_convolution_sidecar, prepare_eq_target, reserve_channel_convolution_sidecar,
    reserve_mixed_crossover_sidecar,
};

/// Compatibility result consumed by the remaining root topology workflows.
pub type ChannelWorkflowResult = (
    ChannelDspChain,
    f64,
    f64,
    Curve,
    Curve,
    Vec<Biquad>,
    f64,
    Option<f64>,
    Option<Vec<f64>>,
    Vec<OptimizerRunEvidence>,
);

/// Load resources, execute one channel, and persist any generated sidecar.
#[allow(clippy::too_many_arguments)]
pub fn process_single_channel(
    channel_name: &str,
    source: &MeasurementSource,
    room_config: &RoomConfig,
    sample_rate: f64,
    output_dir: &Path,
    callback: Option<OptimProgressCallback>,
    probe_arrival_ms: Option<f64>,
    shared_mean_spl: Option<f64>,
) -> Result<ChannelWorkflowResult> {
    process_single_channel_with_frequency_samples(
        channel_name,
        source,
        room_config,
        sample_rate,
        output_dir,
        callback,
        probe_arrival_ms,
        shared_mean_spl,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

/// Process one channel using a configurable measurement frequency grid.
#[allow(clippy::too_many_arguments)]
pub fn process_single_channel_with_frequency_samples(
    channel_name: &str,
    source: &MeasurementSource,
    room_config: &RoomConfig,
    sample_rate: f64,
    output_dir: &Path,
    callback: Option<OptimProgressCallback>,
    probe_arrival_ms: Option<f64>,
    shared_mean_spl: Option<f64>,
    frequency_samples: usize,
) -> Result<ChannelWorkflowResult> {
    let prepared = prepare_channel_input_with_frequency_samples(
        channel_name,
        source,
        room_config,
        sample_rate,
        probe_arrival_ms,
        frequency_samples,
    )
    .map_err(|error| AutoeqError::InvalidMeasurement {
        message: format!("Failed to load measurement for channel {channel_name}: {error}"),
    })?;
    let execution = prepare_channel_execution(
        channel_name,
        &prepared,
        room_config,
        sample_rate,
        shared_mean_spl,
    )?;
    let mut eq_resources = prepared.eq_resources().clone();
    eq_resources.target = prepare_eq_target(execution.target().effective_target(room_config))
        .map_err(|error| AutoeqError::OptimizationFailed {
            message: format!("Failed to prepare EQ resources for channel {channel_name}: {error}"),
        })?;

    let phase_linear = room_config.optimizer.processing_mode == ProcessingMode::PhaseLinear;
    let mut callback = callback;
    if phase_linear && let Some(callback) = callback.as_mut() {
        callback(1, execution.target().pre_score, None);
    }

    let reservation = if room_config.optimizer.processing_mode == ProcessingMode::Hybrid
        && room_config.optimizer.mixed_config.is_some()
    {
        Some(
            reserve_mixed_crossover_sidecar(output_dir, channel_name, sample_rate).map_err(
                |error| AutoeqError::OptimizationFailed {
                    message: format!(
                        "Failed to reserve convolution artifact for channel {channel_name}: {error}"
                    ),
                },
            )?,
        )
    } else {
        reserve_channel_convolution_sidecar(
            output_dir,
            channel_name,
            room_config.optimizer.processing_mode.clone(),
            sample_rate,
        )
        .map_err(|error| AutoeqError::OptimizationFailed {
            message: format!(
                "Failed to reserve convolution artifact for channel {channel_name}: {error}"
            ),
        })?
    };
    let sidecar_reference = reservation
        .as_ref()
        .map(|reservation| reservation.reference().clone());

    let result = execute_prepared_channel(
        channel_name,
        &prepared,
        room_config,
        sample_rate,
        &execution,
        &eq_resources,
        sidecar_reference,
        if phase_linear { None } else { callback.take() },
    )?;

    if let Some(generated) = result.convolution_sidecar.as_ref() {
        let reservation =
            reservation
                .as_ref()
                .ok_or_else(|| AutoeqError::InvalidConfiguration {
                    message: format!(
                        "channel '{channel_name}' generated an unreserved convolution sidecar"
                    ),
                })?;
        let coefficients =
            result
                .fir_coeffs
                .as_deref()
                .ok_or_else(|| AutoeqError::InvalidConfiguration {
                    message: format!(
                        "channel '{channel_name}' generated sidecar metadata without coefficients"
                    ),
                })?;
        if let Err(error) =
            persist_convolution_sidecar(reservation, generated, coefficients, sample_rate as u32)
        {
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
    if phase_linear && let Some(callback) = callback.as_mut() {
        callback(2, result.post_score, None);
    }
    Ok(result_tuple(result))
}

fn result_tuple(result: ChannelProcessingResult) -> ChannelWorkflowResult {
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use ndarray::Array1;
    use roomeq_model::{FirConfig, MixedModeConfig, OptimizerConfig, TargetCurveConfig};

    use super::*;

    fn curve() -> Curve {
        Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 96),
            spl: Array1::from_elem(96, 80.0),
            ..Curve::default()
        }
    }

    fn config(mode: ProcessingMode) -> RoomConfig {
        RoomConfig {
            optimizer: OptimizerConfig {
                processing_mode: mode,
                num_filters: 1,
                max_iter: 8,
                population: 6,
                min_freq: 20.0,
                max_freq: 500.0,
                refine: false,
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        }
    }

    #[test]
    fn low_latency_preserves_probe_arrival_and_shared_level() {
        let directory = tempfile::tempdir().unwrap();
        let result = process_single_channel(
            "left",
            &MeasurementSource::InMemory(curve()),
            &config(ProcessingMode::LowLatency),
            48_000.0,
            directory.path(),
            None,
            Some(3.5),
            Some(82.0),
        )
        .unwrap();

        assert_eq!(result.7, Some(3.5));
        assert!((result.6 - 82.0).abs() < 1e-6);
        assert!(result.8.is_none());
    }

    #[test]
    fn phase_linear_persists_sidecar_before_completion_callback() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = config(ProcessingMode::PhaseLinear);
        config.optimizer.fir = Some(FirConfig {
            taps: 128,
            ..FirConfig::default()
        });
        let events = Arc::new(Mutex::new(Vec::new()));
        let callback_events = Arc::clone(&events);
        let expected_sidecar = directory.path().join("left_fir_48000hz.wav");
        let callback: OptimProgressCallback = Box::new(move |iteration, _, _| {
            if iteration == 2 {
                assert!(expected_sidecar.exists());
            }
            callback_events.lock().unwrap().push(iteration);
            roomeq_engine::CallbackAction::Continue
        });

        let result = process_single_channel(
            "left",
            &MeasurementSource::InMemory(curve()),
            &config,
            48_000.0,
            directory.path(),
            Some(callback),
            None,
            None,
        )
        .unwrap();

        assert_eq!(*events.lock().unwrap(), vec![1, 2]);
        assert_eq!(result.8.as_ref().unwrap().len(), 128);
        assert!(directory.path().join("left_fir_48000hz.wav").exists());
    }

    #[test]
    fn mixed_crossover_hybrid_persists_band_fir() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = config(ProcessingMode::Hybrid);
        config.optimizer.mixed_config = Some(MixedModeConfig {
            crossover_freq: 200.0,
            fir_band: "low".to_string(),
            ..MixedModeConfig::default()
        });
        config.optimizer.fir = Some(FirConfig {
            taps: 64,
            ..FirConfig::default()
        });

        process_single_channel(
            "left",
            &MeasurementSource::InMemory(curve()),
            &config,
            48_000.0,
            directory.path(),
            None,
            None,
            None,
        )
        .unwrap();

        assert!(directory.path().join("left_band_fir_48000hz.wav").exists());
    }

    #[test]
    fn target_curve_path_is_prepared_by_workflow() {
        use std::io::Write;

        let directory = tempfile::tempdir().unwrap();
        let mut target_file = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
        writeln!(target_file, "frequency,spl").unwrap();
        for frequency in curve().freq {
            writeln!(target_file, "{frequency},0").unwrap();
        }
        let mut config = config(ProcessingMode::LowLatency);
        config.target_curve = Some(TargetCurveConfig::Path(target_file.path().to_path_buf()));

        assert!(
            process_single_channel(
                "left",
                &MeasurementSource::InMemory(curve()),
                &config,
                48_000.0,
                directory.path(),
                None,
                None,
                None,
            )
            .is_ok()
        );
    }
}
