//! Preparation and path-free execution for one RoomEQ channel.

use autoeq_core::{AutoeqError, Curve, Result};
use autoeq_optim::optim::OptimProgressCallback;
use log::{debug, info, warn};
use roomeq_model::{OptimizerConfig, ProcessingMode, RoomConfig};

use crate::PreparedChannelInput;
use crate::channel_fir::{FirChannelMode, FirChannelRequest, process_fir_channel};
use crate::channel_iir::{IirChannelMode, IirChannelRequest, process_iir_channel};
use crate::channel_preprocessing::{PreprocessedFeatures, preprocess_channel};
use crate::channel_result::{ChannelProcessingResult, ConvolutionSidecarReference};
use crate::channel_target::{TargetContext, build_target_context_with_prepared_target};
use crate::eq::EqResources;
use crate::mixed_crossover::{MixedCrossoverRequest, process_mixed_crossover};

/// Deterministic state shared by workflow resource preparation and execution.
pub struct PreparedChannelExecution {
    target: TargetContext,
    preprocessed: PreprocessedFeatures,
    optimizer: OptimizerConfig,
}

impl PreparedChannelExecution {
    pub fn target(&self) -> &TargetContext {
        &self.target
    }
}

/// Build deterministic execution state from a workflow-prepared channel.
pub fn prepare_channel_execution(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    room_config: &RoomConfig,
    sample_rate: f64,
    shared_mean_spl: Option<f64>,
) -> Result<PreparedChannelExecution> {
    let curve = prepared.measurements().representative();
    if curve.freq.is_empty() || curve.spl.is_empty() {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!("Empty measurement for channel '{channel_name}'"),
        });
    }
    debug!(
        "  Loaded measurement: {:.1} Hz - {:.1} Hz",
        curve.freq[0],
        curve.freq[curve.freq.len() - 1]
    );
    warn_if_optimizer_bounds_exceed_data(channel_name, curve, &room_config.optimizer);

    let mut target = build_target_context_with_prepared_target(
        channel_name,
        room_config,
        curve,
        shared_mean_spl,
        prepared.eq_resources().target.as_ref(),
    )?;
    // A configured band is not measurement evidence. Clamp before scoring,
    // preprocessing, and FIR/PEQ dispatch so all stages use the same support.
    // In particular, out-of-band PEQs can still change the measured response;
    // warning that such filters will be "ignored" does not make them harmless.
    target.min_freq = target.min_freq.max(curve.freq[0]);
    target.max_freq = target.max_freq.min(curve.freq[curve.freq.len() - 1]);
    if target.min_freq >= target.max_freq {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!(
                "Channel '{channel_name}': configured correction band does not overlap measurement support"
            ),
        });
    }
    let preprocessed = preprocess_channel(
        channel_name,
        prepared,
        room_config,
        sample_rate,
        shared_mean_spl,
        &mut target,
    );
    let optimizer = build_clamped_optimizer(
        channel_name,
        room_config,
        curve,
        &preprocessed.curve_for_optim,
        preprocessed.score_min_freq,
        target.max_freq,
        target.target_tilt_curve.as_ref(),
        preprocessed.broadband_enabled,
    );
    Ok(PreparedChannelExecution {
        target,
        preprocessed,
        optimizer,
    })
}

/// Execute one prepared channel without filesystem or artifact access.
#[allow(clippy::too_many_arguments)]
pub fn execute_prepared_channel(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    room_config: &RoomConfig,
    sample_rate: f64,
    execution: &PreparedChannelExecution,
    eq_resources: &EqResources,
    sidecar_reference: Option<ConvolutionSidecarReference>,
    callback: Option<OptimProgressCallback>,
) -> Result<ChannelProcessingResult> {
    match room_config.optimizer.processing_mode {
        ProcessingMode::PhaseLinear => process_fir_channel(FirChannelRequest {
            mode: FirChannelMode::PhaseLinear,
            channel_name,
            prepared,
            room_config,
            sample_rate,
            target: &execution.target,
            preprocessed: &execution.preprocessed,
            optimizer: &execution.optimizer,
            eq_resources,
            sidecar_reference: required_sidecar(sidecar_reference, channel_name)?,
            callback,
        }),
        ProcessingMode::Hybrid => {
            let sidecar_reference = required_sidecar(sidecar_reference, channel_name)?;
            if let Some(mixed_config) = &room_config.optimizer.mixed_config {
                let preference_filters = crate::channel_preference::build_preference_filters(
                    channel_name,
                    room_config,
                    sample_rate,
                );
                process_mixed_crossover(MixedCrossoverRequest {
                    channel_name,
                    curve: &execution.preprocessed.curve_for_optim,
                    target: &execution.target,
                    preference_filters: &preference_filters,
                    mixed_config,
                    optimizer: &execution.optimizer,
                    eq_resources,
                    sample_rate,
                    min_freq: execution.target.min_freq,
                    max_freq: execution.target.max_freq,
                    mean_spl: execution.target.mean_spl,
                    pre_score: execution.target.pre_score,
                    arrival_time_ms: prepared.arrival_time_ms(),
                    sidecar_reference,
                    callback,
                })
            } else {
                process_fir_channel(FirChannelRequest {
                    mode: FirChannelMode::Hybrid,
                    channel_name,
                    prepared,
                    room_config,
                    sample_rate,
                    target: &execution.target,
                    preprocessed: &execution.preprocessed,
                    optimizer: &execution.optimizer,
                    eq_resources,
                    sidecar_reference,
                    callback,
                })
            }
        }
        ProcessingMode::MixedPhase => process_fir_channel(FirChannelRequest {
            mode: FirChannelMode::MixedPhase,
            channel_name,
            prepared,
            room_config,
            sample_rate,
            target: &execution.target,
            preprocessed: &execution.preprocessed,
            optimizer: &execution.optimizer,
            eq_resources,
            sidecar_reference: required_sidecar(sidecar_reference, channel_name)?,
            callback,
        }),
        ProcessingMode::LowLatency => process_iir(
            IirChannelMode::LowLatency,
            channel_name,
            prepared,
            room_config,
            sample_rate,
            execution,
            eq_resources,
            callback,
        ),
        ProcessingMode::WarpedIir => process_iir(
            IirChannelMode::WarpedIir,
            channel_name,
            prepared,
            room_config,
            sample_rate,
            execution,
            eq_resources,
            callback,
        ),
        ProcessingMode::KautzModal => process_iir(
            IirChannelMode::KautzModal,
            channel_name,
            prepared,
            room_config,
            sample_rate,
            execution,
            eq_resources,
            callback,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn process_iir(
    mode: IirChannelMode,
    channel_name: &str,
    prepared: &PreparedChannelInput,
    room_config: &RoomConfig,
    sample_rate: f64,
    execution: &PreparedChannelExecution,
    eq_resources: &EqResources,
    callback: Option<OptimProgressCallback>,
) -> Result<ChannelProcessingResult> {
    process_iir_channel(IirChannelRequest {
        mode,
        channel_name,
        prepared,
        room_config,
        sample_rate,
        target: &execution.target,
        preprocessed: &execution.preprocessed,
        optimizer: &execution.optimizer,
        eq_resources,
        callback,
    })
}

fn required_sidecar(
    reference: Option<ConvolutionSidecarReference>,
    channel_name: &str,
) -> Result<ConvolutionSidecarReference> {
    reference.ok_or_else(|| AutoeqError::InvalidConfiguration {
        message: format!("channel '{channel_name}' requires a convolution sidecar reference"),
    })
}

fn is_subwoofer_measurement_channel(channel_name: &str, room_config: &RoomConfig) -> bool {
    roomeq_model::home_cinema::role_for_channel(channel_name).is_sub_or_lfe()
        || room_config
            .system
            .as_ref()
            .and_then(|system| {
                let subwoofers = system.subwoofers.as_ref()?;
                let measurement_key = system.speakers.get(channel_name)?;
                Some(subwoofers.mapping.contains_key(measurement_key))
            })
            .unwrap_or(false)
}

#[allow(clippy::too_many_arguments)]
fn sub_optimizer_upper_bound(measured_upper: Option<f64>, crossover_upper: Option<f64>) -> f64 {
    const SUB_UPPER_FALLBACK_HZ: f64 = 160.0;
    match (measured_upper, crossover_upper) {
        (Some(measured), Some(crossover)) => measured.min(crossover),
        (Some(measured), None) => measured,
        (None, Some(crossover)) => crossover,
        (None, None) => SUB_UPPER_FALLBACK_HZ,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_clamped_optimizer(
    channel_name: &str,
    room_config: &RoomConfig,
    curve_raw: &Curve,
    curve_for_optim: &Curve,
    min_freq: f64,
    max_freq: f64,
    target_tilt_curve: Option<&Curve>,
    broadband_enabled: bool,
) -> OptimizerConfig {
    let is_sub_channel = is_subwoofer_measurement_channel(channel_name, room_config);
    let mut optimizer = room_config.optimizer.clone();
    optimizer.min_freq = min_freq;
    optimizer.max_freq = optimizer.max_freq.min(max_freq);
    optimizer.ssir_wav_path = None;

    if is_sub_channel {
        let measured_upper = curve_raw.freq.last().map(|_| {
            crate::group_processing::sub_optimizer_config(
                std::slice::from_ref(curve_raw),
                &room_config.optimizer,
            )
            .max_freq
        });
        let crossover_upper =
            roomeq_model::home_cinema::bass_management_crossover_frequency_hz(room_config)
                .map(|frequency| 2.0 * frequency);
        let upper = sub_optimizer_upper_bound(measured_upper, crossover_upper);
        info!(
            "  Sub channel '{}': clamping optimizer upper bound to {:.1} Hz (final useful response={}, 2*crossover={})",
            channel_name,
            upper,
            measured_upper
                .map(|high| format!("{high:.1} Hz"))
                .unwrap_or_else(|| "n/a".to_string()),
            crossover_upper
                .map(|high| format!("{high:.1} Hz"))
                .unwrap_or_else(|| "n/a".to_string()),
        );
        optimizer.max_freq = optimizer.max_freq.min(upper);
    }

    if is_sub_channel && let Some(sub_config) = &room_config.optimizer.sub_config {
        info!(
            "  Applying sub_config overrides: num_filters={}, max_db={:+.1}, min_db={:+.1}, max_q={:.1}",
            sub_config.num_filters, sub_config.max_db, sub_config.min_db, sub_config.max_q,
        );
        optimizer.num_filters = sub_config.num_filters;
        optimizer.max_db = sub_config.max_db;
        optimizer.min_db = sub_config.min_db;
        optimizer.min_q = sub_config.min_q;
        optimizer.max_q = sub_config.max_q;
    }

    if optimizer
        .auto_optimizer
        .as_ref()
        .is_some_and(|auto| auto.enabled)
    {
        let detected_f3_hz = match crate::excursion::detect_f3_with_config(
            curve_for_optim,
            None,
            optimizer.excursion_protection.as_ref(),
        ) {
            Ok(result) if result.f3_hz > min_freq && result.f3_hz < max_freq => Some(result.f3_hz),
            Ok(_) => None,
            Err(error) => {
                debug!("  Auto optimizer: F3 detection skipped: {error}");
                None
            }
        };
        let context = roomeq_model::auto_tune::AutoOptimizerContext {
            is_sub_channel,
            effective_min_freq: min_freq,
            effective_max_freq: max_freq,
            detected_f3_hz,
            schroeder_hz: roomeq_model::auto_tune::resolved_schroeder_hz(&optimizer),
            target_tilt_active: target_tilt_curve.is_some(),
            broadband_enabled,
        };
        optimizer = roomeq_model::auto_tune::resolve_auto_optimizer_config(
            curve_for_optim,
            &optimizer,
            &context,
        );
    }
    optimizer
}

fn warn_if_optimizer_bounds_exceed_data(
    channel_name: &str,
    curve: &Curve,
    optimizer: &OptimizerConfig,
) {
    let Some(data_min) = curve.freq.first().copied() else {
        return;
    };
    let Some(data_max) = curve.freq.last().copied() else {
        return;
    };
    let log_margin = 0.05;
    let min_tolerance = data_min * 10_f64.powf(-log_margin);
    let max_tolerance = data_max * 10_f64.powf(log_margin);
    if optimizer.min_freq < min_tolerance {
        warn!(
            "Channel '{}': optimizer.min_freq={:.1} Hz is below measurement minimum {:.1} Hz; intersecting the correction band with measured support.",
            channel_name, optimizer.min_freq, data_min,
        );
    }
    if optimizer.max_freq > max_tolerance {
        warn!(
            "Channel '{}': optimizer.max_freq={:.1} Hz is above measurement maximum {:.1} Hz; intersecting the correction band with measured support.",
            channel_name, optimizer.max_freq, data_max,
        );
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use roomeq_model::SubOptimizerConfig;

    use super::*;

    fn bounded_prepared_measurement() -> PreparedChannelInput {
        let curve = Curve {
            freq: ndarray::array![100.0, 160.0, 300.0, 6000.0],
            spl: Array1::from_elem(4, 80.0),
            ..Curve::default()
        };
        PreparedChannelInput::new(
            crate::PreparedChannelMeasurements::new(curve.clone(), vec![curve], false),
            None,
            crate::PreparedCea2034::default(),
            EqResources::default(),
        )
    }

    #[test]
    fn preparation_intersects_requested_band_with_measurement_before_dispatch() {
        let prepared = bounded_prepared_measurement();
        for mode in [
            ProcessingMode::LowLatency,
            ProcessingMode::PhaseLinear,
            ProcessingMode::Hybrid,
        ] {
            let mut config = RoomConfig::default();
            config.optimizer.processing_mode = mode;
            config.optimizer.min_freq = 20.0;
            config.optimizer.max_freq = 20_000.0;
            let execution =
                prepare_channel_execution("left", &prepared, &config, 48_000.0, None).unwrap();
            assert_eq!(execution.target.min_freq, 100.0);
            assert_eq!(execution.optimizer.min_freq, 100.0);
            assert_eq!(execution.preprocessed.score_min_freq, 100.0);
            assert_eq!(execution.target.max_freq, 6000.0);
            assert_eq!(execution.optimizer.max_freq, 6000.0);
        }
    }

    #[test]
    fn preparation_rejects_disjoint_measurement_and_correction_bands() {
        let prepared = bounded_prepared_measurement();
        for (low, high) in [(20.0, 80.0), (7000.0, 20_000.0)] {
            let mut config = RoomConfig::default();
            config.optimizer.min_freq = low;
            config.optimizer.max_freq = high;
            let error = prepare_channel_execution("left", &prepared, &config, 48_000.0, None)
                .err()
                .expect("disjoint bands must not synthesize unsupported correction");
            assert!(
                error
                    .to_string()
                    .contains("does not overlap measurement support")
            );
        }
    }

    fn curve() -> Curve {
        Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(500.0), 64),
            spl: Array1::from_elem(64, 80.0),
            ..Curve::default()
        }
    }

    #[test]
    fn clamping_clears_prepared_ssir_path() {
        let curve = curve();
        let config = RoomConfig {
            optimizer: OptimizerConfig {
                min_freq: 20.0,
                max_freq: 500.0,
                ssir_wav_path: Some("prepared.wav".into()),
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };
        let optimizer =
            build_clamped_optimizer("left", &config, &curve, &curve, 20.0, 500.0, None, false);
        assert_eq!(optimizer.max_freq, 500.0);
        assert!(optimizer.ssir_wav_path.is_none());
    }

    #[test]
    fn clamping_limits_sub_and_applies_overrides() {
        let curve = curve();
        let config = RoomConfig {
            optimizer: OptimizerConfig {
                min_freq: 20.0,
                max_freq: 500.0,
                sub_config: Some(SubOptimizerConfig {
                    num_filters: 7,
                    max_db: 12.0,
                    min_db: -15.0,
                    min_q: 0.5,
                    max_q: 15.0,
                }),
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };
        let optimizer =
            build_clamped_optimizer("LFE", &config, &curve, &curve, 20.0, 500.0, None, false);
        assert!(optimizer.max_freq < 500.0);
        assert_eq!(optimizer.num_filters, 7);
        assert_eq!(optimizer.max_db, 12.0);
        assert_eq!(optimizer.min_db, -15.0);
    }

    #[test]
    fn sub_execution_does_not_reintroduce_first_peak_passband_clamp() {
        let curve = Curve {
            freq: ndarray::array![20.0, 30.0, 35.0, 50.0, 80.0, 130.0, 180.0, 200.0],
            spl: ndarray::array![80.0, 85.0, 60.0, 40.0, 82.0, 68.0, 45.0, 20.0],
            ..Default::default()
        };
        let mut config = RoomConfig::default();
        config.optimizer.min_freq = 20.0;
        config.optimizer.max_freq = 16000.0;
        let bounded =
            build_clamped_optimizer("LFE", &config, &curve, &curve, 20.0, 130.0, None, false);
        assert_eq!(bounded.max_freq, 130.0);
    }

    #[test]
    fn sub_upper_bound_is_the_tighter_of_measurement_and_crossover() {
        assert_eq!(sub_optimizer_upper_bound(Some(300.0), Some(160.0)), 160.0);
        assert_eq!(sub_optimizer_upper_bound(Some(90.0), Some(200.0)), 90.0);
        assert_eq!(sub_optimizer_upper_bound(Some(120.0), None), 120.0);
        assert_eq!(sub_optimizer_upper_bound(None, Some(180.0)), 180.0);
    }
}
