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

/// Reference curve for from-measurement slope inference.
///
/// Single-band and unrestricted channels score the usable curve directly.
/// Multi-segment support restricts inference to the widest segment in
/// octaves so the internal gap never tilts the estimate.
fn slope_reference_curve<'a>(
    prepared: &PreparedChannelInput,
    curve: &'a Curve,
) -> std::borrow::Cow<'a, Curve> {
    let bands = prepared.valid_bands_hz();
    if bands.len() < 2 {
        return std::borrow::Cow::Borrowed(curve);
    }
    let widest = bands
        .iter()
        .max_by(|a, b| (a[1] / a[0]).total_cmp(&(b[1] / b[0])))
        .copied();
    widest
        .and_then(|band| curve.select_frequency_band(band).ok())
        .map(std::borrow::Cow::Owned)
        .unwrap_or_else(|| std::borrow::Cow::Borrowed(curve))
}

/// Build deterministic execution state from a workflow-prepared channel.
pub fn prepare_channel_execution(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    room_config: &RoomConfig,
    sample_rate: f64,
    shared_mean_spl: Option<f64>,
) -> Result<PreparedChannelExecution> {
    let raw_curve = prepared.measurements().representative();
    let usable = prepared.usable_curve(raw_curve)?;
    let curve = usable.as_ref();
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

    // Keep the requested target on the original reporting grid. Only measured
    // slope inference is restricted; level and score are recomputed below.
    // Multi-segment support infers slope on the widest segment in octaves so
    // the internal gap never tilts the estimate.
    let mut target_config = std::borrow::Cow::Borrowed(room_config);
    if prepared.has_valid_bands()
        && room_config
            .optimizer
            .from_measurement_slope_override
            .is_none()
        && room_config
            .optimizer
            .target_response
            .as_ref()
            .is_some_and(|target| target.shape == roomeq_model::TargetShape::FromMeasurement)
        && !roomeq_model::home_cinema::role_for_channel(channel_name).is_sub_or_lfe()
    {
        target_config
            .to_mut()
            .optimizer
            .from_measurement_slope_override = Some(
            roomeq_analysis::slope::estimate_slope_db_per_octave(
                slope_reference_curve(prepared, curve).as_ref(),
                roomeq_analysis::slope::DEFAULT_SLOPE_MIN_FREQ,
                roomeq_analysis::slope::DEFAULT_SLOPE_MAX_FREQ,
            )
            .unwrap_or(0.0),
        );
    }
    let mut target = build_target_context_with_prepared_target(
        channel_name,
        &target_config,
        raw_curve,
        shared_mean_spl,
        prepared.eq_resources().target.as_ref(),
    )?;
    // A configured band is not measurement evidence. Clamp before scoring,
    // preprocessing, and FIR/PEQ dispatch so all stages use the same support.
    // In particular, out-of-band PEQs can still change the measured response;
    // warning that such filters will be "ignored" does not make them harmless.
    target.min_freq = target.min_freq.max(curve.freq[0]);
    target.max_freq = target.max_freq.min(curve.freq[curve.freq.len() - 1]);
    // Clamp to the support hull. Gap bins are absent from the usable curve,
    // so scoring and optimization never consume them; the post-realization
    // segment check additionally refuses filters centered in a gap.
    if let (Some([first_low, _]), Some([_, last_high])) = (
        prepared.valid_bands_hz().first(),
        prepared.valid_bands_hz().last(),
    ) {
        target.min_freq = target.min_freq.max(*first_low);
        target.max_freq = target.max_freq.min(*last_high);
    }
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
    )?;
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
    execute_prepared_channel_with_optional_exact_checkpoint(
        channel_name,
        prepared,
        room_config,
        sample_rate,
        execution,
        eq_resources,
        sidecar_reference,
        callback,
        None,
    )
}

/// Execute one prepared channel with an optional exact-DE recovery state.
#[allow(clippy::too_many_arguments)]
pub fn execute_prepared_channel_with_exact_checkpoint(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    room_config: &RoomConfig,
    sample_rate: f64,
    execution: &PreparedChannelExecution,
    eq_resources: &EqResources,
    sidecar_reference: Option<ConvolutionSidecarReference>,
    callback: Option<OptimProgressCallback>,
    exact: crate::eq::exact_recovery::ExactDERecoveryOptions,
) -> Result<ChannelProcessingResult> {
    execute_prepared_channel_with_optional_exact_checkpoint(
        channel_name,
        prepared,
        room_config,
        sample_rate,
        execution,
        eq_resources,
        sidecar_reference,
        callback,
        Some(exact),
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_prepared_channel_with_optional_exact_checkpoint(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    room_config: &RoomConfig,
    sample_rate: f64,
    execution: &PreparedChannelExecution,
    eq_resources: &EqResources,
    sidecar_reference: Option<ConvolutionSidecarReference>,
    callback: Option<OptimProgressCallback>,
    mut exact: Option<crate::eq::exact_recovery::ExactDERecoveryOptions>,
) -> Result<ChannelProcessingResult> {
    if exact.is_some() && room_config.optimizer.processing_mode != ProcessingMode::LowLatency {
        return Err(AutoeqError::InvalidConfiguration {
            message: "exact DE recovery requires LowLatency processing mode".into(),
        });
    }
    // Multi-segment authorization runs once per channel, after every
    // processing mode assembles its realized result.
    let mut result = match room_config.optimizer.processing_mode {
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
                    prepared: Some(prepared),
                    curve: &execution.preprocessed.curve_for_optim,
                    target: &execution.target,
                    preference_filters: &preference_filters,
                    excursion_filters: &execution.preprocessed.excursion_filters,
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
            exact.take(),
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
            None,
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
            None,
        ),
    }?;
    result.segment_support = crate::segment_support::assess_segment_support(
        channel_name,
        prepared,
        &result.raw_pre_eq_curve,
        &result.raw_post_eq_curve,
        &result.filters,
        result.fir_coeffs.as_deref(),
        sample_rate,
    )?;
    // Measured-room acoustics ride with the delivered chain when the channel
    // declared an optimization-time IR. Incomplete IRs leave the fields
    // absent and the report cells pending.
    if let Some(impulse) = prepared.eq_resources().impulse_response.as_ref() {
        if result.channel.early_reflections.is_none() {
            result.channel.early_reflections = crate::ir_acoustics::measured_early_reflections(
                &impulse.samples,
                impulse.sample_rate,
            );
        }
        if result.channel.t60_octaves.is_none() {
            result.channel.t60_octaves =
                crate::ir_acoustics::measured_octave_t60(&impulse.samples, impulse.sample_rate);
        }
        if (result.channel.waterfall.is_none() || result.channel.resonance_decays.is_none())
            && let Some((waterfall, decays)) =
                crate::ir_acoustics::measured_waterfall(&impulse.samples, impulse.sample_rate)
        {
            result.channel.waterfall = Some(waterfall);
            result.channel.resonance_decays = Some(decays);
        }
        if result.channel.wavelet.is_none() {
            result.channel.wavelet =
                crate::ir_acoustics::measured_wavelet(&impulse.samples, impulse.sample_rate);
        }
    }
    Ok(result)
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
    exact: Option<crate::eq::exact_recovery::ExactDERecoveryOptions>,
) -> Result<ChannelProcessingResult> {
    let request = IirChannelRequest {
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
    };
    match exact {
        Some(exact) => {
            crate::channel_iir::process_iir_channel_with_exact_checkpoint(request, exact)
        }
        None => process_iir_channel(request),
    }
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
                Some(subwoofers.contains_measurement(measurement_key))
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
    use roomeq_model::{ExcursionProtectionConfig, FirConfig, MixedModeConfig, SubOptimizerConfig};

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

    /// One-second 200 Hz decaying tone (τ = 0.2 s): a declared
    /// optimization-time IR with a complete 500 ms post-peak window.
    fn decaying_tone_impulse() -> crate::eq::PreparedImpulseResponse {
        let rate = 48_000.0;
        let samples = (0..48_000)
            .map(|i| {
                let t = f64::from(i) / rate;
                (2.0 * std::f64::consts::PI * 200.0 * t).cos() as f32 * (-t / 0.2).exp() as f32
            })
            .collect();
        crate::eq::PreparedImpulseResponse {
            samples,
            sample_rate: rate,
        }
    }

    #[test]
    fn declared_ir_populates_measured_room_acoustics() {
        let curve = Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(500.0), 64),
            spl: Array1::from_elem(64, 80.0),
            ..Curve::default()
        };
        let resources = EqResources {
            impulse_response: Some(decaying_tone_impulse()),
            ..EqResources::default()
        };
        let prepared = PreparedChannelInput::new(
            crate::PreparedChannelMeasurements::new(curve.clone(), vec![curve], false),
            None,
            crate::PreparedCea2034::default(),
            resources,
        );
        let mut config = RoomConfig::default();
        config.optimizer.processing_mode = ProcessingMode::LowLatency;
        config.optimizer.min_freq = 50.0;
        config.optimizer.max_freq = 200.0;
        config.optimizer.num_filters = 1;
        config.optimizer.max_iter = 10;
        config.optimizer.population = 6;
        config.optimizer.refine = false;
        config.optimizer.seed = Some(7);
        config.optimizer.parallel_threads = Some(1);
        let execution =
            prepare_channel_execution("left", &prepared, &config, 48_000.0, None).unwrap();
        let result = execute_prepared_channel(
            "left",
            &prepared,
            &config,
            48_000.0,
            &execution,
            prepared.eq_resources(),
            None,
            None,
        )
        .unwrap();
        assert!(
            result.channel.early_reflections.is_some(),
            "declared IR must report early reflections"
        );
        assert!(
            result.channel.t60_octaves.is_some(),
            "declared IR must report octave T60"
        );
        let waterfall = result
            .channel
            .waterfall
            .as_ref()
            .expect("declared IR must report a waterfall grid");
        assert_eq!(waterfall.method, "hann_stft_waterfall_v1");
        assert_eq!(waterfall.reference, "full_grid_peak");
        let decays = result
            .channel
            .resonance_decays
            .as_ref()
            .expect("declared IR must report resonance decays");
        assert_eq!(decays.slice_ms, 60.0);
        assert!(
            !decays.decays.is_empty(),
            "decaying tone is a 60 ms resonance"
        );
        let wavelet = result
            .channel
            .wavelet
            .as_ref()
            .expect("declared IR must report a wavelet heatmap");
        assert_eq!(wavelet.method, "complex_morlet_three_cycle_v1");
        assert_eq!(wavelet.display_range_db, [-30.0, 0.0]);
    }

    #[test]
    fn declared_band_excludes_unusable_samples_from_channel_level_reference() {
        let mut config = RoomConfig::default();
        config.optimizer.min_freq = 50.0;
        config.optimizer.max_freq = 200.0;
        let mut references = Vec::new();
        for unusable_level in [60.0, 100.0] {
            let curve = Curve {
                freq: ndarray::array![
                    20.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 4000.0
                ],
                spl: ndarray::array![
                    unusable_level,
                    80.0,
                    85.0,
                    80.0,
                    80.0,
                    unusable_level,
                    unusable_level,
                    unusable_level,
                    unusable_level
                ],
                ..Default::default()
            };
            let prepared = PreparedChannelInput::from_measurements(
                crate::PreparedChannelMeasurements::new(curve.clone(), vec![curve], false),
            )
            .with_valid_band_hz([50.0, 300.0])
            .unwrap();
            let execution =
                prepare_channel_execution("left", &prepared, &config, 48_000.0, None).unwrap();
            references.push(execution.target.mean_spl);
        }
        assert!(
            (references[0] - references[1]).abs() < 1e-10,
            "unusable samples changed the reference: {references:?}"
        );
    }

    #[test]
    fn declared_band_keeps_delivered_correction_independent_of_unusable_samples() {
        for mode in [
            ProcessingMode::LowLatency,
            ProcessingMode::PhaseLinear,
            ProcessingMode::Hybrid,
        ] {
            for multiple in [false, true] {
                let mut config = RoomConfig::default();
                config.optimizer.processing_mode = mode.clone();
                config.optimizer.min_freq = 50.0;
                config.optimizer.max_freq = 200.0;
                config.optimizer.num_filters = 1;
                config.optimizer.max_iter = 10;
                config.optimizer.population = 6;
                config.optimizer.refine = false;
                config.optimizer.seed = Some(19);
                config.optimizer.parallel_threads = Some(1);
                config.optimizer.fir = Some(roomeq_model::FirConfig {
                    taps: 64,
                    phase: "linear".into(),
                    ..Default::default()
                });
                config.optimizer.target_response = Some(roomeq_model::TargetResponseConfig {
                    shape: roomeq_model::TargetShape::Custom,
                    slope_db_per_octave: -0.5,
                    broadband_precorrection: true,
                    ..Default::default()
                });
                if multiple {
                    config.optimizer.multi_measurement = Some(Default::default());
                }
                let mut outputs = Vec::new();
                for unusable_level in [60.0, 100.0] {
                    let curve = Curve {
                        freq: ndarray::array![
                            20.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 2000.0,
                            4000.0
                        ],
                        spl: ndarray::array![
                            unusable_level,
                            80.0,
                            83.0,
                            85.0,
                            83.0,
                            80.0,
                            80.0,
                            unusable_level,
                            unusable_level,
                            unusable_level,
                            unusable_level
                        ],
                        ..Default::default()
                    };
                    let mut second = curve.clone();
                    second.spl += 2.0;
                    let prepared = PreparedChannelInput::from_measurements(
                        crate::PreparedChannelMeasurements::new(
                            curve.clone(),
                            if multiple {
                                vec![curve, second]
                            } else {
                                vec![curve]
                            },
                            multiple,
                        ),
                    )
                    .with_valid_band_hz([50.0, 300.0])
                    .unwrap();
                    let execution =
                        prepare_channel_execution("left", &prepared, &config, 48_000.0, None)
                            .unwrap();
                    assert_eq!(
                        execution.target.target_tilt_curve.as_ref().unwrap().freq,
                        prepared.measurements().representative().freq
                    );
                    let result = execute_prepared_channel(
                        "left",
                        &prepared,
                        &config,
                        48_000.0,
                        &execution,
                        prepared.eq_resources(),
                        Some(ConvolutionSidecarReference::new("left.wav").unwrap()),
                        None,
                    )
                    .unwrap();
                    assert_eq!(result.raw_pre_eq_curve.freq.len(), 11);
                    assert_eq!(result.raw_post_eq_curve.freq.len(), 11);
                    assert_eq!(result.raw_pre_eq_curve.spl[0], unusable_level);
                    if mode == ProcessingMode::LowLatency {
                        assert!(
                            !result.filters.is_empty(),
                            "witness must exercise actual PEQ design"
                        );
                    } else {
                        assert!(
                            result
                                .fir_coeffs
                                .as_ref()
                                .is_some_and(|taps| taps.len() == 64)
                        );
                    }
                    outputs.push((
                        serde_json::to_value(&result.channel.plugins).unwrap(),
                        result.fir_coeffs,
                        result.mean_spl,
                    ));
                }
                assert_eq!(outputs[0], outputs[1], "{mode:?}, multiple={multiple}");
            }
        }
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
    fn hybrid_mixed_crossover_serializes_and_realizes_excursion_protection() {
        const SAMPLE_RATE: f64 = 48_000.0;
        const IR_FILE: &str = "hybrid_low_band.wav";

        let raw_curve = Curve {
            freq: Array1::logspace(10.0, 20.0_f64.log10(), 1_600.0_f64.log10(), 161),
            spl: Array1::from_elem(161, 80.0),
            phase: Some(Array1::zeros(161)),
            ..Curve::default()
        };
        let prepared =
            PreparedChannelInput::from_measurements(crate::PreparedChannelMeasurements::new(
                raw_curve.clone(),
                vec![raw_curve.clone()],
                false,
            ));
        let mut config = RoomConfig::default();
        config.optimizer.processing_mode = ProcessingMode::Hybrid;
        config.optimizer.min_freq = 20.0;
        config.optimizer.max_freq = 1_600.0;
        config.optimizer.num_filters = 1;
        config.optimizer.max_iter = 4;
        config.optimizer.population = 6;
        config.optimizer.refine = false;
        config.optimizer.seed = Some(7);
        config.optimizer.parallel_threads = Some(1);
        config.optimizer.fir = Some(FirConfig {
            taps: 64,
            phase: "linear".into(),
            ..FirConfig::default()
        });
        config.optimizer.mixed_config = Some(MixedModeConfig {
            crossover_freq: 500.0,
            fir_band: "low".into(),
            ..MixedModeConfig::default()
        });
        config.optimizer.excursion_protection = Some(ExcursionProtectionConfig {
            enabled: true,
            auto_detect_f3: false,
            manual_f3_hz: Some(60.0),
            ..ExcursionProtectionConfig::default()
        });

        let execution =
            prepare_channel_execution("left", &prepared, &config, SAMPLE_RATE, None).unwrap();
        assert!(
            !execution.preprocessed.excursion_filters.is_empty(),
            "fixture must generate an excursion-protection filter"
        );
        let result = execute_prepared_channel(
            "left",
            &prepared,
            &config,
            SAMPLE_RATE,
            &execution,
            prepared.eq_resources(),
            Some(ConvolutionSidecarReference::new(IR_FILE).unwrap()),
            None,
        )
        .unwrap();
        let taps = result.fir_coeffs.clone().expect("Hybrid emits FIR taps");
        let serialized_chain: roomeq_model::ChannelDspChain = serde_json::from_slice(
            &serde_json::to_vec(&result.channel).expect("serialize public result chain"),
        )
        .expect("deserialize public result chain");
        let protection_index = serialized_chain
            .plugins
            .iter()
            .position(|plugin| {
                plugin.plugin_type == "eq" && plugin.parameters["label"] == "excursion_protection"
            })
            .expect("serialized Hybrid chain includes excursion protection");
        let crossover_index = serialized_chain
            .plugins
            .iter()
            .position(|plugin| plugin.plugin_type == "band_split")
            .expect("serialized Hybrid chain includes its crossover");
        assert!(
            protection_index < crossover_index,
            "excursion protection must precede the crossover and correction branches"
        );

        struct InlineIr(Vec<f64>);
        impl crate::dsp_realization::ConvolutionIrProvider for InlineIr {
            fn taps(&mut self, ir_file: &str, sample_rate: u32) -> autoeq_core::Result<&[f64]> {
                assert_eq!(ir_file, IR_FILE);
                assert_eq!(sample_rate, SAMPLE_RATE as u32);
                Ok(&self.0)
            }
        }

        let frequencies = ndarray::array![20.0, 30.0, 80.0, 300.0, 1_000.0];
        let mut protected_ir = InlineIr(taps.clone());
        let protected_response = crate::dsp_realization::RealizedDsp::new(
            &serialized_chain,
            SAMPLE_RATE,
            &mut protected_ir,
        )
        .unwrap()
        .complex_response(&frequencies)
        .unwrap();

        let mut unprotected_chain = serialized_chain.clone();
        unprotected_chain
            .plugins
            .retain(|plugin| plugin.parameters["label"] != "excursion_protection");
        let mut unprotected_ir = InlineIr(taps);
        let unprotected_response = crate::dsp_realization::RealizedDsp::new(
            &unprotected_chain,
            SAMPLE_RATE,
            &mut unprotected_ir,
        )
        .unwrap()
        .complex_response(&frequencies)
        .unwrap();
        let expected_protection = autoeq_core::response::compute_peq_complex_response(
            &execution.preprocessed.excursion_filters,
            &frequencies,
            SAMPLE_RATE,
        );
        for ((protected, unprotected), expected) in protected_response
            .iter()
            .zip(&unprotected_response)
            .zip(&expected_protection)
        {
            let ratio = protected / unprotected;
            assert!(
                (ratio - expected).norm() < 1e-8,
                "serialized full-chain response must include the exact pre-correction filter once"
            );
        }
        assert!(
            (protected_response[0] / unprotected_response[0]).norm() < 0.25,
            "the serialized protection must attenuate the measured 20 Hz transfer"
        );

        let mut replay_ir = InlineIr(result.fir_coeffs.clone().unwrap());
        let replayed_curve = crate::dsp_realization::RealizedDsp::new(
            &serialized_chain,
            SAMPLE_RATE,
            &mut replay_ir,
        )
        .unwrap()
        .apply_to_curve(&raw_curve)
        .unwrap();
        assert_eq!(replayed_curve.freq, result.raw_post_eq_curve.freq);
        for (replayed, reported) in replayed_curve.spl.iter().zip(&result.raw_post_eq_curve.spl) {
            assert!(
                (replayed - reported).abs() < 1e-7,
                "Hybrid's reported output must replay from the same raw input and serialized chain"
            );
        }
        assert!(replayed_curve.spl.iter().all(|level| level.is_finite()));
        assert!(
            replayed_curve
                .phase
                .as_ref()
                .unwrap()
                .iter()
                .all(|phase| phase.is_finite())
        );
    }

    #[test]
    fn sub_upper_bound_is_the_tighter_of_measurement_and_crossover() {
        assert_eq!(sub_optimizer_upper_bound(Some(300.0), Some(160.0)), 160.0);
        assert_eq!(sub_optimizer_upper_bound(Some(90.0), Some(200.0)), 90.0);
        assert_eq!(sub_optimizer_upper_bound(Some(120.0), None), 120.0);
        assert_eq!(sub_optimizer_upper_bound(None, Some(180.0)), 180.0);
    }
}
