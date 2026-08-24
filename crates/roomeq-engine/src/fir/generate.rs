use super::apply::apply_fractional_sample_shift;
use crate::Curve;
use crate::eq::{EqResources, PreparedEqTarget};
pub use autoeq_fir::FirPhase;
use ndarray::Array1;
use roomeq_model::OptimizerConfig;
use std::error::Error;

/// Resolve a workflow-prepared FIR target on the measurement grid.
pub fn prepared_fir_target_curve(
    measurement: &Curve,
    config: &OptimizerConfig,
    resources: &EqResources,
) -> Curve {
    let mut target = match resources.target.as_ref() {
        Some(PreparedEqTarget::Curve(target)) => {
            autoeq_core::normalize_and_interpolate_response(&measurement.freq, target)
        }
        Some(PreparedEqTarget::Predefined(name)) => {
            autoeq_core::build_target_curve_by_name(name, &measurement.freq, measurement)
        }
        None => {
            let (sum, count) = measurement
                .freq
                .iter()
                .zip(measurement.spl.iter())
                .filter_map(|(&frequency, &level)| {
                    (frequency >= config.min_freq && frequency <= config.max_freq).then_some(level)
                })
                .fold((0.0, 0_usize), |(sum, count), level| {
                    (sum + level, count + 1)
                });
            let mean_level = if count == 0 { 0.0 } else { sum / count as f64 };
            Curve {
                freq: measurement.freq.clone(),
                spl: Array1::from_elem(measurement.freq.len(), mean_level),
                phase: None,
                ..Curve::default()
            }
        }
    };
    let measurement_mean = roomeq_analysis::response_metrics::mean_response_in_range(
        measurement,
        config.min_freq,
        config.max_freq,
    );
    let target_mean = roomeq_analysis::response_metrics::mean_response_in_range(
        &target,
        config.min_freq,
        config.max_freq,
    );
    if measurement_mean.is_finite() && target_mean.is_finite() {
        target.spl += measurement_mean - target_mean;
    }
    target
}

/// Generate FIR coefficients using only workflow-prepared in-memory resources.
pub fn generate_fir_correction_with_resources(
    measurement: &Curve,
    config: &OptimizerConfig,
    resources: &EqResources,
    sample_rate: f64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let target = prepared_fir_target_curve(measurement, config, resources);
    generate_fir_correction_prepared(measurement, config, &target, sample_rate)
}

/// Generate an FIR correction filter for a single channel
///
/// This is the main entry point for FIR-based room correction. It handles:
/// - Phase type selection (linear, minimum, or kirkeby)
/// - FIR coefficient generation
///
/// # Arguments
/// * `measurement` - The room measurement curve
/// * `config` - Optimizer configuration (contains FIR settings)
/// * `target_curve` - Prepared target curve on the measurement frequency grid
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// * Vector of FIR coefficients on success
pub fn generate_fir_correction_prepared(
    measurement: &Curve,
    config: &OptimizerConfig,
    target_curve: &Curve,
    sample_rate: f64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let fir_config = config.fir.as_ref().ok_or("FIR configuration missing")?;
    let n_taps = fir_config.taps;

    // Optional boost cap: clamp the target-vs-measurement delta to at most
    // `max_boost_db` of positive correction per frequency before designing
    // the filter, so the FIR cannot chase deep nulls past the runtime
    // acceptance policy's boost guard.
    let capped_target;
    let target_curve = if let Some(max_boost_db) = fir_config.max_boost_db {
        let mut capped = target_curve.clone();
        capped.spl = ndarray::Array1::from_iter(
            target_curve
                .spl
                .iter()
                .zip(measurement.spl.iter())
                .map(|(&target, &measured)| measured + (target - measured).min(max_boost_db)),
        );
        capped_target = capped;
        &capped_target
    } else {
        target_curve
    };

    if fir_config.phase.to_lowercase() == "kirkeby" {
        // Use Kirkeby regularized inversion with optional excess phase correction
        let coeffs = autoeq_fir::generate_kirkeby_correction_with_smoothing(
            measurement,
            target_curve,
            sample_rate,
            n_taps,
            config.min_freq,
            config.max_freq,
            fir_config.correct_excess_phase,
            fir_config.phase_smoothing,
        );
        Ok(coeffs)
    } else {
        // Standard magnitude-based generation
        let correction_spl = &target_curve.spl - &measurement.spl;
        let correction_curve = Curve {
            freq: measurement.freq.clone(),
            spl: correction_spl,
            phase: None,
            ..Default::default()
        };

        let phase_type = match fir_config.phase.to_lowercase().as_str() {
            "linear" => FirPhase::Linear,
            "minimum" => FirPhase::Minimum,
            _ => return Err(format!("Unknown FIR phase type: {}", fir_config.phase).into()),
        };

        // Convert pre-ringing config if present
        let pre_ringing =
            fir_config
                .pre_ringing
                .as_ref()
                .map(|pr| math_audio_iir_fir::PreRingingConfig {
                    threshold_db: pr.threshold_db,
                    max_time_s: pr.max_time_s,
                });

        // A causal minimum-phase impulse begins at tap zero. Symmetric windows
        // (the default Blackman) are zero at tap zero and would erase the
        // leading energy, destroying the minimum-phase result. Truncation is
        // sufficient for that phase type.
        let fir_design_config = math_audio_iir_fir::FirDesignConfig {
            n_taps,
            sample_rate,
            phase: phase_type,
            pre_ringing,
            window: if phase_type == FirPhase::Minimum {
                math_audio_iir_fir::WindowType::Rectangular
            } else {
                math_audio_iir_fir::FirDesignConfig::default().window
            },
            ..Default::default()
        };

        let freqs: Vec<f64> = correction_curve.freq.to_vec();
        let magnitude_db: Vec<f64> = correction_curve.spl.to_vec();
        let coeffs = math_audio_iir_fir::generate_fir_from_response(
            &freqs,
            &magnitude_db,
            &fir_design_config,
        );
        Ok(coeffs)
    }
}

/// Generate a correction FIR and apply an optional group-delay alignment
/// target to the selected channel.
pub fn generate_fir_correction_with_gd_target_prepared(
    measurement: &Curve,
    config: &OptimizerConfig,
    target_curve: &Curve,
    sample_rate: f64,
    gd_target: Option<&crate::gd_opt::GdAlignmentTarget>,
    channel_index: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut coeffs =
        generate_fir_correction_prepared(measurement, config, target_curve, sample_rate)?;
    if let Some(delay_ms) = gd_target
        .and_then(|target| target.per_channel_delay_ms.get(channel_index))
        .copied()
        .filter(|delay| delay.abs() > 1e-6)
    {
        coeffs = apply_fractional_sample_shift(&coeffs, delay_ms * 1e-3 * sample_rate);
    }
    if gd_target
        .and_then(|target| target.per_channel_polarity_inverted.get(channel_index))
        .copied()
        .unwrap_or(false)
    {
        for coefficient in &mut coeffs {
            *coefficient = -*coefficient;
        }
    }
    Ok(coeffs)
}
