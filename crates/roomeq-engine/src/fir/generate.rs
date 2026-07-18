use super::apply::apply_fractional_sample_shift;
use crate::Curve;
pub use autoeq_fir::FirPhase;
use roomeq_model::OptimizerConfig;
use std::error::Error;

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

        let fir_design_config = math_audio_iir_fir::FirDesignConfig {
            n_taps,
            sample_rate,
            phase: phase_type,
            pre_ringing,
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
    Ok(coeffs)
}
