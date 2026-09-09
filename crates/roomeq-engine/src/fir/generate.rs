use super::apply::{gd_delay_padding_samples, realize_gd_fir_delay};
use crate::Curve;
use crate::eq::{EqResources, PreparedEqTarget};
pub use autoeq_fir::FirPhase;
use ndarray::Array1;
use roomeq_model::OptimizerConfig;
use std::error::Error;
use std::sync::Once;

static EXCESS_PHASE_IDENTITY_WARNING: Once = Once::new();

/// Fit the realized finite FIR, not just the pre-window inversion spectrum.
/// Windowing a band-limited excess-phase inverse can attenuate its bass even
/// when the requested magnitude was correct. Retain only improving candidates.
pub fn generate_group_residual_fir_prepared(
    measurement: &Curve,
    config: &OptimizerConfig,
    target: &Curve,
    sample_rate: f64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut best = generate_fir_correction_prepared(measurement, config, target, sample_rate)?;
    let Some(fir) = config.fir.as_ref().filter(|fir|
        fir.phase.eq_ignore_ascii_case("kirkeby") && fir.correct_excess_phase)
    else { return Ok(best); };
    let evaluate = |coeffs: &[f64]| {
        let response = crate::response::compute_fir_complex_response(coeffs, &measurement.freq, sample_rate);
        let corrected = crate::response::apply_complex_response(measurement, &response);
        let score = crate::group::target_error_score(&corrected, target, config.min_freq, config.max_freq);
        (score, corrected)
    };
    let (mut best_score, mut realized) = evaluate(&best);
    let initial_score = best_score;
    let mut design_target = target.clone();
    for _ in 0..6 {
        if best_score < 0.05 { break; }
        for i in 0..design_target.freq.len() {
            if (config.min_freq..=config.max_freq).contains(&design_target.freq[i]) {
                let error = target.spl[i] - realized.spl[i];
                design_target.spl[i] = (design_target.spl[i] + 0.8 * error)
                    .clamp(target.spl[i] - 12.0, target.spl[i] + 12.0);
            }
        }
        let candidate = generate_fir_correction_prepared(measurement, config, &design_target, sample_rate)?;
        let (score, candidate_curve) = evaluate(&candidate);
        // Keep the existing Kirkeby boost limit (or a stricter explicit cap).
        let boost_limit = fir.max_boost_db.unwrap_or(15.0);
        let boost_ok = candidate_curve.spl.iter().zip(measurement.spl.iter())
            .all(|(&after, &before)| after.is_finite() && after - before <= boost_limit + 0.1);
        if !score.is_finite() || !boost_ok || score >= best_score { break; }
        best_score = score;
        best = candidate;
        realized = candidate_curve;
    }
    log::info!("Realized residual FIR target RMS: {initial_score:.3} -> {best_score:.3} dB ({} taps)", best.len());
    Ok(best)
}

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

fn correction_rms_db(measurement: &Curve, target: &Curve, min_freq: f64, max_freq: f64) -> f64 {
    let (sum_squares, count) = measurement
        .freq
        .iter()
        .zip(measurement.spl.iter())
        .zip(target.spl.iter())
        .filter(|((frequency, _), _)| **frequency >= min_freq && **frequency <= max_freq)
        .fold((0.0, 0_usize), |(sum, count), ((_, measured), desired)| {
            (sum + (desired - measured).powi(2), count + 1)
        });
    if count == 0 {
        0.0
    } else {
        (sum_squares / count as f64).sqrt()
    }
}

fn is_effectively_identity_fir(coefficients: &[f64]) -> bool {
    let Some((peak_index, peak)) = coefficients
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
    else {
        return false;
    };
    let off_peak = coefficients
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != peak_index)
        .map(|(_, coefficient)| coefficient.abs())
        .reduce(f64::max)
        .unwrap_or(0.0);
    (peak.abs() - 1.0).abs() <= 1.0e-5 && off_peak <= 1.0e-8
}

/// Retry an identity-like excess-phase result only for a nontrivial target.
pub(super) fn recover_excess_phase_identity(
    coefficients: Vec<f64>,
    correct_excess_phase: bool,
    correction_rms: f64,
    magnitude_only: impl FnOnce() -> Vec<f64>,
) -> Vec<f64> {
    if correct_excess_phase && correction_rms > 0.1 && is_effectively_identity_fir(&coefficients) {
        EXCESS_PHASE_IDENTITY_WARNING.call_once(|| {
            log::warn!(
                "One or more excess-phase FIR designs collapsed to identity despite a non-trivial magnitude correction; retrying magnitude-only Kirkeby inversion"
            );
        });
        magnitude_only()
    } else {
        coefficients
    }
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

    // Prepared targets must already share the measurement grid. In particular,
    // boost capping below is pointwise, not an interpolation operation.
    if target_curve.freq != measurement.freq
        || target_curve.spl.len() != measurement.freq.len()
        || measurement.spl.len() != measurement.freq.len()
    {
        return Err("Prepared FIR target and levels must match the measurement frequency grid".into());
    }

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
        let pre_ringing =
            fir_config
                .pre_ringing
                .as_ref()
                .map(|pr| math_audio_iir_fir::PreRingingConfig {
                    threshold_db: pr.threshold_db,
                    max_time_s: pr.max_time_s,
                });
        let coeffs = autoeq_fir::generate_kirkeby_correction_with_smoothing_and_pre_ringing(
            measurement,
            target_curve,
            sample_rate,
            n_taps,
            config.min_freq,
            config.max_freq,
            fir_config.correct_excess_phase,
            fir_config.phase_smoothing,
            pre_ringing,
        );
        let correction_rms =
            correction_rms_db(measurement, target_curve, config.min_freq, config.max_freq);
        let coeffs = recover_excess_phase_identity(
            coeffs,
            fir_config.correct_excess_phase,
            correction_rms,
            || {
                autoeq_fir::generate_kirkeby_correction_with_smoothing_and_pre_ringing(
                    measurement,
                    target_curve,
                    sample_rate,
                    n_taps,
                    config.min_freq,
                    config.max_freq,
                    false,
                    fir_config.phase_smoothing,
                    fir_config.pre_ringing.as_ref().map(|pr| {
                        math_audio_iir_fir::PreRingingConfig {
                            threshold_db: pr.threshold_db,
                            max_time_s: pr.max_time_s,
                        }
                    }),
                )
            },
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
/// All channels receive the common causal padding returned by
/// `gd_delay_padding_samples(&target.per_channel_delay_ms, sample_rate)`.
/// Consumers must include that padding in their absolute latency report.
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
    if let Some(target) = gd_target {
        let delay_ms = target
            .per_channel_delay_ms
            .get(channel_index)
            .copied()
            .ok_or("missing channel in FIR group-delay target")?;
        let padding = gd_delay_padding_samples(&target.per_channel_delay_ms, sample_rate);
        coeffs = realize_gd_fir_delay(&coeffs, delay_ms, sample_rate, padding)?.coefficients;
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
