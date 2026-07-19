//! Path-free cross-talk cancellation matrix solving and diagnostics.

use crate::error::{AutoeqError, Result};
use math_audio_dsp::{
    TransferMatrixBin, half_spectrum_to_fir, position_errors,
    solve_minimax_regularized_inverse_bin, solve_regularized_inverse_bin,
};
use num_complex::Complex64;
use roomeq_model::{
    CtcBinauralDiagnostics, CtcConfig, CtcDeliveredResponseMetrics, CtcHrtfCandidateComparison,
};
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

pub const CTC_CONDITION_WARNING_THRESHOLD: f64 = 1.0e6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedCtcFilter {
    pub speaker: String,
    pub target_ear: String,
    pub taps: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PreparedCtcMatrix {
    pub source: String,
    pub speakers: Vec<String>,
    pub ears: Vec<String>,
    pub positions: Vec<String>,
    pub bins: Vec<Vec<TransferMatrixBin>>,
}

#[derive(Debug, Clone)]
pub struct PreparedCtcSolution {
    pub filters: Vec<PreparedCtcFilter>,
    pub latency_samples: usize,
    pub latency_ms: f64,
    pub max_filter_gain_db: f64,
    pub max_condition_number: f64,
    pub mean_reconstruction_error: f64,
    pub worst_position_error: f64,
    pub mean_crosstalk_residual_db: f64,
    pub max_electrical_sum_gain_db: f64,
    pub driver_headroom_limited: bool,
    pub delivered_response: CtcDeliveredResponseMetrics,
    pub binaural_diagnostics: CtcBinauralDiagnostics,
}

pub fn solve_prepared_ctc(
    spectrum: &PreparedCtcMatrix,
    config: &CtcConfig,
    sample_rate: f64,
) -> Result<PreparedCtcSolution> {
    if config.fir_taps < 16 || !config.fir_taps.is_power_of_two() {
        return Err(invalid_ctc_configuration(
            "ctc.fir_taps must be a power of two >= 16",
        ));
    }
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(invalid_ctc_configuration(
            "ctc sample rate must be finite and positive",
        ));
    }
    if spectrum.speakers.len() < 2 || spectrum.ears.len() != 2 {
        return Err(invalid_ctc_configuration(
            "ctc requires at least two speakers and exactly two ears",
        ));
    }

    let fft_size = config.fir_taps;
    let num_bins = fft_size / 2 + 1;
    if spectrum.bins.len() != num_bins {
        return Err(invalid_ctc_configuration(format!(
            "prepared CTC matrix has {} bins; expected {num_bins}",
            spectrum.bins.len()
        )));
    }

    let target = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let mut solved_bins = Vec::with_capacity(num_bins);
    let mut max_condition = 0.0_f64;
    let mut total_error = 0.0_f64;
    let mut worst_position_error = 0.0_f64;
    let mut headroom_was_limited = false;

    for bin in 0..num_bins {
        let freq = bin as f64 * sample_rate / fft_size as f64;
        let beta = beta_for_frequency(config, freq);
        let solved = if config.robustness == "minimax" {
            solve_minimax_regularized_inverse_bin(
                &spectrum.bins[bin],
                &target,
                beta,
                Some(config.regularization.max_gain_db),
                config.minimax_iterations,
            )
        } else {
            solve_regularized_inverse_bin(
                &spectrum.bins[bin],
                &target,
                beta,
                Some(config.regularization.max_gain_db),
            )
        }
        .map_err(|message| AutoeqError::OptimizationFailed {
            message: format!("ctc inverse failed at bin {bin}: {message}"),
        })?;
        max_condition = max_condition.max(solved.condition_number);
        let mut values = solved.values;
        headroom_was_limited |= enforce_electrical_sum_headroom(
            &mut values,
            spectrum.speakers.len(),
            2,
            config.regularization.max_gain_db,
        );
        let errors = position_errors(&spectrum.bins[bin], &values, &target).map_err(|message| {
            AutoeqError::OptimizationFailed {
                message: format!("ctc reconstruction scoring failed at bin {bin}: {message}"),
            }
        })?;
        total_error += errors.iter().sum::<f64>() / errors.len().max(1) as f64;
        worst_position_error = worst_position_error.max(errors.iter().copied().fold(0.0, f64::max));
        solved_bins.push(values);
    }
    if let Some(message) = ctc_condition_warning(max_condition) {
        log::warn!("  {message}");
    }

    let latency_samples = fft_size / 2;
    let latency_ms = latency_samples as f64 * 1000.0 / sample_rate;
    let max_condition_number = if max_condition.is_finite() {
        max_condition
    } else {
        f64::MAX
    };
    let mean_reconstruction_error = total_error / num_bins as f64;
    let mut filters = Vec::new();
    let mut max_filter_gain_db = f64::NEG_INFINITY;
    let mut max_electrical_sum_gain_db = f64::NEG_INFINITY;

    for speaker_idx in 0..spectrum.speakers.len() {
        for ear_idx in 0..2 {
            let half_spectrum: Vec<Complex64> = solved_bins
                .iter()
                .map(|matrix| matrix[speaker_idx * 2 + ear_idx])
                .collect();
            let max_mag = half_spectrum
                .iter()
                .map(|value| value.norm())
                .fold(0.0, f64::max);
            if max_mag > 0.0 {
                max_filter_gain_db = max_filter_gain_db.max(20.0 * max_mag.log10());
            }
            let taps = half_spectrum_to_fir(&half_spectrum, fft_size, latency_samples as f64)
                .map_err(|message| AutoeqError::OptimizationFailed {
                    message: format!("ctc FIR synthesis failed: {message}"),
                })?;
            filters.push(PreparedCtcFilter {
                speaker: spectrum.speakers[speaker_idx].clone(),
                target_ear: spectrum.ears[ear_idx].clone(),
                taps,
            });
        }
        let max_sum_gain = solved_bins
            .iter()
            .map(|matrix| {
                let row_start = speaker_idx * 2;
                (matrix[row_start].norm_sqr() + matrix[row_start + 1].norm_sqr()).sqrt()
            })
            .fold(0.0, f64::max);
        if max_sum_gain > 0.0 {
            max_electrical_sum_gain_db =
                max_electrical_sum_gain_db.max(20.0 * max_sum_gain.log10());
        }
    }

    if !max_filter_gain_db.is_finite() {
        max_filter_gain_db = 0.0;
    }
    if !max_electrical_sum_gain_db.is_finite() {
        max_electrical_sum_gain_db = 0.0;
    }
    let mean_crosstalk_residual_db = reconstruction_error_to_db(mean_reconstruction_error);
    let driver_headroom_limited = headroom_was_limited
        || max_electrical_sum_gain_db >= config.regularization.max_gain_db - 0.25;
    let delivered_response =
        compute_delivered_response_metrics(spectrum, &filters, fft_size, latency_samples)?;
    let binaural_diagnostics = compute_binaural_diagnostics(
        spectrum,
        &delivered_response,
        max_condition_number,
        driver_headroom_limited,
    );

    Ok(PreparedCtcSolution {
        filters,
        latency_samples,
        latency_ms,
        max_filter_gain_db,
        max_condition_number,
        mean_reconstruction_error,
        worst_position_error,
        mean_crosstalk_residual_db,
        max_electrical_sum_gain_db,
        driver_headroom_limited,
        delivered_response,
        binaural_diagnostics,
    })
}

pub fn compute_binaural_diagnostics(
    spectrum: &PreparedCtcMatrix,
    delivered: &CtcDeliveredResponseMetrics,
    max_condition_number: f64,
    driver_headroom_limited: bool,
) -> CtcBinauralDiagnostics {
    let crosstalk_risk = delivered.worst_crosstalk_db > -12.0;
    let target_risk = delivered.worst_target_error > 1.0;
    let condition_risk = max_condition_number > CTC_CONDITION_WARNING_THRESHOLD;
    let externalization_risk = if driver_headroom_limited || condition_risk || target_risk {
        "high".to_string()
    } else if delivered.worst_crosstalk_db > -20.0 || delivered.mean_channel_balance_db > 2.0 {
        "moderate".to_string()
    } else {
        "low".to_string()
    };
    let imaging_risk = if crosstalk_risk || delivered.mean_channel_balance_db > 3.0 {
        "high".to_string()
    } else if delivered.mean_channel_balance_db > 1.5 {
        "moderate".to_string()
    } else {
        "low".to_string()
    };

    CtcBinauralDiagnostics {
        ild_error_db: delivered.mean_channel_balance_db,
        itd_error_proxy_us: None,
        cue_deviation_score: delivered.mean_target_error
            + 10.0_f64.powf(delivered.mean_crosstalk_db / 20.0)
            + delivered.mean_channel_balance_db / 20.0,
        externalization_risk,
        imaging_risk,
        hrtf_candidate_comparison: spectrum.source.contains("hrtf").then(|| {
            CtcHrtfCandidateComparison {
                candidate_count: spectrum.positions.len().max(1),
                selected_source: spectrum.source.clone(),
                advisory: if spectrum.positions.len() > 1 {
                    "robust_head_position_average".to_string()
                } else {
                    "single_hrtf_candidate".to_string()
                },
            }
        }),
    }
}

pub fn compute_delivered_response_metrics(
    spectrum: &PreparedCtcMatrix,
    filters: &[PreparedCtcFilter],
    fft_size: usize,
    latency_samples: usize,
) -> Result<CtcDeliveredResponseMetrics> {
    let num_bins = fft_size / 2 + 1;
    let speakers = spectrum.speakers.len();
    let mut filter_spectra = Vec::with_capacity(speakers * 2);
    for speaker in &spectrum.speakers {
        for ear in &spectrum.ears {
            let filter = filters
                .iter()
                .find(|filter| filter.speaker == *speaker && filter.target_ear == *ear)
                .ok_or_else(|| AutoeqError::OptimizationFailed {
                    message: format!(
                        "ctc delivered-response scoring missing filter speaker='{speaker}', target_ear='{ear}'"
                    ),
                })?;
            filter_spectra.push(fft_real_to_half_spectrum_f64(&filter.taps, fft_size));
        }
    }

    let mut target_error_sum_sq = 0.0_f64;
    let mut target_count = 0usize;
    let mut worst_target_error = 0.0_f64;
    let mut crosstalk_sum_sq = 0.0_f64;
    let mut crosstalk_count = 0usize;
    let mut worst_crosstalk = 0.0_f64;
    let mut balance_sum_db = 0.0_f64;
    let mut balance_count = 0usize;

    for (bin, positions) in spectrum.bins[..num_bins].iter().enumerate() {
        let latency_phase = 2.0 * PI * bin as f64 * latency_samples as f64 / fft_size as f64;
        let undo_latency = Complex64::from_polar(1.0, latency_phase);
        for position in positions {
            let mut delivered = [Complex64::new(0.0, 0.0); 4];
            for ear_idx in 0..2 {
                for target_ear_idx in 0..2 {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for speaker_idx in 0..speakers {
                        let h = position.values[ear_idx * speakers + speaker_idx];
                        let f = filter_spectra[speaker_idx * 2 + target_ear_idx][bin];
                        sum += h * f;
                    }
                    delivered[ear_idx * 2 + target_ear_idx] = sum * undo_latency;
                }
            }
            for ear_idx in 0..2 {
                let target = delivered[ear_idx * 2 + ear_idx];
                let error = (target - Complex64::new(1.0, 0.0)).norm();
                target_error_sum_sq += error * error;
                worst_target_error = worst_target_error.max(error);
                target_count += 1;
            }
            let left_mag = delivered[0].norm();
            let right_mag = delivered[3].norm();
            balance_sum_db += (amplitude_to_db(left_mag) - amplitude_to_db(right_mag)).abs();
            balance_count += 1;
            for (ear_idx, target_ear_idx) in [(0, 1), (1, 0)] {
                let crosstalk = delivered[ear_idx * 2 + target_ear_idx].norm();
                crosstalk_sum_sq += crosstalk * crosstalk;
                worst_crosstalk = worst_crosstalk.max(crosstalk);
                crosstalk_count += 1;
            }
        }
    }

    let mean_target_error = if target_count == 0 {
        0.0
    } else {
        (target_error_sum_sq / target_count as f64).sqrt()
    };
    let mean_crosstalk = if crosstalk_count == 0 {
        0.0
    } else {
        (crosstalk_sum_sq / crosstalk_count as f64).sqrt()
    };
    let mean_channel_balance_db = if balance_count == 0 {
        0.0
    } else {
        balance_sum_db / balance_count as f64
    };

    Ok(CtcDeliveredResponseMetrics {
        mean_target_error,
        worst_target_error,
        mean_crosstalk_db: amplitude_to_db(mean_crosstalk),
        worst_crosstalk_db: amplitude_to_db(worst_crosstalk),
        mean_channel_balance_db,
    })
}

pub fn build_matrix_spectrum(
    source: String,
    speakers: Vec<String>,
    ears: Vec<String>,
    positions: Vec<String>,
    spectra_by_position: Vec<Vec<[Vec<Complex64>; 2]>>,
    num_bins: usize,
) -> PreparedCtcMatrix {
    let mut bins = Vec::with_capacity(num_bins);
    for bin in 0..num_bins {
        let mut position_bins = Vec::with_capacity(positions.len());
        for speaker_spectra in &spectra_by_position {
            let mut values = vec![Complex64::new(0.0, 0.0); 2 * speakers.len()];
            for (speaker_idx, ear_spectra) in speaker_spectra.iter().enumerate() {
                values[speaker_idx] = ear_spectra[0][bin];
                values[speakers.len() + speaker_idx] = ear_spectra[1][bin];
            }
            position_bins.push(TransferMatrixBin::new(2, speakers.len(), values));
        }
        bins.push(position_bins);
    }
    PreparedCtcMatrix {
        source,
        speakers,
        ears,
        positions,
        bins,
    }
}

pub fn ctc_condition_warning(max_condition: f64) -> Option<String> {
    (max_condition.is_finite() && max_condition > CTC_CONDITION_WARNING_THRESHOLD).then(|| {
        format!(
            "CTC transfer matrix is ill-conditioned: max condition number {max_condition:.3e} exceeds {CTC_CONDITION_WARNING_THRESHOLD:.3e}; filters may amplify measurement noise or need stronger regularization"
        )
    })
}

pub fn enforce_electrical_sum_headroom(
    values: &mut [Complex64],
    speakers: usize,
    ears: usize,
    max_gain_db: f64,
) -> bool {
    let max_gain = 10.0_f64.powf(max_gain_db / 20.0);
    let mut limited = false;
    for speaker_idx in 0..speakers {
        let row_start = speaker_idx * ears;
        let row_end = row_start + ears;
        let row_norm = values[row_start..row_end]
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if row_norm > max_gain && row_norm > 0.0 {
            let scale = max_gain / row_norm;
            for value in &mut values[row_start..row_end] {
                *value *= scale;
            }
            limited = true;
        }
    }
    limited
}

pub fn beta_for_frequency(config: &CtcConfig, freq_hz: f64) -> f64 {
    let beta_db = if freq_hz < 150.0 {
        config.regularization.beta_lf_db
    } else if freq_hz > 6000.0 {
        config.regularization.beta_hf_db
    } else {
        config.regularization.beta_db
    };
    let robustness_scale = if config.robustness == "minimax" {
        2.0
    } else {
        1.0
    };
    10.0_f64.powf(beta_db / 20.0) * robustness_scale
}

pub fn reconstruction_error_to_db(error: f64) -> f64 {
    10.0 * error.max(1e-24).log10()
}

pub fn amplitude_to_db(value: f64) -> f64 {
    20.0 * value.max(1e-12).log10()
}

fn invalid_ctc_configuration(message: impl Into<String>) -> AutoeqError {
    AutoeqError::InvalidConfiguration {
        message: message.into(),
    }
}

pub fn fft_real_to_half_spectrum_f64(input: &[f64], fft_size: usize) -> Vec<Complex64> {
    let mut buffer = vec![Complex64::new(0.0, 0.0); fft_size];
    for (dst, value) in buffer.iter_mut().zip(input.iter().copied()) {
        dst.re = value;
    }
    FftPlanner::<f64>::new()
        .plan_fft_forward(fft_size)
        .process(&mut buffer);
    buffer.truncate(fft_size / 2 + 1);
    buffer
}
