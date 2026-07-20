use hound::{SampleFormat, WavReader};
use math_audio_dsp::{
    biquad_complex_response as dsp_biquad_complex_response, direct_peak_sample,
    direct_peak_windowed_half_spectrum, fdw_complex_half_spectrum,
};
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use num_complex::Complex64;
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_model::CtcWindowConfig;
use std::path::Path;

pub(super) const CTC_ARTIFACT_VERSION: &str = "ctc-recommended-v1";
#[cfg(test)]
pub(super) use roomeq_engine::ctc::{
    CTC_CONDITION_WARNING_THRESHOLD, amplitude_to_db, beta_for_frequency, ctc_condition_warning,
    enforce_electrical_sum_headroom, reconstruction_error_to_db,
};

pub(super) fn invalid_ctc_configuration(message: impl Into<String>) -> AutoeqError {
    let message = message.into();
    log::error!("  CTC configuration invalid: {}", message);
    AutoeqError::InvalidConfiguration { message }
}

pub(super) fn biquad_filter_response(
    filter: &serde_json::Value,
    freq: f64,
    sample_rate: f64,
) -> Result<Complex64> {
    let filter_type = filter
        .get("filter_type")
        .and_then(|value| value.as_str())
        .and_then(parse_biquad_filter_type)
        .ok_or_else(|| AutoeqError::InvalidConfiguration {
            message: format!(
                "unsupported RoomEQ biquad filter type in CTC joint path: {}",
                filter
            ),
        })?;
    let freq_hz = filter
        .get("freq")
        .and_then(|value| value.as_f64())
        .unwrap_or(1000.0);
    let q = filter
        .get("q")
        .and_then(|value| value.as_f64())
        .unwrap_or(1.0);
    let db_gain = filter
        .get("db_gain")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    Ok(dsp_biquad_complex_response(
        &Biquad::new(filter_type, freq_hz, sample_rate, q, db_gain),
        freq,
    ))
}

pub(super) fn parse_biquad_filter_type(value: &str) -> Option<BiquadFilterType> {
    match value {
        "lowpass" => Some(BiquadFilterType::Lowpass),
        "highpass" => Some(BiquadFilterType::Highpass),
        "highpassvariableq" => Some(BiquadFilterType::HighpassVariableQ),
        "bandpass" => Some(BiquadFilterType::Bandpass),
        "peak" => Some(BiquadFilterType::Peak),
        "notch" => Some(BiquadFilterType::Notch),
        "lowshelf" => Some(BiquadFilterType::Lowshelf),
        "highshelf" => Some(BiquadFilterType::Highshelf),
        "allpass" => Some(BiquadFilterType::AllPass),
        "lowshelforf" => Some(BiquadFilterType::LowshelfOrf),
        "highshelforf" => Some(BiquadFilterType::HighshelfOrf),
        "peakmatched" => Some(BiquadFilterType::PeakMatched),
        _ => None,
    }
}

pub(super) fn checked_sample_rate(sample_rate: f64) -> Result<u32> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 || sample_rate > u32::MAX as f64 {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!("invalid sample rate for CTC: {}", sample_rate),
        });
    }
    Ok(sample_rate.round() as u32)
}

pub(super) fn two_channel_ir_spectrum(
    left: &[f64],
    right: &[f64],
    window: &CtcWindowConfig,
    sample_rate: u32,
    fft_size: usize,
) -> Result<[Vec<Complex64>; 2]> {
    Ok([
        ir_to_half_spectrum(left, window, sample_rate, fft_size)?,
        ir_to_half_spectrum(right, window, sample_rate, fft_size)?,
    ])
}

pub(super) fn ir_to_half_spectrum(
    ir: &[f64],
    window: &CtcWindowConfig,
    sample_rate: u32,
    fft_size: usize,
) -> Result<Vec<Complex64>> {
    match window.window_type.as_str() {
        "ctc_direct" => direct_peak_windowed_half_spectrum(
            ir,
            sample_rate as f64,
            fft_size,
            window.start_ms,
            window.length_ms,
            window.fade_ms,
        )
        .map_err(|message| AutoeqError::InvalidMeasurement {
            message: format!("failed direct-windowing CTC IR: {}", message),
        }),
        "fdw" => fdw_complex_half_spectrum(
            ir,
            sample_rate as f64,
            fft_size,
            direct_peak_sample(ir),
            window.fdw_cycles,
            window.fdw_min_ms,
            window.fdw_max_ms,
        )
        .map_err(|message| AutoeqError::InvalidMeasurement {
            message: format!("failed FDW-windowing CTC IR: {}", message),
        }),
        other => Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "unsupported ctc.window.window_type '{}'; expected 'ctc_direct' or 'fdw'",
                other
            ),
        }),
    }
}

pub(super) fn read_wav_channels_f64(
    path: &Path,
    sample_rate: u32,
    label: &str,
) -> Result<Vec<Vec<f64>>> {
    let mut reader = WavReader::open(path).map_err(|err| AutoeqError::InvalidMeasurement {
        message: format!("failed to open {} '{}': {}", label, path.display(), err),
    })?;
    let spec = reader.spec();
    if spec.sample_rate != sample_rate {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!(
                "{} '{}' sample rate {} differs from roomEQ sample rate {}",
                label,
                path.display(),
                spec.sample_rate,
                sample_rate
            ),
        });
    }
    if spec.channels == 0 {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!("{} '{}' has no channels", label, path.display()),
        });
    }
    let mut channels = vec![Vec::new(); spec.channels as usize];
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => {
            for (idx, sample) in reader.samples::<f32>().enumerate() {
                let value = sample.map_err(|err| AutoeqError::InvalidMeasurement {
                    message: format!("failed reading '{}': {}", path.display(), err),
                })? as f64;
                channels[idx % spec.channels as usize].push(value);
            }
        }
        (SampleFormat::Int, bits) if bits <= 16 => {
            let scale = (1_i64 << (bits - 1)) as f64;
            for (idx, sample) in reader.samples::<i16>().enumerate() {
                let value = sample.map_err(|err| AutoeqError::InvalidMeasurement {
                    message: format!("failed reading '{}': {}", path.display(), err),
                })? as f64
                    / scale;
                channels[idx % spec.channels as usize].push(value);
            }
        }
        (SampleFormat::Int, bits) => {
            let scale = (1_i64 << (bits - 1)) as f64;
            for (idx, sample) in reader.samples::<i32>().enumerate() {
                let value = sample.map_err(|err| AutoeqError::InvalidMeasurement {
                    message: format!("failed reading '{}': {}", path.display(), err),
                })? as f64
                    / scale;
                channels[idx % spec.channels as usize].push(value);
            }
        }
        other => {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!(
                    "unsupported {} format {:?} in '{}'",
                    label,
                    other,
                    path.display()
                ),
            });
        }
    }
    Ok(channels)
}
