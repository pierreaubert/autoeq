use super::fft::fft_real_to_half_spectrum;
use super::misc::read_wav_channels_f64;
use super::misc::two_channel_ir_spectrum;
use super::types::MatrixSpectrum;
use super::types::build_matrix_spectrum;
use math_audio_dsp::{
    align_ir_to_reference_peak, deconvolve_sweep_to_ir, direct_peak_sample,
    suppress_log_sweep_harmonic_residues,
};
use num_complex::Complex64;
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_model::{CtcConfig, CtcHrtfConfig, CtcMeasurementConfig, CtcWindowConfig};
use sofa_reader::{Hdf5File, SofaFile, SourcePosition};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;

/// SOFA dataset carrying per-measurement, per-receiver delays separately from
/// `Data.IR`. Units are samples per AES69 (the same index units as `Data.IR`).
const SOFA_DELAY_DATASET: &str = "Data.Delay";

/// Apply a (possibly fractional) sample delay to a half spectrum.
///
/// This is the exact frequency-domain image of shifting the IR: unity
/// magnitude at every bin, linear phase. Applying SOFA `Data.Delay` this way
/// keeps the FFT length fixed, so timing stored inline in `Data.IR` and timing
/// split into `Data.Delay` produce identical transfer matrices.
pub(super) fn apply_sample_delay_to_half_spectrum(
    spectrum: &mut [Complex64],
    delay_samples: f64,
    fft_size: usize,
) {
    if !delay_samples.is_finite() || delay_samples.abs() < 1e-12 || fft_size == 0 {
        return;
    }
    for (bin, value) in spectrum.iter_mut().enumerate() {
        let phase = -2.0 * PI * bin as f64 * delay_samples / fft_size as f64;
        *value *= Complex64::from_polar(1.0, phase);
    }
}

/// Read per-receiver delays (samples) for one SOFA measurement.
///
/// Returns `None` when the file carries no `Data.Delay` dataset (all timing
/// is embedded in `Data.IR`). Fails closed when the dataset exists but cannot
/// be interpreted, instead of silently dropping acoustic timing.
pub(super) fn read_sofa_receiver_delays(
    hdf5: &Hdf5File,
    num_measurements: usize,
    num_receivers: usize,
    measurement: usize,
) -> Result<Option<[f64; 2]>> {
    if !hdf5.has_dataset(SOFA_DELAY_DATASET) {
        return Ok(None);
    }
    let dims = hdf5
        .dataset_dims(SOFA_DELAY_DATASET)
        .map_err(|error| AutoeqError::InvalidMeasurement {
            message: format!("failed to read SOFA {SOFA_DELAY_DATASET} dimensions: {error}"),
        })?;
    let values = hdf5
        .read_f64(SOFA_DELAY_DATASET)
        .map_err(|error| AutoeqError::InvalidMeasurement {
            message: format!("failed to read SOFA {SOFA_DELAY_DATASET}: {error}"),
        })?;
    select_receiver_delays(
        &values,
        &dims,
        num_measurements,
        num_receivers,
        measurement,
    )
}

/// Validate `Data.Delay` shape (`[M, R]`) and select one measurement's
/// per-receiver delays. Pure function over the flattened row-major dataset so
/// the shape/index contract is unit-testable without a SOFA file.
pub(super) fn select_receiver_delays(
    values: &[f64],
    dims: &[u64],
    num_measurements: usize,
    num_receivers: usize,
    measurement: usize,
) -> Result<Option<[f64; 2]>> {
    if dims.len() != 2
        || dims[0] as usize != num_measurements
        || dims[1] as usize != num_receivers
        || values.len() != num_measurements * num_receivers
    {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!(
                "SOFA {SOFA_DELAY_DATASET} has unsupported shape {dims:?} for \
                 {num_measurements} measurements x {num_receivers} receivers; \
                 refusing to silently drop acoustic timing"
            ),
        });
    }
    if num_receivers != 2 {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!(
                "SOFA {SOFA_DELAY_DATASET} expects exactly two receivers, got {num_receivers}"
            ),
        });
    }
    let Some(row) = values.chunks_exact(num_receivers).nth(measurement) else {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!(
                "SOFA {SOFA_DELAY_DATASET} has no entry for measurement {measurement}"
            ),
        });
    };
    let delays = [row[0], row[1]];
    if delays.iter().any(|delay| !delay.is_finite()) {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!(
                "SOFA {SOFA_DELAY_DATASET} entry for measurement {measurement} is non-finite"
            ),
        });
    }
    Ok(Some(delays))
}

pub(super) fn load_measured_spectrum(
    measurements: &CtcMeasurementConfig,
    window: &CtcWindowConfig,
    sample_rate: u32,
    fft_size: usize,
) -> Result<MatrixSpectrum> {
    if window.window_type != "ctc_direct" && window.window_type != "fdw" {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "unsupported ctc.window.window_type '{}'; expected 'ctc_direct' or 'fdw'",
                window.window_type
            ),
        });
    }
    let speakers = measurements.speakers.clone();
    let ears = if measurements.mics.is_empty() {
        vec!["left_ear".to_string(), "right_ear".to_string()]
    } else {
        measurements.mics.clone()
    };
    if ears.len() != 2 {
        return Err(AutoeqError::InvalidConfiguration {
            message: "ctc.measurements.mics must contain exactly two ears".to_string(),
        });
    }
    let positions: Vec<String> = if measurements.head_positions.is_empty() {
        vec!["primary".to_string()]
    } else {
        measurements
            .head_positions
            .iter()
            .map(|position| position.id.clone())
            .collect()
    };

    let mut file_map: HashMap<(String, String), Option<std::path::PathBuf>> = HashMap::new();
    for file in &measurements.files {
        file_map.insert(
            (file.head_position.clone(), file.speaker.clone()),
            file.ir.clone(),
        );
    }

    let mut spectra_by_position = Vec::new();
    for position in &positions {
        let mut speaker_spectra = Vec::new();
        for speaker in &speakers {
            let path = file_map
                .get(&(position.clone(), speaker.clone()))
                .ok_or_else(|| AutoeqError::InvalidConfiguration {
                    message: format!(
                        "missing CTC IR file for head_position='{}', speaker='{}'",
                        position, speaker
                    ),
                })?;
            let path = path.as_ref().ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: format!(
                    "ctc.matrix_source='measured' requires ir for head_position='{}', speaker='{}'",
                    position, speaker
                ),
            })?;
            speaker_spectra.push(load_two_channel_ir_spectrum(
                path,
                window,
                sample_rate,
                fft_size,
            )?);
        }
        spectra_by_position.push(speaker_spectra);
    }

    Ok(build_matrix_spectrum(
        "measured".to_string(),
        speakers,
        ears,
        positions,
        spectra_by_position,
        fft_size / 2 + 1,
    ))
}

pub(super) fn load_raw_sweep_spectrum(
    measurements: &CtcMeasurementConfig,
    config: &CtcConfig,
    sample_rate: u32,
    fft_size: usize,
) -> Result<MatrixSpectrum> {
    if config.window.window_type != "ctc_direct" && config.window.window_type != "fdw" {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "unsupported ctc.window.window_type '{}'; expected 'ctc_direct' or 'fdw'",
                config.window.window_type
            ),
        });
    }
    let reference_sweep =
        config
            .reference_sweep
            .as_ref()
            .ok_or(AutoeqError::InvalidConfiguration {
                message: "ctc.matrix_source='raw_sweep' requires ctc.reference_sweep".to_string(),
            })?;
    let reference_channels =
        read_wav_channels_f64(reference_sweep, sample_rate, "CTC reference sweep")?;
    let reference = reference_channels
        .first()
        .ok_or_else(|| AutoeqError::InvalidMeasurement {
            message: format!(
                "CTC reference sweep '{}' has no channels",
                reference_sweep.display()
            ),
        })?;

    let speakers = measurements.speakers.clone();
    let ears = if measurements.mics.is_empty() {
        vec!["left_ear".to_string(), "right_ear".to_string()]
    } else {
        measurements.mics.clone()
    };
    if ears.len() != 2 {
        return Err(AutoeqError::InvalidConfiguration {
            message: "ctc.measurements.mics must contain exactly two ears".to_string(),
        });
    }
    let positions: Vec<String> = if measurements.head_positions.is_empty() {
        vec!["primary".to_string()]
    } else {
        measurements
            .head_positions
            .iter()
            .map(|position| position.id.clone())
            .collect()
    };

    let mut file_map = HashMap::new();
    for file in &measurements.files {
        file_map.insert(
            (file.head_position.clone(), file.speaker.clone()),
            (file.raw_sweep.clone(), file.loopback.clone()),
        );
    }

    let mut spectra_by_position = Vec::new();
    for position in &positions {
        let mut speaker_spectra = Vec::new();
        for speaker in &speakers {
            let (raw_sweep, loopback) = file_map
                .get(&(position.clone(), speaker.clone()))
                .ok_or_else(|| AutoeqError::InvalidConfiguration {
                    message: format!(
                        "missing CTC raw sweep file for head_position='{}', speaker='{}'",
                        position, speaker
                    ),
                })?;
            let raw_sweep = raw_sweep.as_ref().ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: format!(
                    "ctc.matrix_source='raw_sweep' requires raw_sweep for head_position='{}', speaker='{}'",
                    position, speaker
                ),
            })?;
            let loopback = loopback.as_ref().ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: format!(
                    "ctc.matrix_source='raw_sweep' requires loopback for head_position='{}', speaker='{}'",
                    position, speaker
                ),
            })?;
            speaker_spectra.push(load_two_channel_raw_sweep_spectrum(
                raw_sweep,
                loopback,
                reference,
                config,
                sample_rate,
                fft_size,
            )?);
        }
        spectra_by_position.push(speaker_spectra);
    }

    Ok(build_matrix_spectrum(
        "raw_sweep".to_string(),
        speakers,
        ears,
        positions,
        spectra_by_position,
        fft_size / 2 + 1,
    ))
}

pub(super) fn load_hrtf_spectrum(
    hrtf: &CtcHrtfConfig,
    sample_rate: u32,
    fft_size: usize,
) -> Result<MatrixSpectrum> {
    let sofa =
        SofaFile::load(&hrtf.hrtf_file).map_err(|message| AutoeqError::InvalidMeasurement {
            message: format!(
                "failed to load CTC HRTF '{}': {}",
                hrtf.hrtf_file.display(),
                message
            ),
        })?;
    if let Some(sofa_sr) = sofa.data_sample_rate
        && (sofa_sr - sample_rate as f32).abs() > 1.0
    {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "SOFA sample rate {} Hz differs from roomEQ sample rate {} Hz",
                sofa_sr, sample_rate
            ),
        });
    }

    // `SofaFile` exposes only `Data.IR`; separately stored `Data.Delay`
    // entries (F10) are read through the same file's HDF5 layer and applied
    // as exact spectral phase ramps below.
    let hdf5 = Hdf5File::open(&hrtf.hrtf_file).map_err(|error| {
        AutoeqError::InvalidMeasurement {
            message: format!(
                "failed to open CTC HRTF '{}' delay layer: {}",
                hrtf.hrtf_file.display(),
                error
            ),
        }
    })?;
    let mut speaker_spectra = Vec::new();
    let mut speakers = Vec::new();
    for speaker in &hrtf.speakers {
        let position = SourcePosition::new(
            speaker.azimuth_deg as f32,
            speaker.elevation_deg as f32,
            speaker.distance_m as f32,
        );
        // Same nearest-measurement lookup the reader uses for the IRs, so the
        // delays index the identical measurement.
        let (measurement, _) = sofa.find_nearest(&position);
        let data = sofa.get_hrtf_at_position(&position).ok_or_else(|| {
            AutoeqError::InvalidMeasurement {
                message: format!(
                    "no HRTF measurement found for speaker '{}'",
                    speaker.speaker
                ),
            }
        })?;
        speakers.push(speaker.speaker.clone());
        let mut spectra = [
            fft_real_to_half_spectrum(&data.ir_left, fft_size),
            fft_real_to_half_spectrum(&data.ir_right, fft_size),
        ];
        if let Some(delays) =
            read_sofa_receiver_delays(&hdf5, sofa.num_measurements, 2, measurement)?
        {
            if delays.iter().any(|delay| delay.abs() > 1e-12) {
                log::info!(
                    "CTC HRTF '{}' speaker '{}': applying SOFA Data.Delay [{:.3}, {:.3}] samples from measurement {measurement}",
                    hrtf.hrtf_file.display(),
                    speaker.speaker,
                    delays[0],
                    delays[1],
                );
            }
            apply_sample_delay_to_half_spectrum(&mut spectra[0], delays[0], fft_size);
            apply_sample_delay_to_half_spectrum(&mut spectra[1], delays[1], fft_size);
        }
        speaker_spectra.push(spectra);
    }

    Ok(build_matrix_spectrum(
        "hrtf_database".to_string(),
        speakers,
        vec!["left_ear".to_string(), "right_ear".to_string()],
        vec!["primary".to_string()],
        vec![speaker_spectra],
        fft_size / 2 + 1,
    ))
}

pub(super) fn load_two_channel_ir_spectrum(
    path: &Path,
    window: &CtcWindowConfig,
    sample_rate: u32,
    fft_size: usize,
) -> Result<[Vec<Complex64>; 2]> {
    let channels = read_wav_channels_f64(path, sample_rate, "CTC IR WAV")?;
    if channels.len() != 2 {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!(
                "CTC IR WAV '{}' must have exactly two channels, got {}",
                path.display(),
                channels.len()
            ),
        });
    }
    two_channel_ir_spectrum(&channels[0], &channels[1], window, sample_rate, fft_size)
}

pub(super) fn load_two_channel_raw_sweep_spectrum(
    raw_sweep: &Path,
    loopback: &Path,
    reference: &[f64],
    config: &CtcConfig,
    sample_rate: u32,
    fft_size: usize,
) -> Result<[Vec<Complex64>; 2]> {
    let raw_channels = read_wav_channels_f64(raw_sweep, sample_rate, "CTC raw sweep WAV")?;
    if raw_channels.len() != 2 {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!(
                "CTC raw sweep WAV '{}' must have exactly two channels, got {}",
                raw_sweep.display(),
                raw_channels.len()
            ),
        });
    }
    let loopback_channels = read_wav_channels_f64(loopback, sample_rate, "CTC loopback WAV")?;
    let loopback_signal =
        loopback_channels
            .first()
            .ok_or_else(|| AutoeqError::InvalidMeasurement {
                message: format!("CTC loopback WAV '{}' has no channels", loopback.display()),
            })?;
    let deconvolve_fft_size = raw_channels[0]
        .len()
        .max(raw_channels[1].len())
        .max(loopback_signal.len())
        .max(reference.len())
        .next_power_of_two()
        .max(fft_size);
    let loopback_ir = deconvolve_sweep_to_ir(loopback_signal, reference, deconvolve_fft_size)
        .map_err(|message| AutoeqError::InvalidMeasurement {
            message: format!(
                "failed deconvolving CTC loopback '{}': {}",
                loopback.display(),
                message
            ),
        })?;
    let alignment_peak = direct_peak_sample(&loopback_ir);
    let mut ear_irs = [Vec::new(), Vec::new()];
    for ear_idx in 0..2 {
        let ir = deconvolve_sweep_to_ir(&raw_channels[ear_idx], reference, deconvolve_fft_size)
            .map_err(|message| AutoeqError::InvalidMeasurement {
                message: format!(
                    "failed deconvolving CTC raw sweep '{}' channel {}: {}",
                    raw_sweep.display(),
                    ear_idx + 1,
                    message
                ),
            })?;
        let mut aligned = align_ir_to_reference_peak(&ir, alignment_peak);
        if let (Some(duration), Some(start_hz), Some(end_hz)) = (
            config.sweep_duration_s,
            config.sweep_start_hz,
            config.sweep_end_hz,
        ) {
            suppress_log_sweep_harmonic_residues(
                &mut aligned,
                sample_rate as f64,
                duration,
                start_hz,
                end_hz,
                config.harmonic_suppression_harmonics,
                config.harmonic_suppression_window_ms,
            );
        }
        ear_irs[ear_idx] = aligned;
    }
    two_channel_ir_spectrum(
        &ear_irs[0],
        &ear_irs[1],
        &config.window,
        sample_rate,
        fft_size,
    )
}
