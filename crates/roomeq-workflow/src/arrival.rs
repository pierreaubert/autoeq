//! Workflow preparation of path-free channel arrival metadata.

use std::path::Path;

use autoeq_measurements::MeasurementSource;
use log::debug;
use math_audio_dsp::signals::{gen_dirac, gen_mls};
use roomeq_engine::eq::{EqResources, PreparedImpulseResponse};
use roomeq_engine::{PreparedCea2034, PreparedChannelInput};
use roomeq_model::{RoomConfig, TargetCurveConfig, TargetShape};

use crate::wav::decode_first_channel;

const DEFAULT_MLS_ORDER: u8 = 16;

fn normalize_recording_signal_type(signal_type: &str) -> String {
    signal_type
        .trim()
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

fn matched_reference_from_recording_config(
    room_config: &RoomConfig,
    fallback_sample_rate: f64,
) -> Option<(&'static str, Vec<f32>, u32)> {
    let recording = room_config.recording_config.as_ref()?;
    let signal_type = normalize_recording_signal_type(recording.signal_type.as_deref()?);
    let sample_rate = recording.recording_sample_rate.unwrap_or_else(|| {
        if fallback_sample_rate.is_finite() && fallback_sample_rate > 0.0 {
            fallback_sample_rate.round() as u32
        } else {
            48_000
        }
    });
    let amplitude = 10.0_f32.powf(recording.signal_level_db.unwrap_or(0.0) / 20.0);

    match signal_type.as_str() {
        "mls" | "maximumlengthsequence" | "maximumlengthsequences" => {
            Some(("MLS", gen_mls(DEFAULT_MLS_ORDER, amplitude), sample_rate))
        }
        "dirac" | "impulse" => {
            let duration = recording
                .signal_duration_secs
                .unwrap_or(1.0)
                .max(1.0 / sample_rate as f32);
            Some((
                "Dirac",
                gen_dirac(amplitude, sample_rate, duration),
                sample_rate,
            ))
        }
        _ => None,
    }
}

fn decode_source_wav(
    channel_name: &str,
    source: &MeasurementSource,
) -> Option<crate::wav::DecodedMonoWav> {
    let wav_path = source.wav_path()?;
    let path = Path::new(wav_path);
    if !path.exists() {
        debug!("  WAV file not found for '{}': {:?}", channel_name, path);
        return None;
    }
    match decode_first_channel(path) {
        Ok(decoded) => Some(decoded),
        Err(error) => {
            debug!(
                "  Could not decode arrival WAV for '{}': {}",
                channel_name, error
            );
            None
        }
    }
}

fn arrival_time_from_decoded(
    channel_name: &str,
    room_config: &RoomConfig,
    sample_rate: f64,
    probe_arrival_ms: Option<f64>,
    decoded: Option<&crate::wav::DecodedMonoWav>,
) -> Option<f64> {
    if let Some(probe_ms) = probe_arrival_ms {
        debug!(
            "  Using probe-based arrival time for '{}': {:.2} ms",
            channel_name, probe_ms
        );
        return Some(probe_ms);
    }
    let decoded = decoded?;

    if let Some((reference_name, reference_signal, reference_sample_rate)) =
        matched_reference_from_recording_config(room_config, sample_rate)
        && !reference_signal.is_empty()
    {
        if reference_sample_rate != 0 && reference_sample_rate != decoded.sample_rate {
            debug!(
                "  {} reference rate {} Hz differs from '{}' WAV rate {} Hz; using WAV timing",
                reference_name, reference_sample_rate, channel_name, decoded.sample_rate
            );
        }
        match roomeq_engine::time_align::detect_delay_with_probe(
            &reference_signal,
            &decoded.samples,
            decoded.sample_rate,
        ) {
            Ok(result) => {
                debug!(
                    "  {} matched arrival for '{}': {:.2} ms (peak at sample {}, SNR {:.1} dB)",
                    reference_name,
                    channel_name,
                    result.arrival_ms,
                    result.arrival_samples,
                    result.detection_snr_db
                );
                return Some(result.arrival_ms);
            }
            Err(error) => debug!(
                "  Could not determine {} matched arrival for '{}': {}; falling back to WAV onset",
                reference_name, channel_name, error
            ),
        }
    }

    match roomeq_engine::time_align::find_arrival_time_samples(
        &decoded.samples,
        decoded.sample_rate,
        None,
    ) {
        Ok(result) => {
            debug!(
                "  Arrival time for '{}': {:.2} ms (peak at sample {})",
                channel_name, result.arrival_ms, result.arrival_samples
            );
            Some(result.arrival_ms)
        }
        Err(error) => {
            debug!(
                "  Could not determine arrival time for '{}': {}",
                channel_name, error
            );
            None
        }
    }
}

/// Resolve a channel's acoustic arrival time before engine execution.
pub fn prepare_channel_arrival_time(
    channel_name: &str,
    source: &MeasurementSource,
    room_config: &RoomConfig,
    sample_rate: f64,
    probe_arrival_ms: Option<f64>,
) -> Option<f64> {
    if probe_arrival_ms.is_some() {
        return arrival_time_from_decoded(
            channel_name,
            room_config,
            sample_rate,
            probe_arrival_ms,
            None,
        );
    }
    let decoded = decode_source_wav(channel_name, source);
    arrival_time_from_decoded(
        channel_name,
        room_config,
        sample_rate,
        None,
        decoded.as_ref(),
    )
}

/// Resolve all source-backed resources into one engine-owned channel input.
pub fn prepare_channel_input(
    channel_name: &str,
    source: &MeasurementSource,
    room_config: &RoomConfig,
    sample_rate: f64,
    probe_arrival_ms: Option<f64>,
) -> Result<PreparedChannelInput, Box<dyn std::error::Error>> {
    prepare_channel_input_with_frequency_samples(
        channel_name,
        source,
        room_config,
        sample_rate,
        probe_arrival_ms,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

/// Resolve channel resources using a configurable measurement frequency grid.
pub fn prepare_channel_input_with_frequency_samples(
    channel_name: &str,
    source: &MeasurementSource,
    room_config: &RoomConfig,
    sample_rate: f64,
    probe_arrival_ms: Option<f64>,
    frequency_samples: usize,
) -> Result<PreparedChannelInput, Box<dyn std::error::Error>> {
    let measurements =
        crate::channel_measurements::prepare_channel_measurements_with_frequency_samples(
            source,
            frequency_samples,
        )?;
    let decoded = decode_source_wav(channel_name, source);
    let arrival_time_ms = arrival_time_from_decoded(
        channel_name,
        room_config,
        sample_rate,
        probe_arrival_ms,
        decoded.as_ref(),
    );
    let impulse_response = decoded.map(|decoded| PreparedImpulseResponse {
        samples: decoded.samples,
        sample_rate: f64::from(decoded.sample_rate),
    });
    let target_response_resource = room_config
        .optimizer
        .target_response
        .as_ref()
        .filter(|target| target.shape == TargetShape::File)
        .and_then(|target| target.curve_path.as_ref())
        .cloned()
        .map(TargetCurveConfig::Path);
    let target = crate::prepare_eq_target(target_response_resource.as_ref())?;

    let speaker_name = room_config
        .optimizer
        .cea2034_correction
        .as_ref()
        .and_then(|config| config.speaker_name.clone())
        .or_else(|| source.speaker_name().map(String::from));
    let cea2034_data = room_config
        .optimizer
        .cea2034_correction
        .as_ref()
        .filter(|config| config.enabled)
        .and(speaker_name.as_deref())
        .and_then(|name| room_config.cea2034_cache.as_ref()?.get(name))
        .cloned()
        .map(Box::new);

    Ok(PreparedChannelInput::new(
        measurements,
        arrival_time_ms,
        PreparedCea2034::new(speaker_name, cea2034_data),
        EqResources {
            target,
            impulse_response,
        },
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use autoeq_measurements::{
        Curve, InlineMeasurement, MeasurementRef, MeasurementSingle, SpinoramaBundle,
    };
    use ndarray::Array1;
    use roomeq_model::{Cea2034CorrectionConfig, RecordingConfiguration};

    use super::*;

    fn curve() -> Curve {
        Curve {
            freq: Array1::from_vec(vec![100.0, 1_000.0]),
            spl: Array1::from_vec(vec![80.0, 80.0]),
            ..Curve::default()
        }
    }

    fn source_with_wav(path: &Path) -> MeasurementSource {
        MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Inline(InlineMeasurement {
                frequencies: curve().freq.to_vec(),
                magnitude_db: curve().spl.to_vec(),
                phase_deg: None,
                name: None,
                wav_path: Some(path.to_string_lossy().into_owned()),
                csv_path: None,
            }),
            speaker_name: None,
        })
    }

    fn write_mono_wav(samples: &[f32], sample_rate: u32) -> tempfile::NamedTempFile {
        let file = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(file.path(), spec).unwrap();
        for &sample in samples {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        file
    }

    #[test]
    fn probe_arrival_wins_without_a_wav() {
        let source = MeasurementSource::InMemory(curve());
        let arrival = prepare_channel_arrival_time(
            "left",
            &source,
            &RoomConfig::default(),
            48_000.0,
            Some(2.5),
        );
        assert_eq!(arrival, Some(2.5));
    }

    #[test]
    fn falls_back_to_wav_onset() {
        let sample_rate = 48_000_u32;
        let arrival_sample = 960_usize;
        let mut samples = vec![0.0_f32; 2_048];
        samples[arrival_sample] = 0.8;
        let wav = write_mono_wav(&samples, sample_rate);

        let arrival = prepare_channel_arrival_time(
            "left",
            &source_with_wav(wav.path()),
            &RoomConfig::default(),
            f64::from(sample_rate),
            None,
        );

        assert!((arrival.unwrap() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn matched_reference_detects_delayed_mls() {
        let sample_rate = 48_000_u32;
        let reference = gen_mls(DEFAULT_MLS_ORDER, 0.5);
        let delay_samples = 1_200_usize;
        let mut samples = vec![0.0_f32; delay_samples];
        samples.extend_from_slice(&reference);
        let wav = write_mono_wav(&samples, sample_rate);
        let config = RoomConfig {
            recording_config: Some(RecordingConfiguration {
                signal_type: Some("MLS".to_string()),
                signal_level_db: Some(-6.0206),
                recording_sample_rate: Some(sample_rate),
                ..RecordingConfiguration::default()
            }),
            ..RoomConfig::default()
        };

        let prepared = prepare_channel_input(
            "left",
            &source_with_wav(wav.path()),
            &config,
            f64::from(sample_rate),
            None,
        )
        .unwrap();

        assert!((prepared.arrival_time_ms().unwrap() - 25.0).abs() < 0.2);
        assert!(prepared.eq_resources().impulse_response.is_some());
    }

    #[test]
    fn missing_wav_returns_no_arrival() {
        let source = source_with_wav(Path::new("/nonexistent/roomeq-arrival.wav"));
        assert!(
            prepare_channel_arrival_time("left", &source, &RoomConfig::default(), 48_000.0, None)
                .is_none()
        );
    }

    #[test]
    fn recording_signal_type_normalization_accepts_common_spellings() {
        assert_eq!(normalize_recording_signal_type("  MLS  "), "mls");
        assert_eq!(
            normalize_recording_signal_type("Maximum Length_Sequence"),
            "maximumlengthsequence"
        );
    }

    #[test]
    fn unknown_recording_reference_is_ignored() {
        let config = RoomConfig {
            recording_config: Some(RecordingConfiguration {
                signal_type: Some("Pink Noise".to_string()),
                ..RecordingConfiguration::default()
            }),
            ..RoomConfig::default()
        };
        assert!(matched_reference_from_recording_config(&config, 48_000.0).is_none());
    }

    #[test]
    fn resolves_cea2034_override_and_skips_disabled_data() {
        let response = curve();
        let bundle = SpinoramaBundle {
            on_axis: response.clone(),
            listening_window: response.clone(),
            early_reflections: response.clone(),
            sound_power: response.clone(),
            estimated_in_room: response.clone(),
            er_di: response.clone(),
            sp_di: response,
            curves: HashMap::new(),
        };
        let mut config = RoomConfig {
            optimizer: roomeq_model::OptimizerConfig {
                cea2034_correction: Some(Cea2034CorrectionConfig {
                    enabled: true,
                    speaker_name: Some("Prepared speaker".to_string()),
                    ..Cea2034CorrectionConfig::default()
                }),
                ..roomeq_model::OptimizerConfig::default()
            },
            cea2034_cache: Some(HashMap::from([("Prepared speaker".to_string(), bundle)])),
            ..RoomConfig::default()
        };
        let source_response = curve();
        let source = MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Inline(InlineMeasurement {
                frequencies: source_response.freq.to_vec(),
                magnitude_db: source_response.spl.to_vec(),
                phase_deg: None,
                name: None,
                wav_path: None,
                csv_path: None,
            }),
            speaker_name: Some("Source speaker".to_string()),
        });

        let input = prepare_channel_input("left", &source, &config, 48_000.0, None).unwrap();

        assert_eq!(input.cea2034().speaker_name(), Some("Prepared speaker"));
        assert!(input.cea2034().data().is_some());

        config
            .optimizer
            .cea2034_correction
            .as_mut()
            .unwrap()
            .enabled = false;
        let input = prepare_channel_input("left", &source, &config, 48_000.0, None).unwrap();
        assert_eq!(input.cea2034().speaker_name(), Some("Prepared speaker"));
        assert!(input.cea2034().data().is_none());
    }

    #[test]
    fn prepares_file_target_response_as_channel_resource() {
        let directory = tempfile::TempDir::new().expect("target directory");
        let target_path = directory.path().join("target.csv");
        std::fs::write(&target_path, "frequency,spl\n100,1.5\n1000,-2.0\n").expect("write target");
        let config = RoomConfig {
            optimizer: roomeq_model::OptimizerConfig {
                target_response: Some(roomeq_model::TargetResponseConfig {
                    shape: TargetShape::File,
                    curve_path: Some(target_path),
                    ..roomeq_model::TargetResponseConfig::default()
                }),
                ..roomeq_model::OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };

        let input = prepare_channel_input(
            "left",
            &MeasurementSource::InMemory(curve()),
            &config,
            48_000.0,
            None,
        )
        .expect("prepare file target");

        let Some(roomeq_engine::eq::PreparedEqTarget::Curve(target)) =
            input.eq_resources().target.as_ref()
        else {
            panic!("file target was not prepared as a curve");
        };
        assert_eq!(target.spl.to_vec(), vec![1.5, -2.0]);
    }

    #[test]
    fn missing_file_target_response_is_an_error() {
        let config = RoomConfig {
            optimizer: roomeq_model::OptimizerConfig {
                target_response: Some(roomeq_model::TargetResponseConfig {
                    shape: TargetShape::File,
                    curve_path: Some("/missing/roomeq-target.csv".into()),
                    ..roomeq_model::TargetResponseConfig::default()
                }),
                ..roomeq_model::OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };

        let result = prepare_channel_input(
            "left",
            &MeasurementSource::InMemory(curve()),
            &config,
            48_000.0,
            None,
        );

        assert!(result.is_err());
    }
}
