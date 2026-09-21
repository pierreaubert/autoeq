//! Utility to convert legacy recording.json files to the new RoomConfig format.
//!
//! Usage:
//!   convert_recording <input.json> [output.json]
//!
//! If output is not specified, the input file is overwritten and a .bak backup is created.

use roomeq_model::{
    DspChainOutput, InlineMeasurement, MeasurementRef, MeasurementSource, OptimizerConfig,
    RecordingConfiguration, RoomConfig, SpeakerConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Legacy format: RoomEqMeasurementsFile (from app-gpui)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyMeasurementsFile {
    #[serde(default)]
    version: Option<u32>,
    channels: Vec<LegacyChannelMeasurement>,
    configuration: Option<LegacyRecordingConfiguration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyChannelMeasurement {
    channel_name: String,
    measurement: LegacyRecordingResult,
    #[serde(default)]
    is_group: bool,
    #[serde(default)]
    group_drivers: Vec<LegacyRecordingResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyRecordingResult {
    channel: usize,
    wav_path: Option<String>,
    csv_path: Option<String>,
    frequencies: Vec<f32>,
    magnitude_db: Vec<f32>,
    phase_deg: Vec<f32>,
    #[serde(default)]
    impulse_response: Option<Vec<f32>>,
    #[serde(default)]
    impulse_time_ms: Option<Vec<f32>>,
    #[serde(default)]
    excess_group_delay_ms: Option<Vec<f32>>,
    #[serde(default)]
    thd_percent: Option<Vec<f32>>,
    #[serde(default)]
    harmonic_distortion_db: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    rt60_ms: Option<Vec<f32>>,
    #[serde(default)]
    clarity_c50_db: Option<Vec<f32>>,
    #[serde(default)]
    clarity_c80_db: Option<Vec<f32>>,
    #[serde(default)]
    spectrogram_db: Option<Vec<Vec<f32>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyRecordingConfiguration {
    playback_device_name: String,
    playback_device_id: String,
    playback_sample_rate: u32,
    playback_channels: u32,
    speaker_configuration: String,
    channel_names: Vec<String>,
    recording_device_name: String,
    recording_device_id: String,
    recording_sample_rate: u32,
    recording_channels: u32,
    mic_calibration_path: Option<String>,
    recording_directory: Option<String>,
    signal_type: String,
    signal_duration_secs: f32,
    signal_level_db: f32,
    /// Sweep start frequency in Hz (only applicable when signal_type is "Sweep")
    #[serde(default)]
    sweep_start_freq: Option<f32>,
    /// Sweep end frequency in Hz (only applicable when signal_type is "Sweep")
    #[serde(default)]
    sweep_end_freq: Option<f32>,
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn convert_legacy_to_room_config(legacy: &LegacyMeasurementsFile) -> RoomConfig {
    let mut speakers: HashMap<String, SpeakerConfig> = HashMap::new();

    for ch in &legacy.channels {
        let safe_channel_name = sanitize_filename(&ch.channel_name);
        let result = &ch.measurement;

        // Store only file references, not inline data
        let inline_measurement = InlineMeasurement {
            frequencies: Vec::new(),
            magnitude_db: Vec::new(),
            phase_deg: None,
            name: Some(ch.channel_name.clone()),
            wav_path: result.wav_path.clone(),
            csv_path: Some(format!("{}.csv", safe_channel_name)),
        };

        let measurement_ref = MeasurementRef::Inline(inline_measurement);
        let measurement_source = MeasurementSource::Single(roomeq_model::MeasurementSingle {
            measurement: measurement_ref,
            speaker_name: None,
        });
        let speaker_config = SpeakerConfig::Single(measurement_source);

        speakers.insert(ch.channel_name.clone(), speaker_config);
    }

    // Convert recording configuration if present
    let recording_config = legacy
        .configuration
        .as_ref()
        .map(|cfg| RecordingConfiguration {
            playback_device_name: Some(cfg.playback_device_name.clone()),
            playback_device_id: Some(cfg.playback_device_id.clone()),
            playback_sample_rate: Some(cfg.playback_sample_rate),
            playback_channels: Some(cfg.playback_channels as usize),
            speaker_configuration: Some(cfg.speaker_configuration.clone()),
            channel_names: Some(cfg.channel_names.clone()),
            recording_device_name: Some(cfg.recording_device_name.clone()),
            recording_device_id: Some(cfg.recording_device_id.clone()),
            recording_sample_rate: Some(cfg.recording_sample_rate),
            recording_channels: Some(cfg.recording_channels as usize),
            mic_calibration_path: cfg.mic_calibration_path.clone(),
            mic_calibration_paths: None,
            recording_directory: cfg.recording_directory.clone(),
            signal_type: Some(cfg.signal_type.clone()),
            signal_duration_secs: Some(cfg.signal_duration_secs),
            signal_level_db: Some(cfg.signal_level_db),
            // Sweep parameters for recomputing metrics from WAV
            sweep_start_freq: cfg.sweep_start_freq,
            sweep_end_freq: cfg.sweep_end_freq,
            // Legacy files predate room-info metadata; leave empty so
            // the new fields round-trip as absent.
            ..Default::default()
        });

    RoomConfig {
        version: roomeq_model::default_config_version(),
        system: None,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig::default(),
        recording_config,
        ctc: None,
        cea2034_cache: None,
        provenance: Default::default(),
    }
}

/// Convert a legacy recording JSON string to a `RoomConfig`.
///
/// Provenance, source/seat IDs, calibration state, and timing fields are
/// carried over verbatim: channel names become speaker IDs unchanged, and
/// sample rates, durations, sweep bounds, and calibration paths are copied
/// without rescaling or recentering. Loading through the measurement (L1/L2)
/// acquisition APIs awaits that lane's published loader; until it lands,
/// this converter applies no independent gain, level, or clock adjustment.
pub fn convert_legacy_str_to_room_config(json: &str) -> Result<RoomConfig, String> {
    if !is_legacy_recording_format(json) {
        return Err("input is not a legacy recording file (no \"channels\" array)".to_string());
    }
    let legacy: LegacyMeasurementsFile =
        serde_json::from_str(json).map_err(|e| format!("invalid legacy recording JSON: {e}"))?;
    Ok(convert_legacy_to_room_config(&legacy))
}

fn is_legacy_recording_format(json: &str) -> bool {
    // Check if JSON has "channels" array (legacy) vs "speakers" object (new)
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(json) {
        // New format has "speakers" as object, legacy has "channels" as array
        if value.get("speakers").is_some() {
            return false; // Already new format
        }
        if value.get("channels").is_some() {
            return true; // Legacy format
        }
    }
    false
}

fn backup_path_for(input_path: &std::path::Path) -> PathBuf {
    let mut backup = input_path.to_path_buf();
    let extension = backup
        .extension()
        .map(|e| format!("{}.bak", e.to_string_lossy()))
        .unwrap_or_else(|| "bak".to_string());
    backup.set_extension(extension);
    backup
}

fn write_output(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
    output_json: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if output_path == input_path {
        let backup_path = backup_path_for(input_path);
        std::fs::copy(input_path, &backup_path)?;
        println!("Backup: {}", backup_path.display());
    }

    std::fs::write(output_path, output_json)?;
    println!(
        "Output: {} ({:.2} KB)",
        output_path.display(),
        output_json.len() as f64 / 1024.0
    );
    Ok(())
}

fn strip_deprecated_top_level_keys(value: &mut serde_json::Value) -> Vec<&'static str> {
    let deprecated_keys = ["group_delay"];
    let mut stripped = Vec::new();

    if let Some(obj) = value.as_object_mut() {
        for key in deprecated_keys {
            if obj.remove(key).is_some() {
                stripped.push(key);
            }
        }
    }

    stripped
}

fn normalize_room_config_for_latest_schema(config: &mut RoomConfig) {
    config.version = roomeq_model::default_config_version();

    if config.optimizer.loss_type.eq_ignore_ascii_case("epa")
        && config.optimizer.epa_config.is_none()
    {
        config.optimizer.epa_config = Some(roomeq_model::EpaConfig::default());
    }
}

fn parse_room_config_with_cleanup(json: &str) -> Result<(RoomConfig, Vec<&'static str>), String> {
    let mut value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid JSON: {e}"))?;
    let stripped = strip_deprecated_top_level_keys(&mut value);
    let cleaned_json =
        serde_json::to_string(&value).map_err(|e| format!("failed to encode cleaned JSON: {e}"))?;
    let mut config: RoomConfig = serde_json::from_str(&cleaned_json).map_err(|e| format!("{e}"))?;
    normalize_room_config_for_latest_schema(&mut config);
    Ok((config, stripped))
}

fn parse_dsp_chain_output_with_latest_version(json: &str) -> Result<DspChainOutput, String> {
    let mut output: DspChainOutput = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
    output.version = roomeq_model::default_config_version();
    Ok(output)
}

fn parse_recording_args(args: &[String]) -> Result<(PathBuf, PathBuf), String> {
    if args.len() < 2 {
        let prog = args
            .first()
            .map(|s| s.as_str())
            .unwrap_or("convert_recording");
        return Err(format!(
            "Usage: {prog} <input.json> [output.json]\n\n\
             Converts legacy recording.json files to the new RoomConfig format.\n\n\
             If output is not specified, the input file is overwritten\n\
             and a .bak backup is created."
        ));
    }

    let input_path = PathBuf::from(&args[1]);
    let output_path = if args.len() > 2 {
        PathBuf::from(&args[2])
    } else {
        input_path.clone()
    };
    Ok((input_path, output_path))
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let (input_path, output_path) = match parse_recording_args(&args) {
        Ok(paths) => paths,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(1);
        }
    };

    // Read input file
    let json = std::fs::read_to_string(&input_path)?;
    let file_size = std::fs::metadata(&input_path)?.len();

    println!(
        "Input: {} ({:.2} KB)",
        input_path.display(),
        file_size as f64 / 1024.0
    );

    // Check if already new input or output format. These are still rewritten so
    // their schema version is upgraded to the latest canonical version.
    if !is_legacy_recording_format(&json) {
        match parse_room_config_with_cleanup(&json) {
            Ok((config, stripped_keys)) => {
                for key in stripped_keys {
                    println!("Stripped deprecated key: {}", key);
                }
                println!(
                    "File is in RoomConfig format; rewriting as version {}",
                    config.version
                );
                println!("  {} speaker(s)", config.speakers.len());
                let output_json = serde_json::to_string_pretty(&config)?;
                write_output(&input_path, &output_path, &output_json)?;
                return Ok(());
            }
            Err(room_err) => match parse_dsp_chain_output_with_latest_version(&json) {
                Ok(output) => {
                    println!(
                        "File is in DspChainOutput format; rewriting as version {}",
                        output.version
                    );
                    println!("  {} channel(s)", output.channels.len());
                    let output_json = serde_json::to_string_pretty(&output)?;
                    write_output(&input_path, &output_path, &output_json)?;
                    return Ok(());
                }
                Err(output_err) => {
                    eprintln!("Error: File doesn't appear to be a valid recording format");
                    eprintln!("  RoomConfig parse error: {room_err}");
                    eprintln!("  DspChainOutput parse error: {output_err}");
                    std::process::exit(1);
                }
            },
        }
    }

    // Parse legacy format
    let legacy: LegacyMeasurementsFile = serde_json::from_str(&json)?;
    println!("Detected legacy format (version {:?})", legacy.version);
    println!("  {} channel(s)", legacy.channels.len());

    // Convert to new format
    let room_config = convert_legacy_to_room_config(&legacy);

    // Write output
    let output_json = serde_json::to_string_pretty(&room_config)?;
    write_output(&input_path, &output_path, &output_json)?;
    println!("  {} speaker(s)", room_config.speakers.len());
    println!("Conversion complete!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_recording_args;
    use std::path::PathBuf;

    #[test]
    fn single_input_uses_input_as_output() {
        let args = vec!["prog".to_string(), "in.json".to_string()];
        let (input, output) = parse_recording_args(&args).unwrap();
        assert_eq!(input, PathBuf::from("in.json"));
        assert_eq!(output, PathBuf::from("in.json"));
    }

    #[test]
    fn input_and_output_use_both_paths() {
        let args = vec![
            "prog".to_string(),
            "in.json".to_string(),
            "out.json".to_string(),
        ];
        let (input, output) = parse_recording_args(&args).unwrap();
        assert_eq!(input, PathBuf::from("in.json"));
        assert_eq!(output, PathBuf::from("out.json"));
    }

    #[test]
    fn missing_input_errors() {
        let args = vec!["prog".to_string()];
        assert!(parse_recording_args(&args).is_err());
    }

    #[test]
    fn cli_convert_recording_preserves_timing_and_calibration() {
        // Legacy provenance, IDs, calibration, and timing offsets survive
        // conversion verbatim; nothing is rescaled or recentered.
        let legacy = serde_json::json!({
            "version": 1,
            "channels": [
                {
                    "channel_name": "Left",
                    "measurement": {
                        "channel": 0,
                        "wav_path": "takes/left.wav",
                        "csv_path": "takes/left.csv",
                        "frequencies": [100.0, 1000.0],
                        "magnitude_db": [80.0, 81.0],
                        "phase_deg": [0.0, 1.0]
                    }
                },
                {
                    "channel_name": "Right",
                    "measurement": {
                        "channel": 1,
                        "wav_path": "takes/right.wav",
                        "csv_path": "takes/right.csv",
                        "frequencies": [100.0, 1000.0],
                        "magnitude_db": [79.0, 80.0],
                        "phase_deg": [0.0, 2.0]
                    }
                }
            ],
            "configuration": {
                "playback_device_name": "DAC",
                "playback_device_id": "dac-1",
                "playback_sample_rate": 48000,
                "playback_channels": 2,
                "speaker_configuration": "Stereo",
                "channel_names": ["Left", "Right"],
                "recording_device_name": "Mic",
                "recording_device_id": "mic-1",
                "recording_sample_rate": 48000,
                "recording_channels": 2,
                "mic_calibration_path": "cal/mic.csv",
                "recording_directory": "takes",
                "signal_type": "Sweep",
                "signal_duration_secs": 10.0,
                "signal_level_db": -12.0,
                "sweep_start_freq": 20.0,
                "sweep_end_freq": 20000.0
            }
        });
        let config = super::convert_legacy_str_to_room_config(
            &serde_json::to_string(&legacy).expect("serialize legacy"),
        )
        .expect("legacy conversion must succeed");

        // Source/seat IDs are preserved verbatim as speaker keys.
        let mut ids: Vec<&String> = config.speakers.keys().collect();
        ids.sort();
        assert_eq!(ids, vec!["Left", "Right"]);

        // Calibration state, timing, and signal provenance are unchanged.
        let recording = config
            .recording_config
            .as_ref()
            .expect("recording configuration must survive conversion");
        assert_eq!(
            recording.mic_calibration_path.as_deref(),
            Some("cal/mic.csv")
        );
        assert_eq!(recording.recording_directory.as_deref(), Some("takes"));
        assert_eq!(recording.playback_sample_rate, Some(48000));
        assert_eq!(recording.recording_sample_rate, Some(48000));
        assert_eq!(recording.signal_duration_secs, Some(10.0));
        assert_eq!(recording.signal_level_db, Some(-12.0));
        assert_eq!(recording.sweep_start_freq, Some(20.0));
        assert_eq!(recording.sweep_end_freq, Some(20000.0));
        assert_eq!(recording.signal_type.as_deref(), Some("Sweep"));
        assert_eq!(
            recording.channel_names.as_deref(),
            Some(&["Left".to_string(), "Right".to_string()][..])
        );
    }
}
