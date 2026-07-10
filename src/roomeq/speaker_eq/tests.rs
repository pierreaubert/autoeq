use super::super::types::{
    Cea2034CorrectionConfig, CrossoverConfig, ExcursionProtectionConfig, MeasurementSource,
    OptimizerConfig, ProcessingMode, RecordingConfiguration, RoomConfig, SpeakerConfig,
    SubOptimizerConfig, SubwooferSystemConfig, SystemConfig, TargetResponseConfig,
};
use super::apply::process_single_speaker;
use super::misc::determine_optimization_bands;
use crate::Curve;
use crate::error::AutoeqError;
use crate::{InlineMeasurement, MeasurementRef, MeasurementSingle};
use ndarray::Array1;

use std::collections::HashMap;

fn flat_curve() -> Curve {
    Curve {
        freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20000.0), 96),
        spl: Array1::from_elem(96, 80.0),
        phase: None,
        ..Default::default()
    }
}

fn single_speaker_config(processing_mode: ProcessingMode) -> RoomConfig {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        super::super::types::SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );

    RoomConfig {
        version: super::super::types::default_config_version(),
        system: None,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig {
            processing_mode,
            num_filters: 1,
            max_iter: 20,
            population: 6,
            min_freq: 20.0,
            max_freq: 500.0,
            psychoacoustic: false,
            refine: false,
            ..Default::default()
        },
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    }
}

fn write_mono_wav(samples: &[f32], sample_rate: u32) -> tempfile::NamedTempFile {
    let temp_file = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(temp_file.path(), spec).unwrap();
    for &sample in samples {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    temp_file
}

fn wav_source_with_curve(
    wav_path: &std::path::Path,
    curve: Curve,
    speaker_name: Option<&str>,
) -> MeasurementSource {
    MeasurementSource::Single(MeasurementSingle {
        measurement: MeasurementRef::Inline(InlineMeasurement {
            frequencies: curve.freq.to_vec(),
            magnitude_db: curve.spl.to_vec(),
            phase_deg: curve.phase.as_ref().map(|p| p.to_vec()),
            name: None,
            wav_path: Some(wav_path.to_string_lossy().to_string()),
            csv_path: None,
        }),
        speaker_name: speaker_name.map(|s| s.to_string()),
    })
}

#[test]
fn broadband_rejection_tight_threshold() {
    // A 10 % worse result is accepted.
    assert!(!super::broadband_correction_rejected(1.0, 1.10));
    // A 25 % worse result is rejected.
    assert!(super::broadband_correction_rejected(1.0, 1.25));
    // Slightly past the 20 % boundary is rejected.
    assert!(super::broadband_correction_rejected(1.0, 1.200_000_1));
    // Improvement is always accepted.
    assert!(!super::broadband_correction_rejected(1.0, 0.5));
    // Zero pre-score with any positive post-score is rejected.
    assert!(super::broadband_correction_rejected(0.0, 0.1));
}

#[test]
fn kautz_filter_config_serializes_modal_sections() {
    let config = super::create_kautz_filter_config(&[(42.0, 8.0, -4.5), (71.0, 5.5, 2.0)]);

    assert_eq!(
        config.get("topology").unwrap().as_str().unwrap(),
        "kautz_filter"
    );
    assert_eq!(config.get("freq").unwrap().as_f64().unwrap(), 42.0);
    assert_eq!(
        config
            .get("kautz_sections")
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn kautz_modal_without_detected_modes_returns_error() {
    let source = MeasurementSource::InMemory(flat_curve());
    let config = single_speaker_config(ProcessingMode::KautzModal);
    let output_dir = std::env::temp_dir();

    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );

    assert!(matches!(
        result,
        Err(AutoeqError::OptimizationFailed { ref message })
            if message.contains("KautzModal found no room modes")
    ));
}

#[test]
fn three_way_frequency_range_is_not_treated_as_fixed_crossovers() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 20000.0;
    let crossover = super::super::types::CrossoverConfig {
        crossover_type: "LR24".to_string(),
        frequency: None,
        frequencies: None,
        frequency_range: Some((200.0, 3000.0)),
    };

    let bands = determine_optimization_bands(3, &config, &crossover);

    assert_eq!(bands.len(), 3);
    assert_eq!(bands[0], (20.0, 6000.0));
    assert_eq!(bands[1], (100.0, 6000.0));
    assert_eq!(bands[2], (100.0, 20000.0));
}

#[test]
fn sub_passband_detected_on_raw_curve_not_hpf_corrected() {
    // A sub curve that extends flat from ~20 Hz to ~200 Hz then rolls off.
    // Excursion protection adds an HPF at ~80 Hz.
    // The sub passband detection must see the RAW curve (low bound ~20 Hz),
    // not the HPF-corrected curve (which would incorrectly report a higher
    // low bound because the HPF attenuates the low end).
    let freq = Array1::logspace(10.0, f64::log10(20.0), f64::log10(500.0), 64);
    let spl: Vec<f64> = freq
        .iter()
        .map(|&f| {
            if f < 200.0 {
                80.0
            } else {
                80.0 - 20.0 * ((f / 200.0).log2().max(0.0))
            }
        })
        .collect();
    let raw_curve = Curve {
        freq: freq.clone(),
        spl: Array1::from(spl),
        phase: None,
        ..Default::default()
    };

    // Simulate excursion HPF: a 24 dB/oct HPF at 80 Hz
    let hpf = math_audio_iir_fir::Biquad::new(
        math_audio_iir_fir::BiquadFilterType::Highpass,
        80.0,
        48000.0,
        0.707,
        0.0,
    );
    let hpf_resp = crate::response::compute_peq_complex_response(&[hpf], &raw_curve.freq, 48000.0);
    let hpf_curve = crate::response::apply_complex_response(&raw_curve, &hpf_resp);

    let raw_band = super::super::optimize::detect_sub_passband_3db(&raw_curve);
    let hpf_band = super::super::optimize::detect_sub_passband_3db(&hpf_curve);

    let raw_band = raw_band.expect("raw curve should have passband");
    let hpf_band = hpf_band.expect("hpf curve should have passband");

    // The raw curve has full bass extension, so the low bound should be low
    assert!(
        raw_band.0 < 40.0,
        "raw curve low bound should be ~20-30 Hz, got {:.1}",
        raw_band.0
    );
    // The HPF pulls up the low bound significantly
    assert!(
        hpf_band.0 > 50.0,
        "hpf curve low bound should be pulled up by HPF, got {:.1}",
        hpf_band.0
    );
    // The high bound should be similar for both (HPF doesn't affect high end)
    assert!(
        (raw_band.1 - hpf_band.1).abs() < 30.0,
        "high bounds should be similar: raw={:.1} hpf={:.1}",
        raw_band.1,
        hpf_band.1
    );
}

#[test]
fn process_single_speaker_low_latency_succeeds() {
    let source = MeasurementSource::InMemory(flat_curve());
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let output_dir = std::env::temp_dir();

    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );

    assert!(
        result.is_ok(),
        "low-latency single speaker should succeed: {:?}",
        result.err()
    );
}

#[test]
fn process_single_speaker_phase_linear_succeeds() {
    let source = MeasurementSource::InMemory(flat_curve());
    let mut config = single_speaker_config(ProcessingMode::PhaseLinear);
    config.optimizer.fir = Some(crate::roomeq::types::FirConfig {
        phase: "linear".to_string(),
        ..Default::default()
    });
    let output_dir = std::env::temp_dir();

    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );

    assert!(
        result.is_ok(),
        "phase-linear single speaker should succeed: {:?}",
        result.err()
    );
}

#[test]
fn process_single_speaker_with_probe_arrival() {
    let source = MeasurementSource::InMemory(flat_curve());
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let output_dir = std::env::temp_dir();

    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        Some(3.5),
        None,
    );

    assert!(
        result.is_ok(),
        "single speaker with probe arrival should succeed: {:?}",
        result.err()
    );
}

#[test]
fn process_single_speaker_with_shared_mean_spl() {
    let source = MeasurementSource::InMemory(flat_curve());
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let output_dir = std::env::temp_dir();

    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        Some(82.0),
    );

    assert!(
        result.is_ok(),
        "single speaker with shared mean SPL should succeed: {:?}",
        result.err()
    );
}

#[test]
fn process_single_speaker_returns_chain_and_scores() {
    let source = MeasurementSource::InMemory(flat_curve());
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let output_dir = std::env::temp_dir();

    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    )
    .unwrap();

    // MixedModeResult = (chain, pre_score, post_score, initial_curve, final_curve, biquads, mean_spl, arrival_time_ms, fir_coeffs)
    assert!(
        result.1 >= 0.0,
        "pre_score should be non-negative, got {}",
        result.1
    );
    assert!(
        result.2 >= 0.0,
        "post_score should be non-negative, got {}",
        result.2
    );
}

#[test]
fn process_single_speaker_hybrid_mode_succeeds() {
    let source = MeasurementSource::InMemory(flat_curve());
    let mut config = single_speaker_config(ProcessingMode::Hybrid);
    config.optimizer.num_filters = 2;
    config.optimizer.fir = Some(crate::roomeq::types::FirConfig {
        phase: "linear".to_string(),
        ..Default::default()
    });
    let output_dir = std::env::temp_dir();

    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );

    assert!(
        result.is_ok(),
        "hybrid mode single speaker should succeed: {:?}",
        result.err()
    );
}

#[test]
fn process_single_speaker_mixed_phase_mode_succeeds() {
    let source = MeasurementSource::InMemory(flat_curve());
    let mut config = single_speaker_config(ProcessingMode::MixedPhase);
    config.optimizer.num_filters = 2;
    let output_dir = std::env::temp_dir();

    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );

    assert!(
        result.is_ok(),
        "mixed-phase mode single speaker should succeed: {:?}",
        result.err()
    );
}

#[test]
fn determine_optimization_bands_two_way() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 20000.0;
    let crossover = super::super::types::CrossoverConfig {
        crossover_type: "LR24".to_string(),
        frequency: Some(1000.0),
        frequencies: None,
        frequency_range: None,
    };

    let bands = determine_optimization_bands(2, &config, &crossover);
    assert_eq!(bands.len(), 2);
}

#[test]
fn determine_optimization_bands_with_frequencies() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 20000.0;
    let crossover = super::super::types::CrossoverConfig {
        crossover_type: "LR24".to_string(),
        frequency: None,
        frequencies: Some(vec![200.0, 2000.0]),
        frequency_range: None,
    };

    let bands = determine_optimization_bands(3, &config, &crossover);
    assert_eq!(bands.len(), 3);
}

// ===================================================================
// misc.rs unit tests
// ===================================================================

#[test]
fn normalize_recording_signal_type_handles_whitespace_and_case() {
    assert_eq!(super::normalize_recording_signal_type("  MLS  "), "mls");
    assert_eq!(
        super::normalize_recording_signal_type("Maximum-Length-Sequence"),
        "maximumlengthsequence"
    );
    assert_eq!(super::normalize_recording_signal_type("DIRAC"), "dirac");
    assert_eq!(
        super::normalize_recording_signal_type("Pink Noise"),
        "pinknoise"
    );
}

#[test]
fn matched_reference_mls_and_dirac() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.recording_config = Some(RecordingConfiguration {
        signal_type: Some("MLS".to_string()),
        signal_level_db: Some(-6.0),
        recording_sample_rate: Some(48000),
        ..Default::default()
    });
    let (name, signal, sr) =
        super::matched_reference_from_recording_config(&config, 96000.0).unwrap();
    assert_eq!(name, "MLS");
    assert_eq!(sr, 48000);
    assert!(!signal.is_empty());

    config.recording_config = Some(RecordingConfiguration {
        signal_type: Some("Dirac".to_string()),
        signal_level_db: Some(0.0),
        recording_sample_rate: None,
        signal_duration_secs: Some(0.5),
        ..Default::default()
    });
    let (name, signal, sr) = super::matched_reference_from_recording_config(&config, 0.0).unwrap();
    assert_eq!(name, "Dirac");
    assert_eq!(sr, 48_000);
    assert!(!signal.is_empty());
}

#[test]
fn matched_reference_unknown_returns_none() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.recording_config = Some(RecordingConfiguration {
        signal_type: Some("Pink Noise".to_string()),
        ..Default::default()
    });
    assert!(super::matched_reference_from_recording_config(&config, 48000.0).is_none());
}

#[test]
fn load_channel_measurement_ok_and_warn_bounds() {
    let source = MeasurementSource::InMemory(flat_curve());
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let curve = super::load_channel_measurement("left", &source, &config).unwrap();
    assert_eq!(curve.freq.len(), 96);
}

#[test]
fn detect_channel_arrival_time_probe_wins() {
    let source = MeasurementSource::InMemory(flat_curve());
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let arrival = super::detect_channel_arrival_time("left", &source, &config, 48000.0, Some(2.5));
    assert_eq!(arrival, Some(2.5));
}

#[test]
fn detect_channel_arrival_time_mls_reference() {
    let sr = 48000_u32;
    let reference = math_audio_dsp::signals::gen_mls(10, 0.5);
    let delay = 123_usize;
    let mut recorded = vec![0.0_f32; reference.len() + delay + 256];
    for (i, &sample) in reference.iter().enumerate() {
        recorded[i + delay] += sample * 0.8;
    }
    let wav = write_mono_wav(&recorded, sr);
    let curve = flat_curve();
    let source = wav_source_with_curve(wav.path(), curve, None);

    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.recording_config = Some(RecordingConfiguration {
        signal_type: Some("MLS".to_string()),
        recording_sample_rate: Some(sr),
        ..Default::default()
    });

    let arrival = super::detect_channel_arrival_time("left", &source, &config, sr as f64, None);
    assert!(
        arrival.is_some() && arrival.unwrap() > 0.0 && arrival.unwrap() < 10.0,
        "expected positive arrival < 10 ms, got {:?}",
        arrival
    );
}

#[test]
fn detect_channel_arrival_time_falls_back_to_onset() {
    let sr = 48000_u32;
    let mut recorded = vec![0.0_f32; 4096];
    recorded[120] = 0.1;
    let wav = write_mono_wav(&recorded, sr);
    let curve = flat_curve();
    let source = wav_source_with_curve(wav.path(), curve, None);

    let config = single_speaker_config(ProcessingMode::LowLatency);
    let arrival = super::detect_channel_arrival_time("left", &source, &config, sr as f64, None);
    assert!(arrival.is_some());
}

#[test]
fn detect_channel_arrival_time_missing_wav_returns_none() {
    let curve = flat_curve();
    let source = wav_source_with_curve(std::path::Path::new("/nonexistent/path.wav"), curve, None);
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let arrival = super::detect_channel_arrival_time("left", &source, &config, 48000.0, None);
    assert!(arrival.is_none());
}

#[test]
fn cea2034_correction_active_reflects_config() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    assert!(!super::cea2034_correction_active(&config));
    config.optimizer.cea2034_correction = Some(Cea2034CorrectionConfig {
        enabled: true,
        ..Default::default()
    });
    assert!(super::cea2034_correction_active(&config));
}

#[test]
fn generate_excursion_filters_disabled_returns_empty() {
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let filters = super::generate_excursion_filters(&config, &flat_curve(), 48000.0);
    assert!(filters.is_empty());
}

#[test]
fn generate_excursion_filters_enabled_returns_hpf() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.excursion_protection = Some(ExcursionProtectionConfig {
        enabled: true,
        ..Default::default()
    });
    let filters = super::generate_excursion_filters(&config, &flat_curve(), 48000.0);
    assert!(!filters.is_empty());
}

#[test]
fn system_has_subwoofer_legacy_and_system_v2() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    assert!(!super::system_has_subwoofer(&config));

    config.speakers.insert(
        "LFE".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    assert!(super::system_has_subwoofer(&config));

    config.speakers.clear();
    let mut mapping = HashMap::new();
    mapping.insert("sub1".to_string(), "left".to_string());
    config.system = Some(SystemConfig {
        model: super::super::types::SystemModel::Custom,
        speakers: [("left".to_string(), "left".to_string())]
            .into_iter()
            .collect(),
        subwoofers: Some(SubwooferSystemConfig {
            config: super::super::types::SubwooferStrategy::Single,
            crossover: None,
            mapping,
        }),
        bass_management: None,
        ..Default::default()
    });
    assert!(super::system_has_subwoofer(&config));
}

#[test]
fn maybe_clamp_min_freq_for_target_tilt_branches() {
    let curve = flat_curve();
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig::default());
    // No subwoofer → no clamp
    let min =
        super::maybe_clamp_min_freq_for_target_tilt("left", &config, &curve, None, 20.0, 500.0);
    assert_eq!(min, 20.0);

    // With subwoofer but no tilt → no clamp
    config.speakers.insert(
        "sub1".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let min =
        super::maybe_clamp_min_freq_for_target_tilt("left", &config, &curve, None, 20.0, 500.0);
    assert_eq!(min, 20.0);

    // With tilt and subwoofer but flat curve has no F3 → min_freq unchanged
    let tilt = flat_curve();
    let min = super::maybe_clamp_min_freq_for_target_tilt(
        "left",
        &config,
        &curve,
        Some(&tilt),
        20.0,
        500.0,
    );
    assert_eq!(min, 20.0);
}

#[test]
fn mean_and_flatness_score_in_range() {
    let curve = flat_curve();
    let mean = super::mean_response_in_range(&curve, 20.0, 500.0);
    assert!(
        (mean - 80.0).abs() < 0.5,
        "mean should be ~80 dB, got {}",
        mean
    );
    let score = super::flatness_score_in_range(&curve, 20.0, 500.0);
    assert!(
        score.abs() < 0.1,
        "flat curve should have near-zero score, got {}",
        score
    );
}

#[test]
fn target_mean_spl_prefers_shared() {
    assert_eq!(super::target_mean_spl("left", 78.0, Some(82.0)), 82.0);
    assert_eq!(super::target_mean_spl("left", 78.0, None), 78.0);
}

#[test]
fn existing_ssir_wav_path_respects_existence() {
    let wav = write_mono_wav(&[0.0_f32; 8], 48000);
    let curve = flat_curve();
    let source = wav_source_with_curve(wav.path(), curve, None);
    assert_eq!(
        super::existing_ssir_wav_path(&source),
        Some(wav.path().to_path_buf())
    );

    let source = MeasurementSource::InMemory(flat_curve());
    assert!(super::existing_ssir_wav_path(&source).is_none());
}

#[test]
fn is_subwoofer_measurement_channel_detects_roles_and_mapping() {
    let config = single_speaker_config(ProcessingMode::LowLatency);
    assert!(!super::is_subwoofer_measurement_channel("left", &config));
    assert!(super::is_subwoofer_measurement_channel("LFE", &config));
    assert!(super::is_subwoofer_measurement_channel("sub_rear", &config));

    let mut mapping = HashMap::new();
    // mapping key = subwoofer measurement key, value = main speaker logical role
    mapping.insert("sub_meas".to_string(), "left".to_string());
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.system = Some(SystemConfig {
        model: super::super::types::SystemModel::Custom,
        // channel "left" uses measurement key "sub_meas"
        speakers: [("left".to_string(), "sub_meas".to_string())]
            .into_iter()
            .collect(),
        subwoofers: Some(SubwooferSystemConfig {
            config: super::super::types::SubwooferStrategy::Single,
            crossover: None,
            mapping,
        }),
        bass_management: None,
        ..Default::default()
    });
    assert!(super::is_subwoofer_measurement_channel("left", &config));
}

#[test]
fn determine_optimization_bands_no_crossover_info() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 20000.0;
    let crossover = CrossoverConfig {
        crossover_type: "LR24".to_string(),
        frequency: None,
        frequencies: None,
        frequency_range: None,
    };
    let bands = determine_optimization_bands(2, &config, &crossover);
    assert_eq!(bands.len(), 2);
    assert_eq!(bands[0].0, 20.0);
    assert_eq!(bands[1].1, 20000.0);
}

#[test]
fn optimize_eq_maybe_multi_single_curve_succeeds() {
    let source = MeasurementSource::InMemory(flat_curve());
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let result = super::optimize_eq_maybe_multi(
        &source,
        &flat_curve(),
        &config.optimizer,
        None,
        48000.0,
        "left",
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn optimize_eq_maybe_multi_multi_measurement_weighted() {
    let curves = vec![flat_curve(), flat_curve()];
    let source = MeasurementSource::InMemoryMultiple(curves);
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.multi_measurement = Some(super::super::types::MultiMeasurementConfig {
        strategy: super::super::types::MultiMeasurementStrategy::WeightedSum,
        ..Default::default()
    });
    let result = super::optimize_eq_maybe_multi(
        &source,
        &flat_curve(),
        &config.optimizer,
        None,
        48000.0,
        "left",
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

// ===================================================================
// apply.rs helper tests
// ===================================================================

#[test]
fn apply_excursion_filters_empty_returns_curve_unchanged() {
    let curve = flat_curve();
    let out = super::apply::apply_excursion_filters_to_curve(curve.clone(), &[], 48000.0);
    assert_eq!(out.freq, curve.freq);
    assert_eq!(out.spl, curve.spl);
}

#[test]
fn apply_excursion_filters_non_empty_changes_curve() {
    let curve = flat_curve();
    let hpf = math_audio_iir_fir::Biquad::new(
        math_audio_iir_fir::BiquadFilterType::Highpass,
        100.0,
        48000.0,
        0.707,
        0.0,
    );
    let out = super::apply::apply_excursion_filters_to_curve(curve.clone(), &[hpf], 48000.0);
    // Low end should be attenuated
    let low_idx = out.freq.iter().position(|&f| f > 30.0 && f < 50.0).unwrap();
    let high_idx = out
        .freq
        .iter()
        .position(|&f| f > 200.0 && f < 300.0)
        .unwrap();
    assert!(out.spl[low_idx] < out.spl[high_idx]);
}

#[test]
fn apply_cea2034_speaker_correction_disabled_paths() {
    let curve = flat_curve();
    let source = MeasurementSource::InMemory(curve.clone());
    let mut config = single_speaker_config(ProcessingMode::LowLatency);

    // No config
    let (out, filters, plugins) = super::apply::apply_cea2034_speaker_correction(
        "left",
        &source,
        &config,
        curve.clone(),
        None,
        48000.0,
    );
    assert_eq!(out.spl, curve.spl);
    assert!(filters.is_empty());
    assert!(plugins.is_empty());

    // Config disabled
    config.optimizer.cea2034_correction = Some(Cea2034CorrectionConfig {
        enabled: false,
        ..Default::default()
    });
    let (_out, filters, plugins) = super::apply::apply_cea2034_speaker_correction(
        "left",
        &source,
        &config,
        curve.clone(),
        None,
        48000.0,
    );
    assert!(filters.is_empty());
    assert!(plugins.is_empty());
}

#[test]
fn apply_cea2034_speaker_correction_no_speaker_name_or_cache() {
    let curve = flat_curve();
    let source = MeasurementSource::InMemory(curve.clone());
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.cea2034_correction = Some(Cea2034CorrectionConfig {
        enabled: true,
        ..Default::default()
    });

    // No speaker_name anywhere
    let (_out, filters, plugins) = super::apply::apply_cea2034_speaker_correction(
        "left",
        &source,
        &config,
        curve.clone(),
        None,
        48000.0,
    );
    assert!(filters.is_empty());
    assert!(plugins.is_empty());

    // speaker_name configured but cache missing
    let source = MeasurementSource::Single(MeasurementSingle {
        measurement: MeasurementRef::Inline(InlineMeasurement {
            frequencies: curve.freq.to_vec(),
            magnitude_db: curve.spl.to_vec(),
            phase_deg: None,
            name: None,
            wav_path: None,
            csv_path: None,
        }),
        speaker_name: Some("Some Speaker".to_string()),
    });
    let (_out, _filters, plugins) = super::apply::apply_cea2034_speaker_correction(
        "left",
        &source,
        &config,
        curve.clone(),
        None,
        48000.0,
    );
    assert!(plugins.is_empty());
}

fn make_cea2034_data(curve: &Curve) -> crate::read::Cea2034Data {
    crate::read::Cea2034Data {
        on_axis: curve.clone(),
        listening_window: curve.clone(),
        early_reflections: curve.clone(),
        sound_power: curve.clone(),
        estimated_in_room: curve.clone(),
        er_di: curve.clone(),
        sp_di: curve.clone(),
        curves: std::collections::HashMap::new(),
    }
}

#[test]
fn apply_cea2034_speaker_correction_with_cache_succeeds() {
    let curve = flat_curve();
    let source = MeasurementSource::Single(MeasurementSingle {
        measurement: MeasurementRef::Inline(InlineMeasurement {
            frequencies: curve.freq.to_vec(),
            magnitude_db: curve.spl.to_vec(),
            phase_deg: None,
            name: None,
            wav_path: None,
            csv_path: None,
        }),
        speaker_name: Some("Speaker".to_string()),
    });
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.cea2034_correction = Some(Cea2034CorrectionConfig {
        enabled: true,
        num_filters: 2,
        ..Default::default()
    });
    let mut cache = std::collections::HashMap::new();
    cache.insert("Speaker".to_string(), make_cea2034_data(&curve));
    config.cea2034_cache = Some(cache);

    let (_out, _filters, plugins) = super::apply::apply_cea2034_speaker_correction(
        "left",
        &source,
        &config,
        curve.clone(),
        None,
        48000.0,
    );
    // Correction may return filters or skip if curve can't be improved
    assert!(plugins.is_empty() || plugins.len() == 1);
}

#[test]
fn apply_broadband_precorrection_disabled_returns_identity() {
    let curve = flat_curve();
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let bb = super::apply::apply_broadband_precorrection(
        &config, &curve, None, 80.0, 20.0, 500.0, 48000.0,
    );
    assert_eq!(bb.curve_for_optim.spl, curve.spl);
    assert!(bb.plugins.is_empty());
    assert!(bb.biquads.is_empty());
    assert_eq!(bb.mean_shift, 0.0);
}

#[test]
fn apply_broadband_precorrection_enabled_flat_curve() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig {
        broadband_precorrection: true,
        ..Default::default()
    });
    let curve = flat_curve();
    let bb = super::apply::apply_broadband_precorrection(
        &config, &curve, None, 80.0, 20.0, 500.0, 48000.0,
    );
    // Flat input at target level should produce tiny or no correction
    assert!(bb.mean_shift.abs() < 1.0, "mean_shift={}", bb.mean_shift);
}

#[test]
fn apply_broadband_precorrection_respects_worsening_limit() {
    // A steep rolloff stresses the shelf fit. Depending on the fitted
    // response it may be rejected, but an accepted correction must satisfy
    // the same measured-loss gate as the production path.
    let freq = Array1::logspace(10.0, f64::log10(20.0), f64::log10(20000.0), 96);
    let spl: Vec<f64> = freq
        .iter()
        .map(|&f| {
            if f < 150.0 {
                80.0
            } else {
                80.0 - 25.0 * ((f / 150.0).log2().max(0.0))
            }
        })
        .collect();
    let curve = Curve {
        freq,
        spl: Array1::from(spl),
        phase: None,
        ..Default::default()
    };

    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig {
        broadband_precorrection: true,
        ..Default::default()
    });
    let bb = super::apply::apply_broadband_precorrection(
        &config, &curve, None, 80.0, 20.0, 500.0, 48000.0,
    );
    let target = Array1::from_elem(curve.freq.len(), 80.0);
    let pre_score = crate::loss::flat_loss(&curve.freq, &(&curve.spl - &target), 20.0, 500.0);
    let post_score = crate::loss::flat_loss(
        &bb.curve_for_optim.freq,
        &(&bb.curve_for_optim.spl - &target),
        20.0,
        500.0,
    );
    if bb.curve_for_optim.spl == curve.spl {
        assert!(bb.plugins.is_empty());
        assert!(bb.biquads.is_empty());
        assert_eq!(bb.mean_shift, 0.0);
    } else {
        assert!(
            !super::broadband_correction_rejected(pre_score, post_score),
            "accepted correction exceeded worsening limit: {pre_score} -> {post_score}"
        );
    }
}

#[test]
fn prepare_measurement_in_memory() {
    let curve = flat_curve();
    let source = MeasurementSource::InMemory(curve.clone());
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let input = super::types::ChannelOptimizationInput {
        channel_name: "left",
        source: &source,
        room_config: &config,
        sample_rate: 48000.0,
        output_dir: std::path::Path::new("/tmp"),
        callback: None,
        probe_arrival_ms: Some(1.5),
        shared_mean_spl: None,
    };
    let prepared = super::apply::prepare_measurement(&input).unwrap();
    assert_eq!(prepared.curve.freq.len(), curve.freq.len());
    assert_eq!(prepared.arrival_time_ms, Some(1.5));
}

#[test]
fn build_target_context_flat_returns_none_tilt() {
    let curve = flat_curve();
    let source = MeasurementSource::InMemory(curve.clone());
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let input = super::types::ChannelOptimizationInput {
        channel_name: "left",
        source: &source,
        room_config: &config,
        sample_rate: 48000.0,
        output_dir: std::path::Path::new("/tmp"),
        callback: None,
        probe_arrival_ms: None,
        shared_mean_spl: None,
    };
    let prepared = super::apply::prepare_measurement(&input).unwrap();
    let ctx = super::apply::build_target_context(&input, &prepared).unwrap();
    assert!(ctx.target_tilt_curve.is_none());
    assert_eq!(ctx.min_freq, 20.0);
    assert_eq!(ctx.max_freq, 500.0);
}

#[test]
fn build_target_context_harman_creates_tilt_and_warns_on_target_curve() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig {
        shape: super::super::types::TargetShape::Harman,
        ..Default::default()
    });
    config.target_curve = Some(super::super::types::TargetCurveConfig::Predefined(
        "flat".to_string(),
    ));
    let curve = flat_curve();
    let source = MeasurementSource::InMemory(curve.clone());
    let input = super::types::ChannelOptimizationInput {
        channel_name: "left",
        source: &source,
        room_config: &config,
        sample_rate: 48000.0,
        output_dir: std::path::Path::new("/tmp"),
        callback: None,
        probe_arrival_ms: None,
        shared_mean_spl: None,
    };
    let prepared = super::apply::prepare_measurement(&input).unwrap();
    let ctx = super::apply::build_target_context(&input, &prepared).unwrap();
    assert!(ctx.target_tilt_curve.is_some());
}

#[test]
fn preprocess_features_basic_path() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig {
        broadband_precorrection: true,
        ..Default::default()
    });
    let curve = flat_curve();
    let source = MeasurementSource::InMemory(curve.clone());
    let input = super::types::ChannelOptimizationInput {
        channel_name: "left",
        source: &source,
        room_config: &config,
        sample_rate: 48000.0,
        output_dir: std::path::Path::new("/tmp"),
        callback: None,
        probe_arrival_ms: None,
        shared_mean_spl: None,
    };
    let prepared = super::apply::prepare_measurement(&input).unwrap();
    let mut ctx = super::apply::build_target_context(&input, &prepared).unwrap();
    let features = super::apply::preprocess_features(&input, &prepared, &mut ctx).unwrap();
    assert_eq!(features.curve.freq.len(), curve.freq.len());
    assert!(
        !features.broadband_enabled
            || !features.broadband_plugins.is_empty()
            || features.broadband_mean_shift.abs() < 1.0
    );
}

// ===================================================================
// build.rs tests
// ===================================================================

#[test]
fn build_target_tilt_curve_flat_no_preference_returns_none() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig::default());
    let curve = flat_curve();
    let tilt = super::build::build_target_tilt_curve("left", &config, &curve, false);
    assert!(tilt.is_none());
}

#[test]
fn build_target_tilt_curve_harman_and_custom() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig {
        shape: super::super::types::TargetShape::Harman,
        ..Default::default()
    });
    let curve = flat_curve();
    let tilt = super::build::build_target_tilt_curve("left", &config, &curve, false);
    assert!(tilt.is_some());

    config.optimizer.target_response = Some(TargetResponseConfig {
        shape: super::super::types::TargetShape::Custom,
        slope_db_per_octave: -1.5,
        ..Default::default()
    });
    let tilt = super::build::build_target_tilt_curve("left", &config, &curve, false);
    assert!(tilt.is_some());
}

#[test]
fn build_target_tilt_curve_from_measurement_sub_defaults_flat() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig {
        shape: super::super::types::TargetShape::FromMeasurement,
        ..Default::default()
    });
    let curve = flat_curve();
    let tilt = super::build::build_target_tilt_curve("LFE", &config, &curve, false);
    assert!(tilt.is_some());
}

#[test]
fn build_target_tilt_curve_from_measurement_override_slope() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig {
        shape: super::super::types::TargetShape::FromMeasurement,
        ..Default::default()
    });
    config.optimizer.from_measurement_slope_override = Some(-1.2);
    let curve = flat_curve();
    let tilt = super::build::build_target_tilt_curve("left", &config, &curve, false);
    assert!(tilt.is_some());
}

#[test]
fn build_target_tilt_curve_preference_shelves_create_tilt() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig {
        preference: super::super::types::UserPreference {
            bass_shelf_db: 2.0,
            ..Default::default()
        },
        ..Default::default()
    });
    let curve = flat_curve();
    let tilt = super::build::build_target_tilt_curve("left", &config, &curve, false);
    assert!(tilt.is_some());
}

#[test]
fn build_target_tilt_curve_cea2034_active_strips_preferences() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig {
        shape: super::super::types::TargetShape::Harman,
        preference: super::super::types::UserPreference {
            bass_shelf_db: 2.0,
            ..Default::default()
        },
        ..Default::default()
    });
    let curve = flat_curve();
    let tilt_normal =
        super::build::build_target_tilt_curve("left", &config, &curve, false).unwrap();
    let tilt_cea = super::build::build_target_tilt_curve("left", &config, &curve, true).unwrap();
    // CEA path strips preference shelves, so the curves should differ
    assert!(tilt_normal.spl != tilt_cea.spl);
}

#[test]
fn build_clamped_optimizer_non_sub_passes_through() {
    let curve = flat_curve();
    let source = MeasurementSource::InMemory(curve.clone());
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let opt = super::build::build_clamped_optimizer(
        "left", &source, &config, &curve, &curve, 20.0, 500.0, None, false,
    );
    assert_eq!(opt.min_freq, 20.0);
    assert_eq!(opt.max_freq, 500.0);
}

#[test]
fn build_clamped_optimizer_sub_channel_clamps_max_freq() {
    let freq = Array1::logspace(10.0, f64::log10(20.0), f64::log10(500.0), 64);
    let spl: Vec<f64> = freq
        .iter()
        .map(|&f| {
            if f < 200.0 {
                80.0
            } else {
                80.0 - 20.0 * ((f / 200.0).log2().max(0.0))
            }
        })
        .collect();
    let curve = Curve {
        freq,
        spl: Array1::from(spl),
        phase: None,
        ..Default::default()
    };
    let source = MeasurementSource::InMemory(curve.clone());
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.max_freq = 500.0;
    let opt = super::build::build_clamped_optimizer(
        "LFE", &source, &config, &curve, &curve, 20.0, 500.0, None, false,
    );
    // Sub clamping should reduce max_freq to something below or equal to configured max
    assert!(opt.max_freq <= 500.0);
    assert!(opt.max_freq >= 20.0);
}

#[test]
fn build_clamped_optimizer_sub_config_overrides() {
    let curve = flat_curve();
    let source = MeasurementSource::InMemory(curve.clone());
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.sub_config = Some(SubOptimizerConfig {
        num_filters: 7,
        max_db: 12.0,
        min_db: -15.0,
        min_q: 0.5,
        max_q: 15.0,
    });
    let opt = super::build::build_clamped_optimizer(
        "LFE", &source, &config, &curve, &curve, 20.0, 500.0, None, false,
    );
    assert_eq!(opt.num_filters, 7);
    assert_eq!(opt.max_db, 12.0);
    assert_eq!(opt.min_db, -15.0);
}

#[test]
fn build_clamped_optimizer_ssir_wav_path_set() {
    let wav = write_mono_wav(&[0.0_f32; 8], 48000);
    let curve = flat_curve();
    let source = wav_source_with_curve(wav.path(), curve.clone(), None);
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let opt = super::build::build_clamped_optimizer(
        "left", &source, &config, &curve, &curve, 20.0, 500.0, None, false,
    );
    assert_eq!(opt.ssir_wav_path, Some(wav.path().to_path_buf()));
}

// ===================================================================
// process_single_speaker additional modes
// ===================================================================

#[test]
fn process_single_speaker_warped_iir_succeeds() {
    let source = MeasurementSource::InMemory(flat_curve());
    let config = single_speaker_config(ProcessingMode::WarpedIir);
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn process_single_speaker_mixed_phase_with_phase_data_succeeds() {
    let mut curve = flat_curve();
    curve.phase = Some(Array1::zeros(curve.freq.len()));
    let source = MeasurementSource::InMemory(curve);
    let mut config = single_speaker_config(ProcessingMode::MixedPhase);
    config.optimizer.num_filters = 2;
    config.optimizer.mixed_phase = Some(super::super::types::MixedPhaseSerdeConfig {
        max_fir_length_ms: 5.0,
        pre_ringing_threshold_db: -30.0,
        min_spatial_depth: 0.5,
        phase_smoothing_octaves: 1.0 / 6.0,
    });
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

fn curve_with_room_mode() -> Curve {
    // Add a prominent modal peak around 100 Hz
    let freq = Array1::logspace(10.0, f64::log10(20.0), f64::log10(500.0), 96);
    let spl: Vec<f64> = freq
        .iter()
        .map(|&f| {
            let peak = 8.0 * (-((f - 100.0) / 8.0).powi(2)).exp();
            80.0 + peak
        })
        .collect();
    Curve {
        freq,
        spl: Array1::from(spl),
        phase: None,
        ..Default::default()
    }
}

#[test]
fn process_single_speaker_kautz_modal_with_modes_succeeds() {
    let source = MeasurementSource::InMemory(curve_with_room_mode());
    let config = single_speaker_config(ProcessingMode::KautzModal);
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn process_single_speaker_hybrid_without_mixed_config_succeeds() {
    // Hybrid path without mixed_config falls back to standard IIR+FIR
    let source = MeasurementSource::InMemory(flat_curve());
    let mut config = single_speaker_config(ProcessingMode::Hybrid);
    config.optimizer.num_filters = 2;
    config.optimizer.mixed_config = None;
    config.optimizer.fir = Some(crate::roomeq::types::FirConfig {
        phase: "linear".to_string(),
        ..Default::default()
    });
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn process_single_speaker_low_latency_with_cea2034_and_target_response() {
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.cea2034_correction = Some(Cea2034CorrectionConfig {
        enabled: true,
        ..Default::default()
    });
    config.optimizer.target_response = Some(TargetResponseConfig {
        preference: super::super::types::UserPreference {
            bass_shelf_db: 1.0,
            ..Default::default()
        },
        ..Default::default()
    });
    let source = MeasurementSource::InMemory(flat_curve());
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

// Additional branch coverage for process_single_speaker

#[test]
fn process_single_speaker_phase_linear_succeeds_with_taps() {
    let source = MeasurementSource::InMemory(flat_curve());
    let mut config = single_speaker_config(ProcessingMode::PhaseLinear);
    config.optimizer.fir = Some(crate::roomeq::types::FirConfig {
        taps: 128,
        phase: "linear".to_string(),
        ..Default::default()
    });
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn process_single_speaker_with_target_curve_path_succeeds() {
    let target = Curve {
        freq: flat_curve().freq.clone(),
        spl: Array1::zeros(flat_curve().freq.len()),
        phase: None,
        ..Default::default()
    };
    let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
    use std::io::Write;
    writeln!(tmpfile, "frequency,spl").unwrap();
    for i in 0..target.freq.len() {
        writeln!(tmpfile, "{}, {}", target.freq[i], target.spl[i]).unwrap();
    }
    tmpfile.flush().unwrap();

    let source = MeasurementSource::InMemory(flat_curve());
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.target_curve = Some(super::super::types::TargetCurveConfig::Path(
        tmpfile.path().to_path_buf(),
    ));
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn process_single_speaker_subwoofer_channel_succeeds() {
    let source = MeasurementSource::InMemory(flat_curve());
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "LFE",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn process_single_speaker_with_psychoacoustic_succeeds() {
    let source = MeasurementSource::InMemory(flat_curve());
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.psychoacoustic = true;
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn process_single_speaker_with_refine_succeeds() {
    let source = MeasurementSource::InMemory(flat_curve());
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.refine = true;
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn process_single_speaker_with_schroeder_split_succeeds() {
    let source = MeasurementSource::InMemory(flat_curve());
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.schroeder_split = Some(super::super::types::SchroederSplitConfig {
        enabled: true,
        schroeder_freq: 300.0,
        low_freq_config: super::super::types::LowFreqFilterConfig {
            ..Default::default()
        },
        high_freq_config: super::super::types::HighFreqFilterConfig {
            ..Default::default()
        },
        ..Default::default()
    });
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn process_single_speaker_with_broadband_rejection_succeeds() {
    // Steep rolloff curve triggers broadband rejection branch
    let freq = Array1::logspace(10.0, f64::log10(20.0), f64::log10(20000.0), 96);
    let spl: Vec<f64> = freq
        .iter()
        .map(|&f| {
            if f < 150.0 {
                80.0
            } else {
                80.0 - 25.0 * ((f / 150.0).log2().max(0.0))
            }
        })
        .collect();
    let curve = Curve {
        freq,
        spl: Array1::from(spl),
        phase: None,
        ..Default::default()
    };
    let source = MeasurementSource::InMemory(curve);
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.target_response = Some(TargetResponseConfig {
        broadband_precorrection: true,
        ..Default::default()
    });
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn process_single_speaker_with_cea2034_cache_succeeds() {
    let curve = flat_curve();
    let source = MeasurementSource::Single(MeasurementSingle {
        measurement: MeasurementRef::Inline(InlineMeasurement {
            frequencies: curve.freq.to_vec(),
            magnitude_db: curve.spl.to_vec(),
            phase_deg: None,
            name: None,
            wav_path: None,
            csv_path: None,
        }),
        speaker_name: Some("Speaker".to_string()),
    });
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.cea2034_correction = Some(Cea2034CorrectionConfig {
        enabled: true,
        num_filters: 2,
        ..Default::default()
    });
    let mut cache = std::collections::HashMap::new();
    cache.insert("Speaker".to_string(), make_cea2034_data(&curve));
    config.cea2034_cache = Some(cache);
    let output_dir = std::env::temp_dir();
    let result = process_single_speaker(
        "left",
        &source,
        &config,
        48000.0,
        &output_dir,
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
}
