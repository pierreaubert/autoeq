use super::super::types::{
    Cea2034CorrectionConfig, CrossoverConfig, MeasurementSource, OptimizerConfig, ProcessingMode,
    RoomConfig, SubOptimizerConfig, SubwooferSystemConfig, SystemConfig, TargetResponseConfig,
};
use super::apply::process_single_speaker;
use super::misc::determine_optimization_bands;
use crate::Curve;
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

fn make_cea2034_data(curve: &Curve) -> crate::read::Cea2034Data {
    crate::read::Cea2034Data {
        on_axis: curve.clone(),
        listening_window: curve.clone(),
        early_reflections: curve.clone(),
        sound_power: curve.clone(),
        estimated_in_room: curve.clone(),
        er_di: curve.clone(),
        sp_di: curve.clone(),
        curves: HashMap::new(),
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
    let measurements = roomeq_workflow::prepare_channel_measurements(&source).unwrap();
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let eq_resources = roomeq_engine::eq::EqResources::default();
    let result = super::optimize_eq_maybe_multi(
        &measurements,
        &flat_curve(),
        &config.optimizer,
        &eq_resources,
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
    let measurements = roomeq_workflow::prepare_channel_measurements(&source).unwrap();
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.multi_measurement = Some(super::super::types::MultiMeasurementConfig {
        strategy: super::super::types::MultiMeasurementStrategy::WeightedSum,
        ..Default::default()
    });
    let eq_resources = roomeq_engine::eq::EqResources::default();
    let result = super::optimize_eq_maybe_multi(
        &measurements,
        &flat_curve(),
        &config.optimizer,
        &eq_resources,
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
fn prepare_measurement_in_memory() {
    let curve = flat_curve();
    let source = MeasurementSource::InMemory(curve.clone());
    let measurements = roomeq_workflow::prepare_channel_measurements(&source).unwrap();
    let prepared_input = roomeq_engine::PreparedChannelInput::new(
        measurements,
        Some(1.5),
        roomeq_engine::PreparedCea2034::default(),
        roomeq_engine::eq::EqResources::default(),
    );
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let input = super::types::ChannelOptimizationInput {
        channel_name: "left",
        prepared: &prepared_input,
        room_config: &config,
        sample_rate: 48000.0,
        output_dir: std::path::Path::new("/tmp"),
        callback: None,
        shared_mean_spl: None,
    };
    let prepared = super::apply::prepare_measurement(&input).unwrap();
    assert_eq!(prepared.curve.freq.len(), curve.freq.len());
    assert_eq!(prepared.arrival_time_ms, Some(1.5));
}

#[test]
fn build_clamped_optimizer_non_sub_passes_through() {
    let curve = flat_curve();
    let config = single_speaker_config(ProcessingMode::LowLatency);
    let opt = super::build::build_clamped_optimizer(
        "left", &config, &curve, &curve, 20.0, 500.0, None, false,
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
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.max_freq = 500.0;
    let opt = super::build::build_clamped_optimizer(
        "LFE", &config, &curve, &curve, 20.0, 500.0, None, false,
    );
    // Sub clamping should reduce max_freq to something below or equal to configured max
    assert!(opt.max_freq <= 500.0);
    assert!(opt.max_freq >= 20.0);
}

#[test]
fn build_clamped_optimizer_sub_config_overrides() {
    let curve = flat_curve();
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.sub_config = Some(SubOptimizerConfig {
        num_filters: 7,
        max_db: 12.0,
        min_db: -15.0,
        min_q: 0.5,
        max_q: 15.0,
    });
    let opt = super::build::build_clamped_optimizer(
        "LFE", &config, &curve, &curve, 20.0, 500.0, None, false,
    );
    assert_eq!(opt.num_filters, 7);
    assert_eq!(opt.max_db, 12.0);
    assert_eq!(opt.min_db, -15.0);
}

#[test]
fn build_clamped_optimizer_clears_ssir_wav_path() {
    let curve = flat_curve();
    let mut config = single_speaker_config(ProcessingMode::LowLatency);
    config.optimizer.ssir_wav_path = Some(std::path::PathBuf::from("ignored-ssir.wav"));
    let opt = super::build::build_clamped_optimizer(
        "left", &config, &curve, &curve, 20.0, 500.0, None, false,
    );
    assert!(opt.ssir_wav_path.is_none());
}

// ===================================================================
// process_single_speaker additional modes
// ===================================================================

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
    config.optimizer.num_filters = 2;
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
