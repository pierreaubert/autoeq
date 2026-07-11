// =============================================================================
// Additional branch coverage for optimize_room_impl and decomposed helpers
// =============================================================================

fn optimizer_for_mode(processing_mode: ProcessingMode) -> OptimizerConfig {
    OptimizerConfig {
        processing_mode,
        num_filters: 1,
        max_iter: 20,
        population: 6,
        seed: Some(1),
        min_freq: 20.0,
        max_freq: 500.0,
        psychoacoustic: false,
        refine: false,
        fir: Some(crate::roomeq::types::FirConfig {
            taps: 128,
            phase: "linear".to_string(),
            correct_excess_phase: false,
            phase_smoothing: 1.0 / 6.0,
            pre_ringing: None,
        }),
        mixed_phase: Some(crate::roomeq::types::MixedPhaseSerdeConfig {
            max_fir_length_ms: 5.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
        }),
        ..OptimizerConfig::default()
    }
}

fn stereo_config_for_mode(processing_mode: ProcessingMode) -> RoomConfig {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );

    RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::Stereo,
            speakers: HashMap::from([
                ("Left".to_string(), "left".to_string()),
                ("Right".to_string(), "right".to_string()),
            ]),
            subwoofers: None,
            bass_management: None,
            ..Default::default()
        }),
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: optimizer_for_mode(processing_mode),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    }
}

#[test]
fn optimize_room_impl_workflow_phase_linear_succeeds() {
    let config = stereo_config_for_mode(ProcessingMode::PhaseLinear);
    let result = optimize_room_impl(
        &config,
        48000.0,
        None,
        None,
        None,
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        result.is_ok(),
        "phase-linear stereo workflow should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.channels.is_empty());
}

#[test]
fn optimize_room_impl_workflow_hybrid_succeeds() {
    let config = stereo_config_for_mode(ProcessingMode::Hybrid);
    let result = optimize_room_impl(
        &config,
        48000.0,
        None,
        None,
        None,
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        result.is_ok(),
        "hybrid stereo workflow should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.channels.is_empty());
}

#[test]
fn optimize_room_impl_workflow_mixed_phase_succeeds() {
    let config = stereo_config_for_mode(ProcessingMode::MixedPhase);
    let result = optimize_room_impl(
        &config,
        48000.0,
        None,
        None,
        None,
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        result.is_ok(),
        "mixed-phase stereo workflow should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.channels.is_empty());
}

#[test]
fn optimize_room_impl_workflow_phase_correction_succeeds() {
    let mut config = stereo_config_for_mode(ProcessingMode::LowLatency);
    config.optimizer.phase_correction = Some(crate::roomeq::types::MixedPhaseSerdeConfig {
        max_fir_length_ms: 5.0,
        pre_ringing_threshold_db: -30.0,
        min_spatial_depth: 0.5,
        phase_smoothing_octaves: 1.0 / 6.0,
    });
    let result = optimize_room_impl(
        &config,
        48000.0,
        None,
        None,
        None,
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        result.is_ok(),
        "workflow with phase correction should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_impl_generic_multiple_channels_phase_linear_succeeds() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let mut optimizer = optimizer_for_mode(ProcessingMode::PhaseLinear);
    optimizer.fir = Some(crate::roomeq::types::FirConfig {
        taps: 128,
        phase: "linear".to_string(),
        correct_excess_phase: false,
        phase_smoothing: 1.0 / 6.0,
        pre_ringing: None,
    });
    let config = room_config_with_optimizer(speakers, None, optimizer);

    let result = optimize_room_impl(
        &config,
        48000.0,
        None,
        None,
        None,
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        result.is_ok(),
        "generic multi-channel phase-linear optimization should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(result.channels.len() >= 2);
}

#[test]
fn optimize_room_impl_home_cinema_workflow_succeeds() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "center".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let system = SystemConfig {
        model: SystemModel::HomeCinema,
        speakers: HashMap::from([
            ("Left".to_string(), "left".to_string()),
            ("Right".to_string(), "right".to_string()),
            ("Center".to_string(), "center".to_string()),
        ]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };
    let config = room_config_with_optimizer(speakers, Some(system), tiny_optimizer());

    let result = optimize_room_impl(
        &config,
        48000.0,
        None,
        None,
        None,
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        result.is_ok(),
        "home cinema workflow should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.channels.is_empty());
}

#[test]
fn optimize_speaker_single_channel_succeeds() {
    let source = SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve()));
    let result = optimize_speaker("left", &source, &tiny_optimizer(), None, 48000.0, None);
    assert!(
        result.is_ok(),
        "optimize_speaker should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.biquads.is_empty() || result.fir_coeffs.is_some());
}

#[test]
fn validate_room_optimization_single_speaker_succeeds() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let observer = observer_none();
    let result = validate_room_optimization(&config, &observer);
    assert!(
        result.is_ok(),
        "single-speaker config should validate: {:?}",
        result.err()
    );
}
