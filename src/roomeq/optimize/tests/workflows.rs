#[test]
fn optimize_speaker_group_missing_crossover_fails() {
    let group = SpeakerConfig::Group(SpeakerGroup {
        name: "group".to_string(),
        speaker_name: None,
        measurements: vec![MeasurementSource::InMemory(flat_curve())],
        crossover: None,
    });
    let result = optimize_speaker("group", &group, &tiny_optimizer(), None, 48000.0, None);
    assert!(
        result.is_err(),
        "speaker group without crossover should fail: {:?}",
        result
    );
}

#[test]
fn optimize_speaker_with_target_curve_succeeds() {
    let source = SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve()));
    let target = Some(TargetCurveConfig::Predefined("flat".into()));
    let result = optimize_speaker(
        "left",
        &source,
        &tiny_optimizer(),
        target.as_ref(),
        48000.0,
        None,
    );
    assert!(
        result.is_ok(),
        "optimize_speaker with target curve should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_speaker_group_no_measurements_fails() {
    let group = SpeakerConfig::Group(SpeakerGroup {
        name: "group".to_string(),
        speaker_name: None,
        measurements: vec![],
        crossover: None,
    });
    let result = optimize_speaker("group", &group, &tiny_optimizer(), None, 48000.0, None);
    assert!(result.is_err(), "group with no measurements should fail");
}

#[test]
fn optimize_room_impl_stereo_2_1_workflow_succeeds() {
    let config = stereo_2_1_config();
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
        "stereo 2.1 workflow should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.channels.is_empty());
}

#[test]
fn execute_topology_workflow_home_cinema_with_sub_returns_result() {
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
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let mut crossovers = HashMap::new();
    crossovers.insert(
        "xo".to_string(),
        CrossoverConfig {
            crossover_type: "LR24".to_string(),
            frequency: Some(80.0),
            frequencies: None,
            frequency_range: None,
        },
    );
    let config = RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([
                ("Left".to_string(), "left".to_string()),
                ("Right".to_string(), "right".to_string()),
                ("Center".to_string(), "center".to_string()),
                ("LFE".to_string(), "sub".to_string()),
            ]),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                crossover: Some("xo".to_string()),
                mapping: [("sub".to_string(), "Center".to_string())].into(),
            }),
            bass_management: None,
            ..Default::default()
        }),
        speakers,
        crossovers: Some(crossovers),
        target_curve: None,
        optimizer: tiny_optimizer(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };
    let sys = config.system.as_ref().unwrap();
    let result = execute_topology_workflow(
        &config,
        sys,
        48000.0,
        None,
        &observer_none(),
        TopologyRoute::HomeCinema,
    )
    .unwrap();
    assert!(
        !result.channels.is_empty(),
        "home cinema with sub should produce channels"
    );
}

// Additional coverage tests for high-level optimize_room branches

#[test]
fn optimize_room_with_group_delay_enabled_succeeds() {
    let mut config = stereo_2_0_config();
    config.optimizer.group_delay = Some(crate::roomeq::types::GroupDelayOptimizationConfig {
        enabled: true,
        max_iter: 20,
        popsize: 4,
        tol: 1e-3,
        min_improvement_db: 0.0,
        ..Default::default()
    });
    let result = optimize_room(&config, 48000.0, None, None);
    assert!(
        result.is_ok(),
        "optimize_room with GD enabled should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_with_phase_alignment_enabled_succeeds() {
    let mut config = stereo_2_1_config();
    config.optimizer.allow_delay = Some(true);
    config.optimizer.phase_alignment = Some(crate::roomeq::types::PhaseAlignmentConfig {
        enabled: true,
        ..Default::default()
    });
    let result = optimize_room(&config, 48000.0, None, None);
    assert!(
        result.is_ok(),
        "optimize_room with phase alignment should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_with_vog_enabled_succeeds() {
    let mut config = stereo_2_0_config();
    config.optimizer.vog = Some(crate::roomeq::types::VoiceOfGodConfig {
        enabled: true,
        reference_channel: "Left".to_string(),
    });
    let result = optimize_room(&config, 48000.0, None, None);
    assert!(
        result.is_ok(),
        "optimize_room with VoG should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_with_target_curve_succeeds() {
    let mut config = stereo_2_0_config();
    config.target_curve = Some(crate::roomeq::types::TargetCurveConfig::Predefined(
        "flat".to_string(),
    ));
    let result = optimize_room(&config, 48000.0, None, None);
    assert!(
        result.is_ok(),
        "optimize_room with target curve should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_with_psychoacoustic_smoothing_succeeds() {
    let mut config = stereo_2_0_config();
    config.optimizer.psychoacoustic = true;
    let result = optimize_room(&config, 48000.0, None, None);
    assert!(
        result.is_ok(),
        "optimize_room with psychoacoustic smoothing should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_with_refine_succeeds() {
    let mut config = stereo_2_0_config();
    config.optimizer.refine = true;
    let result = optimize_room(&config, 48000.0, None, None);
    assert!(
        result.is_ok(),
        "optimize_room with refine should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_with_cobyla_algorithm_succeeds() {
    let mut config = stereo_2_0_config();
    config.optimizer.algorithm = "autoeq:cobyla".to_string();
    let result = optimize_room(&config, 48000.0, None, None);
    assert!(
        result.is_ok(),
        "optimize_room with cobyla should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_generic_with_time_alignment_succeeds() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let mut optimizer = tiny_optimizer();
    optimizer.allow_delay = Some(true);
    let config = room_config_with_optimizer(speakers, None, optimizer);
    let result = optimize_room(&config, 48000.0, None, None);
    assert!(
        result.is_ok(),
        "generic optimize_room with time alignment should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_with_multi_measurement_weighted_succeeds() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![
            flat_curve(),
            flat_curve(),
        ])),
    );
    let mut optimizer = tiny_optimizer();
    optimizer.multi_measurement = Some(MultiMeasurementConfig {
        strategy: MultiMeasurementStrategy::WeightedSum,
        weights: Some(vec![0.6, 0.4]),
        ..Default::default()
    });
    let config = room_config_with_optimizer(speakers, None, optimizer);
    let result = optimize_room(&config, 48000.0, None, None);
    assert!(
        result.is_ok(),
        "optimize_room with multi-measurement weighted should succeed: {:?}",
        result.err()
    );
}
