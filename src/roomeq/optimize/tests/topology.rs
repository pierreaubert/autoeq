#[test]
fn select_topology_route_stereo_2_0() {
    let speakers = stereo_speakers();
    let system = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([
            ("Left".to_string(), "left".to_string()),
            ("Right".to_string(), "right".to_string()),
        ]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };
    let config = base_room_config(speakers, Some(system));
    let route = select_topology_route(&config, &observer_none()).unwrap();
    assert_eq!(route, TopologyRoute::Stereo2_0);
}

#[test]
fn select_topology_route_stereo_2_1() {
    let mut speakers = stereo_speakers();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let system = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([
            ("Left".to_string(), "left".to_string()),
            ("Right".to_string(), "right".to_string()),
            ("Sub".to_string(), "sub".to_string()),
        ]),
        subwoofers: Some(SubwooferSystemConfig {
            config: SubwooferStrategy::Single,
            crossover: None,
            mapping: [("sub".to_string(), "Left".to_string())].into(),
        }),
        bass_management: None,
        ..Default::default()
    };
    let config = base_room_config(speakers, Some(system));
    let route = select_topology_route(&config, &observer_none()).unwrap();
    assert_eq!(route, TopologyRoute::Stereo2_1);
}

#[test]
fn select_topology_route_home_cinema_with_sub() {
    let mut speakers = stereo_speakers();
    speakers.insert(
        "center".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "lfe".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let system = SystemConfig {
        model: SystemModel::HomeCinema,
        speakers: HashMap::from([
            ("Left".to_string(), "left".to_string()),
            ("Right".to_string(), "right".to_string()),
            ("Center".to_string(), "center".to_string()),
            ("LFE".to_string(), "lfe".to_string()),
        ]),
        subwoofers: Some(SubwooferSystemConfig {
            config: SubwooferStrategy::Single,
            crossover: None,
            mapping: [("lfe".to_string(), "Center".to_string())].into(),
        }),
        bass_management: None,
        ..Default::default()
    };
    let config = base_room_config(speakers, Some(system));
    let route = select_topology_route(&config, &observer_none()).unwrap();
    assert_eq!(route, TopologyRoute::HomeCinema);
}

#[test]
fn select_topology_route_home_cinema_without_sub() {
    let mut speakers = stereo_speakers();
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
    let config = base_room_config(speakers, Some(system));
    let route = select_topology_route(&config, &observer_none()).unwrap();
    assert_eq!(route, TopologyRoute::HomeCinema);
}

#[test]
fn select_topology_route_custom_is_generic() {
    let speakers = stereo_speakers();
    let system = SystemConfig {
        model: SystemModel::Custom,
        speakers: HashMap::from([
            ("Left".to_string(), "left".to_string()),
            ("Right".to_string(), "right".to_string()),
        ]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };
    let config = base_room_config(speakers, Some(system));
    let route = select_topology_route(&config, &observer_none()).unwrap();
    assert_eq!(route, TopologyRoute::Generic);
}

#[test]
fn select_topology_route_no_system_is_generic() {
    let speakers = stereo_speakers();
    let config = base_room_config(speakers, None);
    let route = select_topology_route(&config, &observer_none()).unwrap();
    assert_eq!(route, TopologyRoute::Generic);
}

#[test]
fn select_topology_route_speaker_group_falls_back_to_generic() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left_group".to_string(),
        SpeakerConfig::Group(SpeakerGroup {
            name: "left_group".to_string(),
            speaker_name: None,
            measurements: vec![MeasurementSource::InMemory(flat_curve())],
            crossover: None,
        }),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let system = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([
            ("Left".to_string(), "left_group".to_string()),
            ("Right".to_string(), "right".to_string()),
        ]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };
    let config = base_room_config(speakers, Some(system));
    let route = select_topology_route(&config, &observer_none()).unwrap();
    assert_eq!(route, TopologyRoute::Generic);
}

#[test]
fn validate_room_optimization_empty_speakers_fails() {
    let config = RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: None,
        speakers: HashMap::new(),
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };
    let observer = observer_none();
    let result = validate_room_optimization(&config, &observer);
    assert!(result.is_err(), "empty speakers should fail validation");
}

fn stereo_2_0_config() -> RoomConfig {
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
        optimizer: OptimizerConfig {
            processing_mode: ProcessingMode::LowLatency,
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
fn execute_topology_workflow_stereo_2_0_returns_result() {
    let config = stereo_2_0_config();
    let sys = config.system.as_ref().unwrap();
    let observer = observer_none();
    let route = TopologyRoute::Stereo2_0;
    let result = execute_topology_workflow(&config, sys, 48000.0, None, &observer, route).unwrap();
    assert!(!result.channels.is_empty(), "result should have channels");
}

#[test]
fn execute_topology_workflow_home_cinema_without_sub() {
    let mut speakers = stereo_speakers();
    speakers.insert(
        "center".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let config = RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([
                ("Left".to_string(), "left".to_string()),
                ("Right".to_string(), "right".to_string()),
                ("Center".to_string(), "center".to_string()),
            ]),
            subwoofers: None,
            bass_management: None,
            ..Default::default()
        }),
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig {
            processing_mode: ProcessingMode::LowLatency,
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
    };
    let sys = config.system.as_ref().unwrap();
    let observer = observer_none();
    let route = TopologyRoute::HomeCinema;
    let result = execute_topology_workflow(&config, sys, 48000.0, None, &observer, route).unwrap();
    assert!(!result.channels.is_empty(), "result should have channels");
}

#[test]
fn execute_generic_channels_single_speaker() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let observer = observer_none();
    let (generic, total_speakers) =
        execute_generic_channels(&config, 48000.0, None, None, &observer).unwrap();
    assert_eq!(total_speakers, 1);
    assert!(
        generic.channel_results.contains_key("left"),
        "generic results should contain 'left'"
    );
}

#[test]
fn assemble_workflow_result_persists_channels() {
    let speakers = stereo_speakers();
    let system = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([
            ("Left".to_string(), "left".to_string()),
            ("Right".to_string(), "right".to_string()),
        ]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };
    let config = base_room_config(speakers, Some(system.clone()));
    let result = single_channel_room_result("left");
    let sys = config.system.as_ref().unwrap();

    let assembled = assemble_workflow_result(
        result,
        &config,
        sys,
        48000.0,
        None,
        &observer_none(),
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        assembled.is_ok(),
        "workflow assembly should succeed: {:?}",
        assembled.err()
    );
    let assembled = assembled.unwrap();
    assert!(
        assembled.channel_results.contains_key("left"),
        "assembled result should preserve channel results"
    );
}

#[test]
fn assemble_generic_result_empty_channels_fails() {
    let config = base_room_config(HashMap::new(), None);
    let generic = GenericChannelCollection {
        channel_chains: HashMap::new(),
        channel_results: HashMap::new(),
        pre_scores: Vec::new(),
        post_scores: Vec::new(),
        curves: HashMap::new(),
        channel_means: HashMap::new(),
        channel_arrivals: HashMap::new(),
    };

    let assembled = assemble_generic_result(
        generic,
        0,
        &config,
        48000.0,
        None,
        &observer_none(),
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        assembled.is_err() || assembled.as_ref().unwrap().channel_results.is_empty(),
        "generic assembly with no channels should error or produce no channels"
    );
}

#[test]
fn sanity_check_result_empty_channels_errors() {
    let result = RoomOptimizationResult {
        channels: HashMap::new(),
        channel_results: HashMap::new(),
        combined_pre_score: 0.0,
        combined_post_score: 0.0,
        metadata: empty_metadata(),
    };
    assert!(
        sanity_check_result(&result).is_err(),
        "sanity check should fail when no channels are produced"
    );
}
