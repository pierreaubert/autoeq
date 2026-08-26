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
fn select_topology_route_home_cinema_keeps_mso_bass_output_on_routed_path() {
    let mut speakers = stereo_speakers();
    speakers.insert(
        "center".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "lfe".to_string(),
        SpeakerConfig::MultiSub(roomeq_model::MultiSubGroup {
            name: "lfe_mso".to_string(),
            speaker_name: None,
            subwoofers: vec![
                MeasurementSource::InMemory(flat_curve()),
                MeasurementSource::InMemory(flat_curve()),
            ],
            allpass_optimization: false,
        }),
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
            config: SubwooferStrategy::Mso,
            crossover: None,
            mapping: [("lfe".to_string(), "Center".to_string())].into(),
        }),
        bass_management: None,
        ..Default::default()
    };
    let mut config = base_room_config(speakers, Some(system));

    let route = select_topology_route(&config, &observer_none()).unwrap();

    assert_eq!(route, TopologyRoute::HomeCinema);

    config.speakers.insert(
        "left".to_string(),
        SpeakerConfig::MultiSub(roomeq_model::MultiSubGroup {
            name: "invalid_multidriver_main".to_string(),
            speaker_name: None,
            subwoofers: vec![MeasurementSource::InMemory(flat_curve())],
            allpass_optimization: false,
        }),
    );
    assert_eq!(
        select_topology_route(&config, &observer_none()).unwrap(),
        TopologyRoute::Generic,
        "only the designated home-cinema bass output may bypass the generic multi-driver route"
    );
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
fn select_topology_route_special_bass_configs_without_system_subs_are_generic() {
    let system = |key: &str| SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([("Left".to_string(), key.to_string())]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };
    let source = || MeasurementSource::InMemory(flat_curve());
    let cases = vec![
        (
            "multi",
            SpeakerConfig::MultiSub(roomeq_model::MultiSubGroup {
                name: "m".into(),
                speaker_name: None,
                subwoofers: vec![source()],
                allpass_optimization: false,
            }),
        ),
        (
            "cardioid",
            SpeakerConfig::Cardioid(Box::new(roomeq_model::CardioidConfig {
                name: "c".into(),
                speaker_name: None,
                front: source(),
                rear: source(),
                separation_meters: 1.0,
            })),
        ),
        (
            "dba",
            SpeakerConfig::Dba(roomeq_model::DBAConfig {
                name: "d".into(),
                speaker_name: None,
                front: vec![source()],
                rear: vec![source()],
            }),
        ),
    ];
    for (key, speaker) in cases {
        let config = base_room_config(
            HashMap::from([(key.to_string(), speaker)]),
            Some(system(key)),
        );
        assert_eq!(
            select_topology_route(&config, &observer_none()).unwrap(),
            TopologyRoute::Generic,
            "{key} must not enter the Stereo 2.0 route"
        );
    }
}

#[test]
fn validate_room_optimization_empty_speakers_fails() {
    let config = RoomConfig {
        version: roomeq_model::default_config_version(),
        system: None,
        speakers: HashMap::new(),
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig::default(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };
    let observer = observer_none();
    let result = validate_room_optimization_with_frequency_samples(
        &config,
        &observer,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    );
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
        version: roomeq_model::default_config_version(),
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
        provenance: Default::default(),
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

fn nonflat_parallel_stereo_config() -> RoomConfig {
    let mut config = stereo_2_0_config();
    config.optimizer.num_filters = 2;
    config.optimizer.max_iter = 120;
    config.optimizer.population = 12;
    config.optimizer.max_freq = 2_000.0;
    config.optimizer.seed = Some(42);
    for speaker in config.speakers.values_mut() {
        if let SpeakerConfig::Single(MeasurementSource::InMemory(curve)) = speaker {
            for (frequency, spl) in curve.freq.iter().zip(curve.spl.iter_mut()) {
                *spl +=
                    (frequency.log10() * 5.7).sin() * 4.0 + (frequency.log10() * 13.0).cos() * 1.5;
            }
        }
    }
    config
}

fn assert_parallel_result_matches(
    serial: &RoomOptimizationResult,
    parallel: &RoomOptimizationResult,
) {
    assert_eq!(serial.channel_results.len(), parallel.channel_results.len());
    for (name, serial_channel) in &serial.channel_results {
        let parallel_channel = parallel
            .channel_results
            .get(name)
            .unwrap_or_else(|| panic!("parallel result is missing channel {name}"));
        assert_eq!(
            serial_channel.biquads.len(),
            parallel_channel.biquads.len(),
            "filter count changed for {name}"
        );
        for (serial_filter, parallel_filter) in
            serial_channel.biquads.iter().zip(&parallel_channel.biquads)
        {
            assert_eq!(serial_filter.filter_type, parallel_filter.filter_type);
            for (field, serial_value, parallel_value) in [
                ("frequency", serial_filter.freq, parallel_filter.freq),
                ("q", serial_filter.q, parallel_filter.q),
                ("gain", serial_filter.db_gain, parallel_filter.db_gain),
            ] {
                assert!(
                    (serial_value - parallel_value).abs() <= 1e-10,
                    "{name} {field} differs between 1 and N workers: {serial_value} vs {parallel_value}"
                );
            }
        }
        assert!(
            (serial_channel.post_score - parallel_channel.post_score).abs() <= 1e-10,
            "{name} post score differs: {} vs {}",
            serial_channel.post_score,
            parallel_channel.post_score
        );
        assert_eq!(
            serial_channel.final_curve.freq, parallel_channel.final_curve.freq,
            "{name} final frequency grid changed"
        );
        assert_eq!(
            serial_channel.final_curve.spl.len(),
            parallel_channel.final_curve.spl.len()
        );
        for (index, (&serial_spl, &parallel_spl)) in serial_channel
            .final_curve
            .spl
            .iter()
            .zip(parallel_channel.final_curve.spl.iter())
            .enumerate()
        {
            assert!(
                (serial_spl - parallel_spl).abs() <= 1e-10,
                "{name} final SPL[{index}] differs: {serial_spl} vs {parallel_spl}"
            );
        }
    }
}

#[test]
fn optimize_room_fixed_seed_matches_between_one_and_four_workers() {
    let mut serial_config = nonflat_parallel_stereo_config();
    serial_config.optimizer.refine = true;
    serial_config.optimizer.parallel_threads = Some(1);
    let serial = optimize_room(&serial_config, 48_000.0, None, None)
        .expect("single-worker RoomEQ run must succeed");

    let mut parallel_config = serial_config.clone();
    parallel_config.optimizer.parallel_threads = Some(4);
    let parallel = optimize_room(&parallel_config, 48_000.0, None, None)
        .expect("four-worker RoomEQ run must succeed");

    assert_parallel_result_matches(&serial, &parallel);
}

#[test]
fn optimize_room_cmaes_fixed_seed_matches_between_one_and_four_workers() {
    let mut serial_config = nonflat_parallel_stereo_config();
    serial_config.optimizer.algorithm = "autoeq:cmaes".to_string();
    serial_config.optimizer.max_iter = 300;
    serial_config.optimizer.parallel_threads = Some(1);
    let serial = optimize_room(&serial_config, 48_000.0, None, None)
        .expect("single-worker CMA-ES RoomEQ run must succeed");

    let mut parallel_config = serial_config.clone();
    parallel_config.optimizer.parallel_threads = Some(4);
    let parallel = optimize_room(&parallel_config, 48_000.0, None, None)
        .expect("four-worker CMA-ES RoomEQ run must succeed");

    assert_parallel_result_matches(&serial, &parallel);
}

#[test]
fn optimize_room_five_position_fixed_seed_matches_between_one_and_four_workers() {
    let mut serial_config = nonflat_parallel_stereo_config();
    for speaker in serial_config.speakers.values_mut() {
        let SpeakerConfig::Single(MeasurementSource::InMemory(base)) = speaker else {
            panic!("parallel stereo fixture must contain in-memory curves");
        };
        let positions = (0..5)
            .map(|position| {
                let mut curve = base.clone();
                let offset = position as f64 * 0.11;
                for (frequency, spl) in curve.freq.iter().zip(curve.spl.iter_mut()) {
                    *spl += (frequency.log10() * 3.3 + offset).sin() * 0.75;
                }
                curve
            })
            .collect();
        *speaker = SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(positions));
    }
    serial_config.optimizer.multi_measurement = Some(MultiMeasurementConfig {
        strategy: MultiMeasurementStrategy::VariancePenalized,
        variance_lambda: 0.5,
        ..Default::default()
    });
    serial_config.optimizer.parallel_threads = Some(1);
    let serial = optimize_room(&serial_config, 48_000.0, None, None)
        .expect("single-worker five-position RoomEQ run must succeed");

    let mut parallel_config = serial_config.clone();
    parallel_config.optimizer.parallel_threads = Some(4);
    let parallel = optimize_room(&parallel_config, 48_000.0, None, None)
        .expect("four-worker five-position RoomEQ run must succeed");

    assert_parallel_result_matches(&serial, &parallel);
}

fn home_cinema_5_1_4_config() -> RoomConfig {
    let roles = [
        ("L", "left"),
        ("R", "right"),
        ("C", "center"),
        ("LFE", "lfe"),
        ("SL", "surround_left"),
        ("SR", "surround_right"),
        ("TFL", "top_front_left"),
        ("TFR", "top_front_right"),
        ("TRL", "top_rear_left"),
        ("TRR", "top_rear_right"),
    ];
    let speakers = roles
        .iter()
        .enumerate()
        .map(|(channel_index, (_, channel))| {
            let mut curve = flat_curve();
            let phase = channel_index as f64 * 0.19;
            for (frequency, spl) in curve.freq.iter().zip(curve.spl.iter_mut()) {
                *spl += (frequency.log10() * 5.7 + phase).sin() * 4.0
                    + (frequency.log10() * 13.0 - phase).cos() * 1.5;
            }
            (
                (*channel).to_string(),
                SpeakerConfig::Single(MeasurementSource::InMemory(curve)),
            )
        })
        .collect();

    RoomConfig {
        version: roomeq_model::default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: roles
                .iter()
                .map(|(role, channel)| ((*role).to_string(), (*channel).to_string()))
                .collect(),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                crossover: Some("main".to_string()),
                mapping: [("lfe".to_string(), "L".to_string())].into(),
            }),
            bass_management: None,
            ..Default::default()
        }),
        speakers,
        crossovers: Some(
            [(
                "main".to_string(),
                CrossoverConfig {
                    crossover_type: "LR24".to_string(),
                    frequency: Some(80.0),
                    frequencies: None,
                    frequency_range: None,
                },
            )]
            .into(),
        ),
        target_curve: None,
        optimizer: OptimizerConfig {
            processing_mode: ProcessingMode::LowLatency,
            num_filters: 2,
            max_iter: 120,
            population: 12,
            min_freq: 20.0,
            max_freq: 2_000.0,
            psychoacoustic: false,
            refine: false,
            seed: Some(42),
            ..Default::default()
        },
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    }
}

#[test]
fn optimize_home_cinema_5_1_4_fixed_seed_matches_one_and_four_workers() {
    let mut serial_config = home_cinema_5_1_4_config();
    serial_config.optimizer.parallel_threads = Some(1);
    let serial = optimize_room(&serial_config, 48_000.0, None, None)
        .expect("single-worker 5.1.4 RoomEQ run must succeed");

    let mut parallel_config = serial_config.clone();
    parallel_config.optimizer.parallel_threads = Some(4);
    let parallel = optimize_room(&parallel_config, 48_000.0, None, None)
        .expect("four-worker 5.1.4 RoomEQ run must succeed");

    assert_parallel_result_matches(&serial, &parallel);
}

#[test]
fn optimize_room_stop_after_parallel_channels_start_cancels_every_job() {
    let mut config = nonflat_parallel_stereo_config();
    config.optimizer.parallel_threads = Some(4);
    config.optimizer.max_iter = 2_000;

    let started_channels = Arc::new(Mutex::new(std::collections::HashSet::<String>::new()));
    let callback_count = Arc::new(AtomicUsize::new(0));
    let started_for_callback = Arc::clone(&started_channels);
    let count_for_callback = Arc::clone(&callback_count);
    let callback: RoomOptimizationCallback = Box::new(move |progress| {
        count_for_callback.fetch_add(1, Ordering::SeqCst);
        if progress.iteration > 0 && !progress.current_speaker.is_empty() {
            let mut started = started_for_callback.lock().unwrap();
            started.insert(progress.current_speaker.clone());
            if started.len() >= 2 {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    });

    let output_dir = tempfile::tempdir().expect("cancellation output directory");
    let result = optimize_room(&config, 48_000.0, Some(callback), Some(output_dir.path()));
    assert!(
        result.is_err(),
        "stop request must abort the RoomEQ workflow"
    );
    assert_eq!(
        started_channels.lock().unwrap().len(),
        2,
        "the stop must be requested only after both parallel channels started"
    );
    let observed_callbacks = callback_count.load(Ordering::SeqCst);
    assert!(
        observed_callbacks < config.optimizer.max_iter,
        "active jobs did not observe cancellation before one full channel budget: \
         observed {observed_callbacks} callbacks for max_iter={}",
        config.optimizer.max_iter
    );
    assert_eq!(
        std::fs::read_dir(output_dir.path()).unwrap().count(),
        0,
        "a cancelled workflow must not publish partial output as complete"
    );
}

#[test]
fn execute_topology_workflow_home_cinema_without_sub() {
    let mut speakers = stereo_speakers();
    speakers.insert(
        "center".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let config = RoomConfig {
        version: roomeq_model::default_config_version(),
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
        provenance: Default::default(),
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
    let (generic, total_speakers) = execute_generic_channels_with_frequency_samples(
        &config,
        48000.0,
        None,
        None,
        &observer,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
    .unwrap();
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

    let assembled = assemble_workflow_result_with_frequency_samples(
        result,
        &config,
        sys,
        48000.0,
        None,
        None,
        &observer_none(),
        &autoeq_artifacts::MemoryArtifactStore::new(),
        crate::DEFAULT_FREQUENCY_SAMPLES,
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

    let assembled = assemble_generic_result_with_frequency_samples(
        generic,
        0,
        &config,
        48000.0,
        None,
        &observer_none(),
        &autoeq_artifacts::MemoryArtifactStore::new(),
        crate::DEFAULT_FREQUENCY_SAMPLES,
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
