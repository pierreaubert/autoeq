// =============================================================================
// Branch coverage for decomposed optimize_room pipeline
// =============================================================================

fn tiny_optimizer() -> OptimizerConfig {
    OptimizerConfig {
        processing_mode: ProcessingMode::LowLatency,
        num_filters: 1,
        max_iter: 20,
        population: 6,
        seed: Some(1),
        min_freq: 20.0,
        max_freq: 500.0,
        psychoacoustic: false,
        refine: false,
        ..Default::default()
    }
}

fn room_config_with_optimizer(
    speakers: HashMap<String, SpeakerConfig>,
    system: Option<SystemConfig>,
    optimizer: OptimizerConfig,
) -> RoomConfig {
    RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer,
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    }
}

// ---- prepare_room_optimization ------------------------------------------------

#[test]
fn prepare_room_optimization_without_system_succeeds() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let config = room_config_with_optimizer(speakers, None, tiny_optimizer());

    let (observer, prepared) = prepare_room_optimization(&config, None).unwrap();
    assert!(prepared.system.is_none());
    assert!(observer.lock().unwrap().is_none());
}

#[test]
fn prepare_room_optimization_with_system_and_bass_management_succeeds() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let system = SystemConfig {
        model: SystemModel::HomeCinema,
        speakers: HashMap::from([("Left".to_string(), "left".to_string())]),
        subwoofers: None,
        bass_management: Some(BassManagementConfig {
            enabled: true,
            ..BassManagementConfig::default()
        }),
        ..Default::default()
    };
    let config = room_config_with_optimizer(speakers, Some(system), tiny_optimizer());

    let (_observer, prepared) = prepare_room_optimization(&config, None).unwrap();
    assert!(
        prepared
            .system
            .as_ref()
            .unwrap()
            .bass_management
            .as_ref()
            .unwrap()
            .enabled
    );
}

// ---- validate_room_optimization error paths ----------------------------------

#[test]
fn validate_room_optimization_mismatched_multi_measurement_weights_fails() {
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
        weights: Some(vec![1.0]), // one weight for two measurements
        ..MultiMeasurementConfig::default()
    });
    let config = room_config_with_optimizer(speakers, None, optimizer);

    let result = validate_room_optimization(&config, &observer_none());
    assert!(
        result.is_err(),
        "mismatched multi-measurement weights should fail validation: {:?}",
        result
    );
}

#[test]
fn validate_room_optimization_missing_measurements_fails() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Group(SpeakerGroup {
            name: "left".to_string(),
            speaker_name: None,
            measurements: vec![],
            crossover: None,
        }),
    );
    let config = room_config_with_optimizer(speakers, None, tiny_optimizer());

    let result = validate_room_optimization(&config, &observer_none());
    assert!(
        result.is_err(),
        "speaker group with no measurements should fail validation: {:?}",
        result
    );
}

// ---- select_topology_route / invalid topology --------------------------------

#[test]
fn execute_topology_workflow_generic_route_errors() {
    let config = stereo_2_0_config();
    let sys = config.system.as_ref().unwrap();
    let result = execute_topology_workflow(
        &config,
        sys,
        48000.0,
        None,
        &observer_none(),
        TopologyRoute::Generic,
    );
    assert!(
        result.is_err(),
        "generic route should not be handled by execute_topology_workflow"
    );
}

// ---- execute_topology_workflow happy paths and early stop --------------------

fn stereo_2_1_config() -> RoomConfig {
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

    RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::Stereo,
            speakers: HashMap::from([
                ("L".to_string(), "left".to_string()),
                ("R".to_string(), "right".to_string()),
                ("Sub".to_string(), "sub".to_string()),
            ]),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                crossover: Some("xo".to_string()),
                mapping: [("sub".to_string(), "L".to_string())].into(),
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
    }
}

#[test]
fn execute_topology_workflow_stereo_2_1_returns_result() {
    let config = stereo_2_1_config();
    let sys = config.system.as_ref().unwrap();
    let result = execute_topology_workflow(
        &config,
        sys,
        48000.0,
        None,
        &observer_none(),
        TopologyRoute::Stereo2_1,
    )
    .unwrap();
    assert!(
        !result.channels.is_empty(),
        "stereo 2.1 should produce channels"
    );
}

#[test]
fn execute_topology_workflow_stops_on_observer_request() {
    let config = stereo_2_0_config();
    let sys = config.system.as_ref().unwrap();
    let observer: SharedPipelineObserver = Arc::new(Mutex::new(Some(Box::new(
        |_event: &PipelineEvent| -> PipelineControl { PipelineControl::Stop },
    ))));
    let result = execute_topology_workflow(
        &config,
        sys,
        48000.0,
        None,
        &observer,
        TopologyRoute::Stereo2_0,
    );
    assert!(
        result.is_err(),
        "workflow should halt when observer requests stop"
    );
}

// ---- execute_generic_channels happy path and early stop ----------------------

#[test]
fn execute_generic_channels_stops_on_observer_request() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let config = room_config_with_optimizer(speakers, None, tiny_optimizer());
    let observer: SharedPipelineObserver = Arc::new(Mutex::new(Some(Box::new(
        |_event: &PipelineEvent| -> PipelineControl { PipelineControl::Stop },
    ))));

    let result = execute_generic_channels(&config, 48000.0, None, None, &observer);
    assert!(
        result.is_err(),
        "generic channel optimization should halt when observer requests stop"
    );
}

// ---- assemble_workflow_result / assemble_generic_result ----------------------

#[test]
fn assemble_workflow_result_empty_channels_succeeds() {
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
    let config = room_config_with_optimizer(speakers, Some(system), tiny_optimizer());
    let sys = config.system.as_ref().unwrap();
    let result = RoomOptimizationResult {
        channels: HashMap::new(),
        channel_results: HashMap::new(),
        combined_pre_score: 0.0,
        combined_post_score: 0.0,
        metadata: empty_metadata(),
    };

    let assembled = assemble_workflow_result(
        result,
        &config,
        sys,
        48000.0,
        None,
        &observer_none(),
        &crate::MemoryArtifactStore::new(),
    )
    .unwrap();
    assert!(assembled.channel_results.is_empty());
}

#[test]
fn assemble_generic_result_non_empty_success() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let config = room_config_with_optimizer(speakers, None, tiny_optimizer());

    let curve = flat_curve();
    let mut channel_chains = HashMap::new();
    channel_chains.insert(
        "left".to_string(),
        ChannelDspChain {
            channel: "left".to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: None,
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        },
    );
    let mut channel_results = HashMap::new();
    channel_results.insert(
        "left".to_string(),
        ChannelOptimizationResult {
            name: "left".to_string(),
            pre_score: 0.5,
            post_score: 0.9,
            initial_curve: curve.clone(),
            final_curve: curve,
            biquads: Vec::new(),
            fir_coeffs: None,
        },
    );
    let mut curves = HashMap::new();
    curves.insert("left".to_string(), flat_curve());
    let mut channel_means = HashMap::new();
    channel_means.insert("left".to_string(), 80.0);

    let generic = GenericChannelCollection {
        channel_chains,
        channel_results,
        pre_scores: vec![0.5],
        post_scores: vec![0.9],
        curves,
        channel_means,
        channel_arrivals: HashMap::new(),
    };

    let assembled = assemble_generic_result(
        generic,
        1,
        &config,
        48000.0,
        None,
        &observer_none(),
        &crate::MemoryArtifactStore::new(),
    )
    .unwrap();
    assert!(
        assembled.channel_results.contains_key("left"),
        "assembled generic result should preserve channel results"
    );
}

// ---- sanity_check_result -----------------------------------------------------

#[test]
fn sanity_check_result_non_empty_ok() {
    let result = single_channel_room_result("left");
    assert!(
        sanity_check_result(&result).is_ok(),
        "sanity check should pass for non-empty results"
    );
}

// ---- optimize_room_impl callback / observer / probe-arrivals -----------------

#[test]
fn optimize_room_impl_without_probe_arrivals_succeeds() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let config = room_config_with_optimizer(speakers, None, tiny_optimizer());

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
        "optimize_room_impl without probe arrivals should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_impl_with_probe_arrivals_succeeds() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let config = room_config_with_optimizer(speakers, None, tiny_optimizer());
    let mut probe = HashMap::new();
    probe.insert("left".to_string(), 5.0);

    let result = optimize_room_impl(
        &config,
        48000.0,
        None,
        Some(&probe),
        None,
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        result.is_ok(),
        "optimize_room_impl with probe arrivals should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_impl_progress_callback_receives_updates() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let config = room_config_with_optimizer(speakers, None, tiny_optimizer());
    let call_count = Arc::new(AtomicUsize::new(0));
    let count_clone = Arc::clone(&call_count);

    let callback: RoomOptimizationCallback =
        Box::new(move |_progress: &RoomOptimizationProgress| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            CallbackAction::Continue
        });
    let observer = callback_pipeline_observer(callback);

    let result = optimize_room_impl(
        &config,
        48000.0,
        None,
        None,
        Some(observer),
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        result.is_ok(),
        "optimize_room_impl with progress callback should succeed: {:?}",
        result.err()
    );
    assert!(
        call_count.load(Ordering::SeqCst) > 0,
        "progress callback should have been invoked"
    );
}

#[test]
fn optimize_room_impl_pipeline_observer_stop_halts() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let config = room_config_with_optimizer(speakers, None, tiny_optimizer());
    let observer = Box::new(|_event: &PipelineEvent| -> PipelineControl { PipelineControl::Stop });

    let result = optimize_room_impl(
        &config,
        48000.0,
        None,
        None,
        Some(observer),
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        result.is_err(),
        "optimize_room_impl should halt when observer requests stop"
    );
}
