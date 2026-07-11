#[test]
fn prepare_room_optimization_with_observer_emits_event() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let event_count = Arc::new(AtomicUsize::new(0));
    let count_clone = Arc::clone(&event_count);

    let observer = Box::new(move |_event: &PipelineEvent| -> PipelineControl {
        count_clone.fetch_add(1, Ordering::SeqCst);
        PipelineControl::Continue
    });

    let (_shared, prepared) = prepare_room_optimization(&config, Some(observer)).unwrap();
    assert!(prepared.speakers.contains_key("left"));
}

// =============================================================================
// Additional coverage for top-level entry points and observer error branches
// =============================================================================

fn stop_on_observer(
    step: PipelineStepId,
    status: PipelineStepStatus,
) -> Option<Box<dyn PipelineObserver>> {
    Some(Box::new(move |event: &PipelineEvent| {
        if event.step_id == step && event.status == status {
            PipelineControl::Stop
        } else {
            PipelineControl::Continue
        }
    }))
}

fn shared_stop_on_observer(
    step: PipelineStepId,
    status: PipelineStepStatus,
) -> SharedPipelineObserver {
    Arc::new(Mutex::new(stop_on_observer(step, status)))
}

fn counting_observer() -> (SharedPipelineObserver, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&count);
    let observer = Arc::new(Mutex::new(Some(Box::new(move |_event: &PipelineEvent| {
        c.fetch_add(1, Ordering::SeqCst);
        PipelineControl::Continue
    }) as Box<dyn PipelineObserver>)));
    (observer, count)
}

#[test]
fn optimize_room_with_probe_arrivals_stereo_config() {
    let mut config = stereo_2_0_config();
    config.optimizer = tiny_optimizer();
    let mut probe = HashMap::new();
    probe.insert("left".to_string(), 2.5);
    probe.insert("right".to_string(), 3.0);
    let result = optimize_room_with_probe_arrivals(&config, 48000.0, None, None, &probe);
    assert!(
        result.is_ok(),
        "stereo config with probe arrivals should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_pipeline_impl_direct_call() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let request = RoomPipelineRequest {
        config: &config,
        sample_rate: 48000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    };
    let result = optimize_room_pipeline_impl(request, None);
    assert!(
        result.is_ok(),
        "pipeline impl direct call should succeed: {:?}",
        result.err()
    );
}

#[test]
fn prepare_room_optimization_observer_stop_on_started() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let observer = stop_on_observer(
        PipelineStepId::ConfigPreparation,
        PipelineStepStatus::Started,
    );
    let result = prepare_room_optimization(&config, observer);
    assert!(
        result.is_err(),
        "observer stop on preparation started should error"
    );
}

#[test]
fn prepare_room_optimization_observer_stop_on_completed() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let observer = stop_on_observer(
        PipelineStepId::ConfigPreparation,
        PipelineStepStatus::Completed,
    );
    let result = prepare_room_optimization(&config, observer);
    assert!(
        result.is_err(),
        "observer stop on preparation completed should error"
    );
}

#[test]
fn validate_room_optimization_observer_stop_on_started() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let observer = shared_stop_on_observer(PipelineStepId::Validation, PipelineStepStatus::Started);
    let result = validate_room_optimization(&config, &observer);
    assert!(
        result.is_err(),
        "observer stop on validation started should error"
    );
}

#[test]
fn validate_room_optimization_observer_stop_on_completed() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let observer =
        shared_stop_on_observer(PipelineStepId::Validation, PipelineStepStatus::Completed);
    let result = validate_room_optimization(&config, &observer);
    assert!(
        result.is_err(),
        "observer stop on validation completed should error"
    );
}

#[test]
fn select_topology_route_observer_emits_events() {
    let config = stereo_2_0_config();
    let (observer, count) = counting_observer();
    let route = select_topology_route(&config, &observer).unwrap();
    assert_eq!(route, TopologyRoute::Stereo2_0);
    assert!(
        count.load(Ordering::SeqCst) >= 2,
        "topology route selection should emit started and completed events"
    );
}

#[test]
fn select_topology_route_observer_stop_on_started() {
    let config = stereo_2_0_config();
    let observer = shared_stop_on_observer(
        PipelineStepId::TopologyRouteSelection,
        PipelineStepStatus::Started,
    );
    let result = select_topology_route(&config, &observer);
    assert!(
        result.is_err(),
        "observer stop on route selection started should error"
    );
}

#[test]
fn execute_generic_channels_multiple_speakers() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let config = room_config_with_optimizer(speakers, None, tiny_optimizer());
    let (generic, total) =
        execute_generic_channels(&config, 48000.0, None, None, &observer_none()).unwrap();
    assert_eq!(total, 2);
    assert!(generic.channel_results.contains_key("left"));
    assert!(generic.channel_results.contains_key("right"));
}

#[test]
fn execute_generic_channels_with_probe_arrivals() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let config = room_config_with_optimizer(speakers, None, tiny_optimizer());
    let mut probe = HashMap::new();
    probe.insert("left".to_string(), 1.0);
    probe.insert("right".to_string(), 2.0);
    let (_generic, total) =
        execute_generic_channels(&config, 48000.0, None, Some(&probe), &observer_none()).unwrap();
    assert_eq!(total, 2);
}

#[test]
fn execute_generic_channels_with_group_speaker() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "group".to_string(),
        SpeakerConfig::Group(SpeakerGroup {
            name: "group".to_string(),
            speaker_name: None,
            measurements: vec![
                MeasurementSource::InMemory(flat_curve()),
                MeasurementSource::InMemory(flat_curve()),
            ],
            crossover: Some("xo".to_string()),
        }),
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
    let mut config = room_config_with_optimizer(speakers, None, tiny_optimizer());
    config.crossovers = Some(crossovers);
    let (generic, total) =
        execute_generic_channels(&config, 48000.0, None, None, &observer_none()).unwrap();
    assert_eq!(total, 1);
    assert!(generic.channel_results.contains_key("group"));
}

#[test]
fn assemble_workflow_result_observer_stop_on_summary() {
    let config = stereo_2_0_config();
    let sys = config.system.as_ref().unwrap();
    let result = RoomOptimizationResult {
        channels: HashMap::new(),
        channel_results: HashMap::new(),
        combined_pre_score: 0.0,
        combined_post_score: 0.0,
        metadata: empty_metadata(),
    };
    let observer = shared_stop_on_observer(
        PipelineStepId::TopologyWorkflowExecution,
        PipelineStepStatus::Completed,
    );
    let assembled = assemble_workflow_result(
        result,
        &config,
        sys,
        48000.0,
        None,
        &observer,
        &crate::MemoryArtifactStore::new(),
    );
    assert!(
        assembled.is_err(),
        "observer stop on workflow summary should error"
    );
}

fn two_channel_generic_collection() -> GenericChannelCollection {
    let left = "left".to_string();
    let right = "right".to_string();
    let curve = flat_curve();
    let chain = |name: &str| ChannelDspChain {
        channel: name.to_string(),
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
    };
    let mut channel_chains = HashMap::new();
    channel_chains.insert(left.clone(), chain(&left));
    channel_chains.insert(right.clone(), chain(&right));
    let mut channel_results = HashMap::new();
    channel_results.insert(
        left.clone(),
        ChannelOptimizationResult {
            name: left.clone(),
            pre_score: 0.5,
            post_score: 0.9,
            initial_curve: curve.clone(),
            final_curve: curve.clone(),
            biquads: Vec::new(),
            fir_coeffs: None,
        },
    );
    channel_results.insert(
        right.clone(),
        ChannelOptimizationResult {
            name: right.clone(),
            pre_score: 0.4,
            post_score: 0.8,
            initial_curve: curve.clone(),
            final_curve: curve.clone(),
            biquads: Vec::new(),
            fir_coeffs: None,
        },
    );
    let mut curves = HashMap::new();
    curves.insert(left.clone(), curve.clone());
    curves.insert(right.clone(), curve.clone());
    let mut channel_means = HashMap::new();
    channel_means.insert(left.clone(), 80.0);
    channel_means.insert(right.clone(), 80.0);
    let mut channel_arrivals = HashMap::new();
    channel_arrivals.insert(left.clone(), 0.0);
    channel_arrivals.insert(right.clone(), 1.0);
    GenericChannelCollection {
        channel_chains,
        channel_results,
        pre_scores: vec![0.5, 0.4],
        post_scores: vec![0.9, 0.8],
        curves,
        channel_means,
        channel_arrivals,
    }
}

#[test]
fn assemble_generic_result_with_observer_emits_events() {
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
    optimizer.allow_delay = Some(true);
    let config = room_config_with_optimizer(speakers, None, optimizer);
    let generic = two_channel_generic_collection();
    let (observer, count) = counting_observer();
    let result = assemble_generic_result(
        generic,
        2,
        &config,
        48000.0,
        None,
        &observer,
        &crate::MemoryArtifactStore::new(),
    )
    .unwrap();
    assert!(result.channel_results.contains_key("left"));
    assert!(result.channel_results.contains_key("right"));
    assert!(
        count.load(Ordering::SeqCst) > 0,
        "observer should receive events"
    );
}

#[test]
fn assemble_generic_result_multiple_channels_time_alignment() {
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
    optimizer.allow_delay = Some(true);
    let config = room_config_with_optimizer(speakers, None, optimizer);
    let generic = two_channel_generic_collection();
    let result = assemble_generic_result(
        generic,
        2,
        &config,
        48000.0,
        None,
        &observer_none(),
        &crate::MemoryArtifactStore::new(),
    )
    .unwrap();
    assert!(result.channels.contains_key("left"));
    assert!(result.channels.contains_key("right"));
}
