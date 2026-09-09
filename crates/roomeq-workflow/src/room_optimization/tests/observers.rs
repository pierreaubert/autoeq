#[test]
#[ignore = "explicit canonical QA diagnosis; not a full five-seed quality gate"]
fn canonical_mso_final_output_loss_diagnostic() {
    assert!(capture_canonical_mso_output_loss(42).is_none());
}

#[test]
#[ignore = "explicit remaining canonical seed diagnosis; not a passing quality gate"]
fn canonical_mso_seed151_output_loss_diagnostic() {
    let error = capture_canonical_mso_output_loss(151).expect("seed 151 failure changed");
    assert!(error.contains("useful output"), "{error}");
}

fn capture_canonical_mso_output_loss(seed: u64) -> Option<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (mut config, _) = crate::load_merged_config_strict(
        &root.join("data_tests/roomeq/generate/fem/small_stereo_2_2_mso/config.json"),
        Some(&root.join("data_tests/roomeq/generate/optimiser-config/small_stereo_2_2_mso/optimiser-iir.json")),
    ).unwrap();
    // Exact default coverage override for the first seed, not a reduced budget.
    config.optimizer.algorithm = "autoeq:cmaes".into();
    config.optimizer.population = 20;
    config.optimizer.asymmetric_loss = false;
    config.optimizer.refine = true;
    config.optimizer.num_filters = 9;
    config.optimizer.max_db = config.optimizer.max_db.min(12.0);
    config.optimizer.max_iter = 600_000;
    config.optimizer.seed = Some(seed);
    config.optimizer.parallel_threads = Some(1);
    let dir = tempfile::tempdir().unwrap();
    let store = autoeq_artifacts::FsArtifactStore::new();
    let captures = seat_replay::capture_training(&config).unwrap();
    let mut result = optimize_room_impl_with_frequency_samples(
        &config, 48_000.0, Some(dir.path()), None, None, &store,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    ).unwrap();
    let pre_validation_graph = result.to_dsp_chain_output();
    let mut ablations = Vec::new();
    for mode in ["without_post_eq", "without_sub_eq", "without_initial_sub_eq", "without_sub_post_eq", "without_main_post_eq",
        "without_channel_matching", "without_sub_eq_and_channel_matching", "without_all_eq"] {
        let mut candidate = result.clone();
        let mut removed_stages = Vec::new();
        for (channel, chain) in &mut candidate.channels {
            let mut plugin_index = 0;
            chain.plugins.retain(|plugin| {
                let post_eq = plugin.parameters.get("label").and_then(|v| v.as_str()) == Some("post_eq");
                let channel_matching = plugin.parameters.get("label").and_then(|v| v.as_str()) == Some("channel_matching");
                let remove = match mode {
                    "without_post_eq" => post_eq,
                    "without_sub_eq" => channel == "LFE" && plugin.plugin_type == "eq",
                    "without_initial_sub_eq" => channel == "LFE" && plugin.plugin_type == "eq" && !post_eq,
                    "without_sub_post_eq" => channel == "LFE" && post_eq,
                    "without_main_post_eq" => channel != "LFE" && post_eq,
                    "without_channel_matching" => channel_matching,
                    "without_sub_eq_and_channel_matching" => channel_matching || (channel == "LFE" && plugin.plugin_type == "eq"),
                    _ => plugin.plugin_type == "eq",
                };
                if remove {
                    removed_stages.push(serde_json::json!({
                        "channel": channel,
                        "original_plugin_index": plugin_index,
                        "plugin": plugin,
                    }));
                }
                plugin_index += 1;
                !remove
            });
        }
        removed_stages.sort_by_key(|stage| (
            stage["channel"].as_str().unwrap().to_owned(),
            stage["original_plugin_index"].as_u64().unwrap(),
        ));
        let check = seat_replay::validate_final_seats(
            &mut candidate, &captures, &HashMap::new(), &config, 48_000.0, dir.path(),
        );
        ablations.push(serde_json::json!({
            "mode": mode, "diagnostic_only": true,
            "removed_stages": removed_stages,
            "final_seat_check_error": check.err().map(|error| error.to_string()),
            "acoustic_quality": candidate.metadata.correction_acceptance
                .as_ref().and_then(|report| report.acoustic_quality.as_ref()),
        }));
    }
    let mut strength_trials = Vec::new();
    for strength in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let mut candidate = result.clone();
        let mut filters_changed = 0;
        for chain in candidate.channels.values_mut() {
            for plugin in &mut chain.plugins {
                if plugin.plugin_type != "eq" { continue; }
                for filter in plugin.parameters["filters"].as_array_mut().expect("EQ filters") {
                    let gain = filter["db_gain"].as_f64().expect("EQ gain");
                    filter["db_gain"] = serde_json::json!(gain * strength);
                    filters_changed += 1;
                }
            }
        }
        let check = seat_replay::validate_final_seats(
            &mut candidate, &captures, &HashMap::new(), &config, 48_000.0, dir.path(),
        );
        let bass = candidate.metadata.bass_management.as_ref().expect("MSO bass metadata");
        let graph = bass.routing_graph.as_ref().expect("MSO routing graph");
        let routed_check = crate::topology::reconstruct_deployed_source_curves(
            &candidate.channels, &retained_fir_coeffs_by_channel(&candidate), graph,
            bass.optimization.as_ref(), 48_000.0, dir.path(),
        );
        let mut safety_candidate = candidate.clone();
        for (name, channel) in &mut safety_candidate.channel_results {
            let chain = safety_candidate.channels.get_mut(name).unwrap();
            let mut logical = chain.clone();
            // Initial curves already contain the physical sub sum.
            logical.drivers = None;
            let realized = crate::ctc::apply_channel_dsp_chain_to_curve_with_sidecar_dir(
                &logical, &channel.initial_curve, 48_000.0, dir.path(),
            ).unwrap();
            channel.final_curve = realized.clone();
            channel.biquads.clear();
            for run in &mut channel.optimizer_evidence { run.selected_for_output = false; }
            chain.final_curve = Some((&realized).into());
        }
        refresh_final_reports(&mut safety_candidate, &config, 48_000.0, dir.path());
        let safety_result = apply_final_correction_safety_gate_preserving_routed_crossover(
            &mut safety_candidate, 48_000.0, config.optimizer.smooth_n,
            (config.optimizer.min_freq, config.optimizer.max_freq), dir.path(),
            config.optimizer.processing_mode.clone(), group_delay_budget_ms(&config),
        );
        let safety_report = safety_candidate.metadata.correction_acceptance.clone();
        let rechecked_seats = seat_replay::validate_final_seats(
            &mut safety_candidate, &captures, &HashMap::new(), &config, 48_000.0, dir.path(),
        );
        let binding = crate::export::bind_final_convolution_artifacts(
            &mut safety_candidate, dir.path(), &store, 48_000.0,
        );
        let electrical = diagnostic_strength_electrical(&safety_candidate, 48_000.0, dir.path());
        let camilladsp = roomeq_export::render_dsp_graph(
            &safety_candidate.to_dsp_chain_output(), roomeq_export::ExportFormat::CamillaDsp, 48_000.0,
        );
        strength_trials.push(serde_json::json!({
            "peq_gain_scale": strength, "filters_changed": filters_changed,
            "diagnostic_only": true, "safety_gains_recomputed": false,
            "final_seat_check_error": check.err().map(|error| error.to_string()),
            "routed_check_error": routed_check.err().map(|error| error.to_string()),
            "after_safety_rebuild": {
                "safety_error": safety_result.err().map(|error| error.to_string()),
                "safety_report_before_seat_check": safety_report,
                "final_seat_check_error": rechecked_seats.err().map(|error| error.to_string()),
                "artifact_binding_error": binding.err().map(|error| error.to_string()),
                "electrical": electrical.unwrap_or_else(|error| serde_json::json!({
                    "status": "error", "error": error.to_string(),
                })),
                "camilladsp_render": match camilladsp {
                    Ok(config) => serde_json::json!({"status": "rendered", "config": config, "backend_executed": false}),
                    Err(error) => serde_json::json!({"status": "error", "error": error.to_string(), "backend_executed": false}),
                },
                "playback": safety_candidate.to_dsp_chain_output(),
            },
            "acoustic_quality": candidate.metadata.correction_acceptance.as_ref()
                .and_then(|report| report.acoustic_quality.as_ref()),
        }));
    }
    let verdict = seat_replay::validate_final_seats(
        &mut result, &captures, &HashMap::new(), &config, 48_000.0, dir.path(),
    );
    let error = verdict.err().map(|error| error.to_string());
    let artifact = serde_json::json!({
        "status": if error.is_none() { "final_seat_checks_passed" } else { "rejected" },
        "scope": "canonical_mso_output_loss_diagnostic", "seed": seed, "error": error,
        "sample_rate": 48_000.0, "requested_optimizer": config.optimizer,
        "before_final_seat_validation": pre_validation_graph,
        "after_final_seat_validation": result.to_dsp_chain_output(),
        "ablations": ablations,
        "strength_trials": strength_trials,
    });
    let output = root.join(if seed == 42 {
        "target/qa/canonical-mso-output-loss.json".to_string()
    } else { format!("target/qa/canonical-mso-output-loss-seed-{seed}.json") });
    std::fs::create_dir_all(output.parent().unwrap()).unwrap();
    std::fs::write(&output, serde_json::to_vec_pretty(&artifact).unwrap()).unwrap();
    eprintln!("canonical seed {seed}: error={error:?}; evidence={}", output.display());
    error
}

/// Independent-input electrical replay for diagnostic strength trials. This
/// deliberately reads filter centers from the serialized graph, not a cache
/// that the diagnostic rebuild may have cleared. It is a sampled sinusoidal
/// bound, not a continuous-frequency or transient/true-peak certificate.
fn diagnostic_strength_electrical(
    result: &RoomOptimizationResult,
    sample_rate: f64,
    dir: &Path,
) -> Result<serde_json::Value> {
    use crate::electrical_headroom::{
        SerializedElectricalPath, canonical_electrical_routing,
        expand_independent_electrical_paths, expand_routed_electrical_paths,
        independent_graph_output_ports, replay_sampled_electrical_headroom,
    };
    let graph = result.to_dsp_chain_output();
    let expanded = if let Some(routing) = canonical_electrical_routing(&graph)? {
        expand_routed_electrical_paths(&graph.channels, routing)?
    } else {
        expand_independent_electrical_paths(
            &graph.channels, &independent_graph_output_ports(&graph.channels),
        )?
    };
    let stages: Vec<Vec<_>> = expanded.iter().map(|path| path.stages.iter().collect()).collect();
    let paths: Vec<_> = expanded.iter().zip(&stages).map(|(path, stages)| {
        SerializedElectricalPath { input: &path.input, output: &path.output, stages }
    }).collect();
    let limits = expanded.iter().map(|path| (path.input.clone(), 1.0)).collect();
    let mut frequencies: Vec<_> = (0..=8192)
        .map(|i| sample_rate * 0.5 * i as f64 / 8192.0).collect();
    for path in &expanded {
        for stage in &path.stages {
            for plugin in &stage.plugins {
                if plugin.plugin_type == "eq"
                    && let Some(filters) = plugin.parameters.get("filters").and_then(|v| v.as_array())
                {
                    for filter in filters {
                        if let Some(frequency) = filter.get("freq").and_then(|v| v.as_f64())
                            && frequency.is_finite() && frequency > 0.0 && frequency < sample_rate / 2.0
                        {
                            frequencies.push(frequency);
                        }
                    }
                }
            }
        }
    }
    frequencies.sort_by(f64::total_cmp);
    frequencies.dedup();
    let outputs = replay_sampled_electrical_headroom(
        &paths, &frequencies, sample_rate, &limits, dir, &HashMap::new(),
    )?;
    Ok(serde_json::json!({
        "status": "evaluated",
        "assessment_kind": "sampled_steady_state_sinusoidal",
        "input_policy": "all_logical_inputs_independently_phased_at_unit_peak",
        "all_outputs_within_full_scale": outputs.iter().all(|output| output.required_attenuation_db <= 1e-6),
        "outputs": outputs,
    }))
}

#[test]
fn strength_electrical_diagnostic_keeps_serialized_centers_without_biquad_cache() {
    let mut result = crate::test_fixtures::single_channel_room_result("left");
    let peak = math_audio_iir_fir::Biquad::new(
        math_audio_iir_fir::BiquadFilterType::Peak, 123.456, 48_000.0, 80.0, 6.0,
    );
    result.channels.get_mut("left").unwrap().plugins = vec![
        roomeq_engine::output::create_eq_plugin(&[peak.clone(), peak]),
    ];
    assert!(result.channel_results["left"].biquads.is_empty());
    let evidence = diagnostic_strength_electrical(&result, 48_000.0, Path::new(".")).unwrap();
    let output = &evidence["outputs"][0];
    assert_eq!(output["peak_frequency_hz"], 123.456);
    assert_eq!(output["grid_points"], 8194);
    assert!((output["required_attenuation_db"].as_f64().unwrap() - 12.0).abs() < 1e-6);
    assert_eq!(evidence["all_outputs_within_full_scale"], false);
}

#[test]
fn reducing_eq_cuts_can_exceed_electrical_full_scale_with_fixed_gain() {
    let mut result = crate::test_fixtures::single_channel_room_result("left");
    let shelves = |gain| roomeq_engine::output::create_eq_plugin(&[
        math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Lowshelf, 1000.0, 48_000.0, 0.7, gain,
        ),
        math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Highshelf, 1000.0, 48_000.0, 0.7, gain,
        ),
    ]);
    result.channels.get_mut("left").unwrap().plugins = vec![
        roomeq_engine::output::create_gain_plugin(5.9), shelves(-6.0),
    ];
    let before = diagnostic_strength_electrical(&result, 48_000.0, Path::new(".")).unwrap();
    result.channels.get_mut("left").unwrap().plugins[1] = shelves(-3.0);
    let after = diagnostic_strength_electrical(&result, 48_000.0, Path::new(".")).unwrap();
    assert_eq!(before["all_outputs_within_full_scale"], true);
    assert_eq!(after["all_outputs_within_full_scale"], false);
    assert!((after["outputs"][0]["required_attenuation_db"].as_f64().unwrap() - 2.9).abs() < 1e-6);
}

#[test]
fn validation_bundle_matches_final_pipeline_playback_evidence() {
    let mut config = minimal_room_config(ProcessingMode::LowLatency);
    config.optimizer.validation_bundle = Some(roomeq_model::ValidationBundleConfig::default());
    let store = autoeq_artifacts::MemoryArtifactStore::new();
    let dir = tempfile::tempdir().unwrap();
    let validation = HashMap::from([("left".to_string(), vec![flat_curve()])]);
    let context = crate::WorkflowContext {
        output_dir: Some(dir.path()), artifact_store: &store,
        validation_measurements: &validation,
    };
    let result = optimize_room_pipeline_impl_with_frequency_samples(
        roomeq_engine::EngineRequest {
            config: &config, sample_rate: 44_100.0, probe_arrival_overrides: None,
        },
        &context, None, crate::DEFAULT_FREQUENCY_SAMPLES,
    ).unwrap();
    let bytes = store.get(&dir.path().join("roomeq_validation_bundle.json")).unwrap();
    let bundle: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(bundle["sample_rate"], 44_100.0);
    assert_eq!(bundle["requested_optimizer"], serde_json::to_value(&config.optimizer).unwrap());
    let mut graph = result.to_dsp_chain_output();
    // The bundle cannot contain its own just-created report pointer.
    graph.metadata.as_mut().unwrap().validation_bundle = None;
    let expected: serde_json::Value = serde_json::from_slice(
        &serde_json::to_vec(&graph).unwrap()
    ).unwrap();
    assert_eq!(bundle["final_playback"], expected);
    let stages = &result.metadata.stage_outcomes;
    let electrical: Vec<_> = stages.iter().filter(|stage|
        stage.stage == "final_graph_sampled_electrical_headroom").collect();
    assert_eq!(electrical.len(), 1);
    assert_eq!(electrical[0], &crate::electrical_headroom::final_graph_unit_peak_stage(
        &result.to_dsp_chain_output(), 44_100.0, dir.path(),
    ));
    assert!(!electrical[0].checks.is_empty());
    assert!(!bundle["final_playback"]["metadata"]["correction_acceptance"].is_null());
    let seats = bundle["final_playback"]["metadata"]["correction_acceptance"]
        ["acoustic_quality"]["final_seats"].as_array().unwrap();
    assert!(seats.iter().any(|seat| seat["partition"] == "held_out"));
}

#[test]
fn rejected_final_seat_validation_does_not_publish_validation_bundle() {
    let mut config = minimal_room_config(ProcessingMode::LowLatency);
    config.optimizer.validation_bundle = Some(roomeq_model::ValidationBundleConfig::default());
    let store = autoeq_artifacts::MemoryArtifactStore::new();
    let dir = tempfile::tempdir().unwrap();
    let curve = crate::test_fixtures::single_channel_room_result("left")
        .channel_results["left"].initial_curve.clone();
    let validation = HashMap::from([("unknown_physical_output".to_string(), vec![curve])]);
    let context = crate::WorkflowContext {
        output_dir: Some(dir.path()), artifact_store: &store,
        validation_measurements: &validation,
    };
    let result = optimize_room_pipeline_impl_with_frequency_samples(
        roomeq_engine::EngineRequest {
            config: &config, sample_rate: 48_000.0, probe_arrival_overrides: None,
        },
        &context, None, crate::DEFAULT_FREQUENCY_SAMPLES,
    );
    let error = result.unwrap_err();
    assert!(error.to_string().contains("unknown held-out physical output"), "{error}");
    assert!(store.get(&dir.path().join("roomeq_validation_bundle.json")).is_none(),
        "a workflow rejected by final-seat validation published a validation bundle");
}

#[test]
fn prepare_room_optimization_with_observer_emits_event() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let event_count = Arc::new(AtomicUsize::new(0));
    let count_clone = Arc::clone(&event_count);

    let observer = Box::new(move |_event: &PipelineEvent| -> PipelineControl {
        count_clone.fetch_add(1, Ordering::SeqCst);
        PipelineControl::Continue
    });

    let (_shared, prepared) = prepare_room_optimization_with_frequency_samples(
        &config,
        Some(observer),
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
    .unwrap();
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
    let request = roomeq_engine::EngineRequest {
        config: &config,
        sample_rate: 48000.0,
        probe_arrival_overrides: None,
    };
    let store = autoeq_artifacts::MemoryArtifactStore::new();
    let validation_measurements = HashMap::new();
    let context = crate::WorkflowContext {
        output_dir: None,
        artifact_store: &store,
        validation_measurements: &validation_measurements,
    };
    let result = optimize_room_pipeline_impl_with_frequency_samples(
        request,
        &context,
        None,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    );
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
    let result = prepare_room_optimization_with_frequency_samples(
        &config,
        observer,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    );
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
    let result = prepare_room_optimization_with_frequency_samples(
        &config,
        observer,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    );
    assert!(
        result.is_err(),
        "observer stop on preparation completed should error"
    );
}

#[test]
fn validate_room_optimization_observer_stop_on_started() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let observer = shared_stop_on_observer(PipelineStepId::Validation, PipelineStepStatus::Started);
    let result = validate_room_optimization_with_frequency_samples(
        &config,
        &observer,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    );
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
    let result = validate_room_optimization_with_frequency_samples(
        &config,
        &observer,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    );
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
    let (generic, total) = execute_generic_channels_with_frequency_samples(
        &config,
        48000.0,
        None,
        None,
        &observer_none(),
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
    .unwrap();
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
    let (_generic, total) = execute_generic_channels_with_frequency_samples(
        &config,
        48000.0,
        None,
        Some(&probe),
        &observer_none(),
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
    .unwrap();
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
    let (generic, total) = execute_generic_channels_with_frequency_samples(
        &config,
        48000.0,
        None,
        None,
        &observer_none(),
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
    .unwrap();
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
        deployed_source_curves: HashMap::new(),
        combined_pre_score: 0.0,
        combined_post_score: 0.0,
        metadata: empty_metadata(),
    };
    let observer = shared_stop_on_observer(
        PipelineStepId::TopologyWorkflowExecution,
        PipelineStepStatus::Completed,
    );
    let assembled = assemble_workflow_result_with_frequency_samples(
        result,
        &config,
        sys,
        48000.0,
        None,
        None,
        &observer,
        &autoeq_artifacts::MemoryArtifactStore::new(),
        crate::DEFAULT_FREQUENCY_SAMPLES,
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
            optimizer_evidence: Vec::new(),
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
            optimizer_evidence: Vec::new(),
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
    let result = assemble_generic_result_with_frequency_samples(
        generic,
        2,
        &config,
        48000.0,
        None,
        &observer,
        &autoeq_artifacts::MemoryArtifactStore::new(),
        crate::DEFAULT_FREQUENCY_SAMPLES,
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
    let result = assemble_generic_result_with_frequency_samples(
        generic,
        2,
        &config,
        48000.0,
        None,
        &observer_none(),
        &autoeq_artifacts::MemoryArtifactStore::new(),
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
    .unwrap();
    assert!(result.channels.contains_key("left"));
    assert!(result.channels.contains_key("right"));
}
