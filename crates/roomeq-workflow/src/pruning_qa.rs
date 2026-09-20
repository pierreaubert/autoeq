//! End-to-end pruning rows verifying the final exported processing chain.

use ndarray::Array1;
use roomeq_model::{
    FilterAudibilityConfig, MeasurementSource, MultiMeasurementConfig, MultiMeasurementStrategy,
    OptimizerConfig, PruningBudget, RoomConfig, SpeakerConfig,
};
use std::collections::HashMap;

fn measurement(seat: usize) -> autoeq_core::Curve {
    let freq = Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 201);
    let spl = freq
        .mapv(|frequency| (6.0 + seat as f64) * (-((frequency / 80.0).log2() / 0.5).powi(2)).exp());
    autoeq_core::Curve {
        freq,
        spl,
        ..Default::default()
    }
}

fn config(measurements: usize, report_only: bool) -> RoomConfig {
    let ids: Vec<_> = (0..measurements)
        .map(|index| format!("seat-{index}"))
        .collect();
    let evaluation = serde_json::from_value(serde_json::json!({
        "version": "spectral-v1",
        "measurement_ids": ids,
        "programmes": [
            {"id": "flat", "frequencies_hz": [20, 20000], "spectrum_db": [0, 0]},
            {"id": "music", "frequencies_hz": [20, 20000], "spectrum_db": [0, -9]}
        ],
        "listening_levels_phon": [55, 85]
    }))
    .unwrap();
    let source = if measurements == 1 {
        MeasurementSource::InMemory(measurement(0))
    } else {
        MeasurementSource::InMemoryMultiple((0..measurements).map(measurement).collect())
    };
    RoomConfig {
        speakers: HashMap::from([(String::from("left"), SpeakerConfig::Single(source))]),
        optimizer: OptimizerConfig {
            algorithm: String::from("autoeq:de"),
            strategy: String::from("lshade"),
            num_filters: 2,
            max_iter: 500,
            population: 8,
            seed: Some(7),
            min_freq: 20.0,
            max_freq: 500.0,
            min_db: -0.5,
            max_db: 0.5,
            min_filter_improvement: 0.0,
            psychoacoustic: false,
            refine: false,
            multi_measurement: (measurements > 1).then_some(MultiMeasurementConfig {
                strategy: MultiMeasurementStrategy::WeightedSum,
                weights: Some(vec![1.0, 0.0]),
                ..Default::default()
            }),
            filter_audibility: Some(FilterAudibilityConfig {
                report_only,
                allow_enforcement_with_experimental_proxy: !report_only,
                ..Default::default()
            }),
            pruning_budget: Some(PruningBudget {
                evaluation: Some(evaluation),
                aggregation: roomeq_model::BudgetAggregation::Max,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn qa_roomeq_pruning_conditions_exported_single_and_multi_matrix() {
    for algorithm in ["autoeq:de", "autoeq:nsga2"] {
        for measurement_count in [1, 2] {
            for (adaptive, refine) in [(false, false), (true, false), (false, true), (true, true)] {
                for report_only in [true, false] {
                    let mut config = config(measurement_count, report_only);
                    config.optimizer.algorithm = algorithm.to_owned();
                    config.optimizer.min_filter_improvement = if adaptive { 0.001 } else { 0.0 };
                    config.optimizer.refine = refine;
                    config.optimizer.parallel_threads = Some(1);
                    let directory = tempfile::tempdir().unwrap();
                    let result =
                        crate::optimize_room(&config, 48000.0, None, Some(directory.path()))
                            .unwrap();
                    let graph = result.to_dsp_chain_output();
                    let report = roomeq_export::roundtrip::verify_biquad_json_roundtrip(
                        &graph, 48000.0, 1e-10,
                    )
                    .unwrap();
                    let adjudications = graph
                        .metadata
                        .as_ref()
                        .expect("native export must retain optimization metadata")
                        .veto_adjudication
                        .as_ref()
                        .expect("native workflow must preserve cumulative pruning evidence");
                    let adjudication = &adjudications["left"];
                    assert_eq!(adjudication.enforced, !report_only);
                    assert!(adjudication.f0_reference_id.contains("conditions-v1:"));
                    if report_only {
                        assert!(adjudication.removed_filter_indices.is_empty());
                        assert!(
                            report.sections > 0,
                            "advisory row must export the retained EQ"
                        );
                    } else {
                        assert!(
                            !adjudication.removed_filter_indices.is_empty(),
                            "enforced row must exercise a real removal"
                        );
                        assert_eq!(
                            report.sections, 0,
                            "removed EQ must not reappear in the exported chain"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn qa_roomeq_pruning_conditions_exported_hybrid_matrix() {
    for crossover in [true, false] {
        for measurement_count in [1, 2] {
            for report_only in [true, false] {
                let mut config = config(measurement_count, report_only);
                // Hybrid acceptance requires explicit phase evidence. These analytic
                // fixtures declare a zero-phase transfer, unlike magnitude-only rows.
                let curves: Vec<_> = (0..measurement_count)
                    .map(|seat| {
                        let mut curve = measurement(seat);
                        curve.phase = Some(Array1::zeros(curve.freq.len()));
                        curve
                    })
                    .collect();
                let source = if measurement_count == 1 {
                    MeasurementSource::InMemory(curves[0].clone())
                } else {
                    MeasurementSource::InMemoryMultiple(curves)
                };
                config
                    .speakers
                    .insert(String::from("left"), SpeakerConfig::Single(source));
                config.optimizer.parallel_threads = Some(1);
                config.optimizer.processing_mode = roomeq_model::ProcessingMode::Hybrid;
                config.optimizer.mixed_config = crossover.then(|| roomeq_model::MixedModeConfig {
                    crossover_freq: 200.0,
                    fir_band: String::from("high"),
                    ..Default::default()
                });
                config.optimizer.fir = Some(roomeq_model::FirConfig {
                    taps: 1024,
                    phase: String::from("minimum"),
                    ..Default::default()
                });
                let directory = tempfile::tempdir().unwrap();
                let result =
                    crate::optimize_room(&config, 48000.0, None, Some(directory.path())).unwrap();
                let graph = result.to_dsp_chain_output();
                let rendered = serde_json::to_vec(&graph).unwrap();
                let decoded: roomeq_model::DspGraph = serde_json::from_slice(&rendered).unwrap();
                let chain = &decoded.channels["left"];
                let adjudication = &decoded
                    .metadata
                    .as_ref()
                    .unwrap()
                    .veto_adjudication
                    .as_ref()
                    .expect("hybrid must retain pruning evidence")["left"];
                assert_eq!(adjudication.enforced, !report_only);
                assert_eq!(adjudication.removed_filter_indices.is_empty(), report_only);
                assert_eq!(
                    chain
                        .plugins
                        .iter()
                        .any(|plugin| plugin.plugin_type == "eq"),
                    report_only,
                    "hybrid graph: {decoded:#?}"
                );
                assert_eq!(
                    chain
                        .plugins
                        .iter()
                        .any(|plugin| plugin.plugin_type == "band_split"),
                    crossover
                );
                assert_eq!(
                    chain
                        .plugins
                        .iter()
                        .any(|plugin| plugin.plugin_type == "band_merge"),
                    crossover
                );
                let convolution = chain
                    .plugins
                    .iter()
                    .find(|plugin| plugin.plugin_type == "convolution")
                    .expect("hybrid must export its FIR branch");
                let reference = convolution.parameters["ir_file"].as_str().unwrap();
                let path = directory.path().join(reference);
                let mut reader = hound::WavReader::open(path).unwrap();
                assert_eq!(reader.spec().sample_rate, 48000);
                assert_eq!(reader.spec().channels, 1);
                let samples: Vec<_> = reader.samples::<f32>().map(Result::unwrap).collect();
                let expected = result.channel_results["left"].fir_coeffs.as_ref().unwrap();
                assert_eq!(samples.len(), expected.len());
                for (sample, expected) in samples.iter().zip(expected) {
                    assert_eq!(*sample, *expected as f32);
                }
            }
        }
    }
}

#[test]
fn qa_roomeq_pruning_conditions_hybrid_missing_seat_phase_retains_safety_fallback() {
    let mut config = config(2, true);
    let mut measured = measurement(0);
    measured.phase = Some(Array1::zeros(measured.freq.len()));
    config.speakers.insert(
        String::from("left"),
        SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![
            measured,
            measurement(1),
        ])),
    );
    config.optimizer.parallel_threads = Some(1);
    config.optimizer.processing_mode = roomeq_model::ProcessingMode::Hybrid;
    config.optimizer.fir = Some(roomeq_model::FirConfig {
        taps: 1024,
        phase: String::from("minimum"),
        ..Default::default()
    });
    let directory = tempfile::tempdir().unwrap();
    let result = crate::optimize_room(&config, 48000.0, None, Some(directory.path())).unwrap();
    let graph = result.to_dsp_chain_output();
    assert!(
        graph.channels["left"]
            .plugins
            .iter()
            .all(|plugin| { !matches!(plugin.plugin_type.as_str(), "eq" | "convolution") })
    );
    let report = graph
        .metadata
        .as_ref()
        .unwrap()
        .correction_acceptance
        .as_ref()
        .unwrap();
    assert!(
        !report
            .acoustic_quality
            .as_ref()
            .unwrap()
            .temporal
            .phase_evidence_available
    );
}

#[test]
fn qa_roomeq_pruning_conditions_exported_all_channel_multiseat_matrix() {
    for report_only in [true, false] {
        let mut config = config(2, report_only);
        config.optimizer.parallel_threads = Some(1);
        config.optimizer.multi_seat = Some(roomeq_model::MultiSeatConfig {
            enabled: true,
            all_channel_enabled: true,
            seat_weights: Some(vec![1.0, 0.0]),
            ..Default::default()
        });
        config
            .speakers
            .insert(String::from("right"), config.speakers["left"].clone());
        config.system = Some(roomeq_model::SystemConfig {
            model: roomeq_model::SystemModel::HomeCinema,
            speakers: HashMap::from([
                (String::from("L"), String::from("left")),
                (String::from("R"), String::from("right")),
            ]),
            ..Default::default()
        });
        let directory = tempfile::tempdir().unwrap();
        let result = crate::optimize_room(&config, 48000.0, None, Some(directory.path())).unwrap();
        let graph = result.to_dsp_chain_output();
        let roundtrip =
            roomeq_export::roundtrip::verify_biquad_json_roundtrip(&graph, 48000.0, 1e-10).unwrap();
        let metadata = graph.metadata.as_ref().unwrap();
        assert!(metadata.multi_seat_correction.is_some());
        let reports = metadata
            .veto_adjudication
            .as_ref()
            .expect("all-channel multi-seat pruning must retain evidence");
        for role in ["L", "R"] {
            let report = &reports[role];
            assert_eq!(report.enforced, !report_only);
            assert_eq!(report.removed_filter_indices.is_empty(), report_only);
            assert!(report.f0_reference_id.contains("conditions-v1:"));
        }
        assert_eq!(roundtrip.sections == 0, !report_only);
    }
}

#[test]
fn qa_roomeq_pruning_conditions_exported_held_out_matrix() {
    for held_seats in [1, 2] {
        let mut advisory_plugins = None;
        for report_only in [true, false] {
            let mut config = config(2, report_only);
            config.optimizer.parallel_threads = Some(1);
            config.optimizer.max_iter = 100;
            config.optimizer.min_db = -0.1;
            config.optimizer.max_db = 0.1;
            let curve = |seat| {
                let mut curve = measurement(seat);
                curve.spl *= 0.05;
                curve
            };
            config.speakers.insert(
                String::from("left"),
                SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(
                    (0..2).map(curve).collect(),
                )),
            );
            let directory = tempfile::tempdir().unwrap();
            let result = crate::RoomPipeline::new(crate::RoomPipelineRequest {
                config: &config,
                sample_rate: 48000.0,
                output_dir: Some(directory.path()),
                probe_arrival_overrides: None,
            })
            .with_validation_measurements(HashMap::from([(
                String::from("left"),
                (0..held_seats).map(curve).collect(),
            )]))
            .run(None)
            .unwrap();
            let graph = result.to_dsp_chain_output();
            let metadata = graph.metadata.as_ref().unwrap();
            let report = &metadata.veto_adjudication.as_ref().unwrap()["final_routed_graph"];
            let verdicts = &metadata.audibility_veto.as_ref().unwrap()["final_routed_graph"];
            assert!(!verdicts.is_empty());
            assert_eq!(
                report.enforced,
                !report_only && held_seats == 2,
                "{verdicts:#?}"
            );
            let plugins = serde_json::to_value(&graph.channels["left"].plugins).unwrap();
            if report_only {
                advisory_plugins = Some(plugins.clone());
            }
            if held_seats == 1 {
                assert!(report.removed_filter_indices.is_empty());
                assert_eq!(plugins, *advisory_plugins.as_ref().unwrap());
                assert!(verdicts.iter().all(|verdict| verdict.acceptance.outcome
                    == roomeq_model::ReportOutcome::InsufficientEvidence));
            } else {
                assert!(
                    verdicts
                        .iter()
                        .any(|verdict| verdict.acceptance.reason.contains("held_out")),
                    "{verdicts:#?}"
                );
                assert_eq!(report.removed_filter_indices.is_empty(), report_only);
            }
            roomeq_export::roundtrip::verify_biquad_json_roundtrip(&graph, 48000.0, 1e-10).unwrap();
        }
    }
}

#[test]
fn qa_roomeq_pruning_conditions_exported_routed_matrix() {
    for (multi_sub, grouped) in [(false, false), (true, false), (true, true)] {
        let mut frozen_reference = None;
        let mut frozen_graph = None;
        for report_only in [true, false] {
            let mut config = config(2, report_only);
            config.optimizer.parallel_threads = Some(1);
            config.optimizer.max_iter = 100;
            config.optimizer.min_db = -0.1;
            config.optimizer.max_db = 0.1;
            // Two correlated programme inputs have declared quarter-scale peaks;
            // the unchanged 0 dBFS output ceiling still applies to their sum.
            config.optimizer.finalization.default_input_peak = 0.25;
            let source = || {
                MeasurementSource::InMemoryMultiple(
                    (0..2)
                        .map(|seat| {
                            let mut curve = measurement(seat);
                            curve.spl *= 0.05;
                            curve.phase = Some(Array1::zeros(curve.freq.len()));
                            curve
                        })
                        .collect(),
                )
            };
            config.speakers = HashMap::from([
                (String::from("left"), SpeakerConfig::Single(source())),
                (String::from("right"), SpeakerConfig::Single(source())),
                (String::from("sub"), SpeakerConfig::Single(source())),
            ]);
            if multi_sub {
                let sub_source = || {
                    let MeasurementSource::InMemoryMultiple(mut curves) = source() else {
                        unreachable!()
                    };
                    for curve in &mut curves {
                        // Two coherent half-pressure subs have the same uncorrected
                        // combined level as the single-sub fixture.
                        curve.spl -= 20.0 * 2.0_f64.log10();
                    }
                    MeasurementSource::InMemoryMultiple(curves)
                };
                config.speakers.remove("sub");
                for index in 0..2 {
                    config
                        .speakers
                        .insert(format!("sub-{index}"), SpeakerConfig::Single(sub_source()));
                }
                if grouped {
                    let subwoofers = (0..2)
                        .map(|index| {
                            let SpeakerConfig::Single(source) =
                                config.speakers.remove(&format!("sub-{index}")).unwrap()
                            else {
                                unreachable!()
                            };
                            source
                        })
                        .collect();
                    config.speakers.insert(
                        String::from("sub"),
                        SpeakerConfig::MultiSub(roomeq_model::MultiSubGroup {
                            name: String::from("subs"),
                            speaker_name: None,
                            subwoofers,
                            allpass_optimization: false,
                        }),
                    );
                }
            }
            config.system = Some(roomeq_model::SystemConfig {
                model: roomeq_model::SystemModel::Stereo,
                speakers: HashMap::from([
                    (String::from("L"), String::from("left")),
                    (String::from("R"), String::from("right")),
                ]),
                subwoofers: Some(roomeq_model::SubwooferSystemConfig {
                    config: if multi_sub {
                        roomeq_model::SubwooferStrategy::Mso
                    } else {
                        roomeq_model::SubwooferStrategy::Single
                    },
                    crossover: Some(roomeq_model::SubwooferCrossoverRef::PerSub(vec![
                    String::from("bass"); if multi_sub { 2 } else { 1 }
                ])),
                    routing: Default::default(),
                    outputs: (0..if multi_sub { 2 } else { 1 })
                        .map(|index| roomeq_model::SubwooferOutput {
                            id: format!("sub-{index}"),
                            speaker: if multi_sub && !grouped {
                                format!("sub-{index}")
                            } else {
                                String::from("sub")
                            },
                        })
                        .collect(),
                }),
                ..Default::default()
            });
            config.crossovers = Some(HashMap::from([(
                String::from("bass"),
                roomeq_model::CrossoverConfig {
                    crossover_type: String::from("LR24"),
                    frequency: Some(80.0),
                    frequency_range: None,
                    frequencies: None,
                },
            )]));
            let directory = tempfile::tempdir().unwrap();
            let result =
                crate::optimize_room(&config, 48000.0, None, Some(directory.path())).unwrap();
            let graph = result.to_dsp_chain_output();
            let metadata = graph.metadata.as_ref().unwrap();
            let report = &metadata
                .veto_adjudication
                .as_ref()
                .expect("routed evidence must survive export")["final_routed_graph"];
            let verdicts = &metadata.audibility_veto.as_ref().unwrap()["final_routed_graph"];
            if let Some(reference) = &frozen_reference {
                assert_eq!(&report.f0_reference_id, reference);
            } else {
                frozen_reference = Some(report.f0_reference_id.clone());
            }
            for (name, channel) in &result.channel_results {
                let chain = &graph.channels[name];
                for filter in &channel.biquads {
                    let filter = roomeq_engine::output::biquad_to_json(filter);
                    assert!(
                        chain.plugins.iter().any(|plugin| {
                            plugin.plugin_type == "eq"
                                && plugin.parameters["filters"]
                                    .as_array()
                                    .is_some_and(|filters| filters.contains(&filter))
                        }),
                        "stale filter metadata for {name}: {filter}"
                    );
                }
            }
            assert!(
                !verdicts.is_empty(),
                "routed row must exercise eligible emitted filters"
            );
            assert_eq!(
                report.enforced, !report_only,
                "{:?}",
                metadata.stage_outcomes
            );
            assert_eq!(
                report.removed_filter_indices.is_empty(),
                report_only,
                "{verdicts:#?}"
            );
            if report_only {
                frozen_graph = Some(serde_json::to_value(&graph).unwrap());
            } else {
                let mut restored = serde_json::to_value(&graph).unwrap();
                // Reinsert in original inventory order so earlier removals do
                // not shift the original positions of later filters.
                for &index in &report.removed_filter_indices {
                    let (_, location) = verdicts[index]
                        .acceptance
                        .reason
                        .rsplit_once("; original_location=")
                        .unwrap();
                    let location: serde_json::Value = serde_json::from_str(location).unwrap();
                    let channel = &mut restored["channels"][location["channel"].as_str().unwrap()];
                    let plugins = if let Some(driver) = location["driver"].as_u64() {
                        &mut channel["drivers"][driver as usize]["plugins"]
                    } else {
                        &mut channel["plugins"]
                    };
                    plugins[location["plugin"].as_u64().unwrap() as usize]["parameters"]["filters"]
                        .as_array_mut()
                        .unwrap()
                        .insert(
                            location["filter"].as_u64().unwrap() as usize,
                            location["original"].clone(),
                        );
                }
                let original = frozen_graph.as_ref().unwrap();
                for (name, channel) in restored["channels"].as_object().unwrap() {
                    assert_eq!(channel["plugins"], original["channels"][name]["plugins"]);
                    if let Some(drivers) = channel["drivers"].as_array() {
                        for (index, driver) in drivers.iter().enumerate() {
                            assert_eq!(
                                driver["plugins"],
                                original["channels"][name]["drivers"][index]["plugins"]
                            );
                        }
                    }
                }
            }
            assert!(
                verdicts
                    .iter()
                    .any(|verdict| verdict.acceptance.reason.contains("correlated-inputs")),
                "{verdicts:#?}"
            );
            assert!(
                verdicts.iter().all(|verdict| verdict.acceptance.confidence
                    == roomeq_model::AssessmentConfidence::Low)
            );
            // Native JSON preserves the matrix and parallel outputs. A serial
            // biquad export cannot represent this playback contract.
            let encoded = serde_json::to_vec(&graph).unwrap();
            let decoded: roomeq_model::DspGraph = serde_json::from_slice(&encoded).unwrap();
            decoded.validate().unwrap();
            assert_eq!(
                serde_json::to_value(&decoded).unwrap(),
                serde_json::to_value(&graph).unwrap()
            );
            let mut delivered = result.clone();
            delivered.channels = decoded.channels;
            delivered.metadata = decoded.metadata.unwrap();
            let captures =
                crate::room_optimization::seat_replay::capture_training(&config).unwrap();
            let physical = crate::room_optimization::seat_replay::training_physical_captures(
                &captures, &result,
            )
            .unwrap();
            for input in ["L", "R"] {
                for seat in 0..2 {
                    let replay = |graph| {
                        crate::room_optimization::seat_replay::replay_final_physical_seat(
                            graph,
                            &physical,
                            input,
                            seat,
                            &config,
                            48000.0,
                            directory.path(),
                            "training",
                        )
                        .unwrap()
                    };
                    let expected = replay(&result);
                    let actual = replay(&delivered);
                    assert_eq!(actual.physical_outputs, expected.physical_outputs);
                    assert_eq!(actual.delivered.freq, expected.delivered.freq);
                    assert_eq!(actual.delivered.spl, expected.delivered.spl);
                    assert_eq!(actual.delivered.phase, expected.delivered.phase);
                }
            }
            assert!(
                metadata
                    .bass_management
                    .as_ref()
                    .unwrap()
                    .routing_graph
                    .is_some()
            );
        }
    }
}
