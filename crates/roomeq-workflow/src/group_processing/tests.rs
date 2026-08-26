use super::{process_cardioid, process_multisub_group};
use roomeq_engine::Curve;
use std::collections::HashMap;
use std::path::Path;

use ndarray::array;
use roomeq_model::{
    MeasurementSource, MultiSeatConfig, MultiSubGroup, OptimizerConfig, RoomConfig,
};

fn flat_curve_without_phase() -> Curve {
    Curve {
        freq: array![40.0, 80.0, 160.0],
        spl: array![80.0, 80.0, 80.0],
        phase: None,
        ..Default::default()
    }
}

#[test]
fn cardioid_rejects_missing_phase() {
    let cardioid = roomeq_model::CardioidConfig {
        name: "card".to_string(),
        speaker_name: None,
        front: MeasurementSource::InMemory(flat_curve_without_phase()),
        rear: MeasurementSource::InMemory(flat_curve_without_phase()),
        separation_meters: 0.5,
    };
    let room_config = RoomConfig {
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

    let err =
        process_cardioid("LFE", &cardioid, &room_config, 48000.0, Path::new(".")).unwrap_err();
    assert!(
        err.to_string().contains("requires measured phase"),
        "unexpected error: {err}"
    );
}

#[test]
fn cardioid_rejects_rear_measurement_that_does_not_cover_front_span() {
    let front = Curve {
        freq: array![100.0, 200.0, 400.0, 800.0],
        spl: array![80.0, 80.0, 80.0, 80.0],
        phase: Some(array![0.0, 0.0, 0.0, 0.0]),
        ..Default::default()
    };
    let rear = Curve {
        freq: array![200.0, 400.0],
        spl: array![80.0, 80.0],
        phase: Some(array![0.0, 0.0]),
        ..Default::default()
    };
    let cardioid = roomeq_model::CardioidConfig {
        name: "card".to_string(),
        speaker_name: None,
        front: MeasurementSource::InMemory(front),
        rear: MeasurementSource::InMemory(rear),
        separation_meters: 0.5,
    };

    let error = process_cardioid(
        "LFE",
        &cardioid,
        &RoomConfig::default(),
        48_000.0,
        Path::new("."),
    )
    .unwrap_err();
    assert!(error.to_string().contains("full front frequency span"));
}

#[test]
fn cardioid_flat_response_does_not_regress() {
    // Front and rear are identical flat curves with measured phase.
    // The cardioid sum will be flat-ish; global EQ should not regress it.
    let front = Curve {
        freq: array![100.0, 200.0, 400.0, 800.0],
        spl: array![80.0, 80.0, 80.0, 80.0],
        phase: Some(array![0.0, 0.0, 0.0, 0.0]),
        ..Default::default()
    };
    let rear = Curve {
        freq: array![100.0, 200.0, 400.0, 800.0],
        spl: array![80.0, 80.0, 80.0, 80.0],
        phase: Some(array![0.0, 0.0, 0.0, 0.0]),
        ..Default::default()
    };
    let cardioid = roomeq_model::CardioidConfig {
        name: "card".to_string(),
        speaker_name: None,
        front: MeasurementSource::InMemory(front),
        rear: MeasurementSource::InMemory(rear),
        separation_meters: 0.5,
    };
    let room_config = RoomConfig {
        version: roomeq_model::default_config_version(),
        system: None,
        speakers: HashMap::new(),
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig {
            min_freq: 100.0,
            max_freq: 800.0,
            num_filters: 1,
            max_iter: 10,
            population: 4,
            seed: Some(42),
            ..Default::default()
        },
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let result = process_cardioid("LFE", &cardioid, &room_config, 48000.0, Path::new("."));
    assert!(
        result.is_ok(),
        "Cardioid processing should succeed: {:?}",
        result
    );
    let (_chain, _pre, post, _initial, _final, _filters, _mean, _arrival, _fir, _evidence) =
        result.unwrap();
    assert!(
        post.is_finite(),
        "post_score must be finite after regression guard"
    );
}

fn phased_sub_curve(spl_offset: f64, phase_offset: f64) -> Curve {
    let freq = array![20.0, 30.0, 45.0, 67.5, 100.0, 120.0];
    let spl = freq.mapv(|f| {
        let mode = if f < 60.0 { 3.0 } else { -1.0 };
        80.0 + spl_offset + mode
    });
    let phase = freq.mapv(|f| -180.0 * f / 100.0 + phase_offset);
    Curve {
        freq,
        spl,
        phase: Some(phase),
        ..Default::default()
    }
}

#[test]
fn multisub_uses_production_multiseat_path_when_subs_have_seat_measurements() {
    let group = MultiSubGroup {
        name: "subs".to_string(),
        speaker_name: None,
        subwoofers: vec![
            MeasurementSource::InMemoryMultiple(vec![
                phased_sub_curve(0.0, 0.0),
                phased_sub_curve(2.0, 12.0),
            ]),
            MeasurementSource::InMemoryMultiple(vec![
                phased_sub_curve(-1.0, 45.0),
                phased_sub_curve(1.0, 60.0),
            ]),
        ],
        allpass_optimization: false,
    };
    let room_config = RoomConfig {
        version: roomeq_model::default_config_version(),
        system: None,
        speakers: HashMap::new(),
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig {
            min_freq: 20.0,
            max_freq: 120.0,
            num_filters: 1,
            max_iter: 3,
            population: 4,
            seed: Some(7),
            refine: false,
            multi_seat: Some(MultiSeatConfig {
                enabled: true,
                per_sub_peq: false,
                global_eq: false,
                ..Default::default()
            }),
            ..OptimizerConfig::default()
        },
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let (
        chain,
        pre_score,
        post_score,
        initial,
        final_curve,
        filters,
        _mean,
        _arrival,
        _fir,
        _evidence,
    ) = process_multisub_group("LFE", &group, &room_config, 48000.0, Path::new("."))
        .expect("multi-seat multi-sub processing should succeed");

    assert!(pre_score.is_finite());
    assert!(post_score.is_finite());
    assert_ne!(
        pre_score, post_score,
        "pre/post scores should include the production MSO stage, not only global EQ"
    );
    assert!(
        filters.is_empty(),
        "global_eq=false should not emit shared EQ"
    );
    assert!(chain.plugins.is_empty());
    assert!(initial.phase.is_some());
    assert!(final_curve.phase.is_some());
    assert!(
        chain
            .initial_curve
            .as_ref()
            .is_some_and(|curve| curve.phase.is_none()),
        "reported spatial aggregate must remain magnitude-only"
    );
    let drivers = chain.drivers.expect("multi-sub output should have drivers");
    assert_eq!(drivers.len(), 2);
}

#[test]
fn production_multiseat_path_emits_per_sub_and_global_eq_when_enabled() {
    let group = MultiSubGroup {
        name: "subs".to_string(),
        speaker_name: None,
        subwoofers: vec![
            MeasurementSource::InMemoryMultiple(vec![
                phased_sub_curve(0.0, 0.0),
                phased_sub_curve(2.0, 12.0),
            ]),
            MeasurementSource::InMemoryMultiple(vec![
                phased_sub_curve(-1.0, 45.0),
                phased_sub_curve(1.0, 60.0),
            ]),
        ],
        allpass_optimization: false,
    };
    let room_config = RoomConfig {
        version: roomeq_model::default_config_version(),
        system: None,
        speakers: HashMap::new(),
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig {
            min_freq: 20.0,
            max_freq: 120.0,
            num_filters: 1,
            max_iter: 3,
            population: 4,
            seed: Some(11),
            refine: false,
            multi_seat: Some(MultiSeatConfig {
                enabled: true,
                per_sub_peq: true,
                global_eq: true,
                ..Default::default()
            }),
            ..OptimizerConfig::default()
        },
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let (chain, pre_score, post_score, _initial, _final, filters, _mean, _arrival, _fir, _evidence) =
        process_multisub_group("LFE", &group, &room_config, 48000.0, Path::new("."))
            .expect("multi-seat multi-sub processing should succeed");

    assert!(pre_score.is_finite());
    assert!(post_score.is_finite());
    let has_global_eq = chain
        .plugins
        .iter()
        .any(|plugin| plugin.plugin_type == "eq");
    assert_eq!(
        has_global_eq,
        !filters.is_empty(),
        "shared EQ filters and exported channel EQ plugin should stay in sync"
    );
    let drivers = chain.drivers.expect("multi-sub output should have drivers");
    assert_eq!(drivers.len(), 2);
    assert!(
        drivers.iter().all(|driver| driver
            .plugins
            .iter()
            .any(|plugin| plugin.plugin_type == "eq")),
        "per_sub_peq=true should export per-driver EQ plugins"
    );
}

mod coverage_tests {
    use super::super::{
        process_dba, process_multisub_group, process_speaker_group, process_speaker_topology,
    };
    use ndarray::array;
    use roomeq_engine::Curve;
    use roomeq_model::DBAConfig;
    use roomeq_model::MeasurementSource;
    use roomeq_model::MultiSubGroup;
    use roomeq_model::OptimizerConfig;
    use roomeq_model::RoomConfig;
    use roomeq_model::SpeakerGroup;
    use roomeq_model::{
        DriverCrossoverBand, ParallelDriverGroup, SpeakerDriver, SpeakerDriverRole, SpeakerTopology,
    };
    use std::collections::HashMap;
    use std::path::Path;

    fn flat_curve() -> Curve {
        Curve {
            freq: array![100.0, 200.0, 400.0, 800.0, 1600.0],
            spl: array![80.0, 80.0, 80.0, 80.0, 80.0],
            phase: None,
            ..Default::default()
        }
    }

    fn sub_optimizer() -> OptimizerConfig {
        OptimizerConfig {
            min_freq: 20.0,
            max_freq: 160.0,
            num_filters: 1,
            max_iter: 3,
            population: 4,
            seed: Some(1),
            refine: false,
            ..Default::default()
        }
    }

    fn room_config_with_optimizer(optimizer: OptimizerConfig) -> RoomConfig {
        RoomConfig {
            version: roomeq_model::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        }
    }

    #[test]
    fn process_speaker_group_rejects_missing_crossover() {
        let group = SpeakerGroup {
            name: "test".to_string(),
            speaker_name: None,
            measurements: vec![MeasurementSource::InMemory(flat_curve())],
            crossover: None,
        };
        let config = room_config_with_optimizer(OptimizerConfig::default());
        let err = process_speaker_group("L", &group, &config, 48000.0, Path::new(".")).unwrap_err();
        assert!(err.to_string().contains("requires crossover configuration"));
    }

    #[test]
    fn process_speaker_group_rejects_unknown_crossover() {
        let group = SpeakerGroup {
            name: "test".to_string(),
            speaker_name: None,
            measurements: vec![MeasurementSource::InMemory(flat_curve())],
            crossover: Some("missing".to_string()),
        };
        let config = room_config_with_optimizer(OptimizerConfig::default());
        let err = process_speaker_group("L", &group, &config, 48000.0, Path::new(".")).unwrap_err();
        assert!(err.to_string().contains("Crossover configuration"));
    }

    #[test]
    fn process_speaker_group_two_way_succeeds() {
        let mut woofer = flat_curve();
        woofer.freq = array![50.0, 100.0, 200.0, 400.0];
        woofer.spl = array![80.0, 80.0, 80.0, 80.0];
        let mut tweeter = flat_curve();
        tweeter.freq = array![1000.0, 2000.0, 4000.0, 8000.0];
        tweeter.spl = array![80.0, 80.0, 80.0, 80.0];
        let group = SpeakerGroup {
            name: "test".to_string(),
            speaker_name: None,
            measurements: vec![
                MeasurementSource::InMemory(tweeter),
                MeasurementSource::InMemory(woofer),
            ],
            crossover: Some("xover".to_string()),
        };
        let mut config = room_config_with_optimizer(OptimizerConfig {
            min_freq: 50.0,
            max_freq: 8000.0,
            num_filters: 1,
            max_iter: 3,
            population: 4,
            seed: Some(3),
            refine: false,
            ..Default::default()
        });
        config.crossovers = Some(HashMap::from([(
            "xover".to_string(),
            roomeq_model::CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: Some(800.0),
                frequencies: None,
                frequency_range: None,
            },
        )]));
        let result = process_speaker_group("L", &group, &config, 48000.0, Path::new("."));
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn explicit_parallel_woofers_share_one_acoustic_crossover_band() {
        let mut woofer = flat_curve();
        woofer.phase = Some(array![0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut tweeter = flat_curve();
        tweeter.phase = Some(array![0.0, 0.0, 0.0, 0.0, 0.0]);
        let topology = SpeakerTopology {
            name: "parallel-two-way".to_string(),
            speaker_name: None,
            drivers: vec![
                SpeakerDriver {
                    id: "woofer_a".to_string(),
                    role: SpeakerDriverRole::Woofer,
                    measurement: MeasurementSource::InMemory(woofer.clone()),
                    crossover_band: Some(DriverCrossoverBand {
                        min_hz: 100.0,
                        max_hz: 1_600.0,
                    }),
                },
                SpeakerDriver {
                    id: "woofer_b".to_string(),
                    role: SpeakerDriverRole::Woofer,
                    measurement: MeasurementSource::InMemory(woofer),
                    crossover_band: Some(DriverCrossoverBand {
                        min_hz: 100.0,
                        max_hz: 1_600.0,
                    }),
                },
                SpeakerDriver {
                    id: "tweeter".to_string(),
                    role: SpeakerDriverRole::Tweeter,
                    measurement: MeasurementSource::InMemory(tweeter),
                    crossover_band: Some(DriverCrossoverBand {
                        min_hz: 400.0,
                        max_hz: 1_600.0,
                    }),
                },
            ],
            parallel_groups: vec![ParallelDriverGroup {
                id: "woofer_pair".to_string(),
                driver_ids: vec!["woofer_a".to_string(), "woofer_b".to_string()],
            }],
            crossover: Some("xover".to_string()),
        };
        let mut config = room_config_with_optimizer(OptimizerConfig {
            min_freq: 100.0,
            max_freq: 1_600.0,
            num_filters: 1,
            max_iter: 3,
            population: 4,
            seed: Some(3),
            refine: false,
            ..Default::default()
        });
        config.crossovers = Some(HashMap::from([(
            "xover".to_string(),
            roomeq_model::CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: Some(800.0),
                frequencies: None,
                frequency_range: None,
            },
        )]));

        let (chain, ..) =
            process_speaker_topology("L", &topology, &config, 48_000.0, Path::new(".")).unwrap();
        let drivers = chain.drivers.unwrap();
        assert_eq!(
            drivers
                .iter()
                .map(|driver| driver.name.as_str())
                .collect::<Vec<_>>(),
            vec!["woofer_a", "woofer_b", "tweeter"]
        );
        assert!(
            drivers[0]
                .plugins
                .iter()
                .any(|plugin| plugin.plugin_type == "crossover")
        );
        assert!(
            drivers[1]
                .plugins
                .iter()
                .any(|plugin| plugin.plugin_type == "crossover")
        );
        assert!(
            drivers[2]
                .plugins
                .iter()
                .any(|plugin| plugin.plugin_type == "crossover")
        );
    }

    #[test]
    fn parallel_only_topology_does_not_require_a_crossover() {
        let topology = SpeakerTopology {
            name: "woofer-array".to_string(),
            speaker_name: None,
            drivers: ["woofer_a", "woofer_b"]
                .into_iter()
                .map(|id| SpeakerDriver {
                    id: id.to_string(),
                    role: SpeakerDriverRole::Woofer,
                    measurement: MeasurementSource::InMemory(phased_curve()),
                    crossover_band: Some(DriverCrossoverBand {
                        min_hz: 100.0,
                        max_hz: 1_600.0,
                    }),
                })
                .collect(),
            parallel_groups: vec![ParallelDriverGroup {
                id: "woofer_pair".to_string(),
                driver_ids: vec!["woofer_a".to_string(), "woofer_b".to_string()],
            }],
            crossover: None,
        };
        let config = room_config_with_optimizer(OptimizerConfig {
            min_freq: 100.0,
            max_freq: 1_600.0,
            num_filters: 1,
            max_iter: 3,
            population: 4,
            seed: Some(4),
            refine: false,
            ..Default::default()
        });
        let (chain, ..) =
            process_speaker_topology("L", &topology, &config, 48_000.0, Path::new(".")).unwrap();
        assert!(chain.drivers.unwrap().iter().all(|driver| {
            driver
                .plugins
                .iter()
                .all(|plugin| plugin.plugin_type != "crossover")
        }));
    }

    #[test]
    fn parallel_missing_phase_disables_relative_temporal_controls() {
        let topology = SpeakerTopology {
            name: "woofer-array-no-phase".to_string(),
            speaker_name: None,
            drivers: ["woofer_a", "woofer_b"]
                .into_iter()
                .map(|id| SpeakerDriver {
                    id: id.to_string(),
                    role: SpeakerDriverRole::Woofer,
                    measurement: MeasurementSource::InMemory(flat_curve()),
                    crossover_band: Some(DriverCrossoverBand {
                        min_hz: 100.0,
                        max_hz: 1_600.0,
                    }),
                })
                .collect(),
            parallel_groups: vec![ParallelDriverGroup {
                id: "woofer_pair".to_string(),
                driver_ids: vec!["woofer_a".to_string(), "woofer_b".to_string()],
            }],
            crossover: None,
        };
        let config = room_config_with_optimizer(OptimizerConfig {
            min_freq: 100.0,
            max_freq: 1_600.0,
            num_filters: 1,
            max_iter: 3,
            population: 4,
            seed: Some(5),
            refine: false,
            ..Default::default()
        });
        let (chain, ..) =
            process_speaker_topology("L", &topology, &config, 48_000.0, Path::new(".")).unwrap();
        assert!(chain.drivers.unwrap().iter().all(|driver| {
            driver
                .plugins
                .iter()
                .all(|plugin| plugin.plugin_type != "delay" && plugin.plugin_type != "gain")
        }));
    }

    #[test]
    fn process_multisub_group_standard_path_succeeds() {
        let group = MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers: vec![
                MeasurementSource::InMemory(flat_curve()),
                MeasurementSource::InMemory(flat_curve()),
            ],
            allpass_optimization: false,
        };
        let config = room_config_with_optimizer(sub_optimizer());
        let result = process_multisub_group("LFE", &group, &config, 48000.0, Path::new("."));
        assert!(result.is_ok(), "{result:?}");
    }

    fn phased_curve() -> Curve {
        Curve {
            freq: array![100.0, 200.0, 400.0, 800.0, 1600.0],
            spl: array![80.0, 80.0, 80.0, 80.0, 80.0],
            phase: Some(array![0.0, 0.0, 0.0, 0.0, 0.0]),
            ..Default::default()
        }
    }

    #[test]
    fn process_dba_succeeds_with_flat_arrays() {
        let dba = DBAConfig {
            name: "dba".to_string(),
            speaker_name: None,
            front: vec![MeasurementSource::InMemory(phased_curve())],
            rear: vec![MeasurementSource::InMemory(phased_curve())],
        };
        let config = room_config_with_optimizer(sub_optimizer());
        let result = process_dba("LFE", &dba, &config, 48000.0, Path::new("."));
        assert!(result.is_ok(), "{result:?}");
    }
}
