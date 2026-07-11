// ---------------------------------------------------------------------------
// Tests for roomeq/types/config/* modules that previously had little or no
// unit-test coverage.  Each test exercises construction, JSON round-tripping,
// and path-resolution helpers where applicable.
// ---------------------------------------------------------------------------

fn sample_single_source(path: &str, speaker_name: Option<&str>) -> MeasurementSource {
    MeasurementSource::Single(MeasurementSingle {
        measurement: MeasurementRef::Path(path.into()),
        speaker_name: speaker_name.map(|s| s.to_string()),
    })
}

#[test]
fn cardioid_config_roundtrip_and_resolve_paths() {
    let mut cfg = CardioidConfig {
        name: "sub-cardioid".into(),
        speaker_name: Some("Subwoofer X".into()),
        front: sample_single_source("front.csv", None),
        rear: sample_single_source("rear.csv", Some("Rear Sub")),
        separation_meters: 0.35,
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: CardioidConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, cfg.name);
    assert_eq!(back.speaker_name, cfg.speaker_name);
    assert_eq!(back.separation_meters, cfg.separation_meters);

    back.resolve_paths(PathBuf::from("/base").as_path());
    match &back.front {
        MeasurementSource::Single(s) => match &s.measurement {
            MeasurementRef::Path(p) => assert_eq!(p, &PathBuf::from("/base/front.csv")),
            other => panic!("expected path front, got {:?}", other),
        },
        other => panic!("expected single front, got {:?}", other),
    }
    match &back.rear {
        MeasurementSource::Single(s) => assert_eq!(s.speaker_name.as_deref(), Some("Rear Sub")),
        other => panic!("expected single rear, got {:?}", other),
    }
    // resolve_paths mutates in place; reuse the resolved value for coverage.
    cfg.resolve_paths(PathBuf::from("/another").as_path());
}

#[test]
fn ctc_config_default_and_roundtrip_and_resolve_paths() {
    let default = CtcConfig::default();
    assert!(!default.enabled);
    assert_eq!(default.matrix_source, "measured");
    assert_eq!(default.robustness, "average");
    assert!(default.include_room_eq_dsp);
    assert_eq!(default.fir_taps, 4096);
    assert_eq!(default.harmonic_suppression_harmonics, 5);
    assert!(default.measurements.is_none());
    assert!(default.hrtf.is_none());
    assert!(default.reference_sweep.is_none());

    let mut cfg = CtcConfig {
        enabled: true,
        measurements: Some(CtcMeasurementConfig {
            speakers: vec!["L".into(), "R".into()],
            mics: vec!["mic1".into()],
            head_positions: vec![CtcHeadPositionConfig {
                id: "center".into(),
                x: 0.0,
                y: 0.0,
                z: 0.0,
                yaw_deg: 0.0,
            }],
            files: vec![CtcMeasurementFileConfig {
                head_position: "center".into(),
                speaker: "L".into(),
                ir: Some("ir.wav".into()),
                raw_sweep: None,
                loopback: None,
            }],
        }),
        reference_sweep: Some("sweep.wav".into()),
        sweep_duration_s: Some(10.0),
        sweep_start_hz: Some(20.0),
        sweep_end_hz: Some(20000.0),
        ..Default::default()
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: CtcConfig = serde_json::from_str(&json).unwrap();
    assert!(back.enabled);
    assert!(back.measurements.is_some());
    assert_eq!(back.measurements.as_ref().unwrap().speakers, vec!["L", "R"]);
    assert_eq!(back.reference_sweep, Some(PathBuf::from("sweep.wav")));

    back.resolve_paths(PathBuf::from("/base").as_path());
    assert_eq!(back.reference_sweep, Some(PathBuf::from("/base/sweep.wav")));
    assert_eq!(
        back.measurements.as_ref().unwrap().files[0].ir,
        Some(PathBuf::from("/base/ir.wav"))
    );

    // Coverage for the hrtf path-resolution branch.
    cfg.hrtf = Some(CtcHrtfConfig {
        hrtf_file: "hrtf.sofa".into(),
        speakers: vec![CtcHrtfSpeakerConfig {
            speaker: "L".into(),
            azimuth_deg: -30.0,
            elevation_deg: 0.0,
            distance_m: 2.0,
        }],
    });
    cfg.resolve_paths(PathBuf::from("/base").as_path());
    assert_eq!(
        cfg.hrtf.as_ref().unwrap().hrtf_file,
        PathBuf::from("/base/hrtf.sofa")
    );
}

#[test]
fn ctc_hrtf_config_roundtrip_and_resolve_paths() {
    let cfg = CtcHrtfConfig {
        hrtf_file: "hrtf.sofa".into(),
        speakers: vec![
            CtcHrtfSpeakerConfig {
                speaker: "L".into(),
                azimuth_deg: -30.0,
                elevation_deg: 0.0,
                distance_m: 2.0,
            },
            CtcHrtfSpeakerConfig {
                speaker: "R".into(),
                azimuth_deg: 30.0,
                elevation_deg: 0.0,
                distance_m: 2.0,
            },
        ],
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: CtcHrtfConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.hrtf_file, cfg.hrtf_file);
    assert_eq!(back.speakers.len(), 2);
    assert_eq!(back.speakers[0].speaker, "L");

    back.resolve_paths(PathBuf::from("/base").as_path());
    assert_eq!(back.hrtf_file, PathBuf::from("/base/hrtf.sofa"));
}

#[test]
fn ctc_measurement_config_roundtrip_and_resolve_paths() {
    let mut cfg = CtcMeasurementConfig {
        speakers: vec!["L".into(), "R".into()],
        mics: vec!["mic1".into(), "mic2".into()],
        head_positions: vec![CtcHeadPositionConfig {
            id: "center".into(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            yaw_deg: 0.0,
        }],
        files: vec![CtcMeasurementFileConfig {
            head_position: "center".into(),
            speaker: "L".into(),
            ir: Some("ir.wav".into()),
            raw_sweep: Some("raw.wav".into()),
            loopback: Some("loopback.wav".into()),
        }],
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: CtcMeasurementConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.speakers, cfg.speakers);
    assert_eq!(back.mics, cfg.mics);
    assert_eq!(back.head_positions.len(), 1);

    back.resolve_paths(PathBuf::from("/base").as_path());
    let file = &back.files[0];
    assert_eq!(file.ir, Some(PathBuf::from("/base/ir.wav")));
    assert_eq!(file.raw_sweep, Some(PathBuf::from("/base/raw.wav")));
    assert_eq!(file.loopback, Some(PathBuf::from("/base/loopback.wav")));

    // Absolute paths should be left untouched.
    cfg.files[0].ir = Some("/abs/ir.wav".into());
    cfg.resolve_paths(PathBuf::from("/base").as_path());
    assert_eq!(cfg.files[0].ir, Some(PathBuf::from("/abs/ir.wav")));
}

#[test]
fn dba_config_roundtrip_and_resolve_paths() {
    let cfg = DBAConfig {
        name: "dba".into(),
        speaker_name: Some("Sub Array".into()),
        front: vec![
            sample_single_source("front0.csv", None),
            sample_single_source("front1.csv", None),
        ],
        rear: vec![sample_single_source("rear0.csv", None)],
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: DBAConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, cfg.name);
    assert_eq!(back.speaker_name, cfg.speaker_name);
    assert_eq!(back.front.len(), 2);
    assert_eq!(back.rear.len(), 1);

    back.resolve_paths(PathBuf::from("/base").as_path());
    match &back.front[0] {
        MeasurementSource::Single(s) => match &s.measurement {
            MeasurementRef::Path(p) => assert_eq!(p, &PathBuf::from("/base/front0.csv")),
            other => panic!("expected path, got {:?}", other),
        },
        other => panic!("expected single, got {:?}", other),
    }
}

#[test]
fn multi_sub_group_roundtrip_and_resolve_paths() {
    let cfg = MultiSubGroup {
        name: "subs".into(),
        speaker_name: Some("Sub".into()),
        subwoofers: vec![
            sample_single_source("sub1.csv", None),
            sample_single_source("sub2.csv", None),
        ],
        allpass_optimization: true,
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: MultiSubGroup = serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, cfg.name);
    assert_eq!(back.speaker_name, cfg.speaker_name);
    assert!(back.allpass_optimization);
    assert_eq!(back.subwoofers.len(), 2);

    back.resolve_paths(PathBuf::from("/base").as_path());
    match &back.subwoofers[1] {
        MeasurementSource::Single(s) => match &s.measurement {
            MeasurementRef::Path(p) => assert_eq!(p, &PathBuf::from("/base/sub2.csv")),
            other => panic!("expected path, got {:?}", other),
        },
        other => panic!("expected single, got {:?}", other),
    }
}

#[test]
fn speaker_group_roundtrip_and_resolve_paths() {
    let cfg = SpeakerGroup {
        name: "L-speaker".into(),
        speaker_name: Some("Model A".into()),
        measurements: vec![
            sample_single_source("woofer.csv", None),
            sample_single_source("tweeter.csv", None),
        ],
        crossover: Some("lr24".into()),
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: SpeakerGroup = serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, cfg.name);
    assert_eq!(back.speaker_name, cfg.speaker_name);
    assert_eq!(back.measurements.len(), 2);
    assert_eq!(back.crossover.as_deref(), Some("lr24"));

    back.resolve_paths(PathBuf::from("/base").as_path());
    match &back.measurements[0] {
        MeasurementSource::Single(s) => match &s.measurement {
            MeasurementRef::Path(p) => assert_eq!(p, &PathBuf::from("/base/woofer.csv")),
            other => panic!("expected path, got {:?}", other),
        },
        other => panic!("expected single, got {:?}", other),
    }
}

#[test]
fn explicit_speaker_topology_roundtrip_validates_parallel_groups() {
    let topology = SpeakerTopology {
        name: "L-speaker".into(),
        speaker_name: Some("Model P".into()),
        drivers: vec![
            SpeakerDriver {
                id: "woofer_left".into(),
                role: SpeakerDriverRole::Woofer,
                measurement: sample_single_source("woofer-left.csv", None),
                crossover_band: Some(DriverCrossoverBand {
                    min_hz: 30.0,
                    max_hz: 1_200.0,
                }),
            },
            SpeakerDriver {
                id: "woofer_right".into(),
                role: SpeakerDriverRole::Woofer,
                measurement: sample_single_source("woofer-right.csv", None),
                crossover_band: Some(DriverCrossoverBand {
                    min_hz: 30.0,
                    max_hz: 1_200.0,
                }),
            },
            SpeakerDriver {
                id: "tweeter".into(),
                role: SpeakerDriverRole::Tweeter,
                measurement: sample_single_source("tweeter.csv", None),
                crossover_band: Some(DriverCrossoverBand {
                    min_hz: 700.0,
                    max_hz: 20_000.0,
                }),
            },
        ],
        parallel_groups: vec![ParallelDriverGroup {
            id: "woofers".into(),
            driver_ids: vec!["woofer_left".into(), "woofer_right".into()],
        }],
        crossover: Some("lr24".into()),
    };
    topology.validate().unwrap();
    assert_eq!(
        topology.acoustic_bands().unwrap(),
        vec![vec![0, 1], vec![2]]
    );

    let json = serde_json::to_string(&SpeakerConfig::Topology(topology)).unwrap();
    let mut back: SpeakerConfig = serde_json::from_str(&json).unwrap();
    let SpeakerConfig::Topology(ref parsed) = back else {
        panic!("explicit drivers input must select the topology variant");
    };
    assert_eq!(parsed.drivers[0].id, "woofer_left");
    assert_eq!(parsed.parallel_groups[0].id, "woofers");
    back.resolve_paths(PathBuf::from("/base").as_path());
    let SpeakerConfig::Topology(parsed) = back else {
        unreachable!();
    };
    match &parsed.drivers[1].measurement {
        MeasurementSource::Single(source) => match &source.measurement {
            MeasurementRef::Path(path) => {
                assert_eq!(path, &PathBuf::from("/base/woofer-right.csv"));
            }
            other => panic!("expected path, got {other:?}"),
        },
        other => panic!("expected single source, got {other:?}"),
    }
}

#[test]
fn legacy_speaker_group_adapter_assigns_stable_driver_entries() {
    let group = SpeakerGroup {
        name: "legacy".into(),
        speaker_name: None,
        measurements: vec![
            sample_single_source("woofer.csv", None),
            sample_single_source("tweeter.csv", None),
        ],
        crossover: Some("lr24".into()),
    };
    let topology = group.to_legacy_topology();
    assert_eq!(topology.drivers[0].id, "legacy_driver_1");
    assert_eq!(topology.drivers[0].role, SpeakerDriverRole::Woofer);
    assert_eq!(topology.drivers[1].role, SpeakerDriverRole::Tweeter);
    assert!(topology.parallel_groups.is_empty());
}

#[test]
fn speaker_config_single_roundtrip_and_resolve_paths() {
    let cfg = SpeakerConfig::Single(sample_single_source("L.csv", Some("Model A")));
    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: SpeakerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.speaker_name(), Some("Model A"));

    back.resolve_paths(PathBuf::from("/base").as_path());
    match &back {
        SpeakerConfig::Single(s) => match s {
            MeasurementSource::Single(ms) => match &ms.measurement {
                MeasurementRef::Path(p) => assert_eq!(p, &PathBuf::from("/base/L.csv")),
                other => panic!("expected path, got {:?}", other),
            },
            other => panic!("expected single source, got {:?}", other),
        },
        other => panic!("expected single variant, got {:?}", other),
    }
}

#[test]
fn speaker_config_group_roundtrip_and_resolve_paths() {
    let group = SpeakerGroup {
        name: "L-speaker".into(),
        speaker_name: Some("Model A".into()),
        measurements: vec![sample_single_source("woofer.csv", None)],
        crossover: None,
    };
    let cfg = SpeakerConfig::Group(group);
    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: SpeakerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.speaker_name(), Some("Model A"));

    back.resolve_paths(PathBuf::from("/base").as_path());
    match &back {
        SpeakerConfig::Group(g) => {
            assert_eq!(g.name, "L-speaker");
        }
        other => panic!("expected group variant, got {:?}", other),
    }
}

#[test]
fn speaker_config_multi_sub_roundtrip_and_resolve_paths() {
    let multi = MultiSubGroup {
        name: "subs".into(),
        speaker_name: Some("Sub".into()),
        subwoofers: vec![sample_single_source("sub.csv", None)],
        allpass_optimization: false,
    };
    let cfg = SpeakerConfig::MultiSub(multi);
    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: SpeakerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.speaker_name(), Some("Sub"));

    back.resolve_paths(PathBuf::from("/base").as_path());
    match &back {
        SpeakerConfig::MultiSub(ms) => {
            assert_eq!(ms.name, "subs");
        }
        other => panic!("expected multi-sub variant, got {:?}", other),
    }
}

#[test]
fn speaker_config_dba_roundtrip_and_resolve_paths() {
    let dba = DBAConfig {
        name: "dba".into(),
        speaker_name: Some("DBA".into()),
        front: vec![sample_single_source("front.csv", None)],
        rear: vec![sample_single_source("rear.csv", None)],
    };
    let cfg = SpeakerConfig::Dba(dba);
    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: SpeakerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.speaker_name(), Some("DBA"));

    back.resolve_paths(PathBuf::from("/base").as_path());
    match &back {
        SpeakerConfig::Dba(d) => {
            assert_eq!(d.name, "dba");
        }
        other => panic!("expected dba variant, got {:?}", other),
    }
}

#[test]
fn speaker_config_cardioid_roundtrip_and_resolve_paths() {
    let cardioid = CardioidConfig {
        name: "cardioid".into(),
        speaker_name: Some("Cardioid".into()),
        front: sample_single_source("front.csv", None),
        rear: sample_single_source("rear.csv", None),
        separation_meters: 0.5,
    };
    let cfg = SpeakerConfig::Cardioid(Box::new(cardioid));
    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: SpeakerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.speaker_name(), Some("Cardioid"));

    back.resolve_paths(PathBuf::from("/base").as_path());
    match &back {
        SpeakerConfig::Cardioid(c) => {
            assert_eq!(c.name, "cardioid");
            assert_eq!(c.separation_meters, 0.5);
        }
        other => panic!("expected cardioid variant, got {:?}", other),
    }
}

#[test]
fn room_config_roundtrip_and_resolve_paths() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "L".into(),
        SpeakerConfig::Single(sample_single_source("L.csv", Some("Model A"))),
    );
    speakers.insert(
        "R".into(),
        SpeakerConfig::Single(sample_single_source("R.csv", None)),
    );

    let mut crossovers = HashMap::new();
    crossovers.insert(
        "L".into(),
        CrossoverConfig {
            crossover_type: "LR24".into(),
            frequency: Some(2000.0),
            frequencies: None,
            frequency_range: None,
        },
    );

    let mut system_speakers = HashMap::new();
    system_speakers.insert("L".into(), "L".into());
    system_speakers.insert("R".into(), "R".into());

    let mut cfg = RoomConfig {
        version: "1.0.0".into(),
        system: Some(SystemConfig {
            model: SystemModel::Stereo,
            speakers: system_speakers,
            subwoofers: None,
            bass_management: None,
            ..Default::default()
        }),
        speakers,
        crossovers: Some(crossovers),
        target_curve: Some(TargetCurveConfig::Predefined("flat".into())),
        optimizer: OptimizerConfig::default(),
        recording_config: None,
        ctc: Some(CtcConfig::default()),
        cea2034_cache: None,
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let mut back: RoomConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.version, cfg.version);
    assert!(back.system.is_some());
    assert_eq!(back.speakers.len(), 2);
    assert!(back.crossovers.is_some());
    assert!(back.ctc.is_some());
    match &back.target_curve {
        Some(TargetCurveConfig::Predefined(name)) => assert_eq!(name, "flat"),
        other => panic!("expected predefined target curve, got {:?}", other),
    }

    // Exercise the path-resolution branch by replacing the target with a Path
    // variant after deserialization (the untagged enum serializes strings as
    // Predefined, so we set it explicitly here).
    back.target_curve = Some(TargetCurveConfig::Path("target.csv".into()));
    back.resolve_paths(PathBuf::from("/base").as_path());
    match &back.target_curve {
        Some(TargetCurveConfig::Path(p)) => {
            assert_eq!(p, &PathBuf::from("/base/target.csv"))
        }
        other => panic!("expected path target curve, got {:?}", other),
    }
    match &back.speakers["L"] {
        SpeakerConfig::Single(s) => match s {
            MeasurementSource::Single(ms) => match &ms.measurement {
                MeasurementRef::Path(p) => assert_eq!(p, &PathBuf::from("/base/L.csv")),
                other => panic!("expected path, got {:?}", other),
            },
            other => panic!("expected single source, got {:?}", other),
        },
        other => panic!("expected single config, got {:?}", other),
    }

    // Absolute target path must remain unchanged.
    cfg.target_curve = Some(TargetCurveConfig::Path("/abs/target.csv".into()));
    cfg.resolve_paths(PathBuf::from("/base").as_path());
    match &cfg.target_curve {
        Some(TargetCurveConfig::Path(p)) => assert_eq!(p, &PathBuf::from("/abs/target.csv")),
        other => panic!("expected path target curve, got {:?}", other),
    }
}
