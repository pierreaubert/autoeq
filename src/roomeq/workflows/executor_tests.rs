// r2factor:facade — do not pass this file back into r2factor
// Integration tests for route-specific workflow executors.

use super::generic::GenericExecutor;
use super::home_cinema::HomeCinemaExecutor;
use super::multiseat::MultiseatExecutor;
use super::stereo::Stereo20Executor;
use super::stereo_sub::Stereo21Executor;
use super::types::{WorkflowAssembly, WorkflowExecutor};
use crate::MeasurementSource;
use crate::roomeq::types::{
    CardioidConfig, CrossoverConfig, DBAConfig, MultiSubGroup, OptimizerConfig, ProcessingMode,
    RoomConfig, SpeakerConfig, SubwooferStrategy, SubwooferSystemConfig, SystemConfig, SystemModel,
};
use ndarray::Array1;
use std::collections::HashMap;
use std::path::Path;

pub(crate) fn flat_curve() -> crate::Curve {
    crate::Curve {
        freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20000.0), 96),
        spl: Array1::from_elem(96, 80.0),
        phase: None,
        ..Default::default()
    }
}

pub(crate) fn flat_curve_with_phase() -> crate::Curve {
    let mut c = flat_curve();
    c.phase = Some(Array1::from_elem(c.freq.len(), 0.0));
    c
}

pub(crate) fn base_optimizer() -> OptimizerConfig {
    OptimizerConfig {
        processing_mode: ProcessingMode::LowLatency,
        num_filters: 1,
        max_iter: 20,
        population: 6,
        min_freq: 20.0,
        max_freq: 500.0,
        psychoacoustic: false,
        refine: false,
        seed: Some(1),
        ..Default::default()
    }
}

fn stereo_speakers() -> HashMap<String, SpeakerConfig> {
    HashMap::from([
        (
            "left".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
        ),
        (
            "right".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
        ),
    ])
}

pub(crate) fn make_assembly<'cfg, 'p, 's>(
    config: &'cfg RoomConfig,
    sys: &'cfg SystemConfig,
) -> WorkflowAssembly<'cfg, 'p, 's> {
    WorkflowAssembly {
        config,
        sys,
        sample_rate: 48000.0,
        output_dir: Path::new("."),
        progress_factory: None,
        stage_callback: None,
    }
}

fn stereo_speakers_with_phase() -> HashMap<String, SpeakerConfig> {
    HashMap::from([
        (
            "left".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
        ),
        (
            "right".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
        ),
    ])
}

fn stereo_21_sub_sys(sub_strategy: SubwooferStrategy) -> SystemConfig {
    SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([
            ("L".to_string(), "left".to_string()),
            ("R".to_string(), "right".to_string()),
            ("LFE".to_string(), "sub".to_string()),
        ]),
        subwoofers: Some(SubwooferSystemConfig {
            config: sub_strategy,
            crossover: Some("bass_xo".to_string()),
            mapping: HashMap::from([("sub".to_string(), "L".to_string())]),
        }),
        bass_management: None,
        ..Default::default()
    }
}

fn stereo_21_crossover_fixed() -> HashMap<String, CrossoverConfig> {
    HashMap::from([(
        "bass_xo".to_string(),
        CrossoverConfig {
            crossover_type: "LR24".to_string(),
            frequency: Some(80.0),
            frequencies: None,
            frequency_range: None,
        },
    )])
}

fn stereo_21_room_config(
    speakers: HashMap<String, SpeakerConfig>,
    sys: &SystemConfig,
    crossovers: HashMap<String, CrossoverConfig>,
    optimizer: OptimizerConfig,
) -> RoomConfig {
    RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: Some(sys.clone()),
        speakers,
        crossovers: Some(crossovers),
        target_curve: None,
        optimizer,
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    }
}

#[test]
fn stereo_2_0_executor_runs() {
    let speakers = stereo_speakers();
    let sys = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([
            ("Left".to_string(), "left".to_string()),
            ("Right".to_string(), "right".to_string()),
        ]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };
    let config = RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: Some(sys.clone()),
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: base_optimizer(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo20Executor.execute(&mut assembly);
    assert!(
        result.is_ok(),
        "stereo 2.0 executor should run: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 2);
    assert!(result.channel_results.contains_key("Left"));
    assert!(result.channel_results.contains_key("Right"));
}

#[test]
fn home_cinema_executor_without_sub_runs() {
    let mut speakers = stereo_speakers();
    speakers.insert(
        "center".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let sys = SystemConfig {
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
    let config = RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: Some(sys.clone()),
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: base_optimizer(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let mut assembly = make_assembly(&config, &sys);
    let result = HomeCinemaExecutor.execute(&mut assembly);
    assert!(
        result.is_ok(),
        "home-cinema executor without sub should run: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 3);
    assert!(result.channel_results.contains_key("Left"));
    assert!(result.channel_results.contains_key("Right"));
    assert!(result.channel_results.contains_key("Center"));
}

#[test]
fn generic_executor_single_speaker_runs() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let sys = SystemConfig {
        model: SystemModel::Custom,
        speakers: HashMap::from([("Left".to_string(), "left".to_string())]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };
    let config = RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: Some(sys.clone()),
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: base_optimizer(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let mut assembly = make_assembly(&config, &sys);
    let result = GenericExecutor.execute(&mut assembly);
    assert!(
        result.is_ok(),
        "generic executor with a single speaker should run: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 1);
    assert!(result.channel_results.contains_key("Left"));
}

#[test]
fn home_cinema_executor_with_sub_runs() {
    let mut speakers = stereo_speakers();
    speakers.insert(
        "center".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let sys = SystemConfig {
        model: SystemModel::HomeCinema,
        speakers: HashMap::from([
            ("Left".to_string(), "left".to_string()),
            ("Right".to_string(), "right".to_string()),
            ("Center".to_string(), "center".to_string()),
            ("LFE".to_string(), "sub".to_string()),
        ]),
        subwoofers: Some(SubwooferSystemConfig {
            config: SubwooferStrategy::Single,
            crossover: Some("bass_xo".to_string()),
            mapping: HashMap::from([("sub".to_string(), "Left".to_string())]),
        }),
        bass_management: None,
        ..Default::default()
    };
    let mut crossovers = HashMap::new();
    crossovers.insert(
        "bass_xo".to_string(),
        CrossoverConfig {
            crossover_type: "LR24".to_string(),
            frequency: Some(80.0),
            frequencies: None,
            frequency_range: None,
        },
    );
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: Some(sys.clone()),
        speakers,
        crossovers: Some(crossovers),
        target_curve: None,
        optimizer,
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let mut assembly = make_assembly(&config, &sys);
    let result = HomeCinemaExecutor.execute(&mut assembly);
    assert!(
        result.is_ok(),
        "home-cinema executor with sub should run: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 4);
    assert!(result.channel_results.contains_key("Left"));
    assert!(result.channel_results.contains_key("Right"));
    assert!(result.channel_results.contains_key("Center"));
    assert!(result.channel_results.contains_key("LFE"));
}

#[test]
fn multiseat_executor_disabled_skips() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let sys = SystemConfig {
        model: SystemModel::Custom,
        speakers: HashMap::from([("Left".to_string(), "left".to_string())]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };
    let config = RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: Some(sys.clone()),
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: base_optimizer(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let mut assembly = make_assembly(&config, &sys);
    let result = MultiseatExecutor.execute(&mut assembly);
    assert!(
        result.is_ok(),
        "disabled multiseat executor should return unchanged result: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(result.channels.is_empty());
    assert!(result.channel_results.is_empty());
    assert_eq!(result.combined_pre_score, 0.0);
    assert_eq!(result.combined_post_score, 0.0);
}

#[test]
fn stereo_2_1_executor_runs() {
    let mut speakers = stereo_speakers();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let sys = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([
            ("L".to_string(), "left".to_string()),
            ("R".to_string(), "right".to_string()),
            ("LFE".to_string(), "sub".to_string()),
        ]),
        subwoofers: Some(SubwooferSystemConfig {
            config: SubwooferStrategy::Single,
            crossover: Some("bass_xo".to_string()),
            mapping: HashMap::from([("sub".to_string(), "L".to_string())]),
        }),
        bass_management: None,
        ..Default::default()
    };
    let mut crossovers = HashMap::new();
    crossovers.insert(
        "bass_xo".to_string(),
        CrossoverConfig {
            crossover_type: "LR24".to_string(),
            frequency: Some(80.0),
            frequencies: None,
            frequency_range: None,
        },
    );
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: Some(sys.clone()),
        speakers,
        crossovers: Some(crossovers),
        target_curve: None,
        optimizer,
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(
        result.is_ok(),
        "stereo 2.1 executor should run: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 3);
    assert!(result.channel_results.contains_key("L"));
    assert!(result.channel_results.contains_key("R"));
    assert!(result.channel_results.contains_key("LFE"));
}

#[test]
fn stereo_21_happy_path_with_phase() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
    );
    let sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, stereo_21_crossover_fixed(), optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(
        result.is_ok(),
        "happy path with phase failed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 3);
    assert!(result.channel_results.contains_key("L"));
    assert!(result.channel_results.contains_key("R"));
    assert!(result.channel_results.contains_key("LFE"));
}

#[test]
fn stereo_21_missing_l_mapping_errs() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
    );
    let mut sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    sys.speakers.remove("L");
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, stereo_21_crossover_fixed(), optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(result.is_err(), "missing L mapping should error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Missing speaker mapping"),
        "unexpected error: {msg}"
    );
}

#[test]
fn stereo_21_l_not_single_errs() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::MultiSub(MultiSubGroup {
            name: "left_subs".to_string(),
            speaker_name: None,
            subwoofers: vec![MeasurementSource::InMemory(flat_curve_with_phase())],
            allpass_optimization: false,
        }),
    );
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
    );
    let sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, stereo_21_crossover_fixed(), optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(result.is_err(), "L not Single should error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("must be a Single speaker config"),
        "unexpected error: {msg}"
    );
}

#[test]
fn stereo_21_missing_subwoofers_config_errs() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
    );
    let mut sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    sys.subwoofers = None;
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, stereo_21_crossover_fixed(), optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(result.is_err(), "missing subwoofers config should error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Missing subwoofers configuration"),
        "unexpected error: {msg}"
    );
}

#[test]
fn stereo_21_missing_lfe_mapping_errs() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
    );
    let mut sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    sys.speakers.remove("LFE");
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, stereo_21_crossover_fixed(), optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(result.is_err(), "missing LFE mapping should error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Missing speaker mapping for 'LFE'"),
        "unexpected error: {msg}"
    );
}

#[test]
fn stereo_21_missing_crossover_key_errs() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
    );
    let mut sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    sys.subwoofers.as_mut().unwrap().crossover = None;
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, stereo_21_crossover_fixed(), optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(result.is_err(), "missing crossover key should error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("'crossover' reference"),
        "unexpected error: {msg}"
    );
}

#[test]
fn stereo_21_crossover_missing_freq_and_range_errs() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
    );
    let sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    let mut crossovers = stereo_21_crossover_fixed();
    {
        let xo = crossovers.get_mut("bass_xo").unwrap();
        xo.frequency = None;
        xo.frequency_range = None;
    }
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, crossovers, optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(
        result.is_err(),
        "crossover missing frequency and range should error"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("'frequency' or 'frequency_range'"),
        "unexpected error: {msg}"
    );
}

#[test]
fn stereo_21_multisub_mso_runs() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::MultiSub(MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers: vec![
                MeasurementSource::InMemory(flat_curve_with_phase()),
                MeasurementSource::InMemory(flat_curve_with_phase()),
            ],
            allpass_optimization: false,
        }),
    );
    let sys = stereo_21_sub_sys(SubwooferStrategy::Mso);
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, stereo_21_crossover_fixed(), optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(
        result.is_ok(),
        "MultiSub(MSO) config failed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 3);
    assert!(result.channel_results.contains_key("LFE"));
}

#[test]
fn stereo_21_cardioid_runs() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Cardioid(Box::new(CardioidConfig {
            name: "cardioid".to_string(),
            speaker_name: None,
            front: MeasurementSource::InMemory(flat_curve_with_phase()),
            rear: MeasurementSource::InMemory(flat_curve_with_phase()),
            separation_meters: 0.5,
        })),
    );
    let sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, stereo_21_crossover_fixed(), optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(result.is_ok(), "Cardioid config failed: {:?}", result.err());
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 3);
    assert!(result.channel_results.contains_key("LFE"));
}

#[test]
fn stereo_21_dba_runs() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Dba(DBAConfig {
            name: "dba".to_string(),
            speaker_name: None,
            front: vec![MeasurementSource::InMemory(flat_curve_with_phase())],
            rear: vec![MeasurementSource::InMemory(flat_curve_with_phase())],
        }),
    );
    let sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, stereo_21_crossover_fixed(), optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(result.is_ok(), "DBA config failed: {:?}", result.err());
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 3);
    assert!(result.channel_results.contains_key("LFE"));
}

#[test]
fn stereo_21_ranged_crossover_runs() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
    );
    let sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    let mut crossovers = stereo_21_crossover_fixed();
    {
        let xo = crossovers.get_mut("bass_xo").unwrap();
        xo.frequency = None;
        xo.frequency_range = Some((60.0, 100.0));
    }
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, crossovers, optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(
        result.is_ok(),
        "ranged crossover failed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 3);
}

#[test]
fn stereo_21_phase_missing_fallback_runs() {
    let mut speakers = stereo_speakers();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, stereo_21_crossover_fixed(), optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(
        result.is_ok(),
        "phase-missing fallback should still run: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 3);
}

#[test]
fn stereo_21_gain_limit_advisory_no_panic() {
    let mut speakers = stereo_speakers_with_phase();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
    );
    let sys = stereo_21_sub_sys(SubwooferStrategy::Single);
    let mut optimizer = base_optimizer();
    optimizer.max_freq = 2000.0;
    let config = stereo_21_room_config(speakers, &sys, stereo_21_crossover_fixed(), optimizer);

    let mut assembly = make_assembly(&config, &sys);
    let result = Stereo21Executor.execute(&mut assembly);
    assert!(
        result.is_ok(),
        "gain-limit advisory path should not panic: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert_eq!(result.channels.len(), 3);
}
