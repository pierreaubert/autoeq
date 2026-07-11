use super::types::{CallbackAction, ChannelOptimizationResult, GenericChannelCollection};
use super::*;
use crate::MeasurementSource;
use crate::roomeq::pipeline::{
    PipelineControl, PipelineEvent, PipelineObserver, PipelineStepId, PipelineStepStatus,
};
use crate::roomeq::test_fixtures::{empty_metadata, single_channel_room_result};
use crate::roomeq::types::{
    BassManagementConfig, ChannelDspChain, CrossoverConfig, MultiMeasurementConfig,
    MultiMeasurementStrategy, MultiSubGroup, OptimizerConfig, ProcessingMode, RoomConfig,
    SpeakerConfig, SpeakerGroup, SubwooferStrategy, SubwooferSystemConfig, SystemConfig,
    SystemModel,
};
use ndarray::Array1;
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

fn flat_curve() -> crate::Curve {
    crate::Curve {
        freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20000.0), 96),
        spl: Array1::from_elem(96, 80.0),
        phase: None,
        ..Default::default()
    }
}

fn minimal_room_config(processing_mode: ProcessingMode) -> RoomConfig {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );

    RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system: None,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig {
            processing_mode,
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
fn optimize_room_empty_speakers_fails_validation() {
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

    let result = optimize_room(&config, 48000.0, None, None);
    assert!(result.is_err(), "empty speakers should fail validation");
}

#[test]
fn optimize_room_single_speaker_low_latency_succeeds() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let result = optimize_room(&config, 48000.0, None, None);
    assert!(
        result.is_ok(),
        "single speaker low-latency optimization should succeed: {:?}",
        result.err()
    );
}

#[test]
#[allow(deprecated)]
fn optimize_room_legacy_vog_alias_records_migration_and_failed_stage() {
    let mut config = minimal_room_config(ProcessingMode::LowLatency);
    config.speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    config.optimizer.vog = Some(crate::roomeq::types::VoiceOfGodConfig {
        enabled: true,
        reference_channel: "missing".to_string(),
        min_improvement_db: 0.0,
    });

    let result = optimize_room(&config, 48_000.0, None, None).unwrap();
    let outcome = result
        .metadata
        .stage_outcomes
        .iter()
        .find(|outcome| outcome.stage == "inter_channel_timbre_matching")
        .expect("inter-channel timbre-matching stage outcome");
    assert_eq!(outcome.status, crate::roomeq::types::StageStatus::Failed);
    assert!(
        outcome
            .advisories
            .iter()
            .any(|advisory| advisory.contains("invalid_reference"))
    );
    assert!(
        outcome
            .advisories
            .contains(&"deprecated_optimizer_vog_alias".to_string())
    );
}

#[test]
fn optimize_room_new_timbre_config_precedes_legacy_alias() {
    let mut config = minimal_room_config(ProcessingMode::LowLatency);
    config.optimizer.inter_channel_timbre_matching =
        Some(crate::roomeq::types::InterChannelTimbreMatchingConfig::default());
    config.optimizer.vog = Some(crate::roomeq::types::InterChannelTimbreMatchingConfig {
        enabled: true,
        reference_channel: "left".to_string(),
        min_improvement_db: 0.0,
    });

    let result = optimize_room(&config, 48_000.0, None, None).unwrap();
    let outcome = result
        .metadata
        .stage_outcomes
        .iter()
        .find(|outcome| outcome.stage == "inter_channel_timbre_matching")
        .expect("ignored legacy alias should be reported");
    assert_eq!(outcome.status, crate::roomeq::types::StageStatus::Skipped);
    assert!(
        outcome
            .advisories
            .contains(&"optimizer_vog_ignored_because_new_config_present".to_string())
    );
}

#[test]
fn optimize_room_height_alignment_records_structured_stage() {
    let mut config = minimal_room_config(ProcessingMode::LowLatency);
    config.speakers.insert(
        "TFL".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    config.optimizer.height_channel_alignment =
        Some(crate::roomeq::types::HeightChannelAlignmentConfig {
            enabled: true,
            min_timbre_improvement_db: 0.0,
            ..Default::default()
        });

    let result = optimize_room(&config, 48_000.0, None, None).unwrap();
    let outcome = result
        .metadata
        .stage_outcomes
        .iter()
        .find(|outcome| outcome.stage == "height_channel_alignment")
        .expect("height-alignment stage outcome");
    assert_ne!(outcome.status, crate::roomeq::types::StageStatus::Failed);
}

#[test]
fn height_arrival_delay_updates_exported_chain_and_reported_phase() {
    let mut config = minimal_room_config(ProcessingMode::LowLatency);
    let mut phased = flat_curve();
    phased.phase = Some(Array1::zeros(phased.freq.len()));
    config.speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(phased.clone())),
    );
    config.speakers.insert(
        "TFL".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(phased)),
    );
    config.optimizer.height_channel_alignment =
        Some(crate::roomeq::types::HeightChannelAlignmentConfig {
            enabled: true,
            match_timbre: false,
            match_level: false,
            match_arrival_time: true,
            ..Default::default()
        });
    let arrivals = HashMap::from([("left".to_string(), 5.0), ("TFL".to_string(), 3.0)]);

    let result =
        optimize_room_with_probe_arrivals(&config, 48_000.0, None, None, &arrivals).unwrap();

    assert!(result.channels["TFL"].plugins.iter().any(|plugin| {
        plugin.plugin_type == "delay"
            && plugin
                .parameters
                .get("delay_ms")
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|delay| (delay - 2.0).abs() < 1e-9)
    }));
    assert!(
        result.channel_results["TFL"]
            .final_curve
            .phase
            .as_ref()
            .is_some_and(|phase| phase.iter().any(|value| value.abs() > 1e-6))
    );
}

#[test]
fn phase_complete_multisub_reaches_downstream_phase_alignment() {
    let mut config = minimal_room_config(ProcessingMode::LowLatency);
    let mut phased = flat_curve();
    phased.phase = Some(Array1::zeros(phased.freq.len()));
    config.speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(phased.clone())),
    );
    config.speakers.insert(
        "sub".to_string(),
        SpeakerConfig::MultiSub(MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers: vec![
                MeasurementSource::InMemory(phased.clone()),
                MeasurementSource::InMemory(phased),
            ],
            allpass_optimization: false,
        }),
    );
    config.optimizer.phase_alignment = Some(Default::default());
    config.optimizer.allow_delay = Some(true);

    let events = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&events);
    let observer = Box::new(move |event: &PipelineEvent| {
        observed.lock().unwrap().push((event.step_id, event.status));
        PipelineControl::Continue
    });
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(Some(observer));

    assert!(
        result.is_ok(),
        "phase-complete multi-sub run failed: {result:?}"
    );
    assert!(events.lock().unwrap().iter().any(|(step, status)| {
        *step == PipelineStepId::PhaseAlignment && *status == PipelineStepStatus::Started
    }));
}

#[test]
fn optimize_room_with_probe_arrivals_uses_overrides() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let mut probe = HashMap::new();
    probe.insert("left".to_string(), 5.0);
    let result = optimize_room_with_probe_arrivals(&config, 48000.0, None, None, &probe);
    assert!(
        result.is_ok(),
        "optimization with probe arrivals should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_pipeline_events_are_emitted() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let event_count = Arc::new(AtomicUsize::new(0));
    let count_clone = Arc::clone(&event_count);

    let observer = Box::new(move |_event: &PipelineEvent| -> PipelineControl {
        count_clone.fetch_add(1, Ordering::SeqCst);
        PipelineControl::Continue
    });

    let _result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(Some(observer));

    assert!(
        event_count.load(Ordering::SeqCst) > 0,
        "observer should have received at least one event"
    );
}

#[test]
fn optimize_room_pipeline_stop_request_halts() {
    let config = minimal_room_config(ProcessingMode::LowLatency);

    let observer = Box::new(|_event: &PipelineEvent| -> PipelineControl { PipelineControl::Stop });

    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(Some(observer));

    assert!(
        result.is_err(),
        "pipeline should return error when observer requests stop"
    );
}

#[test]
fn emit_pipeline_event_no_observer_ok() {
    let observer: SharedPipelineObserver = Arc::new(Mutex::new(None));
    let event = PipelineEvent::started(PipelineStepId::ConfigPreparation, "test");
    assert!(emit_pipeline_event(&observer, event).is_ok());
}

#[test]
fn emit_pipeline_event_continue_observer_ok() {
    let observer: SharedPipelineObserver = Arc::new(Mutex::new(Some(Box::new(
        |_event: &PipelineEvent| -> PipelineControl { PipelineControl::Continue },
    ))));
    let event = PipelineEvent::started(PipelineStepId::ConfigPreparation, "test");
    assert!(emit_pipeline_event(&observer, event).is_ok());
}

#[test]
fn emit_pipeline_event_stop_observer_errors() {
    let observer: SharedPipelineObserver = Arc::new(Mutex::new(Some(Box::new(
        |_event: &PipelineEvent| -> PipelineControl { PipelineControl::Stop },
    ))));
    let event = PipelineEvent::started(PipelineStepId::Validation, "test");
    let result = emit_pipeline_event(&observer, event);
    assert!(result.is_err());
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(err_str.contains("stopped by observer"));
}

#[test]
fn emit_pipeline_event_step_id_in_error() {
    let observer: SharedPipelineObserver = Arc::new(Mutex::new(Some(Box::new(
        |_event: &PipelineEvent| -> PipelineControl { PipelineControl::Stop },
    ))));
    let event = PipelineEvent::started(PipelineStepId::Validation, "test");
    let result = emit_pipeline_event(&observer, event);
    assert!(result.is_err());
}

#[test]
fn optimize_room_stereo_2_0_workflow() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );

    let config = RoomConfig {
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
    };

    let result = optimize_room(&config, 48000.0, None, None);
    assert!(
        result.is_ok(),
        "stereo 2.0 workflow should succeed: {:?}",
        result.err()
    );
}

#[test]
fn optimize_room_result_has_channel_results() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let result = optimize_room(&config, 48000.0, None, None).unwrap();
    assert!(
        !result.channel_results.is_empty(),
        "result should contain at least one channel result"
    );
}

#[test]
fn optimize_room_progress_callback_receives_updates() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let call_count = Arc::new(AtomicUsize::new(0));
    let count_clone = Arc::clone(&call_count);

    let callback: RoomOptimizationCallback =
        Box::new(move |_progress: &RoomOptimizationProgress| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            CallbackAction::Continue
        });

    let _result = optimize_room(&config, 48000.0, Some(callback), None);

    assert!(
        call_count.load(Ordering::SeqCst) > 0,
        "progress callback should have been called"
    );
}

#[test]
fn optimize_room_progress_callback_stop_halts() {
    let config = minimal_room_config(ProcessingMode::LowLatency);

    let callback: RoomOptimizationCallback =
        Box::new(|_progress: &RoomOptimizationProgress| CallbackAction::Stop);

    let result = optimize_room(&config, 48000.0, Some(callback), None);
    assert!(
        result.is_err(),
        "optimization should stop when callback returns Stop"
    );
}

fn base_room_config(
    speakers: HashMap<String, SpeakerConfig>,
    system: Option<SystemConfig>,
) -> RoomConfig {
    RoomConfig {
        version: crate::roomeq::types::default_config_version(),
        system,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
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

fn observer_none() -> SharedPipelineObserver {
    Arc::new(Mutex::new(None))
}

include!("tests/topology.rs");
include!("tests/pipeline.rs");
include!("tests/processing_modes.rs");
include!("tests/observers.rs");
include!("tests/workflows.rs");
