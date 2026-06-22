//! Integration test for supporting-source room compensation.

use autoeq::Curve;
use autoeq::read::MeasurementSource;
use autoeq::roomeq::{
    OptimizerConfig, ProcessingMode, RoomConfig, SpeakerConfig, SupportingSourceConfig,
    SupportingSourceDecorrelation, SupportingSourceGroup, SystemConfig, SystemModel, optimize_room,
};
use ndarray::Array1;
use std::collections::HashMap;

fn flat_curve(spl_db: f64) -> Curve {
    Curve {
        freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20000.0), 96),
        spl: Array1::from_elem(96, spl_db),
        phase: None,
        ..Default::default()
    }
}

fn primary_with_notch() -> Curve {
    let mut c = flat_curve(80.0);
    // Create a 6 dB notch around 500 Hz.
    for (f, s) in c.freq.iter().zip(c.spl.iter_mut()) {
        if *f >= 400.0 && *f <= 600.0 {
            *s -= 6.0;
        }
    }
    c
}

fn base_optimizer() -> OptimizerConfig {
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

#[test]
fn stereo_workflow_emits_supporting_source_channels_and_metadata() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::SupportingSource(SupportingSourceGroup {
            name: "Left Main + Support".to_string(),
            speaker_name: None,
            primary: MeasurementSource::InMemory(primary_with_notch()),
            support: MeasurementSource::InMemory(flat_curve(80.0)),
            supporting_source: SupportingSourceConfig {
                delay_ms: 5.0,
                fir_taps: 256,
                decorrelation: SupportingSourceDecorrelation::None,
                ..Default::default()
            },
        }),
    );

    let sys = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([("L".to_string(), "left".to_string())]),
        ..Default::default()
    };

    let config = RoomConfig {
        version: autoeq::roomeq::types::default_config_version(),
        system: Some(sys),
        speakers,
        optimizer: base_optimizer(),
        target_curve: None,
        crossovers: None,
        recording_config: None,
        cea2034_cache: None,
        ctc: None,
    };

    let output_dir = tempfile::tempdir().unwrap();
    let result = optimize_room(&config, 48000.0, None, Some(output_dir.path()))
        .expect("optimization should succeed");

    assert!(
        result.channels.contains_key("L"),
        "primary channel L should be present"
    );
    assert!(
        result.channels.contains_key("L_support"),
        "support channel L_support should be present"
    );
    assert!(
        result.channel_results.contains_key("L"),
        "primary channel result L should be present"
    );
    assert!(
        result.channel_results.contains_key("L_support"),
        "support channel result L_support should be present"
    );

    let report = result
        .metadata
        .supporting_source
        .as_ref()
        .expect("supporting_source metadata should exist")
        .get("L")
        .expect("report for L should exist");
    assert!(report.enabled);
    assert_eq!(report.primary_output, "L");
    assert_eq!(report.support_output, "L_support");
    assert_eq!(report.fir_length, 256);

    let support_chain = &result.channels["L_support"];
    assert!(
        support_chain
            .plugins
            .iter()
            .any(|p| p.plugin_type == "convolution"),
        "support chain should contain a convolution plugin for the FIR"
    );
}

#[test]
fn home_cinema_workflow_emits_supporting_source_channels_and_metadata() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(80.0))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(80.0))),
    );
    speakers.insert(
        "left_ss".to_string(),
        SpeakerConfig::SupportingSource(SupportingSourceGroup {
            name: "Left Wide".to_string(),
            speaker_name: None,
            primary: MeasurementSource::InMemory(primary_with_notch()),
            support: MeasurementSource::InMemory(flat_curve(80.0)),
            supporting_source: SupportingSourceConfig {
                delay_ms: 4.0,
                fir_taps: 256,
                decorrelation: SupportingSourceDecorrelation::None,
                ..Default::default()
            },
        }),
    );

    let sys = SystemConfig {
        model: SystemModel::HomeCinema,
        speakers: HashMap::from([
            ("Left".to_string(), "left".to_string()),
            ("Right".to_string(), "right".to_string()),
            ("WideLeft".to_string(), "left_ss".to_string()),
        ]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };

    let config = RoomConfig {
        version: autoeq::roomeq::types::default_config_version(),
        system: Some(sys),
        speakers,
        optimizer: base_optimizer(),
        target_curve: None,
        crossovers: None,
        recording_config: None,
        cea2034_cache: None,
        ctc: None,
    };

    let output_dir = tempfile::tempdir().unwrap();
    let result = optimize_room(&config, 48000.0, None, Some(output_dir.path()))
        .expect("optimization should succeed");

    assert!(result.channels.contains_key("Left"));
    assert!(result.channels.contains_key("Right"));
    assert!(result.channels.contains_key("WideLeft"));
    assert!(result.channels.contains_key("WideLeft_support"));

    let report = result
        .metadata
        .supporting_source
        .as_ref()
        .expect("supporting_source metadata should exist")
        .get("WideLeft")
        .expect("report for WideLeft should exist");
    assert!(report.enabled);
    assert_eq!(report.primary_output, "WideLeft");
    assert_eq!(report.support_output, "WideLeft_support");
}

#[test]
fn spatial_robustness_advisories_raised_for_multiple_measurements() {
    // Two positions that differ by 12 dB across the whole spectrum should
    // trigger a high spatial-variance advisory.
    let seat1 = flat_curve(86.0);
    let seat2 = flat_curve(74.0);

    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::SupportingSource(SupportingSourceGroup {
            name: "Left Main + Support".to_string(),
            speaker_name: None,
            primary: MeasurementSource::InMemoryMultiple(vec![seat1, seat2]),
            support: MeasurementSource::InMemory(flat_curve(80.0)),
            supporting_source: SupportingSourceConfig {
                delay_ms: 3.0,
                fir_taps: 256,
                velvet_noise_taps: 128,
                decorrelation: SupportingSourceDecorrelation::None,
                ..Default::default()
            },
        }),
    );

    let sys = SystemConfig {
        model: SystemModel::Stereo,
        speakers: HashMap::from([("L".to_string(), "left".to_string())]),
        ..Default::default()
    };

    let config = RoomConfig {
        version: autoeq::roomeq::types::default_config_version(),
        system: Some(sys),
        speakers,
        optimizer: base_optimizer(),
        target_curve: None,
        crossovers: None,
        recording_config: None,
        cea2034_cache: None,
        ctc: None,
    };

    let output_dir = tempfile::tempdir().unwrap();
    let result = optimize_room(&config, 48000.0, None, Some(output_dir.path()))
        .expect("optimization should succeed");

    let report = result
        .metadata
        .supporting_source
        .as_ref()
        .unwrap()
        .get("L")
        .unwrap();
    assert!(
        report
            .advisories
            .iter()
            .any(|a| a.contains("spatial_variance")),
        "should raise a spatial-robustness advisory for multiple measurements: {:?}",
        report.advisories
    );
}
