#![allow(clippy::field_reassign_with_default)]
use super::validate::{RoomValidationContext, validate_room_config, validate_room_config_staged};
use super::validation_result::ValidationResult;
use crate::config::{
    Cea2034CorrectionMode, OptimizerConfig, ProcessingMode, RoomConfig, SpeakerConfig,
};
use crate::{Curve, MeasurementRef, MeasurementSingle, MeasurementSource};
use std::collections::HashMap;

use crate::config::*;
use std::path::PathBuf;

mod misc;

#[test]
fn test_validation_result_default_is_valid() {
    let result = ValidationResult::default();
    assert!(result.is_valid);
    assert!(result.errors.is_empty());
    assert!(result.warnings.is_empty());
    assert!(result.staged_report.is_none());
}

#[test]
fn compatibility_validator_identifies_structural_only_strength() {
    let mut config = RoomConfig::default();
    config.speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(crate::Curve {
            freq: ndarray::arr1(&[20.0, 20_000.0]),
            spl: ndarray::arr1(&[0.0, 0.0]),
            phase: None,
            ..Default::default()
        })),
    );

    let result = validate_room_config(&config);
    let staged = result
        .staged_report
        .expect("public compatibility result must state which stages ran");
    assert!(staged.stage_ran(ValidationStage::Structural));
    assert!(!staged.production_ready());
}

#[test]
fn test_validation_result_add_error_invalidates() {
    let mut result = ValidationResult::valid();
    result.add_error("Test error".to_string());
    assert!(!result.is_valid);
    assert_eq!(result.errors.len(), 1);
}

#[test]
fn test_validation_result_add_warning_keeps_valid() {
    let mut result = ValidationResult::valid();
    result.add_warning("Test warning".to_string());
    assert!(result.is_valid);
    assert_eq!(result.warnings.len(), 1);
}

#[test]
fn production_validation_reports_every_named_stage() {
    let mut config = RoomConfig::default();
    config.speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(crate::Curve {
            freq: ndarray::arr1(&[20.0, 100.0, 1_000.0, 20_000.0]),
            spl: ndarray::arr1(&[0.0, 1.0, -1.0, 0.0]),
            phase: None,
            ..Default::default()
        })),
    );

    let report = validate_room_config_staged(&config, RoomValidationContext::production());

    for stage in ValidationStage::ALL {
        assert_ne!(report.stage(stage).status, ValidationStageStatus::NotRun);
    }
    assert_eq!(
        report.stage(ValidationStage::ExportTarget).status,
        ValidationStageStatus::NotApplicable
    );
    assert!(report.production_ready(), "{report:#?}");
}

#[test]
fn test_validate_empty_speakers() {
    let config = RoomConfig {
        version: default_config_version(),
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

    let result = validate_room_config(&config);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("No speakers")));
}

#[test]
fn test_validate_min_freq_greater_than_max() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Path(PathBuf::from("test.csv")),
            speaker_name: None,
        })),
    );

    let mut optimizer = OptimizerConfig::default();
    optimizer.min_freq = 20000.0;
    optimizer.max_freq = 20.0;

    let config = RoomConfig {
        version: default_config_version(),
        system: None,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer,
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let result = validate_room_config(&config);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("min_freq")));
}

#[test]
fn test_validate_cea2034_score_mode_is_invalid_for_roomeq() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Path(PathBuf::from("spinorama_left.csv")),
            speaker_name: Some("Example Speaker".to_string()),
        })),
    );

    let optimizer = OptimizerConfig {
        cea2034_correction: Some(Cea2034CorrectionConfig {
            enabled: true,
            correction_mode: Cea2034CorrectionMode::Score,
            ..Default::default()
        }),
        ..Default::default()
    };

    let config = RoomConfig {
        version: default_config_version(),
        system: None,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer,
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let result = validate_room_config(&config);

    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| {
        e.contains("cea2034_correction.correction_mode=score is not supported in roomeq")
    }));
}

#[test]
fn test_validate_warped_iir_mode_is_valid() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(Curve {
            freq: ndarray::Array1::from_vec(vec![20.0, 100.0, 1000.0]),
            spl: ndarray::Array1::from_vec(vec![80.0, 80.0, 80.0]),
            phase: None,
            ..Default::default()
        })),
    );

    let optimizer = OptimizerConfig {
        processing_mode: ProcessingMode::WarpedIir,
        ..Default::default()
    };

    let config = RoomConfig {
        version: default_config_version(),
        system: None,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer,
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let result = validate_room_config(&config);

    assert!(
        result.is_valid,
        "expected valid warped_iir config, got {:?}",
        result.errors
    );
}

#[test]
fn test_validate_crossover_reference() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Group(SpeakerGroup {
            name: "Test".to_string(),
            speaker_name: None,
            measurements: vec![
                MeasurementSource::Single(MeasurementSingle {
                    measurement: MeasurementRef::Path(PathBuf::from("woofer.csv")),
                    speaker_name: None,
                }),
                MeasurementSource::Single(MeasurementSingle {
                    measurement: MeasurementRef::Path(PathBuf::from("tweeter.csv")),
                    speaker_name: None,
                }),
            ],
            crossover: Some("nonexistent".to_string()),
        }),
    );

    let config = RoomConfig {
        version: default_config_version(),
        system: None,
        speakers,
        crossovers: Some(HashMap::new()), // Empty crossovers
        target_curve: None,
        optimizer: OptimizerConfig::default(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let result = validate_room_config(&config);
    assert!(!result.is_valid);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("non-existent crossover"))
    );
}

#[test]
fn test_validate_crossover_frequency_range_rejects_reversed_bounds() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Group(SpeakerGroup {
            name: "Test".to_string(),
            speaker_name: None,
            measurements: vec![
                MeasurementSource::Single(MeasurementSingle {
                    measurement: MeasurementRef::Path(PathBuf::from("woofer.csv")),
                    speaker_name: None,
                }),
                MeasurementSource::Single(MeasurementSingle {
                    measurement: MeasurementRef::Path(PathBuf::from("tweeter.csv")),
                    speaker_name: None,
                }),
            ],
            crossover: Some("xo".to_string()),
        }),
    );
    let config = RoomConfig {
        version: default_config_version(),
        speakers,
        crossovers: Some(HashMap::from([(
            "xo".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: None,
                frequencies: None,
                frequency_range: Some((300.0, 100.0)),
            },
        )])),
        ..RoomConfig::default()
    };

    let result = validate_room_config(&config);
    assert!(
        result
            .errors
            .iter()
            .any(|error| error.contains("frequency_range"))
    );
}

#[test]
fn test_validate_speaker_name() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Path(PathBuf::from("left.csv")),
            speaker_name: Some("Invalid @ Name".to_string()),
        })),
    );

    let config = RoomConfig {
        version: default_config_version(),
        system: None,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig::default(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let result = validate_room_config(&config);
    assert!(!result.is_valid);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("invalid speaker_name"))
    );
}

#[test]
fn validate_bass_management_rejects_negative_headroom_and_boost() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Path(PathBuf::from("sub.csv")),
            speaker_name: None,
        })),
    );
    let config = RoomConfig {
        version: default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([("Sub".to_string(), "sub".to_string())]),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                crossover: Some("xo".into()),
                mapping: HashMap::new(),
            }),
            bass_management: Some(BassManagementConfig {
                max_sub_boost_db: -1.0,
                headroom_margin_db: -3.0,
                ..Default::default()
            }),
            ..Default::default()
        }),
        speakers,
        crossovers: Some(HashMap::from([(
            "xo".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: Some(80.0),
                frequencies: None,
                frequency_range: None,
            },
        )])),
        target_curve: None,
        optimizer: OptimizerConfig::default(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let result = validate_room_config(&config);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("max_sub_boost_db")));
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("headroom_margin_db"))
    );
}

#[test]
fn validate_role_targets_rejects_invalid_bands_and_distances() {
    let mut speakers = HashMap::new();
    speakers.insert(
        "C".to_string(),
        SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Path(PathBuf::from("center.csv")),
            speaker_name: None,
        })),
    );
    let config = RoomConfig {
        version: default_config_version(),
        system: None,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: OptimizerConfig {
            target_response: Some(TargetResponseConfig {
                role_targets: Some(RoleTargetConfig {
                    center_dialog_low_hz: 5000.0,
                    center_dialog_high_hz: 500.0,
                    cinema_x_curve_start_hz: 0.0,
                    listening_distance_m: Some(-1.0),
                    cinema_reference_distance_m: 0.0,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        },
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    let result = validate_room_config(&config);
    assert!(!result.is_valid);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("center dialog band"))
    );
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("cinema_x_curve_start_hz"))
    );
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("listening_distance_m"))
    );
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("cinema_reference_distance_m"))
    );
}

/// Two-sub MSO config mirroring `data_tests/roomeq/measured/2.2`: the
/// `subs` speaker holds two subwoofers and the crossovers map holds
/// `bass_xover1` / `bass_xover2`.
fn mso_two_sub_config(crossover: Option<SubwooferCrossoverRef>) -> RoomConfig {
    fn single(path: &str) -> MeasurementSource {
        MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Path(PathBuf::from(path)),
            speaker_name: None,
        })
    }
    let mut speakers = HashMap::new();
    speakers.insert("L".to_string(), SpeakerConfig::Single(single("L.csv")));
    speakers.insert("R".to_string(), SpeakerConfig::Single(single("R.csv")));
    speakers.insert(
        "subs".to_string(),
        SpeakerConfig::MultiSub(MultiSubGroup {
            name: "Two subs".to_string(),
            speaker_name: None,
            subwoofers: vec![single("sub1.csv"), single("sub2.csv")],
            allpass_optimization: false,
        }),
    );
    let lr24_range = |minimum: f64, maximum: f64| CrossoverConfig {
        crossover_type: "LR24".to_string(),
        frequency: None,
        frequencies: None,
        frequency_range: Some((minimum, maximum)),
    };
    RoomConfig {
        version: default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::Stereo,
            speakers: HashMap::from([
                ("L".to_string(), "L".to_string()),
                ("R".to_string(), "R".to_string()),
                ("LFE".to_string(), "subs".to_string()),
            ]),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Mso,
                crossover,
                mapping: HashMap::from([("subs".to_string(), "L".to_string())]),
            }),
            bass_management: None,
            supporting_source_outputs: None,
        }),
        speakers,
        crossovers: Some(HashMap::from([
            ("bass_xover1".to_string(), lr24_range(40.0, 130.0)),
            ("bass_xover2".to_string(), lr24_range(60.0, 130.0)),
        ])),
        target_curve: None,
        optimizer: OptimizerConfig::default(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    }
}

#[test]
fn subwoofer_crossover_list_matching_sub_count_is_valid() {
    let config = mso_two_sub_config(Some(SubwooferCrossoverRef::PerSub(vec![
        "bass_xover1".to_string(),
        "bass_xover2".to_string(),
    ])));
    let result = validate_room_config(&config);
    assert!(
        result.is_valid,
        "per-sub list matching the sub count must validate: {:?}",
        result.errors
    );
}

#[test]
fn subwoofer_crossover_shared_and_single_entry_forms_stay_valid() {
    for crossover in [
        SubwooferCrossoverRef::Shared("bass_xover1".to_string()),
        SubwooferCrossoverRef::PerSub(vec!["bass_xover1".to_string()]),
    ] {
        let result = validate_room_config(&mso_two_sub_config(Some(crossover.clone())));
        assert!(
            result.is_valid,
            "shared/single-entry crossover {crossover:?} must validate: {:?}",
            result.errors
        );
    }
}

#[test]
fn subwoofer_crossover_rejects_unknown_key() {
    // Shared form with the dangling `bass_xover` key from the 2.2 fixture.
    let result = validate_room_config(&mso_two_sub_config(Some(SubwooferCrossoverRef::Shared(
        "bass_xover".to_string(),
    ))));
    assert!(!result.is_valid);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("non-existent crossover 'bass_xover'")),
        "unexpected errors: {:?}",
        result.errors
    );

    // List form where only the second entry is unknown.
    let result = validate_room_config(&mso_two_sub_config(Some(SubwooferCrossoverRef::PerSub(
        vec!["bass_xover1".to_string(), "nope".to_string()],
    ))));
    assert!(!result.is_valid);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("non-existent crossover 'nope'")),
        "unexpected errors: {:?}",
        result.errors
    );
}

#[test]
fn subwoofer_crossover_rejects_arity_mismatch() {
    // Three entries for two subwoofers.
    let result = validate_room_config(&mso_two_sub_config(Some(SubwooferCrossoverRef::PerSub(
        vec![
            "bass_xover1".to_string(),
            "bass_xover2".to_string(),
            "bass_xover1".to_string(),
        ],
    ))));
    assert!(!result.is_valid);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("has 3 entries") && e.contains("expected 1 or 2")),
        "unexpected errors: {:?}",
        result.errors
    );

    // Empty list is never a valid arity.
    let result = validate_room_config(&mso_two_sub_config(Some(SubwooferCrossoverRef::PerSub(
        Vec::new(),
    ))));
    assert!(!result.is_valid);
    assert!(
        result.errors.iter().any(|e| e.contains("has 0 entries")),
        "unexpected errors: {:?}",
        result.errors
    );
}
