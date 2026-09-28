// =============================================================================
// Additional branch coverage for optimize_room_impl and decomposed helpers
// =============================================================================

fn optimizer_for_mode(processing_mode: ProcessingMode) -> OptimizerConfig {
    OptimizerConfig {
        processing_mode,
        num_filters: 1,
        max_iter: 20,
        population: 6,
        seed: Some(1),
        min_freq: 20.0,
        max_freq: 500.0,
        psychoacoustic: false,
        refine: false,
        fir: Some(roomeq_model::FirConfig {
            placement: Default::default(),
            taps: 128,
            phase: "linear".to_string(),
            correct_excess_phase: false,
            phase_smoothing: 1.0 / 6.0,
            pre_ringing: None,
            max_boost_db: None,
        }),
        mixed_phase: Some(roomeq_model::MixedPhaseSerdeConfig {
            max_fir_length_ms: 5.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
            assessment: Default::default(),
            max_correction_latency_ms: None,
        }),
        ..OptimizerConfig::default()
    }
}

fn stereo_config_for_mode(processing_mode: ProcessingMode) -> RoomConfig {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );

    RoomConfig {
        version: roomeq_model::default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::Stereo,
            speakers: HashMap::from([
                ("L".to_string(), "left".to_string()),
                ("R".to_string(), "right".to_string()),
            ]),
            subwoofers: None,
            bass_management: None,
            ..Default::default()
        }),
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: optimizer_for_mode(processing_mode),
        provenance: Default::default(),
        recording_config: None,
        measured_impulse_responses: Default::default(),
        ctc: None,
        reporting: None,
        cea2034_cache: None,
    }
}

/// Inline measurement source with declared provenance for intake tests.
///
/// In-memory curves carry no declaration by design, so fixtures that need
/// evidence authorization use inline data with an explicit provenance.
fn inline_source_with_provenance(
    name: &str,
    with_phase: bool,
    provenance: autoeq_core::MeasurementProvenance,
) -> MeasurementSource {
    let freq: Vec<f64> = flat_curve().freq.to_vec();
    let phase_deg = if with_phase {
        Some(freq.iter().map(|frequency| -0.02 * frequency).collect())
    } else {
        None
    };
    MeasurementSource::Single(autoeq_core::MeasurementSingle {
        measurement: autoeq_core::MeasurementRef::Inline(autoeq_core::InlineMeasurement {
            frequencies: freq,
            magnitude_db: vec![80.0; 96],
            phase_deg,
            name: Some(name.to_string()),
            wav_path: None,
            csv_path: None,
        }),
        speaker_name: Some(name.to_string()),
        provenance,
    })
}

fn stationary_provenance() -> autoeq_core::MeasurementProvenance {
    autoeq_core::MeasurementProvenance {
        capture_kind: autoeq_core::ProvenanceCaptureKind::StationaryIr,
        calibration_id: None,
        timing_reference_id: Some(String::from("loopback-1")),
        has_measured_spl: false,
        valid_band_hz: None,
        valid_bands_hz: Vec::new(),
        has_direct_angular: false,
        direct_sound: None,
        capture: None,
    }
}

fn optimize_room_with_temp_output(config: &RoomConfig) -> Result<RoomOptimizationResult> {
    let output_dir = tempfile::tempdir()?;
    optimize_room_impl_with_frequency_samples(
        config,
        48000.0,
        Some(output_dir.path()),
        None,
        None,
        &autoeq_artifacts::MemoryArtifactStore::new(),
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

#[test]
fn optimize_room_impl_workflow_phase_linear_succeeds() {
    let config = stereo_config_for_mode(ProcessingMode::PhaseLinear);
    let result = optimize_room_with_temp_output(&config);
    assert!(
        result.is_ok(),
        "phase-linear stereo workflow should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.channels.is_empty());
}

#[test]
fn optimize_room_impl_workflow_hybrid_succeeds() {
    let config = stereo_config_for_mode(ProcessingMode::Hybrid);
    let result = optimize_room_with_temp_output(&config);
    assert!(
        result.is_ok(),
        "hybrid stereo workflow should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.channels.is_empty());
}

#[test]
fn roadmap_measurement_receipts_reach_iir_fir_and_hybrid_reports() {
    for mode in [
        ProcessingMode::LowLatency,
        ProcessingMode::PhaseLinear,
        ProcessingMode::Hybrid,
    ] {
        let config = stereo_config_for_mode(mode.clone());
        let output_dir = tempfile::tempdir().unwrap();
        let result =
            crate::optimize_room(&config, 48_000.0, None, Some(output_dir.path())).unwrap();
        let stage = result
            .metadata
            .stage_outcomes
            .iter()
            .find(|stage| stage.stage == "measurement_input_conditioning")
            .expect("final output must retain the preparation receipt stage");
        assert_eq!(stage.status, roomeq_model::StageStatus::Applied, "{mode:?}");
        assert_eq!(stage.checks.len(), 2, "{mode:?}");
        assert!(
            stage.checks.iter().all(|check| check.passed),
            "{mode:?}: {stage:?}"
        );
        for channel in result.channel_results.values() {
            let receipt = channel.measurement_conditioning.as_ref().unwrap();
            assert_eq!(
                receipt.representative_identity,
                channel.initial_curve.content_hash().unwrap()
            );
        }
    }
}

#[test]
fn optimize_room_impl_workflow_mixed_phase_succeeds() {
    let mut config = stereo_config_for_mode(ProcessingMode::MixedPhase);
    for speaker in config.speakers.values_mut() {
        if let SpeakerConfig::Single(MeasurementSource::InMemory(curve)) = speaker {
            curve.phase = Some(Array1::from_iter(
                curve.freq.iter().map(|frequency| -0.02 * frequency),
            ));
        }
    }
    let result = optimize_room_with_temp_output(&config);
    assert!(
        result.is_ok(),
        "mixed-phase stereo workflow should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.channels.is_empty());
    if let Some(reports) = result.metadata.mixed_phase_per_channel.as_ref() {
        assert_eq!(reports.len(), 2);
        for report in reports.values() {
            assert!(report.estimated_delay_ms.is_finite());
            assert!(report.fir_taps > 0);
            assert!(report.residual_excess_phase_min_deg.is_finite());
            assert!(report.residual_excess_phase_max_deg.is_finite());
            assert!(report.residual_excess_phase_rms_deg.is_finite());
            assert!(report.residual_excess_phase_min_deg <= report.residual_excess_phase_max_deg);
        }
    } else {
        assert!(
            result
                .metadata
                .correction_acceptance
                .as_ref()
                .is_some_and(|acceptance| !acceptance.accepted),
            "mixed-phase reports may be absent only when the final safety gate reverted correction"
        );
    }
}

fn phase_correction_config() -> roomeq_model::MixedPhaseSerdeConfig {
    roomeq_model::MixedPhaseSerdeConfig {
        max_fir_length_ms: 5.0,
        pre_ringing_threshold_db: -30.0,
        min_spatial_depth: 0.5,
        phase_smoothing_octaves: 1.0 / 6.0,
        assessment: Default::default(),
        max_correction_latency_ms: None,
    }
}

fn stereo_inline_config(
    processing_mode: ProcessingMode,
    with_phase: bool,
    provenance: autoeq_core::MeasurementProvenance,
) -> RoomConfig {
    let mut config = stereo_config_for_mode(processing_mode);
    for (name, speaker) in config.speakers.iter_mut() {
        let source = inline_source_with_provenance(name, with_phase, provenance.clone());
        *speaker = SpeakerConfig::Single(source);
    }
    config
}

fn gate_for<'a>(
    result: &'a RoomOptimizationResult,
    measurement: &str,
) -> &'a roomeq_model::eligibility::ChannelOperationGate {
    result
        .metadata
        .operation_gates
        .as_ref()
        .expect("intake gates attach on every run")
        .iter()
        .find(|gate| gate.measurement_id == measurement || gate.channel == measurement)
        .expect("channel gate present")
}

#[test]
fn optimize_room_impl_workflow_phase_correction_succeeds() {
    // Stationary captures with a shared timing reference authorize phase work.
    let mut config =
        stereo_inline_config(ProcessingMode::LowLatency, true, stationary_provenance());
    config.optimizer.phase_correction = Some(phase_correction_config());
    let result = optimize_room_with_temp_output(&config);
    assert!(
        result.is_ok(),
        "workflow with phase correction should succeed: {:?}",
        result.as_ref().err()
    );
    let result = result.unwrap();
    assert!(
        gate_for(&result, "left")
            .authorizes(roomeq_model::eligibility::CorrectionOperation::ExcessPhaseCorrection),
        "stationary evidence authorizes excess-phase work"
    );
    if let Some(reports) = result.metadata.mixed_phase_per_channel.as_ref() {
        assert_eq!(reports.len(), 2);
    } else {
        assert!(
            result
                .metadata
                .correction_acceptance
                .as_ref()
                .is_some_and(|acceptance| !acceptance.accepted),
            "phase-correction reports may be absent only when the final safety gate reverted correction"
        );
    }
}

/// MMM input permits magnitude correction but refuses phase work, even with
/// phase data loaded and phase correction requested.
#[test]
fn roadmap_correction_mmm_permits_magnitude_refuses_phase() {
    let provenance = autoeq_core::MeasurementProvenance {
        capture_kind: autoeq_core::ProvenanceCaptureKind::SpatialMagnitude,
        ..Default::default()
    };
    let mut config = stereo_inline_config(ProcessingMode::LowLatency, true, provenance);
    config.optimizer.phase_correction = Some(phase_correction_config());
    let result = optimize_room_with_temp_output(&config);
    assert!(
        result.is_ok(),
        "mmm run keeps magnitude path: {:?}",
        result.err()
    );
    let result = result.unwrap();
    // Magnitude correction still delivered.
    assert!(!result.channels.is_empty());
    // Phase work refused on every channel with a spatial-magnitude reason.
    for channel in ["left", "right"] {
        let gate = gate_for(&result, channel);
        assert!(
            !gate.authorizes(roomeq_model::eligibility::CorrectionOperation::ExcessPhaseCorrection),
            "mmm must refuse excess-phase work on {channel}"
        );
        assert!(
            gate.records.iter().any(|record| {
                record.operation
                    == roomeq_model::eligibility::CorrectionOperation::ExcessPhaseCorrection
                    && record
                        .observations
                        .iter()
                        .any(|observation| observation.contains("spatial magnitude"))
            }),
            "refusal cites the spatial-magnitude capture"
        );
    }
    // No phase FIR was emitted.
    assert!(
        result
            .metadata
            .mixed_phase_per_channel
            .as_ref()
            .is_none_or(|reports| reports.is_empty()),
        "refused phase work emits no phase reports"
    );
}

/// Undeclared provenance refuses phase work: unknown never authorizes.
#[test]
fn roadmap_correction_unknown_provenance_refuses_phase() {
    let mut config = stereo_config_for_mode(ProcessingMode::LowLatency);
    for speaker in config.speakers.values_mut() {
        if let SpeakerConfig::Single(MeasurementSource::InMemory(curve)) = speaker {
            curve.phase = Some(Array1::from_iter(
                curve.freq.iter().map(|frequency| -0.02 * frequency),
            ));
        }
    }
    config.optimizer.phase_correction = Some(phase_correction_config());
    let result = optimize_room_with_temp_output(&config);
    assert!(result.is_ok(), "unknown provenance keeps magnitude path");
    let result = result.unwrap();
    // Stereo channels are keyed by role; in-memory curves carry no
    // measurement identity, so look the gates up by channel.
    for channel in ["L", "R"] {
        assert!(
            !gate_for(&result, channel)
                .authorizes(roomeq_model::eligibility::CorrectionOperation::ExcessPhaseCorrection),
            "unknown provenance refuses excess-phase work on {channel}"
        );
    }
    assert!(
        result
            .metadata
            .mixed_phase_per_channel
            .as_ref()
            .is_none_or(|reports| reports.is_empty())
    );
}

/// A short gate authorizes its valid band only: support outside the valid
/// band carries explicit refusals while the valid band stays eligible.
#[test]
fn roadmap_correction_short_gate_limits_authorized_band() {
    // The fixture's 96-point native grid is below the workflow cap. Its
    // declared edges lie between samples and must not become assessed bins.
    let retained: Vec<_> = flat_curve()
        .freq
        .iter()
        .copied()
        .filter(|frequency| (50.0..=8000.0).contains(frequency))
        .collect();
    let low = retained[0];
    let high = *retained.last().unwrap();
    assert!(low > 50.0 && high < 8000.0);
    let provenance = autoeq_core::MeasurementProvenance {
        valid_band_hz: Some([50.0, 8000.0]),
        ..stationary_provenance()
    };
    let config = stereo_inline_config(ProcessingMode::LowLatency, true, provenance);
    let result = optimize_room_with_temp_output(&config);
    assert!(
        result.is_ok(),
        "short-gate run succeeds: {:?}",
        result.err()
    );
    let result = result.unwrap();
    let gate = gate_for(&result, "left");
    // Grid endpoints carry floating-point dust from log spacing: compare
    // bands with tolerance instead of exact equality.
    let band_near = |band: Option<[f64; 2]>, lo: f64, hi: f64| {
        band.is_some_and(|[a, b]| (a - lo).abs() < 1e-6 && (b - hi).abs() < 1e-6)
    };
    let phase_records: Vec<_> = gate
        .records
        .iter()
        .filter(|record| {
            record.operation
                == roomeq_model::eligibility::CorrectionOperation::ExcessPhaseCorrection
        })
        .collect();
    assert!(
        phase_records.iter().any(|record| {
            record.verdict == roomeq_model::eligibility::EligibilityVerdict::Eligible
                && band_near(record.band_hz, low, high)
        }),
        "valid band stays eligible"
    );
    assert!(
        phase_records.iter().any(|record| {
            record.verdict == roomeq_model::eligibility::EligibilityVerdict::Unsupported
                && band_near(record.band_hz, high, 20000.0)
        }),
        "support above the valid band is refused"
    );
    assert!(
        phase_records.iter().any(|record| {
            record.verdict == roomeq_model::eligibility::EligibilityVerdict::Unsupported
                && band_near(record.band_hz, 20.0, low)
        }),
        "support below the retained valid band is refused"
    );
}

/// Direct-sound detail needs declared angular coverage; SPL-absent input
/// refuses absolute loudness claims. Both are record-level verdicts on the
/// public path for the target policy to consume.
#[test]
fn roadmap_correction_detail_needs_angular_and_spl() {
    use roomeq_model::eligibility::{CorrectionOperation, EligibilityVerdict};
    let direct = autoeq_core::MeasurementProvenance {
        capture_kind: autoeq_core::ProvenanceCaptureKind::DirectSound,
        timing_reference_id: Some(String::from("loopback-1")),
        ..Default::default()
    };
    let config = stereo_inline_config(ProcessingMode::LowLatency, false, direct);
    let result = optimize_room_with_temp_output(&config);
    assert!(result.is_ok());
    let result = result.unwrap();
    let gate = gate_for(&result, "left");
    let detail = gate
        .records
        .iter()
        .find(|record| record.operation == CorrectionOperation::DirectSoundSpeakerCorrection)
        .expect("detail record present");
    assert_eq!(
        detail.verdict,
        EligibilityVerdict::Unknown,
        "detail unassessed without angular coverage"
    );
    let loudness = gate
        .records
        .iter()
        .find(|record| record.operation == CorrectionOperation::AbsoluteLoudnessAnalysis)
        .expect("loudness record present");
    assert_eq!(
        loudness.verdict,
        EligibilityVerdict::Unknown,
        "absolute loudness unassessed without measured SPL"
    );

    let mut angular = stationary_provenance();
    angular.capture_kind = autoeq_core::ProvenanceCaptureKind::DirectSound;
    angular.has_direct_angular = true;
    angular.has_measured_spl = true;
    let config = stereo_inline_config(ProcessingMode::LowLatency, false, angular);
    let result = optimize_room_with_temp_output(&config);
    assert!(result.is_ok());
    let result = result.unwrap();
    let gate = gate_for(&result, "left");
    let detail = gate
        .records
        .iter()
        .find(|record| record.operation == CorrectionOperation::DirectSoundSpeakerCorrection)
        .expect("detail record present");
    assert_eq!(
        detail.verdict,
        EligibilityVerdict::Unknown,
        "a legacy angular boolean cannot validate geometry, gate, or angular sampling"
    );
    let loudness = gate
        .records
        .iter()
        .find(|record| record.operation == CorrectionOperation::AbsoluteLoudnessAnalysis)
        .expect("loudness record present");
    assert_eq!(loudness.verdict, EligibilityVerdict::Eligible);
}

/// Mismatched timing references refuse coherent summation while per-channel
/// excess-phase verdicts stand on their own evidence.
#[test]
fn roadmap_correction_mismatched_reference_rejects_coherent() {
    use roomeq_model::eligibility::{CorrectionOperation, EligibilityVerdict};
    let mut left_provenance = stationary_provenance();
    left_provenance.timing_reference_id = Some(String::from("loopback-left"));
    let mut right_provenance = stationary_provenance();
    right_provenance.timing_reference_id = Some(String::from("loopback-right"));
    let mut config = stereo_config_for_mode(ProcessingMode::LowLatency);
    for (name, speaker) in config.speakers.iter_mut() {
        let provenance = if name == "left" {
            left_provenance.clone()
        } else {
            right_provenance.clone()
        };
        *speaker = SpeakerConfig::Single(inline_source_with_provenance(name, true, provenance));
    }
    let result = optimize_room_with_temp_output(&config);
    assert!(result.is_ok());
    let result = result.unwrap();
    for channel in ["left", "right"] {
        let gate = gate_for(&result, channel);
        // Per-channel phase evidence is self-consistent: still eligible.
        assert!(
            gate.authorizes(CorrectionOperation::ExcessPhaseCorrection),
            "{channel} keeps its own phase authorization"
        );
        let coherent = gate
            .records
            .iter()
            .find(|record| record.operation == CorrectionOperation::CoherentSummation)
            .expect("coherent record present");
        assert_eq!(
            coherent.verdict,
            EligibilityVerdict::Unsupported,
            "{channel}"
        );
    }
}

/// Real intake assesses facts rather than trusting a declared usable band.
#[test]
fn roadmap_correction_quasi_anechoic_public_intake() {
    use roomeq_model::eligibility::{CorrectionOperation, EligibilityVerdict};

    let valid: autoeq_core::MeasurementProvenance = serde_json::from_value(serde_json::json!({
        "capture_kind": "direct_sound",
        "timing_reference_id": "loopback-1",
        "valid_band_hz": [50.0, 20000.0],
        "direct_sound": {
            "facts": {
                "gate_s": 0.002,
                "direct_path_m": 1.0,
                "first_reflection_path_m": 2.0,
                "sound_speed_m_s": 343.0,
                "angular": {"angles_deg": [0.0, -30.0, 30.0]},
                "averaging": "stationary",
                "capture_kind": "direct_sound",
                "sample_rate_hz": 16000.0
            },
            "policy": {
                "version": "quasi-anechoic-v1",
                "cycles_for_valid_band": 2.0,
                "min_off_axis_count": 2,
                "min_off_axis_abs_deg": 15.0
            }
        }
    }))
    .unwrap();
    for case in [
        "valid",
        "missing_facts",
        "missing_policy",
        "overlong",
        "negative_geometry",
        "unknown_averaging",
        "moving",
        "missing_angles",
        "duplicate_angles",
        "contradictory_kind",
        "missing_rate",
        "no_band",
        "zero_angle_policy",
        "zero_count_policy",
    ] {
        let mut provenance = valid.clone();
        let direct = provenance.direct_sound.as_mut().unwrap();
        match case {
            "valid" => {}
            "missing_facts" => provenance.direct_sound = None,
            "missing_policy" => direct.policy = None,
            "overlong" => direct.facts.gate_s = Some(0.01),
            "negative_geometry" => direct.facts.direct_path_m = Some(-1.0),
            "unknown_averaging" => {
                direct.facts.averaging = autoeq_core::direct_sound::AveragingMethod::Unknown
            }
            "moving" => {
                direct.facts.averaging =
                    autoeq_core::direct_sound::AveragingMethod::MovingMicrophone
            }
            "missing_angles" => direct.facts.angular.angles_deg.clear(),
            "duplicate_angles" => direct.facts.angular.angles_deg = vec![30.0, 30.0],
            "contradictory_kind" => {
                direct.facts.capture_kind = autoeq_core::evidence::CaptureKind::SpatialMagnitude
            }
            "missing_rate" => direct.facts.sample_rate_hz = None,
            "no_band" => direct.facts.sample_rate_hz = Some(1000.0),
            "zero_angle_policy" => direct.policy.as_mut().unwrap().min_off_axis_abs_deg = 0.0,
            "zero_count_policy" => direct.policy.as_mut().unwrap().min_off_axis_count = 0,
            _ => unreachable!(),
        }
        // A JSON round trip exercises the same public source contract as file configs.
        let mut config = stereo_inline_config(ProcessingMode::LowLatency, true, provenance);
        if case != "valid" {
            config.optimizer.phase_correction = Some(phase_correction_config());
        }
        let config = serde_json::from_value(serde_json::to_value(config).unwrap()).unwrap();
        let result = optimize_room_with_temp_output(&config)
            .unwrap_or_else(|error| panic!("{case}: {error}"));
        assert!(
            !result.channels.is_empty(),
            "{case}: supported magnitude workflow remains available"
        );
        if case != "valid" {
            assert!(
                result
                    .metadata
                    .mixed_phase_per_channel
                    .as_ref()
                    .is_none_or(|reports| reports.is_empty()),
                "{case}: no phase realization may be delivered without supported direct evidence"
            );
        }
        let gate = gate_for(&result, "left");
        let detail_allowed = gate.authorizes(CorrectionOperation::DirectSoundSpeakerCorrection);
        assert_eq!(detail_allowed, case == "valid", "{case}: detail permission");
        let phase_allowed = gate.authorizes(CorrectionOperation::ExcessPhaseCorrection);
        assert_eq!(
            phase_allowed,
            matches!(case, "valid" | "missing_angles" | "duplicate_angles"),
            "{case}: phase source permission"
        );
        if case == "valid" {
            for operation in [
                CorrectionOperation::DirectSoundSpeakerCorrection,
                CorrectionOperation::ExcessPhaseCorrection,
            ] {
                let eligible = gate
                    .records
                    .iter()
                    .find(|record| {
                        record.operation == operation
                            && record.verdict == EligibilityVerdict::Eligible
                    })
                    .unwrap();
                assert_eq!(eligible.band_hz, Some([1000.0, 8000.0]));
                assert!(
                    eligible
                        .observations
                        .iter()
                        .any(|note| note.contains("quasi-anechoic-v1"))
                );
                assert!(
                    gate.records
                        .iter()
                        .any(|record| record.operation == operation
                            && record.verdict == EligibilityVerdict::Unsupported
                            && record.band_hz.is_some_and(|band| band[1] == 1000.0))
                );
            }
        }
        // Evidence and refusal reasons survive the actual public output serialization.
        let output = result.to_dsp_chain_output();
        let json = serde_json::to_string(&output).unwrap();
        assert!(
            json.contains(if case == "valid" {
                "quasi-anechoic-v1"
            } else {
                "direct"
            }),
            "{case}"
        );
    }
}

#[test]
fn roadmap_correction_timing_id_does_not_upgrade_mmm_or_unknown_capture() {
    use roomeq_model::eligibility::CorrectionOperation;
    for capture_kind in [
        autoeq_core::ProvenanceCaptureKind::SpatialMagnitude,
        autoeq_core::ProvenanceCaptureKind::Unknown,
    ] {
        let provenance = autoeq_core::MeasurementProvenance {
            capture_kind,
            timing_reference_id: Some(String::from("claimed-clock")),
            ..Default::default()
        };
        let mut config = stereo_inline_config(ProcessingMode::LowLatency, true, provenance);
        config.optimizer.phase_correction = Some(phase_correction_config());
        let result = optimize_room_with_temp_output(&config).unwrap();
        let gate = gate_for(&result, "left");
        assert!(!gate.authorizes(CorrectionOperation::ExcessPhaseCorrection));
        assert!(!gate.authorizes(CorrectionOperation::CoherentSummation));
        assert!(
            !result.channels.is_empty(),
            "relative magnitude processing remains supported"
        );
        assert!(
            result
                .metadata
                .mixed_phase_per_channel
                .as_ref()
                .is_none_or(|reports| reports.is_empty())
        );
    }
}

#[test]
fn optimize_room_impl_generic_multiple_channels_phase_linear_succeeds() {
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
    optimizer.fir = Some(roomeq_model::FirConfig {
        placement: Default::default(),
        taps: 128,
        phase: "linear".to_string(),
        correct_excess_phase: false,
        phase_smoothing: 1.0 / 6.0,
        pre_ringing: None,
        max_boost_db: None,
    });
    let config = room_config_with_optimizer(speakers, None, optimizer);

    let result = optimize_room_with_temp_output(&config);
    assert!(
        result.is_ok(),
        "generic multi-channel phase-linear optimization should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(result.channels.len() >= 2);
}

#[test]
fn optimize_room_impl_home_cinema_workflow_succeeds() {
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
        "center".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
    );
    let system = SystemConfig {
        model: SystemModel::HomeCinema,
        speakers: HashMap::from([
            ("L".to_string(), "left".to_string()),
            ("R".to_string(), "right".to_string()),
            ("Center".to_string(), "center".to_string()),
        ]),
        subwoofers: None,
        bass_management: None,
        ..Default::default()
    };
    let config = room_config_with_optimizer(speakers, Some(system), tiny_optimizer());

    let result = optimize_room_with_temp_output(&config);
    assert!(
        result.is_ok(),
        "home cinema workflow should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.channels.is_empty());
}

#[test]
fn optimize_speaker_single_channel_succeeds() {
    let source = SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve()));
    let result = optimize_speaker("left", &source, &tiny_optimizer(), None, 48000.0, None);
    assert!(
        result.is_ok(),
        "optimize_speaker should succeed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert!(!result.biquads.is_empty() || result.fir_coeffs.is_some());
}

#[test]
fn optimize_speaker_forwards_progress_callback() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let source = SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve()));
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_callback = Arc::clone(&calls);
    let callback: SpeakerOptimizationCallback = Box::new(move |progress| {
        assert_eq!(progress.current_speaker, "left");
        calls_for_callback.fetch_add(1, Ordering::Relaxed);
        CallbackAction::Continue
    });

    optimize_speaker(
        "left",
        &source,
        &tiny_optimizer(),
        None,
        48_000.0,
        Some(callback),
    )
    .unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);
}

#[test]
fn optimize_speaker_topology_forwards_progress_callback() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let source = SpeakerConfig::Topology(SpeakerTopology {
        name: "full-range".to_string(),
        speaker_name: None,
        drivers: vec![SpeakerDriver {
            id: "driver".to_string(),
            role: SpeakerDriverRole::FullRange,
            measurement: MeasurementSource::InMemory(flat_curve()),
            crossover_band: None,
        }],
        parallel_groups: Vec::new(),
        crossover: None,
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_callback = Arc::clone(&calls);
    let callback: SpeakerOptimizationCallback = Box::new(move |progress| {
        assert_eq!(progress.current_speaker, "center");
        calls_for_callback.fetch_add(1, Ordering::Relaxed);
        CallbackAction::Continue
    });

    optimize_speaker(
        "center",
        &source,
        &tiny_optimizer(),
        None,
        48_000.0,
        Some(callback),
    )
    .unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);
}

#[test]
fn validate_room_optimization_single_speaker_succeeds() {
    let config = minimal_room_config(ProcessingMode::LowLatency);
    let observer = observer_none();
    let result = validate_room_optimization_with_frequency_samples(
        &config,
        &observer,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    );
    assert!(
        result.is_ok(),
        "single-speaker config should validate: {:?}",
        result.err()
    );
}
