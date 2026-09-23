//! CI admission: stereo, home-cinema (main/sub), and multi-sub entry paths
//! optimize end to end, and every emitted graph finalizes a bound ledger.
//!
//! The stereo leg pins the baseline path. The home-cinema leg carries
//! measured phase so the overlap-band summation search runs in production
//! (its advisories prove the hook ran; adoption is covered at unit level).
//! The multi-sub legs pin the detailed mode and explicit joint refusal
//! when the timing scope is unknown. Engine construction leaves
//! `correction_decisions` empty; only reconciliation at emission attaches
//! the finalized, graph-bound ledger.

use autoeq::Curve;
use autoeq::roomeq::{
    CrossoverConfig, MeasurementSource, OptimizerConfig, ProcessingMode, RoomConfig, RoomPipeline,
    RoomPipelineRequest, SpeakerConfig, SubwooferCrossoverRef, SubwooferOutput, SubwooferStrategy,
    SubwooferSystemConfig, SystemConfig, SystemModel, default_config_version, final_ledger,
};
use std::collections::HashMap;

fn phased_curve(base_level: f64, delay_s: f64) -> Curve {
    let n = 60;
    let freq: Vec<f64> = (0..n)
        .map(|i| 20.0 * (25.0f64).powf(i as f64 / n as f64))
        .collect();
    let spl: Vec<f64> = freq
        .iter()
        .map(|f| base_level + (f / 100.0).ln() * 1.5)
        .collect();
    let phase: Vec<f64> = freq.iter().map(|f| -360.0 * f * delay_s).collect();
    Curve {
        freq: ndarray::Array1::from_vec(freq),
        spl: ndarray::Array1::from_vec(spl),
        phase: Some(ndarray::Array1::from_vec(phase)),
        ..Default::default()
    }
}

fn tiny_optimizer() -> OptimizerConfig {
    OptimizerConfig {
        max_iter: 20,
        population: 8,
        num_filters: 1,
        processing_mode: ProcessingMode::LowLatency,
        refine: false,
        seed: Some(7),
        ..OptimizerConfig::default()
    }
}

fn stereo_config() -> RoomConfig {
    let mut speakers = HashMap::new();
    speakers.insert(
        "left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(phased_curve(80.0, 0.0))),
    );
    speakers.insert(
        "right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(phased_curve(82.0, 0.0))),
    );
    RoomConfig {
        version: default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::Stereo,
            speakers: HashMap::from([
                ("L".to_string(), "left".to_string()),
                ("R".to_string(), "right".to_string()),
            ]),
            subwoofers: None,
            bass_management: None,
            supporting_source_outputs: None,
        }),
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: tiny_optimizer(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    }
}

fn declared_voltage_config(limit: f64) -> RoomConfig {
    let mut config = stereo_config();
    config.optimizer.finalization.default_input_peak = 0.01;
    let envelope = serde_json::json!({
        "quantity": "voltage_rms", "calibration_id": "synthetic-voltmeter-reference",
        "reference_conditions_id": "synthetic-resistive-load", "limit_conditions_id": "synthetic-resistive-load",
        "sine_duration_seconds": 1.0, "reference_output_peak": 0.1, "linear_valid_output_peak": 1.0,
        "frequencies_hz": [50.0, 100.0], "demand_at_reference": [1.0, 1.0], "limits": [limit, limit]
    });
    config.optimizer.finalization.physical_drive = Some(serde_json::from_value(serde_json::json!({
        "outputs": { "[\"channel\",\"L\"]": [envelope.clone()], "[\"channel\",\"R\"]": [envelope] }
    })).unwrap());
    config
}

#[test]
fn roadmap_correction_admission_physical_search_finds_permitted_lower_drive() {
    let mut config = declared_voltage_config(1000.0);
    let run = |config: &RoomConfig| {
        RoomPipeline::new(RoomPipelineRequest {
            config,
            sample_rate: 48_000.0,
            output_dir: None,
            probe_arrival_overrides: None,
        })
        .run(None)
    };
    let original = run(&config).unwrap();
    let original_assessments =
        roomeq_workflow::electrical_headroom::assess_final_graph_physical_drive(
            &original.to_dsp_chain_output(),
            48_000.0,
            std::path::Path::new("."),
            &config.optimizer.finalization,
        )
        .unwrap();
    for assessment in &original_assessments {
        let envelope = &mut config
            .optimizer
            .finalization
            .physical_drive
            .as_mut()
            .unwrap()
            .outputs
            .get_mut(&assessment.output)
            .unwrap()[0];
        // This synthetic limit forces the search to change the delivered DSP.
        envelope.limits = assessment
            .demands
            .iter()
            .map(|value| value * 0.98)
            .collect();
    }
    let constrained = run(&config).expect("a modest physical cut fits the existing search budgets");
    let graph = finalize_bound(&constrained);
    let after = roomeq_workflow::electrical_headroom::assess_final_graph_physical_drive(
        &graph,
        48_000.0,
        std::path::Path::new("."),
        &config.optimizer.finalization,
    )
    .unwrap();
    assert!(
        after
            .iter()
            .all(|assessment| assessment.passes_declared_samples)
    );
    assert!(
        after
            .iter()
            .zip(&original_assessments)
            .all(|(after, before)| {
                after
                    .demands
                    .iter()
                    .zip(&before.demands)
                    .all(|(a, b)| *a <= b * 0.98)
            })
    );
    assert!(
        constrained.metadata.stage_outcomes.iter().any(|stage| {
            stage.stage == "final_correction_selection"
                && stage.checks.iter().any(|check| check.passed)
        }),
        "must select an assessed candidate, not merely return an attenuated rejected fallback"
    );
}

#[test]
fn roadmap_correction_admission_physical_drive_changes_candidate_ranking() {
    let mut config = declared_voltage_config(1.0);
    // A broad dip admits partial positive correction: reducing its strength
    // trades target fit for lower drive without destroying all improvement.
    config.optimizer.min_db = 0.0;
    for source in config.speakers.values_mut() {
        let SpeakerConfig::Single(MeasurementSource::InMemory(curve)) = source else {
            panic!("in-memory fixture");
        };
        curve.spl = curve
            .freq
            .mapv(|frequency| 80.0 - 6.0 * (-((frequency / 100.0).ln() / 0.3).powi(2)).exp());
    }
    let run = |config: &RoomConfig| {
        RoomPipeline::new(RoomPipelineRequest {
            config,
            sample_rate: 48_000.0,
            output_dir: None,
            probe_arrival_overrides: None,
        })
        .run(None)
        .unwrap()
    };
    let before = run(&config);
    config.optimizer.finalization.physical_drive_weight = 1000.0;
    let after = run(&config);
    let objective = |result: &roomeq_workflow::RoomOptimizationResult| {
        let stage = result
            .metadata
            .stage_outcomes
            .iter()
            .find(|stage| stage.stage == "final_candidate_objective")
            .unwrap();
        serde_json::from_str::<serde_json::Value>(stage.checks[0].diagnostic.as_ref().unwrap())
            .unwrap()
    };
    let initial = objective(&before);
    let ranked = objective(&after);
    assert!(
        ranked["worst_declared_drive_utilization"].as_f64().unwrap()
            < initial["worst_declared_drive_utilization"]
                .as_f64()
                .unwrap(),
        "{initial} -> {ranked}"
    );
    let replay = roomeq_workflow::electrical_headroom::assess_final_graph_physical_drive(
        &after.to_dsp_chain_output(),
        48_000.0,
        std::path::Path::new("."),
        &config.optimizer.finalization,
    )
    .unwrap();
    assert!(
        replay
            .iter()
            .all(|assessment| assessment.passes_declared_samples)
    );
    let utilization = replay
        .iter()
        .map(|assessment| assessment.max_utilization)
        .fold(0.0_f64, f64::max);
    assert!(
        (ranked["worst_declared_drive_utilization"].as_f64().unwrap() - utilization).abs() < 1e-12
    );
    assert!(
        after
            .metadata
            .correction_acceptance
            .as_ref()
            .unwrap()
            .accepted
    );
}

#[test]
fn roadmap_correction_admission_physical_voltage_limit_is_enforced() {
    let config = declared_voltage_config(1e-9);
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None);
    assert!(
        result.is_err(),
        "an explicit physical voltage overload must not be delivered"
    );
    let error = result.err().unwrap().to_string();
    assert!(
        error.contains("declared physical-drive check failed"),
        "{error}"
    );
    assert!(error.contains("V RMS"), "{error}");
}

#[test]
fn roadmap_correction_admission_physical_attenuation_does_not_grant_acoustic_budget() {
    let mut config = declared_voltage_config(0.01);
    config.optimizer.finalization.max_attenuation_db = 60.0;
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .expect("retain a safely attenuated but explicitly rejected fallback artifact");
    let report = result.metadata.correction_acceptance.as_ref().unwrap();
    assert!(!report.accepted);
    assert_eq!(report.decision, roomeq_model::CorrectionDecision::Rejected);
    assert!(
        report
            .violations
            .iter()
            .any(|reason| reason == "baseline_requires_safety_attenuation")
    );
    assert!(result.metadata.stage_outcomes.iter().any(|stage| {
        stage.stage == "final_graph_declared_physical_drive"
            && !stage.checks.is_empty()
            && stage.checks.iter().all(|check| check.passed)
    }));
    assert!(
        !result
            .metadata
            .stage_outcomes
            .iter()
            .filter(|stage| stage.stage == "final_correction_selection")
            .flat_map(|stage| &stage.checks)
            .any(|check| check.passed)
    );
}

#[test]
fn roadmap_correction_admission_physical_voltage_pass_is_scoped_and_bound() {
    let config = declared_voltage_config(1000.0);
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .unwrap();
    let stage = result
        .metadata
        .stage_outcomes
        .iter()
        .find(|stage| stage.stage == "final_graph_declared_physical_drive")
        .unwrap();
    assert_eq!(stage.checks.len(), 2);
    assert!(
        stage
            .checks
            .iter()
            .all(|check| check.passed && check.observed.unwrap() <= 1.0)
    );
    assert!(
        stage
            .advisories
            .iter()
            .any(|text| text.contains("undeclared_physical_constraints_remain_unknown"))
    );
    for check in &stage.checks {
        let evidence: serde_json::Value =
            serde_json::from_str(check.diagnostic.as_ref().unwrap()).unwrap();
        assert_eq!(evidence["unit"], "V RMS");
        assert_eq!(
            evidence["declaration"]["calibration_id"],
            "synthetic-voltmeter-reference"
        );
        assert_eq!(
            evidence["digital_output_peaks"].as_array().unwrap().len(),
            2
        );
    }
    finalize_bound(&result);
}

#[test]
fn roadmap_correction_admission_physical_evidence_is_fail_closed() {
    for invalid in 0..3 {
        let mut config = declared_voltage_config(1000.0);
        let policy = config
            .optimizer
            .finalization
            .physical_drive
            .as_mut()
            .unwrap();
        let expected = match invalid {
            0 => {
                policy.outputs.remove("[\"channel\",\"R\"]");
                "output coverage mismatch"
            }
            1 => {
                policy.outputs.values_mut().next().unwrap()[0].calibration_id =
                    "uncalibrated".into();
                "missing calibration"
            }
            _ => {
                let envelope = &mut policy.outputs.values_mut().next().unwrap()[0];
                envelope.reference_output_peak = 1e-6;
                envelope.linear_valid_output_peak = 1e-6;
                envelope.limits = vec![1e12; 2];
                "declared physical-drive check failed"
            }
        };
        let result = RoomPipeline::new(RoomPipelineRequest {
            config: &config,
            sample_rate: 48_000.0,
            output_dir: None,
            probe_arrival_overrides: None,
        })
        .run(None);
        let error = result
            .err()
            .expect("invalid physical evidence must refuse delivery")
            .to_string();
        assert!(error.contains(expected), "{error}");
    }
}

fn home_cinema_config() -> RoomConfig {
    let mut speakers = HashMap::new();
    speakers.insert(
        "l".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(phased_curve(80.0, 0.002))),
    );
    speakers.insert(
        "r".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(phased_curve(80.0, 0.002))),
    );
    speakers.insert(
        "sub".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(phased_curve(85.0, 0.0))),
    );
    let mut crossovers = HashMap::new();
    crossovers.insert(
        "sub_xover".to_string(),
        CrossoverConfig {
            crossover_type: "LR24".to_string(),
            frequency: Some(80.0),
            frequencies: None,
            frequency_range: None,
        },
    );
    RoomConfig {
        version: default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::Stereo,
            speakers: HashMap::from([
                ("L".to_string(), "l".to_string()),
                ("R".to_string(), "r".to_string()),
            ]),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                crossover: Some(SubwooferCrossoverRef::PerSub(vec!["sub_xover".to_string()])),
                routing: Default::default(),
                outputs: vec![SubwooferOutput {
                    id: "Sub1".to_string(),
                    speaker: "sub".to_string(),
                }],
            }),
            bass_management: None,
            supporting_source_outputs: None,
        }),
        speakers,
        crossovers: Some(crossovers),
        target_curve: None,
        optimizer: tiny_optimizer(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    }
}

#[test]
fn roadmap_correction_mso_unknown_timing_emits_gain_only_routing() {
    let mut config = home_cinema_config();
    config.speakers.remove("sub");
    config.speakers.insert(
        "sub1".into(),
        SpeakerConfig::Single(MeasurementSource::InMemory(phased_curve(85.0, 0.0))),
    );
    config.speakers.insert(
        "sub2".into(),
        SpeakerConfig::Single(MeasurementSource::InMemory(phased_curve(82.0, 0.001))),
    );
    let subwoofers = config.system.as_mut().unwrap().subwoofers.as_mut().unwrap();
    subwoofers.config = SubwooferStrategy::Mso;
    subwoofers.crossover = Some(SubwooferCrossoverRef::PerSub(vec![
        "sub_xover".into(),
        "sub_xover".into(),
    ]));
    subwoofers.outputs = vec![
        SubwooferOutput {
            id: "Sub1".into(),
            speaker: "sub1".into(),
        },
        SubwooferOutput {
            id: "Sub2".into(),
            speaker: "sub2".into(),
        },
    ];
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .expect("unknown timing retains the gain-only RoomEQ workflow");
    let bass = result.metadata.bass_management.as_ref().unwrap();
    let optimization = bass.optimization.as_ref().unwrap();
    assert!(
        optimization
            .advisories
            .iter()
            .any(|reason| reason == "unverified_timing_gain_only")
    );
    let graph = bass.routing_graph.as_ref().unwrap();
    let redirected: Vec<_> = graph
        .routes
        .iter()
        .filter(|route| route.route_kind == "redirected_bass_lowpass_to_sub")
        .collect();
    assert_eq!(redirected.len(), 2);
    assert!(redirected.iter().all(|route| route.delay_ms == 0.0));
    let sub = result
        .channels
        .values()
        .find(|chain| {
            chain
                .drivers
                .as_ref()
                .is_some_and(|drivers| drivers.len() == 2)
        })
        .expect("the delivered multi-sub chain has two physical drivers");
    assert!(
        sub.drivers
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(|driver| &driver.plugins)
            .all(|plugin| plugin.plugin_type != "delay")
    );
    finalize_bound(&result);
}

fn declare_stationary_captures(config: &mut RoomConfig) {
    for speaker in config.speakers.values_mut() {
        let SpeakerConfig::Single(MeasurementSource::InMemory(curve)) = speaker else {
            panic!("fixture starts with explicit in-memory curves");
        };
        *speaker = SpeakerConfig::Single(
            serde_json::from_value(serde_json::json!({
                "inline": {
                    "frequencies": curve.freq.to_vec(),
                    "magnitude_db": curve.spl.to_vec(),
                    "phase_deg": curve.phase.as_ref().unwrap().to_vec()
                },
                "provenance": {
                    "capture_kind": "stationary_ir",
                    "timing_reference_id": "fixture-common-clock"
                }
            }))
            .expect("declared inline measurement"),
        );
    }
}

#[test]
fn roadmap_correction_admission_crossover_refuses_unshared_capture_reference() {
    for (kind, reference) in [
        ("stationary_ir", None),
        ("stationary_ir", Some("different-clock")),
        ("spatial_magnitude", Some("fixture-common-clock")),
    ] {
        let mut config = home_cinema_config();
        declare_stationary_captures(&mut config);
        let SpeakerConfig::Single(MeasurementSource::Single(source)) =
            config.speakers.get_mut("sub").unwrap()
        else {
            panic!("single fixture");
        };
        source.provenance.capture_kind = serde_json::from_value(serde_json::json!(kind)).unwrap();
        source.provenance.timing_reference_id = reference.map(str::to_string);
        let result = RoomPipeline::new(RoomPipelineRequest {
            config: &config,
            sample_rate: 48_000.0,
            output_dir: None,
            probe_arrival_overrides: None,
        })
        .run(None)
        .expect("unsupported timing retains magnitude workflow");
        let advisories = result
            .metadata
            .bass_management
            .as_ref()
            .and_then(|report| report.optimization.as_ref())
            .map(|optimization| optimization.advisories.join(";"))
            .unwrap_or_default();
        assert!(
            advisories.contains("crossover_timing_reference_refused"),
            "{kind} {reference:?}: {advisories}"
        );
        assert!(!advisories.contains("xover_summation_search_selected"));
        let optimization = result
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .optimization
            .as_ref()
            .unwrap();
        assert!(
            !optimization.applied,
            "unreferenced phase must not drive alignment"
        );
        assert!(!optimization.phase_available);
        assert!(
            advisories.contains("source_route_optimizer_skipped_unverified_timing")
                || (!optimization.source_results.is_empty()
                    && optimization.source_results.iter().all(|source| source
                        .advisories
                        .iter()
                        .any(|reason| reason.contains("unverified_timing"))))
        );
    }
}

fn finalize_bound(
    result: &autoeq::roomeq::RoomOptimizationResult,
) -> autoeq::roomeq::DspChainOutput {
    let output = result.to_dsp_chain_output();
    assert!(
        output.correction_decisions.is_some(),
        "public workflow output must finalize its ledger without caller assistance"
    );
    let mut payload = serde_json::to_value(&output).unwrap();
    payload
        .as_object_mut()
        .unwrap()
        .remove("correction_decisions");
    let identity = final_ledger::canonical_value_identity(&payload);
    let ledger = output
        .correction_decisions
        .as_ref()
        .expect("finalized output carries a ledger");
    assert!(ledger.validate().is_ok(), "finalized ledger validates");
    let evidence = ledger
        .acceptance_evidence
        .as_ref()
        .expect("public workflow must emit acceptance diagnostics without caller assistance");
    assert!(evidence.matches(&identity.fingerprint));
    assert_eq!(
        evidence.payload["channels"].as_object().unwrap().len(),
        output.channels.len()
    );
    assert!(
        final_ledger::verify_final_binding(ledger, &identity).is_ok(),
        "finalized ledger binds the shipped bytes"
    );
    output
}

#[test]
fn roadmap_correction_admission_records_analysis_normalization() {
    for adaptive in [false, true] {
        let mut config = stereo_config();
        config.optimizer.num_filters = if adaptive { 2 } else { 1 };
        config.optimizer.min_filter_improvement = if adaptive { 0.01 } else { 0.0 };
        for (key, level) in [("left", 80.0), ("right", 82.0)] {
            let SpeakerConfig::Single(MeasurementSource::InMemory(curve)) =
                config.speakers.get_mut(key).unwrap()
            else {
                panic!("in-memory fixture");
            };
            curve.spl.fill(level);
        }
        let result = RoomPipeline::new(RoomPipelineRequest {
            config: &config,
            sample_rate: 48_000.0,
            output_dir: None,
            probe_arrival_overrides: None,
        })
        .run(None)
        .unwrap();
        let stage = result
            .metadata
            .stage_outcomes
            .iter()
            .find(|s| s.stage == "optimizer_input_conditioning")
            .expect("normalization must reach canonical conditioning ledger");
        assert!(!stage.checks.is_empty());
        for (channel, output) in &result.channel_results {
            let source_key = &config.system.as_ref().unwrap().speakers[channel];
            let SpeakerConfig::Single(MeasurementSource::InMemory(original)) =
                &config.speakers[source_key]
            else {
                panic!("in-memory fixture");
            };
            let expected = -original.spl.mean().unwrap();
            assert!(!output.optimizer_evidence.is_empty());
            for (index, run) in output.optimizer_evidence.iter().enumerate() {
                let normalization = run
                    .input_normalization
                    .as_ref()
                    .expect("every ordinary/adaptive run retains preparation evidence");
                let mut legacy = serde_json::to_value(run).unwrap();
                legacy
                    .as_object_mut()
                    .unwrap()
                    .remove("input_normalization");
                let legacy: roomeq_model::OptimizerRunEvidence =
                    serde_json::from_value(legacy).unwrap();
                assert!(
                    legacy.input_normalization.is_none(),
                    "legacy absence must not invent zero conditioning"
                );
                assert!(
                    (normalization.applied_gain_db - expected).abs() < 1e-9,
                    "{channel}: {normalization:?}, expected {expected}"
                );
                let check = stage
                    .checks
                    .iter()
                    .find(|check| check.id == format!("optimizer-conditioning:{channel}:{index}"))
                    .unwrap();
                let payload: serde_json::Value =
                    serde_json::from_str(check.diagnostic.as_ref().unwrap()).unwrap();
                assert_eq!(
                    payload["normalization"],
                    serde_json::to_value(normalization).unwrap()
                );
                assert_eq!(
                    payload["conditioning"]["gains"][0]["gain_db"],
                    normalization.applied_gain_db
                );
                assert_eq!(check.kind, roomeq_model::StageCheckKind::Structural);
            }
        }
        let output = finalize_bound(&result);
        let encoded = serde_json::to_string(&output).unwrap();
        assert!(encoded.contains("input_normalization"));
        assert!(encoded.contains("optimizer_input_conditioning"));
        let _: roomeq_model::DspChainOutput = serde_json::from_str(&encoded).unwrap();
    }
}

#[test]
fn roadmap_correction_admission_records_multi_normalization() {
    let mut config = stereo_config();
    config.optimizer.multi_measurement = Some(roomeq_model::MultiMeasurementConfig::default());
    for source in config.speakers.values_mut() {
        let mut first = phased_curve(80.0, 0.0);
        first.spl.fill(80.0);
        let mut second = first.clone();
        second.spl.fill(86.0);
        *source = SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![first, second]));
    }
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .unwrap();
    let stage = result
        .metadata
        .stage_outcomes
        .iter()
        .find(|stage| stage.stage == "optimizer_input_conditioning")
        .unwrap();
    assert!(!stage.checks.is_empty());
    for (channel, output) in &result.channel_results {
        assert!(!output.optimizer_evidence.is_empty());
        for (run_index, run) in output.optimizer_evidence.iter().enumerate() {
            let receipt = run
                .multi_input_normalization
                .as_ref()
                .expect("multi dispatch receipt");
            assert_eq!(
                receipt.population,
                roomeq_model::NormalizationPopulation::AlignedMeasurements
            );
            assert_eq!(receipt.objectives.len(), 2);
            for (index, (record, expected)) in
                receipt.objectives.iter().zip([-80.0, -86.0]).enumerate()
            {
                assert!((record.applied_gain_db - expected).abs() < 1e-9);
                let id = format!("optimizer-conditioning:{channel}:{run_index}:objective:{index}");
                let check = stage.checks.iter().find(|check| check.id == id).unwrap();
                let payload: serde_json::Value =
                    serde_json::from_str(check.diagnostic.as_ref().unwrap()).unwrap();
                assert_eq!(
                    payload["conditioning"]["gains"][0]["gain_db"],
                    record.applied_gain_db
                );
                assert_eq!(payload["objective_index"], index);
                assert_eq!(payload["objective_population"], "aligned_measurements");
            }
            let mut legacy = serde_json::to_value(run).unwrap();
            legacy
                .as_object_mut()
                .unwrap()
                .remove("multi_input_normalization");
            let legacy: roomeq_model::OptimizerRunEvidence =
                serde_json::from_value(legacy).unwrap();
            assert!(legacy.multi_input_normalization.is_none());
        }
    }
    let encoded = serde_json::to_string(&result.to_dsp_chain_output()).unwrap();
    assert!(encoded.contains("multi_input_normalization"));
    let _: roomeq_model::DspChainOutput = serde_json::from_str(&encoded).unwrap();
}

#[test]
fn roadmap_correction_admission_snapshot_survives_source_file_change() {
    use autoeq_core::{MeasurementRef, MeasurementSingle};
    use roomeq_engine::{PipelineControl, PipelineEvent};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    let directory = tempfile::tempdir_in("/Volumes/home_tmp/tmp").unwrap();
    let path = directory.path().join("source.csv");
    let original = phased_curve(80.0, 0.0);
    let mut csv = String::from("freq,spl,phase\n");
    for index in 0..original.freq.len() {
        csv.push_str(&format!(
            "{},{},{}\n",
            original.freq[index],
            original.spl[index],
            original.phase.as_ref().unwrap()[index]
        ));
    }
    std::fs::write(&path, csv).unwrap();
    let mut config = stereo_config();
    config.speakers.insert(
        "left".into(),
        SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Named {
                path: path.clone(),
                name: Some("left-source".into()),
            },
            speaker_name: None,
            provenance: Default::default(),
        })),
    );
    let changed = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&changed);
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(Some(Box::new(move |_event: &PipelineEvent| {
        if !signal.swap(true, Ordering::SeqCst) {
            std::fs::write(&path, "not valid measurement data").unwrap();
        }
        PipelineControl::Continue
    })))
    .expect("optimization and final replay must consume the frozen numerical response");
    assert!(changed.load(Ordering::SeqCst));
    let stage = result
        .metadata
        .stage_outcomes
        .iter()
        .find(|s| s.stage == "final_seat_input_retention")
        .unwrap();
    let receipt = stage
        .checks
        .iter()
        .map(|check| {
            serde_json::from_str::<serde_json::Value>(check.diagnostic.as_ref().unwrap()).unwrap()
        })
        .find(|receipt| receipt["configuration_source_key"] == "left")
        .unwrap();
    let retained: Curve = serde_json::from_value(receipt["takes"][0]["curve"].clone()).unwrap();
    assert_eq!(retained.freq, original.freq);
    assert_eq!(retained.spl, original.spl);
    assert_eq!(retained.phase, original.phase);
    let preparation = result
        .metadata
        .stage_outcomes
        .iter()
        .find(|stage| stage.stage == "measurement_input_conditioning")
        .unwrap();
    let check = preparation
        .checks
        .iter()
        .find(|check| check.id == "measurement-conditioning:L")
        .unwrap();
    assert!(
        check.passed,
        "source mutation must not change the frozen root binding"
    );
    let binding: serde_json::Value =
        serde_json::from_str(check.diagnostic.as_ref().unwrap()).unwrap();
    assert_eq!(
        binding["source_snapshot_binding"],
        "verified_parsed_curve_snapshot"
    );
    assert_eq!(
        binding["frozen_native_identities"],
        serde_json::json!([retained.content_hash().unwrap()])
    );
    assert_eq!(
        binding["receipt"]["native_identities"],
        binding["frozen_native_identities"]
    );
    assert_eq!(
        receipt["declared_measurement_labels"],
        serde_json::json!(["left-source"])
    );
    finalize_bound(&result);
}

#[test]
fn roadmap_correction_admission_retains_partitioned_replay_inputs() {
    let config = stereo_config();
    let held = phased_curve(80.0, 0.0);
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .with_validation_measurements(HashMap::from([("L".into(), vec![held.clone()])]))
    .run(None)
    .expect("public workflow retains replay inputs");
    let stage = result
        .metadata
        .stage_outcomes
        .iter()
        .find(|stage| stage.stage == "final_seat_input_retention")
        .expect("production receipt must survive final selection");
    assert_eq!(stage.checks.len(), 3);
    let mut training = 0;
    let mut held_out = 0;
    for check in &stage.checks {
        assert_eq!(check.kind, roomeq_model::StageCheckKind::Structural);
        let receipt: serde_json::Value =
            serde_json::from_str(check.diagnostic.as_ref().unwrap()).unwrap();
        assert_eq!(receipt["evidence_kind"], "loaded_response_snapshot");
        assert_eq!(receipt["declared_provenance"]["capture_kind"], "unknown");
        assert!(receipt["raw_refs"][0]["artifact_hash"].is_null());
        assert!(receipt["takes"][0]["reference_id"].is_null());
        assert_eq!(receipt["conditioning"]["gains"], serde_json::json!([]));
        let expected = match receipt["partition"].as_str().unwrap() {
            "training" => {
                training += 1;
                let channel = receipt["configuration_source_key"].as_str().unwrap();
                let SpeakerConfig::Single(MeasurementSource::InMemory(curve)) =
                    &config.speakers[channel]
                else {
                    panic!("fixture must use in-memory training curves");
                };
                curve
            }
            "held_out" => {
                held_out += 1;
                assert!(receipt["configuration_source_key"].is_null());
                &held
            }
            partition => panic!("unexpected partition {partition}"),
        };
        assert_eq!(
            receipt["takes"][0]["curve"],
            serde_json::to_value(expected).unwrap()
        );
    }
    assert_eq!((training, held_out), (2, 1));
    let output = finalize_bound(&result);
    let serialized = serde_json::to_string(&output).unwrap();
    assert!(serialized.contains("final_seat_input_retention"));
    let decoded: roomeq_model::DspChainOutput = serde_json::from_str(&serialized).unwrap();
    assert_eq!(
        serde_json::to_value(decoded).unwrap(),
        serde_json::to_value(output).unwrap()
    );
}

#[test]
fn roadmap_correction_admission_stereo_optimizes_and_finalizes() {
    let config = stereo_config();
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .expect("stereo admission pipeline");
    assert!(!result.channels.is_empty());
    let output = finalize_bound(&result);
    let decisions = &output.correction_decisions.as_ref().unwrap().decisions;
    assert!(
        decisions.iter().any(|record| record
            .reason_codes
            .iter()
            .any(|reason| reason == "optimizer_selected_magnitude_correction")),
        "ordinary PEQ processing must produce real decision records"
    );
    for record in decisions
        .iter()
        .filter(|record| record.decision_id.starts_with("candidate-peq-"))
    {
        assert!(
            record.frequency_band_hz.is_none(),
            "requested limits are not measured affected bands"
        );
        assert!(
            record
                .limits
                .iter()
                .any(|limit| limit.name == "requested_min_frequency")
        );
        assert!(
            record
                .observed
                .iter()
                .any(|value| value.name == "candidate_pre_score")
        );
    }
}

#[test]
fn roadmap_correction_admission_delivered_gains_have_bound_explanations() {
    let mut config = stereo_config();
    // Explicit engineering headroom request, not an audibility threshold.
    config.optimizer.finalization.output_ceiling_dbfs = -1.0;
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .expect("public workflow with final headroom gains");
    let output = finalize_bound(&result);
    let ledger = output.correction_decisions.as_ref().unwrap();
    let mut checked = 0;
    for (name, channel) in &output.channels {
        for plugin in &channel.plugins {
            if plugin.plugin_type != "gain"
                || !matches!(
                    plugin.parameters["label"].as_str(),
                    Some("final_electrical_headroom" | "final_channel_level_alignment")
                )
            {
                continue;
            }
            checked += 1;
            let gain_db = plugin.parameters["gain_db"].as_f64().unwrap();
            assert!(
                ledger.decisions.iter().any(|record| {
                    record.logical_input == *name
                        && record.action
                            == roomeq_model::decision_ledger::DecisionAction::GainAdjust
                        && record.stage == roomeq_model::decision_ledger::DecisionStage::Final
                        && record
                            .reason_codes
                            .iter()
                            .any(|reason| reason == "serialized_final_gain")
                        && record.observed.iter().any(|value| {
                            value.name == "delivered_gain_db"
                                && (value.value - gain_db).abs() < 1e-12
                        })
                }),
                "missing final gain explanation for {name}: {gain_db} dB"
            );
        }
    }
    assert!(checked > 0, "fixture must actually install a final gain");
    assert_python_report(
        &output,
        r#"
import json, sys
from scripts.src.correction_explanation import correction_explanation_html
data = json.load(sys.stdin)
html = correction_explanation_html(data)
assert 'Gain present in the delivered DSP' in html, html
assert 'Attenuation for the configured digital headroom ceiling' in html
assert 'not net parallel-path gain or measured acoustic benefit' in html
assert 'delivered_gain_db:' in html
assert 'Delivered payload and referenced resource bytes verified' in html
"#,
    );
}

#[test]
fn roadmap_correction_admission_report_binds_production_payload() {
    let config = stereo_config();
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .expect("public workflow");
    let output = finalize_bound(&result);
    for (name, channel) in &output.channels {
        let evidence = output
            .correction_decisions
            .as_ref()
            .unwrap()
            .acceptance_evidence
            .as_ref()
            .unwrap();
        assert!(
            !evidence.payload["channels"][name]["bundle"]["magnitude"].is_null(),
            "magnitude unavailable: {name}; pre grid {:?}, post grid {:?}; normalization {:?} / {:?}",
            channel.initial_curve.as_ref().map(|curve| (
                curve.freq.len(),
                curve.freq.first(),
                curve.freq.last()
            )),
            channel.final_curve.as_ref().map(|curve| (
                curve.freq.len(),
                curve.freq.first(),
                curve.freq.last()
            )),
            channel.initial_curve.as_ref().map(|curve| curve.norm_range),
            channel.final_curve.as_ref().map(|curve| curve.norm_range)
        );
    }
    assert!(
        !output
            .correction_decisions
            .as_ref()
            .unwrap()
            .decisions
            .is_empty()
    );
    // Python independently recomputes the digest from the actual serialized
    // public output; it does not receive an oracle pass flag from Rust.
    let script = r#"
import copy, json, sys, tempfile
from pathlib import Path
from scripts.src.payload_binding import verify_payload_binding
from scripts.src.correction_explanation import correction_explanation_html
from scripts.src.loaders import load_roomeq_json
from scripts.src.report import create_html_report, create_comparison_html_report
data = json.load(sys.stdin)
valid, reason, _ = verify_payload_binding(data)
assert valid, reason
html = correction_explanation_html(data)
assert 'Delivered payload and referenced resource bytes verified' in html
from scripts.src.acceptance_views import acceptance_views_html
views = acceptance_views_html(data)
assert 'General magnitude' in views, views
assert 'Fine magnitude' in views
assert 'no complete routed true-peak trial' in views
assert 'No complete acoustic/headroom acceptance' in views
tampered_views = copy.deepcopy(data)
tampered_views['correction_decisions']['acceptance_evidence']['payload']['scope'] = 'forged'
assert 'acceptance-view payload or graph binding changed' in acceptance_views_html(tampered_views)
changed = copy.deepcopy(data)
changed.setdefault('global_plugins', []).append({'plugin_type': 'gain', 'parameters': {'gain_db': 12.0}})
assert not verify_payload_binding(changed)[0]
assert 'Delivered payload changed' in correction_explanation_html(changed)
assert '0 applied delivery claims' in correction_explanation_html(changed)
assert 'Delivered payload changed' in acceptance_views_html(changed)
with tempfile.TemporaryDirectory(dir='/Volumes/home_tmp/tmp') as directory:
    root = Path(directory)
    result_path = root / 'production.json'
    result_path.write_text(json.dumps(data), encoding='utf-8')
    loaded = load_roomeq_json(result_path)
    assert verify_payload_binding(loaded)[0]
    report_path = root / 'report.html'
    create_html_report(loaded, report_path, result_path)
    rendered = report_path.read_text()
    assert rendered.index('<section class="playback-status"') < rendered.index('<section class="correction-explanation"')
    assert 'Delivered payload and referenced resource bytes verified' in rendered
    assert rendered.index('<section class="correction-explanation"') < rendered.index('<section class="acceptance-views"')
    assert 'General magnitude' in rendered
    create_comparison_html_report([('original', loaded), ('mutated', changed)], root / 'comparison.html')
    compared = (root / 'comparison.html').read_text()
    assert 'Delivered payload changed' in compared
    assert 'Not approved for playback' in compared
assert compared.count('<section class="acceptance-views"') == 2
"#;
    assert_python_report(&output, script);
}

fn assert_python_report(output: &autoeq::roomeq::DspChainOutput, script: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = std::env::var_os("ROOMEQ_REPORT_PYTHON").unwrap_or_else(|| {
        let venv = root.join("venv/bin/python");
        if venv.is_file() {
            venv.into_os_string()
        } else {
            "python3".into()
        }
    });
    let mut child = Command::new(python)
        .arg("-c")
        .arg(script)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Python 3 with report dependencies is required (ROOMEQ_REPORT_PYTHON)");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&output).unwrap())
        .unwrap();
    let checked = child.wait_with_output().unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
}

#[test]
fn roadmap_correction_admission_phase_application_survives_finalization() {
    use roomeq_model::decision_ledger::{DecisionAction, DecisionStage, DecisionStatus};
    use std::fmt::Write as _;

    // Exercise the real CSV/provenance loader and both production phase callers.
    // These are synthetic declarations, not independently validated acoustic captures.
    // The idealized 125 ms gate and 50 m first-reflection path deliberately
    // supply full-band support for this numerical fixture, not a room setup recipe.
    let temp = tempfile::TempDir::new_in("/Volumes/home_tmp/tmp").unwrap();
    for (generic, retained) in [(false, true), (true, true), (false, false), (true, false)] {
        let output_dir = temp
            .path()
            .join(format!("generic-{generic}-retained-{retained}"));
        std::fs::create_dir(&output_dir).unwrap();
        let measurement = output_dir.join("declared-phase.csv");
        let base = autoeq::roomeq::synthetic::generate_flat_curve(20.0, 20_000.0, 128);
        let modes = if retained {
            vec![math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                100.0,
                48_000.0,
                1.0,
                6.0,
            )]
        } else {
            Vec::new()
        };
        let curve =
            autoeq::roomeq::synthetic::generate_channel_curve(&base, &modes, 0.2, 0.0, 7, 48_000.0);
        let mut csv = String::from("frequency_hz,spl,phase_deg,noise_floor_db\n");
        for index in 0..curve.freq.len() {
            writeln!(
                csv,
                "{},{},{},{}",
                curve.freq[index],
                curve.spl[index],
                curve.phase.as_ref().unwrap()[index],
                curve.spl[index] - 50.0
            )
            .unwrap();
        }
        std::fs::write(&measurement, csv).unwrap();
        let mut config = stereo_config();
        if generic {
            config.system = None;
        }
        for speaker in config.speakers.values_mut() {
            *speaker = SpeakerConfig::Single(
                serde_json::from_value(serde_json::json!({
                    "path": measurement,
                    "provenance": {
                        "capture_kind": "direct_sound",
                        "timing_reference_id": "fixture-clock",
                        "has_direct_angular": true,
                        "direct_sound": {
                            "facts": {
                                "gate_s": 0.125,
                                "direct_path_m": 1.0,
                                "first_reflection_path_m": 50.0,
                                "sound_speed_m_s": 343.0,
                                "angular": {"angles_deg": [0.0, -30.0, 30.0]},
                                "averaging": "stationary",
                                "capture_kind": "direct_sound",
                                "sample_rate_hz": 48000.0
                            },
                            "policy": {
                                "version": "quasi-anechoic-v1",
                                "cycles_for_valid_band": 2.0,
                                "min_off_axis_count": 2,
                                "min_off_axis_abs_deg": 15.0
                            }
                        }
                    }
                }))
                .unwrap(),
            );
        }
        config.optimizer.phase_correction = Some(roomeq_model::MixedPhaseSerdeConfig {
            max_fir_length_ms: 10.0,
            max_correction_latency_ms: Some(6.0),
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
            assessment: Default::default(),
        });
        let result = RoomPipeline::new(RoomPipelineRequest {
            config: &config,
            sample_rate: 48_000.0,
            output_dir: Some(&output_dir),
            probe_arrival_overrides: None,
        })
        .run(None)
        .expect("supported phase workflow");
        let saved: roomeq_model::DspChainOutput =
            serde_json::from_slice(&serde_json::to_vec(&result.to_dsp_chain_output()).unwrap())
                .unwrap();
        let ledger = saved.correction_decisions.as_ref().expect("final ledger");
        for (name, channel) in &saved.channels {
            let records: Vec<_> = ledger
                .decisions
                .iter()
                .filter(|record| {
                    record.physical_output == *name
                        && record.action == DecisionAction::PhaseCorrect
                        && record.stage == DecisionStage::Final
                })
                .collect();
            assert_eq!(records.len(), 1, "{name}: {:?}", ledger.decisions);
            let record = records[0];
            assert_eq!(
                record.status,
                if retained {
                    DecisionStatus::Applied
                } else {
                    DecisionStatus::Reverted
                },
                "generic={generic}: {record:?}"
            );
            assert_eq!(record.stage, DecisionStage::Final, "{record:?}");
            if !retained {
                assert!(
                    !channel
                        .plugins
                        .iter()
                        .any(|plugin| plugin.plugin_type == "convolution")
                );
                let attempted = ledger
                    .decisions
                    .iter()
                    .find(|candidate| record.supersedes_ids.contains(&candidate.decision_id))
                    .expect("reversion must preserve the attempted phase correction");
                assert_eq!(attempted.stage, DecisionStage::Provisional);
                assert!(
                    attempted
                        .reason_codes
                        .iter()
                        .any(|reason| reason == "phase_fir_removed_before_delivery")
                );
                continue;
            }
            assert!(
                record
                    .reason_codes
                    .iter()
                    .any(|reason| reason == "target_ok")
            );
            assert!(
                record
                    .reason_codes
                    .iter()
                    .any(|reason| reason == "phase_fir_realized")
            );
            assert!(
                record.frequency_band_hz.is_none(),
                "design scope is not realized support"
            );
            assert!(
                record
                    .observed
                    .iter()
                    .any(|value| value.name == "requested_phase_band_low")
            );
            let plugin = channel
                .plugins
                .iter()
                .find(|plugin| plugin.plugin_type == "convolution")
                .unwrap_or_else(|| panic!("phase decision must describe a delivered convolution: generic={generic}, channel={name}, plugins={:?}, global={:?}", channel.plugins, saved.global_plugins));
            let path = plugin.parameters["ir_file"].as_str().unwrap();
            assert!(
                output_dir.join(path).is_file(),
                "phase artifact {path} must exist"
            );
        }
        let mut payload = serde_json::to_value(&saved).unwrap();
        payload
            .as_object_mut()
            .unwrap()
            .remove("correction_decisions");
        let binding = ledger.payload_binding.as_ref().unwrap();
        assert!(binding.matches(&payload, &binding.graph_identity));
    }
}

#[test]
fn roadmap_correction_admission_phase_refusal_survives_workflow_and_serialization() {
    use roomeq_model::decision_ledger::{DecisionAction, DecisionStatus};
    // Exercise both the named-topology and generic production phase callers.
    for generic in [false, true] {
        let mut config = stereo_config();
        if generic {
            config.system = None;
        }
        config.optimizer.phase_correction = Some(roomeq_model::MixedPhaseSerdeConfig {
            max_fir_length_ms: 10.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
            assessment: Default::default(),
            max_correction_latency_ms: None,
        });
        let result = RoomPipeline::new(RoomPipelineRequest {
            config: &config,
            sample_rate: 48_000.0,
            output_dir: None,
            probe_arrival_overrides: None,
        })
        .run(None)
        .expect("phase refusal pipeline");
        let output = finalize_bound(&result);
        let json = serde_json::to_vec(&output).unwrap();
        let saved: roomeq_model::DspChainOutput = serde_json::from_slice(&json).unwrap();
        let ledger = saved.correction_decisions.unwrap();
        assert_eq!(result.channels.len(), 2);
        for channel in result.channels.keys() {
            let refusal = ledger
                .decisions
                .iter()
                .find(|record| {
                    record.physical_output == *channel
                        && record.action == DecisionAction::PhaseCorrect
                })
                .unwrap_or_else(|| {
                    panic!(
                        "phase refusal missing: generic={generic}, channel={channel}, records={:?}",
                        ledger.decisions
                    )
                });
            assert_eq!(refusal.status, DecisionStatus::InsufficientEvidence);
            assert!(
                refusal
                    .reason_codes
                    .iter()
                    .any(|reason| reason == "phase_evidence_refused")
            );
            assert!(!refusal.is_final_claim());
            assert!(refusal.validate().is_ok());
        }
    }
}

#[test]
fn roadmap_correction_admission_routed_missing_phase_publishes_unverified_baseline() {
    let mut config = home_cinema_config();
    for speaker in config.speakers.values_mut() {
        if let SpeakerConfig::Single(MeasurementSource::InMemory(curve)) = speaker {
            curve.phase = None;
        }
    }
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .expect("missing routed phase must publish an unverified structural baseline");
    let report = result.metadata.correction_acceptance.as_ref().unwrap();
    assert!(
        !report.accepted,
        "missing phase cannot prove acoustic acceptance"
    );
    let advisories = result
        .metadata
        .stage_outcomes
        .iter()
        .flat_map(|stage| &stage.advisories)
        .cloned()
        .collect::<Vec<_>>()
        .join(";");
    assert!(
        advisories.contains("structural_baseline_published"),
        "{advisories}"
    );
    assert!(
        advisories.contains("baseline_evidence_insufficient"),
        "{advisories}"
    );
    assert!(advisories.contains("phase"), "{advisories}");
    assert!(
        result.metadata.bass_management.is_some(),
        "structural routing must survive"
    );
    finalize_bound(&result);
}

#[test]
fn roadmap_correction_admission_home_cinema_runs_summation_search() {
    let mut config = home_cinema_config();
    declare_stationary_captures(&mut config);
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .expect("home-cinema admission pipeline");
    assert!(!result.channels.is_empty());
    let advisories = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|report| report.optimization.as_ref())
        .map(|optimization| optimization.advisories.join(";"))
        .unwrap_or_default();
    assert!(
        advisories.contains("xover_summation_search_selected"),
        "summation search must run in production; advisories: {advisories}"
    );
    finalize_bound(&result);
}

fn multisub_group(joint: bool) -> autoeq::roomeq::MultiSubGroup {
    autoeq::roomeq::MultiSubGroup {
        name: "subs".to_string(),
        speaker_name: None,
        subwoofers: vec![
            MeasurementSource::InMemory(phased_curve(80.0, 0.0)),
            MeasurementSource::InMemory(phased_curve(78.0, 0.001)),
        ],
        allpass_optimization: false,
        joint_optimization: joint,
    }
}

fn multisub_room_config() -> RoomConfig {
    RoomConfig {
        version: default_config_version(),
        system: None,
        speakers: HashMap::new(),
        crossovers: None,
        target_curve: None,
        optimizer: tiny_optimizer(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    }
}

fn declared_joint_sub_fixture() -> autoeq::roomeq::MultiSubGroup {
    let mut group = multisub_group(true);
    group.subwoofers = (0..2)
        .map(|sub| {
            let measurements: Vec<_> = (0..3).map(|seat| {
            let curve = phased_curve(80.0 + (sub + seat) as f64, 0.001 * sub as f64);
            serde_json::json!({
                "name": format!("seat-{seat}"), "frequencies": curve.freq.to_vec(),
                "magnitude_db": curve.spl.to_vec(), "phase_deg": curve.phase.unwrap().to_vec()
            })
        }).collect();
            serde_json::from_value(serde_json::json!({
            "measurements": measurements,
            "speaker_name": "identical-sub-model",
            "provenance": {"capture_kind": "stationary_ir", "timing_reference_id": "fixture-clock"}
        })).unwrap()
        })
        .collect();
    group
}

#[test]
fn roadmap_correction_joint_drive_search_reaches_final_graph_trials() {
    for source_specific_peak in [false, true] {
        let mut config = multisub_room_config();
        config.optimizer.min_freq = 30.0;
        config.optimizer.max_freq = 120.0;
        config.optimizer.min_db = -4.0;
        config.optimizer.finalization.default_input_peak = 0.1;
        config.optimizer.finalization.physical_drive_weight = 1000.0;
        let outputs: serde_json::Map<String, serde_json::Value> = (0..2)
        .map(|index| {
            let name = format!("subs_{}", index + 1);
            let id = serde_json::json!(["driver", "subs", index, name]).to_string();
            (
                id,
                serde_json::json!([{
                    "quantity":"voltage_rms", "calibration_id":"synthetic-drive",
                    "reference_conditions_id":"load", "limit_conditions_id":"load",
                    "sine_duration_seconds":1.0, "reference_output_peak":0.1,
                    "linear_valid_output_peak":1.0, "frequencies_hz":[40.0,80.0],
                    "demand_at_reference": if index == 0 { vec![0.01,0.01] } else { vec![1.0,1.0] }, "limits":[10.0,10.0]
                }]),
            )
        })
        .collect();
        config.optimizer.finalization.physical_drive =
            Some(serde_json::from_value(serde_json::json!({"outputs": outputs})).unwrap());
        let mut group = declared_joint_sub_fixture();
        for (sub, source) in group.subwoofers.iter_mut().enumerate() {
            let mut value = serde_json::to_value(&*source).unwrap();
            for measurement in value["measurements"].as_array_mut().unwrap() {
                let frequencies = measurement["frequencies"].as_array().unwrap().clone();
                for (frequency, magnitude) in frequencies
                    .iter()
                    .zip(measurement["magnitude_db"].as_array_mut().unwrap())
                {
                    let offset = (frequency.as_f64().unwrap() / 70.0).log2();
                    let peak = if !source_specific_peak || sub == 1 {
                        10.0 * (-offset * offset).exp()
                    } else {
                        0.0
                    };
                    *magnitude = serde_json::json!(magnitude.as_f64().unwrap() + peak);
                }
            }
            *source = serde_json::from_value(value).unwrap();
        }
        config
            .speakers
            .insert("subs".into(), SpeakerConfig::MultiSub(group));
        let result = RoomPipeline::new(RoomPipelineRequest {
            config: &config,
            sample_rate: 48_000.0,
            output_dir: None,
            probe_arrival_overrides: None,
        })
        .run(None)
        .expect("calibrated joint candidate workflow");
        if source_specific_peak {
            assert_joint_drive_weight_discriminates(&config);
        }
        let trials: Vec<_> = result
            .metadata
            .stage_outcomes
            .iter()
            .flat_map(|stage| &stage.checks)
            .filter(|check| check.id.starts_with("joint_drive_gain_"))
            .collect();
        let refined = result.channels["subs"]
            .drivers
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(|driver| &driver.plugins)
            .any(|plugin| plugin.parameters["label"] == "joint_physical_drive_refinement");
        let gain_delay_trials: Vec<_> = trials
            .iter()
            .copied()
            .filter(|trial| trial.id.starts_with("joint_drive_gain_delay_"))
            .collect();
        let gain_only_trials: Vec<_> = trials
            .iter()
            .copied()
            .filter(|trial| !trial.id.starts_with("joint_drive_gain_delay_"))
            .collect();
        if source_specific_peak {
            assert!(refined, "source-specific drive must select a gain trim");
            assert!(trials.iter().any(|trial| trial.passed), "{trials:?}");
            let drivers = result.channels["subs"].drivers.as_ref().unwrap();
            assert!(
                drivers[0]
                    .plugins
                    .iter()
                    .all(|plugin| plugin.parameters["label"] != "joint_physical_drive_refinement")
            );
            let gain: f64 = drivers[1]
                .plugins
                .iter()
                .filter(|plugin| plugin.plugin_type == "gain")
                .map(|plugin| plugin.parameters["gain_db"].as_f64().unwrap())
                .sum();
            assert!((config.optimizer.min_db..=config.optimizer.max_db).contains(&gain));
            let output = result.to_dsp_chain_output();
            assert!(
                !output.channels["subs"]
                    .joint_sub
                    .as_ref()
                    .unwrap()
                    .channel_processing_matches
            );
            let check = result
                .metadata
                .stage_outcomes
                .iter()
                .find(|stage| stage.stage == "final_candidate_objective")
                .unwrap()
                .checks
                .first()
                .unwrap();
            let score: serde_json::Value =
                serde_json::from_str(check.diagnostic.as_ref().unwrap()).unwrap();
            let expected = score["mean_seat_target_error_db"].as_f64().unwrap()
                + config.optimizer.finalization.physical_drive_weight
                    * score["worst_declared_drive_utilization"]
                        .as_f64()
                        .unwrap()
                        .powi(2);
            assert!((check.observed.unwrap() - expected).abs() < 1e-10);
        } else {
            assert!(
                gain_only_trials.iter().all(|trial| !trial.passed),
                "gain-only proposals must still fail the primary-target gate: {trials:?}"
            );
            assert!(
                refined,
                "this fixture must select a gain-and-delay candidate unavailable to gain-only search"
            );
            let drivers = result.channels["subs"].drivers.as_ref().unwrap();
            assert!(
                drivers
                    .iter()
                    .any(|driver| driver.plugins.iter().any(|plugin| {
                        plugin.parameters["label"] == "joint_physical_drive_delay_refinement"
                    }))
            );
            let selected_score = result
                .metadata
                .stage_outcomes
                .iter()
                .find(|stage| stage.stage == "final_candidate_objective")
                .unwrap()
                .checks[0]
                .observed
                .unwrap();
            assert!(
                gain_delay_trials.iter().any(|trial| {
                    trial.passed
                        && trial
                            .observed
                            .is_some_and(|score| (score - selected_score).abs() < 1e-9)
                }),
                "selected gain must belong to a passing combined trial"
            );
        }
        assert!(
            !trials.is_empty(),
            "calibrated ranking must explore array controls, not only common trims"
        );
        assert!(
            trials
                .iter()
                .all(|trial| trial.observed.is_some() || trial.diagnostic.is_some())
        );
        assert!(
            result
                .metadata
                .stage_outcomes
                .iter()
                .any(|stage| stage.stage == "final_graph_declared_physical_drive")
        );
        // Independent amplitude oracle for this gain/delay-only fixture. Read
        // the delivered JSON, not the optimizer controls or reported demand.
        let delivered: autoeq::roomeq::DspChainOutput =
            serde_json::from_slice(&serde_json::to_vec(&result.to_dsp_chain_output()).unwrap())
                .unwrap();
        let physical = delivered
            .metadata
            .as_ref()
            .unwrap()
            .stage_outcomes
            .iter()
            .find(|stage| stage.stage == "final_graph_declared_physical_drive")
            .unwrap();
        let chain = &delivered.channels["subs"];
        let policy = config
            .optimizer
            .finalization
            .physical_drive
            .as_ref()
            .unwrap();
        let mut worst = 0.0_f64;
        for driver in chain.drivers.as_ref().unwrap() {
            let mut gain_db = 0.0;
            for plugin in delivered
                .global_plugins
                .iter()
                .chain(&chain.plugins)
                .chain(&driver.plugins)
            {
                match plugin.plugin_type.as_str() {
                    "gain" => gain_db += plugin.parameters["gain_db"].as_f64().unwrap(),
                    "delay" => {} // A pure delay has unit steady-sine magnitude.
                    other => {
                        panic!("independent fixture oracle does not support {other}: {plugin:?}")
                    }
                }
            }
            let peak = 0.1 * 10.0_f64.powf(gain_db / 20.0);
            let id = serde_json::json!(["driver", "subs", driver.index, driver.name]).to_string();
            let declaration = &policy.outputs[&id][0];
            let assessment: serde_json::Value = physical
                .checks
                .iter()
                .map(|check| {
                    serde_json::from_str::<serde_json::Value>(check.diagnostic.as_ref().unwrap())
                        .unwrap()
                })
                .find(|assessment| assessment["output"] == id)
                .unwrap();
            for index in 0..declaration.frequencies_hz.len() {
                let demand = declaration.demand_at_reference[index] * peak
                    / declaration.reference_output_peak;
                let ratio = demand / declaration.limits[index];
                worst = worst.max(ratio);
                assert!(
                    (assessment["digital_output_peaks"][index].as_f64().unwrap() - peak).abs()
                        < 1e-12
                );
                assert!((assessment["demands"][index].as_f64().unwrap() - demand).abs() < 1e-12);
                assert!(ratio <= 1.0 && peak <= declaration.linear_valid_output_peak);
            }
        }
        if source_specific_peak {
            let selected = result
                .metadata
                .stage_outcomes
                .iter()
                .find(|stage| stage.stage == "final_candidate_objective")
                .unwrap();
            let score: serde_json::Value =
                serde_json::from_str(selected.checks[0].diagnostic.as_ref().unwrap()).unwrap();
            assert!(
                (score["worst_declared_drive_utilization"].as_f64().unwrap() - worst).abs() < 1e-12
            );
            let mut acoustic_config = config.clone();
            acoustic_config.optimizer.finalization.physical_drive_weight = 0.0;
            let acoustic = RoomPipeline::new(RoomPipelineRequest {
                config: &acoustic_config,
                sample_rate: 48_000.0,
                output_dir: None,
                probe_arrival_overrides: None,
            })
            .run(None)
            .unwrap();
            assert!(
                acoustic
                    .metadata
                    .stage_outcomes
                    .iter()
                    .flat_map(|stage| &stage.checks)
                    .all(|check| !check.id.starts_with("joint_drive_gain_"))
            );
            let acoustic_output = acoustic.to_dsp_chain_output();
            let original_drivers = acoustic_output.channels["subs"].drivers.as_ref().unwrap();
            let selected_drivers = chain.drivers.as_ref().unwrap();
            assert_eq!(
                serde_json::to_value(&original_drivers[0].plugins).unwrap(),
                serde_json::to_value(&selected_drivers[0].plugins).unwrap()
            );
            // This fixture has no frequency-dependent electrical filters.
            // Compare the complete demanding branch, including common gains.
            let branch_gain = |graph: &autoeq::roomeq::DspChainOutput| -> f64 {
                let channel = &graph.channels["subs"];
                graph
                    .global_plugins
                    .iter()
                    .chain(&channel.plugins)
                    .chain(&channel.drivers.as_ref().unwrap()[1].plugins)
                    .map(|plugin| match plugin.plugin_type.as_str() {
                        "gain" => plugin.parameters["gain_db"].as_f64().unwrap(),
                        "delay" => 0.0,
                        other => panic!("unsupported independent comparison stage {other}"),
                    })
                    .sum()
            };
            let initial_peak = 0.1 * 10.0_f64.powf(branch_gain(&acoustic_output) / 20.0);
            let selected_peak = 0.1 * 10.0_f64.powf(branch_gain(&delivered) / 20.0);
            assert!(
                selected_peak < initial_peak - 1e-12,
                "independent demanding-branch peak: {initial_peak} -> {selected_peak}"
            );
        }
        finalize_bound(&result);
        if !source_specific_peak {
            if let Ok(path) = std::env::var("ROOMEQ_SELECTED_GAIN_DELAY_ARTIFACT") {
                std::fs::write(
                    path,
                    serde_json::to_vec_pretty(&result.to_dsp_chain_output()).unwrap(),
                )
                .expect("write selected gain-delay QA artifact");
            }
        }
    }
}

#[test]
fn roadmap_correction_joint_drive_pair_reaches_public_finalization() {
    let mut config = multisub_room_config();
    config.optimizer.min_freq = 30.0;
    config.optimizer.max_freq = 120.0;
    config.optimizer.min_db = -4.0;
    config.optimizer.finalization.default_input_peak = 0.1;
    config.optimizer.finalization.physical_drive_weight = 1000.0;
    let outputs: serde_json::Map<String, serde_json::Value> = (0..3)
        .map(|index| {
            let name = format!("subs_{}", index + 1);
            let id = serde_json::json!(["driver", "subs", index, name]).to_string();
            (
                id,
                serde_json::json!([{
                    "quantity": "voltage_rms", "calibration_id": "synthetic-pair-drive",
                    "reference_conditions_id": "load", "limit_conditions_id": "load",
                    "sine_duration_seconds": 1.0, "reference_output_peak": 0.1,
                    "linear_valid_output_peak": 1.0, "frequencies_hz": [40.0, 80.0],
                    "demand_at_reference": [0.01, 0.01], "limits": [10.0, 10.0]
                }]),
            )
        })
        .collect();
    config.optimizer.finalization.physical_drive =
        Some(serde_json::from_value(serde_json::json!({ "outputs": outputs })).unwrap());
    let mut group = declared_joint_sub_fixture();
    group.subwoofers.push(group.subwoofers[1].clone());
    config
        .speakers
        .insert("subs".into(), SpeakerConfig::MultiSub(group));
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .expect("three-driver calibrated joint candidate workflow");
    let pairs: Vec<_> = result
        .metadata
        .stage_outcomes
        .iter()
        .flat_map(|stage| &stage.checks)
        .filter(|check| check.id.starts_with("joint_drive_gain_pair_"))
        .collect();
    assert!(
        !pairs.is_empty(),
        "pairwise proposals must reach public selection"
    );
    assert!(
        pairs
            .iter()
            .all(|check| check.observed.is_some() || check.diagnostic.is_some())
    );
    assert!(
        pairs.iter().any(|check| check.passed),
        "at least one pair must survive complete-graph gates: {pairs:?}"
    );
    let best_pair = pairs
        .iter()
        .filter_map(|check| check.observed)
        .fold(f64::INFINITY, f64::min);
    let best_single = result
        .metadata
        .stage_outcomes
        .iter()
        .flat_map(|stage| &stage.checks)
        .filter(|check| {
            check.id.starts_with("joint_drive_gain_")
                && !check.id.starts_with("joint_drive_gain_pair_")
                && !check.id.starts_with("joint_drive_gain_delay_")
        })
        .filter_map(|check| check.observed)
        .fold(f64::INFINITY, f64::min);
    assert!(best_single.is_finite() && best_pair.is_finite());
    let gain_delay_trials: Vec<_> = result
        .metadata
        .stage_outcomes
        .iter()
        .flat_map(|stage| &stage.checks)
        .filter(|check| check.id.starts_with("joint_drive_gain_delay_"))
        .collect();
    assert!(
        !gain_delay_trials.is_empty(),
        "combined gain-and-delay proposals must reach public selection"
    );
    assert!(
        gain_delay_trials.iter().any(|check| check.passed),
        "at least one combined candidate must survive complete-graph gates: {gain_delay_trials:?}"
    );
    assert!(
        best_pair < best_single - 1e-9,
        "pair must beat every valid single-driver alternative: pair={best_pair}, single={best_single}"
    );
    assert!(pairs.iter().all(|check| {
        !check
            .diagnostic
            .as_deref()
            .unwrap_or("")
            .contains("ownership")
    }));
    let drivers = result.channels["subs"].drivers.as_ref().unwrap();
    let pair_selected = drivers[1..].iter().all(|driver| {
        driver
            .plugins
            .iter()
            .any(|plugin| plugin.parameters["label"] == "joint_physical_drive_refinement")
    });
    let delay_selected = drivers.iter().any(|driver| {
        driver
            .plugins
            .iter()
            .any(|plugin| plugin.parameters["label"] == "joint_physical_drive_delay_refinement")
    });
    assert!(
        pair_selected || delay_selected,
        "selected output has no joint-control trim"
    );
    if delay_selected {
        let selected_score = result
            .metadata
            .stage_outcomes
            .iter()
            .find(|stage| stage.stage == "final_candidate_objective")
            .unwrap()
            .checks[0]
            .observed
            .unwrap();
        assert!(
            selected_score < best_pair - 1e-9,
            "selected delay must beat the best valid gain pair: delay={selected_score}, pair={best_pair}"
        );
        let selected_delay = result
            .metadata
            .stage_outcomes
            .iter()
            .flat_map(|stage| &stage.checks)
            .find(|check| {
                check.id.starts_with("joint_drive_delay_")
                    && check
                        .observed
                        .is_some_and(|score| (score - selected_score).abs() < 1e-9)
            })
            .expect("selected delay must have a passing complete-graph trial");
        let suffix = selected_delay
            .id
            .strip_prefix("joint_drive_delay_subs_")
            .unwrap();
        let (index, target_ms) = suffix.split_once("_to_").unwrap();
        let index: usize = index.parse().unwrap();
        let target_ms: f64 = target_ms.parse().unwrap();
        let delay_inventory: Vec<(f64, f64)> = drivers
            .iter()
            .map(|driver| {
                let mut emitted_ms = 0.0;
                let mut added_ms = 0.0;
                for plugin in driver
                    .plugins
                    .iter()
                    .filter(|plugin| plugin.plugin_type == "delay")
                {
                    let delay = plugin.parameters["delay_ms"].as_f64().unwrap();
                    assert!(delay.is_finite() && delay >= 0.0);
                    emitted_ms += delay;
                    if plugin.parameters["label"] == "joint_physical_drive_delay_refinement" {
                        assert_eq!(plugin.parameters["room_eq_stage"], "post_route");
                        assert_eq!(plugin.parameters["room_eq_correction_delay"], true);
                        assert!(delay > 0.0);
                        added_ms += delay;
                    }
                }
                (emitted_ms, added_ms)
            })
            .collect();
        assert!(
            (delay_inventory[index].0 - delay_inventory[0].0 - target_ms).abs() < 1e-9,
            "selected physical delay target is absent from delivered plugins"
        );
        for other in 1..drivers.len() {
            if other != index {
                assert!(
                    (delay_inventory[other].1 - delay_inventory[0].1).abs() < 1e-9,
                    "unselected correction delay moved relative to reference"
                );
            }
        }
        let delivered: autoeq::roomeq::DspChainOutput =
            serde_json::from_slice(&serde_json::to_vec(&result.to_dsp_chain_output()).unwrap())
                .unwrap();
        assert_eq!(
            serde_json::to_value(delivered.channels["subs"].drivers.as_ref().unwrap()).unwrap(),
            serde_json::to_value(drivers).unwrap(),
            "serialized public output lost physical delay controls"
        );
    }
    finalize_bound(&result);
    if let Ok(path) = std::env::var("ROOMEQ_SELECTED_DELAY_ARTIFACT") {
        assert!(
            delay_selected,
            "QA handoff requires a selected physical delay"
        );
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&result.to_dsp_chain_output()).unwrap(),
        )
        .expect("write selected-delay QA artifact");
    }
}

#[test]
fn roadmap_correction_physical_phase_refusal_is_delivered_with_opt_in_policy() {
    for (mixed, upper_band) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut config = multisub_room_config();
        config.optimizer.min_freq = 30.0;
        config.optimizer.max_freq = if upper_band { 400.0 } else { 120.0 };
        config.optimizer.processing_mode = if mixed {
            ProcessingMode::MixedPhase
        } else {
            ProcessingMode::PhaseLinear
        };
        config.optimizer.fir = Some(autoeq::roomeq::FirConfig {
            placement: autoeq::roomeq::FirPlacement::PerDriver,
            phase: "kirkeby".into(),
            correct_excess_phase: true,
            taps: 1024,
            ..Default::default()
        });
        let mut group = declared_joint_sub_fixture();
        group.joint_optimization = false;
        // A coherent physical FIR needs one phase-referenced capture per
        // branch, not the phase-less spatial power average of several seats.
        for (index, source) in group.subwoofers.iter_mut().enumerate() {
            let curve = phased_curve(80.0 + index as f64, 0.001 * index as f64);
            *source = MeasurementSource::Single(autoeq_core::MeasurementSingle {
                measurement: autoeq_core::MeasurementRef::Inline(autoeq_core::InlineMeasurement {
                    frequencies: curve.freq.to_vec(),
                    magnitude_db: curve.spl.to_vec(),
                    phase_deg: curve.phase.map(|phase| phase.to_vec()),
                    name: Some("fixture-seat".into()),
                    wav_path: None,
                    csv_path: None,
                }),
                speaker_name: None,
                provenance: autoeq_core::MeasurementProvenance {
                    capture_kind: autoeq_core::ProvenanceCaptureKind::StationaryIr,
                    timing_reference_id: Some("fixture-clock".into()),
                    ..Default::default()
                },
            });
        }
        config
            .speakers
            .insert("subs".into(), SpeakerConfig::MultiSub(group));
        let dir = tempfile::tempdir().unwrap();
        let run = |config: &RoomConfig| {
            RoomPipeline::new(RoomPipelineRequest {
                config,
                sample_rate: 48_000.0,
                output_dir: Some(dir.path()),
                probe_arrival_overrides: None,
            })
            .run(None)
        };
        let error = run(&config).expect_err("default policy must still abort unsupported phase");
        let refusal_kind = if upper_band {
            "target policy refused"
        } else {
            "excess-phase assessment"
        };
        assert!(error.to_string().contains(refusal_kind), "{error}");
        config
            .optimizer
            .mixed_phase
            .get_or_insert_with(|| serde_json::from_value(serde_json::json!({})).unwrap())
            .assessment
            .retain_magnitude_on_refusal = true;
        let result = run(&config).expect("opt-in must retain supported magnitude processing");
        finalize_bound(&result);
        let output = result.to_dsp_chain_output();
        let ledger = output.correction_decisions.as_ref().unwrap();
        let refusals: Vec<_> = ledger
            .decisions
            .iter()
            .filter(|record| {
                record.action == roomeq_model::decision_ledger::DecisionAction::PhaseCorrect
            })
            .collect();
        assert!(
            refusals.iter().any(|record| record
                .reason_codes
                .iter()
                .any(|reason| reason == "existing_magnitude_retained_by_policy")),
            "{refusals:?}"
        );
        assert!(
            refusals
                .iter()
                .all(|record| record.status
                    != roomeq_model::decision_ledger::DecisionStatus::Applied)
        );
        let refusal = refusals
            .iter()
            .find(|record| {
                record
                    .reason_codes
                    .iter()
                    .any(|reason| reason == "existing_magnitude_retained_by_policy")
            })
            .unwrap();
        // The bound final ledger retains refusal history; it must not promote
        // missing evidence into a final claim of delivered phase processing.
        assert_eq!(
            refusal.status,
            roomeq_model::decision_ledger::DecisionStatus::InsufficientEvidence
        );
        assert_eq!(
            refusal.stage,
            roomeq_model::decision_ledger::DecisionStage::Provisional
        );
        assert!(refusal.final_graph_identity.is_none());
        assert_eq!(refusal.measurement_refs.len(), 2);
        assert!(
            refusal
                .reason_codes
                .iter()
                .any(|reason| reason.contains(refusal_kind))
        );
        assert!(
            output.channels["subs"]
                .drivers
                .as_ref()
                .unwrap()
                .iter()
                .flat_map(|driver| &driver.plugins)
                .all(|plugin| plugin.plugin_type != "convolution")
        );
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}

fn assert_joint_drive_weight_discriminates(config: &RoomConfig) {
    let mut tradeoff = config.clone();
    let SpeakerConfig::MultiSub(group) = tradeoff.speakers.get_mut("subs").unwrap() else {
        panic!("tradeoff requires the joint-sub fixture");
    };
    // Put the peak on the fixed reference: filling from the flatter second
    // source reduces spectral error but costs more declared electrical drive.
    group.subwoofers.swap(0, 1);
    // Both positive weights retain exactly the same candidate family. These
    // deliberately separated engineering weights are not perceptual thresholds.
    let weights = [0.000001, 1000.0];
    let runs = weights.map(|weight| {
        let mut config = tradeoff.clone();
        config.optimizer.finalization.physical_drive_weight = weight;
        let result = RoomPipeline::new(RoomPipelineRequest {
            config: &config,
            sample_rate: 48_000.0,
            output_dir: None,
            probe_arrival_overrides: None,
        })
        .run(None)
        .unwrap();
        finalize_bound(&result);
        let output: autoeq::roomeq::DspChainOutput =
            serde_json::from_slice(&serde_json::to_vec(&result.to_dsp_chain_output()).unwrap())
                .unwrap();
        let chain = &output.channels["subs"];
        let declarations = &config
            .optimizer
            .finalization
            .physical_drive
            .as_ref()
            .unwrap()
            .outputs;
        let mut utilization = 0.0_f64;
        for driver in chain.drivers.as_ref().unwrap() {
            // Independent steady-sine oracle from delivered plugin parameters,
            // with no production DSP response or physical-assessment helper.
            let mut gain_db = 0.0;
            for plugin in output
                .global_plugins
                .iter()
                .chain(&chain.plugins)
                .chain(&driver.plugins)
            {
                match plugin.plugin_type.as_str() {
                    "gain" => gain_db += plugin.parameters["gain_db"].as_f64().unwrap(),
                    "delay" => {}
                    other => panic!("unsupported oracle plugin {other}"),
                }
            }
            let peak =
                config.optimizer.finalization.default_input_peak * 10.0_f64.powf(gain_db / 20.0);
            let id = serde_json::json!(["driver", "subs", driver.index, driver.name]).to_string();
            for declaration in &declarations[&id] {
                assert!(peak <= declaration.linear_valid_output_peak);
                for (reference, limit) in declaration
                    .demand_at_reference
                    .iter()
                    .zip(&declaration.limits)
                {
                    let demand = reference * peak / declaration.reference_output_peak;
                    assert!(demand <= *limit);
                    utilization = utilization.max(demand / limit);
                }
            }
        }
        let check = &result
            .metadata
            .stage_outcomes
            .iter()
            .find(|stage| stage.stage == "final_candidate_objective")
            .unwrap()
            .checks[0];
        let score: serde_json::Value =
            serde_json::from_str(check.diagnostic.as_ref().unwrap()).unwrap();
        let error = score["mean_seat_target_error_db"].as_f64().unwrap();
        assert!(
            (score["worst_declared_drive_utilization"].as_f64().unwrap() - utilization).abs()
                < 1e-12
        );
        assert!((check.observed.unwrap() - (error + weight * utilization.powi(2))).abs() < 1e-10);
        let trials: std::collections::BTreeMap<_, _> = result
            .metadata
            .stage_outcomes
            .iter()
            .flat_map(|stage| &stage.checks)
            .filter(|trial| {
                trial.id.starts_with("joint_drive_gain_")
                    || trial.id.starts_with("correction_strength_")
            })
            .map(|trial| {
                (
                    trial.id.clone(),
                    (trial.passed, trial.observed, trial.diagnostic.clone()),
                )
            })
            .collect();
        assert!(
            trials
                .iter()
                .any(|(id, (passed, _, _))| id.starts_with("joint_drive_gain_") && *passed)
        );
        trials
    });
    let [low_trials, high_trials] = runs;
    assert_eq!(
        low_trials.keys().collect::<Vec<_>>(),
        high_trials.keys().collect::<Vec<_>>(),
        "weights must not change candidate identities"
    );
    for (id, (passed, _, diagnostic)) in &low_trials {
        let other = &high_trials[id];
        assert_eq!((*passed, diagnostic), (other.0, &other.2), "{id}");
    }
    let flips = low_trials
        .iter()
        .any(|(gain_id, (gain_passed, gain_low, _))| {
            if !gain_id.starts_with("joint_drive_gain_") || !gain_passed {
                return false;
            }
            let (Some(gain_low), Some(gain_high)) = (*gain_low, high_trials[gain_id].1) else {
                return false;
            };
            low_trials
                .iter()
                .any(|(other_id, (other_passed, other_low, _))| {
                    if gain_id == other_id || !other_passed {
                        return false;
                    }
                    let (Some(other_low), Some(other_high)) = (*other_low, high_trials[other_id].1)
                    else {
                        return false;
                    };
                    (gain_low - other_low) * (gain_high - other_high) < -1e-12
                })
        });
    assert!(
        flips,
        "positive physical-drive weights must reverse at least one valid gain-candidate ranking"
    );
}

#[test]
fn roadmap_correction_admission_joint_sub_preserves_seat_diagnostics() {
    let group = declared_joint_sub_fixture();
    let mut config = multisub_room_config();
    config.optimizer.min_freq = 30.0;
    config.optimizer.max_freq = 120.0;
    config
        .speakers
        .insert("subs".into(), SpeakerConfig::MultiSub(group));
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .expect("declared two-sub, three-seat joint workflow");
    let output = serde_json::to_value(result.to_dsp_chain_output()).unwrap();
    let report = &output["channels"]["subs"]["joint_sub"];
    assert!(
        report.is_object(),
        "joint diagnostics must survive the public pipeline: {report}"
    );
    let seats = report["seats"].as_array().expect("per-seat diagnostics");
    assert_eq!(seats.len(), 3);
    assert_eq!(report["array_gains_db"].as_array().unwrap().len(), 2);
    assert_eq!(report["array_delays_ms"].as_array().unwrap().len(), 2);
    assert_eq!(
        report["physical_outputs"],
        serde_json::json!(["subs_1", "subs_2"])
    );
    for seat in seats {
        for stage in ["before", "after_array", "after_shared_eq"] {
            assert!(!seat[stage]["spl"].as_array().unwrap().is_empty());
        }
        assert!(
            seat["reference_scope"]
                .as_str()
                .unwrap()
                .contains("fixture-clock")
        );
    }
    let actual_ids: Vec<_> = output["channels"]["subs"]["drivers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|driver| driver["name"].clone())
        .collect();
    assert_eq!(report["physical_outputs"], serde_json::json!(actual_ids));
    // One common EQ cannot change pairwise per-seat level differences.
    for (index, value) in seats[0]["after_array"]["spl"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let before =
            value.as_f64().unwrap() - seats[1]["after_array"]["spl"][index].as_f64().unwrap();
        let after = seats[0]["after_shared_eq"]["spl"][index].as_f64().unwrap()
            - seats[1]["after_shared_eq"]["spl"][index].as_f64().unwrap();
        assert!((before - after).abs() < 1e-8);
    }
    // A valid finalized graph retains the complete diagnostic history in packaging.
    let original: autoeq::roomeq::DspChainOutput = serde_json::from_value(output.clone()).unwrap();
    let (packaged, _) = roomeq_export::package_convolution_sidecars(
        &original,
        &[],
        &Default::default(),
        &Default::default(),
    )
    .expect("unchanged finalized output must package successfully");
    assert_eq!(
        serde_json::to_value(packaged.channels["subs"].joint_sub.as_ref().unwrap()).unwrap(),
        *report
    );
    // A forged current-processing flag cannot authorize rebinding stale decisions.
    let mut changed = original;
    let channel = changed.channels.get_mut("subs").unwrap();
    channel.drivers.as_mut().unwrap()[1]
        .plugins
        .push(autoeq::roomeq_engine::output::create_gain_plugin(1.0));
    channel
        .joint_sub
        .as_mut()
        .unwrap()
        .channel_processing_matches = true;
    let error = roomeq_export::package_convolution_sidecars(
        &changed,
        &[],
        &Default::default(),
        &Default::default(),
    )
    .expect_err("packaging must refuse a mutation made after finalization");
    assert!(error.to_string().contains("stale source"), "{error:#}");
}

#[test]
fn roadmap_correction_admission_joint_sub_reverts_damaged_seat() {
    let mut group = declared_joint_sub_fixture();
    group.subwoofers = (0..2).map(|_| {
        let measurements: Vec<_> = [80.0, 60.0, 60.0].into_iter().enumerate().map(|(seat, level)| {
            let curve = phased_curve(level, 0.0);
            serde_json::json!({
                "name": format!("seat-{seat}"), "frequencies": curve.freq.to_vec(),
                "magnitude_db": curve.spl.to_vec(), "phase_deg": curve.phase.unwrap().to_vec()
            })
        }).collect();
        serde_json::from_value(serde_json::json!({
            "measurements": measurements,
            "speaker_name": "identical-sub-model",
            "provenance": {"capture_kind": "stationary_ir", "timing_reference_id": "fixture-clock"}
        })).unwrap()
    }).collect();
    let mut config = multisub_room_config();
    config.optimizer.min_freq = 30.0;
    config.optimizer.max_freq = 120.0;
    config
        .speakers
        .insert("subs".into(), SpeakerConfig::MultiSub(group));
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .unwrap();
    let output = serde_json::to_value(result.to_dsp_chain_output()).unwrap();
    let report = &output["channels"]["subs"]["joint_sub"];
    assert!(
        report["array_rejection_reason"]
            .as_str()
            .unwrap()
            .contains("seat_target_weighted_rms_regressed")
    );
    assert_eq!(report["array_gains_db"], serde_json::json!([0.0, 0.0]));
    assert_eq!(report["array_delays_ms"], serde_json::json!([0.0, 0.0]));
    assert_eq!(report["before_objective"], report["after_array_objective"]);
    for seat in report["seats"].as_array().unwrap() {
        assert_eq!(seat["before"], seat["after_array"]);
    }
    let decoded: autoeq::roomeq::DspChainOutput = serde_json::from_value(output).unwrap();
    assert!(
        decoded.channels["subs"]
            .joint_sub
            .as_ref()
            .unwrap()
            .array_rejection_reason
            .is_some()
    );
    assert_joint_rejection_ledger(&decoded, "joint_array_proposal_reverted");
}

#[test]
fn roadmap_correction_admission_joint_shared_eq_protects_flat_seat() {
    let mut group = declared_joint_sub_fixture();
    group.subwoofers = (0..2)
        .map(|_| {
            let measurements: Vec<_> = (0..3).map(|seat| {
            let mut curve = phased_curve(80.0, 0.0);
            // Only the other two seats have this peak. Fitting the spatial
            // average with a shared cut must not damage the flat seat.
            for (frequency, level) in curve.freq.iter().zip(curve.spl.iter_mut()) {
                *level = 80.0 + if seat == 0 { 0.0 } else {
                    12.0 * (-((frequency / 65.0).ln() / 0.22).powi(2)).exp()
                };
            }
            serde_json::json!({
                "name": format!("seat-{seat}"), "frequencies": curve.freq.to_vec(),
                "magnitude_db": curve.spl.to_vec(), "phase_deg": curve.phase.unwrap().to_vec()
            })
        }).collect();
            serde_json::from_value(serde_json::json!({
            "measurements": measurements, "speaker_name": "identical-sub-model",
            "provenance": {"capture_kind": "stationary_ir", "timing_reference_id": "fixture-clock"}
        })).unwrap()
        })
        .collect();
    let mut config = multisub_room_config();
    config.optimizer.min_freq = 30.0;
    config.optimizer.max_freq = 120.0;
    config
        .speakers
        .insert("subs".into(), SpeakerConfig::MultiSub(group));
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .unwrap();
    let output = serde_json::to_value(result.to_dsp_chain_output()).unwrap();
    let report = &output["channels"]["subs"]["joint_sub"];
    assert!(
        report["shared_eq_rejection_reason"]
            .as_str()
            .unwrap_or("")
            .contains("seat_target_weighted_rms_regressed"),
        "expected numerical rejection, got {}",
        report["shared_eq_rejection_reason"]
    );
    for seat in report["seats"].as_array().unwrap() {
        assert_eq!(seat["after_array"]["spl"], seat["after_shared_eq"]["spl"]);
    }
    let decoded: autoeq::roomeq::DspChainOutput = serde_json::from_value(output).unwrap();
    let channel = &decoded.channels["subs"];
    assert!(
        channel
            .plugins
            .iter()
            .all(|plugin| plugin.plugin_type != "eq")
    );
    assert!(
        channel
            .joint_sub
            .as_ref()
            .unwrap()
            .shared_eq_rejection_reason
            .is_some()
    );
    assert_joint_rejection_ledger(&decoded, "joint_shared_eq_proposal_reverted");
}

fn assert_joint_rejection_ledger(output: &autoeq::roomeq::DspChainOutput, reason: &str) {
    use roomeq_model::decision_ledger::{DecisionStage, DecisionStatus};
    let ledger = output.correction_decisions.as_ref().unwrap();
    let records: Vec<_> = ledger
        .decisions
        .iter()
        .filter(|record| record.reason_codes.iter().any(|code| code == reason))
        .collect();
    assert_eq!(
        records.len(),
        2,
        "expected rejection history for both physical subs"
    );
    for record in records {
        assert_eq!(record.status, DecisionStatus::Reverted);
        assert_eq!(record.stage, DecisionStage::Provisional);
        assert!(record.final_graph_identity.is_none());
        assert_eq!(record.seat_refs.len(), 3);
        assert!(record.frequency_band_hz.is_some());
        assert!(
            record
                .reason_codes
                .iter()
                .any(|code| code.contains("seat_target_weighted_rms_regressed"))
        );
    }
    let mut rebound = output.clone();
    final_ledger::finalize_output_ledger(
        &mut rebound,
        &ledger.decisions,
        &final_ledger::ReconciliationEvents::default(),
    )
    .unwrap();
    let repeated = rebound.correction_decisions.as_ref().unwrap();
    assert_eq!(
        repeated
            .decisions
            .iter()
            .filter(|record| record.reason_codes.iter().any(|code| code == reason))
            .count(),
        2,
        "repeated finalization must not duplicate stage history",
    );
    let mut forged = ledger.decisions.clone();
    let record = forged
        .iter_mut()
        .find(|record| record.reason_codes.iter().any(|code| code == reason))
        .unwrap();
    record.status = DecisionStatus::Applied;
    let error = final_ledger::finalize_output_ledger(
        &mut rebound,
        &forged,
        &final_ledger::ReconciliationEvents::default(),
    )
    .unwrap_err();
    assert!(error.contains("conflicting joint-stage history"), "{error}");
    assert_python_report(
        output,
        r#"
import json, sys, tempfile
from pathlib import Path
from scripts.src.payload_binding import verify_payload_binding
from scripts.src.correction_explanation import correction_explanation_html
from scripts.src.report import create_html_report, create_comparison_html_report
from scripts.src.loaders import load_roomeq_json
data = json.load(sys.stdin)
assert verify_payload_binding(data)[0]
section = correction_explanation_html(data)
assert 'proposal reverted' in section
assert 'stage history, not approval of later routing or recorded playback' in section
assert 'seat_target_weighted_rms_regressed' in section
assert 'fixture-clock:seat-index-0' in section
assert 'subs_1' in section and 'subs_2' in section
assert 'stage_correction_band_min' in section
with tempfile.TemporaryDirectory(dir='/Volumes/home_tmp/tmp') as directory:
    root = Path(directory)
    path = root / 'joint.json'
    path.write_text(json.dumps(data))
    loaded = load_roomeq_json(path)
    create_html_report(loaded, root / 'report.html', path)
    rendered = (root / 'report.html').read_text()
    verdict = rendered.index('<section class="playback-status"')
    explanation = rendered.index('<section class="correction-explanation"')
    views = rendered.index('<section class="acceptance-views"')
    assert verdict < explanation < views
    assert 'proposal reverted' in rendered[explanation:views]
    create_comparison_html_report([('joint', loaded), ('same', loaded)], root / 'comparison.html')
    compared = (root / 'comparison.html').read_text()
    assert compared.count('<section class="correction-explanation"') == 2
    assert compared.count('stage history, not approval of later routing or recorded playback') >= 2
"#,
    );
}

#[test]
fn roadmap_correction_admission_multisub_detailed_and_joint_refusal() {
    use autoeq::roomeq_engine::eq::EqResources;
    use autoeq::roomeq_engine::group_processing::{PreparedMultiSubGroup, process_multisub_group};
    let room_config = multisub_room_config();
    let resources = EqResources::default();
    for joint in [false, true] {
        // A phased seat matrix with unknown timing scope must not authorize
        // selected joint processing through a different optimizer's fallback.
        // Explicit legacy detailed selection remains a separate contract.
        let prepared = PreparedMultiSubGroup {
            subwoofers: vec![phased_curve(80.0, 0.0), phased_curve(78.0, 0.0)],
            seat_measurements: Some(vec![
                vec![phased_curve(80.0, 0.0)],
                vec![phased_curve(78.0, 0.001)],
            ]),
            reference_scope: None,
        };
        let result = process_multisub_group(
            "subs",
            &multisub_group(joint),
            &room_config,
            48_000.0,
            &prepared,
            &resources,
            &resources,
        );
        if joint {
            let error = result.expect_err("unknown timing must refuse selected joint mode");
            assert!(error.to_string().contains("timing reference"), "{error}");
            continue;
        }
        let (chain, ..) = result.expect("explicit detailed mode");
        assert!(
            !chain.plugins.is_empty(),
            "multisub joint={joint} must emit a chain"
        );
    }
}

#[test]
fn roadmap_correction_admission_routed_joint_sub_retains_stage_history() {
    check_routed_joint_sub_refinement(false);
}

#[test]
fn roadmap_correction_admission_routed_joint_sub_named_outputs_reach_refinement() {
    check_routed_joint_sub_refinement(true);
}

fn check_routed_joint_sub_refinement(named_outputs: bool) {
    let mut config = home_cinema_config();
    // Exercise joint array control and routed transport, not a second route
    // optimization benchmark. Structural bass routing still changes the chain.
    config
        .system
        .as_mut()
        .unwrap()
        .bass_management
        .get_or_insert_with(Default::default)
        .optimize_groups = false;
    let group = declared_joint_sub_fixture();
    for key in ["l", "r"] {
        config.speakers.insert(
            key.into(),
            SpeakerConfig::Single(group.subwoofers[0].clone()),
        );
    }
    config
        .speakers
        .insert("sub".into(), SpeakerConfig::MultiSub(group));
    config
        .system
        .as_mut()
        .unwrap()
        .subwoofers
        .as_mut()
        .unwrap()
        .config = SubwooferStrategy::Mso;
    if named_outputs {
        let subs = config.system.as_mut().unwrap().subwoofers.as_mut().unwrap();
        subs.outputs.push(SubwooferOutput {
            id: "RearBass".into(),
            speaker: "sub".into(),
        });
        subs.crossover = Some(SubwooferCrossoverRef::PerSub(vec![
            "sub_xover".into(),
            "sub_xover".into(),
        ]));
    }
    let result = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .expect("routed joint-sub pipeline");
    let output = result.to_dsp_chain_output();
    let report = output.channels["Sub1"]
        .joint_sub
        .as_ref()
        .expect("routed preprocessing must retain joint stage report");
    assert_eq!(report.seats.len(), 3);
    assert!(
        !report.channel_processing_matches,
        "routing adds processing after the joint stage"
    );
    assert!(report.scope.contains("excludes later trims"));
    if named_outputs {
        assert_eq!(
            output.channels["Sub1"].drivers.as_ref().unwrap()[1].name,
            "RearBass"
        );
        assert_ne!(report.physical_outputs[1], "RearBass");
    }
    config.optimizer.finalization.default_input_peak = 0.1;
    let ports = roomeq_workflow::electrical_headroom::assess_final_graph(
        &output,
        48_000.0,
        std::path::Path::new("."),
        &config.optimizer.finalization,
    )
    .unwrap();
    let outputs: serde_json::Map<String, serde_json::Value> = ports
        .iter()
        .map(|port| {
            (
                port.output.clone(),
                serde_json::json!([{
                    "quantity":"voltage_rms", "calibration_id":"synthetic-routed-drive",
                    "reference_conditions_id":"load", "limit_conditions_id":"load",
                    "sine_duration_seconds":1.0, "reference_output_peak":0.1,
                    "linear_valid_output_peak":1.0, "frequencies_hz":[40.0,80.0],
                    "demand_at_reference":[1.0,1.0], "limits":[100.0,100.0]
                }]),
            )
        })
        .collect();
    config.optimizer.finalization.physical_drive =
        Some(serde_json::from_value(serde_json::json!({"outputs": outputs})).unwrap());
    config.optimizer.finalization.physical_drive_weight = 1.0;
    let ranked = RoomPipeline::new(RoomPipelineRequest {
        config: &config,
        sample_rate: 48_000.0,
        output_dir: None,
        probe_arrival_overrides: None,
    })
    .run(None)
    .expect("routed calibrated joint-sub pipeline");
    assert_eq!(
        ranked.channels["Sub1"]
            .joint_sub
            .as_ref()
            .unwrap()
            .physical_outputs,
        report.physical_outputs,
        "refinement must preserve historical stage identities"
    );
    let trials: Vec<_> = ranked
        .metadata
        .stage_outcomes
        .iter()
        .flat_map(|stage| &stage.checks)
        .filter(|check| check.id.starts_with("joint_drive_gain_"))
        .collect();
    assert!(
        !trials.is_empty(),
        "routed joint groups must reach calibrated refinement"
    );
    for trial in trials {
        let diagnostic = trial.diagnostic.as_deref().unwrap_or("");
        assert!(!diagnostic.contains("ownership"), "{trial:?}");
        assert!(
            !diagnostic.contains("inconsistent physical identities"),
            "{trial:?}"
        );
    }
    finalize_bound(&ranked);
    if named_outputs {
        if let Ok(path) = std::env::var("ROOMEQ_ROUTED_JOINT_ARTIFACT") {
            roomeq_workflow::output::save_dsp_chain(
                &ranked.to_dsp_chain_output(),
                std::path::Path::new(&path),
            )
            .expect("write routed joint-sub QA artifact");
        }
    }
}
