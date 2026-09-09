use super::apply::clamp_strict_measured_maxeval;
use super::consts::{QA_MAXEVAL, qa_seed};
use super::metric_scorecard::MetricScorecard;
use super::metric_scorecard::compare_scorecards;
use super::misc::level_matched_rms_curve_difference_db;
use super::option::isolate_schroeder_split_from_multi_measurement;
use super::option_override::OptionOverride;
use super::parse_maxeval;
use super::parse_seed_runs;
use super::run::deployed_final_curve;
use super::types::TestResult;
use super::validate::{
    TargetTiltValidationOptions, validate_option_effect, validate_phase_alignment,
    validate_target_tilt,
};
use roomeq_model::{
    ChannelDspChain, Curve, MultiMeasurementConfig, OptimizationMetadata, RoomConfig, StageOutcome,
    StageStatus,
};
use std::collections::HashMap;

use roomeq_engine::room_result::{ChannelOptimizationResult, RoomOptimizationResult};

#[test]
fn schroeder_split_option_clears_inherited_multi_measurement_mode() {
    let mut config = RoomConfig::default();
    config.optimizer.multi_measurement = Some(MultiMeasurementConfig::default());
    let options = [OptionOverride::SchroederSplit {
        schroeder_freq: 300.0,
        low_max_q: 10.0,
        high_max_q: 1.0,
    }];

    isolate_schroeder_split_from_multi_measurement(&mut config, &options);

    assert!(config.optimizer.multi_measurement.is_none());
}

#[test]
fn unrelated_option_preserves_inherited_multi_measurement_mode() {
    let mut config = RoomConfig::default();
    config.optimizer.multi_measurement = Some(MultiMeasurementConfig::default());

    isolate_schroeder_split_from_multi_measurement(&mut config, &[OptionOverride::AsymmetricLoss]);

    assert!(config.optimizer.multi_measurement.is_some());
}

#[test]
fn level_matched_cross_mode_rms_ignores_offset_but_detects_shape() {
    let freq = ndarray::arr1(&[100.0, 200.0, 400.0, 800.0]);
    let reference = Curve {
        freq: freq.clone(),
        spl: ndarray::arr1(&[0.0, 1.0, 2.0, 3.0]),
        phase: None,
        ..Default::default()
    };
    let level_shifted = Curve {
        freq: freq.clone(),
        spl: ndarray::arr1(&[6.0, 7.0, 8.0, 9.0]),
        phase: None,
        ..Default::default()
    };
    let reshaped = Curve {
        freq,
        spl: ndarray::arr1(&[0.0, 2.0, 4.0, 6.0]),
        phase: None,
        ..Default::default()
    };

    let shifted_rms =
        level_matched_rms_curve_difference_db(&reference, &level_shifted, 100.0, 800.0).unwrap();
    let shape_rms =
        level_matched_rms_curve_difference_db(&reference, &reshaped, 100.0, 800.0).unwrap();
    assert!(shifted_rms < 1.0e-12);
    assert!(shape_rms > 1.0);
}

#[test]
fn deployed_curve_uses_routed_result_curve_before_raw_channel_curve() {
    let mut result = result_with_channel_slopes(0.0, 0.0, 0.0);
    let raw_channel_curve = curve_with_slope(0.0);
    let deployed_routed_curve = curve_with_slope(12.0);
    result.channels.get_mut("L").unwrap().final_curve = Some((&raw_channel_curve).into());
    result.channel_results.get_mut("L").unwrap().final_curve = raw_channel_curve;
    result
        .deployed_source_curves
        .insert("L".to_string(), deployed_routed_curve.clone());

    let actual = deployed_final_curve(&result, "L").unwrap();

    assert_eq!(actual.spl, deployed_routed_curve.spl);
}

fn curve_with_slope(slope_db_per_octave: f64) -> Curve {
    let freq = ndarray::arr1(&[100.0, 200.0, 400.0, 500.0]);
    let spl = freq.mapv(|f: f64| slope_db_per_octave * (f / 100.0).log2());
    Curve {
        freq,
        spl,
        phase: None,
        ..Default::default()
    }
}

fn channel_chain_with_slopes(
    initial_slope_db_per_octave: f64,
    final_slope_db_per_octave: f64,
    target_slope_db_per_octave: f64,
) -> ChannelDspChain {
    ChannelDspChain {
        channel: "L".to_string(),
        plugins: Vec::new(),
        drivers: None,
        initial_curve: Some((&curve_with_slope(initial_slope_db_per_octave)).into()),
        final_curve: Some((&curve_with_slope(final_slope_db_per_octave)).into()),
        eq_response: None,
        target_curve: Some((&curve_with_slope(target_slope_db_per_octave)).into()),
        pre_ir: None,
        post_ir: None,
        fir_temporal_masking: None,
        direct_early_late_correction: None,
    }
}

fn result_with_channel_slopes(
    initial_slope_db_per_octave: f64,
    final_slope_db_per_octave: f64,
    target_slope_db_per_octave: f64,
) -> RoomOptimizationResult {
    let initial_curve = curve_with_slope(initial_slope_db_per_octave);
    let final_curve = curve_with_slope(final_slope_db_per_octave);
    let channel = ChannelOptimizationResult {
        name: "L".to_string(),
        pre_score: 0.0,
        post_score: 0.0,
        initial_curve,
        final_curve,
        biquads: Vec::new(),
        fir_coeffs: None,
        optimizer_evidence: Vec::new(),
    };
    RoomOptimizationResult {
        channels: HashMap::from([(
            "L".to_string(),
            channel_chain_with_slopes(
                initial_slope_db_per_octave,
                final_slope_db_per_octave,
                target_slope_db_per_octave,
            ),
        )]),
        channel_results: HashMap::from([("L".to_string(), channel)]),
        deployed_source_curves: HashMap::new(),
        combined_pre_score: 0.0,
        combined_post_score: 0.0,
        metadata: OptimizationMetadata {
            pre_score: 0.0,
            post_score: 0.0,
            algorithm: "test".to_string(),
            loss_type: None,
            iterations: 0,
            timestamp: "test".to_string(),
            inter_channel_deviation: None,
            epa_per_channel: None,
            epa_multichannel: None,
            group_delay: None,
            mixed_phase_per_channel: None,
            perceptual_metrics: None,
            home_cinema_layout: None,
            multi_seat_coverage: None,
            multi_seat_correction: None,
            bass_management: None,
            timing_diagnostics: None,
            ctc: None,
            perceptual_policy: None,
            bootstrap_uncertainty: None,
            validation_bundle: None,
            final_convolution_sha256: None,
            supporting_source: None,
            correction_acceptance: None,
            optimizer_evidence: None,
            stage_outcomes: Vec::new(),
            qa_seed_distribution: None,
            effective_config: None,
        },
    }
}

fn empty_room_config() -> RoomConfig {
    RoomConfig {
        version: "test".to_string(),
        system: None,
        speakers: HashMap::new(),
        crossovers: None,
        target_curve: None,
        optimizer: Default::default(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    }
}

fn result_with_inter_channel_slope(channel_slope_db_per_octave: f64) -> RoomOptimizationResult {
    let mut result = result_with_channel_slopes(0.0, 0.0, 0.0);
    let reference_curve = curve_with_slope(0.0);
    let channel_curve = curve_with_slope(channel_slope_db_per_octave);
    result.channel_results = HashMap::from([
        (
            "C".to_string(),
            ChannelOptimizationResult {
                name: "C".to_string(),
                pre_score: 0.0,
                post_score: 0.0,
                initial_curve: reference_curve.clone(),
                final_curve: reference_curve,
                biquads: Vec::new(),
                fir_coeffs: None,
                optimizer_evidence: Vec::new(),
            },
        ),
        (
            "L".to_string(),
            ChannelOptimizationResult {
                name: "L".to_string(),
                pre_score: 0.0,
                post_score: 0.0,
                initial_curve: channel_curve.clone(),
                final_curve: channel_curve,
                biquads: Vec::new(),
                fir_coeffs: None,
                optimizer_evidence: Vec::new(),
            },
        ),
    ]);
    result.combined_post_score = 1.0;
    result
}

#[test]
fn electrical_scorecard_gates_cascade_and_retains_sidecar_evidence() {
    for sample_rate in [44100.0, 48000.0, 96000.0] {
        let directory = tempfile::tempdir().unwrap();
        let mut result = result_with_channel_slopes(0.0, 0.0, 0.0);
        result.combined_pre_score = 10.0;
        result.combined_post_score = 5.0;
        let filter = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            80.0,
            sample_rate,
            2.0,
            3.0,
        );
        result.channel_results.get_mut("L").unwrap().biquads = vec![filter.clone(); 6];
        result.channels.get_mut("L").unwrap().plugins = vec![
            roomeq_engine::output::create_eq_plugin(&vec![filter; 6]),
            roomeq_engine::output::create_convolution_plugin("gain.wav"),
        ];
        let mut writer = hound::WavWriter::create(
            directory.path().join("gain.wav"),
            hound::WavSpec {
                channels: 1,
                sample_rate: sample_rate as u32,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        writer.write_sample(2.0_f32).unwrap();
        writer.finalize().unwrap();
        let assessment = super::electrical::assess(&result, sample_rate, directory.path());
        let scorecard = super::metric_scorecard::compute_result_scorecard(&result, assessment);
        assert!((scorecard.max_section_gain_db - 3.0).abs() < 1e-9);
        assert!(
            (scorecard.max_boost_db - 24.020599913).abs() < 1e-5,
            "{scorecard:?}"
        );
        let missing_directory = directory.path().to_path_buf();
        let retained = tempfile::tempdir().unwrap();
        let bundle = retained.path().join("replay");
        super::electrical::retain_replay_bundle(&result, sample_rate, directory.path(), &bundle)
            .unwrap();
        assert!(
            super::electrical::retain_replay_bundle(
                &result,
                sample_rate,
                directory.path(),
                &bundle
            )
            .is_err()
        );
        directory.close().unwrap();
        let graph: roomeq_model::DspGraph =
            serde_json::from_slice(&std::fs::read(bundle.join("graph.json")).unwrap()).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(bundle.join("assessment.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["sample_rate_hz"], sample_rate);
        assert!(
            manifest["frequencies_hz"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!(80.0))
        );
        let mut packaged_result = result.clone();
        packaged_result.channels = graph.channels;
        packaged_result.metadata = graph.metadata.unwrap();
        let packaged_peaks =
            super::electrical::assess(&packaged_result, sample_rate, &bundle).unwrap();
        assert_eq!(
            &packaged_peaks,
            scorecard.electrical.as_ref().unwrap().as_ref().unwrap()
        );
        // Retained evidence remains usable after the same cleanup boundary as
        // the runner; a fresh replay must fail, not substitute a PEQ statistic.
        let peaks = scorecard.electrical.as_ref().unwrap().as_ref().unwrap();
        assert_eq!(peaks[0].sample_rate_hz, sample_rate);
        assert_eq!(peaks[0].input_peak_limits["L"], 1.0);
        assert_eq!(
            peaks[0].output,
            serde_json::json!(["channel", "L"]).to_string()
        );
        let missing = super::metric_scorecard::compute_result_scorecard(
            &result,
            super::electrical::assess(&result, sample_rate, &missing_directory),
        );
        assert!(missing.max_boost_db.is_infinite());
        assert!(missing.electrical.as_ref().unwrap().is_err());
        for reverted in [false, true] {
            let mut row = TestResult {
                label: "cascade +50% max_db".into(),
                pre_score: 10.0,
                scorecard: scorecard.clone(),
                pass: true,
                reason: "shape passed".into(),
            };
            row.scorecard.correction_reverted = reverted;
            super::enforce_registry_expectations(
                "electrical/cascade",
                &[],
                crate::registry::ScenarioExpect {
                    improvement_min_pct: 0.0,
                    max_post_score: 20.0,
                    max_boost_db: 12.0,
                    allow_safe_revert: true,
                    gate_purpose: crate::registry::QaGatePurpose::Safety,
                },
                std::slice::from_mut(&mut row),
            );
            assert!(
                !row.pass,
                "electrical clipping cannot pass via rollback or wider section bounds"
            );
            assert!(row.reason.contains("sampled electrical max boost"));
        }
    }
}

#[test]
fn electrical_qa_expands_canonical_global_bass_routes_once() {
    let mut result = result_with_channel_slopes(0.0, 0.0, 0.0);
    result.channels.clear();
    for (name, gain, stage) in [
        ("L", 6.0, "pre_route"),
        ("R", 6.0, "pre_route"),
        ("sub", 3.0, "post_route"),
    ] {
        let mut chain = channel_chain_with_slopes(0.0, 0.0, 0.0);
        chain.channel = name.into();
        let mut plugin = roomeq_engine::output::create_gain_plugin(gain);
        plugin.parameters["room_eq_stage"] = serde_json::json!(stage);
        chain.plugins = vec![plugin];
        result.channels.insert(name.into(), chain);
    }
    let routing = roomeq_model::BassManagementRoutingGraph {
        physical_sub_output: "sub".into(),
        input_channels: vec!["L".into(), "R".into()],
        output_channels: vec!["sub".into()],
        routes: ["L", "R"]
            .iter()
            .enumerate()
            .map(|(index, name)| roomeq_model::BassManagementRoute {
                group_id: None,
                source_channel: (*name).into(),
                source_index: index,
                destination: "sub".into(),
                destination_index: 0,
                pre_chain_channel: Some((*name).into()),
                post_chain_channel: Some("sub".into()),
                route_kind: "low".into(),
                crossover_type: "LR24".into(),
                high_pass_hz: None,
                low_pass_hz: None,
                gain_db: -6.0,
                gain_linear: 10.0_f64.powf(-6.0 / 20.0),
                matrix_gain: 0.5,
                delay_ms: 0.0,
                polarity_inverted: index == 1,
            })
            .collect(),
        matrix: Some(roomeq_model::BassManagementMatrix {
            input_channel_map: vec![0, 1],
            output_channel_map: vec![0],
            matrix: vec![0.5, -0.5],
            route_count: 2,
        }),
        input_trim_db: HashMap::new(),
        advisories: Vec::new(),
    };
    result.metadata.bass_management = Some(roomeq_model::BassManagementReport {
        enabled: true,
        crossover_type: "LR24".into(),
        crossover_frequency_hz: None,
        redirected_bass_enabled: true,
        lfe_channel: "LFE".into(),
        lfe_playback_gain_db: 0.0,
        lfe_low_pass_hz: 120.0,
        lfe_gain_applied_to_chain: false,
        sub_trim_db: 0.0,
        max_sub_boost_db: 12.0,
        headroom_margin_db: 0.0,
        applied_sub_gain_db: None,
        gain_limited: false,
        physical_sub_output: "sub".into(),
        redirected_bass_channel_count: 2,
        main_high_pass_hz: None,
        sub_low_pass_hz: None,
        lfe_headroom_required_db: 0.0,
        signal_flow: Vec::new(),
        signal_flow_advisories: Vec::new(),
        routing_graph: Some(routing),
        optimization: None,
        groups: Vec::new(),
        sub_outputs: Vec::new(),
        headroom_simulation: None,
        advisory: String::new(),
    });
    let peaks = super::electrical::assess(&result, 48000.0, std::path::Path::new(".")).unwrap();
    assert_eq!(peaks.len(), 1);
    assert_eq!(peaks[0].output, "sub");
    assert_eq!(peaks[0].inputs, vec!["L", "R"]);
    // +6 pre -6 route +3 post, plus 6.0206 dB concurrent-input sum.
    assert!((peaks[0].required_attenuation_db - 9.020599913).abs() < 1e-6);
    let mut graph = result.to_dsp_chain_output();
    assert_eq!(graph.global_plugins.len(), 1);
    graph.global_plugins[0].parameters["matrix"][0] = serde_json::json!(99.0);
    assert!(roomeq_workflow::electrical_headroom::canonical_electrical_routing(&graph).is_err());
    graph = result.to_dsp_chain_output();
    graph
        .global_plugins
        .push(roomeq_engine::output::create_gain_plugin(20.0));
    assert!(roomeq_workflow::electrical_headroom::canonical_electrical_routing(&graph).is_err());
    graph = result.to_dsp_chain_output();
    graph.global_plugins.clear();
    assert!(roomeq_workflow::electrical_headroom::canonical_electrical_routing(&graph).is_err());
}

#[test]
fn electrical_qa_keeps_same_named_driver_ports_separate() {
    let mut result = result_with_channel_slopes(0.0, 0.0, 0.0);
    result.channels.clear();
    for (name, invert) in [("L", false), ("R", true)] {
        let mut chain = channel_chain_with_slopes(0.0, 0.0, 0.0);
        chain.channel = name.into();
        chain.plugins = vec![roomeq_engine::output::create_gain_plugin(6.0)];
        chain.drivers = Some(vec![
            roomeq_model::DriverDspChain {
                name: "woofer".into(),
                index: 0,
                initial_curve: None,
                plugins: vec![roomeq_engine::output::create_gain_plugin_with_invert(
                    3.0, invert,
                )],
            },
            roomeq_model::DriverDspChain {
                name: "tweeter".into(),
                index: 1,
                initial_curve: None,
                plugins: vec![roomeq_engine::output::create_gain_plugin(-12.0)],
            },
        ]);
        result.channels.insert(name.into(), chain);
    }
    let peaks = super::electrical::assess(&result, 48000.0, std::path::Path::new(".")).unwrap();
    assert_eq!(peaks.len(), 4);
    for name in ["L", "R"] {
        let woofer = serde_json::json!(["driver", name, 0, "woofer"]).to_string();
        let tweeter = serde_json::json!(["driver", name, 1, "tweeter"]).to_string();
        let woofer = peaks.iter().find(|p| p.output == woofer).unwrap();
        let tweeter = peaks.iter().find(|p| p.output == tweeter).unwrap();
        assert_eq!(woofer.inputs, vec![name]);
        assert_eq!(tweeter.inputs, vec![name]);
        assert!((woofer.required_attenuation_db - 9.0).abs() < 1e-9);
        assert!((tweeter.peak_dbfs.unwrap() + 6.0).abs() < 1e-9);
        assert_eq!(tweeter.required_attenuation_db, 0.0);
    }
    let drivers = result
        .channels
        .get_mut("L")
        .unwrap()
        .drivers
        .as_mut()
        .unwrap();
    drivers[1].name = "woofer".into();
    assert!(super::electrical::assess(&result, 48000.0, std::path::Path::new(".")).is_err());
}

#[test]
fn electrical_artifact_retains_assessed_and_unassessed_rows() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let result = result_with_channel_slopes(0.0, 0.0, 0.0);
    let scorecard = super::metric_scorecard::compute_result_scorecard(
        &result,
        super::electrical::assess(&result, 48000.0, std::path::Path::new(".")),
    );
    let mut row = TestResult {
        label: "identity".into(),
        pre_score: 10.0,
        scorecard,
        pass: true,
        reason: "registry passed".into(),
    };
    super::electrical::append_evidence(file.path(), "quality/example", std::slice::from_ref(&row))
        .unwrap();
    row.scorecard.electrical = Some(Err("missing required FIR".into()));
    row.pass = false;
    super::electrical::append_evidence(file.path(), "quality/example", &[row]).unwrap();
    let records: Vec<serde_json::Value> = std::fs::read_to_string(file.path())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0]["registry_id"], "quality/example");
    assert_eq!(records[0]["status"], "assessed");
    assert_eq!(
        records[0]["assessment_kind"],
        "sampled_steady_state_sinusoidal"
    );
    assert_eq!(records[0]["outputs"][0]["sample_rate_hz"], 48000.0);
    assert_eq!(records[0]["outputs"][0]["input_peak_limits"]["L"], 1.0);
    assert_eq!(records[0]["transient_peak_certified"], false);
    assert_eq!(records[1]["status"], "unassessed");
    assert_eq!(records[1]["error"], "missing required FIR");
    assert_eq!(records[1]["registry_pass"], false);
    assert!(records[1]["outputs"].is_null());
    super::electrical::append_execution_failure(file.path(), "quality/failed", "seat regressed")
        .unwrap();
    let text = std::fs::read_to_string(file.path()).unwrap();
    let failure: serde_json::Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
    assert_eq!(failure["status"], "execution_failed");
    assert_eq!(failure["registry_id"], "quality/failed");
    assert_eq!(failure["registry_pass"], false);
}

#[test]
fn target_tilt_validator_accepts_response_that_does_not_regress_from_target() {
    let baseline = result_with_channel_slopes(0.0, 0.0, 0.0);
    let option = result_with_channel_slopes(1.0, 0.8, -0.8);
    let config = empty_room_config();

    let (pass, detail) = validate_target_tilt(
        -0.8,
        &baseline,
        &config,
        &option,
        TargetTiltValidationOptions {
            num_options: 1,
            has_schroeder: false,
            has_broadband: false,
            has_excursion: false,
        },
    );

    assert!(pass, "{detail}");
}

#[test]
fn target_tilt_validator_rejects_response_that_regresses_from_target() {
    let baseline = result_with_channel_slopes(0.0, 0.0, 0.0);
    let option = result_with_channel_slopes(0.0, 1.0, -0.8);
    let config = empty_room_config();

    let (pass, _) = validate_target_tilt(
        -0.8,
        &baseline,
        &config,
        &option,
        TargetTiltValidationOptions {
            num_options: 1,
            has_schroeder: false,
            has_broadband: false,
            has_excursion: false,
        },
    );

    assert!(!pass);
}

#[test]
fn target_tilt_validator_rejects_wrong_target_curve_slope() {
    let baseline = result_with_channel_slopes(0.0, 0.0, 0.0);
    let option = result_with_channel_slopes(0.0, 0.0, 0.0);
    let config = empty_room_config();

    let (pass, _) = validate_target_tilt(
        -0.8,
        &baseline,
        &config,
        &option,
        TargetTiltValidationOptions {
            num_options: 1,
            has_schroeder: false,
            has_broadband: false,
            has_excursion: false,
        },
    );

    assert!(!pass);
}

#[test]
fn target_tilt_validator_does_not_exempt_excursion_regression() {
    let baseline = result_with_channel_slopes(0.0, 0.0, 0.0);
    let option = result_with_channel_slopes(1.6, 5.3, -0.8);
    let config = empty_room_config();

    let (without_excursion, _) = validate_target_tilt(
        -0.8,
        &baseline,
        &config,
        &option,
        TargetTiltValidationOptions {
            num_options: 3,
            has_schroeder: false,
            has_broadband: false,
            has_excursion: false,
        },
    );
    let (with_excursion, detail) = validate_target_tilt(
        -0.8,
        &baseline,
        &config,
        &option,
        TargetTiltValidationOptions {
            num_options: 3,
            has_schroeder: false,
            has_broadband: false,
            has_excursion: true,
        },
    );

    assert!(!without_excursion);
    assert!(
        !with_excursion,
        "excursion protection bypassed tilt gate: {detail}"
    );
}

#[test]
fn timbre_matching_validator_requires_reduced_normalized_spread() {
    let baseline = result_with_inter_channel_slope(3.0);
    let option = result_with_inter_channel_slope(1.0);
    let config = empty_room_config();
    let override_option = OptionOverride::InterChannelTimbreMatching {
        reference_channel: "C".to_string(),
    };

    let (pass, detail) = validate_option_effect(
        &override_option,
        &config,
        &baseline,
        &config,
        &option,
        std::slice::from_ref(&override_option),
    );

    assert!(pass, "{detail}");
}

#[test]
fn timbre_matching_validator_rejects_increased_normalized_spread() {
    let baseline = result_with_inter_channel_slope(1.0);
    let option = result_with_inter_channel_slope(3.0);
    let config = empty_room_config();
    let override_option = OptionOverride::InterChannelTimbreMatching {
        reference_channel: "C".to_string(),
    };

    let (pass, _) = validate_option_effect(
        &override_option,
        &config,
        &baseline,
        &config,
        &option,
        std::slice::from_ref(&override_option),
    );

    assert!(!pass);
}

#[test]
fn timbre_matching_validator_allows_small_parallel_drift_for_applied_stage() {
    let baseline = result_with_inter_channel_slope(1.0);
    let mut option = result_with_inter_channel_slope(1.02);
    option.metadata.stage_outcomes.push(StageOutcome {
        checks: Vec::new(),
        stage: "inter_channel_timbre_matching".to_string(),
        status: StageStatus::Applied,
        advisories: Vec::new(),
    });
    let config = empty_room_config();
    let override_option = OptionOverride::InterChannelTimbreMatching {
        reference_channel: "C".to_string(),
    };

    let (pass, detail) = validate_option_effect(
        &override_option,
        &config,
        &baseline,
        &config,
        &option,
        std::slice::from_ref(&override_option),
    );

    assert!(pass, "{detail}");
}

#[test]
fn scorecard_allows_small_roughness_regression_when_baseline_already_violates_limit() {
    let baseline = MetricScorecard {
        flat_loss: 10.0,
        peak_residual_db: 1.0,
        max_boost_db: 0.0,
        max_section_gain_db: 0.0,
        replay_bundle: None,
        electrical: None,
        correction_reverted: false,
        epa_preference: None,
        epa_sharpness: None,
        epa_roughness: Some(0.95),
        group_delay_std_ms: None,
    };
    let candidate = MetricScorecard {
        flat_loss: 9.0,
        peak_residual_db: 1.0,
        max_boost_db: 0.0,
        max_section_gain_db: 0.0,
        replay_bundle: None,
        electrical: None,
        correction_reverted: false,
        epa_preference: None,
        epa_sharpness: None,
        epa_roughness: Some(0.99),
        group_delay_std_ms: None,
    };

    let checks = compare_scorecards(&baseline, &candidate);
    let roughness = checks
        .iter()
        .find(|(name, _, _)| *name == "roughness")
        .expect("roughness check");

    assert!(roughness.1, "{}", roughness.2);
}

#[test]
fn scorecard_allows_absolute_slack_at_flat_loss_ratio_boundary() {
    let baseline = MetricScorecard {
        flat_loss: 9.0,
        peak_residual_db: 10.0,
        max_boost_db: 0.0,
        max_section_gain_db: 0.0,
        replay_bundle: None,
        electrical: None,
        correction_reverted: false,
        epa_preference: None,
        epa_sharpness: None,
        epa_roughness: None,
        group_delay_std_ms: None,
    };
    let candidate = MetricScorecard {
        flat_loss: 14.30,
        peak_residual_db: 10.0,
        max_boost_db: 0.0,
        max_section_gain_db: 0.0,
        replay_bundle: None,
        electrical: None,
        correction_reverted: false,
        epa_preference: None,
        epa_sharpness: None,
        epa_roughness: None,
        group_delay_std_ms: None,
    };

    let checks = compare_scorecards(&baseline, &candidate);
    let flat_loss = checks
        .iter()
        .find(|(name, _, _)| *name == "flat_loss")
        .expect("flat-loss check");
    assert!(flat_loss.1, "{}", flat_loss.2);
}

fn scorecard_with_epa(
    preference: Option<f64>,
    sharpness: Option<f64>,
    roughness: Option<f64>,
) -> MetricScorecard {
    MetricScorecard {
        flat_loss: 1.0,
        peak_residual_db: 1.0,
        max_boost_db: 0.0,
        max_section_gain_db: 0.0,
        replay_bundle: None,
        electrical: None,
        correction_reverted: false,
        epa_preference: preference,
        epa_sharpness: sharpness,
        epa_roughness: roughness,
        group_delay_std_ms: None,
    }
}

#[test]
fn scorecard_rejects_missing_candidate_psychoacoustic_metrics() {
    let baseline = scorecard_with_epa(Some(8.0), Some(1.2), Some(0.3));
    let candidate = scorecard_with_epa(None, None, None);
    let checks = compare_scorecards(&baseline, &candidate);

    for metric in ["epa_preference", "sharpness", "roughness"] {
        let check = checks
            .iter()
            .find(|(name, _, _)| *name == metric)
            .unwrap_or_else(|| panic!("missing {metric} QA check"));
        assert!(!check.1, "{metric} omission passed: {}", check.2);
        assert!(check.2.contains("omitted"), "{}", check.2);
    }
}

#[test]
fn scorecard_rejects_large_psychoacoustic_regressions() {
    let baseline = scorecard_with_epa(Some(8.0), Some(1.2), Some(0.3));
    let candidate = scorecard_with_epa(Some(4.0), Some(2.5), Some(1.1));
    let checks = compare_scorecards(&baseline, &candidate);

    for metric in ["epa_preference", "sharpness", "roughness"] {
        let check = checks
            .iter()
            .find(|(name, _, _)| *name == metric)
            .unwrap_or_else(|| panic!("missing {metric} QA check"));
        assert!(!check.1, "{metric} regression passed: {}", check.2);
    }
}

#[test]
fn qa_seed_is_stable_and_label_specific() {
    assert_eq!(qa_seed("case:a"), qa_seed("case:a"));
    assert_ne!(qa_seed("case:a"), qa_seed("case:b"));
}

#[test]
fn target_reshaping_options_are_only_tilt_and_broadband() {
    assert!(
        OptionOverride::TargetTilt {
            slope_db_per_octave: -0.8
        }
        .reshapes_target()
    );
    assert!(OptionOverride::BroadbandTargetMatching.reshapes_target());
    assert!(!OptionOverride::Psychoacoustic.reshapes_target());
    assert!(!OptionOverride::ExcursionProtection.reshapes_target());
    assert!(!OptionOverride::PhaseAlignment.reshapes_target());
    assert!(!OptionOverride::AsymmetricLoss.reshapes_target());
}

#[test]
fn phase_alignment_validator_skips_flat_ratio_when_target_reshaped() {
    let mut baseline = result_with_channel_slopes(0.0, 0.0, 0.0);
    baseline.combined_post_score = 10.0;
    let mut option = result_with_channel_slopes(0.0, 0.0, 0.0);
    // Inflated by a companion tilt option; exceeds any flat-ratio limit.
    option.combined_post_score = 17.0;

    let (pass_without, _) = validate_phase_alignment(&baseline, &option, 1, false);
    assert!(
        !pass_without,
        "flat-ratio gate must reject 17.0 vs 10.0 without target reshaping"
    );

    let (pass_with, reason) = validate_phase_alignment(&baseline, &option, 3, true);
    assert!(
        pass_with,
        "target-reshaped combo should skip the flat-ratio gate: {reason}"
    );
    assert!(reason.contains("target reshaped"));

    // The exemption is not a blank cheque: non-finite scores still fail.
    option.combined_post_score = f64::NAN;
    let (pass_nan, _) = validate_phase_alignment(&baseline, &option, 3, true);
    assert!(!pass_nan, "non-finite scores must fail even when reshaped");
}

#[test]
fn registry_expectations_block_weak_or_overboosted_quality_results() {
    let mut results = vec![TestResult {
        label: "candidate".to_string(),
        pre_score: 10.0,
        scorecard: MetricScorecard {
            flat_loss: 9.9995,
            peak_residual_db: 1.0,
            max_boost_db: 12.5,
            max_section_gain_db: 3.0,
            replay_bundle: None,
            electrical: None,
            correction_reverted: false,
            epa_preference: None,
            epa_sharpness: None,
            epa_roughness: None,
            group_delay_std_ms: None,
        },
        pass: true,
        reason: "local checks passed".to_string(),
    }];
    super::enforce_registry_expectations(
        "quality/example",
        &["option_effect".to_string()],
        crate::registry::ScenarioExpect {
            improvement_min_pct: 0.01,
            max_post_score: 20.0,
            max_boost_db: 12.0,
            allow_safe_revert: false,
            gate_purpose: crate::registry::QaGatePurpose::Quality,
        },
        &mut results,
    );
    assert!(!results[0].pass);
    assert!(!results[0].reason.contains("improvement"));
    assert!(results[0].reason.contains("max boost"));
    assert!(results[0].reason.contains("registry=quality/example"));

    results[0].label = "candidate +50% max_db".to_string();
    results[0].pass = true;
    results[0].reason = "relationship checks passed".to_string();
    results[0].scorecard.flat_loss = 9.0;
    super::enforce_registry_expectations(
        "quality/max-db-probe",
        &["workflow".to_string()],
        crate::registry::ScenarioExpect {
            improvement_min_pct: 0.01,
            max_post_score: 20.0,
            max_boost_db: 12.0,
            allow_safe_revert: false,
            gate_purpose: crate::registry::QaGatePurpose::Quality,
        },
        &mut results,
    );
    assert!(
        !results[0].pass,
        "relaxed section bounds do not relax electrical limits"
    );

    results[0].pass = true;
    results[0].reason = "runtime safety fallback".to_string();
    results[0].pre_score = 22.0;
    results[0].scorecard.flat_loss = 22.0;
    results[0].scorecard.max_boost_db = 0.0;
    results[0].scorecard.correction_reverted = true;
    super::enforce_registry_expectations(
        "quality/allowed-revert",
        &["workflow".to_string()],
        crate::registry::ScenarioExpect {
            improvement_min_pct: 0.01,
            max_post_score: 20.0,
            max_boost_db: 12.0,
            allow_safe_revert: true,
            gate_purpose: crate::registry::QaGatePurpose::Safety,
        },
        &mut results,
    );
    assert!(results[0].pass);
    assert_eq!(results[0].outcome(), super::types::QaOutcome::Reverted);
}

#[test]
fn registry_functional_artifact_accepts_safe_revert() {
    let mut results = vec![TestResult {
        label: "measured artifact".to_string(),
        pre_score: 10.0,
        scorecard: MetricScorecard {
            flat_loss: 11.0,
            peak_residual_db: 1.0,
            max_boost_db: 3.0,
            max_section_gain_db: 3.0,
            replay_bundle: None,
            electrical: None,
            correction_reverted: true,
            epa_preference: None,
            epa_sharpness: None,
            epa_roughness: None,
            group_delay_std_ms: None,
        },
        pass: true,
        reason: "artifact and safety checks passed".to_string(),
    }];

    super::enforce_registry_expectations(
        "quality/measured-artifact",
        &["cross_mode".to_string(), "functional_artifact".to_string()],
        crate::registry::ScenarioExpect {
            improvement_min_pct: 0.01,
            max_post_score: 20.0,
            max_boost_db: 12.0,
            allow_safe_revert: false,
            gate_purpose: crate::registry::QaGatePurpose::Functional,
        },
        &mut results,
    );

    assert!(results[0].pass, "{}", results[0].reason);
    assert_eq!(results[0].outcome(), super::types::QaOutcome::Reverted);
}

#[test]
fn registry_correction_thresholds_skip_relationship_only_rows() {
    let mut results = vec![TestResult {
        label: "cross-mode relationship".to_string(),
        pre_score: 0.0,
        scorecard: MetricScorecard {
            flat_loss: 50.0,
            peak_residual_db: 0.0,
            max_boost_db: 50.0,
            max_section_gain_db: 0.0,
            replay_bundle: None,
            electrical: None,
            correction_reverted: false,
            epa_preference: None,
            epa_sharpness: None,
            epa_roughness: None,
            group_delay_std_ms: None,
        },
        pass: true,
        reason: "relationship-specific bound passed".to_string(),
    }];
    super::enforce_registry_expectations(
        "quality/cross-mode",
        &["cross_mode".to_string()],
        crate::registry::ScenarioExpect {
            improvement_min_pct: 0.01,
            max_post_score: 20.0,
            max_boost_db: 12.0,
            allow_safe_revert: false,
            gate_purpose: crate::registry::QaGatePurpose::Functional,
        },
        &mut results,
    );
    assert!(results[0].pass, "{}", results[0].reason);
}

#[test]
fn quality_maxeval_defaults_to_convergence_budget() {
    assert_eq!(
        parse_maxeval(&["roomeq-qa-quality".into()]).unwrap(),
        QA_MAXEVAL
    );
}

#[test]
fn quality_maxeval_accepts_positive_contract_budget() {
    let args = [
        "roomeq-qa-quality".into(),
        "--maxeval".into(),
        "1234".into(),
    ];
    assert_eq!(parse_maxeval(&args).unwrap(), 1234);
}

#[test]
fn strict_measured_maxeval_clamps_only_optimizer_iterations() {
    let mut config = RoomConfig::default();
    config.optimizer.algorithm = "autoeq:cmaes".to_string();
    config.optimizer.population = 20;
    config.optimizer.num_filters = 7;
    config.optimizer.max_iter = 600_000;
    config.optimizer.seed = Some(42);
    let mut expected = config.clone();
    expected.optimizer.max_iter = 15_000;

    clamp_strict_measured_maxeval(&mut config, 15_000);

    assert_eq!(
        serde_json::to_value(&config).unwrap(),
        serde_json::to_value(&expected).unwrap()
    );
}

#[test]
fn quality_maxeval_rejects_missing_zero_and_invalid_values() {
    for args in [
        vec!["roomeq-qa-quality".into(), "--maxeval".into()],
        vec!["roomeq-qa-quality".into(), "--maxeval".into(), "0".into()],
        vec![
            "roomeq-qa-quality".into(),
            "--maxeval".into(),
            "invalid".into(),
        ],
    ] {
        assert!(parse_maxeval(&args).is_err(), "accepted {args:?}");
    }
}

#[test]
fn quality_seed_runs_defaults_to_convergence_distribution() {
    assert_eq!(parse_seed_runs(&["roomeq-qa-quality".into()]).unwrap(), 5);
}

#[test]
fn quality_seed_runs_accepts_single_contract_seed() {
    let args = ["roomeq-qa-quality".into(), "--seed-runs".into(), "1".into()];
    assert_eq!(parse_seed_runs(&args).unwrap(), 1);
}

#[test]
fn quality_seed_runs_rejects_unsupported_values() {
    for args in [
        vec!["roomeq-qa-quality".into(), "--seed-runs".into()],
        vec!["roomeq-qa-quality".into(), "--seed-runs".into(), "2".into()],
        vec![
            "roomeq-qa-quality".into(),
            "--seed-runs".into(),
            "invalid".into(),
        ],
    ] {
        assert!(parse_seed_runs(&args).is_err(), "accepted {args:?}");
    }
}
