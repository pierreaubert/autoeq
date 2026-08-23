use super::consts::qa_seed;
use super::metric_scorecard::MetricScorecard;
use super::metric_scorecard::compare_scorecards;
use super::option_override::OptionOverride;
use super::types::TestResult;
use super::validate::{
    TargetTiltValidationOptions, validate_option_effect, validate_phase_alignment,
    validate_target_tilt,
};
use roomeq_model::{
    ChannelDspChain, Curve, OptimizationMetadata, RoomConfig, StageOutcome, StageStatus,
};
use std::collections::HashMap;

use roomeq_engine::room_result::{ChannelOptimizationResult, RoomOptimizationResult};

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
            supporting_source: None,
            correction_acceptance: None,
            optimizer_evidence: None,
            stage_outcomes: Vec::new(),
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
        },
        &mut results,
    );
    assert!(results[0].pass, "{}", results[0].reason);

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
        },
        &mut results,
    );
    assert!(results[0].pass);
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
        },
        &mut results,
    );
    assert!(results[0].pass, "{}", results[0].reason);
}
