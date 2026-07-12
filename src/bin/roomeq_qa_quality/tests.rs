use super::consts::qa_seed;
use super::metric_scorecard::MetricScorecard;
use super::metric_scorecard::compare_scorecards;
use super::option_override::OptionOverride;
use super::validate::{validate_option_effect, validate_target_tilt};
use autoeq::Curve;
use autoeq::roomeq::RoomConfig;
use std::collections::HashMap;

use autoeq::roomeq::{
    ChannelDspChain, ChannelOptimizationResult, OptimizationMetadata, RoomOptimizationResult,
    StageOutcome, StageStatus,
};

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
            stage_outcomes: Vec::new(),
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

    let (pass, detail) = validate_target_tilt(-0.8, &baseline, &config, &option, 1, false, false);

    assert!(pass, "{detail}");
}

#[test]
fn target_tilt_validator_rejects_response_that_regresses_from_target() {
    let baseline = result_with_channel_slopes(0.0, 0.0, 0.0);
    let option = result_with_channel_slopes(0.0, 1.0, -0.8);
    let config = empty_room_config();

    let (pass, _) = validate_target_tilt(-0.8, &baseline, &config, &option, 1, false, false);

    assert!(!pass);
}

#[test]
fn target_tilt_validator_rejects_wrong_target_curve_slope() {
    let baseline = result_with_channel_slopes(0.0, 0.0, 0.0);
    let option = result_with_channel_slopes(0.0, 0.0, 0.0);
    let config = empty_room_config();

    let (pass, _) = validate_target_tilt(-0.8, &baseline, &config, &option, 1, false, false);

    assert!(!pass);
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
        epa_preference: None,
        epa_sharpness: None,
        epa_roughness: Some(0.95),
        group_delay_std_ms: None,
    };
    let candidate = MetricScorecard {
        flat_loss: 9.0,
        peak_residual_db: 1.0,
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

fn scorecard_with_epa(
    preference: Option<f64>,
    sharpness: Option<f64>,
    roughness: Option<f64>,
) -> MetricScorecard {
    MetricScorecard {
        flat_loss: 1.0,
        peak_residual_db: 1.0,
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
