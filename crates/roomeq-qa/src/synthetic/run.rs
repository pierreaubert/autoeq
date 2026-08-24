use super::build::build_config;
use super::build::build_multichannel_config;
use super::build::build_multisub_config;
use super::build::configure_processing_mode;
use super::channel_layout::ChannelLayout;
use super::consts::MS_OPTIONS;
use super::consts::OPTIONS;
use super::consts::SAMPLE_RATE;
use super::consts::TEMP_DIR_COUNTER;
use super::misc::avg_epa_preference;
use super::misc::make_multiseat_qa_curve;
use super::types::DifficultyLevel;
use super::types::MultiSubDifficulty;
use super::types::MultiSubTopology;
use super::types::SubTopology;
use super::types::TestResult;
use anyhow::Result;
use roomeq_engine::multiseat::{MultiSeatMeasurements, optimize_multiseat};
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{
    CorrectionDecision, Curve, MultiSeatConfig, MultiSeatStrategy, ProcessingMode, RoomConfig,
};
use std::sync::atomic::Ordering;

fn correction_was_reverted(result: &RoomOptimizationResult) -> bool {
    result
        .metadata
        .correction_acceptance
        .as_ref()
        .is_some_and(|report| {
            matches!(
                report.decision,
                CorrectionDecision::RevertedStage | CorrectionDecision::IdentityFallback
            )
        })
}

pub(super) fn run_optimization(config: &RoomConfig) -> Result<RoomOptimizationResult> {
    let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_dir =
        std::env::temp_dir().join(format!("roomeq_qa_syn_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&temp_dir)?;
    let result = crate::optimize_room(config, SAMPLE_RATE, Some(&temp_dir));
    let _ = std::fs::remove_dir_all(&temp_dir);
    result
}

pub(super) fn run_single_test(
    degraded: &Curve,
    mode: ProcessingMode,
    target_name: &str,
    option_names: &[&str],
    difficulty: &DifficultyLevel,
) -> TestResult {
    let mode_str = match mode {
        ProcessingMode::LowLatency => "IIR",
        ProcessingMode::PhaseLinear => "FIR",
        ProcessingMode::Hybrid => "Mixed",
        ProcessingMode::MixedPhase => "MixedPhase",
        ProcessingMode::WarpedIir => "WarpedIIR",
        ProcessingMode::KautzModal => "KautzModal",
    };
    let options_str = if option_names.is_empty() {
        "baseline".to_string()
    } else {
        option_names.join("+")
    };

    let test_name = format!(
        "{}/{}/{}/{}",
        difficulty.name, mode_str, target_name, options_str
    );

    let mut config = build_config(degraded, mode);

    // Apply option overrides
    for opt_name in option_names {
        if let Some(opt) = OPTIONS.iter().find(|o| o.name == *opt_name) {
            (opt.apply)(&mut config);
        }
    }

    let result = match run_optimization(&config) {
        Ok(r) => r,
        Err(e) => {
            return TestResult {
                name: test_name,
                passed: false,
                pre_score: 0.0,
                post_score: 0.0,
                epa_preference: None,
                reason: format!("Optimization failed: {}", e),
            };
        }
    };

    // Every quality scenario must improve over its uncorrected baseline.
    let pre = result.combined_pre_score;
    let post = result.combined_post_score;
    let epa = avg_epa_preference(&result);

    if post >= pre {
        let reverted = correction_was_reverted(&result);
        let verdict = if reverted {
            "REVERTED"
        } else {
            "No improvement"
        };
        return TestResult {
            name: test_name,
            // A runtime safety revert preserves the baseline by design and is
            // accepted by the coverage matrix as a successful safety outcome.
            passed: reverted,
            pre_score: pre,
            post_score: post,
            epa_preference: epa,
            reason: format!("{verdict}: pre={:.3}, post={:.3}", pre, post),
        };
    }

    TestResult {
        name: test_name,
        passed: true,
        pre_score: pre,
        post_score: post,
        epa_preference: epa,
        reason: format!(
            "OK: {:.3} -> {:.3} ({:+.1}%)",
            pre,
            post,
            (1.0 - post / pre) * 100.0
        ),
    }
}

pub(super) fn run_multisub_test(
    sub_curves: &[Curve],
    topology: &MultiSubTopology,
    option_names: &[&str],
    difficulty: &MultiSubDifficulty,
) -> TestResult {
    let options_str = if option_names.is_empty() {
        "baseline".to_string()
    } else {
        option_names.join("+")
    };
    let test_name = format!(
        "multisub/{}/{}sub_{}/{}",
        difficulty.name, difficulty.n_subs, topology.name, options_str,
    );

    let mut config = build_multisub_config(sub_curves, topology.allpass);

    // Apply option overrides (use MS_OPTIONS lookup)
    for opt_name in option_names {
        if let Some(opt) = MS_OPTIONS.iter().find(|o| o.name == *opt_name) {
            (opt.apply)(&mut config);
        }
    }

    let result = match run_optimization(&config) {
        Ok(r) => r,
        Err(e) => {
            return TestResult {
                name: test_name,
                passed: false,
                pre_score: 0.0,
                post_score: 0.0,
                epa_preference: None,
                reason: format!("Optimization failed: {}", e),
            };
        }
    };

    let pre = result.combined_pre_score;
    let post = result.combined_post_score;
    let epa = avg_epa_preference(&result);

    // `pre` is already the response after multi-sub gain/delay/all-pass
    // optimization; this assertion covers the secondary global-EQ refinement,
    // not the total raw-to-MSO benefit. Require a deterministic positive
    // refinement without mislabeling it as the full MSO audibility margin.
    let required_improvement = 0.05_f64.max(pre.abs() * 0.02);
    if post > pre - required_improvement {
        let reverted = correction_was_reverted(&result);
        let verdict = if reverted {
            "REVERTED"
        } else {
            "Meaningful audibility margin not reached"
        };
        return TestResult {
            name: test_name,
            passed: reverted,
            pre_score: pre,
            post_score: post,
            epa_preference: epa,
            reason: format!(
                "{verdict}: pre={:.3}, post={:.3}, required improvement={:.3}",
                pre, post, required_improvement,
            ),
        };
    }

    TestResult {
        name: test_name,
        passed: true,
        pre_score: pre,
        post_score: post,
        epa_preference: epa,
        reason: format!(
            "OK: {:.3} -> {:.3} ({:+.1}%)",
            pre,
            post,
            (1.0 - post / pre) * 100.0
        ),
    }
}

pub(super) fn run_multiseat_missing_phase_guard() -> TestResult {
    let test_name = "multiseat/api/missing_phase_rejected".to_string();
    let flat = |_: f64| 90.0;
    let measurements = vec![
        vec![
            make_multiseat_qa_curve(flat, 0.0, true),
            make_multiseat_qa_curve(flat, 12.0, false),
        ],
        vec![
            make_multiseat_qa_curve(flat, 24.0, true),
            make_multiseat_qa_curve(flat, 36.0, true),
        ],
    ];
    let ms = match MultiSeatMeasurements::new(measurements) {
        Ok(ms) => ms,
        Err(e) if e.to_string().contains("missing phase") => {
            return TestResult {
                name: test_name,
                passed: true,
                pre_score: 0.0,
                post_score: 0.0,
                epa_preference: None,
                reason: "OK: missing phase rejected while building measurements".to_string(),
            };
        }
        Err(e) => {
            return TestResult {
                name: test_name,
                passed: false,
                pre_score: 0.0,
                post_score: 0.0,
                epa_preference: None,
                reason: format!("Failed to build measurements: {}", e),
            };
        }
    };
    let config = MultiSeatConfig {
        enabled: true,
        strategy: MultiSeatStrategy::MinimizeVariance,
        primary_seat: 0,
        max_deviation_db: 6.0,
        ..Default::default()
    };

    match optimize_multiseat(&ms, &config, (20.0, 120.0), SAMPLE_RATE) {
        Ok(result) => TestResult {
            name: test_name,
            passed: false,
            pre_score: result.objective_before,
            post_score: result.objective_after,
            epa_preference: None,
            reason: "MSO accepted a missing phase trace".to_string(),
        },
        Err(e) if e.to_string().contains("requires phase data") => TestResult {
            name: test_name,
            passed: true,
            pre_score: 0.0,
            post_score: 0.0,
            epa_preference: None,
            reason: "OK: missing phase rejected".to_string(),
        },
        Err(e) => TestResult {
            name: test_name,
            passed: false,
            pre_score: 0.0,
            post_score: 0.0,
            epa_preference: None,
            reason: format!("Unexpected rejection: {}", e),
        },
    }
}

pub(super) fn run_multiseat_strategy_metric_guard(strategy: MultiSeatStrategy) -> TestResult {
    let expected_name = match strategy {
        MultiSeatStrategy::MinimizeVariance => "seat_variance",
        MultiSeatStrategy::Average => "average_flatness",
        MultiSeatStrategy::PrimaryWithConstraints => "primary_constrained",
        MultiSeatStrategy::ModalBasis => "modal_basis",
        MultiSeatStrategy::ContinuousArea => "continuous_area",
    };
    let test_name = format!("multiseat/api/{}_metrics", expected_name);

    let flat = |_: f64| 90.0;
    let dipped = |f: f64| if f < 60.0 { 84.0 } else { 90.0 };
    let peaked = |f: f64| if f < 60.0 { 96.0 } else { 90.0 };
    let measurements = vec![
        vec![
            make_multiseat_qa_curve(flat, 0.0, true),
            make_multiseat_qa_curve(dipped, 12.0, true),
        ],
        vec![
            make_multiseat_qa_curve(peaked, 24.0, true),
            make_multiseat_qa_curve(flat, 36.0, true),
        ],
    ];
    let ms = match MultiSeatMeasurements::new(measurements) {
        Ok(ms) => ms,
        Err(e) => {
            return TestResult {
                name: test_name,
                passed: false,
                pre_score: 0.0,
                post_score: 0.0,
                epa_preference: None,
                reason: format!("Failed to build measurements: {}", e),
            };
        }
    };
    let config = MultiSeatConfig {
        enabled: true,
        strategy: strategy.clone(),
        primary_seat: 0,
        max_deviation_db: 6.0,
        ..Default::default()
    };

    let result = match optimize_multiseat(&ms, &config, (20.0, 120.0), SAMPLE_RATE) {
        Ok(result) => result,
        Err(e) => {
            return TestResult {
                name: test_name,
                passed: false,
                pre_score: 0.0,
                post_score: 0.0,
                epa_preference: None,
                reason: format!("Optimization failed: {}", e),
            };
        }
    };

    let objective_metric_matches = result.objective_name == expected_name;
    let improvement_matches_objective =
        (result.improvement_db - result.objective_improvement_db).abs() < 1e-9;
    let objective_not_worse = result.objective_after <= result.objective_before + 0.05;
    let reference_sub_fixed =
        result.gains.first() == Some(&0.0) && result.delays.first() == Some(&0.0);
    let finite_solution = result
        .gains
        .iter()
        .chain(result.delays.iter())
        .all(|value| value.is_finite());

    if !objective_metric_matches
        || !improvement_matches_objective
        || !objective_not_worse
        || !reference_sub_fixed
        || !finite_solution
    {
        return TestResult {
            name: test_name,
            passed: false,
            pre_score: result.objective_before,
            post_score: result.objective_after,
            epa_preference: None,
            reason: format!(
                "Bad MSO result: objective_name={} expected={}, objective {:.3}->{:.3}, improvement={:.3}, objective_improvement={:.3}, gains={:?}, delays={:?}",
                result.objective_name,
                expected_name,
                result.objective_before,
                result.objective_after,
                result.improvement_db,
                result.objective_improvement_db,
                result.gains,
                result.delays,
            ),
        };
    }

    TestResult {
        name: test_name,
        passed: true,
        pre_score: result.objective_before,
        post_score: result.objective_after,
        epa_preference: None,
        reason: format!(
            "OK: {} {:.3} -> {:.3} ({:+.3} dB); variance diagnostic {:+.3} dB",
            result.objective_name,
            result.objective_before,
            result.objective_after,
            result.objective_improvement_db,
            result.variance_improvement_db,
        ),
    }
}

pub(super) fn run_multiseat_phase_control_guard() -> TestResult {
    let test_name = "multiseat/api/polarity_allpass_controls".to_string();
    let flat = |_: f64| 90.0;
    let peaked = |f: f64| if f < 70.0 { 94.0 } else { 90.0 };
    // Keep the seats non-proportional across subs. If every sub has the same
    // seat-to-seat level ratio, shared per-sub controls cannot reduce variance.
    let measurements = vec![
        vec![
            make_multiseat_qa_curve(flat, 0.0, true),
            make_multiseat_qa_curve(flat, 15.0, true),
        ],
        vec![
            make_multiseat_qa_curve(peaked, 170.0, true),
            make_multiseat_qa_curve(flat, -170.0, true),
        ],
    ];
    let ms = match MultiSeatMeasurements::new(measurements) {
        Ok(ms) => ms,
        Err(e) => {
            return TestResult {
                name: test_name,
                passed: false,
                pre_score: 0.0,
                post_score: 0.0,
                epa_preference: None,
                reason: format!("Failed to build measurements: {}", e),
            };
        }
    };
    let config = MultiSeatConfig {
        enabled: true,
        strategy: MultiSeatStrategy::MinimizeVariance,
        optimize_polarity: true,
        allpass_filters_per_sub: 1,
        primary_seat: 0,
        max_deviation_db: 6.0,
        ..Default::default()
    };

    let result = match optimize_multiseat(&ms, &config, (20.0, 120.0), SAMPLE_RATE) {
        Ok(result) => result,
        Err(e) => {
            return TestResult {
                name: test_name,
                passed: false,
                pre_score: 0.0,
                post_score: 0.0,
                epa_preference: None,
                reason: format!("Optimization failed: {}", e),
            };
        }
    };

    let phase_controls_present = result.polarities.len() == 2
        && result.allpass_filters.len() == 2
        && !result.polarities[0]
        && result.allpass_filters[0].is_empty()
        && result.allpass_filters[1].len() == 1;
    let allpass_in_bounds = result
        .allpass_filters
        .get(1)
        .and_then(|filters| filters.first())
        .is_some_and(|&(freq, q)| (20.0..=120.0).contains(&freq) && (0.3..=5.0).contains(&q));
    let objective_not_worse = result.objective_after <= result.objective_before + 0.05;

    if !phase_controls_present || !allpass_in_bounds || !objective_not_worse {
        return TestResult {
            name: test_name,
            passed: false,
            pre_score: result.objective_before,
            post_score: result.objective_after,
            epa_preference: None,
            reason: format!(
                "Bad phase-control result: objective {:.3}->{:.3}, polarities={:?}, allpass={:?}",
                result.objective_before,
                result.objective_after,
                result.polarities,
                result.allpass_filters,
            ),
        };
    }

    TestResult {
        name: test_name,
        passed: true,
        pre_score: result.objective_before,
        post_score: result.objective_after,
        epa_preference: None,
        reason: format!(
            "OK: polarity/all-pass controls optimized; objective {:.3} -> {:.3}",
            result.objective_before, result.objective_after
        ),
    }
}

pub(super) fn run_multiseat_api_guard_tests() -> Vec<TestResult> {
    vec![
        run_multiseat_missing_phase_guard(),
        run_multiseat_strategy_metric_guard(MultiSeatStrategy::Average),
        run_multiseat_strategy_metric_guard(MultiSeatStrategy::PrimaryWithConstraints),
        run_multiseat_phase_control_guard(),
    ]
}

pub(super) fn report_multiseat_api_guard_tests() -> Result<bool> {
    println!("RoomEQ Synthetic QA -- multi-seat API guards");
    println!("============================================================");

    let results = run_multiseat_api_guard_tests();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;

    for result in &results {
        let status = if result.passed { "PASS" } else { "FAIL" };
        println!("  {}: {} -- {}", status, result.name, result.reason);
    }

    println!();
    println!(
        "Results: {} passed, {} failed, {} total",
        passed,
        failed,
        results.len()
    );

    Ok(failed > 0)
}

pub(super) fn run_multichannel_test(
    layout: &ChannelLayout,
    sub_topo: Option<&SubTopology>,
    difficulty: &DifficultyLevel,
    base_curve: &Curve,
    processing_mode: ProcessingMode,
    sample_rate: f64,
) -> TestResult {
    let sub_str = sub_topo.map(|s| s.name).unwrap_or("no_lfe");
    let test_name = format!(
        "multichannel/{}/{}/{}/{:?}",
        layout.name, sub_str, difficulty.name, processing_mode
    );

    let mut config = build_multichannel_config(
        layout,
        sub_topo,
        difficulty,
        base_curve,
        processing_mode.clone(),
        sample_rate,
    );
    configure_processing_mode(&mut config.optimizer, processing_mode);

    let result = match run_optimization(&config) {
        Ok(r) => r,
        Err(e) => {
            return TestResult {
                name: test_name,
                passed: false,
                pre_score: 0.0,
                post_score: 0.0,
                epa_preference: None,
                reason: format!("Optimization failed: {}", e),
            };
        }
    };

    let pre = result.combined_pre_score;
    let post = result.combined_post_score;
    let epa = avg_epa_preference(&result);

    let expected_channels = layout.mains.len() + layout.heights.len() + usize::from(layout.has_lfe);
    if result.channels.len() != expected_channels {
        return TestResult {
            name: test_name,
            passed: false,
            pre_score: pre,
            post_score: post,
            epa_preference: epa,
            reason: format!(
                "Topology mismatch: expected {expected_channels} logical channels, got {}",
                result.channels.len()
            ),
        };
    }

    if let Some(expected_subs) = sub_topo.and_then(|topology| match topology.name {
        "mso_2sub" | "mso_2sub_allpass" => Some(2),
        "mso_4sub" => Some(4),
        "mso_8sub" => Some(8),
        _ => None,
    }) {
        let actual_subs = result
            .channels
            .values()
            .filter_map(|chain| chain.drivers.as_ref())
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        if actual_subs != expected_subs {
            return TestResult {
                name: test_name,
                passed: false,
                pre_score: pre,
                post_score: post,
                epa_preference: epa,
                reason: format!(
                    "Sub topology mismatch: expected {expected_subs} physical subs, got {actual_subs}"
                ),
            };
        }
    }

    if post >= pre {
        let reverted = correction_was_reverted(&result);
        let verdict = if reverted {
            "REVERTED"
        } else {
            "No improvement"
        };
        let mut channel_scores: Vec<_> = result
            .channel_results
            .iter()
            .map(|(name, channel)| {
                format!("{name}={:.3}->{:.3}", channel.pre_score, channel.post_score)
            })
            .collect();
        channel_scores.sort();
        let stage_outcomes = result
            .metadata
            .stage_outcomes
            .iter()
            .filter(|outcome| outcome.status != roomeq_model::StageStatus::Skipped)
            .map(|outcome| format!("{}:{:?}", outcome.stage, outcome.status))
            .collect::<Vec<_>>()
            .join(",");
        let acceptance_violations = result
            .metadata
            .correction_acceptance
            .as_ref()
            .map(|report| report.violations.join(","))
            .unwrap_or_default();
        return TestResult {
            name: test_name,
            passed: reverted,
            pre_score: pre,
            post_score: post,
            epa_preference: epa,
            reason: format!(
                "{verdict}: pre={:.3}, post={:.3} ({:.1}% change); {}; stages=[{}]; acceptance=[{}]",
                pre,
                post,
                (post / pre - 1.0) * 100.0,
                channel_scores.join(", "),
                stage_outcomes,
                acceptance_violations,
            ),
        };
    }

    TestResult {
        name: test_name,
        passed: true,
        pre_score: pre,
        post_score: post,
        epa_preference: epa,
        reason: format!(
            "OK: {:.3} -> {:.3} ({:.1}% reduction)",
            pre,
            post,
            (1.0 - post / pre) * 100.0
        ),
    }
}
