use super::apply::apply_group_delay_qa_passthrough_eq;
use super::apply::apply_mutation;
use super::apply::apply_option_override;
use super::apply::apply_qa_overrides;
use super::apply::clamp_strict_measured_maxeval;
use super::consts::CORRECTION_OUT_OF_BAND_SPREAD_DB;
use super::consts::CROSS_MODE_BASS_MAX_RMS_DB;
use super::consts::CROSS_MODE_BASS_MEDIAN_RMS_DB;
use super::consts::CROSS_MODE_FR_MAX_DIFF_DB;
use super::consts::CROSS_MODE_MAIN_MEDIAN_RMS_DB;
use super::consts::CROSS_MODE_RATIO_LIMIT;
use super::consts::CROSS_MODE_SCORE_RATIO_LIMIT;
use super::consts::CROSS_MODE_TIMING_MAX_STD_MS;
use super::consts::CROSS_MODE_UPPER_MEDIAN_RMS_DB;
use super::consts::FIR_MUTATIONS;
use super::consts::IIR_MUTATIONS;
use super::consts::MIXED_MUTATIONS;
use super::consts::MIXED_PHASE_MUTATIONS;
use super::consts::SAMPLE_RATE;
use super::consts::TEMP_DIR_COUNTER;
use super::group::group_delay_std_dev;
use super::group_delay_qa_profile::disable_option;
use super::group_delay_qa_profile::prepare_option_measurement_paths;
use super::metric_scorecard::MetricScorecard;
use super::metric_scorecard::compare_scorecards;
use super::metric_scorecard::compute_scorecard;
use super::metric_scorecard::evaluate_scorecard;
use super::metric_scorecard::placeholder_scorecard;
use super::misc::convergence_epsilon;
use super::misc::level_matched_rms_curve_difference_db;
use super::misc::load_config_for_generic_path;
use super::misc::load_config_for_path;
use super::misc::max_curve_difference_db;
use super::mutation::Mutation;
use super::option::isolate_schroeder_split_from_multi_measurement;
use super::option::option_gd_profile;
use super::option::option_is_group_delay;
use super::option::option_needs_gd_trusted_measurements;
use super::option::option_needs_multi_measurement_paths;
use super::option::option_needs_multisub_multi_seat_paths;
use super::option_override::OptionOverride;
use super::types::TestResult;
use super::validate::validate_option_effect;
use anyhow::{Context, Result};
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{Curve, ProcessingMode, RoomConfig, TargetResponseConfig, TargetShape};
use roomeq_workflow::load_config;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::Ordering;

pub(super) fn run_optimization(
    config: &RoomConfig,
    seed_runs: usize,
) -> Result<RoomOptimizationResult> {
    let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_dir = std::env::temp_dir().join(format!("roomeq_qa_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&temp_dir)?;
    let result = if seed_runs == 1 {
        crate::optimize_room_single_seed(config, SAMPLE_RATE)
    } else {
        crate::optimize_room(config, SAMPLE_RATE, Some(&temp_dir))
    };
    let _ = std::fs::remove_dir_all(&temp_dir);
    result
}

pub(super) fn run_stereo_workflow_tests(
    name: &str,
    base_config_path: &Path,
    override_config_path: Option<&Path>,
    maxeval: usize,
    seed_runs: usize,
) -> Result<(String, Vec<TestResult>)> {
    let mut out = String::new();
    let mut results = Vec::new();

    writeln!(out, "\n--- {} (IIR workflow) ---", name).unwrap();

    let mut baseline_scorecard: Option<MetricScorecard> = None;

    for mutation in IIR_MUTATIONS {
        let (mut config, _, _validation) = load_config(base_config_path, override_config_path)?;
        apply_qa_overrides(&mut config, &format!("{name}:iir:{mutation}"), maxeval);
        apply_mutation(&mut config, *mutation);

        let result = run_optimization(&config, seed_runs)
            .with_context(|| format!("{} IIR {}", name, mutation))?;

        let pre = result.combined_pre_score;
        let scorecard = compute_scorecard(&result);

        let (pass, reason) =
            evaluate_scorecard(*mutation, pre, &scorecard, &mut baseline_scorecard);

        let status = if pass { "PASS" } else { "FAIL" };
        writeln!(
            out,
            "  IIR {:>14}: {}  {}  ({})",
            mutation.to_string(),
            scorecard,
            status,
            reason
        )
        .unwrap();

        results.push(TestResult {
            label: format!("{} IIR {}", name, mutation),
            pre_score: pre,
            scorecard,
            pass,
            reason,
        });
    }

    Ok((out, results))
}

/// Exercise a non-IIR override through the production config-loading path.
///
/// The generic-path matrix deliberately mutates processing modes in memory;
/// this smoke gate is separate so a broken or misleading checked-in FIR or
/// Hybrid override cannot remain hidden behind that mutation.
pub(super) fn run_workflow_override_smoke(
    name: &str,
    mode_name: &str,
    expected_mode: ProcessingMode,
    base_config_path: &Path,
    override_config_path: &Path,
    maxeval: usize,
    seed_runs: usize,
) -> Result<(String, Vec<TestResult>)> {
    let mut out = String::new();
    let (mut config, _, _validation) = load_config(base_config_path, Some(override_config_path))?;
    anyhow::ensure!(
        config.optimizer.processing_mode == expected_mode,
        "{} workflow override {} claims {:?}, but its merged config selects {:?}",
        name,
        override_config_path.display(),
        expected_mode,
        config.optimizer.processing_mode
    );

    apply_qa_overrides(
        &mut config,
        &format!(
            "{name}:workflow:{}:baseline",
            mode_name.to_ascii_lowercase()
        ),
        maxeval,
    );
    let result = run_optimization(&config, seed_runs)
        .with_context(|| format!("{name} {mode_name} workflow baseline"))?;
    let pre = result.combined_pre_score;
    let scorecard = compute_scorecard(&result);
    let mut baseline_scorecard = None;
    let (pass, reason) =
        evaluate_scorecard(Mutation::Baseline, pre, &scorecard, &mut baseline_scorecard);
    let status = if pass { "PASS" } else { "FAIL" };
    writeln!(
        out,
        "  {mode_name:>10} workflow: {scorecard} {status} ({reason})"
    )
    .unwrap();

    Ok((
        out,
        vec![TestResult {
            label: format!("{name} {mode_name} workflow baseline"),
            pre_score: pre,
            scorecard,
            pass,
            reason,
        }],
    ))
}

pub(super) fn run_generic_path_tests(
    name: &str,
    base_config_path: &Path,
    override_config_dir: &Path,
    maxeval: usize,
    seed_runs: usize,
) -> Result<(String, Vec<TestResult>)> {
    let mut out = String::new();
    let mut results = Vec::new();

    writeln!(out, "\n--- Generic Path ({}, all modes) ---", name).unwrap();

    let modes: &[(&str, ProcessingMode, &str, &[Mutation])] = &[
        (
            "IIR",
            ProcessingMode::LowLatency,
            "optimiser-iir.json",
            IIR_MUTATIONS,
        ),
        (
            "FIR",
            ProcessingMode::PhaseLinear,
            "optimiser-fir.json",
            FIR_MUTATIONS,
        ),
        (
            "Mixed",
            ProcessingMode::Hybrid,
            "optimiser-mixed.json",
            MIXED_MUTATIONS,
        ),
        (
            "MixedPhase",
            ProcessingMode::MixedPhase,
            "../modes/optimiser-mixed-phase.json",
            MIXED_PHASE_MUTATIONS,
        ),
    ];

    let mut mode_baselines: Vec<(&str, f64)> = Vec::new();

    for (mode_name, processing_mode, override_file, mutations) in modes {
        let scenario_override = override_config_dir.join(override_file);
        let shared_override = override_config_dir
            .parent()
            .unwrap_or(override_config_dir)
            .join("modes")
            .join(
                Path::new(override_file)
                    .file_name()
                    .unwrap_or_else(|| override_file.as_ref()),
            );
        let override_path = if scenario_override.exists() {
            scenario_override
        } else {
            shared_override
        };
        let mut baseline_scorecard: Option<MetricScorecard> = None;

        for mutation in *mutations {
            let (mut config, _) = load_config_for_generic_path(
                base_config_path,
                Some(&override_path),
                processing_mode.clone(),
            )?;
            apply_qa_overrides(
                &mut config,
                &format!("{name}:generic:{mode_name}:{mutation}"),
                maxeval,
            );
            // Generic-path cases compare processing modes and budget/filter
            // mutations. Keep their scalar objective aligned with the flat-loss
            // scorecard and runtime acceptance metric; psychoacoustic and
            // asymmetric objectives have dedicated option-effect cases.
            config.optimizer.psychoacoustic = false;
            config.optimizer.asymmetric_loss = false;
            apply_mutation(&mut config, *mutation);

            let result = run_optimization(&config, seed_runs)
                .with_context(|| format!("{} {} generic {}", name, mode_name, mutation))?;

            let pre = result.combined_pre_score;
            let scorecard = compute_scorecard(&result);

            let (pass, reason) =
                evaluate_scorecard(*mutation, pre, &scorecard, &mut baseline_scorecard);

            // Record baseline for cross-mode comparison
            if matches!(mutation, Mutation::Baseline) {
                mode_baselines.push((mode_name, scorecard.flat_loss));
            }

            let status = if pass { "PASS" } else { "FAIL" };
            writeln!(
                out,
                "  {} {:>14}: {}  {}  ({})",
                mode_name,
                mutation.to_string(),
                scorecard,
                status,
                reason
            )
            .unwrap();

            results.push(TestResult {
                label: format!("{} generic {} {}", name, mode_name, mutation),
                pre_score: pre,
                scorecard,
                pass,
                reason,
            });
        }
    }

    // Cross-mode comparison
    if mode_baselines.len() >= 2 {
        let scores: Vec<f64> = mode_baselines.iter().map(|(_, s)| *s).collect();
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ratio = if min_score > 0.0 {
            max_score / min_score
        } else {
            f64::INFINITY
        };
        let pass = ratio <= CROSS_MODE_RATIO_LIMIT;
        let status = if pass { "PASS" } else { "FAIL" };

        let mode_scores: String = mode_baselines
            .iter()
            .map(|(name, score)| format!("{}={:.4}", name, score))
            .collect::<Vec<_>>()
            .join(" ");

        writeln!(
            out,
            "\n  Cross-mode: {} ratio={:.2}x  {}",
            mode_scores, ratio, status
        )
        .unwrap();

        results.push(TestResult {
            label: format!("{} cross-mode", name),
            pre_score: 0.0,
            scorecard: placeholder_scorecard(ratio),
            pass,
            reason: format!("ratio={:.2}x (limit={:.1}x)", ratio, CROSS_MODE_RATIO_LIMIT),
        });
    }

    Ok((out, results))
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

pub(super) fn deployed_final_curve(
    result: &RoomOptimizationResult,
    channel: &str,
) -> Option<Curve> {
    result
        .deployed_source_curves
        .get(channel)
        .cloned()
        .or_else(|| {
            result
                .channel_results
                .get(channel)
                .map(|channel| channel.final_curve.clone())
        })
        .or_else(|| {
            result
                .channels
                .get(channel)
                .and_then(|chain| chain.final_curve.clone())
                .map(Curve::from)
        })
}

fn redirected_main_channels(result: &RoomOptimizationResult) -> Vec<String> {
    let routed: std::collections::BTreeSet<String> = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|report| report.routing_graph.as_ref())
        .into_iter()
        .flat_map(|graph| graph.routes.iter())
        .filter(|route| route.route_kind == "main_highpass_to_self")
        .map(|route| route.source_channel.clone())
        .collect();
    if !routed.is_empty() {
        return routed.into_iter().collect();
    }
    result
        .channel_results
        .keys()
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            !lower.contains("lfe") && !lower.contains("sub")
        })
        .cloned()
        .collect()
}

fn correction_passband_violations(result: &RoomOptimizationResult) -> Vec<String> {
    let mut violations = Vec::new();
    for (channel_name, channel_result) in &result.channel_results {
        let Some(reliable_upper_hz) = roomeq_engine::spectral_align::reliable_upper_passband_hz(
            &channel_result.initial_curve,
        ) else {
            continue;
        };
        let Some(chain) = result.channels.get(channel_name) else {
            continue;
        };
        for plugin in &chain.plugins {
            if plugin.plugin_type != "eq" {
                continue;
            }
            let Some(filters) = plugin
                .parameters
                .get("filters")
                .and_then(serde_json::Value::as_array)
            else {
                continue;
            };
            let stage = plugin
                .parameters
                .get("label")
                .or_else(|| plugin.parameters.get("room_eq_correction_stage"))
                .or_else(|| plugin.parameters.get("room_eq_stage"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("eq");
            for frequency_hz in filters
                .iter()
                .filter_map(|filter| filter.get("freq").and_then(serde_json::Value::as_f64))
            {
                if frequency_hz > reliable_upper_hz {
                    violations.push(format!(
                        "{channel_name}:{stage}:{frequency_hz:.1}Hz>{reliable_upper_hz:.1}Hz"
                    ));
                }
            }
        }

        let lower_channel_name = channel_name.to_ascii_lowercase();
        let is_bass_output =
            lower_channel_name.contains("lfe") || lower_channel_name.starts_with("sub");
        if !is_bass_output && let Some(eq_response) = chain.eq_response.as_ref() {
            let audit_start_hz = reliable_upper_hz * 2.0_f64.powf(1.0 / 6.0);
            let mut minimum_db = f64::INFINITY;
            let mut maximum_db = f64::NEG_INFINITY;
            let mut sample_count = 0usize;
            for (frequency_hz, level_db) in eq_response.freq.iter().zip(&eq_response.spl) {
                if *frequency_hz >= audit_start_hz && level_db.is_finite() {
                    minimum_db = minimum_db.min(*level_db);
                    maximum_db = maximum_db.max(*level_db);
                    sample_count += 1;
                }
            }
            let spread_db = maximum_db - minimum_db;
            if sample_count >= 3 && spread_db > CORRECTION_OUT_OF_BAND_SPREAD_DB {
                violations.push(format!(
                    "{channel_name}:out-of-band correction spread {spread_db:.2}dB>{CORRECTION_OUT_OF_BAND_SPREAD_DB:.2}dB above {audit_start_hz:.1}Hz"
                ));
            }
        }
    }
    violations
}

pub(super) fn run_cross_mode_convergence_tests(
    name: &str,
    base_config_path: &Path,
    override_config_dir: &Path,
    preserve_system: bool,
    strict: bool,
    maxeval: usize,
    seed_runs: usize,
) -> Result<(String, Vec<TestResult>)> {
    let mut out = String::new();
    let mut results = Vec::new();

    writeln!(out, "\n--- {} (cross-mode convergence) ---", name).unwrap();

    let modes: &[(&str, ProcessingMode, &str)] = &[
        ("IIR", ProcessingMode::LowLatency, "optimiser-iir.json"),
        ("FIR", ProcessingMode::PhaseLinear, "optimiser-fir.json"),
        ("Hybrid", ProcessingMode::Hybrid, "optimiser-mixed.json"),
        (
            "MixedPhase",
            ProcessingMode::MixedPhase,
            "optimiser-mixed-phase.json",
        ),
    ];

    // Run every production processing mode and collect comparable artifacts.
    let mut mode_results: Vec<(&str, RoomOptimizationResult)> = Vec::new();

    for (mode_name, processing_mode, override_file) in modes {
        let override_path = override_config_dir.join(override_file);
        let (mut config, _) = load_config_for_path(
            base_config_path,
            Some(&override_path),
            processing_mode.clone(),
            preserve_system,
        )?;
        if !strict {
            apply_qa_overrides(
                &mut config,
                &format!("{name}:cross-mode:{mode_name}"),
                maxeval,
            );
        } else {
            clamp_strict_measured_maxeval(&mut config, maxeval);
        }
        // Strict measured regressions exercise the checked-in production
        // fixture unchanged except for clamping optimizer.max_iter. Filter
        // count, algorithm, population, seed, and every acoustic option remain
        // the checked-in production values.

        let result = run_optimization(&config, seed_runs)
            .with_context(|| format!("{} {} cross-mode", name, mode_name))?;

        writeln!(
            out,
            "  {}: post={:.4} (pre={:.4})",
            mode_name, result.combined_post_score, result.combined_pre_score
        )
        .unwrap();

        let pre = result.combined_pre_score;
        let scorecard = compute_scorecard(&result);
        let (pass, reason) = if strict {
            let pass = pre.is_finite()
                && scorecard.flat_loss.is_finite()
                && scorecard.max_boost_db.is_finite();
            (
                pass,
                "measured-mode artifact produced with finite metrics".to_string(),
            )
        } else {
            let mut baseline_scorecard = None;
            evaluate_scorecard(Mutation::Baseline, pre, &scorecard, &mut baseline_scorecard)
        };
        results.push(TestResult {
            label: format!("{name} {mode_name} correction"),
            pre_score: pre,
            scorecard,
            pass,
            reason,
        });

        if strict {
            let violations = correction_passband_violations(&result);
            let pass = violations.is_empty();
            results.push(TestResult {
                label: format!("{name} {mode_name} measured-passband correction bounds"),
                pre_score: 0.0,
                scorecard: placeholder_scorecard(if pass { 0.0 } else { f64::INFINITY }),
                pass,
                reason: if pass {
                    "all correction stages stay within each measured passband".to_string()
                } else {
                    format!("out-of-passband correction: {}", violations.join(", "))
                },
            });
        }

        mode_results.push((mode_name, result));
    }

    // CM-1: Frequency-response convergence from the final deployed channel
    // curves. Strict cases use level-matched RMS bands; legacy generic cases
    // retain their historical broad maximum-difference smoke gate.
    if strict {
        let channel_names = redirected_main_channels(&mode_results[0].1);
        let mode_pairs: Vec<(usize, usize)> = (0..mode_results.len())
            .flat_map(|first| ((first + 1)..mode_results.len()).map(move |second| (first, second)))
            .collect();
        let bands = [
            (
                "bass",
                25.0,
                250.0,
                CROSS_MODE_BASS_MEDIAN_RMS_DB,
                Some(CROSS_MODE_BASS_MAX_RMS_DB),
            ),
            ("main", 100.0, 10_000.0, CROSS_MODE_MAIN_MEDIAN_RMS_DB, None),
            (
                "upper",
                300.0,
                10_000.0,
                CROSS_MODE_UPPER_MEDIAN_RMS_DB,
                None,
            ),
        ];
        for (band_name, fmin, fmax, median_limit, max_limit) in bands {
            let mut differences = Vec::new();
            for channel in &channel_names {
                let curves: Vec<Option<Curve>> = mode_results
                    .iter()
                    .map(|(_, result)| deployed_final_curve(result, channel))
                    .collect();
                for &(first, second) in &mode_pairs {
                    if let (Some(first), Some(second)) = (&curves[first], &curves[second])
                        && let Some(rms) =
                            level_matched_rms_curve_difference_db(first, second, fmin, fmax)
                    {
                        differences.push(rms);
                    }
                }
            }
            let median_rms = median(differences.clone()).unwrap_or(f64::INFINITY);
            let max_rms = differences.into_iter().fold(f64::NEG_INFINITY, f64::max);
            let max_rms = if max_rms.is_finite() {
                max_rms
            } else {
                f64::INFINITY
            };
            let pass = median_rms <= median_limit && max_limit.is_none_or(|limit| max_rms <= limit);
            let status = if pass { "PASS" } else { "FAIL" };
            writeln!(
                out,
                "  CM-1 {band_name} parity: median_rms={median_rms:.2}dB max_rms={max_rms:.2}dB  {status}"
            )
            .unwrap();
            results.push(TestResult {
                label: format!("{name} CM-1 {band_name} parity"),
                pre_score: 0.0,
                scorecard: placeholder_scorecard(max_rms),
                pass,
                reason: format!(
                    "median_rms={median_rms:.2}dB (limit={median_limit:.2}dB), max_rms={max_rms:.2}dB{}",
                    max_limit.map_or_else(String::new, |limit| format!(" (limit={limit:.2}dB)"))
                ),
            });
        }
    } else {
        let channel_names = redirected_main_channels(&mode_results[0].1);
        let mut cm1_max_diff = 0.0_f64;
        for ch_name in &channel_names {
            let curves: Vec<Curve> = mode_results
                .iter()
                .filter_map(|(_, result)| deployed_final_curve(result, ch_name))
                .collect();
            if curves.len() >= 2 {
                let curve_refs: Vec<&Curve> = curves.iter().collect();
                let diff = max_curve_difference_db(&curve_refs, 20.0, 500.0);
                cm1_max_diff = cm1_max_diff.max(diff);
            }
        }
        let pass = cm1_max_diff <= CROSS_MODE_FR_MAX_DIFF_DB;
        let status = if pass { "PASS" } else { "FAIL" };
        writeln!(
            out,
            "  CM-1 FR convergence: max_diff={cm1_max_diff:.2}dB (limit={CROSS_MODE_FR_MAX_DIFF_DB:.1}dB)  {status}"
        )
        .unwrap();
        results.push(TestResult {
            label: format!("{name} CM-1 FR convergence"),
            pre_score: 0.0,
            scorecard: placeholder_scorecard(cm1_max_diff),
            pass,
            reason: format!(
                "max_diff={cm1_max_diff:.2}dB (limit={CROSS_MODE_FR_MAX_DIFF_DB:.1}dB)"
            ),
        });
    }

    // CM-2: strict home-cinema cases keep group-delay dispersion bounded for
    // every mode. Hybrid is a frequency-band split and does not promise lower
    // group-delay dispersion than IIR; MixedPhase is validated independently.
    if strict {
        let channel_names = redirected_main_channels(&mode_results[0].1);
        let mut by_mode = vec![Vec::new(); mode_results.len()];
        for channel in &channel_names {
            for (mode_index, (_, result)) in mode_results.iter().enumerate() {
                if let Some(curve) = deployed_final_curve(result, channel)
                    && let Some(gd_std) = group_delay_std_dev(&curve, 100.0, 1_000.0)
                {
                    by_mode[mode_index].push(gd_std);
                }
            }
        }
        let medians: Vec<f64> = by_mode
            .into_iter()
            .map(|values| median(values).unwrap_or(f64::INFINITY))
            .collect();
        let pass = medians
            .iter()
            .all(|value| value.is_finite() && *value <= CROSS_MODE_TIMING_MAX_STD_MS);
        let detail = mode_results
            .iter()
            .zip(&medians)
            .map(|((mode_name, _), value)| format!("{mode_name}={value:.2}ms"))
            .collect::<Vec<_>>()
            .join(" ");
        let status = if pass { "PASS" } else { "FAIL" };
        writeln!(
            out,
            "  CM-2 timing sanity: {detail} limit={CROSS_MODE_TIMING_MAX_STD_MS:.2}ms  {status}"
        )
        .unwrap();
        results.push(TestResult {
            label: format!("{name} CM-2 timing sanity"),
            pre_score: 0.0,
            scorecard: placeholder_scorecard(
                medians.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ),
            pass,
            reason: format!(
                "{detail}; every mode must remain <= {CROSS_MODE_TIMING_MAX_STD_MS:.2}ms"
            ),
        });
    } else {
        let channel_names: Vec<String> =
            mode_results[0].1.channel_results.keys().cloned().collect();
        let mut iir_gd_max = 0.0_f64;
        let mut fir_gd_max = 0.0_f64;
        let mut mixed_gd_max = 0.0_f64;
        let mut has_phase = false;
        for ch_name in &channel_names {
            for (mode_name, result) in &mode_results {
                if let Some(ch) = result.channel_results.get(ch_name)
                    && let Some(gd_std) = group_delay_std_dev(&ch.final_curve, 20.0, 500.0)
                {
                    has_phase = true;
                    match *mode_name {
                        "IIR" => iir_gd_max = iir_gd_max.max(gd_std),
                        "FIR" => fir_gd_max = fir_gd_max.max(gd_std),
                        "Mixed" => mixed_gd_max = mixed_gd_max.max(gd_std),
                        _ => {}
                    }
                }
            }
        }
        if has_phase {
            let max_gd = iir_gd_max.max(fir_gd_max).max(mixed_gd_max);
            let pass = max_gd < 50.0;
            let status = if pass { "PASS" } else { "FAIL" };
            writeln!(
                out,
                "  CM-2 GD flatness: IIR={iir_gd_max:.2}ms FIR={fir_gd_max:.2}ms Mixed={mixed_gd_max:.2}ms  {status}"
            )
            .unwrap();
            results.push(TestResult {
                label: format!("{name} CM-2 GD flatness"),
                pre_score: 0.0,
                scorecard: placeholder_scorecard(fir_gd_max.max(mixed_gd_max)),
                pass,
                reason: format!(
                    "IIR={iir_gd_max:.2}ms FIR={fir_gd_max:.2}ms Mixed={mixed_gd_max:.2}ms"
                ),
            });
        } else {
            writeln!(out, "  CM-2 GD flatness: SKIP (no phase data)").unwrap();
        }
    }

    // CM-3: Score convergence (ratio of max/min post scores)
    {
        let scores: Vec<f64> = mode_results
            .iter()
            .map(|(_, r)| r.combined_post_score)
            .collect();
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ratio = if min_score > 0.0 {
            max_score / min_score
        } else {
            f64::INFINITY
        };
        let cm3_pass = ratio <= CROSS_MODE_SCORE_RATIO_LIMIT;
        let status = if cm3_pass { "PASS" } else { "FAIL" };

        let mode_scores: String = mode_results
            .iter()
            .map(|(name, r)| format!("{}={:.4}", name, r.combined_post_score))
            .collect::<Vec<_>>()
            .join(" ");

        writeln!(
            out,
            "  CM-3 Score convergence: {} ratio={:.2}x (limit={:.1}x)  {}",
            mode_scores, ratio, CROSS_MODE_SCORE_RATIO_LIMIT, status
        )
        .unwrap();

        results.push(TestResult {
            label: format!("{} CM-3 score convergence", name),
            pre_score: 0.0,
            scorecard: placeholder_scorecard(ratio),
            pass: cm3_pass,
            reason: format!(
                "{} ratio={:.2}x (limit={:.1}x)",
                mode_scores, ratio, CROSS_MODE_SCORE_RATIO_LIMIT
            ),
        });
    }

    Ok((out, results))
}

pub(super) fn run_option_effect_test(
    name: &str,
    fem_dir: &Path,
    fem_subdir: &str,
    optim_dir: &Path,
    optim_subdir: &str,
    options: &[OptionOverride],
    maxeval: usize,
    seed_runs: usize,
) -> Result<(String, Vec<TestResult>)> {
    let mut out = String::new();
    let mut results = Vec::new();

    let options_str: String = options
        .iter()
        .map(|o| o.to_string())
        .collect::<Vec<_>>()
        .join(" + ");
    writeln!(out, "\n--- {} ({}) ---", name, options_str).unwrap();

    let base_config_path = fem_dir.join(format!("{}/config.json", fem_subdir));
    let override_path = optim_dir.join(format!("{}/optimiser-iir.json", optim_subdir));
    let override_path = if override_path.exists() {
        Some(override_path)
    } else {
        None
    };

    let needs_multi_measurement = options.iter().any(option_needs_multi_measurement_paths);
    let needs_gd_trusted_measurements = options.iter().any(option_needs_gd_trusted_measurements);
    let needs_multisub_multi_seat = options.iter().any(option_needs_multisub_multi_seat_paths);
    let gd_profile = options.iter().find_map(option_gd_profile);
    let isolate_group_delay = options.iter().any(option_is_group_delay);

    // BroadbandTargetMatching needs a target tilt to have something to match.
    // When the combo doesn't include an explicit TargetTilt, both baseline and
    // option get a default -0.8 dB/oct tilt so the only variable is broadband.
    let has_broadband = options
        .iter()
        .any(|o| matches!(o, OptionOverride::BroadbandTargetMatching));
    let has_tilt = options
        .iter()
        .any(|o| matches!(o, OptionOverride::TargetTilt { .. }));
    let default_target_response = if has_broadband && !has_tilt {
        Some(TargetResponseConfig {
            shape: TargetShape::Custom,
            slope_db_per_octave: -0.8,
            ..TargetResponseConfig::default()
        })
    } else {
        None
    };

    // Load and run baseline (all options disabled)
    let (mut baseline_config, _, _validation) =
        load_config(&base_config_path, override_path.as_deref())?;
    apply_qa_overrides(
        &mut baseline_config,
        &format!("{name}:option-baseline"),
        maxeval,
    );
    for option in options {
        disable_option(&mut baseline_config, option);
    }
    isolate_schroeder_split_from_multi_measurement(&mut baseline_config, options);
    if let Some(ref tr) = default_target_response {
        baseline_config.optimizer.target_response = Some(tr.clone());
    }
    if isolate_group_delay {
        apply_group_delay_qa_passthrough_eq(&mut baseline_config);
    }
    prepare_option_measurement_paths(
        &mut baseline_config,
        fem_dir,
        fem_subdir,
        needs_multi_measurement,
        needs_gd_trusted_measurements,
        needs_multisub_multi_seat,
        gd_profile,
    )?;

    let baseline_result = run_optimization(&baseline_config, seed_runs)
        .with_context(|| format!("{} baseline", name))?;

    writeln!(
        out,
        "  baseline: post={:.4} (pre={:.4})",
        baseline_result.combined_post_score, baseline_result.combined_pre_score
    )
    .unwrap();

    // Load and run with all options enabled
    let (mut option_config, _, _validation) =
        load_config(&base_config_path, override_path.as_deref())?;
    apply_qa_overrides(
        &mut option_config,
        &format!("{name}:option-enabled"),
        maxeval,
    );
    for option in options {
        apply_option_override(&mut option_config, option);
    }
    isolate_schroeder_split_from_multi_measurement(&mut option_config, options);
    if let Some(ref tr) = default_target_response {
        option_config.optimizer.target_response = Some(tr.clone());
    }
    if isolate_group_delay {
        apply_group_delay_qa_passthrough_eq(&mut option_config);
    }
    prepare_option_measurement_paths(
        &mut option_config,
        fem_dir,
        fem_subdir,
        needs_multi_measurement,
        needs_gd_trusted_measurements,
        needs_multisub_multi_seat,
        gd_profile,
    )?;

    let option_result = run_optimization(&option_config, seed_runs)
        .with_context(|| format!("{} with-options", name))?;

    writeln!(
        out,
        "  with-options: post={:.4} (pre={:.4})",
        option_result.combined_post_score, option_result.combined_pre_score
    )
    .unwrap();

    // Validate each per-option invariant individually
    let mut all_pass = true;
    for option in options {
        let (pass, reason) = validate_option_effect(
            option,
            &baseline_config,
            &baseline_result,
            &option_config,
            &option_result,
            options,
        );

        let status = if pass { "PASS" } else { "FAIL" };
        writeln!(out, "  {}: {}  ({})", option, status, reason).unwrap();

        if !pass {
            all_pass = false;
            results.push(TestResult {
                label: format!("{} [{}]", name, option),
                pre_score: option_result.combined_pre_score,
                scorecard: compute_scorecard(&option_result),
                pass: false,
                reason,
            });
        }
    }

    // Combo-level scorecard check: compare option result against baseline
    // using the multi-metric scorecard. Combos with multiple options face
    // conflicting constraints (e.g., schroeder split + asymmetric loss) that
    // shrink the feasible region. Allow a small convergence margin that scales
    // with the number of options.
    let option_scorecard = compute_scorecard(&option_result);
    let baseline_scorecard = compute_scorecard(&baseline_result);

    let convergence_margin = match options.len() {
        0..=1 => option_result.combined_pre_score * 0.01, // 1% — optimizer budget is tight, allow noise
        2..=3 => option_result.combined_pre_score * 0.05, // 5% for 2-3 options
        _ => option_result.combined_pre_score * 0.15,     // 15% for 4+ options
    };
    // Target-reshaping options (tilt, broadband matching) deliberately move the
    // response away from flat, so the flat-loss convergence gate is not a valid
    // acceptance criterion for combos containing them; the per-option validators
    // above (tilt slope error, broadband shelves, double-tilt check) are the
    // authoritative gates in that case.
    let target_reshaped = options.iter().any(OptionOverride::reshapes_target);
    let converged = option_result.combined_post_score
        < option_result.combined_pre_score
            + convergence_margin
            // Degenerate flat-in/flat-out fixtures (e.g. group-delay isolation
            // with 0 dB EQ bounds) score exactly 0 == 0: count equality as
            // non-regression instead of failing the strict comparison.
            + convergence_epsilon(option_result.combined_pre_score);

    // Run scorecard comparison (informational for option tests — per-option
    // validators remain the primary gates, but EPA/peak/GD violations are surfaced)
    let scorecard_checks = compare_scorecards(&baseline_scorecard, &option_scorecard);
    let scorecard_failures: Vec<String> = scorecard_checks
        .iter()
        .filter(|(_, pass, _)| !pass)
        .map(|(name, _, detail)| format!("{}: {}", name, detail))
        .collect();

    if !converged && target_reshaped {
        writeln!(
            out,
            "  convergence: SKIP  (flat-loss gate n/a: option set reshapes the target; post {:.6} vs pre {:.6} informational)",
            option_result.combined_post_score, option_result.combined_pre_score
        )
        .unwrap();
    } else if !converged {
        all_pass = false;
        let reason = format!(
            "no convergence: post {:.6} >= pre {:.6} (+{:.6} margin)",
            option_result.combined_post_score,
            option_result.combined_pre_score,
            convergence_margin + convergence_epsilon(option_result.combined_pre_score)
        );
        writeln!(out, "  convergence: FAIL  ({})", reason).unwrap();
        results.push(TestResult {
            label: format!("{} [convergence]", name),
            pre_score: option_result.combined_pre_score,
            scorecard: option_scorecard.clone(),
            pass: false,
            reason,
        });
    }

    // Scorecard failures are blocking quality failures, not warnings.
    if !scorecard_failures.is_empty() {
        all_pass = false;
        let reason = scorecard_failures.join("; ");
        writeln!(out, "  scorecard: FAIL [{}]", reason).unwrap();
        results.push(TestResult {
            label: format!("{} [scorecard]", name),
            pre_score: option_result.combined_pre_score,
            scorecard: option_scorecard.clone(),
            pass: false,
            reason,
        });
    }
    // Registry improvement is correction quality (the option run's own
    // uncorrected input -> corrected output). The per-option validators above
    // independently compare the requested effect against the default baseline.
    // Requiring every tuning value to outperform the default tuning would make
    // parameter sweeps fail even when they are safe, effective, and distinct.
    // If everything passed, push a single PASS result.
    if all_pass {
        results.push(TestResult {
            label: name.to_string(),
            pre_score: option_result.combined_pre_score,
            scorecard: option_scorecard,
            pass: true,
            reason: format!(
                "all {} invariants pass [{}]",
                options.len(),
                compute_scorecard(&option_result)
            ),
        });
    }

    Ok((out, results))
}
