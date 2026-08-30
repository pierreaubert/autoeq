//! RoomEQ Synthetic QA: Tests optimization against synthetic speaker scenarios.
//!
//! Uses deterministic synthetic curves with known room modes and noise to validate
//! that optimization consistently improves the response across all processing modes,
//! targets, and option combinations.
//!
//! Usage:
//!   cargo run --bin roomeq-qa-synthetic --no-default-features --release
//!   cargo run --bin roomeq-qa-synthetic --no-default-features --release -- --list
//!   cargo run --bin roomeq-qa-synthetic --no-default-features --release -- --difficulty easy
//!   cargo run --bin roomeq-qa-synthetic --no-default-features --release -- --multiseat-guards-only

use anyhow::Result;
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use roomeq_model::{Curve, ProcessingMode};
use roomeq_synthetic::{
    generate_flat_curve, generate_harman_tilt_curve, generate_multisub_scenario, generate_scenario,
    generate_speaker_rolloff_curve,
};
use std::fmt::Write as _;
use std::time::Instant;

mod build;
mod channel_layout;
mod consts;
mod generate;
mod misc;
mod option;
mod run;
mod types;

use channel_layout::sub_topos_for_layout;
use consts::ALL_DIFFICULTIES;
use consts::ALL_LAYOUTS;
use consts::ALL_MS_DIFFICULTIES;
use consts::ALL_SUB_TOPOS;
use consts::KAUTZ_REFERENCE_MODES;
use consts::MS_OPTIONS;
use consts::MS_TOPOLOGIES;
use consts::OPTIONS;
use consts::SAMPLE_RATE;
use consts::SEED;
use generate::generate_ms_option_combos;
use generate::generate_option_combos;
use misc::fmt_epa;
use run::report_multiseat_api_guard_tests;
use run::run_multichannel_test;
use run::run_multiseat_api_guard_tests;
use run::run_multisub_test;
use run::run_single_test;
use types::DifficultyLevel;
use types::MultiSubDifficulty;
use types::QaOutcome;
use types::TestResult;

fn multichannel_mode_supported(
    layout: &channel_layout::ChannelLayout,
    mode: &ProcessingMode,
) -> bool {
    // Kautz correction is validated for individual speakers and no-LFE
    // multichannel systems. Independently corrected mains and subs cannot yet
    // be safely recombined by the routed-bass workflow.
    !(layout.has_lfe && *mode == ProcessingMode::KautzModal)
}

/// Run the synthetic QA command and report whether the binary should exit
/// unsuccessfully because one or more scenarios failed.
pub fn run() -> Result<bool> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let registry = crate::registry::load_registry()?;
    let suite = registry
        .suite_for_runner("synthetic")
        .ok_or_else(|| anyhow::anyhow!("RoomEQ QA registry has no synthetic suite"))?;
    for required_claim in ["pairwise_options", "topology_mode_cartesian", "multi_seed"] {
        anyhow::ensure!(
            suite.claims.iter().any(|claim| claim == required_claim),
            "synthetic registry suite is missing claim '{required_claim}'"
        );
    }
    let registered_options = suite.cases.iter().map(String::as_str).collect::<Vec<_>>();
    let implemented_options = OPTIONS.iter().map(|option| option.name).collect::<Vec<_>>();
    anyhow::ensure!(
        registered_options == implemented_options,
        "synthetic option axes drifted from the RoomEQ QA registry: registry={registered_options:?}, implemented={implemented_options:?}"
    );

    let args: Vec<String> = std::env::args().collect();
    let help = args.iter().any(|a| a == "--help" || a == "-h");
    let list_only = args.iter().any(|a| a == "--list");
    let multiseat_guards_only = args.iter().any(|a| a == "--multiseat-guards-only");
    let full_matrix = args.iter().any(|a| a == "--full-matrix");
    let pr_matrix = args.iter().any(|a| a == "--pr");
    let fail_fast = args.iter().any(|a| a == "--fail-fast");
    let difficulty_filter = args
        .windows(2)
        .find(|w| w[0] == "--difficulty")
        .map(|w| w[1].clone());
    let mode_filter = args
        .windows(2)
        .find(|w| w[0] == "--mode")
        .map(|w| w[1].clone());
    let layout_filter = args
        .windows(2)
        .find(|w| w[0] == "--layout")
        .map(|w| w[1].clone());
    let sub_topology_filter = args
        .windows(2)
        .find(|w| w[0] == "--sub-topology")
        .map(|w| w[1].clone());

    if help {
        println!("RoomEQ Synthetic QA");
        println!();
        println!("Usage:");
        println!(
            "  roomeq-qa-synthetic [--list] [--pr] [--difficulty NAME] [--mode NAME] [--layout NAME] [--sub-topology NAME] [--full-matrix] [--multiseat-guards-only]"
        );
        println!();
        println!("Options:");
        println!("  --list                   Print the synthetic QA matrix and exit");
        println!("  --difficulty NAME        Run only one difficulty: easy, medium, hard");
        println!(
            "  --mode NAME              Run one mode: LowLatency, PhaseLinear, Hybrid, MixedPhase, WarpedIir, KautzModal"
        );
        println!("  --layout NAME            Run one multichannel layout, for example 7.1.4");
        println!("  --sub-topology NAME      Run one sub topology, for example mso_8sub");
        println!("  --multiseat-guards-only  Run only multi-seat API guard tests");
        println!(
            "  --full-matrix            Include WarpedIir/KautzModal and every multichannel processing mode"
        );
        println!("  --pr                     Run the bounded pull-request audibility matrix");
        println!("  --help, -h               Print this help");
        return Ok(false);
    }

    if multiseat_guards_only {
        return report_multiseat_api_guard_tests();
    }

    let difficulties: Vec<&DifficultyLevel> = if pr_matrix {
        vec![&consts::EASY]
    } else if let Some(ref filter) = difficulty_filter {
        ALL_DIFFICULTIES
            .iter()
            .filter(|d| d.name == filter.as_str())
            .collect()
    } else {
        ALL_DIFFICULTIES.iter().collect()
    };

    let ms_difficulties: Vec<&MultiSubDifficulty> = if pr_matrix {
        vec![&consts::MS_EASY]
    } else if let Some(ref filter) = difficulty_filter {
        ALL_MS_DIFFICULTIES
            .iter()
            .filter(|d| d.name == filter.as_str())
            .collect()
    } else {
        ALL_MS_DIFFICULTIES.iter().collect()
    };

    let default_modes = [
        ProcessingMode::LowLatency,
        ProcessingMode::PhaseLinear,
        ProcessingMode::Hybrid,
        ProcessingMode::MixedPhase,
    ];
    let full_modes = [
        ProcessingMode::LowLatency,
        ProcessingMode::PhaseLinear,
        ProcessingMode::Hybrid,
        ProcessingMode::MixedPhase,
        ProcessingMode::WarpedIir,
        ProcessingMode::KautzModal,
    ];
    let pr_modes = [ProcessingMode::LowLatency, ProcessingMode::Hybrid];
    let selected_modes: &[ProcessingMode] = if pr_matrix {
        &pr_modes
    } else if full_matrix {
        &full_modes
    } else {
        &default_modes
    };
    let selected_multichannel_modes: &[ProcessingMode] = if full_matrix {
        &full_modes
    } else {
        &default_modes
    };
    let mode_matches = |mode: &ProcessingMode, filter: &str| {
        let filter = filter.to_ascii_lowercase().replace(['-', '_'], "");
        let name = format!("{mode:?}").to_ascii_lowercase();
        name == filter
    };
    let modes: Vec<ProcessingMode> = selected_modes
        .iter()
        .filter(|mode| {
            mode_filter
                .as_deref()
                .is_none_or(|filter| mode_matches(mode, filter))
        })
        .cloned()
        .collect();
    let multichannel_modes: Vec<ProcessingMode> = selected_multichannel_modes
        .iter()
        .filter(|mode| {
            mode_filter
                .as_deref()
                .is_none_or(|filter| mode_matches(mode, filter))
        })
        .cloned()
        .collect();
    if modes.is_empty() || multichannel_modes.is_empty() {
        anyhow::bail!(
            "mode filter '{}' does not select a mode in this matrix",
            mode_filter.as_deref().unwrap_or_default()
        );
    }

    let flat_target = generate_flat_curve(20.0, 20000.0, 200);
    let harman_target = generate_harman_tilt_curve(20.0, 20000.0, 200);
    let targets: Vec<(&str, &Curve)> = vec![("flat", &flat_target), ("harman", &harman_target)];

    // Speaker rolloff: 0 dB above 80 Hz, -12 dB/oct below (realistic 2nd-order highpass)
    let speaker_rolloff = generate_speaker_rolloff_curve(20.0, 20000.0, 200, 80.0, -12.0);

    let mut option_combos = generate_option_combos();
    let mut ms_option_combos = generate_ms_option_combos();
    if pr_matrix {
        option_combos.truncate(OPTIONS.len() + 1);
        ms_option_combos.truncate(1);
    }
    let layouts: Vec<_> = ALL_LAYOUTS
        .iter()
        .filter(|layout| {
            layout_filter
                .as_ref()
                .is_none_or(|filter| layout.name == filter)
        })
        .filter(|layout| !pr_matrix || matches!(layout.name, "2.0" | "2.1" | "5.1" | "7.1.4"))
        .collect();

    // Count total tests
    let single_total = difficulties.len() * modes.len() * targets.len() * option_combos.len();
    let ms_total = ms_difficulties.len() * MS_TOPOLOGIES.len() * ms_option_combos.len();
    let multiseat_guard_total = 4;
    let mc_total: usize = layouts
        .iter()
        .map(|layout| {
            let n_topos = sub_topos_for_layout(layout)
                .iter()
                .filter(|topology| {
                    !pr_matrix
                        || matches!(
                            topology.name,
                            "single_sub" | "mso_2sub" | "cardioid" | "dba"
                        )
                })
                .count();
            let topology_cases = if n_topos == 0 {
                difficulties.len() // no LFE → 1 test per difficulty
            } else {
                n_topos * difficulties.len()
            };
            let supported_modes = multichannel_modes
                .iter()
                .filter(|mode| multichannel_mode_supported(layout, mode))
                .count();
            topology_cases * supported_modes
        })
        .sum();
    let total = single_total + ms_total + multiseat_guard_total + mc_total;

    if list_only {
        println!("Synthetic QA Test Matrix:");
        println!();
        println!("  Single-speaker:");
        println!(
            "    Difficulties: {}",
            difficulties
                .iter()
                .map(|d| d.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Modes: {}",
            modes
                .iter()
                .map(|mode| format!("{mode:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("    Targets: flat, harman");
        println!(
            "    Option combos: {} (baseline + {} singles + {} pairs + 1 all)",
            option_combos.len(),
            OPTIONS.len(),
            OPTIONS.len() * (OPTIONS.len() - 1) / 2,
        );
        println!("    Subtotal: {}", single_total);
        println!();
        println!("  Multi-sub:");
        println!(
            "    Difficulties: {}",
            ms_difficulties
                .iter()
                .map(|d| d.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Topologies: {}",
            MS_TOPOLOGIES
                .iter()
                .map(|t| t.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Option combos: {} (baseline + {} singles)",
            ms_option_combos.len(),
            MS_OPTIONS.len(),
        );
        println!("    Subtotal: {}", ms_total);
        println!();
        println!("  Multi-seat API guards:");
        println!(
            "    Checks: missing phase rejection, Average metrics, PrimaryWithConstraints metrics, polarity/all-pass controls"
        );
        println!("    Subtotal: {}", multiseat_guard_total);
        println!();
        println!("  Multi-channel:");
        println!(
            "    Layouts: {}",
            layouts
                .iter()
                .map(|l| l.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Sub topologies (with LFE): {}",
            ALL_SUB_TOPOS
                .iter()
                .map(|t| t.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Difficulties: {}",
            difficulties
                .iter()
                .map(|d| d.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Modes: {}",
            multichannel_modes
                .iter()
                .map(|mode| format!("{mode:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        if multichannel_modes.contains(&ProcessingMode::KautzModal) {
            println!("    KautzModal: no-LFE layouts only (routed-bass integration unsupported)");
        }
        println!("    Subtotal: {}", mc_total);
        println!();
        println!("  Total tests: {}", total);
        return Ok(false);
    }

    println!(
        "RoomEQ Synthetic QA -- {} tests ({} single + {} multi-sub + {} multi-seat guards + {} multi-channel)",
        total, single_total, ms_total, multiseat_guard_total, mc_total
    );
    println!("============================================================");

    let start = Instant::now();
    let mut all_results = Vec::new();
    let mut passed = 0;
    let mut failed = 0;

    for difficulty in &difficulties {
        // Build room modes from difficulty config
        let modes_biquad: Vec<Biquad> = difficulty
            .modes
            .iter()
            .map(|&(freq, q, gain)| Biquad::new(BiquadFilterType::Peak, freq, SAMPLE_RATE, q, gain))
            .collect();
        let kautz_modes_biquad: Vec<Biquad> = difficulty
            .modes
            .iter()
            .copied()
            .chain(KAUTZ_REFERENCE_MODES.iter().copied())
            .map(|(freq, q, gain)| Biquad::new(BiquadFilterType::Peak, freq, SAMPLE_RATE, q, gain))
            .collect();

        for &(target_name, target) in &targets {
            // Combine target shape with speaker rolloff so that broadband/excursion
            // options see a realistic low-frequency limit in the measurement.
            let speaker_base = Curve {
                freq: target.freq.clone(),
                spl: &target.spl + &speaker_rolloff.spl,
                phase: None,
                ..Default::default()
            };
            let scenario = generate_scenario(
                &format!("{}/{}", difficulty.name, target_name),
                &speaker_base,
                &modes_biquad,
                difficulty.noise_rms * 0.3,
                difficulty.noise_rms * 0.7,
                SEED,
                SAMPLE_RATE,
            );
            let kautz_scenario = generate_scenario(
                &format!("{}/{}-kautz", difficulty.name, target_name),
                &speaker_base,
                &kautz_modes_biquad,
                difficulty.noise_rms * 0.3,
                difficulty.noise_rms * 0.7,
                SEED,
                SAMPLE_RATE,
            );

            for mode in &modes {
                let degraded = if *mode == ProcessingMode::KautzModal {
                    &kautz_scenario.degraded_curve
                } else {
                    &scenario.degraded_curve
                };
                let mut baseline_post_score = None;
                for combo in &option_combos {
                    let result = run_single_test(
                        degraded,
                        mode.clone(),
                        target_name,
                        combo,
                        difficulty,
                        baseline_post_score,
                    );
                    if combo.is_empty() {
                        baseline_post_score = Some(result.post_score);
                    }

                    if result.passed {
                        passed += 1;
                    } else {
                        failed += 1;
                        println!(
                            "  FAIL: {} -- {} (epa={})",
                            result.name,
                            result.reason,
                            fmt_epa(result.epa_preference)
                        );
                        if fail_fast {
                            return Ok(true);
                        }
                    }

                    all_results.push(result);
                }
            }
        }
    }

    // ====================================================================
    // Multi-sub tests
    // ====================================================================
    for ms_diff in &ms_difficulties {
        let shared_biquads: Vec<Biquad> = ms_diff
            .shared_modes
            .iter()
            .map(|&(f, q, g)| Biquad::new(BiquadFilterType::Peak, f, SAMPLE_RATE, q, g))
            .collect();

        let per_sub_biquads: Vec<Vec<Biquad>> = ms_diff
            .per_sub_modes
            .iter()
            .map(|modes| {
                modes
                    .iter()
                    .map(|&(f, q, g)| Biquad::new(BiquadFilterType::Peak, f, SAMPLE_RATE, q, g))
                    .collect()
            })
            .collect();

        let scenario = generate_multisub_scenario(
            &format!("multisub/{}", ms_diff.name),
            ms_diff.n_subs,
            &shared_biquads,
            &per_sub_biquads,
            ms_diff.delays_ms,
            ms_diff.noise_rms,
            SEED,
            SAMPLE_RATE,
        );

        for topo in MS_TOPOLOGIES {
            for combo in &ms_option_combos {
                let result = run_multisub_test(&scenario.sub_curves, topo, combo, ms_diff);

                if result.passed {
                    passed += 1;
                } else {
                    failed += 1;
                    println!(
                        "  FAIL: {} -- {} (epa={})",
                        result.name,
                        result.reason,
                        fmt_epa(result.epa_preference)
                    );
                    if fail_fast {
                        return Ok(true);
                    }
                }

                all_results.push(result);
            }
        }
    }

    // ====================================================================
    // Multi-seat public API guard tests
    // ====================================================================
    for result in run_multiseat_api_guard_tests() {
        if result.passed {
            passed += 1;
        } else {
            failed += 1;
            println!("  FAIL: {} -- {}", result.name, result.reason);
            if fail_fast {
                return Ok(true);
            }
        }
        all_results.push(result);
    }

    // ====================================================================
    // Multi-channel topology tests
    // ====================================================================
    let base_fullrange = generate_speaker_rolloff_curve(20.0, 20000.0, 200, 80.0, -6.0);

    for layout in layouts {
        let topos: Vec<_> = sub_topos_for_layout(layout)
            .iter()
            .filter(|topology| {
                sub_topology_filter
                    .as_ref()
                    .is_none_or(|filter| topology.name == filter)
            })
            .filter(|topology| {
                !pr_matrix
                    || matches!(
                        topology.name,
                        "single_sub" | "mso_2sub" | "cardioid" | "dba"
                    )
            })
            .collect();

        if topos.is_empty() {
            // No LFE — test mains only
            for difficulty in &difficulties {
                for mode in multichannel_modes
                    .iter()
                    .filter(|mode| multichannel_mode_supported(layout, mode))
                {
                    let result = run_multichannel_test(
                        layout,
                        None,
                        difficulty,
                        &base_fullrange,
                        mode.clone(),
                        SAMPLE_RATE,
                    );
                    if result.passed {
                        passed += 1;
                    } else {
                        failed += 1;
                        println!(
                            "  FAIL: {} -- {} (epa={})",
                            result.name,
                            result.reason,
                            fmt_epa(result.epa_preference)
                        );
                        if fail_fast {
                            return Ok(true);
                        }
                    }
                    all_results.push(result);
                }
            }
        } else {
            // With LFE — test each sub topology
            for sub_topo in topos {
                for difficulty in &difficulties {
                    for mode in multichannel_modes
                        .iter()
                        .filter(|mode| multichannel_mode_supported(layout, mode))
                    {
                        let result = run_multichannel_test(
                            layout,
                            Some(sub_topo),
                            difficulty,
                            &base_fullrange,
                            mode.clone(),
                            SAMPLE_RATE,
                        );
                        if result.passed {
                            passed += 1;
                        } else {
                            failed += 1;
                            println!(
                                "  FAIL: {} -- {} (epa={})",
                                result.name,
                                result.reason,
                                fmt_epa(result.epa_preference)
                            );
                            if fail_fast {
                                return Ok(true);
                            }
                        }
                        all_results.push(result);
                    }
                }
            }
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("============================================================");
    println!(
        "Results: {} passed, {} failed, {} total ({:.1}s)",
        passed,
        failed,
        all_results.len(),
        elapsed.as_secs_f64()
    );

    // Print summary table
    let mut summary = String::new();
    for difficulty in &difficulties {
        let diff_results: Vec<&TestResult> = all_results
            .iter()
            .filter(|r| r.name.starts_with(difficulty.name))
            .collect();
        let diff_pass = diff_results.iter().filter(|r| r.passed).count();
        let diff_total = diff_results.len();
        writeln!(
            &mut summary,
            "  {}: {}/{} passed ({:.1}%)",
            difficulty.name,
            diff_pass,
            diff_total,
            diff_pass as f64 / diff_total as f64 * 100.0
        )
        .ok();
    }
    // Multi-sub summary
    let ms_results: Vec<&TestResult> = all_results
        .iter()
        .filter(|r| r.name.starts_with("multisub/"))
        .collect();
    if !ms_results.is_empty() {
        let ms_pass = ms_results.iter().filter(|r| r.passed).count();
        let ms_total_count = ms_results.len();
        writeln!(
            &mut summary,
            "  multi-sub: {}/{} passed ({:.1}%)",
            ms_pass,
            ms_total_count,
            ms_pass as f64 / ms_total_count as f64 * 100.0
        )
        .ok();
    }

    // Multi-seat API guard summary
    let multiseat_results: Vec<&TestResult> = all_results
        .iter()
        .filter(|r| r.name.starts_with("multiseat/"))
        .collect();
    if !multiseat_results.is_empty() {
        let multiseat_pass = multiseat_results.iter().filter(|r| r.passed).count();
        let multiseat_total_count = multiseat_results.len();
        writeln!(
            &mut summary,
            "  multi-seat API guards: {}/{} passed ({:.1}%)",
            multiseat_pass,
            multiseat_total_count,
            multiseat_pass as f64 / multiseat_total_count as f64 * 100.0
        )
        .ok();
    }

    // Multi-channel summary
    let mc_results: Vec<&TestResult> = all_results
        .iter()
        .filter(|r| r.name.starts_with("multichannel/"))
        .collect();
    if !mc_results.is_empty() {
        let mc_pass = mc_results.iter().filter(|r| r.passed).count();
        let mc_total_count = mc_results.len();
        writeln!(
            &mut summary,
            "  multi-channel: {}/{} passed ({:.1}%)",
            mc_pass,
            mc_total_count,
            mc_pass as f64 / mc_total_count as f64 * 100.0
        )
        .ok();
    }

    println!("\nPer-difficulty summary:");
    print!("{}", summary);

    let passed_outcomes = all_results
        .iter()
        .filter(|result| result.outcome() == QaOutcome::Passed)
        .count();
    let reverted_outcomes = all_results
        .iter()
        .filter(|result| result.outcome() == QaOutcome::Reverted)
        .count();
    let failed_outcomes = all_results
        .iter()
        .filter(|result| result.outcome() == QaOutcome::Failed)
        .count();
    println!(
        "Outcome summary: PASS={passed_outcomes}, REVERTED={reverted_outcomes}, FAIL={failed_outcomes}"
    );

    if failed > 0 {
        println!("\nFailed tests:");
        for r in &all_results {
            if !r.passed {
                println!(
                    "  {} -- {} (epa={})",
                    r.name,
                    r.reason,
                    fmt_epa(r.epa_preference)
                );
            }
        }
    }

    Ok(failed > 0)
}
