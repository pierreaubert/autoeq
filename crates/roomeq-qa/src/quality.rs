//! RoomEQ QA: Convergence, Monotonicity, Cross-Mode & Per-Option Tests
//!
//! Validates that roomeq optimization modes produce converging results,
//! that giving the optimizer more resources always improves or maintains loss,
//! that IIR/FIR/Mixed modes converge to similar frequency responses,
//! and that each optimizer option has its expected effect.
//!
//! Uses autoeq:cmaes with fixed seed for deterministic results.
//! Test cases run in parallel for maximum throughput.
//!
//! Usage:
//!   cargo run --bin roomeq-qa --release              # run all tests
//!   cargo run --bin roomeq-qa --release -- --list     # list available cases
//!   cargo run --bin roomeq-qa --release -- --case "Stereo 2.0"
//!   cargo run --bin roomeq-qa --release -- --case "Cross-Mode"
//!   cargo run --bin roomeq-qa --release -- --case "target_tilt"

use anyhow::{Result, anyhow};
use roomeq_model::ProcessingMode;
use std::sync::Arc;

mod apply;
mod consts;
mod count;
mod counting_semaphore;
mod enable;
mod group;
mod group_delay_qa_profile;
mod metric_scorecard;
mod misc;
mod mutation;
mod option;
mod option_override;
mod peak;
mod residual;
mod run;
mod test_case;
#[cfg(test)]
mod tests;
mod types;
mod validate;

use consts::FEM_DIR;
use consts::OPTIM_CONFIG_DIR;
use consts::SEED;
use counting_semaphore::CountingSemaphore;
use group_delay_qa_profile::all_test_cases;
use misc::default_parallel_jobs;
use misc::find_project_root;
use run::run_cross_mode_convergence_tests;
use run::run_generic_path_tests;
use run::run_option_effect_test;
use run::run_stereo_workflow_tests;
use run::run_workflow_override_smoke;
use test_case::{RegisteredTestCase, TestCase};
use types::QaOutcome;
use types::TestResult;

fn enforce_registry_expectations(
    id: &str,
    claims: &[String],
    expect: crate::registry::ScenarioExpect,
    results: &mut [TestResult],
) {
    for result in results {
        let original_outcome = result.outcome();
        let mut failures = Vec::new();
        let flat_loss = result.scorecard.flat_loss;
        if !result.pre_score.is_finite()
            || !flat_loss.is_finite()
            || !result.scorecard.max_boost_db.is_finite()
        {
            failures.push("non-finite registry metric".to_string());
        } else {
            if result.pre_score > 0.0 && flat_loss > expect.max_post_score {
                failures.push(format!(
                    "post score {flat_loss:.4} exceeds registry limit {:.4}",
                    expect.max_post_score
                ));
            }
            if result.pre_score > 0.0 && result.scorecard.max_boost_db > expect.max_boost_db {
                failures.push(format!(
                    "max boost {:.2} dB exceeds registry limit {:.2} dB",
                    result.scorecard.max_boost_db, expect.max_boost_db
                ));
            }
            if expect.improvement_min_pct > 0.0 {
                // Relationship-only rows (for example the cross-mode ratio)
                // deliberately use pre_score=0 and have no raw-response
                // improvement claim of their own.
                if result.pre_score > 0.0 {
                    let improvement_pct = (1.0 - flat_loss / result.pre_score) * 100.0;
                    if improvement_pct + 1e-9 < expect.improvement_min_pct {
                        failures.push(format!(
                            "improvement {improvement_pct:.3}% below registry minimum {:.3}%",
                            expect.improvement_min_pct
                        ));
                    }
                }
            }
        }
        if matches!(original_outcome, QaOutcome::Reverted) && !expect.allow_safe_revert {
            failures.push("safe revert is not allowed by registry".to_string());
        }
        if !failures.is_empty() {
            result.pass = false;
            result.reason = format!("{}; registry: {}", result.reason, failures.join("; "));
        }
        result.reason = format!(
            "[registry={id} claims={}] {}",
            claims.join(","),
            result.reason
        );
    }
}

/// Run the quality QA command and report whether failed cases should produce
/// a non-zero process exit.
pub fn run() -> Result<bool> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let registry = crate::registry::load_registry()?;
    let suite = registry
        .suite_for_runner("quality")
        .ok_or_else(|| anyhow::anyhow!("RoomEQ QA registry has no quality suite"))?;
    for required_claim in [
        "blocking_scorecard",
        "option_effects",
        "cross_mode",
        "workflow_overrides",
    ] {
        anyhow::ensure!(
            suite.claims.iter().any(|claim| claim == required_claim),
            "quality registry suite is missing claim '{required_claim}'"
        );
    }

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "RoomEQ QA: Convergence, Monotonicity & Invariants\n\n\
             Usage:\n\
               roomeq-qa-quality [--jobs N] [--list] [--case SUBSTRING]\n\n\
             Options:\n\
               --jobs N          Number of test cases to run concurrently\n\
               --list            List available test cases and exit\n\
               --case TEXT       Run cases whose name contains TEXT, case-insensitive\n\
               --help, -h        Print this help"
        );
        return Ok(false);
    }
    let list_mode = args.iter().any(|a| a == "--list");
    let case_filter: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--case")
        .map(|w| w[1].clone());
    let jobs: usize = args
        .windows(2)
        .find(|w| w[0] == "--jobs")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or_else(default_parallel_jobs)
        .max(1);

    let all_cases = all_test_cases();

    // --list: print available cases and exit
    if list_mode {
        println!("Available test cases:");
        for tc in &all_cases {
            println!("  {}", tc.name());
        }
        return Ok(false);
    }

    println!(
        "=== RoomEQ QA: Convergence, Monotonicity & Invariants (CMA-ES, seed={}, parallel) ===",
        SEED
    );

    let project_root = find_project_root()?;
    let fem_dir = project_root.join(FEM_DIR);
    let optim_dir = project_root.join(OPTIM_CONFIG_DIR);

    // Filter cases if --case is provided (substring match)
    let cases_to_run: Vec<RegisteredTestCase> = if let Some(ref filter) = case_filter {
        let filter_lower = filter.to_lowercase();
        let matched: Vec<_> = all_cases
            .into_iter()
            .filter(|tc| tc.name().to_lowercase().contains(&filter_lower))
            .collect();
        if matched.is_empty() {
            return Err(anyhow!(
                "No test case matches '{}'. Use --list to see available cases.",
                filter
            ));
        }
        println!("Running {} case(s) matching '{}'", matched.len(), filter);
        matched
    } else {
        all_cases
    };

    println!("Using {} parallel job(s) (override with --jobs N).", jobs);

    // Run all test cases with a bounded permit pool. The outer thread is
    // spawned immediately but `sem.acquire()` gates entry to the actual
    // optimization — so at most `jobs` cases are resident simultaneously.
    let semaphore = Arc::new(CountingSemaphore::new(jobs));
    let handles: Vec<_> = cases_to_run
        .into_iter()
        .map(|tc| {
            let fem_dir = fem_dir.clone();
            let optim_dir = optim_dir.clone();
            let sem = Arc::clone(&semaphore);
            std::thread::spawn(move || -> Result<(String, Vec<TestResult>)> {
                sem.acquire();
                let RegisteredTestCase {
                    id,
                    claims,
                    expect,
                    case,
                } = tc;
                let mut result = match case {
                    TestCase::Workflow {
                        name,
                        fem_subdir,
                        optim_subdir,
                    } => {
                        let base_path = fem_dir.join(format!("{}/config.json", fem_subdir));
                        let scenario_override_dir = optim_dir.join(optim_subdir);
                        let override_path = scenario_override_dir.join("optimiser-iir.json");
                        let (_, mut results) =
                            run_stereo_workflow_tests(&name, &base_path, Some(&override_path))?;

                        for (mode_name, expected_mode, file_name) in [
                            ("FIR", ProcessingMode::PhaseLinear, "optimiser-fir.json"),
                            ("Hybrid", ProcessingMode::Hybrid, "optimiser-mixed.json"),
                        ] {
                            let scenario_override = scenario_override_dir.join(file_name);
                            let mode_override = optim_dir.join("modes").join(file_name);
                            let override_path = if scenario_override.exists() {
                                scenario_override
                            } else {
                                mode_override
                            };
                            let (_, mode_results) = run_workflow_override_smoke(
                                &name,
                                mode_name,
                                expected_mode,
                                &base_path,
                                &override_path,
                            )?;
                            results.extend(mode_results);
                        }

                        let (_, generic_results) =
                            run_generic_path_tests(&name, &base_path, &scenario_override_dir)?;
                        results.extend(generic_results);
                        Ok((name.to_string(), results))
                    }
                    TestCase::Generic {
                        name,
                        fem_subdir,
                        optim_subdir,
                    } => {
                        let base_path = fem_dir.join(format!("{}/config.json", fem_subdir));
                        let override_dir = optim_dir.join(optim_subdir);
                        run_generic_path_tests(&name, &base_path, &override_dir)
                    }
                    TestCase::CrossModeConvergence {
                        name,
                        fem_subdir,
                        optim_subdir,
                    } => {
                        let base_path = fem_dir.join(format!("{}/config.json", fem_subdir));
                        let override_dir = optim_dir.join(optim_subdir);
                        run_cross_mode_convergence_tests(&name, &base_path, &override_dir)
                    }
                    TestCase::OptionEffect {
                        name,
                        fem_subdir,
                        optim_subdir,
                        options,
                    } => run_option_effect_test(
                        &name,
                        &fem_dir,
                        &fem_subdir,
                        &optim_dir,
                        &optim_subdir,
                        &options,
                    ),
                };
                if let Ok((_, results)) = &mut result {
                    enforce_registry_expectations(&id, &claims, expect, results);
                }
                sem.release();
                result
            })
        })
        .collect();

    // Collect results in order, printing output as each completes
    let mut all_results: Vec<TestResult> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    for handle in handles {
        match handle.join() {
            Ok(Ok((output, results))) => {
                print!("{}", output);
                all_results.extend(results);
            }
            Ok(Err(e)) => {
                errors.push(format!("{:#}", e));
            }
            Err(_) => {
                errors.push("Thread panicked".to_string());
            }
        }
    }

    // Print any thread errors
    for err in &errors {
        eprintln!("ERROR: {}", err);
    }

    // Summary
    let execution_failures = errors.len();
    let total = all_results.len() + execution_failures;
    let passed = all_results
        .iter()
        .filter(|result| result.outcome() == QaOutcome::Passed)
        .count();
    let reverted = all_results
        .iter()
        .filter(|result| result.outcome() == QaOutcome::Reverted)
        .count();
    let disallowed_reverted = all_results
        .iter()
        .filter(|result| result.outcome() == QaOutcome::Reverted && !result.pass)
        .count();
    let failed_results = all_results
        .iter()
        .filter(|result| result.outcome() == QaOutcome::Failed)
        .count();
    let failed = failed_results + execution_failures;

    println!(
        "\n=== Summary: PASS={passed}, REVERTED={reverted} (allowed={}, disallowed={disallowed_reverted}), FAIL={failed}, TOTAL={total} ===",
        reverted - disallowed_reverted
    );

    if failed > 0 || disallowed_reverted > 0 {
        println!("\nNon-passing tests:");
        for r in &all_results {
            if r.outcome() == QaOutcome::Failed || (r.outcome() == QaOutcome::Reverted && !r.pass) {
                println!(
                    "  - {} (pre={:.4}, {}): {}",
                    r.label, r.pre_score, r.scorecard, r.reason
                );
            }
        }
    }

    Ok(failed > 0 || disallowed_reverted > 0)
}
