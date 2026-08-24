//! RoomEQ Full QA: Comprehensive Scenario Testing
//!
//! Tests the general FEM corpus across processing modes and a correctness-gated
//! FEM/Sonium fast-hybrid home-cinema feature matrix.
//! Validates both the library and CLI binary output.
//!
//! Checks performed per test case:
//! 1. Minimum improvement threshold (varies by room size)
//! 2. Per-channel regression (no individual channel may get worse)
//! 3. Output sanity (filters exist, frequencies/gains valid)
//! 4. Absolute score ceiling (post_score below maximum for room size)
//!
//! Usage:
//!   cargo run --bin roomeq-qa-full --release              # run all tests
//!   cargo run --bin roomeq-qa-full --release -- --quick    # fast subset
//!   cargo run --bin roomeq-qa-full --release -- --list     # list scenarios
//!   cargo run --bin roomeq-qa-full --release -- --matrix    # show test matrix
//!   cargo run --bin roomeq-qa-full --release -- --junit     # JUnit XML output

use anyhow::Result;
use clap::Parser;

mod args;
mod consts;
mod home_cinema;
mod is;
mod misc;
mod processing_method;
mod room_size;
mod run;
mod solver;
mod test_case;
mod test_result;

use args::Args;
use home_cinema::{build_home_cinema_matrix, build_quick_home_cinema_matrix};
use misc::find_project_root;
use misc::scenario_description;
pub use processing_method::ProcessingMethod;
use processing_method::{build_quick_test_matrix, build_test_matrix_for_tier};
use run::run_parallel;
pub use run::run_regression_case;
use test_case::TestCase;
use test_case::print_matrix;
use test_result::{QaOutcome, write_junit_xml};

/// Default optimizer evaluation budget used by the coverage matrix.
pub const DEFAULT_MAXEVAL: usize = consts::QA_MAXEVAL;

/// Run the coverage command and report whether failed cases should produce a
/// non-zero process exit. Process termination remains the binary adapter's job.
pub fn run() -> Result<bool> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let args = Args::parse();
    let project_root = find_project_root()?;
    std::env::set_current_dir(&project_root)?;

    // List scenarios
    if args.list {
        let registry = crate::registry::load_registry()?;
        println!("Available scenarios:");
        for family in registry.families_for(args.tier) {
            println!(
                "  {}: {} [{}]",
                family.scenario,
                scenario_description(&family.scenario),
                family.id
            );
        }
        return Ok(false);
    }

    // Build test matrix
    let mut test_cases = if args.home_cinema {
        build_home_cinema_matrix(args.tier, args.solver.as_deref(), args.mode.as_deref())
    } else if args.quick {
        build_quick_test_matrix(args.tier, args.solver.as_deref(), args.mode.as_deref())
    } else {
        build_test_matrix_for_tier(
            args.tier,
            false,
            args.solver.as_deref(),
            args.mode.as_deref(),
        )
    };
    if args.quick && !args.home_cinema {
        test_cases.extend(build_quick_home_cinema_matrix(
            args.tier,
            args.solver.as_deref(),
            args.mode.as_deref(),
        ));
    } else if !args.home_cinema {
        test_cases.extend(build_home_cinema_matrix(
            args.tier,
            args.solver.as_deref(),
            args.mode.as_deref(),
        ));
    }

    // Show matrix
    if args.matrix {
        print_matrix(&test_cases);
        return Ok(false);
    }

    // Apply scenario filter
    let test_cases: Vec<TestCase> = if let Some(ref filter) = args.scenario {
        let filter_lower = filter.to_lowercase();
        test_cases
            .into_iter()
            .filter(|tc| tc.scenario.to_lowercase().contains(&filter_lower))
            .collect()
    } else {
        test_cases
    };
    let test_cases: Vec<TestCase> = if let Some(ref filter) = args.case_name {
        let filter_lower = filter.to_lowercase();
        test_cases
            .into_iter()
            .filter(|test_case| test_case.name().to_lowercase().contains(&filter_lower))
            .collect()
    } else {
        test_cases
    };

    if test_cases.is_empty() {
        println!("No test cases to run.");
        return Ok(false);
    }

    println!("=== RoomEQ Full QA ===");
    println!(
        "Running {} test cases with {} parallel jobs",
        test_cases.len(),
        args.jobs()
    );
    if args.quick {
        println!(
            "QUICK MODE: bounded FEM/IIR plus routed home-cinema smoke matrix with safety-level acceptance"
        );
    }
    println!();

    // Run tests
    let results = run_parallel(test_cases, args.maxeval(), args.jobs());

    // Print results
    let mut passed = 0;
    let mut reverted = 0;
    let mut failed = 0;

    for result in &results {
        let status = result.outcome.label();
        let epa_str = match result.epa_preference {
            Some(v) => format!("epa={:.3}", v),
            None => "epa=n/a".to_string(),
        };
        if result.passed {
            match result.outcome {
                QaOutcome::Passed => passed += 1,
                QaOutcome::Reverted => reverted += 1,
                QaOutcome::Failed => unreachable!("failed outcome cannot have passed=true"),
            }
            println!(
                "[{}] {} ({}ms): {:.4} -> {:.4} ({:.1}% improvement) {}",
                status,
                result.name,
                result.duration_ms,
                result.pre_score,
                result.post_score,
                (1.0 - result.post_score / result.pre_score.max(0.001)) * 100.0,
                epa_str,
            );
        } else {
            failed += 1;
            if let Some(ref err) = result.error {
                eprintln!("[{}] {} {}: {}", status, result.name, epa_str, err);
            } else {
                eprintln!(
                    "[{}] {}: pre={:.4}, post={:.4} {}",
                    status, result.name, result.pre_score, result.post_score, epa_str,
                );
            }
        }
    }

    // Summary
    println!(
        "\n=== Summary: {} PASS, {} REVERTED, {} FAIL ({} total) ===",
        passed,
        reverted,
        failed,
        passed + reverted + failed
    );
    println!(
        "Total time: {}ms",
        results.iter().map(|r| r.duration_ms).sum::<u64>()
    );

    // JUnit output
    if let Some(ref junit_path) = args.junit {
        write_junit_xml(&results, junit_path)?;
        println!("JUnit XML written to: {}", junit_path.display());
    }

    // Exit code
    Ok(args.fail && failed > 0)
}
