//! Fuzzer for roomeq binary
//!
//! Generates stratified and random speaker configurations and checks the
//! roomeq CLI produces structurally valid DSP output. Required scenario
//! buckets keep feature coverage stable while extra tests continue random
//! exploration.

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::Ordering;

mod coverage_counters;
mod driver_type;
mod generate;
mod misc;
mod scenario_kind;
mod types;
mod validate;

use coverage_counters::CoverageCounters;
use generate::generate_plots_for_multi_drivers;
use generate::generate_random_config;
use misc::CURRENT_TEST_INDEX;
use scenario_kind::ScenarioKind;
use types::Args;
use validate::validate_config;
use validate::validate_roomeq_output;

fn read_fuzz_post_score(output_path: &std::path::Path) -> Result<f64, String> {
    let json = fs::read_to_string(output_path).map_err(|error| error.to_string())?;
    let output: serde_json::Value =
        serde_json::from_str(&json).map_err(|error| error.to_string())?;
    let post = output
        .pointer("/metadata/post_score")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| "output metadata.post_score missing or non-numeric".to_string())?;
    if !post.is_finite() {
        return Err("output metadata.post_score is not finite".to_string());
    }
    Ok(post)
}

fn validate_fuzz_seed_distribution(
    config: &roomeq_model::RoomConfig,
    test_dir: &std::path::Path,
    canonical_output_path: &std::path::Path,
    sample_rate: f64,
    verbose: bool,
) -> Result<(), String> {
    let base_seed = config.optimizer.seed.unwrap_or(42);
    let base_score = read_fuzz_post_score(canonical_output_path)?;
    let mut scores = vec![(base_seed, base_score, canonical_output_path.to_path_buf())];
    let sample_rate_arg = sample_rate.to_string();

    for offset in crate::QA_SEED_OFFSETS.into_iter().skip(1) {
        let seed = base_seed.wrapping_add(offset);
        let mut seeded_config = config.clone();
        seeded_config.optimizer.seed = Some(seed);
        let config_path = test_dir.join(format!("config-seed-{seed}.json"));
        fs::write(
            &config_path,
            serde_json::to_string_pretty(&seeded_config).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let output_path = test_dir.join(format!("output-seed-{seed}.json"));

        let mut command = Command::new("cargo");
        command.args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "roomeq",
            "--features",
            "cli",
            "--",
            "--config",
            config_path
                .to_str()
                .ok_or_else(|| "non-UTF-8 fuzz config path".to_string())?,
            "--output",
            output_path
                .to_str()
                .ok_or_else(|| "non-UTF-8 fuzz output path".to_string())?,
            "--sample-rate",
            sample_rate_arg.as_str(),
        ]);
        let status = if verbose {
            command.status().map_err(|error| error.to_string())?
        } else {
            command.env("RUST_LOG", "error");
            let output = command.output().map_err(|error| error.to_string())?;
            if !output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stdout.trim().is_empty() {
                    println!("  seed {seed} stdout:\n{}", stdout.trim_end());
                }
                if !stderr.trim().is_empty() {
                    eprintln!("  seed {seed} stderr:\n{}", stderr.trim_end());
                }
            }
            output.status
        };
        if !status.success() {
            return Err(format!(
                "seed {seed} failed with exit code {:?}",
                status.code()
            ));
        }
        scores.push((seed, read_fuzz_post_score(&output_path)?, output_path));
    }

    scores.sort_by(|left, right| left.1.total_cmp(&right.1));
    let selected = &scores[scores.len() / 2];
    let min_score = scores.first().map(|entry| entry.1).unwrap_or_default();
    let max_score = scores.last().map(|entry| entry.1).unwrap_or_default();
    let details = scores
        .iter()
        .map(|(seed, score, _)| format!("{seed}:{score:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "  Seed distribution: selected={}, min={:.6}, max={:.6}, spread={:.6}, scores=[{}]",
        selected.0,
        min_score,
        max_score,
        max_score - min_score,
        details
    );
    if selected.2 != canonical_output_path {
        fs::copy(&selected.2, canonical_output_path).map_err(|error| error.to_string())?;
    }
    validate_roomeq_output(canonical_output_path).map(|_| ())
}

/// Rendering boundary supplied by the feature-gated root launcher.
pub trait DriverPlotter {
    #[allow(clippy::too_many_arguments)]
    fn plot_drivers_results(
        &self,
        data: &autoeq_optim::loss::DriversLossData,
        gains: &[f64],
        crossover_freqs: &[f64],
        sample_rate: f64,
        output: &std::path::Path,
    ) -> anyhow::Result<()>;
}

/// Run the fuzzer and report whether failed scenarios or missing coverage
/// should produce a non-zero process exit.
pub fn run(plotter: &dyn DriverPlotter) -> Result<bool, Box<dyn Error>> {
    let registry = crate::registry::load_registry()?;
    let suite = registry
        .suite_for_runner("fuzzer")
        .ok_or_else(|| anyhow::anyhow!("RoomEQ QA registry has no fuzzer suite"))?;
    for required_claim in [
        "finite_scores",
        "strict_improvement",
        "corrective_filter",
        "multi_seed",
    ] {
        if !suite.claims.iter().any(|claim| claim == required_claim) {
            return Err(anyhow::anyhow!(
                "fuzzer registry suite is missing claim '{required_claim}'"
            )
            .into());
        }
    }
    let args = Args::parse();

    // Create output directory
    let output_dir = args.output_dir.unwrap_or_else(|| {
        let dir = PathBuf::from("fuzzer_output");
        if !dir.exists() {
            fs::create_dir_all(&dir).unwrap();
        }
        dir
    });

    println!("Starting fuzzer with {} tests...", args.num_tests);
    println!("Output directory: {}", output_dir.display());

    let mut successful_tests = 0;
    let mut failed_tests = 0;
    let mut coverage = CoverageCounters::default();

    // Use seed if provided
    let mut rng = if let Some(seed) = args.seed {
        ChaCha8Rng::seed_from_u64(seed)
    } else {
        ChaCha8Rng::from_rng(&mut rand::rng())
    };

    for i in 0..args.num_tests {
        CURRENT_TEST_INDEX.store(i, Ordering::SeqCst);
        println!("Running test {}/{}...", i + 1, args.num_tests);
        let scenario_kind = ScenarioKind::for_test(i, &mut rng, args.skip_kautz_modal);
        println!("  Scenario bucket: {}", scenario_kind.name());

        // Create a subdirectory for this test
        let test_dir = output_dir.join(format!("test_{}", i));
        if test_dir.exists() {
            fs::remove_dir_all(&test_dir)?;
        }
        fs::create_dir_all(&test_dir)?;

        // Generate random configuration and measurements
        let (config, _measurement_files, multi_driver_groups) =
            generate_random_config(&test_dir, i, &mut rng, args.max_speakers, scenario_kind)?;

        // Validate config
        if let Err(e) = validate_config(&config) {
            println!("  Invalid config generated: {}", e);
            failed_tests += 1;
            continue;
        }

        // Save config
        let config_path = test_dir.join("config.json");
        let config_json = serde_json::to_string_pretty(&config)?;
        fs::write(&config_path, config_json)?;
        coverage.record(scenario_kind, &config);

        // Run roomeq binary
        let output_json_path = test_dir.join("output.json");
        let sample_rate_arg = args.sample_rate.to_string();
        let mut command = Command::new("cargo");
        command.args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "roomeq",
            "--features",
            "cli",
            "--",
            "--config",
            config_path.to_str().unwrap(),
            "--output",
            output_json_path.to_str().unwrap(),
            "--sample-rate",
            sample_rate_arg.as_str(),
        ]);

        let status = if args.verbose {
            command.status()?
        } else {
            command.env("RUST_LOG", "error");
            let output = command.output()?;
            if !output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stdout.trim().is_empty() {
                    println!("  stdout:\n{}", stdout.trim_end());
                }
                if !stderr.trim().is_empty() {
                    eprintln!("  stderr:\n{}", stderr.trim_end());
                }
            }
            output.status
        };

        if status.success() {
            if let Err(error) = validate_fuzz_seed_distribution(
                &config,
                &test_dir,
                &output_json_path,
                args.sample_rate,
                args.verbose,
            ) {
                println!(
                    "  Test {} failed optimizer-seed distribution: {}",
                    i + 1,
                    error
                );
                failed_tests += 1;
                continue;
            }

            println!("  Test {} successful!", i + 1);
            successful_tests += 1;

            // Generate plots for multi-driver groups
            if !multi_driver_groups.is_empty()
                && let Err(e) = generate_plots_for_multi_drivers(
                    plotter,
                    &output_json_path,
                    &multi_driver_groups,
                    &test_dir,
                    i,
                    args.sample_rate,
                    args.verbose,
                )
            {
                println!("  Warning: failed to generate plots: {}", e);
            }
        } else {
            println!(
                "  Test {} failed with exit code: {:?}",
                i + 1,
                status.code()
            );
            failed_tests += 1;
        }
    }

    println!("\nFuzzing complete!");
    println!("Successful tests: {}", successful_tests);
    println!("Failed tests: {}", failed_tests);
    coverage.print(args.skip_kautz_modal);

    let missing_required = coverage.missing_required(args.num_tests, args.skip_kautz_modal);
    let missing_required_coverage = !missing_required.is_empty();
    if missing_required_coverage {
        println!("\nMissing required coverage buckets:");
        for name in missing_required {
            println!("  {}", name);
        }
    }

    Ok(missing_required_coverage || failed_tests > 0)
}
