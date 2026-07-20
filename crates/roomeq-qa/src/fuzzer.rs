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
            if let Err(e) = validate_roomeq_output(&output_json_path) {
                println!("  Test {} failed output validation: {}", i + 1, e);
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
