//! AutoEQ - A library for audio equalization and filter optimization
//!
//! Copyright (C) 2025-2026 Pierre Aubert pierre(at)spinorama(dot)org
//!
//! This program is free software: you can redistribute and/or modify
//! it under the terms of the GNU General Public License as published by
//! the Free Software Foundation, either version 3 of the License, or
//! (at your option) any later version.
//!
//! This program is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//! GNU General Public License for more details.
//!
//! You should have received a copy of the GNU General Public License
//! along with this program.  If not, see <https://www.gnu.org/licenses/>.

use anyhow::{Context, Result, anyhow};
use autoeq_plot as plot;
use clap::Parser;
use log::warn;
use log::{error, info};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// Include split modules
#[path = "autoeq/load.rs"]
mod load;
#[path = "autoeq/postscore.rs"]
mod postscore;
#[path = "autoeq/prescore.rs"]
mod prescore;
#[path = "autoeq/progress.rs"]
mod progress;
#[path = "autoeq/qa.rs"]
mod qa;
#[path = "autoeq/runopt.rs"]
mod runopt;
#[path = "autoeq/save.rs"]
mod save;
#[path = "autoeq/spacing.rs"]
mod spacing;

#[cfg(test)]
#[path = "autoeq/load_tests.rs"]
mod load_tests;
#[cfg(test)]
#[path = "autoeq/postscore_tests.rs"]
mod postscore_tests;
#[cfg(test)]
#[path = "autoeq/prescore_tests.rs"]
mod prescore_tests;
#[cfg(test)]
#[path = "autoeq/qa_tests.rs"]
mod qa_tests;
#[cfg(test)]
#[path = "autoeq/runopt_tests.rs"]
mod runopt_tests;
#[cfg(test)]
#[path = "autoeq/save_tests.rs"]
mod save_tests;
#[cfg(test)]
#[path = "autoeq/spacing_tests.rs"]
mod spacing_tests;

/// A command-line tool to find optimal IIR filters to match a frequency curve.
#[tokio::main]
pub async fn run_command() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut args = autoeq::cli::Args::parse();

    // Apply preset (if specified) before processing other flags
    args.apply_preset();

    // Check if user wants to see algorithm list
    if args.algo_list {
        autoeq::cli::display_algorithm_list();
    }

    // Check if user wants to see strategy list
    if args.strategy_list {
        autoeq::cli::display_strategy_list();
    }

    // Check if user wants to see PEQ model list
    if args.peq_model_list {
        autoeq::cli::display_peq_model_list();
    }

    // Validate CLI arguments
    autoeq::cli::validate_args_or_exit(&args);

    // Run the main logic
    if let Err(e) = run(args).await {
        error!("Application error: {:#}", e);
        std::process::exit(1);
    }

    Ok(())
}

fn cli_config_identity(
    params: &autoeq::OptimParams,
    input: &autoeq::Curve,
    target: &autoeq::Curve,
    deviation: &autoeq::Curve,
    spin_data: Option<&std::collections::HashMap<String, autoeq::Curve>>,
) -> Result<String> {
    let mut canonical_params = params.clone();
    canonical_params.algo = autoeq::workflow::resume::canonical_optimizer_identity(&params.algo)
        .map_err(|error| anyhow!("{error}"))?;
    let mut parts = vec![
        format!("cli-optimizer-params:{canonical_params:#?}"),
        format!("input:{}", input.content_hash()?),
        format!("target:{}", target.content_hash()?),
        format!("deviation:{}", deviation.content_hash()?),
    ];
    if let Some(spin_data) = spin_data {
        let mut hashes = spin_data
            .iter()
            .map(|(name, curve)| Ok((name, curve.content_hash()?)))
            .collect::<Result<Vec<_>>>()?;
        hashes.sort_by(|left, right| left.0.cmp(right.0));
        for (name, hash) in hashes {
            parts.push(format!("spin:{name}:{hash}"));
        }
    }
    Ok(autoeq::workflow::resume::config_identity_digest(
        parts.iter().map(String::as_str),
    ))
}

#[derive(Clone)]
struct CliCheckpointIdentity {
    measurement: String,
    config: String,
    normalization: String,
    sample_rate: f64,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    algorithm: String,
    algorithm_version: String,
    budget: usize,
    seed: Option<u64>,
}

impl CliCheckpointIdentity {
    fn as_identity(&self) -> autoeq::workflow::resume::WarmStartIdentity<'_> {
        autoeq::workflow::resume::WarmStartIdentity {
            measurement_identity: &self.measurement,
            config_identity: &self.config,
            normalization_hash: Some(&self.normalization),
            sample_rate: self.sample_rate,
            lower_bounds: &self.lower_bounds,
            upper_bounds: &self.upper_bounds,
            algorithm: &self.algorithm,
            algorithm_version: &self.algorithm_version,
            budget: self.budget,
        }
    }
}

fn cli_candidate_within_bounds(candidate: &[f64], lower: &[f64], upper: &[f64]) -> bool {
    !candidate.is_empty()
        && candidate.len() == lower.len()
        && candidate.len() == upper.len()
        && candidate
            .iter()
            .zip(lower.iter().zip(upper))
            .all(|(&value, (&minimum, &maximum))| {
                value.is_finite()
                    && minimum.is_finite()
                    && maximum.is_finite()
                    && minimum <= value
                    && value <= maximum
            })
}

fn cli_checkpoint_callback(
    path: PathBuf,
    identity: CliCheckpointIdentity,
    objective_data: autoeq::optim::ObjectiveData,
    constraint_spec: autoeq::optim::OwnedConstraintSpec,
    best_loss: Arc<Mutex<Option<f64>>>,
) -> runopt::CandidateProgressCallback {
    Box::new(move |update| {
        if !update.loss.is_finite()
            || !cli_candidate_within_bounds(
                &update.params,
                &identity.lower_bounds,
                &identity.upper_bounds,
            )
        {
            return Ok(());
        }

        let Ok(finalized) = autoeq::optim::finalize_candidate(
            "cli-checkpoint-progress",
            &update.params,
            &objective_data,
            &constraint_spec.as_spec(),
        ) else {
            return Ok(());
        };
        if !finalized.loss.is_finite()
            || !cli_candidate_within_bounds(
                &finalized.params,
                &identity.lower_bounds,
                &identity.upper_bounds,
            )
        {
            return Ok(());
        }
        let mut best_loss = best_loss
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if best_loss.is_some_and(|best| finalized.loss >= best) {
            return Ok(());
        }

        let state = autoeq::workflow::resume::OptimizerState::from_candidate(
            &finalized.params,
            finalized.loss,
            update.iteration.min(identity.budget),
            identity.budget,
            false,
            identity.seed,
            true,
            &identity.as_identity(),
        );
        autoeq::workflow::resume::save_optimizer_state(&state, &path).map_err(|error| {
            format!(
                "failed to save progress checkpoint {}: {error}",
                path.display()
            )
        })?;
        *best_loss = Some(finalized.loss);
        Ok(())
    })
}

async fn run(args: autoeq::cli::Args) -> Result<()> {
    // Check if this is multi-driver mode
    if args.loss == autoeq::LossType::DriversFlat {
        if args.resume_state.is_some() || args.checkpoint_state.is_some() {
            return Err(anyhow!(
                "warm-start checkpoints are not supported for multi-driver optimization yet"
            ));
        }
        return run_multi_driver_optimization(&args).await;
    }

    // Load and prepare all input data
    let (standard_freq, input_curve, target_curve, deviation_curve, spin_data) =
        load::load_and_prepare(&args)
            .await
            .map_err(|e| anyhow!("{}", e))
            .context("Failed to load and prepare input data")?;

    // Objective data
    let optim_params = autoeq::OptimParams::from(&args);
    let (objective_data, use_cea) = autoeq::workflow::setup_objective_data(
        &optim_params,
        &input_curve,
        &target_curve,
        &deviation_curve,
        &spin_data,
    )
    .map_err(|e| anyhow!("{}", e))
    .context("Failed to setup objective data")?;

    // Compute pre-optimization metrics
    let pre_metrics = prescore::compute_pre_optimization_metrics(
        &args,
        &objective_data,
        use_cea,
        &deviation_curve,
        &spin_data,
    )
    .await
    .map_err(|e| anyhow!("{}", e))
    .context("Failed to compute pre-optimization metrics")?;

    // Checkpoint identity binds both the prepared measurement and all current
    // settings/data that can change search or correction behavior.
    let measurement_identity = input_curve.content_hash()?;
    let normalization_hash = measurement_identity.clone();
    let config_identity = cli_config_identity(
        &optim_params,
        &input_curve,
        &target_curve,
        &deviation_curve,
        spin_data.as_ref(),
    )?;
    let algorithm_identity =
        autoeq::workflow::resume::canonical_optimizer_identity(&optim_params.algo)
            .map_err(|error| anyhow!("{error}"))?;
    let (lower_bounds, upper_bounds) = autoeq::workflow::setup_bounds(&optim_params);
    let checkpoint_identity = CliCheckpointIdentity {
        measurement: measurement_identity.clone(),
        config: config_identity.clone(),
        normalization: normalization_hash.clone(),
        sample_rate: optim_params.sample_rate,
        lower_bounds: lower_bounds.clone(),
        upper_bounds: upper_bounds.clone(),
        algorithm: algorithm_identity.clone(),
        algorithm_version: autoeq::optim::OPTIMIZER_IMPLEMENTATION_VERSION.to_owned(),
        budget: optim_params.maxeval,
        seed: optim_params.seed,
    };
    let identity = checkpoint_identity.as_identity();
    let warm_state = match args.resume_state.as_ref() {
        Some(path) => {
            let state = autoeq::workflow::resume::load_optimizer_state(path)
                .map_err(|error| anyhow!("failed to load warm-start checkpoint: {error}"))?
                .ok_or_else(|| {
                    anyhow!(
                        "requested warm-start checkpoint {} does not exist",
                        path.display()
                    )
                })?;
            state
                .check_warm_start_compatible(&identity)
                .map_err(|reason| anyhow!("warm-start checkpoint rejected: {reason}"))?;
            Some(state)
        }
        None => None,
    };
    let constraint_spec = autoeq::optim::OwnedConstraintSpec::from_params(&optim_params)
        .map_err(|reason| anyhow!("invalid current optimizer constraints: {reason}"))?;
    let warm_candidate = match warm_state.as_ref() {
        Some(state) => {
            let finalized = autoeq::optim::finalize_candidate(
                "cli-warm-start",
                &state.best_params,
                &objective_data,
                &constraint_spec.as_spec(),
            )
            .map_err(|reason| {
                anyhow!("warm-start candidate failed current constraint validation: {reason}")
            })?;
            if !cli_candidate_within_bounds(&finalized.params, &lower_bounds, &upper_bounds) {
                return Err(anyhow!(
                    "warm-start candidate is outside current optimizer bounds after constraint validation"
                ));
            }
            Some(finalized.params)
        }
        None => None,
    };
    if warm_candidate.is_some() {
        let backend = autoeq::optim::backend::resolve(&optim_params.algo)
            .ok_or_else(|| anyhow!("unknown optimizer backend: {}", optim_params.algo))?;
        if !backend.supports_initial_candidate() {
            return Err(anyhow!(
                "warm-start reuse is unsupported for {} because this optimizer path does not use the supplied initial candidate",
                backend.name()
            ));
        }
    }

    // Save a feasible candidate before the potentially long run. AutoEQ DE
    // also exposes full parameter snapshots, so it can replace this state
    // periodically; other backends only produce a final replacement.
    let checkpoint_best_loss = Arc::new(Mutex::new(None));
    let supports_candidate_progress = algorithm_identity.eq_ignore_ascii_case("autoeq:de");
    if args.checkpoint_state.is_some() && !supports_candidate_progress {
        warn!(
            "{} does not expose candidate snapshots; checkpoints retain the initial candidate until the final result",
            algorithm_identity
        );
    }
    if let Some(path) = args.checkpoint_state.as_ref() {
        let candidate = match warm_candidate.as_ref() {
            Some(candidate) => candidate.clone(),
            None => autoeq::workflow::initial_guess(&optim_params, &lower_bounds, &upper_bounds),
        };
        let finalized = autoeq::optim::finalize_candidate(
            "cli-checkpoint-start",
            &candidate,
            &objective_data,
            &constraint_spec.as_spec(),
        )
        .map_err(|reason| anyhow!("cannot checkpoint starting candidate: {reason}"))?;
        let state = autoeq::workflow::resume::OptimizerState::from_candidate(
            &finalized.params,
            finalized.loss,
            0,
            optim_params.maxeval,
            false,
            optim_params.seed,
            true,
            &identity,
        );
        autoeq::workflow::resume::save_optimizer_state(&state, path)
            .map_err(|error| anyhow!("failed to write warm-start checkpoint: {error}"))?;
        *checkpoint_best_loss
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(finalized.loss);
    }

    // A saved candidate seeds a fresh optimizer population and random stream;
    // this path does not restore exact optimizer continuation state.
    if warm_candidate.is_some() {
        info!("Starting optimization from a saved candidate with a fresh optimizer run...");
    } else {
        info!("🚀 Starting optimization...");
    }
    let opt_result = if let Some(path) = args
        .checkpoint_state
        .as_ref()
        .filter(|_| supports_candidate_progress)
    {
        let callback = cli_checkpoint_callback(
            path.clone(),
            checkpoint_identity.clone(),
            objective_data.clone(),
            constraint_spec.clone(),
            Arc::clone(&checkpoint_best_loss),
        );
        runopt::perform_optimization_with_progress_callback(
            &optim_params,
            &objective_data,
            None,
            warm_candidate.as_deref(),
            callback,
        )
    } else {
        match warm_candidate.as_deref() {
            Some(candidate) => runopt::perform_optimization_with_candidate(
                &optim_params,
                &objective_data,
                None,
                candidate,
            ),
            None => runopt::perform_optimization(&optim_params, &objective_data),
        }
    }
    .map_err(|e| anyhow!("{}", e))
    .context("Optimization failed")?;
    if let Some(path) = args.checkpoint_state.as_ref() {
        let finalized = autoeq::optim::finalize_candidate(
            "cli-checkpoint-final",
            &opt_result.params,
            &objective_data,
            &constraint_spec.as_spec(),
        )
        .map_err(|reason| {
            anyhow!("final optimizer candidate failed constraint validation: {reason}")
        })?;
        let selected_iteration = opt_result
            .optimizer_evidence
            .iter()
            .find(|evidence| evidence.selected_for_output)
            .and_then(|evidence| evidence.evaluation_count)
            .unwrap_or(optim_params.maxeval)
            .min(optim_params.maxeval);
        let checkpoint_loss = *checkpoint_best_loss
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if checkpoint_loss.is_none_or(|best_loss| finalized.loss < best_loss) {
            let state = autoeq::workflow::resume::OptimizerState::from_candidate(
                &finalized.params,
                finalized.loss,
                selected_iteration,
                optim_params.maxeval,
                opt_result.converged,
                optim_params.seed,
                true,
                &identity,
            );
            autoeq::workflow::resume::save_optimizer_state(&state, path)
                .map_err(|error| anyhow!("failed to write final warm-start checkpoint: {error}"))?;
        }
    }
    for evidence in &opt_result.optimizer_evidence {
        log::debug!(
            "Optimizer evidence: {} termination={:?} confidence={:?} selected={} status={}",
            evidence.algorithm,
            evidence.termination,
            evidence.confidence,
            evidence.selected_for_output,
            evidence.status
        );
    }

    // Compute post-optimization metrics
    let post_metrics = postscore::compute_post_optimization_metrics(
        &args,
        &objective_data,
        use_cea,
        &opt_result.params,
        &standard_freq,
        &target_curve,
        &input_curve,
        &spin_data,
        pre_metrics.cea2034_metrics,
        pre_metrics.headphone_loss,
    )
    .await
    .map_err(|e| anyhow!("{}", e))
    .context("Failed to compute post-optimization metrics")?;

    // Print pre and post optimization scores
    postscore::print_optimization_scores(
        &args,
        &post_metrics,
        opt_result.pre_objective,
        opt_result.post_objective,
    );

    // Extract scores for QA summary
    let (pre_score, post_score) = match objective_data.loss_type {
        autoeq::LossType::HeadphoneFlat | autoeq::LossType::HeadphoneScore => {
            (post_metrics.pre_headphone_loss, post_metrics.headphone_loss)
        }
        autoeq::LossType::SpeakerFlat
        | autoeq::LossType::SpeakerFlatAsymmetric
        | autoeq::LossType::SpeakerScore
        | autoeq::LossType::Epa => (
            post_metrics.pre_cea2034.as_ref().map(|m| m.pref_score),
            post_metrics.cea2034_metrics.as_ref().map(|m| m.pref_score),
        ),
        autoeq::LossType::DriversFlat | autoeq::LossType::MultiSubFlat => {
            // Unreachable: DriversFlat mode uses a separate code path
            unreachable!("DriversFlat mode should not reach this point");
        }
    };

    // Check spacing constraints
    let spacing_ok = spacing::check_spacing_constraints(&opt_result.params, &optim_params);

    // Output QA summary if in QA mode
    if let Some(qa_threshold) = args.qa {
        let converge_str = if opt_result.converged {
            "true"
        } else {
            "false"
        };
        let spacing_str = if spacing_ok { "ok" } else { "ko" };

        // Use scores if available, otherwise use objective function values
        let (pre_str, post_str) = if let (Some(pre), Some(post)) = (pre_score, post_score) {
            (format!("{:.3}", pre), format!("{:.3}", post))
        } else {
            // Fall back to objective function values
            let pre_obj = opt_result.pre_objective.unwrap_or(f64::NAN);
            let post_obj = opt_result.post_objective.unwrap_or(f64::NAN);
            (format!("{:.6}", pre_obj), format!("{:.6}", post_obj))
        };

        // Always output the standard QA summary line for backward compatibility
        // This uses println! because it is the "result" output for scripts
        println!(
            "Converge: {} | Spacing: {} | Pre: {} | Post: {}",
            converge_str, spacing_str, pre_str, post_str
        );

        // Perform additional QA analysis if threshold was provided
        let qa_result = qa::perform_qa_analysis(
            opt_result.converged,
            spacing_ok,
            pre_score,
            post_score,
            qa_threshold,
        );
        qa::display_qa_analysis(&qa_result);

        return Ok(());
    }

    // Normal mode: plot and report
    let output_path = args.output.clone().unwrap_or_else(|| {
        let mut path = PathBuf::from("data_generated");
        path.push("autoeq");
        if let Some(speaker) = &args.speaker {
            // Use speaker name for default filename
            let safe_name = speaker.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
            path.push(format!("autoeq_{}", safe_name));
        } else {
            path.push("autoeq_results");
        }
        path
    });

    {
        info!("📊 Generating plots: {}", output_path.display());
        if let Err(e) = plot::plot_results(
            &autoeq_plot::PlotConfig::from(&args),
            &opt_result.params,
            &input_curve,
            &target_curve,
            &deviation_curve,
            &spin_data,
            &output_path,
        ) {
            warn!("Failed to generate plots: {}", e);
        } else {
            info!("✅ Plots generated successfully");
        }
    }

    // The shipped preset serializes frequencies to integer Hz; surface
    // any drift between the optimizer response and the serialized one.
    if let Some(gap) = opt_result.apo_roundtrip_gap {
        if gap > save::APO_ROUNDTRIP_WARN_THRESHOLD {
            log::warn!(
                "APO serialization drifted the objective by {:.6} (> {:.0e}); \
                 reported evidence reflects the optimizer response, the preset \
                 the integer-Hz response",
                gap,
                save::APO_ROUNDTRIP_WARN_THRESHOLD
            );
        } else {
            log::debug!("APO round-trip objective gap: {:.3e}", gap);
        }
    }

    // Save PEQ settings to APO format file
    save::save_peq_to_file(
        &args,
        &opt_result.params,
        &output_path,
        &objective_data.loss_type,
        None,
    )
    .await
    .map_err(|e| anyhow!("{}", e))
    .context("Failed to save PEQ file")?;

    Ok(())
}

/// Run multi-driver crossover optimization
async fn run_multi_driver_optimization(args: &autoeq::cli::Args) -> Result<()> {
    info!("🎵 Multi-driver crossover optimization mode");

    // Collect driver file paths
    let driver_paths: Vec<_> = [&args.driver1, &args.driver2, &args.driver3, &args.driver4]
        .iter()
        .filter_map(|p| p.as_ref())
        .cloned()
        .collect();

    if driver_paths.len() < 2 {
        return Err(anyhow!(
            "At least 2 driver files are required for multi-driver optimization"
        ));
    }

    // Load driver measurements
    let measurements = autoeq::workflow::load_driver_measurements_from_files(&driver_paths)
        .map_err(|e| anyhow!("{}", e))
        .context("Failed to load driver measurements")?;

    // Parse crossover type
    let crossover_type: autoeq::loss::CrossoverType = args
        .crossover_type
        .parse()
        .map_err(|e: String| anyhow!("{}", e))?;

    // Create DriversLossData
    let drivers_data = autoeq::loss::DriversLossData::new(measurements, crossover_type);

    info!(
        "✓ Initialized {} drivers with {:?} crossover",
        drivers_data.drivers.len(),
        drivers_data.crossover_type
    );

    info!("📊 Drivers sorted by frequency (lowest to highest):");
    for (i, driver) in drivers_data.drivers.iter().enumerate() {
        let (min_f, max_f) = driver.freq_range();
        info!(
            "   Driver {}: {:.0} Hz - {:.0} Hz (mean: {:.0} Hz)",
            i + 1,
            min_f,
            max_f,
            driver.mean_freq()
        );
    }

    info!("🎯 Optimization parameters:");
    info!(
        "   {} driver gains + {} crossover frequencies = {} parameters",
        drivers_data.drivers.len(),
        drivers_data.drivers.len() - 1,
        drivers_data.drivers.len() + (drivers_data.drivers.len() - 1)
    );
    info!(
        "   Gain bounds: [{:.1}, {:.1}] dB",
        -args.max_db, args.max_db
    );

    // Optimize using shared function
    info!("🚀 Starting optimization...");
    let smoothness_penalty = autoeq::OptimParams::from(args).smoothness_penalty;
    let result = autoeq::workflow::optimize_drivers_crossover_with_smoothness(
        drivers_data.clone(),
        args.min_freq,
        args.max_freq,
        args.sample_rate,
        &args.algo,
        args.maxeval,
        args.population,
        args.min_db,
        args.max_db,
        None, // No fixed crossover frequencies - optimize them
        args.seed,
        smoothness_penalty,
    )
    .map_err(|e| anyhow!("{}", e))
    .context("Driver optimization failed")?;

    // Extract results
    let gains = &result.gains;
    let delays = &result.delays;
    let xover_freqs = &result.crossover_freqs;

    // Display results
    info!("✅ Optimization complete!");
    info!("📊 Results:");
    info!("Driver Gains:");
    for (i, gain) in gains.iter().enumerate() {
        info!("   Driver {}: {:+.2} dB", i + 1, gain);
    }
    info!("Driver Delays:");
    for (i, delay) in delays.iter().enumerate() {
        info!("   Driver {}: {:.2} ms", i + 1, delay);
    }
    info!("Crossover Frequencies:");
    for (i, freq) in xover_freqs.iter().enumerate() {
        info!("   Between Driver {} and {}: {:.0} Hz", i + 1, i + 2, freq);
    }
    info!("Crossover Type: {:?}", drivers_data.crossover_type);

    // Display pre and post objective values
    info!("Loss (RMS deviation from flat):");
    info!("   Before optimization: {:.6} dB", result.pre_objective);
    info!("   After optimization:  {:.6} dB", result.post_objective);
    info!(
        "   Improvement: {:.2}%",
        (result.pre_objective - result.post_objective) / result.pre_objective * 100.0
    );

    // Generate plot
    let output_path = args.output.clone().unwrap_or_else(|| {
        let mut path = std::path::PathBuf::from("data_generated");
        path.push("autoeq");
        path.push("drivers_crossover_results");
        path
    });

    {
        info!("📊 Generating plots: {}", output_path.display());
        if let Err(e) = autoeq_plot::plot_drivers_results(
            &drivers_data,
            gains,
            xover_freqs,
            Some(delays),
            args.sample_rate,
            &output_path,
        ) {
            warn!("Failed to generate plots: {}", e);
        } else {
            info!("✅ Plots generated successfully");
        }
    }

    // QA mode output
    if let Some(_qa_threshold) = args.qa {
        let converge_str = if result.converged { "true" } else { "false" };

        println!(
            "Converge: {} | Pre: {:.6} | Post: {:.6}",
            converge_str, result.pre_objective, result.post_objective
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use autoeq::cli::Args;
    use clap::Parser;
    use ndarray::Array1;

    #[test]
    fn setup_bounds_hp_pk_mode_overrides_first_triplet() {
        use autoeq::PeqModel;
        let mut args = Args::parse_from(["autoeq-test"]);
        args.num_filters = 2;
        args.min_freq = 30.0;
        args.max_freq = 2000.0;
        args.min_q = 0.3;
        args.max_q = 8.0;
        args.min_db = -9.0;
        args.max_db = 12.0;
        args.peq_model = PeqModel::HpPk;

        let (lb, ub) = autoeq::workflow::setup_bounds(&autoeq::OptimParams::from(&args));
        assert_eq!(lb.len(), args.num_filters * 3);
        assert_eq!(ub.len(), args.num_filters * 3);

        // First triplet should be overridden for HP
        assert!((lb[0] - 20.0_f64.max(args.min_freq).log10()).abs() < 1e-12);
        assert!((ub[0] - 120.0_f64.min(args.min_freq + 20.0).log10()).abs() < 1e-12);
        assert!((lb[1] - 1.0).abs() < 1e-12);
        assert!((ub[1] - 1.5).abs() < 1e-12);
        assert!((lb[2] - 0.0).abs() < 1e-12);
        assert!((ub[2] - 0.0).abs() < 1e-12);

        // Second filter should follow the general pattern
        let q_lower = args.min_q.max(0.1);
        assert!((lb[3] - args.min_freq.log10()).abs() < 1e-12);
        assert!((lb[4] - q_lower).abs() < 1e-12);
        assert!((lb[5] - args.min_db).abs() < 1e-12);
        assert!((ub[3] - args.max_freq.log10()).abs() < 1e-12);
        assert!((ub[4] - args.max_q).abs() < 1e-12);
        assert!((ub[5] - args.max_db).abs() < 1e-12);
    }

    #[test]
    fn listening_window_target_profile() {
        use autoeq::PeqModel;
        let mut args = Args::parse_from(["autoeq-test"]);
        // Ensure we hit the custom target branch and avoid clamping negatives
        args.curve_name = "Listening Window".to_string();
        args.peq_model = PeqModel::HpPk;

        let freqs = Array1::from_vec(vec![500.0_f64, 1000.0_f64, 20000.0_f64]);
        let spl = Array1::<f64>::zeros(freqs.len());
        let curve = autoeq::Curve {
            freq: freqs.clone(),
            spl,
            phase: None,
            ..Default::default()
        };

        let target_curve = autoeq::workflow::build_target_curve(
            &autoeq::workflow::TargetConfig::from(&args),
            &freqs,
            &curve,
        )
        .expect("build_target_curve should succeed");
        // Since SPL is zero, target_curve.spl == base_target
        assert!((target_curve.spl[0] - 0.0).abs() < 1e-12);
        assert!((target_curve.spl[1] - 0.0).abs() < 1e-12);
        assert!((target_curve.spl[2] - (-0.5)).abs() < 1e-12);
    }
}
