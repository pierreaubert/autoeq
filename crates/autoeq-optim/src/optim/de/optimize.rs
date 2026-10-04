use super::super::ObjectiveData;
use super::super::constraint_envelope::{
    OwnedConstraintSpec, finalize_candidate, project_gains_onto_envelopes,
};
use super::create::create_de_callback;
use super::create::create_de_objective;
use super::misc::process_de_results;
use super::misc::register_de_constraint;
use super::types::setup_de_common;
use crate::constraints::{
    CeilingConstraintData, MinGainConstraintData, SpacingConstraintData, constraint_ceiling,
    constraint_min_gain, constraint_spacing,
};
use crate::de::init_sobol::init_halton;
use crate::de::{
    CallbackAction, DECheckpoint, DEConfigBuilder, DEIntermediate, DifferentialEvolution, Init,
    Mutation, ParallelConfig, Strategy, differential_evolution,
};
use crate::initial_guess::{SmartInitConfig, create_smart_initial_guesses};
use ndarray::Array1;

/// Persistence callback used by exact DE continuation.
pub type DECheckpointSaveCallback =
    Box<dyn FnMut(&DECheckpoint) -> std::result::Result<(), String> + Send>;

/// Exact continuation inputs and generation-barrier persistence callback.
pub struct DEExactContinuation {
    /// Previously saved DE state; None starts a new exact-checkpointed run.
    pub checkpoint: Option<DECheckpoint>,
    /// Caller identity for the objective and opaque callback semantics.
    pub run_identity: String,
    /// Saves each complete barrier state; errors stop optimization.
    pub save_callback: DECheckpointSaveCallback,
}

impl std::fmt::Debug for DEExactContinuation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DEExactContinuation")
            .field("checkpoint", &self.checkpoint)
            .field("run_identity", &self.run_identity)
            .field("save_callback", &"<generation-barrier callback>")
            .finish()
    }
}

/// Optimize filter parameters using AutoEQ custom algorithms
pub fn optimize_filters_autoeq(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    autoeq_name: &str,
    params: &crate::OptimParams,
) -> Result<(String, f64), (String, f64)> {
    // Create the callback with all the logging and user feedback
    let callback = create_de_callback("autoeq::DE", params.quiet);

    // Delegate to the callback-based version
    optimize_filters_autoeq_with_callback(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        autoeq_name,
        params,
        callback,
    )
}

/// AutoEQ DE optimization with external progress callback
pub fn optimize_filters_autoeq_with_callback(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    autoeq_name: &str,
    params: &crate::OptimParams,
    callback: Box<dyn FnMut(&DEIntermediate) -> CallbackAction + Send>,
) -> Result<(String, f64), (String, f64)> {
    optimize_filters_autoeq_with_callback_and_initial(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        autoeq_name,
        params,
        None,
        callback,
    )
}

/// AutoEQ DE optimization whose first individual is an explicit candidate
/// when one is supplied. The remaining population is initialized normally.
#[expect(
    clippy::too_many_arguments,
    reason = "the explicit DE inputs keep bounds, objective, candidate, parameters, and callback visible"
)]
pub fn optimize_filters_autoeq_with_callback_and_initial(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    _autoeq_name: &str,
    params: &crate::OptimParams,
    initial_candidate: Option<&[f64]>,
    callback: Box<dyn FnMut(&DEIntermediate) -> CallbackAction + Send>,
) -> Result<(String, f64), (String, f64)> {
    optimize_filters_autoeq_with_callback_and_initial_and_exact(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        _autoeq_name,
        params,
        initial_candidate,
        callback,
        None,
    )
}

/// AutoEQ DE optimization with safe-barrier exact-state persistence.
///
/// This mode is deliberately separate from candidate warm starts. It always
/// uses a deterministic seed and validates the state against the full DE
/// configuration before any objective evaluation.
///
/// # Errors
///
/// Returns an error tuple when bounds, configuration, or the saved checkpoint
/// are invalid, the exact run identity differs, or checkpoint persistence
/// fails. The error value is set to positive infinity.
#[expect(
    clippy::too_many_arguments,
    reason = "the explicit optimizer inputs and exact-continuation control stay visible at the production boundary"
)]
pub fn optimize_filters_autoeq_with_exact_checkpoint(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    autoeq_name: &str,
    params: &crate::OptimParams,
    callback: Box<dyn FnMut(&DEIntermediate) -> CallbackAction + Send>,
    continuation: DEExactContinuation,
) -> Result<(String, f64), (String, f64)> {
    optimize_filters_autoeq_with_callback_and_initial_and_exact(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        autoeq_name,
        params,
        None,
        callback,
        Some(continuation),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the legacy typed callback plus optional initial candidate are preserved while exact state is additive"
)]
fn optimize_filters_autoeq_with_callback_and_initial_and_exact(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    _autoeq_name: &str,
    params: &crate::OptimParams,
    initial_candidate: Option<&[f64]>,
    mut callback: Box<dyn FnMut(&DEIntermediate) -> CallbackAction + Send>,
    mut exact: Option<DEExactContinuation>,
) -> Result<(String, f64), (String, f64)> {
    let explicit_initial_candidate = if let Some(candidate) = initial_candidate {
        if candidate.is_empty()
            || candidate.len() != x.len()
            || candidate.len() != lower_bounds.len()
            || candidate.len() != upper_bounds.len()
        {
            return Err((
                format!(
                    "warm-start candidate has {} parameters; x/lower/upper dimensions are {}/{}/{}",
                    candidate.len(),
                    x.len(),
                    lower_bounds.len(),
                    upper_bounds.len()
                ),
                f64::INFINITY,
            ));
        }
        for (index, (&value, (&lower, &upper))) in candidate
            .iter()
            .zip(lower_bounds.iter().zip(upper_bounds))
            .enumerate()
        {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err((
                    format!(
                        "warm-start bounds at parameter {index} are invalid: [{lower}, {upper}]"
                    ),
                    f64::INFINITY,
                ));
            }
            if !value.is_finite() {
                return Err((
                    format!("warm-start candidate parameter {index} is not finite"),
                    f64::INFINITY,
                ));
            }
            if value < lower || value > upper {
                return Err((
                    format!(
                        "warm-start candidate parameter {index}={value} is outside bounds [{lower}, {upper}]"
                    ),
                    f64::INFINITY,
                ));
            }
        }
        let constraint_spec = OwnedConstraintSpec::from_params(params).map_err(|reason| {
            (
                format!("current warm-start constraint spec is invalid: {reason}"),
                f64::INFINITY,
            )
        })?;
        let finalized = finalize_candidate(
            "de-warm-start",
            candidate,
            &objective_data,
            &constraint_spec.as_spec(),
        )
        .map_err(|reason| {
            (
                format!("warm-start candidate fails current constraint checks: {reason}"),
                f64::INFINITY,
            )
        })?;
        for (index, (&value, (&lower, &upper))) in finalized
            .params
            .iter()
            .zip(lower_bounds.iter().zip(upper_bounds))
            .enumerate()
        {
            if !value.is_finite() || value < lower || value > upper {
                return Err((
                    format!(
                        "finalized warm-start parameter {index}={value} is outside bounds [{lower}, {upper}]"
                    ),
                    f64::INFINITY,
                ));
            }
        }
        Some(finalized.params)
    } else {
        None
    };

    // Extract parameters from args
    let population = params.population;
    let maxeval = params.maxeval;

    // Reuse same setup as standard AutoEQ DE
    let setup = setup_de_common(
        lower_bounds,
        upper_bounds,
        objective_data.clone(),
        population,
        maxeval,
        params.quiet,
    );
    let base_objective_fn = create_de_objective(setup.penalty_data.clone());

    // Create smart initialization based on frequency response analysis
    // Skip for drivers-flat loss as it uses a different parameter layout
    let smart_guesses = if matches!(
        setup.penalty_data.loss_type,
        crate::LossType::DriversFlat | crate::LossType::MultiSubFlat
    ) {
        Vec::new()
    } else {
        let params_per_filter = crate::param_utils::params_per_filter(params.peq_model);
        let num_filters = x.len() / params_per_filter;
        // If the caller (typically roomeq's `prepare_single_channel_eq`)
        // already detected high-quality room-mode problems via SSIR /
        // decomposed correction, feed them into the smart-guess
        // generator instead of letting it run its own cruder
        // find_peaks over the smoothed deviation. Empty list → fall
        // back to the legacy auto-detection.
        let pre_detected_problems = setup.penalty_data.detected_problems.clone();
        if !pre_detected_problems.is_empty() && !params.quiet {
            log::debug!(
                "🎯 Seeding smart initial guesses with {} pre-detected problem(s) from upstream analysis",
                pre_detected_problems.len()
            );
        }
        let smart_config = SmartInitConfig {
            seed: params.seed, // Pass seed for deterministic initialization
            pre_detected_problems,
            ..SmartInitConfig::default()
        };

        // Use the deviation curve (target - measurement) to identify problems.
        // Positive deviation = needs boost, negative = needs cut.
        let target_response = &setup.penalty_data.deviation;
        let freq_grid = &setup.penalty_data.freqs;

        if !params.quiet {
            log::debug!(
                "🧠 Generating smart initial guesses based on frequency response analysis..."
            );
        }
        let guesses = create_smart_initial_guesses(
            target_response,
            freq_grid,
            num_filters,
            &setup.bounds,
            &smart_config,
            params.peq_model,
        );

        if !params.quiet {
            log::debug!("📊 Generated {} smart initial guesses", guesses.len());
        }
        // O1: project seeds onto the configured per-filter gain envelopes.
        // Deterministic post-processing that consumes no RNG; bit-identical
        // without envelopes.
        guesses
            .into_iter()
            .map(|guess| {
                project_gains_onto_envelopes(
                    &guess,
                    params.peq_model,
                    setup.penalty_data.loss_type,
                    setup.penalty_data.max_boost_envelope.as_deref(),
                    setup.penalty_data.min_cut_envelope.as_deref(),
                )
                .0
            })
            .collect()
    };

    // Generate Sobol quasi-random population for better space coverage
    let sobol_samples = init_halton(
        x.len(),
        setup.population_size.saturating_sub(smart_guesses.len()),
        &setup.bounds,
    );

    if !params.quiet {
        log::debug!(
            "🎯 Generated {} Sobol quasi-random samples",
            sobol_samples.len()
        );
    }

    // A validated warm-start candidate takes precedence as DE's x0.
    let best_initial_guess = choose_best_initial_guess(
        explicit_initial_candidate.as_deref(),
        &smart_guesses,
        &sobol_samples,
        x,
    );

    if !params.quiet {
        log::debug!("🚀 Using smart initial guess with Sobol population initialization");
    }

    // Parse strategy from CLI args
    use std::str::FromStr;
    let strategy = Strategy::from_str(&params.strategy).unwrap_or_else(|_| {
        if !params.quiet {
            log::debug!(
                "⚠️ Warning: Invalid strategy '{}', falling back to CurrentToBest1Bin",
                params.strategy
            );
        }
        Strategy::CurrentToBest1Bin
    });

    // Set up adaptive configuration if using adaptive strategies
    let adaptive_config = if matches!(strategy, Strategy::AdaptiveBin | Strategy::AdaptiveExp) {
        Some(crate::de::AdaptiveConfig {
            adaptive_mutation: true,
            wls_enabled: false,                    // Disable WLS for stability
            w_max: 0.8,                            // Reduce max weight for more stability
            w_min: 0.2,                            // Increase min weight for more stability
            w_f: params.adaptive_weight_f * 0.5,   // Make adaptation even more conservative
            w_cr: params.adaptive_weight_cr * 0.5, // Make adaptation even more conservative
            f_m: 0.6,                              // Start with slightly higher F
            cr_m: 0.5,                             // Start with slightly lower CR
            wls_prob: 0.0,                         // Completely disable WLS
            wls_scale: 0.0,                        // Completely disable WLS
        })
    } else {
        None
    };

    // Adjust tolerance for adaptive strategies (they need much more relaxed convergence)
    let (tolerance, atolerance) =
        if matches!(strategy, Strategy::AdaptiveBin | Strategy::AdaptiveExp) {
            // Use much more relaxed tolerances for adaptive strategies - they converge differently
            (params.tolerance * 10.0, params.atolerance * 10.0)
        } else {
            (params.tolerance, params.atolerance)
        };

    // Use constraint helpers for nonlinear constraints
    let mut config_builder = DEConfigBuilder::new()
        .maxiter(setup.max_iter)
        // Speaker-score searches can have a narrow population-fitness spread
        // before the scored preference has improved. Use the existing
        // generation/evaluation cap before accepting population convergence.
        .min_convergence_iter(if setup.penalty_data.loss_type == crate::LossType::SpeakerScore {
            setup.max_iter
        } else {
            0
        })
        .popsize(setup.pop_multiplier)
        .tol(tolerance)
        .atol(atolerance)
        .strategy(strategy)
        .mutation(Mutation::Range { min: 0.4, max: 1.2 })
        .recombination(params.recombination)
        .init(Init::LatinHypercube) // Use Latin Hypercube sampling for population
        .x0(best_initial_guess) // Use smart guess as initial best individual
        .disp(false)
        .callback(Box::new(move |intermediate| callback(intermediate)));

    // Unspecified never means nondeterministic: unseeded runs use the shared
    // default seed instead of OS entropy.
    let seed_value = params.seed.unwrap_or(crate::DEFAULT_SEED);
    config_builder = config_builder.seed(seed_value);
    if !params.quiet {
        log::debug!("🎲 Using deterministic seed: {}", seed_value);
    }

    // Add adaptive configuration if present
    if let Some(adaptive_cfg) = adaptive_config {
        config_builder = config_builder.adaptive(adaptive_cfg);
    }

    // Configure parallel evaluation
    let parallel_config = ParallelConfig {
        enabled: !params.no_parallel,
        num_threads: if params.parallel_threads == 0 {
            None // Use all available cores
        } else {
            Some(params.parallel_threads)
        },
    };
    config_builder = config_builder.parallel(parallel_config);

    if !params.no_parallel && !params.quiet {
        log::debug!(
            "🚄 Parallel evaluation enabled with {} threads",
            if params.parallel_threads.eq(&0) {
                "all available".to_string()
            } else {
                params.parallel_threads.to_string()
            }
        );
    }

    // Add native nonlinear constraints
    let mut config = config_builder
        .build()
        .map_err(|e| (format!("DE config build failed: {:?}", e), f64::INFINITY))?;

    // Register nonlinear constraints using helper
    if setup.penalty_data.max_db > 0.0 {
        register_de_constraint(
            &mut config,
            constraint_ceiling,
            CeilingConstraintData {
                freqs: setup.penalty_data.freqs.clone(),
                srate: setup.penalty_data.srate,
                max_db: setup.penalty_data.max_db,
                peq_model: setup.penalty_data.peq_model,
            },
        );
    }

    if setup.penalty_data.min_db > 0.0 {
        register_de_constraint(
            &mut config,
            constraint_min_gain,
            MinGainConstraintData {
                min_db: setup.penalty_data.min_db,
                peq_model: setup.penalty_data.peq_model,
            },
        );
    }

    if setup.penalty_data.min_spacing_oct > 0.0 {
        register_de_constraint(
            &mut config,
            constraint_spacing,
            SpacingConstraintData {
                min_spacing_oct: setup.penalty_data.min_spacing_oct,
                peq_model: setup.penalty_data.peq_model,
            },
        );
    }

    if exact.is_some() && explicit_initial_candidate.is_some() {
        return Err((
            "exact DE continuation cannot be combined with a warm-start candidate".to_owned(),
            f64::INFINITY,
        ));
    }
    let result = if let Some(continuation) = exact.as_mut() {
        let lower = Array1::from_iter(setup.bounds.iter().map(|(lower, _)| *lower));
        let upper = Array1::from_iter(setup.bounds.iter().map(|(_, upper)| *upper));
        let mut solver = DifferentialEvolution::new(&base_objective_fn, lower, upper)
            .map_err(|error| (format!("DE initialization failed: {error}"), f64::INFINITY))?;
        *solver.config_mut() = config;
        solver.solve_with_checkpoint(
            continuation.checkpoint.as_ref(),
            &continuation.run_identity,
            Some(&mut *continuation.save_callback),
        )
    } else {
        differential_evolution(&base_objective_fn, &setup.bounds, config)
    }
    .map_err(|error| (format!("DE optimization failed: {error:?}"), f64::INFINITY))?;
    process_de_results(x, result, "AutoDE")
}

fn choose_best_initial_guess(
    explicit_candidate: Option<&[f64]>,
    smart_guesses: &[Vec<f64>],
    sobol_samples: &[Vec<f64>],
    current: &[f64],
) -> Array1<f64> {
    if let Some(candidate) = explicit_candidate {
        Array1::from(candidate.to_vec())
    } else if let Some(candidate) = smart_guesses.first() {
        Array1::from(candidate.clone())
    } else if let Some(candidate) = sobol_samples.first() {
        Array1::from(candidate.clone())
    } else {
        Array1::from(current.to_vec())
    }
}

#[cfg(test)]
mod warm_start_initialization_tests {
    use super::choose_best_initial_guess;

    #[test]
    fn explicit_candidate_is_the_de_initial_vector() {
        let saved = [0.25, 1.75, -2.0];
        let smart_guesses = vec![vec![0.5, 0.9, 1.0]];
        let sobol_samples = vec![vec![0.7, 0.8, 1.5]];

        let selected = choose_best_initial_guess(
            Some(&saved),
            &smart_guesses,
            &sobol_samples,
            &[0.9, 0.6, 0.0],
        );

        assert_eq!(selected.to_vec(), saved);
    }
}
