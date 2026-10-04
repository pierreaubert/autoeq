// Metaheuristics-specific optimization code

use super::backend::{
    AlgorithmType, BackendSearchEvidence, BackendSearchStopCause, ConstraintCapabilities, FilterOptimizer,
    FilterOptimizerOutput,
};
use super::callback::{ProgressTracker, format_param_summary};
use super::constraints_install::install_constraints;
use super::optimize::OptimizerBackendCompletion;
use super::params::OptimParams;
use super::run_control::OptimizerBudgetProfile;
use super::{ObjectiveData, OptimProgressCallback, PenaltyMode, compute_fitness_penalties_ref};
use ndarray::Array1;

/// Metaheuristics-backed `FilterOptimizer` (one instance per algorithm
/// variant: de, pso, rga, tlbo, firefly).
pub struct MhBackend {
    name: &'static str,
    algo_suffix: &'static str,
    fallback_mode: PenaltyMode,
}

impl MhBackend {
    pub fn new_de(name: &'static str) -> Self {
        Self {
            name,
            algo_suffix: "de",
            fallback_mode: PenaltyMode::Standard,
        }
    }
    pub fn new_pso(name: &'static str) -> Self {
        Self {
            name,
            algo_suffix: "pso",
            fallback_mode: PenaltyMode::Pso,
        }
    }
    pub fn new_rga(name: &'static str) -> Self {
        Self {
            name,
            algo_suffix: "rga",
            fallback_mode: PenaltyMode::Standard,
        }
    }
    pub fn new_tlbo(name: &'static str) -> Self {
        Self {
            name,
            algo_suffix: "tlbo",
            fallback_mode: PenaltyMode::Standard,
        }
    }
    pub fn new_firefly(name: &'static str) -> Self {
        Self {
            name,
            algo_suffix: "firefly",
            fallback_mode: PenaltyMode::Standard,
        }
    }
}

impl FilterOptimizer for MhBackend {
    fn name(&self) -> &'static str {
        self.name
    }
    fn library(&self) -> &'static str {
        "Metaheuristics"
    }
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::Global
    }
    fn capabilities(&self) -> ConstraintCapabilities {
        ConstraintCapabilities {
            nonlinear_ineq: false,
            nonlinear_eq: false,
            linear: false,
            iteration_callback: true,
            fallback_penalty_mode: self.fallback_mode,
        }
    }

    fn evaluation_budget_profile(
        &self,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        params: &OptimParams,
    ) -> Option<OptimizerBudgetProfile> {
        if lower_bounds.len() != upper_bounds.len() {
            return None;
        }
        let population_size = params.population.max(1);
        let generation_limit = params.maxeval.max(population_size).div_ceil(population_size);
        Some(OptimizerBudgetProfile::new(
            params.maxeval,
            None,
            population_size,
            population_size,
            Some(population_size),
            Some(population_size),
            Some(generation_limit),
        ))
    }
    fn optimize(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        callback: Option<OptimProgressCallback>,
    ) -> Result<(String, f64), (String, f64)> {
        self.optimize_with_report(x, lower, upper, objective, params, callback)
            .into_legacy_result()
    }

    fn optimize_with_report(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        callback: Option<OptimProgressCallback>,
    ) -> FilterOptimizerOutput {
        let mut objective = objective;
        let seed = params.seed.unwrap_or(0);
        // MH has no native nonlinear constraints — install_constraints will
        // configure penalty weights matching `self.fallback_mode`.
        let _ = install_constraints(self.capabilities(), &mut objective);

        let (result, search) = match callback {
            Some(mut user_cb) => {
                // Adapt the unified `OptimProgressCallback` to MH's
                // intermediate type. EPA progress is `None` here (only the
                // AutoEQ DE path computes EPA mid-run).
                let mh_cb: Box<dyn FnMut(&MHIntermediate) -> CallbackAction + Send> =
                    Box::new(move |im| user_cb(im.iter, im.fun, None));
                optimize_filters_mh_with_callback_seeded_report(
                    x,
                    lower,
                    upper,
                    objective,
                    self.algo_suffix,
                    params.population,
                    params.maxeval,
                    mh_cb,
                    seed,
                )
            }
            None => optimize_filters_mh_with_callback_seeded_report(
                x,
                lower,
                upper,
                objective,
                self.algo_suffix,
                params.population,
                params.maxeval,
                create_mh_callback(&format!("mh::{}", self.algo_suffix)),
                seed,
            ),
        };
        match search {
            Some(search) => FilterOptimizerOutput::CompletedWithSearchEvidence { result, search },
            None => FilterOptimizerOutput::Completed {
                result,
                pareto_report: None,
            },
        }
    }
}

#[allow(unused_imports)]
use metaheuristics_nature as mh;
#[allow(unused_imports)]
use mh::methods::{De as MhDe, Fa as MhFa, Pso as MhPso, Rga as MhRga, Tlbo as MhTlbo};
#[allow(unused_imports)]
use mh::{Bounded as MhBounded, Fitness as MhFitness, ObjFunc as MhObjFunc, Solver as MhSolver};

/// Information passed to callback after each generation.
///
/// Similar to DEIntermediate but for metaheuristics optimizers.
pub struct MHIntermediate {
    /// Current best solution vector.
    pub x: Array1<f64>,
    /// Current best fitness value.
    pub fun: f64,
    /// Current iteration number.
    pub iter: usize,
}

/// Callback action - shared with DE module for consistency
pub use crate::de::CallbackAction;

// ---------------- Metaheuristics objective and utilities ----------------
use std::sync::{Arc, Mutex};

/// Objective function wrapper for metaheuristics optimizers.
#[derive(Clone)]
pub struct MHObjective {
    /// Objective data containing target curves and loss parameters.
    pub data: ObjectiveData,
    /// Parameter bounds as [min, max] pairs.
    pub bounds: Vec<[f64; 2]>,
    /// Optional callback state for tracking progress.
    pub callback_state: Option<Arc<Mutex<CallbackState>>>,
}

/// State tracked across fitness evaluations for callback reporting.
pub struct CallbackState {
    /// Best fitness value found so far.
    pub best_fitness: f64,
    /// Parameters corresponding to best fitness.
    pub best_params: Vec<f64>,
    /// Total number of fitness evaluations.
    pub eval_count: usize,
    /// Evaluation count at last callback report.
    pub last_report_eval: usize,
    /// Number of task callbacks observed by the solver.
    pub iterations: usize,
    /// Completed solver generations reported by the upstream context.
    pub generations: usize,
    /// Final finite population mean, if all fitness values are finite.
    pub population_mean: Option<f64>,
    /// Final finite population standard deviation.
    pub population_stddev: Option<f64>,
    /// Whether the progress callback stopped the solver before its generation cap.
    pub callback_stopped: bool,
}

impl MhBounded for MHObjective {
    fn bound(&self) -> &[[f64; 2]] {
        self.bounds.as_slice()
    }
}

impl MhObjFunc for MHObjective {
    type Ys = f64;
    fn fitness(&self, xs: &[f64]) -> Self::Ys {
        let fitness_val = compute_fitness_penalties_ref(xs, &self.data);

        // Update callback state if present
        if let Some(ref state_arc) = self.callback_state
            && let Ok(mut state) = state_arc.lock()
        {
            state.eval_count += 1;

            // Track best solution
            if fitness_val < state.best_fitness {
                state.best_fitness = fitness_val;
                state.best_params = xs.to_vec();
            }
        }

        fitness_val
    }
}

/// Create a default callback for metaheuristics that prints progress
pub fn create_mh_callback(
    algo_name: &str,
) -> Box<dyn FnMut(&MHIntermediate) -> CallbackAction + Send> {
    let name = algo_name.to_string();
    let mut tracker = ProgressTracker::default();

    Box::new(move |intermediate: &MHIntermediate| -> CallbackAction {
        let (improvement, _) = tracker.update(intermediate.fun);

        // Print when stalling or periodically
        if tracker.just_started_stalling()
            || tracker.stall_at_interval(25)
            || intermediate.iter.is_multiple_of(10)
        {
            crate::qa_println!(
                "{} iter {:4}  fitness={:.6e} {}",
                name,
                intermediate.iter,
                intermediate.fun,
                improvement
            );
        }

        // Show parameter details every 50 iterations
        if intermediate.iter > 0 && intermediate.iter.is_multiple_of(50) {
            let summary = format_param_summary(intermediate.x.as_slice().unwrap(), 3);
            crate::qa_println!("  --> Best params: {}", summary);
        }

        CallbackAction::Continue
    })
}

/// Optimize filter parameters using metaheuristics algorithms
pub fn optimize_filters_mh(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    mh_name: &str,
    population: usize,
    maxeval: usize,
) -> Result<(String, f64), (String, f64)> {
    optimize_filters_mh_seeded(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        mh_name,
        population,
        maxeval,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn optimize_filters_mh_seeded(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    mh_name: &str,
    population: usize,
    maxeval: usize,
    seed: u64,
) -> Result<(String, f64), (String, f64)> {
    // Create default callback for terminal output
    let callback = create_mh_callback(&format!("mh::{}", mh_name));

    // Delegate to callback version
    optimize_filters_mh_with_callback_seeded(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        mh_name,
        population,
        maxeval,
        callback,
        seed,
    )
}

/// Optimize filter parameters using metaheuristics algorithms with callback support
#[allow(clippy::too_many_arguments)]
pub fn optimize_filters_mh_with_callback(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    mh_name: &str,
    population: usize,
    maxeval: usize,
    callback: Box<dyn FnMut(&MHIntermediate) -> CallbackAction + Send>,
) -> Result<(String, f64), (String, f64)> {
    optimize_filters_mh_with_callback_seeded(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        mh_name,
        population,
        maxeval,
        callback,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn optimize_filters_mh_with_callback_seeded(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    mh_name: &str,
    population: usize,
    maxeval: usize,
    callback: Box<dyn FnMut(&MHIntermediate) -> CallbackAction + Send>,
    seed: u64,
) -> Result<(String, f64), (String, f64)> {
    optimize_filters_mh_with_callback_seeded_report(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        mh_name,
        population,
        maxeval,
        callback,
        seed,
    )
    .0
}

#[allow(clippy::too_many_arguments)]
fn optimize_filters_mh_with_callback_seeded_report(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    mh_name: &str,
    population: usize,
    maxeval: usize,
    mut callback: Box<dyn FnMut(&MHIntermediate) -> CallbackAction + Send>,
    seed: u64,
) -> (Result<(String, f64), (String, f64)>, Option<BackendSearchEvidence>) {
    let num_params = x.len();

    // Build bounds for metaheuristics (as pairs)
    if lower_bounds.len() != num_params || upper_bounds.len() != num_params {
        return (
            Err((
                format!(
                    "Metaheuristics dimension mismatch: x={}, lower={}, upper={}",
                    num_params,
                    lower_bounds.len(),
                    upper_bounds.len()
                ),
                f64::INFINITY,
            )),
            None,
        );
    }
    let mut bounds: Vec<[f64; 2]> = Vec::with_capacity(num_params);
    for i in 0..num_params {
        bounds.push([lower_bounds[i], upper_bounds[i]]);
    }

    // Create objective with penalties (metaheuristics don't support native constraints)
    let mut penalty_data = objective_data.clone();
    let penalty_mode = if mh_name == "pso" {
        PenaltyMode::Pso
    } else {
        PenaltyMode::Standard
    };
    penalty_data.configure_penalties(penalty_mode);

    // Create callback state
    let callback_state = Arc::new(Mutex::new(CallbackState {
        best_fitness: f64::INFINITY,
        best_params: vec![],
        eval_count: 0,
        last_report_eval: 0,
        iterations: 0,
        generations: 0,
        population_mean: None,
        population_stddev: None,
        callback_stopped: false,
    }));

    // Clone for the task closure
    let callback_state_task = Arc::clone(&callback_state);

    // Simple objective function wrapper for metaheuristics
    let mh_obj = MHObjective {
        data: penalty_data,
        bounds,
        callback_state: Some(Arc::clone(&callback_state)),
    };

    // Choose algorithm configuration
    // Use boxed builder to allow runtime selection with unified type
    let builder = match mh_name {
        "de" => MhSolver::build_boxed(MhDe::default(), mh_obj),
        "pso" => {
            // Tuned PSO parameters for this implementation
            // This PSO uses: v = velocity*x + cognition*r1*(pbest-x) + social*r2*(gbest-x)
            // where v becomes the new position (not standard PSO)
            // Balance exploration and exploitation
            let pso_tuned = MhPso::default()
                .cognition(1.0) // Equal personal best influence
                .social(1.5) // Stronger global best attraction
                .velocity(0.9); // Moderate inertia for gradual convergence
            MhSolver::build_boxed(pso_tuned, mh_obj)
        }
        "rga" => {
            // RGA works well for constrained optimization with default parameters
            // Note: RGA benefits from larger populations (recommended: 100+)
            MhSolver::build_boxed(MhRga::default(), mh_obj)
        }
        "tlbo" => MhSolver::build_boxed(MhTlbo, mh_obj),
        "fa" | "firefly" => {
            // Firefly works well for constrained optimization
            // alpha: randomization parameter (exploration)
            // beta_min: minimum attractiveness (exploitation)
            // gamma: light absorption coefficient (distance sensitivity)
            let fa_tuned = MhFa::default()
                .alpha(0.5) // Reduced randomization for more focused search
                .beta_min(1.0) // Keep default attractiveness
                .gamma(0.01); // Keep default absorption
            MhSolver::build_boxed(fa_tuned, mh_obj)
        }
        _ => MhSolver::build_boxed(MhDe::default(), mh_obj),
    };

    // Estimate generations from maxeval and population
    let pop = population.max(1);
    let gens = (maxeval.max(pop)).div_ceil(pop); // ceil(maxeval/pop)

    // Track iteration count
    let mut current_iter = 0_usize;
    let report_interval = 100; // Report every N evaluations

    let solver = builder
        .seed(seed)
        .pop_num(pop)
        .task(move |ctx| {
            current_iter += 1;

            // Report progress periodically
            if let Ok(mut state) = callback_state_task.lock() {
                state.iterations = current_iter;
                state.generations = ctx.r#gen as usize;
                let fitness = &ctx.pool_y;
                state.population_mean = None;
                state.population_stddev = None;
                if !fitness.is_empty() && fitness.iter().all(|value| value.is_finite()) {
                    let mean = fitness.iter().sum::<f64>() / fitness.len() as f64;
                    let variance = fitness.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
                        / fitness.len() as f64;
                    if mean.is_finite() && variance.is_finite() {
                        state.population_mean = Some(mean);
                        state.population_stddev = Some(variance.sqrt());
                    }
                }
                let evals_since_last = state.eval_count.saturating_sub(state.last_report_eval);

                if evals_since_last >= report_interval {
                    // Create intermediate state for callback
                    let x_array = Array1::from(state.best_params.clone());
                    let intermediate = MHIntermediate {
                        x: x_array,
                        fun: state.best_fitness,
                        iter: current_iter,
                    };

                    // Call the callback
                    let action = callback(&intermediate);
                    state.last_report_eval = state.eval_count;

                    // Check if user wants to stop
                    if matches!(action, CallbackAction::Stop) {
                        state.callback_stopped = true;
                        return true; // Signal to stop optimization
                    }
                }
            }

            // Continue until max generations
            current_iter >= gens
        })
        .solve();

    // Write back the best parameters
    let best_xs = solver.as_best_xs();
    if best_xs.len() == x.len() {
        x.copy_from_slice(best_xs);
    }
    let best_val = *solver.as_best_fit();
    let state = match callback_state.lock() {
        Ok(state) => state,
        Err(_) => {
            return (
                Err((
                    format!("Metaheuristics({mh_name}) could not inspect search counters"),
                    best_val,
                )),
                None,
            );
        }
    };
    let search = BackendSearchEvidence {
        completion: if state.callback_stopped {
            OptimizerBackendCompletion::NonConverged
        } else {
            OptimizerBackendCompletion::EvaluationLimit
        },
        stop_cause: if state.callback_stopped {
            BackendSearchStopCause::ProgressCallbackStop
        } else {
            BackendSearchStopCause::GenerationLimit
        },
        evaluations: state.eval_count,
        generations: state.generations,
        generation_limit: gens,
        task_callbacks: state.iterations,
        population_mean: state.population_mean,
        population_stddev: state.population_stddev,
    };
    (
        Ok((format!("Metaheuristics({})", mh_name), best_val)),
        Some(search),
    )
}
