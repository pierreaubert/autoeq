use super::objective_data::ObjectiveData;
use super::objective_data::run_autoeq_de_with_epa_callback;
use super::types::OptimProgressCallback;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerTermination {
    Converged,
    EvaluationLimit,
    NonConverged,
    UserStopped,
    BackendFailure,
    InvalidResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerConfidence {
    High,
    Low,
    Unusable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OptimizerRestartEvidence {
    pub attempt: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub termination: OptimizerTermination,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<f64>,
}

/// Structured evidence for one optimizer invocation.
///
/// Backends retain their historical tuple API, but callers should use this
/// type for production acceptance. In particular, an `Ok` tuple containing
/// "not converged" is classified as best-effort rather than success.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OptimizerRunEvidence {
    pub algorithm: String,
    pub termination: OptimizerTermination,
    pub converged: bool,
    pub best_effort: bool,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_count: Option<usize>,
    pub evaluation_limit: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub max_constraint_violation: f64,
    pub confidence: OptimizerConfidence,
    /// Whether this invocation supplied the parameters used in the emitted
    /// result. Attempts superseded by a better pass/refinement remain in the
    /// report but are not production-acceptance inputs.
    #[serde(default = "default_true")]
    pub selected_for_output: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub restart_history: Vec<OptimizerRestartEvidence>,
}

const fn default_true() -> bool {
    true
}

impl OptimizerRunEvidence {
    #[allow(clippy::too_many_arguments)]
    pub fn from_backend_result(
        algorithm: &str,
        result: Result<(String, f64), (String, f64)>,
        parameters: &[f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        evaluation_limit: usize,
        seed: Option<u64>,
    ) -> Self {
        let backend_accepted = result.is_ok();
        let (status, raw_objective) = match result {
            Ok(value) | Err(value) => value,
        };
        let objective = raw_objective.is_finite().then_some(raw_objective);
        let max_constraint_violation = max_bound_violation(parameters, lower_bounds, upper_bounds);
        let lower_status = status.to_ascii_lowercase();
        let mut termination = if !backend_accepted {
            OptimizerTermination::BackendFailure
        } else if lower_status.contains("stopped") || lower_status.contains("cancelled") {
            OptimizerTermination::UserStopped
        } else if lower_status.contains("not converged")
            && (lower_status.contains("maximum")
                || lower_status.contains("maxeval")
                || lower_status.contains("limit")
                || lower_status.contains("budget"))
        {
            OptimizerTermination::EvaluationLimit
        } else if lower_status.contains("not converged") {
            OptimizerTermination::NonConverged
        } else {
            OptimizerTermination::Converged
        };
        if objective.is_none()
            || !max_constraint_violation.is_finite()
            || max_constraint_violation > 1e-9
        {
            termination = OptimizerTermination::InvalidResult;
        }
        let converged = termination == OptimizerTermination::Converged;
        let best_effort = !converged
            && objective.is_some()
            && max_constraint_violation.is_finite()
            && max_constraint_violation <= 1e-9;
        let confidence = if converged {
            OptimizerConfidence::High
        } else if best_effort {
            OptimizerConfidence::Low
        } else {
            OptimizerConfidence::Unusable
        };
        Self {
            algorithm: algorithm.to_string(),
            termination,
            converged,
            best_effort,
            evaluation_count: parse_evaluation_count(&status),
            evaluation_limit,
            seed,
            status,
            objective,
            max_constraint_violation,
            confidence,
            selected_for_output: true,
            restart_history: Vec::new(),
        }
    }
}

fn parse_evaluation_count(status: &str) -> Option<usize> {
    let start = status.find("nfev=")? + "nfev=".len();
    let digits: String = status[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn max_bound_violation(parameters: &[f64], lower_bounds: &[f64], upper_bounds: &[f64]) -> f64 {
    if parameters.len() != lower_bounds.len() || parameters.len() != upper_bounds.len() {
        return f64::INFINITY;
    }
    parameters
        .iter()
        .zip(lower_bounds)
        .zip(upper_bounds)
        .map(|((&value, &lower), &upper)| (lower - value).max(value - upper).max(0.0))
        .fold(0.0, f64::max)
}

/// Optimize filter parameters using global optimization algorithms
///
/// # Arguments
/// * `x` - Initial parameter vector to optimize (modified in place)
/// * `lower_bounds` - Lower bounds for each parameter
/// * `upper_bounds` - Upper bounds for each parameter
/// * `objective_data` - Data structure containing optimization parameters
/// * `cli_args` - CLI arguments containing algorithm, population, maxeval, and other parameters
///
/// # Returns
/// * Result containing (status, optimal value) or (error, value)
///
/// # Details
/// Dispatches to appropriate library-specific optimizer based on algorithm name.
/// The parameter vector is organized as [freq, Q, gain] triplets for each filter.
pub fn optimize_filters(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
) -> Result<(String, f64), (String, f64)> {
    optimize_filters_with_algo_override(x, lower_bounds, upper_bounds, objective_data, params, None)
}

/// Optimize filters and return structured termination/convergence evidence.
pub fn optimize_filters_detailed(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
) -> OptimizerRunEvidence {
    let result = optimize_filters(x, lower_bounds, upper_bounds, objective_data, params);
    OptimizerRunEvidence::from_backend_result(
        &params.algo,
        result,
        x,
        lower_bounds,
        upper_bounds,
        params.maxeval,
        params.seed,
    )
}

/// Optimize filter parameters with optional algorithm override.
///
/// `algo_override` is used by the local-refine step in
/// [`setup::perform_optimization`] to switch from the global algorithm
/// (`params.algo`) to a local one (`params.local_algo`) without rebuilding
/// the params struct.
pub fn optimize_filters_with_algo_override(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
    algo_override: Option<&str>,
) -> Result<(String, f64), (String, f64)> {
    let algo = algo_override.unwrap_or(&params.algo);
    let backend = super::registry::resolve(algo)
        .ok_or_else(|| (format!("Unknown algorithm: {}", algo), f64::INFINITY))?;
    backend.optimize(x, lower_bounds, upper_bounds, objective_data, params, None)
}

/// Optimize filter parameters with a progress callback for per-iteration updates.
///
/// Backends that report iteration progress (`autoeq:*`, `mh:*`) invoke the
/// callback; NLopt silently drops it. The `autoeq:*` path is specialised
/// here to compute the EPA preference score every 10 iterations and pass
/// it as the third argument of `OptimProgressCallback` — that bookkeeping
/// is loss-specific, so it stays in this dispatcher rather than the
/// generic trait. All other backends go through [`registry::resolve`].
pub fn optimize_filters_with_callback(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
    callback: OptimProgressCallback,
) -> Result<(String, f64), (String, f64)> {
    let backend = super::registry::resolve(&params.algo)
        .ok_or_else(|| (format!("Unknown algorithm: {}", params.algo), f64::INFINITY))?;

    // Specialised EPA-aware path: only meaningful for the AutoEQ DE
    // backend (the only backend that exposes per-iteration `DEIntermediate`
    // states the EPA wrapper consumes).
    //
    // Match by exact name — earlier this checked `library() == "AutoEQ"`,
    // which now also matches `autoeq:cobyla` and `autoeq:isres` and would
    // silently route them through DE instead of the chosen backend.
    if backend.name().eq_ignore_ascii_case("autoeq:de") {
        return run_autoeq_de_with_epa_callback(
            x,
            lower_bounds,
            upper_bounds,
            objective_data,
            params,
            backend.name(),
            callback,
        );
    }

    // Generic path: delegate to the trait. Backends without callback
    // capability (NLopt) silently drop the callback inside `optimize`.
    let cb_for_backend: Option<OptimProgressCallback> = if backend.capabilities().iteration_callback
    {
        Some(callback)
    } else {
        None
    };
    backend.optimize(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        params,
        cb_for_backend,
    )
}

/// Callback variant of [`optimize_filters_detailed`].
pub fn optimize_filters_with_callback_detailed(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
    callback: OptimProgressCallback,
) -> OptimizerRunEvidence {
    let result = optimize_filters_with_callback(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        params,
        callback,
    );
    OptimizerRunEvidence::from_backend_result(
        &params.algo,
        result,
        x,
        lower_bounds,
        upper_bounds,
        params.maxeval,
        params.seed,
    )
}
