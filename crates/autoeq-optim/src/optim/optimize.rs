use super::backend::FilterOptimizerOutput;
use super::constraint_envelope::finalize_candidate;
use super::objective_data::ObjectiveData;
use super::objective_data::run_autoeq_de_with_epa_callback;
use super::run_control::{OptimizerRunControl, OptimizerStageSnapshot};
use super::types::OptimProgressCallback;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Finalize one backend winner through the shared envelope choke-point.
///
/// Every optimizer backend funnels through the three dispatchers below, so
/// finalizing here leaves no bypass: gains repair onto their envelopes, Q
/// enforces the global cap plus the policy-declared local caps, composite
/// breaches refuse, and the returned loss is re-verified at the repaired
/// parameters. Failed backend runs propagate untouched.
fn finalize_dispatch_winner(
    candidate_id: &str,
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    data: &ObjectiveData,
    params: &crate::OptimParams,
    result: Result<(String, f64), (String, f64)>,
) -> Result<(String, f64), (String, f64)> {
    let (algo, _) = result?;
    let failed =
        |reason: String| -> Result<(String, f64), (String, f64)> { Err((reason, f64::INFINITY)) };
    let owned = match super::constraint_envelope::OwnedConstraintSpec::from_params(params) {
        Ok(owned) => owned,
        Err(reason) => return failed(reason),
    };
    let required_spacing = params.min_spacing_oct.max(data.min_spacing_oct);
    let repaired = if super::constraint_envelope::is_peq_layout_loss(data.loss_type) {
        match crate::constraints::project_min_spacing(
            x,
            lower_bounds,
            upper_bounds,
            data.peq_model,
            required_spacing,
        ) {
            Ok(repaired) => repaired,
            Err(reason) => return failed(reason),
        }
    } else {
        x.to_vec()
    };
    let adjusted = repaired != x;
    match finalize_candidate(candidate_id, &repaired, data, &owned.as_spec()) {
        Ok(finalized) => {
            let is_peq = super::constraint_envelope::is_peq_layout_loss(data.loss_type);
            let spacing = if is_peq {
                crate::constraints::viol_spacing_from_xs(
                    &finalized.params,
                    data.peq_model,
                    required_spacing,
                )
            } else {
                0.0
            };
            let ceiling = if is_peq && data.max_db > 0.0 {
                super::compute::compute_ceiling_violation_into(
                    &data.freqs,
                    &finalized.params,
                    data.srate,
                    data.peq_model,
                    data.max_db,
                )
            } else {
                0.0
            };
            let min_gain = if is_peq && data.min_db > 0.0 {
                crate::constraints::viol_min_gain_from_xs(
                    &finalized.params,
                    data.peq_model,
                    data.min_db,
                )
            } else {
                0.0
            };
            let within_bounds = finalized
                .params
                .iter()
                .zip(lower_bounds.iter().zip(upper_bounds))
                .all(|(&value, (&lower, &upper))| {
                    value.is_finite() && value >= lower && value <= upper
                });
            if !within_bounds
                || spacing > 0.0
                || ceiling > 0.0
                || min_gain > 0.0
                || !finalized.loss.is_finite()
            {
                return failed(format!(
                    "spacing repair refused: within_bounds={within_bounds}, spacing={spacing}, ceiling={ceiling}, min_gain={min_gain}"
                ));
            }
            x.copy_from_slice(&finalized.params);
            let status = if adjusted {
                format!("{algo}; minimum-spacing projection applied")
            } else {
                algo
            };
            Ok((status, finalized.loss))
        }
        Err(reason) => failed(reason),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerTermination {
    Converged,
    /// The backend reached its configured evaluation or generation limit.
    /// Actual evaluation and generation counts are reported separately.
    EvaluationLimit,
    NonConverged,
    UserStopped,
    TimedOut,
    BackendFailure,
    InvalidResult,
}

/// Typed completion reported by a backend whose API can distinguish outcomes.
///
/// Legacy backends return a status string only. Their text remains diagnostic
/// and is not treated as proof of convergence or user cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerBackendCompletion {
    /// The solver reached its declared convergence condition.
    Converged,
    /// The solver stopped because its configured evaluation or iteration cap ran out.
    EvaluationLimit,
    /// The solver returned a usable result without claiming convergence.
    NonConverged,
}

/// A budget refusal detected before the optimizer backend is invoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizerBudgetPreflightRefusal {
    /// Requested hard evaluation cap.
    pub requested_evaluations: usize,
    /// Minimum required for one complete backend search unit.
    pub required_evaluations: usize,
}

/// Whether a controlled optimizer invocation entered backend execution.
///
/// Preflight failures are structurally distinct from solver failures. The
/// legacy tuple API still returns its historical error message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerDispatchOutcome {
    /// The registered backend was invoked, whether it succeeded or failed.
    BackendInvoked,
    /// The configured score cap could not admit the backend's minimum complete unit.
    NotStartedBudgetRefusal(OptimizerBudgetPreflightRefusal),
    /// A progress observer was requested but this backend cannot invoke it.
    NotStartedCallbackUnsupported,
    /// An explicit user stop or deadline was already latched before dispatch.
    NotStartedRunStopped,
    /// Resolution or budget-profile validation failed before the backend was invoked.
    NotStartedDispatchFailure,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_input_normalization: Option<roomeq_model::MultiInputNormalizationEvidence>,
    /// Analysis conditioning recorded by the owning preparation path, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_normalization: Option<roomeq_model::InputNormalizationEvidence>,
    pub algorithm: String,
    pub termination: OptimizerTermination,
    pub converged: bool,
    pub best_effort: bool,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<f64>,
    /// Search evaluations from backend status, when available.
    /// Detailed controlled calls replace this with the authoritative run-control
    /// counter; validation/finalization scores remain in the run snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_count: Option<usize>,
    /// Fitness calls counted by the backend objective before finalization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_evaluation_count: Option<usize>,
    /// Solver task stop cause, distinct from the run-control verdict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_stop_cause: Option<super::backend::BackendSearchStopCause>,
    /// Completed solver generations reported by the backend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_count: Option<usize>,
    /// Configured solver task-callback limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_limit: Option<usize>,
    /// Solver task callbacks, including the initial population callback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_callback_count: Option<usize>,
    /// Final finite population fitness mean and standard deviation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_fitness_mean: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_fitness_stddev: Option<f64>,
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
    /// Pareto selection evidence emitted by this exact optimizer invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pareto_report: Option<roomeq_model::ParetoDispatchReport>,
    /// Envelope finalization report for the emitted parameters, when the
    /// emitter ran the shared choke-point. Dispatchers finalize before
    /// returning; engine emission re-checks and attaches the diagnostics
    /// here so refusal evidence survives with the result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_report: Option<super::constraint_envelope::ConstrainedCandidate>,
}

/// Result of a controlled optimizer call with its stop and scoring evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlledOptimizerRun {
    /// Legacy backend result, including its unchanged raw status text.
    pub result: Result<(String, f64), (String, f64)>,
    /// Structured evidence after applying the final run-control snapshot.
    pub evidence: OptimizerRunEvidence,
    /// Counters and stop causes captured after finalization completed.
    pub snapshot: super::run_control::OptimizerRunSnapshot,
    /// Per-stage counters, when this call used a staged run-control view.
    pub stage_snapshot: Option<OptimizerStageSnapshot>,
    /// Structural dispatch outcome, independent of the legacy error string.
    pub dispatch: OptimizerDispatchOutcome,
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
        let invalid_parameters =
            !max_constraint_violation.is_finite() || max_constraint_violation > 1e-9;
        let invalid_success_objective = backend_accepted && objective.is_none();
        let termination = if invalid_parameters || invalid_success_objective {
            OptimizerTermination::InvalidResult
        } else if !backend_accepted {
            OptimizerTermination::BackendFailure
        } else if lower_status.contains("maxevalreached")
            || lower_status.contains("maximum evaluations reached")
            || lower_status.contains("maximum iterations reached")
            || lower_status.contains("evaluation budget exhausted")
            || (lower_status.contains("not converged")
                && (lower_status.contains("maximum")
                    || lower_status.contains("maxeval")
                    || lower_status.contains("limit")
                    || lower_status.contains("budget")))
        {
            OptimizerTermination::EvaluationLimit
        } else {
            // A successful legacy tuple is not structured evidence of
            // convergence, even if its status happens to contain that word.
            OptimizerTermination::NonConverged
        };
        let mut evidence = Self {
            algorithm: algorithm.to_string(),
            input_normalization: None,
            multi_input_normalization: None,
            termination,
            converged: false,
            best_effort: false,
            evaluation_count: parse_evaluation_count(&status),
            backend_evaluation_count: None,
            backend_stop_cause: None,
            generation_count: None,
            generation_limit: None,
            task_callback_count: None,
            population_fitness_mean: None,
            population_fitness_stddev: None,
            evaluation_limit,
            seed,
            status,
            objective,
            max_constraint_violation,
            confidence: OptimizerConfidence::Unusable,
            selected_for_output: true,
            restart_history: Vec::new(),
            constraint_report: None,
            pareto_report: None,
        };
        evidence.refresh_quality_flags();
        evidence
    }

    /// Apply a structured backend completion without interpreting its status text.
    pub fn apply_backend_completion(&mut self, completion: OptimizerBackendCompletion) {
        if matches!(
            self.termination,
            OptimizerTermination::BackendFailure
                | OptimizerTermination::InvalidResult
                | OptimizerTermination::EvaluationLimit
                | OptimizerTermination::UserStopped
                | OptimizerTermination::TimedOut
        ) {
            return;
        }
        self.termination = match completion {
            OptimizerBackendCompletion::Converged => OptimizerTermination::Converged,
            OptimizerBackendCompletion::EvaluationLimit => OptimizerTermination::EvaluationLimit,
            OptimizerBackendCompletion::NonConverged => OptimizerTermination::NonConverged,
        };
        self.refresh_quality_flags();
    }

    fn apply_budget_preflight_refusal(&mut self) {
        if self.termination != OptimizerTermination::InvalidResult {
            self.termination = OptimizerTermination::EvaluationLimit;
            self.refresh_quality_flags();
        }
    }

    /// Refine the termination using typed stop causes and admitted-score counters.
    ///
    /// Backend failures and invalid successful results remain authoritative.
    /// User cancellation takes precedence over a simultaneous deadline; both
    /// flags remain visible in the snapshot. A deadline takes precedence over
    /// score-budget exhaustion, and budget exhaustion takes precedence over a
    /// backend convergence claim.
    pub fn apply_run_control(&mut self, snapshot: &super::run_control::OptimizerRunSnapshot) {
        self.apply_run_control_with_refusals(snapshot, true);
    }

    fn apply_run_control_with_refusals(
        &mut self,
        snapshot: &super::run_control::OptimizerRunSnapshot,
        root_refusals_are_budget_stop: bool,
    ) {
        if matches!(
            self.termination,
            OptimizerTermination::InvalidResult | OptimizerTermination::BackendFailure
        ) {
            return;
        }
        if snapshot.cancellation_requested {
            self.termination = OptimizerTermination::UserStopped;
        } else if self.termination == OptimizerTermination::UserStopped {
            return;
        } else if snapshot.deadline_reached {
            self.termination = OptimizerTermination::TimedOut;
        } else if self.termination == OptimizerTermination::TimedOut {
            return;
        } else if snapshot.budget_exhausted
            || (root_refusals_are_budget_stop && snapshot.evaluations_refused > 0)
        {
            self.termination = OptimizerTermination::EvaluationLimit;
        }
        self.refresh_quality_flags();
    }

    /// Apply the current dispatch's stage-local budget result.
    pub fn apply_stage_run_control(&mut self, snapshot: &OptimizerStageSnapshot) {
        if matches!(
            self.termination,
            OptimizerTermination::InvalidResult
                | OptimizerTermination::BackendFailure
                | OptimizerTermination::UserStopped
                | OptimizerTermination::TimedOut
        ) || snapshot.cancellation_requested
            || snapshot.deadline_reached
        {
            return;
        }
        if snapshot.budget_exhausted || snapshot.evaluations_refused > 0 {
            self.termination = OptimizerTermination::EvaluationLimit;
            self.refresh_quality_flags();
        }
    }

    fn apply_not_started_stop(&mut self, snapshot: &super::run_control::OptimizerRunSnapshot) {
        self.termination = if snapshot.cancellation_requested {
            OptimizerTermination::UserStopped
        } else if snapshot.deadline_reached {
            OptimizerTermination::TimedOut
        } else {
            return;
        };
        self.objective = None;
        self.refresh_quality_flags();
    }

    fn apply_validation_refusal_stop(
        &mut self,
        snapshot: &super::run_control::OptimizerRunSnapshot,
        confirmed_typed_refusal: bool,
    ) {
        if !confirmed_typed_refusal || self.termination != OptimizerTermination::BackendFailure {
            return;
        }
        self.termination = if snapshot.cancellation_requested {
            OptimizerTermination::UserStopped
        } else if snapshot.deadline_reached {
            OptimizerTermination::TimedOut
        } else {
            return;
        };
        self.objective = None;
        self.refresh_quality_flags();
    }

    /// Whether the retained parameters and objective are finite and satisfy the recorded bounds.
    ///
    /// This only reports parameter validity; it does not override stop cause,
    /// confidence, or acceptance policy.
    pub fn has_valid_candidate(&self) -> bool {
        self.objective.is_some_and(f64::is_finite)
            && self.max_constraint_violation.is_finite()
            && self.max_constraint_violation <= 1e-9
    }

    fn refresh_quality_flags(&mut self) {
        self.converged = self.termination == OptimizerTermination::Converged;
        self.best_effort = !self.converged
            && self.termination != OptimizerTermination::UserStopped
            && self.has_valid_candidate();
        self.confidence = if self.converged {
            OptimizerConfidence::High
        } else if self.best_effort {
            OptimizerConfidence::Low
        } else {
            OptimizerConfidence::Unusable
        };
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
        .map(|((&value, &lower), &upper)| {
            // f64::max discards NaNs; reject invalid values before reductions
            // so a backend cannot receive usable evidence for a NaN candidate.
            if !value.is_finite() || !lower.is_finite() || !upper.is_finite() || lower > upper {
                f64::INFINITY
            } else {
                (lower - value).max(value - upper).max(0.0)
            }
        })
        .fold(0.0, f64::max)
}

fn validation_refusal_caused_by_stop(
    backend_result: &Result<(String, f64), (String, f64)>,
    finalized_result: &Result<(String, f64), (String, f64)>,
    parameters: &[f64],
    bounds: (&[f64], &[f64]),
    snapshots: (
        &super::run_control::OptimizerRunSnapshot,
        &super::run_control::OptimizerRunSnapshot,
    ),
    stage_snapshots: (
        Option<&OptimizerStageSnapshot>,
        Option<&OptimizerStageSnapshot>,
    ),
) -> bool {
    let (lower_bounds, upper_bounds) = bounds;
    let (before, after) = snapshots;
    let (before_stage, after_stage) = stage_snapshots;
    let validation_refusal_count_increased = match (before_stage, after_stage) {
        (Some(before), Some(after)) => {
            after.validation_evaluations_refused > before.validation_evaluations_refused
        }
        _ => after.validation_evaluations_refused > before.validation_evaluations_refused,
    };
    let stop_was_latched = after.cancellation_requested
        || after.deadline_reached
        || after_stage.is_some_and(|stage| stage.cancellation_requested || stage.deadline_reached);
    backend_result
        .as_ref()
        .is_ok_and(|(_, loss)| loss.is_finite())
        && max_bound_violation(parameters, lower_bounds, upper_bounds) <= 1e-9
        && finalized_result.is_err()
        && stop_was_latched
        && validation_refusal_count_increased
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

/// Preserve the actual DE completion counters when the ordinary production
/// path selects AutoEQ DE. Other backends retain their legacy result.
pub fn optimize_filters_with_de_completion(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
) -> (Result<(String, f64), (String, f64)>, Option<super::de::DECompletion>) {
    let Some(backend) = super::registry::resolve(&params.algo) else {
        return (optimize_filters(x, lower_bounds, upper_bounds, objective_data, params), None);
    };
    if !backend.name().eq_ignore_ascii_case("autoeq:de") {
        return (optimize_filters(x, lower_bounds, upper_bounds, objective_data, params), None);
    }
    let snapshot = objective_data.clone();
    let (result, completion) = super::de::optimize_filters_autoeq_with_completion(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        backend.name(),
        params,
    );
    (
        finalize_dispatch_winner(
            backend.name(),
            x,
            lower_bounds,
            upper_bounds,
            &snapshot,
            params,
            result,
        ),
        completion,
    )
}

/// Preserve same-invocation search diagnostics on the ordinary MH route.
///
/// This dispatch performs the same solver call and candidate finalization as
/// [`optimize_filters_with_de_completion`]. The extra value is observational:
/// it does not classify a generation-limit result as converged.
pub fn optimize_filters_with_completion_evidence(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
) -> (
    Result<(String, f64), (String, f64)>,
    Option<super::de::DECompletion>,
    Option<super::backend::BackendSearchEvidence>,
) {
    if let Some(backend) = super::registry::resolve(&params.algo)
        && backend.name().starts_with("mh:")
    {
        let snapshot = objective_data.clone();
        let output = backend.optimize_with_report(
            x, lower_bounds, upper_bounds, objective_data, params, None,
        );
        let normalized = normalize_backend_output(output, backend.name(), x);
        return (
            finalize_dispatch_winner(
                &params.algo, x, lower_bounds, upper_bounds, &snapshot, params,
                normalized.result,
            ),
            None,
            normalized.search_evidence,
        );
    }
    let (result, de_completion) = optimize_filters_with_de_completion(
        x, lower_bounds, upper_bounds, objective_data, params,
    );
    (result, de_completion, None)
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

/// Optimize with a hard candidate-evaluation budget and cooperative cancellation.
///
/// The control gate applies to search-time candidate scores across all
/// registered built-in backends. Final constraint realization uses an
/// uncontrolled snapshot so an exhausted search budget cannot invalidate its
/// best candidate. `params.maxeval` is set to the control's budget for the
/// backend invocation.
///
/// # Errors
///
/// Returns an error when the algorithm is unknown, has no budget profile, or
/// the budget cannot fill its minimum complete solver search unit.
pub fn optimize_filters_with_run_control(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
    run_control: &OptimizerRunControl,
) -> Result<(String, f64), (String, f64)> {
    optimize_filters_with_run_control_dispatch(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        params,
        RunControlDispatchOptions {
            algo_override: None,
            callback: None,
            run_control,
        },
    )
    .result
}

struct ControlledOptimizationDispatch {
    result: Result<(String, f64), (String, f64)>,
    dispatch: OptimizerDispatchOutcome,
    evaluation_limit: usize,
    validation_stop_refusal: bool,
    pareto_report: Option<roomeq_model::ParetoDispatchReport>,
    search_evidence: Option<super::backend::BackendSearchEvidence>,
}

struct NormalizedBackendOutput {
    result: Result<(String, f64), (String, f64)>,
    pareto_report: Option<roomeq_model::ParetoDispatchReport>,
    validation_stop_refusal: bool,
    search_evidence: Option<super::backend::BackendSearchEvidence>,
}

fn normalize_backend_output(
    output: FilterOptimizerOutput,
    expected_backend: &str,
    parameters: &[f64],
) -> NormalizedBackendOutput {
    match output {
        FilterOptimizerOutput::StoppedDuringValidation { reason } => NormalizedBackendOutput {
            result: Err((reason, f64::INFINITY)),
            pareto_report: None,
            validation_stop_refusal: true,
            search_evidence: None,
        },
        FilterOptimizerOutput::CompletedWithSearchEvidence { result, search } => {
            NormalizedBackendOutput {
                result,
                pareto_report: None,
                validation_stop_refusal: false,
                search_evidence: Some(search),
            }
        }
        FilterOptimizerOutput::Completed {
            result,
            pareto_report,
        } => {
            if result.is_err() && pareto_report.is_some() {
                return NormalizedBackendOutput {
                    result: Err((
                        "optimizer backend attached Pareto evidence to a failed result".to_string(),
                        f64::INFINITY,
                    )),
                    pareto_report: None,
                    validation_stop_refusal: false,
                    search_evidence: None,
                };
            }
            let Some(report) = pareto_report.as_ref() else {
                return NormalizedBackendOutput {
                    result,
                    pareto_report: None,
                    validation_stop_refusal: false,
                    search_evidence: None,
                };
            };
            let invalid = report.validate().err().or_else(|| {
                (report.backend != expected_backend).then(|| {
                    "optimizer Pareto report backend does not match the resolved backend".to_string()
                })
            }).or_else(|| {
                (report.returned_parameters != parameters).then(|| {
                    "optimizer Pareto report does not match its returned parameters".to_string()
                })
            }).or_else(|| {
                let selected_scalar = report
                    .candidates
                    .get(report.selection.selected_candidate_index)
                    .and_then(|candidate| candidate.scalar_loss);
                result.as_ref().ok().and_then(|(_, loss)| {
                    selected_scalar
                        .filter(|scalar| scalar != loss)
                        .map(|_| "optimizer Pareto report selected scalar loss does not match the returned score".to_string())
                })
            });
            match invalid {
                Some(reason) => NormalizedBackendOutput {
                    result: Err((
                        format!("optimizer returned malformed Pareto evidence: {reason}"),
                        f64::INFINITY,
                    )),
                    pareto_report: None,
                    validation_stop_refusal: false,
                    search_evidence: None,
                },
                None => NormalizedBackendOutput {
                    result,
                    pareto_report,
                    validation_stop_refusal: false,
                    search_evidence: None,
                },
            }
        }
    }
}

struct RunControlDispatchOptions<'a> {
    algo_override: Option<&'a str>,
    callback: Option<OptimProgressCallback>,
    run_control: &'a OptimizerRunControl,
}

fn optimize_filters_with_run_control_dispatch(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
    options: RunControlDispatchOptions<'_>,
) -> ControlledOptimizationDispatch {
    let RunControlDispatchOptions {
        algo_override,
        mut callback,
        run_control,
    } = options;
    let evaluation_limit = run_control.effective_evaluation_limit();
    let not_started = |message: String, dispatch| ControlledOptimizationDispatch {
        result: Err((message, f64::INFINITY)),
        dispatch,
        evaluation_limit,
        validation_stop_refusal: false,
        pareto_report: None,
        search_evidence: None,
    };
    let algorithm = algo_override.unwrap_or(&params.algo);
    let Some(backend) = super::registry::resolve(algorithm) else {
        return not_started(
            format!("Unknown algorithm: {algorithm}"),
            OptimizerDispatchOutcome::NotStartedDispatchFailure,
        );
    };
    let mut controlled_params = params.clone();
    controlled_params.algo = algorithm.to_owned();
    // Some backend profiles clamp the configured value to a positive solver
    // minimum. A zero effective cap is still refused below, before scoring.
    controlled_params.maxeval = evaluation_limit.max(1);
    let supports_callback =
        backend.supports_iteration_callback(&controlled_params, &objective_data);
    if callback.is_some() && !supports_callback {
        return not_started(
            format!(
                "{} cannot provide requested optimizer progress callbacks",
                backend.name()
            ),
            OptimizerDispatchOutcome::NotStartedCallbackUnsupported,
        );
    }
    let root_before_dispatch = run_control.snapshot();
    if root_before_dispatch.cancellation_requested || root_before_dispatch.deadline_reached {
        return not_started(
            "optimizer stop was requested before backend dispatch".to_owned(),
            OptimizerDispatchOutcome::NotStartedRunStopped,
        );
    }
    let Some(profile) =
        backend.evaluation_budget_profile(lower_bounds, upper_bounds, &controlled_params)
    else {
        return not_started(
            format!(
                "{} does not report a matched objective-evaluation budget profile",
                backend.name()
            ),
            OptimizerDispatchOutcome::NotStartedDispatchFailure,
        );
    };
    if profile.requested_evaluations != controlled_params.maxeval {
        return not_started(
            format!(
                "{} budget profile reports {} evaluations for a control cap of {}",
                backend.name(),
                profile.requested_evaluations,
                controlled_params.maxeval
            ),
            OptimizerDispatchOutcome::NotStartedDispatchFailure,
        );
    }
    if evaluation_limit < profile.minimum_complete_batch {
        let refusal = OptimizerBudgetPreflightRefusal {
            requested_evaluations: evaluation_limit,
            required_evaluations: profile.minimum_complete_batch,
        };
        return ControlledOptimizationDispatch {
            result: Err((
                format!(
                    "{} requires at least {} objective evaluations for one complete search unit; requested budget is {}",
                    backend.name(),
                    refusal.required_evaluations,
                    refusal.requested_evaluations
                ),
                f64::INFINITY,
            )),
            dispatch: OptimizerDispatchOutcome::NotStartedBudgetRefusal(refusal),
            evaluation_limit,
            validation_stop_refusal: false,
            pareto_report: None,
            search_evidence: None,
        };
    }

    let validation_snapshot = objective_data.with_validation_tracking(run_control.clone());
    let controlled_objective = objective_data.with_run_control(run_control.clone());
    let control_for_callback = run_control.clone();
    let backend_callback: Option<OptimProgressCallback> = if supports_callback {
        Some(Box::new(move |iteration, loss, preference| {
            if control_for_callback.stop_requested() {
                return crate::de::CallbackAction::Stop;
            }
            if let Some(observer) = callback.as_mut() {
                let action = observer(iteration, loss, preference);
                match action {
                    crate::de::CallbackAction::Continue => {}
                    crate::de::CallbackAction::Stop => {
                        control_for_callback.request_cancel();
                        return crate::de::CallbackAction::Stop;
                    }
                }
            }
            if control_for_callback.stop_requested() {
                crate::de::CallbackAction::Stop
            } else {
                crate::de::CallbackAction::Continue
            }
        }))
    } else {
        None
    };
    let backend_output = backend.optimize_with_report(
        x,
        lower_bounds,
        upper_bounds,
        controlled_objective,
        &controlled_params,
        backend_callback,
    );
    let NormalizedBackendOutput {
        result: backend_result,
        pareto_report,
        validation_stop_refusal: typed_validation_stop,
        search_evidence,
    } = normalize_backend_output(backend_output, backend.name(), x);
    let snapshot = run_control.snapshot();
    let stage_snapshot_before_finalization = run_control.stage_snapshot();
    let stop_before_finalization = snapshot.cancellation_requested
        || snapshot.deadline_reached
        || typed_validation_stop
        || stage_snapshot_before_finalization
            .as_ref()
            .is_some_and(|stage| stage.cancellation_requested || stage.deadline_reached);
    let (finalized, validation_stop_refusal) = if stop_before_finalization {
        (backend_result, typed_validation_stop)
    } else {
        let finalized = finalize_dispatch_winner(
            backend.name(),
            x,
            lower_bounds,
            upper_bounds,
            &validation_snapshot,
            &controlled_params,
            backend_result.clone(),
        );
        let snapshot_after_finalization = run_control.snapshot();
        let stage_snapshot_after_finalization = run_control.stage_snapshot();
        let validation_stop_refusal = validation_refusal_caused_by_stop(
            &backend_result,
            &finalized,
            x,
            (lower_bounds, upper_bounds),
            (&snapshot, &snapshot_after_finalization),
            (
                stage_snapshot_before_finalization.as_ref(),
                stage_snapshot_after_finalization.as_ref(),
            ),
        );
        (finalized, validation_stop_refusal)
    };
    let pareto_report = pareto_report.filter(|report| {
        finalized.is_ok()
            && !snapshot.cancellation_requested
            && !snapshot.deadline_reached
            && report.returned_parameters == x
    });
    ControlledOptimizationDispatch {
        result: finalized,
        dispatch: OptimizerDispatchOutcome::BackendInvoked,
        evaluation_limit,
        validation_stop_refusal,
        pareto_report,
        search_evidence,
    }
}

/// Optimize under run control and return both the backend tuple and typed evidence.
///
/// The evidence preserves the backend's raw status and applies explicit user,
/// deadline and admitted-score stop causes from the control snapshot. A
/// minimum-budget refusal is returned as a typed not-started outcome rather
/// than inferred from the legacy error string. Legacy backend strings alone
/// never imply convergence.
pub fn optimize_filters_with_run_control_detailed(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
    run_control: &OptimizerRunControl,
) -> ControlledOptimizerRun {
    optimize_filters_with_run_control_and_algo_override_detailed(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        params,
        None,
        None,
        run_control,
    )
}

/// Optimize under shared run control with an optional algorithm override and progress observer.
///
/// The override is resolved through the same registry and budget profile as
/// the configured algorithm. A requested observer is refused before scoring
/// when the selected backend cannot report iteration progress. The root
/// snapshot remains cumulative; stage-local counts are attached separately.
///
/// # Errors
///
/// The returned tuple is an error when resolution, budget profiling, callback
/// support, or backend execution fails. The typed `dispatch` field identifies
/// preflight failures without parsing the legacy error string.
#[expect(
    clippy::too_many_arguments,
    reason = "the public dispatch keeps objective, bounds, algorithm override, observer, and shared run-control explicit"
)]
pub fn optimize_filters_with_run_control_and_algo_override_detailed(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
    algo_override: Option<&str>,
    callback: Option<OptimProgressCallback>,
    run_control: &OptimizerRunControl,
) -> ControlledOptimizerRun {
    let dispatch_result = optimize_filters_with_run_control_dispatch(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        params,
        RunControlDispatchOptions {
            algo_override,
            callback,
            run_control,
        },
    );
    let result = dispatch_result.result;
    let dispatch = dispatch_result.dispatch;
    let evaluation_limit = dispatch_result.evaluation_limit;
    let validation_stop_refusal = dispatch_result.validation_stop_refusal;
    let pareto_report = dispatch_result.pareto_report;
    let search_evidence = dispatch_result.search_evidence;
    let snapshot = run_control.snapshot();
    let stage_snapshot = run_control.stage_snapshot();
    let algorithm = algo_override.unwrap_or(&params.algo);
    let mut evidence = OptimizerRunEvidence::from_backend_result(
        algorithm,
        result.clone(),
        x,
        lower_bounds,
        upper_bounds,
        evaluation_limit,
        params.seed,
    );
    evidence.evaluation_count = Some(
        stage_snapshot.map_or(snapshot.evaluations_started, |stage| {
            stage.evaluations_started
        }),
    );
    if let Some(search) = search_evidence {
        evidence.backend_evaluation_count = Some(search.evaluations);
        evidence.backend_stop_cause = Some(search.stop_cause);
        evidence.generation_count = Some(search.generations);
        evidence.generation_limit = Some(search.generation_limit);
        evidence.task_callback_count = Some(search.task_callbacks);
        evidence.population_fitness_mean = search.population_mean;
        evidence.population_fitness_stddev = search.population_stddev;
        if result.is_ok() {
            evidence.apply_backend_completion(search.completion);
        }
    }
    if matches!(
        dispatch,
        OptimizerDispatchOutcome::NotStartedBudgetRefusal(_)
    ) {
        evidence.apply_budget_preflight_refusal();
    }
    if let Some(stage_snapshot) = &stage_snapshot {
        evidence.apply_run_control_with_refusals(&snapshot, false);
        evidence.apply_stage_run_control(stage_snapshot);
    } else {
        evidence.apply_run_control(&snapshot);
    }
    if dispatch == OptimizerDispatchOutcome::NotStartedRunStopped {
        evidence.apply_not_started_stop(&snapshot);
    }
    if validation_stop_refusal {
        evidence.apply_validation_refusal_stop(&snapshot, true);
    }
    evidence.pareto_report = if result.is_ok()
        && evidence.termination != OptimizerTermination::UserStopped
        && evidence.termination != OptimizerTermination::TimedOut
    {
        pareto_report.filter(|report| report.returned_parameters == x)
    } else {
        None
    };
    ControlledOptimizerRun {
        result,
        evidence,
        snapshot,
        stage_snapshot,
        dispatch,
    }
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
    let snapshot = objective_data.clone();
    let result = backend.optimize(x, lower_bounds, upper_bounds, objective_data, params, None);
    finalize_dispatch_winner(algo, x, lower_bounds, upper_bounds, &snapshot, params, result)
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
        let snapshot = objective_data.clone();
        let result = run_autoeq_de_with_epa_callback(
            x,
            lower_bounds,
            upper_bounds,
            objective_data,
            params,
            backend.name(),
            callback,
        );
        return finalize_dispatch_winner(
            backend.name(), x, lower_bounds, upper_bounds, &snapshot, params, result,
        );
    }

    if backend.capabilities().iteration_callback
        && !backend.supports_iteration_callback(params, &objective_data)
    {
        return Err((
            format!(
                "{} cannot provide requested optimizer progress callbacks in this mode",
                backend.name()
            ),
            f64::INFINITY,
        ));
    }

    // Generic path: delegate to the trait. Backends without callback
    // capability (NLopt) silently drop the callback inside `optimize`.
    let cb_for_backend: Option<OptimProgressCallback> = if backend.capabilities().iteration_callback
    {
        Some(callback)
    } else {
        None
    };
    let snapshot = objective_data.clone();
    let result = backend.optimize(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        params,
        cb_for_backend,
    );
    finalize_dispatch_winner(
        backend.name(), x, lower_bounds, upper_bounds, &snapshot, params, result,
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

#[cfg(test)]
mod evidence_validation_tests {
    use super::*;

    fn pareto_report(parameters: Vec<f64>) -> roomeq_model::ParetoDispatchReport {
        roomeq_model::ParetoDispatchReport {
            schema: "roomeq.pareto_dispatch/v1".into(),
            backend: "autoeq:nsga2".into(),
            submitted_count: 1,
            refused_source_indices: Vec::new(),
            candidates: vec![roomeq_model::ParetoCandidateEvidence {
                source_index: 0,
                search_parameters: parameters.clone(),
                search_objectives: vec![1.0],
                validated_parameters: parameters.clone(),
                validated_objectives: vec![1.0],
                rank: Some(0),
                crowding_distance: Some(roomeq_model::ParetoCrowdingDistance::Unbounded),
                scalar_loss: Some(1.0),
            }],
            selection: roomeq_model::ParetoSelectionEvidence {
                rule: "normalized_compromise".into(),
                weights: vec![1.0],
                ideal: vec![1.0],
                nadir: vec![1.0],
                selected_candidate_index: 0,
                selected_source_index: 0,
                scalar_baseline_rule: Some("configured_scalar:single".into()),
                scalar_best_source_index: Some(0),
                scalar_best_loss: Some(1.0),
            },
            returned_parameters: parameters,
            search_evaluations: Some(16),
            generations: Some(1),
        }
    }

    #[test]
    fn backend_output_accepts_only_a_valid_report_matching_the_returned_vector() {
        let parameters = vec![0.25];
        let valid = normalize_backend_output(
            FilterOptimizerOutput::Completed {
                result: Ok(("done".into(), 1.0)),
                pareto_report: Some(pareto_report(parameters.clone())),
            },
            "autoeq:nsga2",
            &parameters,
        );
        assert!(valid.result.is_ok());
        assert_eq!(valid.pareto_report.unwrap().returned_parameters, parameters);
        assert!(!valid.validation_stop_refusal);

        let mismatched = normalize_backend_output(
            FilterOptimizerOutput::Completed {
                result: Ok(("done".into(), 1.0)),
                pareto_report: Some(pareto_report(vec![0.5])),
            },
            "autoeq:nsga2",
            &[0.25],
        );
        assert!(mismatched.result.is_err());
        assert!(mismatched.pareto_report.is_none());
        assert!(!mismatched.validation_stop_refusal);

        let wrong_backend = normalize_backend_output(
            FilterOptimizerOutput::Completed {
                result: Ok(("done".into(), 2.0)),
                pareto_report: Some(pareto_report(parameters.clone())),
            },
            "autoeq:bo",
            &parameters,
        );
        assert!(
            wrong_backend
                .result
                .unwrap_err()
                .0
                .contains("resolved backend")
        );
        assert!(wrong_backend.pareto_report.is_none());

        let wrong_score = normalize_backend_output(
            FilterOptimizerOutput::Completed {
                result: Ok(("done".into(), 2.0)),
                pareto_report: Some(pareto_report(parameters.clone())),
            },
            "autoeq:nsga2",
            &parameters,
        );
        assert!(wrong_score.result.unwrap_err().0.contains("scalar loss"));
        assert!(wrong_score.pareto_report.is_none());
    }

    #[test]
    fn backend_output_never_keeps_reports_on_errors_or_typed_stops() {
        let parameters = vec![0.25];
        let failed = normalize_backend_output(
            FilterOptimizerOutput::Completed {
                result: Err(("real backend failure".into(), f64::INFINITY)),
                pareto_report: Some(pareto_report(parameters.clone())),
            },
            "autoeq:nsga2",
            &parameters,
        );
        assert!(failed.result.is_err());
        assert!(
            failed
                .result
                .unwrap_err()
                .0
                .contains("attached Pareto evidence")
        );
        assert!(failed.pareto_report.is_none());
        assert!(!failed.validation_stop_refusal);

        let stopped = normalize_backend_output(
            FilterOptimizerOutput::StoppedDuringValidation {
                reason: "cancelled during front validation".into(),
            },
            "autoeq:nsga2",
            &parameters,
        );
        assert!(stopped.result.is_err());
        assert!(stopped.pareto_report.is_none());
        assert!(stopped.validation_stop_refusal);
    }

    #[test]
    fn converged_backend_cannot_authorize_nonfinite_parameters_or_invalid_bounds() {
        for (parameters, lower, upper) in [
            (vec![f64::NAN], vec![0.0], vec![1.0]),
            (vec![f64::INFINITY], vec![0.0], vec![1.0]),
            (vec![0.5], vec![f64::NAN], vec![1.0]),
            (vec![0.5], vec![0.0], vec![f64::NAN]),
            (vec![0.5], vec![1.0], vec![0.0]),
            (vec![0.5], vec![f64::NEG_INFINITY], vec![f64::INFINITY]),
            (vec![0.5], vec![], vec![1.0]),
        ] {
            let evidence = OptimizerRunEvidence::from_backend_result(
                "autoeq:de",
                Ok(("converged nfev=10".into(), 0.0)),
                &parameters,
                &lower,
                &upper,
                10,
                Some(7),
            );
            assert_eq!(evidence.termination, OptimizerTermination::InvalidResult);
            assert_eq!(evidence.confidence, OptimizerConfidence::Unusable);
            assert!(!evidence.converged && !evidence.best_effort);
            assert!(evidence.max_constraint_violation.is_infinite());
        }
    }

    #[test]
    fn valid_budget_limited_candidate_retains_best_effort_evidence() {
        let evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Ok(("maximum evaluations reached nfev=10".into(), 0.5)),
            &[0.0, 1.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            10,
            Some(7),
        );
        assert_eq!(evidence.termination, OptimizerTermination::EvaluationLimit);
        assert!(evidence.best_effort);
        assert_eq!(evidence.max_constraint_violation, 0.0);
    }

    #[test]
    fn failed_result_with_invalid_parameters_is_invalid_result() {
        let evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Err(("solver returned an invalid candidate".into(), f64::INFINITY)),
            &[f64::NAN],
            &[0.0],
            &[1.0],
            10,
            Some(7),
        );

        assert_eq!(evidence.termination, OptimizerTermination::InvalidResult);
        assert!(!evidence.has_valid_candidate());
        assert!(!evidence.best_effort);
    }

    #[test]
    fn failed_result_with_valid_candidate_and_infinite_sentinel_is_backend_failure() {
        let evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Err(("solver failed independently".into(), f64::INFINITY)),
            &[0.5],
            &[0.0],
            &[1.0],
            10,
            Some(7),
        );

        assert_eq!(evidence.termination, OptimizerTermination::BackendFailure);
        assert!(!evidence.has_valid_candidate());
        assert!(!evidence.best_effort);
    }

    #[test]
    fn validation_stop_refusal_maps_only_a_valid_backend_winner_to_stop_evidence() {
        use super::super::run_control::{EvaluationStage, OptimizerRunControl};
        use std::num::NonZeroUsize;

        let parameters = [0.5];
        let lower = [0.0];
        let upper = [1.0];
        let backend_winner = Ok(("backend best".to_owned(), 0.25));
        let refused_finalization = Err(("finalizer score refused".to_owned(), f64::INFINITY));

        for deadline in [false, true] {
            let control = OptimizerRunControl::new(NonZeroUsize::new(3).unwrap());
            let before = control.snapshot();
            if deadline {
                control.request_deadline();
            } else {
                control.request_cancel();
            }
            assert!(
                control
                    .begin_evaluation(EvaluationStage::Validation, 1)
                    .is_none()
            );
            let after = control.snapshot();

            assert!(validation_refusal_caused_by_stop(
                &backend_winner,
                &refused_finalization,
                &parameters,
                (&lower, &upper),
                (&before, &after),
                (None, None),
            ));
            let mut evidence = OptimizerRunEvidence::from_backend_result(
                "autoeq:cobra",
                refused_finalization.clone(),
                &parameters,
                &lower,
                &upper,
                3,
                Some(1),
            );
            assert_eq!(evidence.termination, OptimizerTermination::BackendFailure);
            evidence.apply_validation_refusal_stop(&after, true);
            assert_eq!(
                evidence.termination,
                if deadline {
                    OptimizerTermination::TimedOut
                } else {
                    OptimizerTermination::UserStopped
                }
            );

            let real_backend_failure = Err(("solver failed before finalization".to_owned(), 1.0));
            assert!(!validation_refusal_caused_by_stop(
                &real_backend_failure,
                &refused_finalization,
                &parameters,
                (&lower, &upper),
                (&before, &after),
                (None, None),
            ));
            let mut failure = OptimizerRunEvidence::from_backend_result(
                "autoeq:cobra",
                real_backend_failure,
                &parameters,
                &lower,
                &upper,
                3,
                Some(1),
            );
            failure.apply_validation_refusal_stop(&after, false);
            assert_eq!(failure.termination, OptimizerTermination::BackendFailure);

            let invalid_parameters = [f64::NAN];
            assert!(!validation_refusal_caused_by_stop(
                &backend_winner,
                &refused_finalization,
                &invalid_parameters,
                (&lower, &upper),
                (&before, &after),
                (None, None),
            ));
            let mut invalid = OptimizerRunEvidence::from_backend_result(
                "autoeq:cobra",
                refused_finalization.clone(),
                &invalid_parameters,
                &lower,
                &upper,
                3,
                Some(1),
            );
            assert_eq!(invalid.termination, OptimizerTermination::InvalidResult);
            invalid.apply_validation_refusal_stop(&after, true);
            assert_eq!(invalid.termination, OptimizerTermination::InvalidResult);
        }
    }

    #[test]
    fn validation_refusal_in_another_stage_does_not_reclassify_finalizer_failure() {
        use super::super::run_control::{EvaluationStage, OptimizerRunControl};
        use std::num::NonZeroUsize;

        let root = OptimizerRunControl::new(NonZeroUsize::new(3).unwrap());
        let finalizing_stage = root.with_stage_budget(NonZeroUsize::new(2).unwrap());
        let unrelated_stage = root.with_stage_budget(NonZeroUsize::new(2).unwrap());
        let before_root = finalizing_stage.snapshot();
        let before_stage = finalizing_stage.stage_snapshot();

        root.request_cancel();
        assert!(
            unrelated_stage
                .begin_evaluation(EvaluationStage::Validation, 1)
                .is_none()
        );

        let after_root = finalizing_stage.snapshot();
        let after_stage = finalizing_stage.stage_snapshot();
        assert_eq!(after_root.validation_evaluations_refused, 1);
        assert_eq!(
            after_stage.as_ref().unwrap().validation_evaluations_refused,
            0
        );
        assert!(!validation_refusal_caused_by_stop(
            &Ok(("backend best".to_owned(), 0.25)),
            &Err(("finalizer failed".to_owned(), f64::INFINITY)),
            &[0.5],
            (&[0.0], &[1.0]),
            (&before_root, &after_root),
            (before_stage.as_ref(), after_stage.as_ref()),
        ));
    }
}

#[cfg(test)]
mod staged_run_control_tests {
    use super::*;
    use crate::Curve;
    use crate::cli::Args;
    use clap::Parser;
    use ndarray::Array1;
    use std::num::NonZeroUsize;

    fn scalar_fixture() -> (
        ObjectiveData,
        crate::OptimParams,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        let mut args = Args::parse_from(["autoeq"]);
        args.num_filters = 1;
        args.population = 6;
        args.maxeval = 60;
        args.seed = Some(1);
        args.min_freq = 20.0;
        args.max_freq = 20_000.0;
        args.min_db = -12.0;
        args.max_db = 12.0;
        let params = crate::OptimParams::from(&args);
        let frequencies = Array1::from(vec![
            20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 10_240.0,
        ]);
        let input = Curve {
            freq: frequencies.clone(),
            spl: Array1::from_elem(frequencies.len(), 5.0),
            ..Default::default()
        };
        let target = Curve {
            freq: frequencies.clone(),
            spl: Array1::zeros(frequencies.len()),
            ..Default::default()
        };
        let deviation = Curve {
            freq: frequencies,
            spl: Array1::from_elem(10, 5.0),
            ..Default::default()
        };
        let (objective, _) =
            super::super::setup::setup_objective_data(&params, &input, &target, &deviation, &None)
                .expect("valid fixture objective");
        let (lower, upper) = super::super::setup::setup_bounds(&params);
        let initial = super::super::setup::initial_guess(&params, &lower, &upper);
        (objective, params, lower, upper, initial)
    }

    #[test]
    fn dispatch_limits_are_remaining_global_and_stage_budgets() {
        let (objective, mut params, lower, upper, mut initial) = scalar_fixture();
        params.algo = "autoeq:cobra".to_owned();
        params.maxeval = 60;
        let root = OptimizerRunControl::new(NonZeroUsize::new(30).unwrap());

        let first_stage = root.with_stage_budget(NonZeroUsize::new(12).unwrap());
        assert_eq!(first_stage.remaining_evaluations(), 12);
        let first = optimize_filters_with_run_control_detailed(
            &mut initial,
            &lower,
            &upper,
            objective.clone(),
            &params,
            &first_stage,
        );
        assert!(first.result.is_ok(), "{:?}", first.result);
        assert_eq!(first.evidence.evaluation_limit, 12);
        assert_eq!(first.evidence.evaluation_count, Some(12));
        assert_eq!(first.stage_snapshot.unwrap().evaluations_started, 12);
        assert_eq!(first.snapshot.evaluations_started, 12);

        params.algo = "autoeq:cobyla".to_owned();
        let second_stage = root.with_stage_budget(NonZeroUsize::new(25).unwrap());
        assert_eq!(second_stage.effective_evaluation_limit(), 18);
        let second = optimize_filters_with_run_control_and_algo_override_detailed(
            &mut initial,
            &lower,
            &upper,
            objective,
            &params,
            Some("autoeq:cobra"),
            None,
            &second_stage,
        );
        assert!(second.result.is_ok(), "{:?}", second.result);
        assert_eq!(second.evidence.algorithm, "autoeq:cobra");
        assert_eq!(second.evidence.evaluation_limit, 18);
        assert_eq!(second.evidence.evaluation_count, Some(18));
        assert_eq!(second.stage_snapshot.unwrap().evaluations_started, 18);
        assert_eq!(second.snapshot.evaluations_started, 30);
        assert_eq!(second.snapshot.evaluation_budget, 30);
    }

    #[test]
    fn stage_preflight_uses_effective_cap_and_refuses_before_scoring() {
        let (objective, mut params, lower, upper, mut initial) = scalar_fixture();
        params.algo = "autoeq:de".to_owned();
        params.population = 6;
        let root = OptimizerRunControl::new(NonZeroUsize::new(20).unwrap());
        let stage = root.with_stage_budget(NonZeroUsize::new(1).unwrap());

        let run = optimize_filters_with_run_control_detailed(
            &mut initial,
            &lower,
            &upper,
            objective,
            &params,
            &stage,
        );

        let OptimizerDispatchOutcome::NotStartedBudgetRefusal(refusal) = run.dispatch else {
            panic!(
                "expected stage-budget preflight refusal, got {:?}",
                run.dispatch
            );
        };
        assert_eq!(refusal.requested_evaluations, 1);
        assert!(refusal.required_evaluations > 1);
        assert_eq!(run.snapshot.evaluations_started, 0);
        assert_eq!(run.stage_snapshot.unwrap().evaluations_started, 0);
        assert_eq!(run.evidence.evaluation_count, Some(0));
        assert_eq!(run.evidence.evaluation_limit, 1);
    }

    #[test]
    fn callbackless_backend_refuses_observer_without_scoring() {
        let (objective, mut params, lower, upper, mut initial) = scalar_fixture();
        params.algo = "autoeq:cobyla".to_owned();
        let control = OptimizerRunControl::new(NonZeroUsize::new(20).unwrap());
        let callback: OptimProgressCallback =
            Box::new(|_, _, _| crate::de::CallbackAction::Continue);

        let run = optimize_filters_with_run_control_and_algo_override_detailed(
            &mut initial,
            &lower,
            &upper,
            objective,
            &params,
            None,
            Some(callback),
            &control,
        );

        assert_eq!(
            run.dispatch,
            OptimizerDispatchOutcome::NotStartedCallbackUnsupported
        );
        assert_eq!(run.snapshot.evaluations_started, 0);
        assert_eq!(run.snapshot.validation_evaluations_started, 0);
    }

    #[test]
    fn bo_callback_capability_tracks_actual_objective_mode() {
        let (objective, mut params, _, _, _) = scalar_fixture();
        let backend = super::super::registry::resolve("autoeq:bo").unwrap();
        params.bo_ehvi = true;
        assert!(backend.supports_iteration_callback(&params, &objective));
        let mut multi = objective.clone();
        multi.multi_objective = Some(super::super::types::MultiObjectiveData {
            objectives: vec![objective.clone(), objective],
            strategy: crate::roomeq::MultiMeasurementStrategy::WeightedSum,
            weights: vec![0.5, 0.5],
            variance_lambda: 0.0,
            uncertainty_cvar_alpha: None,
        });
        assert!(!backend.supports_iteration_callback(&params, &multi));
        params.bo_ehvi = false;
        assert!(backend.supports_iteration_callback(&params, &multi));
    }

    #[test]
    fn bo_ehvi_observer_refuses_all_entry_points_before_scoring() {
        let (mut objective, mut params, lower, upper, mut initial) = scalar_fixture();
        params.algo = "autoeq:bo".into();
        params.bo_ehvi = true;
        objective.multi_objective = Some(super::super::types::MultiObjectiveData {
            objectives: vec![objective.clone(), objective.clone()],
            strategy: crate::roomeq::MultiMeasurementStrategy::WeightedSum,
            weights: vec![0.5, 0.5],
            variance_lambda: 0.0,
            uncertainty_cvar_alpha: None,
        });
        let control = OptimizerRunControl::new(NonZeroUsize::new(60).unwrap());
        let before = initial.clone();
        let observer = || -> OptimProgressCallback {
            Box::new(|_, _, _| panic!("unsupported EHVI observer must not run"))
        };
        let run = optimize_filters_with_run_control_and_algo_override_detailed(
            &mut initial,
            &lower,
            &upper,
            objective.clone(),
            &params,
            None,
            Some(observer()),
            &control,
        );
        assert_eq!(
            run.dispatch,
            OptimizerDispatchOutcome::NotStartedCallbackUnsupported
        );
        let error = optimize_filters_with_callback(
            &mut initial,
            &lower,
            &upper,
            objective.clone().with_run_control(control.clone()),
            &params,
            observer(),
        )
        .unwrap_err();
        assert!(
            error
                .0
                .contains("cannot provide requested optimizer progress callbacks")
        );
        let backend = super::super::registry::resolve("autoeq:bo").unwrap();
        let error = backend
            .optimize(
                &mut initial,
                &lower,
                &upper,
                objective.with_run_control(control.clone()),
                &params,
                Some(observer()),
            )
            .unwrap_err();
        assert!(
            error
                .0
                .contains("cannot provide requested optimizer progress callbacks")
        );
        assert_eq!(initial, before);
        let snapshot = control.snapshot();
        assert_eq!(snapshot.evaluations_started, 0);
        assert_eq!(snapshot.validation_evaluations_started, 0);
    }

    #[test]
    fn observer_stop_latches_cancel_and_skips_finalization_scores() {
        let (objective, mut params, lower, upper, mut initial) = scalar_fixture();
        params.algo = "autoeq:cobra".to_owned();
        params.maxeval = 30;
        let control = OptimizerRunControl::new(NonZeroUsize::new(30).unwrap());
        let callback: OptimProgressCallback = Box::new(|_, _, _| crate::de::CallbackAction::Stop);

        let run = optimize_filters_with_run_control_and_algo_override_detailed(
            &mut initial,
            &lower,
            &upper,
            objective,
            &params,
            None,
            Some(callback),
            &control,
        );

        assert!(run.snapshot.cancellation_requested);
        assert_eq!(run.snapshot.validation_evaluations_started, 0);
        assert_eq!(run.evidence.termination, OptimizerTermination::UserStopped);
        assert!(run.snapshot.evaluations_started > 0);
        assert_eq!(run.dispatch, OptimizerDispatchOutcome::BackendInvoked);
    }

    #[test]
    fn prelatched_stop_returns_before_backend_and_finalization_scoring() {
        let (objective, mut params, lower, upper, mut initial) = scalar_fixture();
        params.algo = "autoeq:cobra".to_owned();
        let control = OptimizerRunControl::new(NonZeroUsize::new(20).unwrap());
        control.request_cancel();

        let run = optimize_filters_with_run_control_detailed(
            &mut initial,
            &lower,
            &upper,
            objective,
            &params,
            &control,
        );

        assert_eq!(run.dispatch, OptimizerDispatchOutcome::NotStartedRunStopped);
        assert!(run.result.is_err());
        assert_eq!(run.evidence.termination, OptimizerTermination::UserStopped);
        assert_eq!(run.evidence.evaluation_count, Some(0));
        assert_eq!(run.snapshot.evaluations_started, 0);
        assert_eq!(run.snapshot.validation_evaluations_started, 0);
    }
}
