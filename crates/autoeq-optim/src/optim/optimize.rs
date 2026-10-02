use super::constraint_envelope::finalize_candidate;
use super::objective_data::ObjectiveData;
use super::objective_data::run_autoeq_de_with_epa_callback;
use super::run_control::OptimizerRunControl;
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
    match finalize_candidate(candidate_id, x, data, &owned.as_spec()) {
        Ok(finalized) => {
            x.copy_from_slice(&finalized.params);
            Ok((algo, finalized.loss))
        }
        Err(reason) => failed(reason),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerTermination {
    Converged,
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
            evaluation_limit,
            seed,
            status,
            objective,
            max_constraint_violation,
            confidence: OptimizerConfidence::Unusable,
            selected_for_output: true,
            restart_history: Vec::new(),
            constraint_report: None,
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
        } else if snapshot.budget_exhausted || snapshot.evaluations_refused > 0 {
            self.termination = OptimizerTermination::EvaluationLimit;
        }
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
        run_control,
    )
    .0
}

type ControlledOptimizationDispatch = (
    Result<(String, f64), (String, f64)>,
    OptimizerDispatchOutcome,
);

fn optimize_filters_with_run_control_dispatch(
    x: &mut [f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    objective_data: ObjectiveData,
    params: &crate::OptimParams,
    run_control: &OptimizerRunControl,
) -> ControlledOptimizationDispatch {
    let not_started = |message: String| {
        (
            Err((message, f64::INFINITY)),
            OptimizerDispatchOutcome::NotStartedDispatchFailure,
        )
    };
    let Some(backend) = super::registry::resolve(&params.algo) else {
        return not_started(format!("Unknown algorithm: {}", params.algo));
    };
    let mut controlled_params = params.clone();
    controlled_params.maxeval = run_control.evaluation_budget();
    let Some(profile) =
        backend.evaluation_budget_profile(lower_bounds, upper_bounds, &controlled_params)
    else {
        return not_started(format!(
            "{} does not report a matched objective-evaluation budget profile",
            backend.name()
        ));
    };
    if profile.requested_evaluations != run_control.evaluation_budget() {
        return not_started(format!(
            "{} budget profile reports {} evaluations for a control cap of {}",
            backend.name(),
            profile.requested_evaluations,
            run_control.evaluation_budget()
        ));
    }
    if run_control.evaluation_budget() < profile.minimum_complete_batch {
        let refusal = OptimizerBudgetPreflightRefusal {
            requested_evaluations: run_control.evaluation_budget(),
            required_evaluations: profile.minimum_complete_batch,
        };
        return (
            Err((
                format!(
                    "{} requires at least {} objective evaluations for one complete search unit; requested budget is {}",
                    backend.name(),
                    refusal.required_evaluations,
                    refusal.requested_evaluations
                ),
                f64::INFINITY,
            )),
            OptimizerDispatchOutcome::NotStartedBudgetRefusal(refusal),
        );
    }

    let validation_snapshot = objective_data.with_validation_tracking(run_control.clone());
    let controlled_objective = objective_data.with_run_control(run_control.clone());
    let control_for_callback = run_control.clone();
    let cancellation_callback: OptimProgressCallback = Box::new(move |_, _, _| {
        if control_for_callback.stop_requested() {
            crate::de::CallbackAction::Stop
        } else {
            crate::de::CallbackAction::Continue
        }
    });
    let callback = backend
        .capabilities()
        .iteration_callback
        .then_some(cancellation_callback);
    let result = backend.optimize(
        x,
        lower_bounds,
        upper_bounds,
        controlled_objective,
        &controlled_params,
        callback,
    );
    (
        finalize_dispatch_winner(
            backend.name(),
            x,
            &validation_snapshot,
            &controlled_params,
            result,
        ),
        OptimizerDispatchOutcome::BackendInvoked,
    )
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
    let (result, dispatch) = optimize_filters_with_run_control_dispatch(
        x,
        lower_bounds,
        upper_bounds,
        objective_data,
        params,
        run_control,
    );
    let snapshot = run_control.snapshot();
    let mut evidence = OptimizerRunEvidence::from_backend_result(
        &params.algo,
        result.clone(),
        x,
        lower_bounds,
        upper_bounds,
        run_control.evaluation_budget(),
        params.seed,
    );
    evidence.evaluation_count = Some(snapshot.evaluations_started);
    if matches!(
        dispatch,
        OptimizerDispatchOutcome::NotStartedBudgetRefusal(_)
    ) {
        evidence.apply_budget_preflight_refusal();
    }
    evidence.apply_run_control(&snapshot);
    ControlledOptimizerRun {
        result,
        evidence,
        snapshot,
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
    finalize_dispatch_winner(algo, x, &snapshot, params, result)
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
        return finalize_dispatch_winner(backend.name(), x, &snapshot, params, result);
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
    finalize_dispatch_winner(backend.name(), x, &snapshot, params, result)
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
}
