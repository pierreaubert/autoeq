//! Shared search budgets across RoomEQ passes and local refinement.

use super::optimize::EqOptimizationResult;
use autoeq_optim::OptimParams;
use autoeq_optim::optim::run_control::{
    OptimizerRunControl, OptimizerRunSnapshot, OptimizerStageSnapshot,
};
use autoeq_optim::optim::{
    ObjectiveData, OptimProgressCallback, OptimizerBackend, OptimizerDispatchOutcome,
    OptimizerRunEvidence, OptimizerTermination,
};
use std::cell::RefCell;
use std::error::Error;
use std::num::NonZeroUsize;

/// Evidence and counters captured when one optimizer stage returned.
#[derive(Debug, Clone)]
pub struct EqOptimizerStageRecord {
    /// Typed backend and termination evidence for this stage.
    pub evidence: OptimizerRunEvidence,
    /// Whether the backend was invoked or refused before search.
    pub dispatch: OptimizerDispatchOutcome,
    /// Shared cumulative counters at the optimizer boundary.
    pub snapshot: OptimizerRunSnapshot,
    /// Counters for this stage, including its optimizer finalization.
    pub stage_snapshot: Option<OptimizerStageSnapshot>,
}

/// RoomEQ output with a shared search cap and stage accounting.
#[derive(Debug, Clone)]
pub struct ControlledEqOptimizationResult {
    /// Filters, realized loss, and existing RoomEQ evidence.
    pub result: EqOptimizationResult,
    /// Final shared counters, including pipeline validation after search.
    pub snapshot: OptimizerRunSnapshot,
    /// Ordered optimizer invocations and preflight refusals.
    pub stages: Vec<EqOptimizerStageRecord>,
}

/// A controlled RoomEQ request failed without emitting a filter result.
#[derive(Debug, Clone)]
pub struct ControlledEqError {
    /// Explanation from preparation, dispatch, or result validation.
    pub reason: String,
    details: Box<ControlledEqErrorDetails>,
}

#[derive(Debug, Clone)]
struct ControlledEqErrorDetails {
    snapshot: OptimizerRunSnapshot,
    stages: Vec<EqOptimizerStageRecord>,
}

impl ControlledEqError {
    /// Return shared counters after the failed synchronous request returned.
    pub fn snapshot(&self) -> OptimizerRunSnapshot {
        self.details.snapshot
    }

    /// Return completed optimizer boundaries, including any refused stage.
    pub fn stages(&self) -> &[EqOptimizerStageRecord] {
        &self.details.stages
    }
}

impl std::fmt::Display for ControlledEqError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.reason)
    }
}
impl Error for ControlledEqError {}

pub(super) struct EqRunControl<'a> {
    pub control: &'a OptimizerRunControl,
    stage_budget: NonZeroUsize,
    stages: RefCell<Vec<EqOptimizerStageRecord>>,
}

impl<'a> EqRunControl<'a> {
    pub fn new(control: &'a OptimizerRunControl, stage_budget: NonZeroUsize) -> Self {
        Self {
            control,
            stage_budget,
            stages: RefCell::new(Vec::new()),
        }
    }

    pub fn check_terminal(&self) -> Result<(), Box<dyn Error>> {
        let snapshot = self.control.snapshot();
        if snapshot.cancellation_requested {
            return Err("controlled EQ cancelled before output".into());
        }
        if snapshot.deadline_reached {
            return Err("controlled EQ deadline reached before output".into());
        }
        Ok(())
    }

    pub fn validation_objective(&self, objective: &ObjectiveData) -> ObjectiveData {
        objective.with_validation_tracking(self.control.clone())
    }

    pub fn finish(
        self,
        result: Result<EqOptimizationResult, Box<dyn Error>>,
    ) -> Result<ControlledEqOptimizationResult, ControlledEqError> {
        let result = result.and_then(|result| self.check_terminal().map(|()| result));
        let snapshot = self.control.snapshot();
        let stages = self.stages.into_inner();
        match result {
            Ok(result) => Ok(ControlledEqOptimizationResult {
                result,
                snapshot,
                stages,
            }),
            Err(error) => Err(ControlledEqError {
                reason: error.to_string(),
                details: Box::new(ControlledEqErrorDetails { snapshot, stages }),
            }),
        }
    }
}

/// A stage produced no usable candidate because its minimum batch did not fit.
#[derive(Debug)]
pub(super) struct EqBudgetRefusal(pub OptimizerRunEvidence);
impl std::fmt::Display for EqBudgetRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0.status)
    }
}
impl Error for EqBudgetRefusal {}

pub(super) fn no_search_remaining(control: Option<&EqRunControl<'_>>) -> bool {
    control.is_some_and(|control| control.control.remaining_evaluations() == 0)
}

pub(super) fn validation_objective(
    objective: &ObjectiveData,
    control: Option<&EqRunControl<'_>>,
) -> ObjectiveData {
    control.map_or_else(
        || objective.clone(),
        |control| control.validation_objective(objective),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "preserves the optimizer boundary inputs"
)]
pub(super) fn run_optimizer(
    backend: &dyn OptimizerBackend,
    x: &mut [f64],
    lower: &[f64],
    upper: &[f64],
    objective: ObjectiveData,
    params: &OptimParams,
    algo_override: Option<&str>,
    callback: Option<OptimProgressCallback>,
    control: Option<&EqRunControl<'_>>,
) -> Result<OptimizerRunEvidence, Box<dyn Error>> {
    if let Some(control) = control {
        control.check_terminal()?;
        let stage_control = control.control.with_stage_budget(control.stage_budget);
        let run = backend.optimize_filters_controlled(
            x,
            lower,
            upper,
            objective,
            params,
            algo_override,
            callback,
            &stage_control,
        )?;
        control.stages.borrow_mut().push(EqOptimizerStageRecord {
            evidence: run.evidence.clone(),
            dispatch: run.dispatch,
            snapshot: run.snapshot,
            stage_snapshot: run.stage_snapshot,
        });
        control.check_terminal()?;
        if matches!(
            run.dispatch,
            OptimizerDispatchOutcome::NotStartedBudgetRefusal(_)
        ) {
            return Err(Box::new(EqBudgetRefusal(run.evidence)));
        }
        if matches!(
            run.evidence.termination,
            OptimizerTermination::UserStopped | OptimizerTermination::TimedOut
        ) {
            return Err(run.evidence.status.into());
        }
        return Ok(run.evidence);
    }
    let result = if algo_override.is_some() {
        backend.optimize_filters_with_algo_override(
            x,
            lower,
            upper,
            objective,
            params,
            algo_override,
        )
    } else if let Some(callback) = callback {
        backend.optimize_filters_with_callback(x, lower, upper, objective, params, callback)
    } else {
        backend.optimize_filters(x, lower, upper, objective, params)
    };
    Ok(OptimizerRunEvidence::from_backend_result(
        algo_override.unwrap_or(&params.algo),
        result,
        x,
        lower,
        upper,
        params.maxeval,
        params.seed,
    ))
}
