//! Hard objective-evaluation budgets and cooperative stop requests.

use std::num::NonZeroUsize;
use std::sync::{Arc, Condvar, Mutex};

/// Counts one complete candidate evaluation, including every measurement loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizerBudgetProfile {
    /// Requested objective evaluations in the controlled run.
    pub requested_evaluations: usize,
    /// Evaluation cap configured inside the selected solver.
    pub solver_evaluation_limit: Option<usize>,
    /// Minimum evaluations needed for the backend's smallest complete run unit.
    pub minimum_complete_batch: usize,
    /// Number of candidate scores in the solver's initial batch.
    pub initial_batch_size: usize,
    /// Number of candidate scores in a later generation or batch, if applicable.
    pub generation_batch_size: Option<usize>,
    /// Effective solver population, if the algorithm uses one.
    pub population_size: Option<usize>,
    /// Effective generation limit, if the algorithm uses one.
    pub generation_limit: Option<usize>,
}

impl OptimizerBudgetProfile {
    /// Build a profile from the settings used by a concrete backend.
    pub const fn new(
        requested_evaluations: usize,
        solver_evaluation_limit: Option<usize>,
        minimum_complete_batch: usize,
        initial_batch_size: usize,
        generation_batch_size: Option<usize>,
        population_size: Option<usize>,
        generation_limit: Option<usize>,
    ) -> Self {
        Self {
            requested_evaluations,
            solver_evaluation_limit,
            minimum_complete_batch,
            initial_batch_size,
            generation_batch_size,
            population_size,
            generation_limit,
        }
    }
}

/// Thread-safe gate for objective evaluations in one optimizer run.
#[derive(Clone)]
pub struct OptimizerRunControl {
    inner: Arc<RunControlInner>,
    stage: Option<Arc<RunStageInner>>,
}

/// Separates solver search work from safety/result validation work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationStage {
    /// Candidate scores requested by the selected backend, subject to the cap.
    Search,
    /// Finalization and explicit post-run checks, counted outside the cap.
    Validation,
}

struct RunControlInner {
    budget: NonZeroUsize,
    state: Mutex<RunState>,
    idle: Condvar,
}

struct RunStageInner {
    budget: NonZeroUsize,
    state: Mutex<RunStageState>,
}

#[derive(Default)]
struct RunStageState {
    evaluations_started: usize,
    evaluations_completed: usize,
    evaluations_failed: usize,
    component_evaluations_started: usize,
    component_evaluations_completed: usize,
    evaluations_refused: usize,
    evaluations_in_flight: usize,
    validation_evaluations_started: usize,
    validation_evaluations_completed: usize,
    validation_evaluations_failed: usize,
    validation_evaluations_refused: usize,
    validation_component_evaluations_started: usize,
    validation_component_evaluations_completed: usize,
    validation_evaluations_in_flight: usize,
}

#[derive(Default)]
struct RunState {
    evaluations_started: usize,
    evaluations_completed: usize,
    evaluations_failed: usize,
    component_evaluations_started: usize,
    component_evaluations_completed: usize,
    evaluations_refused: usize,
    evaluations_in_flight: usize,
    validation_evaluations_started: usize,
    validation_evaluations_completed: usize,
    validation_evaluations_failed: usize,
    validation_evaluations_refused: usize,
    validation_component_evaluations_started: usize,
    validation_component_evaluations_completed: usize,
    validation_evaluations_in_flight: usize,
    cancellation_requested: bool,
    deadline_reached: bool,
}

/// Point-in-time objective-evaluation and cancellation accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizerRunSnapshot {
    /// Hard cap on complete candidate evaluations.
    pub evaluation_budget: usize,
    /// Candidate evaluations admitted by the gate.
    pub evaluations_started: usize,
    /// Candidate evaluations that returned from the objective function.
    pub evaluations_completed: usize,
    /// Candidate evaluations whose objective function unwound with a panic.
    pub evaluations_failed: usize,
    /// Per-measurement objective components admitted by the gate.
    pub component_evaluations_started: usize,
    /// Per-measurement objective components that returned.
    pub component_evaluations_completed: usize,
    /// Candidate evaluations rejected after cancellation, deadline, or budget exhaustion.
    pub evaluations_refused: usize,
    /// Candidate evaluations still scoring when the snapshot was taken.
    pub evaluations_in_flight: usize,
    /// Finalizer and post-run candidate scores outside the search cap.
    pub validation_evaluations_started: usize,
    /// Validation scores that returned.
    pub validation_evaluations_completed: usize,
    /// Validation scores whose objective function unwound with a panic.
    pub validation_evaluations_failed: usize,
    /// Validation scores refused because cancellation or the deadline was latched.
    pub validation_evaluations_refused: usize,
    /// Measurement components scored during finalization and post-run checks.
    pub validation_component_evaluations_started: usize,
    /// Validation measurement components that returned.
    pub validation_component_evaluations_completed: usize,
    /// Validation scores active when the snapshot was taken.
    pub validation_evaluations_in_flight: usize,
    /// Whether an explicit user cancellation was requested.
    pub cancellation_requested: bool,
    /// Whether the run's time deadline elapsed.
    pub deadline_reached: bool,
    /// Whether the hard search-score budget has been fully admitted.
    pub budget_exhausted: bool,
}

/// Point-in-time accounting for one stage sharing a run-wide control.
///
/// Clones of a staged control share this quota and its counters. Creating a
/// second stage with [`OptimizerRunControl::with_stage_budget`] creates fresh
/// stage counters while retaining the same root budget and stop flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizerStageSnapshot {
    /// Hard cap for candidate evaluations in this stage.
    pub evaluation_budget: usize,
    /// Candidate evaluations admitted by the root and stage gates.
    pub evaluations_started: usize,
    /// Candidate evaluations that returned from the objective function.
    pub evaluations_completed: usize,
    /// Candidate evaluations whose objective function unwound with a panic.
    pub evaluations_failed: usize,
    /// Per-measurement objective components admitted by the gates.
    pub component_evaluations_started: usize,
    /// Per-measurement objective components that returned.
    pub component_evaluations_completed: usize,
    /// Candidate evaluations refused by either the root or stage gate.
    pub evaluations_refused: usize,
    /// Candidate evaluations still scoring when the snapshot was taken.
    pub evaluations_in_flight: usize,
    /// Validation scores made by this stage, outside both search caps.
    pub validation_evaluations_started: usize,
    /// Validation scores that returned.
    pub validation_evaluations_completed: usize,
    /// Validation scores whose objective function unwound with a panic.
    pub validation_evaluations_failed: usize,
    /// Validation scores refused because cancellation or the deadline was latched.
    pub validation_evaluations_refused: usize,
    /// Measurement components scored during this stage's validation work.
    pub validation_component_evaluations_started: usize,
    /// Validation measurement components that returned.
    pub validation_component_evaluations_completed: usize,
    /// Validation scores active when the snapshot was taken.
    pub validation_evaluations_in_flight: usize,
    /// Whether an explicit user cancellation was requested for the root run.
    pub cancellation_requested: bool,
    /// Whether the root run's deadline elapsed.
    pub deadline_reached: bool,
    /// Whether this stage's score cap has been fully admitted.
    pub budget_exhausted: bool,
}

impl OptimizerRunControl {
    /// Create a run gate with a positive objective-evaluation limit.
    pub fn new(evaluation_budget: NonZeroUsize) -> Self {
        Self {
            inner: Arc::new(RunControlInner {
                budget: evaluation_budget,
                state: Mutex::new(RunState::default()),
                idle: Condvar::new(),
            }),
            stage: None,
        }
    }

    /// Create a fresh stage quota that shares this run's global cap and stop flags.
    ///
    /// Clones of the returned control share the new stage quota. Calling this
    /// method again creates another independent stage quota against the same
    /// remaining root budget.
    pub fn with_stage_budget(&self, stage_budget: NonZeroUsize) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            stage: Some(Arc::new(RunStageInner {
                budget: stage_budget,
                state: Mutex::new(RunStageState::default()),
            })),
        }
    }

    /// Return the hard candidate-evaluation limit.
    pub fn evaluation_budget(&self) -> usize {
        self.inner.budget.get()
    }

    /// Return the score capacity remaining in both the root run and this stage.
    pub fn remaining_evaluations(&self) -> usize {
        let state = self.lock_state();
        let root_remaining = self
            .inner
            .budget
            .get()
            .saturating_sub(state.evaluations_started);
        self.stage.as_ref().map_or(root_remaining, |stage| {
            let stage_state = stage
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            root_remaining.min(
                stage
                    .budget
                    .get()
                    .saturating_sub(stage_state.evaluations_started),
            )
        })
    }

    /// Return the maximum search-score cap to configure for the next dispatch.
    ///
    /// This is the remaining root cap, limited by the remaining stage quota
    /// when this control represents a stage. A fresh, unstaged control returns
    /// its original cap, preserving single-dispatch behavior.
    pub fn effective_evaluation_limit(&self) -> usize {
        self.remaining_evaluations()
    }

    /// Read this control's stage counters, if it was created as a stage view.
    pub fn stage_snapshot(&self) -> Option<OptimizerStageSnapshot> {
        let root_state = self.lock_state();
        self.stage.as_ref().map(|stage| {
            let stage_state = stage
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            stage_snapshot(
                stage.budget.get(),
                &stage_state,
                root_state.cancellation_requested,
                root_state.deadline_reached,
            )
        })
    }

    /// Whether the gate is closed by a user stop, deadline, or budget exhaustion.
    pub fn stop_requested(&self) -> bool {
        let state = self.lock_state();
        let root_stopped = state.cancellation_requested
            || state.deadline_reached
            || state.evaluations_started >= self.inner.budget.get();
        root_stopped
            || self.stage.as_ref().is_some_and(|stage| {
                let stage_state = stage
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                stage_state.evaluations_started >= stage.budget.get()
            })
    }

    /// Record an explicit user cancellation and close the gate.
    pub fn request_cancel(&self) {
        let mut state = self.lock_state();
        state.cancellation_requested = true;
        if state.evaluations_in_flight + state.validation_evaluations_in_flight == 0 {
            self.inner.idle.notify_all();
        }
    }

    /// Record that the run deadline elapsed and close the gate.
    pub fn request_deadline(&self) {
        let mut state = self.lock_state();
        state.deadline_reached = true;
        if state.evaluations_in_flight + state.validation_evaluations_in_flight == 0 {
            self.inner.idle.notify_all();
        }
    }

    /// Record explicit user cancellation, close the gate, and wait for active scores.
    ///
    /// The optimizer worker may still be unwinding its own loop; callers must
    /// also join or await that worker before reporting the run as finished.
    pub fn cancel_and_wait_for_evaluations(&self) -> OptimizerRunSnapshot {
        let mut state = self.lock_state();
        state.cancellation_requested = true;
        while state.evaluations_in_flight + state.validation_evaluations_in_flight != 0 {
            state = self
                .inner
                .idle
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        snapshot(self.inner.budget.get(), &state)
    }

    /// Read current counters without closing the evaluation gate.
    pub fn snapshot(&self) -> OptimizerRunSnapshot {
        let state = self.lock_state();
        snapshot(self.inner.budget.get(), &state)
    }

    pub(crate) fn begin_evaluation(
        &self,
        stage: EvaluationStage,
        objective_components: usize,
    ) -> Option<ObjectiveEvaluationGuard> {
        let mut state = self.lock_state();
        let mut stage_state = self.stage.as_ref().map(|stage| {
            stage
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });
        let objective_components = objective_components.max(1);
        match stage {
            EvaluationStage::Search => {
                if state.cancellation_requested
                    || state.deadline_reached
                    || state.evaluations_started >= self.inner.budget.get()
                    || self.stage.as_ref().is_some_and(|stage| {
                        stage_state
                            .as_ref()
                            .is_some_and(|state| state.evaluations_started >= stage.budget.get())
                    })
                {
                    state.evaluations_refused = state.evaluations_refused.saturating_add(1);
                    if let Some(stage_state) = stage_state.as_mut() {
                        stage_state.evaluations_refused =
                            stage_state.evaluations_refused.saturating_add(1);
                    }
                    return None;
                }
                state.evaluations_started += 1;
                state.component_evaluations_started = state
                    .component_evaluations_started
                    .saturating_add(objective_components);
                state.evaluations_in_flight += 1;
                if let Some(stage_state) = stage_state.as_mut() {
                    stage_state.evaluations_started += 1;
                    stage_state.component_evaluations_started = stage_state
                        .component_evaluations_started
                        .saturating_add(objective_components);
                    stage_state.evaluations_in_flight += 1;
                }
            }
            EvaluationStage::Validation => {
                if state.cancellation_requested || state.deadline_reached {
                    state.validation_evaluations_refused =
                        state.validation_evaluations_refused.saturating_add(1);
                    if let Some(stage_state) = stage_state.as_mut() {
                        stage_state.validation_evaluations_refused =
                            stage_state.validation_evaluations_refused.saturating_add(1);
                    }
                    return None;
                }
                state.validation_evaluations_started += 1;
                state.validation_component_evaluations_started = state
                    .validation_component_evaluations_started
                    .saturating_add(objective_components);
                state.validation_evaluations_in_flight += 1;
                if let Some(stage_state) = stage_state.as_mut() {
                    stage_state.validation_evaluations_started += 1;
                    stage_state.validation_component_evaluations_started = stage_state
                        .validation_component_evaluations_started
                        .saturating_add(objective_components);
                    stage_state.validation_evaluations_in_flight += 1;
                }
            }
        }
        Some(ObjectiveEvaluationGuard {
            control: self.clone(),
            stage,
            objective_components,
        })
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, RunState> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl std::fmt::Debug for OptimizerRunControl {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OptimizerRunControl")
            .field("snapshot", &self.snapshot())
            .field("stage_snapshot", &self.stage_snapshot())
            .finish()
    }
}

pub(crate) struct ObjectiveEvaluationGuard {
    control: OptimizerRunControl,
    stage: EvaluationStage,
    objective_components: usize,
}

impl Drop for ObjectiveEvaluationGuard {
    fn drop(&mut self) {
        let mut state = self.control.lock_state();
        let mut stage_state = self.control.stage.as_ref().map(|stage| {
            stage
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });
        let panicking = std::thread::panicking();
        match self.stage {
            EvaluationStage::Search => {
                if panicking {
                    state.evaluations_failed += 1;
                } else {
                    state.evaluations_completed += 1;
                    state.component_evaluations_completed = state
                        .component_evaluations_completed
                        .saturating_add(self.objective_components);
                }
                state.evaluations_in_flight = state.evaluations_in_flight.saturating_sub(1);
                if let Some(stage_state) = stage_state.as_mut() {
                    if panicking {
                        stage_state.evaluations_failed += 1;
                    } else {
                        stage_state.evaluations_completed += 1;
                        stage_state.component_evaluations_completed = stage_state
                            .component_evaluations_completed
                            .saturating_add(self.objective_components);
                    }
                    stage_state.evaluations_in_flight =
                        stage_state.evaluations_in_flight.saturating_sub(1);
                }
            }
            EvaluationStage::Validation => {
                if panicking {
                    state.validation_evaluations_failed += 1;
                } else {
                    state.validation_evaluations_completed += 1;
                    state.validation_component_evaluations_completed = state
                        .validation_component_evaluations_completed
                        .saturating_add(self.objective_components);
                }
                state.validation_evaluations_in_flight =
                    state.validation_evaluations_in_flight.saturating_sub(1);
                if let Some(stage_state) = stage_state.as_mut() {
                    if panicking {
                        stage_state.validation_evaluations_failed += 1;
                    } else {
                        stage_state.validation_evaluations_completed += 1;
                        stage_state.validation_component_evaluations_completed = stage_state
                            .validation_component_evaluations_completed
                            .saturating_add(self.objective_components);
                    }
                    stage_state.validation_evaluations_in_flight = stage_state
                        .validation_evaluations_in_flight
                        .saturating_sub(1);
                }
            }
        }
        if state.evaluations_in_flight + state.validation_evaluations_in_flight == 0 {
            self.control.inner.idle.notify_all();
        }
    }
}

fn stage_snapshot(
    budget: usize,
    state: &RunStageState,
    cancellation_requested: bool,
    deadline_reached: bool,
) -> OptimizerStageSnapshot {
    OptimizerStageSnapshot {
        evaluation_budget: budget,
        evaluations_started: state.evaluations_started,
        evaluations_completed: state.evaluations_completed,
        evaluations_failed: state.evaluations_failed,
        component_evaluations_started: state.component_evaluations_started,
        component_evaluations_completed: state.component_evaluations_completed,
        evaluations_refused: state.evaluations_refused,
        evaluations_in_flight: state.evaluations_in_flight,
        validation_evaluations_started: state.validation_evaluations_started,
        validation_evaluations_completed: state.validation_evaluations_completed,
        validation_evaluations_failed: state.validation_evaluations_failed,
        validation_evaluations_refused: state.validation_evaluations_refused,
        validation_component_evaluations_started: state.validation_component_evaluations_started,
        validation_component_evaluations_completed: state
            .validation_component_evaluations_completed,
        validation_evaluations_in_flight: state.validation_evaluations_in_flight,
        cancellation_requested,
        deadline_reached,
        budget_exhausted: state.evaluations_started >= budget,
    }
}

fn snapshot(budget: usize, state: &RunState) -> OptimizerRunSnapshot {
    OptimizerRunSnapshot {
        evaluation_budget: budget,
        evaluations_started: state.evaluations_started,
        evaluations_completed: state.evaluations_completed,
        evaluations_failed: state.evaluations_failed,
        component_evaluations_started: state.component_evaluations_started,
        component_evaluations_completed: state.component_evaluations_completed,
        evaluations_refused: state.evaluations_refused,
        evaluations_in_flight: state.evaluations_in_flight,
        validation_evaluations_started: state.validation_evaluations_started,
        validation_evaluations_completed: state.validation_evaluations_completed,
        validation_evaluations_failed: state.validation_evaluations_failed,
        validation_evaluations_refused: state.validation_evaluations_refused,
        validation_component_evaluations_started: state.validation_component_evaluations_started,
        validation_component_evaluations_completed: state
            .validation_component_evaluations_completed,
        validation_evaluations_in_flight: state.validation_evaluations_in_flight,
        cancellation_requested: state.cancellation_requested,
        deadline_reached: state.deadline_reached,
        budget_exhausted: state.evaluations_started >= budget,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Barrier, mpsc};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn budget_is_a_hard_limit_and_records_refused_calls() {
        let control = OptimizerRunControl::new(NonZeroUsize::new(2).unwrap());
        for _ in 0..2 {
            let _guard = control
                .begin_evaluation(EvaluationStage::Search, 3)
                .expect("budget admits two scores");
        }
        assert!(
            control
                .begin_evaluation(EvaluationStage::Search, 3)
                .is_none()
        );
        let snapshot = control.snapshot();
        assert_eq!(snapshot.evaluations_started, 2);
        assert_eq!(snapshot.evaluations_completed, 2);
        assert_eq!(snapshot.component_evaluations_started, 6);
        assert_eq!(snapshot.component_evaluations_completed, 6);
        assert_eq!(snapshot.evaluations_refused, 1);
        assert!(snapshot.budget_exhausted);
    }

    #[test]
    fn unstaged_control_keeps_the_original_single_dispatch_cap() {
        let control = OptimizerRunControl::new(NonZeroUsize::new(3).unwrap());
        assert_eq!(control.effective_evaluation_limit(), 3);
        assert!(control.stage_snapshot().is_none());
        for _ in 0..2 {
            drop(
                control
                    .begin_evaluation(EvaluationStage::Search, 1)
                    .unwrap(),
            );
        }
        assert_eq!(control.effective_evaluation_limit(), 1);
        drop(
            control
                .begin_evaluation(EvaluationStage::Search, 1)
                .unwrap(),
        );
        assert_eq!(control.effective_evaluation_limit(), 0);
        assert!(control.stage_snapshot().is_none());
    }

    #[test]
    fn cancel_closes_gate_and_waits_for_active_scores() {
        let control = OptimizerRunControl::new(NonZeroUsize::new(4).unwrap());
        let guard = control
            .begin_evaluation(EvaluationStage::Search, 1)
            .unwrap();
        let worker_control = control.clone();
        let (done_sender, done_receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            let snapshot = worker_control.cancel_and_wait_for_evaluations();
            done_sender.send(snapshot).unwrap();
        });

        for _ in 0..100 {
            if control.snapshot().cancellation_requested {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert!(control.snapshot().cancellation_requested);
        assert!(
            control
                .begin_evaluation(EvaluationStage::Search, 1)
                .is_none()
        );
        assert!(done_receiver.try_recv().is_err());

        drop(guard);
        let snapshot = done_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("cancel waiter should return after active score exits");
        worker.join().unwrap();
        assert_eq!(snapshot.evaluations_in_flight, 0);
        assert_eq!(snapshot.evaluations_completed, 1);
        assert_eq!(snapshot.evaluations_refused, 1);
    }

    #[test]
    fn deadline_is_distinct_from_user_cancellation_and_closes_search_gate() {
        let control = OptimizerRunControl::new(NonZeroUsize::new(2).unwrap());
        control.request_deadline();

        let snapshot = control.snapshot();
        assert!(!snapshot.cancellation_requested);
        assert!(snapshot.deadline_reached);
        assert!(!snapshot.budget_exhausted);
        assert!(control.stop_requested());
        assert!(
            control
                .begin_evaluation(EvaluationStage::Search, 1)
                .is_none()
        );
        assert_eq!(control.snapshot().evaluations_refused, 1);
    }

    #[test]
    fn simultaneous_user_and_deadline_requests_remain_inspectable() {
        let control = OptimizerRunControl::new(NonZeroUsize::new(2).unwrap());
        control.request_deadline();
        control.request_cancel();

        let snapshot = control.snapshot();
        assert!(snapshot.cancellation_requested);
        assert!(snapshot.deadline_reached);
    }

    #[test]
    fn search_budget_exhaustion_allows_validation_but_cancel_refuses_it() {
        let root = OptimizerRunControl::new(NonZeroUsize::new(1).unwrap());
        let control = root.with_stage_budget(NonZeroUsize::new(1).unwrap());
        drop(
            control
                .begin_evaluation(EvaluationStage::Search, 2)
                .unwrap(),
        );
        assert!(
            control
                .begin_evaluation(EvaluationStage::Search, 2)
                .is_none()
        );
        drop(
            control
                .begin_evaluation(EvaluationStage::Validation, 2)
                .expect("search budget exhaustion does not block validation"),
        );

        let cancelled_root = OptimizerRunControl::new(NonZeroUsize::new(2).unwrap());
        let cancelled = cancelled_root.with_stage_budget(NonZeroUsize::new(2).unwrap());
        cancelled.request_cancel();
        assert!(
            cancelled
                .begin_evaluation(EvaluationStage::Validation, 3)
                .is_none()
        );

        let root_snapshot = cancelled_root.snapshot();
        let stage_snapshot = cancelled.stage_snapshot().unwrap();
        assert_eq!(root_snapshot.validation_evaluations_started, 0);
        assert_eq!(root_snapshot.validation_evaluations_refused, 1);
        assert_eq!(stage_snapshot.validation_evaluations_started, 0);
        assert_eq!(stage_snapshot.validation_evaluations_refused, 1);
        assert!(root_snapshot.cancellation_requested);
    }

    #[test]
    fn cancel_or_deadline_before_validation_admission_is_never_raced_through() {
        for stop_kind in ["cancel", "deadline"] {
            let root = OptimizerRunControl::new(NonZeroUsize::new(3).unwrap());
            let control = root.with_stage_budget(NonZeroUsize::new(2).unwrap());
            let barrier = Arc::new(Barrier::new(2));
            let worker_control = control.clone();
            let worker_barrier = Arc::clone(&barrier);
            let (ready_sender, ready_receiver) = mpsc::channel();
            let worker = thread::spawn(move || {
                ready_sender.send(()).unwrap();
                worker_barrier.wait();
                worker_control
                    .begin_evaluation(EvaluationStage::Validation, 1)
                    .is_none()
            });

            ready_receiver
                .recv_timeout(Duration::from_secs(1))
                .expect("validation worker reached admission barrier");
            if stop_kind == "cancel" {
                control.request_cancel();
            } else {
                control.request_deadline();
            }
            barrier.wait();
            assert!(worker.join().unwrap(), "{stop_kind} must refuse validation");

            let root_snapshot = root.snapshot();
            let stage_snapshot = control.stage_snapshot().unwrap();
            assert_eq!(root_snapshot.validation_evaluations_started, 0);
            assert_eq!(root_snapshot.validation_evaluations_refused, 1);
            assert_eq!(stage_snapshot.validation_evaluations_started, 0);
            assert_eq!(stage_snapshot.validation_evaluations_refused, 1);
            assert_eq!(root_snapshot.cancellation_requested, stop_kind == "cancel");
            assert_eq!(root_snapshot.deadline_reached, stop_kind == "deadline");
        }
    }

    #[test]
    fn already_admitted_validation_completes_before_cancel_waiter_returns() {
        let root = OptimizerRunControl::new(NonZeroUsize::new(3).unwrap());
        let control = root.with_stage_budget(NonZeroUsize::new(2).unwrap());
        let barrier = Arc::new(Barrier::new(2));
        let worker_control = control.clone();
        let worker_barrier = Arc::clone(&barrier);
        let (admitted_sender, admitted_receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            let guard = worker_control
                .begin_evaluation(EvaluationStage::Validation, 2)
                .expect("score is admitted before cancellation");
            admitted_sender.send(()).unwrap();
            worker_barrier.wait();
            drop(guard);
        });
        admitted_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("validation score reached barrier while admitted");

        let waiter_control = control.clone();
        let (done_sender, done_receiver) = mpsc::channel();
        let (waiter_started_sender, waiter_started_receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            waiter_started_sender.send(()).unwrap();
            done_sender
                .send(waiter_control.cancel_and_wait_for_evaluations())
                .unwrap();
        });
        waiter_started_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("cancel waiter started");
        for _ in 0..100 {
            if control.snapshot().cancellation_requested {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        let waiting = control.snapshot();
        assert!(
            waiting.cancellation_requested,
            "cancel waiter must latch stop"
        );
        assert_eq!(waiting.validation_evaluations_in_flight, 1);
        assert!(done_receiver.try_recv().is_err());
        barrier.wait();
        worker.join().unwrap();
        let completed = done_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("cancel waiter returns after admitted validation score drains");
        waiter.join().unwrap();
        assert_eq!(completed.validation_evaluations_completed, 1);
        assert_eq!(completed.validation_evaluations_in_flight, 0);
        assert_eq!(completed.validation_component_evaluations_completed, 2);
        assert_eq!(
            control
                .stage_snapshot()
                .unwrap()
                .validation_evaluations_completed,
            1
        );
    }

    #[test]
    fn panicking_objectives_are_counted_as_failed_not_completed() {
        let control = OptimizerRunControl::new(NonZeroUsize::new(2).unwrap());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = control
                .begin_evaluation(EvaluationStage::Search, 3)
                .unwrap();
            panic!("simulated scorer panic");
        }));
        assert!(result.is_err());

        let snapshot = control.snapshot();
        assert_eq!(snapshot.evaluations_started, 1);
        assert_eq!(snapshot.evaluations_completed, 0);
        assert_eq!(snapshot.evaluations_failed, 1);
        assert_eq!(snapshot.component_evaluations_started, 3);
        assert_eq!(snapshot.component_evaluations_completed, 0);
        assert_eq!(snapshot.evaluations_in_flight, 0);
    }

    #[test]
    fn stage_quotas_share_root_budget_but_start_with_fresh_counters() {
        let root = OptimizerRunControl::new(NonZeroUsize::new(7).unwrap());
        let first = root.with_stage_budget(NonZeroUsize::new(4).unwrap());
        for _ in 0..4 {
            drop(first.begin_evaluation(EvaluationStage::Search, 2).unwrap());
        }
        assert!(first.begin_evaluation(EvaluationStage::Search, 2).is_none());
        assert_eq!(first.effective_evaluation_limit(), 0);
        assert_eq!(root.snapshot().evaluations_started, 4);
        assert_eq!(root.snapshot().evaluation_budget, 7);
        assert!(!root.snapshot().budget_exhausted);
        assert_eq!(first.stage_snapshot().unwrap().evaluations_started, 4);
        assert!(first.stage_snapshot().unwrap().budget_exhausted);

        let second = root.with_stage_budget(NonZeroUsize::new(5).unwrap());
        assert_eq!(second.effective_evaluation_limit(), 3);
        for _ in 0..3 {
            drop(second.begin_evaluation(EvaluationStage::Search, 1).unwrap());
        }
        assert!(
            second
                .begin_evaluation(EvaluationStage::Search, 1)
                .is_none()
        );
        assert_eq!(second.stage_snapshot().unwrap().evaluations_started, 3);
        assert_eq!(root.snapshot().evaluations_started, 7);
        assert!(root.snapshot().budget_exhausted);
    }

    #[test]
    fn concurrent_stage_clones_never_exceed_stage_or_root_caps() {
        let root = OptimizerRunControl::new(NonZeroUsize::new(17).unwrap());
        let first_stage = root.with_stage_budget(NonZeroUsize::new(13).unwrap());
        let mut workers = Vec::new();
        for _ in 0..8 {
            let control = first_stage.clone();
            workers.push(thread::spawn(move || {
                while let Some(guard) = control.begin_evaluation(EvaluationStage::Search, 1) {
                    drop(guard);
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(
            first_stage.stage_snapshot().unwrap().evaluations_started,
            13
        );
        assert_eq!(root.snapshot().evaluations_started, 13);
        assert_eq!(first_stage.effective_evaluation_limit(), 0);

        let second_stage = root.with_stage_budget(NonZeroUsize::new(10).unwrap());
        assert_eq!(second_stage.effective_evaluation_limit(), 4);
        let mut workers = Vec::new();
        for _ in 0..4 {
            let control = second_stage.clone();
            workers.push(thread::spawn(move || {
                drop(control.begin_evaluation(EvaluationStage::Search, 1));
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(
            second_stage.stage_snapshot().unwrap().evaluations_started,
            4
        );
        assert_eq!(root.snapshot().evaluations_started, 17);
        assert!(root.snapshot().budget_exhausted);
    }

    #[test]
    fn stage_validation_is_separate_and_does_not_consume_search_quota() {
        let root = OptimizerRunControl::new(NonZeroUsize::new(3).unwrap());
        let stage = root.with_stage_budget(NonZeroUsize::new(1).unwrap());
        drop(stage.begin_evaluation(EvaluationStage::Search, 2).unwrap());
        drop(
            stage
                .begin_evaluation(EvaluationStage::Validation, 4)
                .unwrap(),
        );

        let stage_snapshot = stage.stage_snapshot().unwrap();
        let root_snapshot = root.snapshot();
        assert_eq!(stage_snapshot.evaluations_started, 1);
        assert_eq!(stage_snapshot.validation_evaluations_started, 1);
        assert_eq!(stage_snapshot.validation_component_evaluations_started, 4);
        assert_eq!(root_snapshot.evaluations_started, 1);
        assert_eq!(root_snapshot.validation_evaluations_started, 1);
        assert_eq!(root_snapshot.evaluation_budget, 3);
        assert_eq!(stage.effective_evaluation_limit(), 0);

        let next_stage = root.with_stage_budget(NonZeroUsize::new(2).unwrap());
        assert_eq!(next_stage.effective_evaluation_limit(), 2);
    }
}
