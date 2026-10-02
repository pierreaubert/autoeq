//! Hard objective-evaluation budgets and cooperative cancellation.

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
    validation_component_evaluations_started: usize,
    validation_component_evaluations_completed: usize,
    validation_evaluations_in_flight: usize,
    cancellation_requested: bool,
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
    /// Candidate evaluations rejected after cancellation or budget exhaustion.
    pub evaluations_refused: usize,
    /// Candidate evaluations still scoring when the snapshot was taken.
    pub evaluations_in_flight: usize,
    /// Finalizer and post-run candidate scores outside the search cap.
    pub validation_evaluations_started: usize,
    /// Validation scores that returned.
    pub validation_evaluations_completed: usize,
    /// Validation scores whose objective function unwound with a panic.
    pub validation_evaluations_failed: usize,
    /// Measurement components scored during finalization and post-run checks.
    pub validation_component_evaluations_started: usize,
    /// Validation measurement components that returned.
    pub validation_component_evaluations_completed: usize,
    /// Validation scores active when the snapshot was taken.
    pub validation_evaluations_in_flight: usize,
    /// Whether cancellation was requested.
    pub cancellation_requested: bool,
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
        }
    }

    /// Return the hard candidate-evaluation limit.
    pub fn evaluation_budget(&self) -> usize {
        self.inner.budget.get()
    }

    /// Whether the gate is closed by cancellation or budget exhaustion.
    pub fn stop_requested(&self) -> bool {
        let state = self.lock_state();
        state.cancellation_requested || state.evaluations_started >= self.inner.budget.get()
    }

    /// Close the gate so no later expensive score can start.
    pub fn request_cancel(&self) {
        let mut state = self.lock_state();
        state.cancellation_requested = true;
        if state.evaluations_in_flight + state.validation_evaluations_in_flight == 0 {
            self.inner.idle.notify_all();
        }
    }

    /// Close the gate and wait for active objective scores to finish.
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
        let objective_components = objective_components.max(1);
        match stage {
            EvaluationStage::Search => {
                if state.cancellation_requested
                    || state.evaluations_started >= self.inner.budget.get()
                {
                    state.evaluations_refused = state.evaluations_refused.saturating_add(1);
                    return None;
                }
                state.evaluations_started += 1;
                state.component_evaluations_started = state
                    .component_evaluations_started
                    .saturating_add(objective_components);
                state.evaluations_in_flight += 1;
            }
            EvaluationStage::Validation => {
                state.validation_evaluations_started += 1;
                state.validation_component_evaluations_started = state
                    .validation_component_evaluations_started
                    .saturating_add(objective_components);
                state.validation_evaluations_in_flight += 1;
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
            }
        }
        if state.evaluations_in_flight + state.validation_evaluations_in_flight == 0 {
            self.control.inner.idle.notify_all();
        }
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
        validation_component_evaluations_started: state.validation_component_evaluations_started,
        validation_component_evaluations_completed: state
            .validation_component_evaluations_completed,
        validation_evaluations_in_flight: state.validation_evaluations_in_flight,
        cancellation_requested: state.cancellation_requested,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
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
    fn validation_scores_are_counted_separately_and_remain_available_after_cancel() {
        let control = OptimizerRunControl::new(NonZeroUsize::new(1).unwrap());
        let search = control
            .begin_evaluation(EvaluationStage::Search, 2)
            .unwrap();
        drop(search);
        control.request_cancel();

        let validation = control
            .begin_evaluation(EvaluationStage::Validation, 2)
            .expect("validation is not limited by search cancellation");
        drop(validation);

        let snapshot = control.snapshot();
        assert_eq!(snapshot.evaluations_started, 1);
        assert_eq!(snapshot.component_evaluations_started, 2);
        assert_eq!(snapshot.validation_evaluations_started, 1);
        assert_eq!(snapshot.validation_component_evaluations_started, 2);
        assert!(snapshot.cancellation_requested);
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
}
