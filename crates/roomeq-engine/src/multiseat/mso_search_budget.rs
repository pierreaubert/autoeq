use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Stage-specific execution budget for the continuous MSO differential-evolution
/// search (`optimize_continuous_mso`).
///
/// All fields are optional and default to the historical hardcoded search: a
/// `None` budget reproduces the legacy population/generation counts, seed,
/// unbounded runtime, no early stopping, and no cancellation exactly. This is
/// additive: callers that do not set a budget observe byte-identical search
/// behavior to before.
///
/// Evaluation accounting: one "evaluation" is one call of the area-loss
/// closure, which itself sweeps Q quadrature points (and, for the worst-case
/// scalarisation, nests another inner DE per call — see `inner_maxiter`).
#[derive(Debug, Clone, Default)]
pub(super) struct MsoSearchBudget {
    /// Cap on total area-loss evaluations (initial population scoring counts).
    /// `None` runs the legacy `population_size * (1 + generations)` schedule.
    pub(super) max_evaluations: Option<usize>,
    /// Wall-clock cap on the search. Checked once per generation.
    /// `None` runs to completion.
    pub(super) max_duration: Option<Duration>,
    /// Override the DE seed. `None` keeps `MSO_DE_SEED ^ num_subs`.
    pub(super) seed: Option<u64>,
    /// Stop after this many generations without improvement. `None` never
    /// stops early on convergence.
    pub(super) stall_generations: Option<usize>,
    /// Cooperative cancellation flag, checked once per generation.
    /// `None` disables cancellation.
    pub(super) cancelled: Option<Arc<AtomicBool>>,
    /// Cap on the nested worst-case inner search iterations. `None` keeps the
    /// configured `inner_maxiter`. Applied as `min(configured, cap)` at the
    /// dispatch site; the outer DE always honors `max_evaluations`.
    pub(super) max_inner_iterations: Option<usize>,
}

/// Consumed-work report for one `optimize_continuous_mso_with_budget` run.
#[derive(Debug, Clone)]
pub(super) struct MsoSearchReport {
    /// Total area-loss evaluations consumed (population scoring + trials).
    pub(super) evaluations: usize,
    /// Generations completed (initial scoring is generation 0 work).
    pub(super) generations_run: usize,
    /// Best loss observed.
    pub(super) best_loss: f64,
    /// Whether the search stopped before the full generation schedule.
    pub(super) stopped_early: bool,
    /// Machine-readable stop cause: "completed", "evaluation_budget",
    /// "time_budget", "cancelled", or "converged".
    pub(super) stop_reason: &'static str,
    /// Wall-clock time consumed by the search.
    pub(super) elapsed: Duration,
}

pub(super) fn is_cancelled(budget: &MsoSearchBudget) -> bool {
    budget
        .cancelled
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
}

/// Improvement tolerance for stall detection: matches the objective-regression
/// tolerance scale so "no improvement" means the same thing as "regressed".
pub(super) const MSO_STALL_IMPROVEMENT_TOLERANCE: f64 = 1e-12;

pub(super) fn search_started() -> Instant {
    Instant::now()
}
