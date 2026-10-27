//! Backend execution callbacks and shared objective inputs.

pub use autoeq_objective::optim::types::MultiObjectiveData;

/// Progress callback: (iteration, best_loss, epa_preference) -> continue/stop
///
/// Used to thread per-iteration optimizer progress through the room EQ call chain.
/// `epa_preference` is `Some` when EPA is computed (every N iterations), `None` otherwise.
pub type OptimProgressCallback =
    Box<dyn FnMut(usize, f64, Option<f64>) -> crate::de::CallbackAction + Send>;
