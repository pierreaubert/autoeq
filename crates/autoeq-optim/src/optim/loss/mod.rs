//! Strategy-based single-curve objective functions for the optimizer.
//!
//! Each [`LossType`] is mapped to an implementation of the [`Objective`]
//! trait.  This keeps the per-loss math in isolated modules and makes the
//! dispatcher in [`crate::optim::compute`] a small match instead of a
//! mega-function.

pub mod context;
pub mod strategies;

pub use context::ObjectiveContext;
pub use strategies::*;

/// Interchangeable single-curve objective function.
pub trait Objective: Send + Sync {
    /// Compute the scalar loss for parameter vector `x`.
    fn compute(&self, x: &[f64], ctx: &ObjectiveContext) -> f64;

    /// Evaluate an already-realized correction magnitude (dB), on `ctx.freqs`.
    /// This is the complete correction, not the corrected measurement or a
    /// new target. It lets FIR stages retain the same loss/deadband/regularizer
    /// as PEQ stages. Physical driver/array objectives need more than a scalar
    /// response and explicitly remain unsupported here.
    fn compute_response(
        &self,
        _correction_db: &ndarray::Array1<f64>,
        _ctx: &ObjectiveContext,
    ) -> Option<f64> {
        None
    }
}
