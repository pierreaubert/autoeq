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
}
