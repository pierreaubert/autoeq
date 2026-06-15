//! High-level optimizer seam for EQ unit tests.
//!
//! [`OptimizerBackend`] wraps the three [`crate::optim`] dispatch entry points
//! (`optimize_filters`, `optimize_filters_with_callback`,
//! `optimize_filters_with_algo_override`) so RoomEQ code can be exercised with
//! a deterministic fake optimizer instead of running real global/local searches.
//!
//! Production code uses [`RealOptimizerBackend`]; tests can inject
//! [`MockOptimizerBackend`] to avoid flaky stochastic optimization while still
//! exercising curve preparation, target construction, and filter conversion.

use super::{ObjectiveData, OptimProgressCallback};
use crate::OptimParams;

/// High-level optimizer backend used by the RoomEQ filter-fitting pipeline.
pub trait OptimizerBackend: Send + Sync {
    /// Run the configured global optimizer.
    fn optimize_filters(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
    ) -> Result<(String, f64), (String, f64)>;

    /// Run the configured global optimizer with a per-iteration progress callback.
    fn optimize_filters_with_callback(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        callback: OptimProgressCallback,
    ) -> Result<(String, f64), (String, f64)>;

    /// Run an optimizer with an optional algorithm override.
    ///
    /// Used for the local-refinement step, which switches from the global
    /// algorithm to [`OptimParams::local_algo`] without rebuilding the params.
    fn optimize_filters_with_algo_override(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        algo_override: Option<&str>,
    ) -> Result<(String, f64), (String, f64)>;
}

/// Production backend: delegates to the real [`crate::optim`] dispatchers.
#[derive(Debug, Default, Clone, Copy)]
pub struct RealOptimizerBackend;

impl RealOptimizerBackend {
    /// Create a new production optimizer backend.
    pub fn new() -> Self {
        Self
    }
}

impl OptimizerBackend for RealOptimizerBackend {
    fn optimize_filters(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
    ) -> Result<(String, f64), (String, f64)> {
        super::optimize_filters(x, lower_bounds, upper_bounds, objective, params)
    }

    fn optimize_filters_with_callback(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        callback: OptimProgressCallback,
    ) -> Result<(String, f64), (String, f64)> {
        super::optimize_filters_with_callback(
            x,
            lower_bounds,
            upper_bounds,
            objective,
            params,
            callback,
        )
    }

    fn optimize_filters_with_algo_override(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        algo_override: Option<&str>,
    ) -> Result<(String, f64), (String, f64)> {
        super::optimize_filters_with_algo_override(
            x,
            lower_bounds,
            upper_bounds,
            objective,
            params,
            algo_override,
        )
    }
}

/// Deterministic fake backend for tests.
///
/// Returns fixed results without mutating the parameter vector, so callers can
/// cheaply verify preparation/filter-conversion plumbing without invoking a
/// real stochastic optimizer.
#[derive(Debug, Clone)]
pub struct MockOptimizerBackend {
    /// Result returned by global and spatial-robustness optimization calls.
    pub result: Result<(String, f64), (String, f64)>,
    /// Optional separate result for the local-refinement (`algo_override`) call.
    /// If `None`, [`Self::result`] is returned for refinement as well.
    pub refine_result: Option<Result<(String, f64), (String, f64)>>,
}

impl Default for MockOptimizerBackend {
    fn default() -> Self {
        Self {
            result: Ok(("mock".to_string(), 0.0)),
            refine_result: None,
        }
    }
}

impl MockOptimizerBackend {
    /// Create a fake backend that reports success with the given message and loss.
    pub fn ok(msg: impl Into<String>, loss: f64) -> Self {
        Self {
            result: Ok((msg.into(), loss)),
            refine_result: None,
        }
    }

    /// Create a fake backend that reports non-convergence with the given message
    /// and loss.
    pub fn err(msg: impl Into<String>, loss: f64) -> Self {
        Self {
            result: Err((msg.into(), loss)),
            refine_result: None,
        }
    }

    /// Set a separate result for local-refinement calls.
    pub fn with_refine_result(
        mut self,
        result: Result<(String, f64), (String, f64)>,
    ) -> Self {
        self.refine_result = Some(result);
        self
    }
}

impl OptimizerBackend for MockOptimizerBackend {
    fn optimize_filters(
        &self,
        _x: &mut [f64],
        _lower_bounds: &[f64],
        _upper_bounds: &[f64],
        _objective: ObjectiveData,
        _params: &OptimParams,
    ) -> Result<(String, f64), (String, f64)> {
        self.result.clone()
    }

    fn optimize_filters_with_callback(
        &self,
        _x: &mut [f64],
        _lower_bounds: &[f64],
        _upper_bounds: &[f64],
        _objective: ObjectiveData,
        _params: &OptimParams,
        _callback: OptimProgressCallback,
    ) -> Result<(String, f64), (String, f64)> {
        self.result.clone()
    }

    fn optimize_filters_with_algo_override(
        &self,
        _x: &mut [f64],
        _lower_bounds: &[f64],
        _upper_bounds: &[f64],
        _objective: ObjectiveData,
        _params: &OptimParams,
        _algo_override: Option<&str>,
    ) -> Result<(String, f64), (String, f64)> {
        self.refine_result.clone().unwrap_or_else(|| self.result.clone())
    }
}
