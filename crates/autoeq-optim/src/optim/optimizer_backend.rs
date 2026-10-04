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

use super::run_control::{OptimizerBudgetProfile, OptimizerRunControl};
use super::{ControlledOptimizerRun, ObjectiveData, OptimProgressCallback};
use crate::OptimParams;

/// Effective registered backend and native budget profile for a dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizerDispatchBudgetProfile {
    /// Canonical backend name resolved by the production dispatcher.
    pub backend: String,
    /// Native profile computed for the effective controlled-run cap, if supported.
    pub profile: Option<OptimizerBudgetProfile>,
}

/// A backend cannot enforce the requested controlled-run contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlledBackendUnsupported;

impl std::fmt::Display for ControlledBackendUnsupported {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("optimizer backend does not support controlled runs")
    }
}

impl std::error::Error for ControlledBackendUnsupported {}

/// High-level optimizer backend used by the RoomEQ filter-fitting pipeline.
pub trait OptimizerBackend: Send + Sync {
    /// Describe the actual solver profile for a controlled dispatch.
    ///
    /// Custom backends should return `None` unless they can report settings
    /// from the exact implementation that will receive this call.
    fn evaluation_budget_profile(
        &self,
        _lower_bounds: &[f64],
        _upper_bounds: &[f64],
        _params: &OptimParams,
        _algo_override: Option<&str>,
        _evaluation_limit: usize,
    ) -> Option<OptimizerDispatchBudgetProfile> {
        None
    }

    /// Run the configured global optimizer.
    fn optimize_filters(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
    ) -> Result<(String, f64), (String, f64)>;

    /// Retain typed DE completion when the backend can provide it.
    /// Test doubles and non-DE backends keep the historical result contract.
    fn optimize_filters_with_de_completion(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
    ) -> (
        Result<(String, f64), (String, f64)>,
        Option<super::de::DECompletion>,
    ) {
        (
            self.optimize_filters(x, lower_bounds, upper_bounds, objective, params),
            None,
        )
    }

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

    /// Run with a shared score cap and retain typed dispatch evidence.
    ///
    /// Existing custom backends must implement this explicitly. The default
    /// refuses the request without invoking an uncontrolled optimizer.
    ///
    /// # Errors
    /// Returns `ControlledBackendUnsupported` unless the backend supports run control.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the existing optimizer seam"
    )]
    fn optimize_filters_controlled(
        &self,
        _x: &mut [f64],
        _lower_bounds: &[f64],
        _upper_bounds: &[f64],
        _objective: ObjectiveData,
        _params: &OptimParams,
        _algo_override: Option<&str>,
        _callback: Option<OptimProgressCallback>,
        _run_control: &OptimizerRunControl,
    ) -> Result<ControlledOptimizerRun, ControlledBackendUnsupported> {
        Err(ControlledBackendUnsupported)
    }
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
    fn evaluation_budget_profile(
        &self,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        params: &OptimParams,
        algo_override: Option<&str>,
        evaluation_limit: usize,
    ) -> Option<OptimizerDispatchBudgetProfile> {
        let algorithm = algo_override.unwrap_or(&params.algo);
        let backend = super::registry::resolve(algorithm)?;
        let profile = if evaluation_limit == 0 {
            None
        } else {
            let mut profile_params = params.clone();
            profile_params.algo = algorithm.to_string();
            profile_params.maxeval = evaluation_limit;
            backend.evaluation_budget_profile(lower_bounds, upper_bounds, &profile_params)
        };
        Some(OptimizerDispatchBudgetProfile {
            backend: backend.name().to_string(),
            profile,
        })
    }

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

    fn optimize_filters_with_de_completion(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
    ) -> (
        Result<(String, f64), (String, f64)>,
        Option<super::de::DECompletion>,
    ) {
        super::optimize_filters_with_de_completion(
            x,
            lower_bounds,
            upper_bounds,
            objective,
            params,
        )
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

    fn optimize_filters_controlled(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        algo_override: Option<&str>,
        callback: Option<OptimProgressCallback>,
        run_control: &OptimizerRunControl,
    ) -> Result<ControlledOptimizerRun, ControlledBackendUnsupported> {
        Ok(
            super::optimize_filters_with_run_control_and_algo_override_detailed(
                x,
                lower_bounds,
                upper_bounds,
                objective,
                params,
                algo_override,
                callback,
                run_control,
            ),
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
    pub fn with_refine_result(mut self, result: Result<(String, f64), (String, f64)>) -> Self {
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
        self.refine_result
            .clone()
            .unwrap_or_else(|| self.result.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::{MockOptimizerBackend, OptimizerBackend, RealOptimizerBackend};
    use crate::OptimParams;

    #[test]
    fn only_dispatchers_with_native_knowledge_report_profiles() {
        let lower = [0.0, 0.0, 0.0];
        let upper = [1.0, 1.0, 1.0];
        let mut args = crate::cli::Args::speaker_defaults();
        args.algo = "autoeq:de".into();
        let params = OptimParams::from(&args);
        let custom = MockOptimizerBackend::default();
        assert!(
            custom
                .evaluation_budget_profile(&lower, &upper, &params, None, 128)
                .is_none(),
            "the trait default must not infer a custom solver's profile from its algorithm label"
        );

        let real = RealOptimizerBackend::new();
        assert!(
            real.evaluation_budget_profile(&lower, &upper, &params, None, 0)
                .is_some_and(|dispatch| dispatch.profile.is_none()),
            "a dispatch with no remaining search budget has no solver profile"
        );
        let dispatch = real
            .evaluation_budget_profile(&lower, &upper, &params, Some("cobyla"), 128)
            .expect("the production resolver recognizes the configured alias");
        assert_eq!(dispatch.backend, "autoeq:cobyla");
        assert_eq!(
            dispatch
                .profile
                .map(|profile| profile.requested_evaluations),
            Some(128)
        );
    }
}
