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
use super::{ControlledOptimizerRun, ObjectiveData, OptimProgressCallback, OptimizerRunEvidence};
use crate::OptimParams;

/// Legacy backend result paired with its optional DE completion counters.
pub type OptimizerBackendDeCompletion = (
    Result<(String, f64), (String, f64)>,
    Option<super::de::DECompletion>,
);

/// Legacy backend result paired with its optional DE and search evidence.
pub type OptimizerBackendCompletionEvidence = (
    Result<(String, f64), (String, f64)>,
    Option<super::de::DECompletion>,
    Option<super::backend::BackendSearchEvidence>,
);

/// Backend result and all same-run typed completion evidence.
pub type OptimizerBackendTypedCompletionEvidence = (
    Result<(String, f64), (String, f64)>,
    Option<super::de::DECompletion>,
    Option<super::backend::BackendSearchEvidence>,
    Option<super::optimize::OptimizerBackendCompletion>,
);

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
    /// Whether this backend can enforce a shared candidate-evaluation budget.
    /// Custom backends remain uncontrolled unless they explicitly opt in.
    fn supports_controlled_runs(&self) -> bool {
        false
    }

    /// Run one dispatch and retain its structured completion evidence.
    ///
    /// Custom backends keep their existing dispatch semantics. Production
    /// uses counted evaluations and typed completion for both global search
    /// and algorithm overrides, including searches with progress callbacks.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the optimizer dispatch boundary"
    )]
    fn optimize_filters_evidence(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        algo_override: Option<&str>,
        callback: Option<OptimProgressCallback>,
    ) -> OptimizerRunEvidence {
        let result = if algo_override.is_some() {
            self.optimize_filters_with_algo_override(
                x,
                lower,
                upper,
                objective,
                params,
                algo_override,
            )
        } else if let Some(callback) = callback {
            self.optimize_filters_with_callback(x, lower, upper, objective, params, callback)
        } else {
            self.optimize_filters(x, lower, upper, objective, params)
        };
        OptimizerRunEvidence::from_backend_result(
            algo_override.unwrap_or(&params.algo),
            result,
            x,
            lower,
            upper,
            params.maxeval,
            params.seed,
        )
    }

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
    ) -> crate::optim::OptimizerOutcome;

    /// Retain typed DE completion when the backend can provide it.
    /// Test doubles and non-DE backends keep the historical result contract.
    fn optimize_filters_with_de_completion(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
    ) -> OptimizerBackendDeCompletion {
        (
            self.optimize_filters(x, lower_bounds, upper_bounds, objective, params),
            None,
        )
    }

    /// Retain same-invocation MH search diagnostics when available.
    ///
    /// Test doubles preserve their historical result unless they explicitly
    /// supply diagnostics. The search report is observational and cannot turn
    /// a best-effort result into a converged one.
    fn optimize_filters_with_completion_evidence(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
    ) -> OptimizerBackendCompletionEvidence {
        let (result, de_completion) = self.optimize_filters_with_de_completion(
            x,
            lower_bounds,
            upper_bounds,
            objective,
            params,
        );
        (result, de_completion, None)
    }

    /// Retain typed completion while preserving the legacy evidence method.
    ///
    /// Existing backend implementations inherit `None` for completion.
    fn optimize_filters_with_typed_completion_evidence(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
    ) -> OptimizerBackendTypedCompletionEvidence {
        let (result, de_completion, search_evidence) = self
            .optimize_filters_with_completion_evidence(
                x,
                lower_bounds,
                upper_bounds,
                objective,
                params,
            );
        (result, de_completion, search_evidence, None)
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
    ) -> crate::optim::OptimizerOutcome;

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
    ) -> crate::optim::OptimizerOutcome;

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
    fn supports_controlled_runs(&self) -> bool {
        true
    }

    fn optimize_filters_evidence(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        algo_override: Option<&str>,
        callback: Option<OptimProgressCallback>,
    ) -> OptimizerRunEvidence {
        super::optimize::ordinary_dispatch_evidence(
            x,
            lower,
            upper,
            objective,
            params,
            algo_override,
            callback,
        )
    }

    fn optimize_filters_with_completion_evidence(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
    ) -> (
        Result<(String, f64), (String, f64)>,
        Option<super::de::DECompletion>,
        Option<super::backend::BackendSearchEvidence>,
    ) {
        let (result, de_completion, search_evidence, _) =
            super::optimize_filters_with_typed_completion_evidence(
                x,
                lower_bounds,
                upper_bounds,
                objective,
                params,
            );
        (result, de_completion, search_evidence)
    }

    fn optimize_filters_with_typed_completion_evidence(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
    ) -> (
        Result<(String, f64), (String, f64)>,
        Option<super::de::DECompletion>,
        Option<super::backend::BackendSearchEvidence>,
        Option<super::optimize::OptimizerBackendCompletion>,
    ) {
        super::optimize_filters_with_typed_completion_evidence(
            x,
            lower_bounds,
            upper_bounds,
            objective,
            params,
        )
    }

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
    ) -> crate::optim::OptimizerOutcome {
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
        super::optimize_filters_with_de_completion(x, lower_bounds, upper_bounds, objective, params)
    }

    fn optimize_filters_with_callback(
        &self,
        x: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        callback: OptimProgressCallback,
    ) -> crate::optim::OptimizerOutcome {
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
    ) -> crate::optim::OptimizerOutcome {
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
    pub result: crate::optim::OptimizerOutcome,
    /// Optional separate result for the local-refinement (`algo_override`) call.
    /// If `None`, [`Self::result`] is returned for refinement as well.
    pub refine_result: Option<crate::optim::OptimizerOutcome>,
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
    pub fn with_refine_result(mut self, result: crate::optim::OptimizerOutcome) -> Self {
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
    ) -> crate::optim::OptimizerOutcome {
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
    ) -> crate::optim::OptimizerOutcome {
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
    ) -> crate::optim::OptimizerOutcome {
        self.refine_result
            .clone()
            .unwrap_or_else(|| self.result.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::{MockOptimizerBackend, OptimizerBackend, RealOptimizerBackend};
    use crate::OptimParams;

    fn flat_objective() -> super::ObjectiveData {
        super::super::ObjectiveDataBuilder::speaker_flat(
            ndarray::array![20.0, 100.0, 1000.0, 20_000.0],
            ndarray::Array1::zeros(4),
            ndarray::Array1::zeros(4),
            48_000.0,
            crate::PeqModel::Pk,
        )
        .freq_range(20.0, 20_000.0)
        .min_db(-3.0)
        .max_db(3.0)
        .build()
        .unwrap()
    }

    #[test]
    fn ordinary_native_dispatch_retains_typed_completion_and_counts() {
        let backend = RealOptimizerBackend::new();
        let mut params = OptimParams::from(&crate::OptimizationConfig::speaker_defaults());
        params.algo = "autoeq:cmaes".into();
        params.maxeval = 6000;
        params.population = 6;
        params.num_filters = 1;
        params.peq_model = crate::PeqModel::Pk;
        params.min_spacing_oct = 0.0;
        params.seed = Some(7);
        let lower = [20.0_f64.log10(), 0.5, -3.0];
        let upper = [20_000.0_f64.log10(), 2.0, 3.0];
        for override_algorithm in [None, Some("autoeq:cobyla")] {
            let mut x = [100.0_f64.log10(), 1.0, 0.0];
            let evidence = backend.optimize_filters_evidence(
                &mut x,
                &lower,
                &upper,
                flat_objective(),
                &params,
                override_algorithm,
                None,
            );
            assert!(evidence.evaluation_count.is_some_and(|count| count > 0));
            assert!(evidence.converged, "{evidence:?}");
            assert_eq!(
                evidence.algorithm,
                override_algorithm.unwrap_or(&params.algo)
            );
            assert!(evidence.has_valid_candidate());
        }
    }

    #[test]
    fn ordinary_native_dispatch_distinguishes_budget_and_callback_stops() {
        use crate::optim::OptimizerTermination;
        let backend = RealOptimizerBackend::new();
        let mut params = OptimParams::from(&crate::OptimizationConfig::speaker_defaults());
        params.algo = "autoeq:cmaes".into();
        params.population = 6;
        params.maxeval = 60;
        params.num_filters = 1;
        params.peq_model = crate::PeqModel::Pk;
        params.min_spacing_oct = 0.0;
        params.seed = Some(7);
        let lower = [20.0_f64.log10(), 0.5, -3.0];
        let upper = [20_000.0_f64.log10(), 2.0, 3.0];
        for stopped in [false, true] {
            let mut x = [100.0_f64.log10(), 1.0, 0.0];
            let callback = stopped.then(|| {
                Box::new(|_, _, _| crate::de::CallbackAction::Stop)
                    as crate::optim::OptimProgressCallback
            });
            let evidence = backend.optimize_filters_evidence(
                &mut x,
                &lower,
                &upper,
                flat_objective(),
                &params,
                None,
                callback,
            );
            assert_eq!(
                evidence.termination,
                if stopped {
                    OptimizerTermination::UserStopped
                } else {
                    OptimizerTermination::EvaluationLimit
                },
                "{evidence:?}",
            );
            assert!(!evidence.converged);
            assert!(evidence.has_valid_candidate());
            assert!(evidence.evaluation_count.is_some_and(|count| count > 0));
        }
    }

    #[test]
    fn custom_backend_keeps_override_dispatch_and_cannot_claim_typed_convergence() {
        let backend = MockOptimizerBackend::ok("converged", 1.0)
            .with_refine_result(Ok(("local nfev=3".into(), 0.5)));
        let params = OptimParams::from(&crate::OptimizationConfig::speaker_defaults());
        let mut x = [2.0, 1.0, 0.0];
        let evidence = backend.optimize_filters_evidence(
            &mut x,
            &[1.0, 0.5, -3.0],
            &[4.0, 2.0, 3.0],
            flat_objective(),
            &params,
            Some("cobyla"),
            None,
        );
        assert_eq!(evidence.objective, Some(0.5));
        assert_eq!(evidence.evaluation_count, Some(3));
        assert!(!evidence.converged);
    }

    #[test]
    fn only_dispatchers_with_native_knowledge_report_profiles() {
        let lower = [0.0, 0.0, 0.0];
        let upper = [1.0, 1.0, 1.0];
        let mut args = crate::OptimizationConfig::speaker_defaults();
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
