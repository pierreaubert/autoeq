use super::super::backend::{AlgorithmType, ConstraintCapabilities, FilterOptimizer};
use super::super::params::OptimParams as BackendOptimParams;
use super::super::run_control::OptimizerBudgetProfile;
use super::super::{ObjectiveData, OptimProgressCallback, PenaltyMode};
use super::optimize::optimize_filters_autoeq;
use super::optimize::optimize_filters_autoeq_with_callback;
use crate::de::{CallbackAction, DEIntermediate};

/// AutoEQ DE-backed `FilterOptimizer`. Single instance — name is `"autoeq:de"`
/// today; the strategy variants (best1bin, lshadebin, …) are picked from
/// `OptimParams::strategy` inside `optimize_filters_autoeq`.
pub struct AutoeqDeBackend {
    pub(super) name: &'static str,
}

impl AutoeqDeBackend {
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl FilterOptimizer for AutoeqDeBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn supports_initial_candidate(&self) -> bool {
        true
    }

    fn evaluation_budget_profile(
        &self,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        params: &BackendOptimParams,
    ) -> Option<OptimizerBudgetProfile> {
        if lower_bounds.len() != upper_bounds.len() {
            return None;
        }
        let (_, population_size, generation_limit) = super::misc::derive_de_budget(
            lower_bounds,
            upper_bounds,
            params.population,
            params.maxeval,
        );
        // Every fresh solver start scores the initial population and then
        // replaces its best member with a separately scored x0. Exact resume
        // bypasses both scores and is exposed through its dedicated API.
        let initial_batch = population_size.saturating_add(1);
        let solver_limit =
            initial_batch.saturating_add(generation_limit.saturating_mul(population_size));
        Some(OptimizerBudgetProfile::new(
            params.maxeval,
            Some(solver_limit),
            population_size.saturating_mul(2).saturating_add(1),
            initial_batch,
            Some(population_size),
            Some(population_size),
            Some(generation_limit),
        ))
    }

    fn library(&self) -> &'static str {
        "AutoEQ"
    }
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::Global
    }
    fn capabilities(&self) -> ConstraintCapabilities {
        ConstraintCapabilities {
            nonlinear_ineq: true,
            nonlinear_eq: true,
            linear: true,
            iteration_callback: true,
            // Unused (nonlinear_ineq=true means install_constraints disables penalties)
            // but keep a sensible value for completeness.
            fallback_penalty_mode: PenaltyMode::Disabled,
        }
    }
    fn optimize(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &BackendOptimParams,
        callback: Option<OptimProgressCallback>,
    ) -> Result<(String, f64), (String, f64)> {
        match callback {
            Some(user_cb) => {
                // Adapt unified callback to DE's typed callback. EPA mid-run
                // computation (the original behaviour from optim.rs:1163-1211)
                // is performed by the caller wrapping `user_cb` before passing
                // it in — the trait stays generic; EPA is autoeq-loss-specific.
                let de_cb: Box<dyn FnMut(&DEIntermediate) -> CallbackAction + Send> = {
                    let mut user_cb = user_cb;
                    Box::new(move |im| user_cb(im.iter, im.fun, None))
                };
                optimize_filters_autoeq_with_callback(
                    x, lower, upper, objective, self.name, params, de_cb,
                )
            }
            None => optimize_filters_autoeq(x, lower, upper, objective, self.name, params),
        }
    }
}
