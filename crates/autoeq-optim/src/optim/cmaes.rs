//! AutoEQ CMA-ES backend.
//!
//! Wraps [`math_audio_optimisation::cma_es`] in the shared
//! [`FilterOptimizer`](super::backend::FilterOptimizer) interface. CMA-ES does
//! not support native AutoEQ constraints, so ceiling/spacing/min-gain are
//! folded into the scalar objective as penalties.

use super::backend::{
    AlgorithmType, ConstraintCapabilities, FilterOptimizer, FilterOptimizerOutput,
};
use super::constraints_install::install_constraints;
use super::optimize::OptimizerBackendCompletion;
use super::params::OptimParams;
use super::run_control::OptimizerBudgetProfile;
use super::{ObjectiveData, OptimProgressCallback, PenaltyMode, compute_fitness_penalties_ref};
use math_audio_optimisation::{
    CmaEsConfig, CmaEsIntermediate, CmaEsReport, ParallelConfig, cma_es,
};
use ndarray::Array1;
use std::sync::Arc;

/// Keep scored CMA-ES candidates and returned controls inside exact rails.
///
/// Normalized box interpolation can round a boundary point a few ULPs outside
/// its physical interval. Project before scoring and on return so the reported
/// loss describes the exact emitted parameters, without loosening final checks.
///
/// # Errors
/// Returns the underlying solver's configuration or execution error.
pub(super) fn run_bounded_cma_es<F>(
    objective: &F,
    config: CmaEsConfig,
) -> math_audio_optimisation::Result<CmaEsReport>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    let bounds = config.bounds.clone();
    let bounded_objective = |point: &Array1<f64>| {
        if point
            .iter()
            .zip(&bounds)
            .all(|(value, (low, high))| value >= low && value <= high)
        {
            objective(point)
        } else {
            let bounded = Array1::from_iter(
                point
                    .iter()
                    .zip(&bounds)
                    .map(|(value, (low, high))| value.clamp(*low, *high)),
            );
            objective(&bounded)
        }
    };
    let mut report = cma_es(&bounded_objective, config)?;
    for (value, (low, high)) in report.x.iter_mut().zip(&bounds) {
        *value = value.clamp(*low, *high);
    }
    Ok(report)
}

/// Pure-Rust CMA-ES `FilterOptimizer`.
pub struct AutoeqCmaEsBackend {
    name: &'static str,
}

impl AutoeqCmaEsBackend {
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }

    fn optimize_with_completion(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        callback: Option<OptimProgressCallback>,
    ) -> CmaEsOptimizationCompletion {
        if lower.len() != x.len() || upper.len() != x.len() {
            return (
                Err((
                    format!(
                        "bounds dimension mismatch: x={}, lower={}, upper={}",
                        x.len(),
                        lower.len(),
                        upper.len(),
                    ),
                    f64::INFINITY,
                )),
                None,
            );
        }

        let mut objective = objective;
        let _ = install_constraints(self.capabilities(), &mut objective);
        let objective = Arc::new(objective);
        let obj_for_call = objective.clone();
        let f = move |x: &Array1<f64>| -> f64 {
            compute_fitness_penalties_ref(x.as_slice().unwrap(), &obj_for_call)
        };

        let bounds: Vec<(f64, f64)> = lower
            .iter()
            .zip(upper.iter())
            .map(|(&lo, &hi)| (lo, hi))
            .collect();
        let x0 = Array1::from(
            x.iter()
                .zip(bounds.iter())
                .map(|(&xi, (lo, hi))| xi.clamp(*lo, *hi))
                .collect::<Vec<_>>(),
        );

        let lambda = params.population.max(4);
        let mut cfg = CmaEsConfig {
            bounds,
            x0: Some(x0),
            sigma0: Some(0.20),
            lambda,
            mu: 0,
            maxeval: params.maxeval.max(lambda + 1),
            seed: params.seed,
            f_tol: params.atolerance.max(1e-12),
            stagnation_window: 80,
            parallel: ParallelConfig {
                enabled: !params.no_parallel,
                num_threads: if params.parallel_threads == 0 {
                    None
                } else {
                    Some(params.parallel_threads)
                },
            },
            ..Default::default()
        };

        if let Some(mut user_cb) = callback {
            cfg.callback = Some(Box::new(move |im: &CmaEsIntermediate| {
                user_cb(im.iter, im.fun, None)
            }));
        }

        match run_bounded_cma_es(&f, cfg) {
            Ok(report) => {
                if report.x.len() == x.len() {
                    x.copy_from_slice(report.x.as_slice().unwrap());
                }
                completed_report_result(report)
            }
            Err(e) => (
                Err((format!("CMA-ES setup failed: {:?}", e), f64::INFINITY)),
                None,
            ),
        }
    }
}

type CmaEsOptimizationCompletion = (
    Result<(String, f64), (String, f64)>,
    Option<OptimizerBackendCompletion>,
);

fn completed_report_result(report: CmaEsReport) -> CmaEsOptimizationCompletion {
    // The solver reports success on stagnation, step-size collapse, target,
    // or callback stops; only budget exhaustion reports success=false. Carry
    // the solver-side convergence verdict through the typed completion so a
    // converged run is not downgraded to best-effort. Budget stops keep the
    // legacy tuple classification downstream.
    let completion = report
        .success
        .then_some(OptimizerBackendCompletion::Converged);
    let label = if report.success {
        format!("AutoEQ CMA-ES: {}", report.message)
    } else {
        format!("AutoEQ CMA-ES: {} (not converged)", report.message)
    };
    (Ok((label, report.fun)), completion)
}

impl FilterOptimizer for AutoeqCmaEsBackend {
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
        params: &OptimParams,
    ) -> Option<OptimizerBudgetProfile> {
        if lower_bounds.len() != upper_bounds.len() {
            return None;
        }
        let lambda = params.population.max(4);
        let minimum = lambda.saturating_add(1);
        let solver_limit = params.maxeval.max(minimum);
        Some(OptimizerBudgetProfile::new(
            params.maxeval,
            Some(solver_limit),
            minimum,
            minimum,
            Some(lambda),
            Some(lambda),
            Some(solver_limit.saturating_sub(1) / lambda),
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
            nonlinear_ineq: false,
            nonlinear_eq: false,
            linear: false,
            iteration_callback: true,
            fallback_penalty_mode: PenaltyMode::Standard,
        }
    }

    fn optimize(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        callback: Option<OptimProgressCallback>,
    ) -> Result<(String, f64), (String, f64)> {
        self.optimize_with_completion(x, lower, upper, objective, params, callback)
            .0
    }

    fn optimize_with_typed_completion(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        callback: Option<OptimProgressCallback>,
    ) -> (
        FilterOptimizerOutput,
        Option<super::optimize::OptimizerBackendCompletion>,
    ) {
        let (result, completion) =
            self.optimize_with_completion(x, lower, upper, objective, params, callback);
        (
            FilterOptimizerOutput::Completed {
                result,
                pareto_report: None,
            },
            completion,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn report(success: bool, message: &str) -> CmaEsReport {
        CmaEsReport {
            x: Array1::from(vec![0.5]),
            fun: 0.25,
            success,
            message: message.to_string(),
            nfev: 1607,
            nit: 80,
            sigma: 0.01,
            restarts: 0,
            feasible: true,
            max_violation: 0.0,
        }
    }

    #[test]
    fn completed_stagnation_report_carries_typed_convergence() {
        let (result, completion) = completed_report_result(report(
            true,
            "stagnated for 80 generations below f_tol=1.000e-4",
        ));
        assert!(result.is_ok(), "completed solver report: {result:?}");
        let (status, objective) = result.unwrap();
        assert!(!status.contains("not converged"));
        assert_eq!(objective, 0.25);

        let mut evidence = super::super::optimize::OptimizerRunEvidence::from_backend_result(
            "autoeq:cmaes",
            Ok((status, objective)),
            &[0.5],
            &[0.0],
            &[1.0],
            20000,
            None,
        );
        evidence.apply_backend_completion(completion.unwrap());
        assert_eq!(
            evidence.termination,
            super::super::optimize::OptimizerTermination::Converged
        );
        assert!(evidence.converged);
        assert!(!evidence.best_effort);
        assert_eq!(
            evidence.confidence,
            super::super::optimize::OptimizerConfidence::High
        );
        assert_eq!(evidence.objective, Some(0.25));
    }

    #[test]
    fn budget_stop_keeps_legacy_evaluation_limit_classification() {
        let (result, completion) =
            completed_report_result(report(false, "maximum evaluations reached"));
        assert!(result.is_ok(), "completed solver report: {result:?}");
        assert!(
            completion.is_none(),
            "budget stop carries no typed completion"
        );
        let (status, objective) = result.unwrap();

        let evidence = super::super::optimize::OptimizerRunEvidence::from_backend_result(
            "autoeq:cmaes",
            Ok((status, objective)),
            &[0.5],
            &[0.0],
            &[1.0],
            20000,
            None,
        );
        assert_eq!(
            evidence.termination,
            super::super::optimize::OptimizerTermination::EvaluationLimit
        );
        assert!(!evidence.converged);
        assert!(evidence.best_effort);
    }
}
