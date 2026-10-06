//! Pure-Rust COBYLA backend (no NLopt C-FFI).
//!
//! Wraps [`math_audio_optimisation::cobyla::cobyla`] in a [`FilterOptimizer`]
//! impl. Native nonlinear inequalities are honored directly — the four
//! [`crate::constraints`] are installed via
//! [`super::constraints_install::install_constraints`] in the same way the
//! NLopt backend does.
//!
//! This backend is registered as `"autoeq:cobyla"` (and resolves the
//! unprefixed legacy alias `"cobyla"`). The NLopt-backed `"nlopt:cobyla"`
//! remains available for A/B comparison while the wider migration to
//! pure-Rust optimizers is in flight.

use super::backend::{
    AlgorithmType, ConstraintCapabilities, ConstraintInstallation, FilterOptimizer,
    FilterOptimizerOutput, NativeConstraint,
};
use super::constraints_install::{build_crossover_monotonicity_constraint, install_constraints};
use super::optimize::OptimizerBackendCompletion;
use super::params::OptimParams;
use super::run_control::OptimizerBudgetProfile;
use super::{ObjectiveData, OptimProgressCallback, PenaltyMode, compute_fitness_penalties_ref};
use math_audio_optimisation::cobyla::{
    CobylaConfig, CobylaConstraint, CobylaConstraintFn, CobylaReport, CobylaRhoBegin,
    CobylaStopTols, CobylaTermination, cobyla_with_termination,
};
use ndarray::Array1;
use std::sync::Arc;

type CobylaOptimizationCompletion = (
    Result<(String, f64), (String, f64)>,
    Option<OptimizerBackendCompletion>,
);

fn completed_report_result(
    report: CobylaReport,
    termination: CobylaTermination,
) -> CobylaOptimizationCompletion {
    let completion = match termination {
        CobylaTermination::Success
        | CobylaTermination::StopValueReached
        | CobylaTermination::FunctionToleranceReached
        | CobylaTermination::ParameterToleranceReached => OptimizerBackendCompletion::Converged,
        CobylaTermination::EvaluationLimit => OptimizerBackendCompletion::EvaluationLimit,
        CobylaTermination::RoundoffLimited | CobylaTermination::Failure => {
            OptimizerBackendCompletion::NonConverged
        }
    };
    let label = if report.success {
        format!("AutoEQ COBYLA: {}", report.message)
    } else {
        format!("AutoEQ COBYLA: {} (not converged)", report.message)
    };

    // A returned report is a completed solver call with a best candidate;
    // preserve the legacy tuple's success channel and carry stop quality in
    // the typed completion. Invalid candidates are rejected downstream.
    (Ok((label, report.fun)), Some(completion))
}

/// Pure-Rust COBYLA `FilterOptimizer`.
pub struct AutoeqCobylaBackend {
    name: &'static str,
}

impl AutoeqCobylaBackend {
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
    ) -> CobylaOptimizationCompletion {
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

        // Native constraint installation matches the prior NLopt path.
        let mut objective = objective;
        let installation = install_constraints(self.capabilities(), &mut objective);
        let crossover = build_crossover_monotonicity_constraint(&objective);
        let mut constraints: Vec<CobylaConstraint> = Vec::new();
        if let ConstraintInstallation::Native(ncs) = installation {
            for c in ncs {
                constraints.push(native_to_cobyla(c));
            }
        }
        if let Some(c) = crossover {
            constraints.push(native_to_cobyla(c));
        }

        let obj = Arc::new(objective);
        let obj_for_call = obj.clone();
        let f = move |x: &Array1<f64>| -> f64 {
            compute_fitness_penalties_ref(x.as_slice().unwrap(), &obj_for_call)
        };
        // Small initial radii keep standalone COBYLA refinement near the
        // supplied candidate; fixed dimensions still need a nonzero simplex.
        let rho_per_dim: Vec<f64> = lower
            .iter()
            .zip(upper.iter())
            .map(|(lo, hi)| {
                let span = (hi - lo).max(0.0);
                if span <= 0.0 {
                    1e-6
                } else {
                    (span * 0.05).max(1e-6)
                }
            })
            .collect();
        let bounds: Vec<(f64, f64)> = lower
            .iter()
            .zip(upper.iter())
            .map(|(&lo, &hi)| (lo, hi))
            .collect();
        // Clip the seed in case an upstream refine landed on a bound.
        let x0: Vec<f64> = x
            .iter()
            .zip(bounds.iter())
            .map(|(&xi, (lo, hi))| xi.clamp(*lo, *hi))
            .collect();
        let cfg = CobylaConfig {
            x0: Array1::from(x0),
            bounds,
            rho_begin: CobylaRhoBegin::PerDim(rho_per_dim),
            maxeval: params.maxeval.max(1),
            stop_tol: CobylaStopTols::default(),
        };

        match cobyla_with_termination(&f, &constraints, cfg) {
            Ok((report, termination)) => {
                if report.x.len() == x.len() {
                    x.copy_from_slice(report.x.as_slice().unwrap());
                }
                completed_report_result(report, termination)
            }
            Err(error) => (
                Err((format!("COBYLA setup failed: {error:?}"), f64::INFINITY)),
                None,
            ),
        }
    }
}

impl FilterOptimizer for AutoeqCobylaBackend {
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
        (lower_bounds.len() == upper_bounds.len()).then(|| {
            OptimizerBudgetProfile::new(
                params.maxeval,
                Some(params.maxeval.max(1)),
                1,
                1,
                None,
                None,
                None,
            )
        })
    }
    fn library(&self) -> &'static str {
        "AutoEQ"
    }
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::Local
    }
    fn capabilities(&self) -> ConstraintCapabilities {
        ConstraintCapabilities {
            nonlinear_ineq: true,
            // The underlying solver does not handle equalities; matches
            // NLopt's COBYLA in autoeq's previous behaviour.
            nonlinear_eq: false,
            linear: true,
            iteration_callback: false,
            fallback_penalty_mode: PenaltyMode::Disabled,
        }
    }
    fn optimize(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        _callback: Option<OptimProgressCallback>,
    ) -> Result<(String, f64), (String, f64)> {
        self.optimize_with_completion(x, lower, upper, objective, params)
            .0
    }

    fn optimize_with_report(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        callback: Option<OptimProgressCallback>,
    ) -> FilterOptimizerOutput {
        self.optimize_with_typed_completion(x, lower, upper, objective, params, callback)
            .0
    }

    fn optimize_with_typed_completion(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        objective: ObjectiveData,
        params: &OptimParams,
        _callback: Option<OptimProgressCallback>,
    ) -> (
        FilterOptimizerOutput,
        Option<super::OptimizerBackendCompletion>,
    ) {
        let (result, completion) =
            self.optimize_with_completion(x, lower, upper, objective, params);
        (
            FilterOptimizerOutput::Completed {
                result,
                pareto_report: None,
            },
            completion,
        )
    }
}

/// Convert a unified `NativeConstraint` (closure over `&[f64]`) to the
/// `CobylaConstraint` shape (closure over `&Array1<f64>`).
fn native_to_cobyla(c: NativeConstraint) -> CobylaConstraint {
    let f = c.fun;
    let wrapped: CobylaConstraintFn = Arc::new(move |x: &Array1<f64>| f(x.as_slice().unwrap()));
    CobylaConstraint { fun: wrapped }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math_audio_optimisation::cobyla::CobylaTermination;
    use ndarray::Array1;

    #[test]
    fn completed_roundoff_report_remains_legacy_ok_and_typed_best_effort() {
        let report = CobylaReport {
            x: Array1::from(vec![0.5]),
            fun: 0.25,
            success: false,
            message: "RoundoffLimited".to_string(),
            nfev: 7,
        };

        let (result, completion) =
            completed_report_result(report, CobylaTermination::RoundoffLimited);
        assert!(result.is_ok(), "completed solver report: {result:?}");
        let (status, objective) = result.unwrap();
        assert!(status.contains("RoundoffLimited (not converged)"));
        assert_eq!(objective, 0.25);

        let mut evidence = super::super::optimize::OptimizerRunEvidence::from_backend_result(
            "autoeq:cobyla",
            Ok((status, objective)),
            &[0.5],
            &[0.0],
            &[1.0],
            100,
            None,
        );
        evidence.apply_backend_completion(completion.unwrap());
        assert_eq!(
            evidence.termination,
            super::super::optimize::OptimizerTermination::NonConverged
        );
        assert!(!evidence.converged);
        assert!(evidence.best_effort);
        assert_eq!(
            evidence.confidence,
            super::super::optimize::OptimizerConfidence::Low
        );
        assert_eq!(evidence.objective, Some(0.25));
    }
}
