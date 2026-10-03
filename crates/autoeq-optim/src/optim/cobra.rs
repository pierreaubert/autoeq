//! COBRA surrogate optimization with native RoomEQ inequality constraints.
//!
//! Register as `autoeq:cobra` or `cobra`. The solver uses a Halton initial
//! design rather than the incoming candidate, so saved-candidate starts are
//! unsupported. Progress callbacks run after surrogate infill evaluations;
//! the initial design and an active model search are not interrupted.
//! Internal true-function polishing is disabled so the reported evaluation
//! count covers each objective call and stop cannot enter that polish phase.

use super::backend::{
    AlgorithmType, ConstraintCapabilities, ConstraintInstallation, FilterOptimizer,
    NativeConstraint,
};
use super::constraints_install::{build_crossover_monotonicity_constraint, install_constraints};
use super::params::OptimParams;
use super::run_control::OptimizerBudgetProfile;
use super::{ObjectiveData, OptimProgressCallback, PenaltyMode, compute_fitness_penalties_ref};
use math_audio_optimisation::cobra::{CobraConfig, CobraConstraint, CobraIntermediate, cobra};
use ndarray::Array1;
use std::sync::Arc;

/// COBRA backend for bounded filter optimization with native inequalities.
#[derive(Debug)]
pub struct AutoeqCobraBackend {
    name: &'static str,
}

impl AutoeqCobraBackend {
    /// Create a backend with its canonical registry name.
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

fn initial_samples(dimensions: usize, maxeval: usize) -> usize {
    dimensions
        .saturating_mul(3)
        .saturating_add(1)
        .min(maxeval.max(1))
}

impl FilterOptimizer for AutoeqCobraBackend {
    fn name(&self) -> &'static str {
        self.name
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
            nonlinear_eq: false,
            linear: false,
            iteration_callback: true,
            fallback_penalty_mode: PenaltyMode::Disabled,
        }
    }
    fn evaluation_budget_profile(
        &self,
        lower: &[f64],
        upper: &[f64],
        params: &OptimParams,
    ) -> Option<OptimizerBudgetProfile> {
        if lower.is_empty() || lower.len() != upper.len() {
            return None;
        }
        let initial = initial_samples(lower.len(), params.maxeval);
        Some(OptimizerBudgetProfile::new(
            params.maxeval,
            Some(params.maxeval.max(1)),
            initial,
            initial,
            Some(1),
            None,
            None,
        ))
    }
    fn optimize(
        &self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        mut objective: ObjectiveData,
        params: &OptimParams,
        callback: Option<OptimProgressCallback>,
    ) -> Result<(String, f64), (String, f64)> {
        if x.is_empty() || lower.len() != x.len() || upper.len() != x.len() {
            return Err(("COBRA bounds dimension mismatch".into(), f64::INFINITY));
        }
        if lower
            .iter()
            .zip(upper)
            .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
        {
            return Err(("COBRA requires finite ordered bounds".into(), f64::INFINITY));
        }
        let installation = install_constraints(self.capabilities(), &mut objective);
        let mut constraints = match installation {
            ConstraintInstallation::Native(native) => {
                native.into_iter().map(native_to_cobra).collect::<Vec<_>>()
            }
            ConstraintInstallation::Penalty => Vec::new(),
        };
        if let Some(constraint) = build_crossover_monotonicity_constraint(&objective) {
            constraints.push(native_to_cobra(constraint));
        }
        let objective = Arc::new(objective);
        let f = |candidate: &Array1<f64>| {
            compute_fitness_penalties_ref(
                candidate.as_slice().expect("COBRA candidate is contiguous"),
                &objective,
            )
        };
        let callback = callback.map(|mut callback| {
            Box::new(move |progress: &CobraIntermediate| {
                match callback(progress.iter, progress.fun, None) {
                    crate::de::CallbackAction::Continue => {
                        math_audio_optimisation::CallbackAction::Continue
                    }
                    // This backend has no exact-resume state for Pause.
                    _ => math_audio_optimisation::CallbackAction::Stop,
                }
            }) as math_audio_optimisation::cobra::CobraCallback
        });
        let config = CobraConfig {
            bounds: lower.iter().zip(upper).map(|(&lo, &hi)| (lo, hi)).collect(),
            maxeval: params.maxeval.max(1),
            seed: params.seed,
            initial_samples: initial_samples(x.len(), params.maxeval),
            // Shared RoomEQ refinement owns true-function local polishing.
            // Native COBRA's polish has no callback and reports its whole cap.
            polish_fraction: 0.0,
            callback,
            ..Default::default()
        };
        match cobra(&f, &constraints, config) {
            Ok(report) => {
                if report.x.len() != x.len() || !report.fun.is_finite() || !report.feasible {
                    return Err((
                        format!(
                            "COBRA returned no finite feasible candidate (nfev={})",
                            report.nfev
                        ),
                        report.fun,
                    ));
                }
                x.copy_from_slice(report.x.as_slice().expect("COBRA result is contiguous"));
                // Callback stop is not evidence of numerical convergence.
                Ok((
                    format!(
                        "AutoEQ COBRA: {} (not converged, nfev={})",
                        report.message, report.nfev
                    ),
                    report.fun,
                ))
            }
            Err(error) => Err((format!("COBRA setup failed: {error:?}"), f64::INFINITY)),
        }
    }
}

fn native_to_cobra(constraint: NativeConstraint) -> CobraConstraint {
    // COBRA has no separate tolerance field. Preserve the installed tolerance
    // by expressing feasibility as g(x) - tolerance <= 0.
    CobraConstraint {
        fun: Arc::new(move |x| {
            (constraint.fun)(x.as_slice().expect("COBRA candidate is contiguous")) - constraint.tol
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_constraint_preserves_installed_tolerance() {
        let constraint = native_to_cobra(NativeConstraint {
            label: "test",
            fun: Box::new(|x| x[0]),
            tol: 1e-6,
        });
        assert!((constraint.fun)(&Array1::from(vec![0.5e-6])) < 0.0);
        assert!((constraint.fun)(&Array1::from(vec![2e-6])) > 0.0);
    }
}
