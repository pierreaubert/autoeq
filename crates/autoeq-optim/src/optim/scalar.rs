//! Shared bounded scalar-objective optimizer dispatch.
//!
//! RoomEQ has a few optimisation problems that are not PEQ filter fitting:
//! GD-Opt v2, multi-sub all-pass, and other small bounded black-box searches.
//! They should still honor the user-selected optimizer algorithm without each
//! caller wiring algorithm-specific code.

use crate::optim::registry;
use math_audio_optimisation::cobra::{CobraConfig, CobraIntermediate, cobra};
use math_audio_optimisation::cobyla::{CobylaRhoBegin, cobyla};
use math_audio_optimisation::{
    CmaEsConfig, CobylaConfig, CobylaStopTols, DEConfigBuilder, Init, IsresConfig, Mutation,
    Strategy, differential_evolution, isres,
};
use ndarray::Array1;
use std::str::FromStr;

/// Options for bounded scalar minimization.
#[derive(Debug, Clone)]
pub struct ScalarOptimConfig {
    /// User-facing optimizer name, e.g. `"autoeq:cmaes"` or `"autoeq:de"`.
    pub algorithm: String,
    /// Maximum iteration/evaluation budget. DE interprets this as generations;
    /// evaluation-budget optimizers interpret it as objective evaluations.
    pub max_iter: usize,
    /// Population size or population multiplier, depending on backend.
    pub population: usize,
    /// Relative convergence tolerance.
    pub tolerance: f64,
    /// Absolute convergence tolerance.
    pub atolerance: f64,
    /// DE mutation strategy. Ignored by non-DE backends.
    pub strategy: String,
    /// Stochastic seed (`None` selects [`crate::DEFAULT_SEED`]).
    pub seed: Option<u64>,
}

impl Default for ScalarOptimConfig {
    fn default() -> Self {
        Self {
            algorithm: "autoeq:cmaes".to_string(),
            max_iter: 10_000,
            population: 20,
            tolerance: 1e-8,
            atolerance: 1e-8,
            strategy: "lshade".to_string(),
            seed: None,
        }
    }
}

/// Result of a bounded scalar minimization.
#[derive(Debug, Clone)]
pub struct ScalarOptimResult {
    /// Actual objective calls, excluding refused attempts and later validation.
    pub evaluation_count: usize,
    /// Hard search cap when a run control was supplied.
    pub evaluation_limit: Option<usize>,
    /// Elapsed search time in milliseconds, excluding caller-side validation.
    pub elapsed_ms: f64,
    /// Best parameter vector found.
    pub x: Vec<f64>,
    /// Objective value at [`Self::x`].
    pub fun: f64,
    /// Canonical resolved algorithm name.
    pub algorithm: String,
    /// Whether the backend reported convergence.
    pub success: bool,
    /// Human-readable backend status.
    pub message: String,
}

/// Minimize `objective` over box bounds using a configured AutoEQ optimizer.
pub fn optimize_bounded_scalar<F>(
    bounds: &[(f64, f64)],
    initial: &[f64],
    config: &ScalarOptimConfig,
    objective: F,
) -> Result<ScalarOptimResult, String>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    optimize_bounded_scalar_with_callback(bounds, initial, config, objective, None)
}

/// Cancel DE/CMA-ES at generation boundaries and COBRA after surrogate infill.
/// A stopped run returns an error, never a deliverable best-effort candidate.
/// Backends without native callbacks reject callback requests before scoring.
/// This does not interrupt an active evaluation, initial design, or model search.
pub fn optimize_bounded_scalar_with_callback<F>(
    bounds: &[(f64, f64)],
    initial: &[f64],
    config: &ScalarOptimConfig,
    objective: F,
    callback: Option<super::OptimProgressCallback>,
) -> Result<ScalarOptimResult, String>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    if callback.is_some()
        && registry::resolve(&config.algorithm).is_some_and(|backend| {
            !matches!(
                backend.name(),
                "autoeq:cmaes" | "autoeq:de" | "autoeq:cobra"
            )
        })
    {
        return Err(format!(
            "native scalar cancellation is not supported by {}",
            config.algorithm
        ));
    }
    optimize_bounded_scalar_with_control(bounds, initial, config, objective, callback, None)
}

/// Minimize a scalar objective with shared strict budgets and cancellation.
///
/// Backend iteration/generation limits retain their existing interpretation.
/// The optional control imposes a separate hard cap on actual objective calls.
/// Budget exhaustion returns the best finite scored candidate as best effort;
/// user cancellation or deadline expiry never returns a deliverable candidate.
///
/// # Errors
///
/// Returns an error for invalid problems, unsupported dispatch/callbacks,
/// cancellation, deadlines, or a run without a finite scored candidate.
pub fn optimize_bounded_scalar_with_control<F>(
    bounds: &[(f64, f64)],
    initial: &[f64],
    config: &ScalarOptimConfig,
    objective: F,
    mut callback: Option<super::OptimProgressCallback>,
    control: Option<&super::run_control::OptimizerRunControl>,
) -> Result<ScalarOptimResult, String>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    validate_problem(bounds, initial)?;

    let backend = registry::resolve(&config.algorithm)
        .ok_or_else(|| format!("Unknown algorithm: {}", config.algorithm))?;
    let canonical = backend.name().to_string();

    let stopped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let native_callback = matches!(
        canonical.as_str(),
        "autoeq:cmaes" | "autoeq:de" | "autoeq:cobra"
    );
    // Backends without native task callbacks still poll an observer after each
    // admitted objective. Once stopped, no further expensive score is computed.
    let evaluation_callback = if native_callback {
        None
    } else {
        callback.take().map(Mutex::new)
    };
    let callback = (callback.is_some() || (control.is_some() && native_callback)).then(|| {
        let stopped = stopped.clone();
        let control = control.cloned();
        Box::new(move |iteration, loss, preference| {
            if control.as_ref().is_some_and(|control| {
                let snapshot = control.snapshot();
                snapshot.cancellation_requested
                    || snapshot.deadline_reached
                    || control.remaining_evaluations() == 0
            }) {
                return crate::de::CallbackAction::Stop;
            }
            let action = callback
                .as_mut()
                .map_or(crate::de::CallbackAction::Continue, |callback| {
                    callback(iteration, loss, preference)
                });
            if matches!(action, crate::de::CallbackAction::Stop) {
                stopped.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            action
        }) as super::OptimProgressCallback
    });

    let evaluation_limit = control.map(|control| control.remaining_evaluations());
    let evaluations = AtomicUsize::new(0);
    let best = Mutex::new(None::<(Vec<f64>, f64)>);
    let start = std::time::Instant::now();
    let f = |x: &Array1<f64>| {
        if stopped.load(Ordering::Relaxed) {
            return f64::INFINITY;
        }
        let score = || {
            let evaluation = evaluations.fetch_add(1, Ordering::Relaxed) + 1;
            let value = objective(x.as_slice().unwrap());
            if control.is_some() && value.is_finite() {
                let mut best = best
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if best.as_ref().is_none_or(|(_, previous)| value < *previous) {
                    *best = Some((x.to_vec(), value));
                }
            }
            if let Some(callback) = &evaluation_callback {
                let action = callback
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)(
                    evaluation, value, None
                );
                if matches!(action, crate::de::CallbackAction::Stop) {
                    stopped.store(true, Ordering::Relaxed);
                }
            }
            value
        };
        control.map_or_else(score, |control| {
            control
                .evaluate(super::run_control::EvaluationStage::Search, 1, score)
                .unwrap_or(f64::INFINITY)
        })
    };
    let x0 = || clamp_initial(initial, bounds);

    let result = match canonical.as_str() {
        "autoeq:cmaes" => optimize_cmaes(&canonical, bounds, x0(), config, &f, callback),
        "autoeq:de" => optimize_de(&canonical, bounds, x0(), config, &f, callback),
        "autoeq:cobra" => optimize_cobra(&canonical, bounds, config, &f, callback),
        "autoeq:cobyla" => optimize_cobyla(&canonical, bounds, x0(), config, &f),
        "autoeq:isres" => optimize_isres(&canonical, bounds, x0(), config, &f),
        other => Err(format!(
            "Algorithm '{}' is registered for PEQ filter optimization but is not supported for bounded scalar RoomEQ objectives",
            other
        )),
    };
    if stopped.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(format!("{canonical} scalar optimization stopped by user"));
    }
    if control.is_some_and(|control| {
        let snapshot = control.snapshot();
        snapshot.cancellation_requested || snapshot.deadline_reached
    }) {
        return Err(format!(
            "{canonical} scalar optimization cancelled or deadline reached"
        ));
    }
    let capped = control.is_some_and(|control| control.remaining_evaluations() == 0);
    let mut result = if capped {
        let (x, fun) = best
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .ok_or_else(|| format!("{canonical} evaluation budget has no finite candidate"))?;
        ScalarOptimResult {
            x,
            fun,
            algorithm: canonical,
            success: false,
            message: "strict objective evaluation budget exhausted".into(),
            evaluation_count: 0,
            evaluation_limit: None,
            elapsed_ms: 0.0,
        }
    } else {
        result?
    };
    result.evaluation_count = evaluations.load(Ordering::Relaxed);
    result.evaluation_limit = evaluation_limit;
    result.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(result)
}

fn validate_problem(bounds: &[(f64, f64)], initial: &[f64]) -> Result<(), String> {
    if bounds.is_empty() {
        return Err("scalar optimizer requires at least one parameter".to_string());
    }
    if bounds.len() != initial.len() {
        return Err(format!(
            "scalar optimizer dimension mismatch: bounds={}, initial={}",
            bounds.len(),
            initial.len()
        ));
    }
    for (idx, (lo, hi)) in bounds.iter().enumerate() {
        if lo > hi {
            return Err(format!(
                "invalid scalar optimizer bounds at {}: lower {} > upper {}",
                idx, lo, hi
            ));
        }
    }
    Ok(())
}

fn clamp_initial(initial: &[f64], bounds: &[(f64, f64)]) -> Array1<f64> {
    Array1::from(
        initial
            .iter()
            .zip(bounds.iter())
            .map(|(&x, (lo, hi))| x.clamp(*lo, *hi))
            .collect::<Vec<_>>(),
    )
}

fn optimize_cmaes<F>(
    canonical: &str,
    bounds: &[(f64, f64)],
    x0: Array1<f64>,
    config: &ScalarOptimConfig,
    f: &F,
    callback: Option<super::OptimProgressCallback>,
) -> Result<ScalarOptimResult, String>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    let lambda = config.population.max(4);
    let report = super::cmaes::run_bounded_cma_es(
        f,
        CmaEsConfig {
            bounds: bounds.to_vec(),
            x0: Some(x0),
            sigma0: Some(0.20),
            lambda,
            mu: 0,
            maxeval: config.max_iter.max(lambda + 1),
            seed: Some(config.seed.unwrap_or(crate::DEFAULT_SEED)),
            f_tol: config.atolerance.max(1e-12),
            stagnation_window: 80,
            callback: callback.map(|mut callback| {
                Box::new(
                    move |progress: &math_audio_optimisation::cmaes::CmaEsIntermediate| {
                        callback(progress.iter, progress.fun, None)
                    },
                ) as math_audio_optimisation::cmaes::CmaEsCallback
            }),
            ..Default::default()
        },
    )
    .map_err(|e| format!("CMA-ES failed: {e:?}"))?;

    Ok(ScalarOptimResult {
        evaluation_count: 0,
        evaluation_limit: None,
        elapsed_ms: 0.0,
        x: report.x.to_vec(),
        fun: report.fun,
        algorithm: canonical.to_string(),
        success: report.success,
        message: report.message,
    })
}

fn optimize_de<F>(
    canonical: &str,
    bounds: &[(f64, f64)],
    x0: Array1<f64>,
    config: &ScalarOptimConfig,
    f: &F,
    callback: Option<super::OptimProgressCallback>,
) -> Result<ScalarOptimResult, String>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    let strategy = Strategy::from_str(&config.strategy).unwrap_or(Strategy::LShadeBin);
    let mut builder = DEConfigBuilder::new()
        .maxiter(config.max_iter)
        .popsize(config.population.max(4))
        .tol(config.tolerance)
        .atol(config.atolerance)
        .strategy(strategy)
        .mutation(Mutation::Range { min: 0.4, max: 1.2 })
        .init(Init::LatinHypercube)
        .x0(x0)
        .disp(false);
    builder = builder.seed(config.seed.unwrap_or(crate::DEFAULT_SEED));
    if let Some(mut callback) = callback {
        builder = builder.callback(Box::new(move |progress| {
            callback(progress.iter, progress.fun, None)
        }));
    }

    let de_config = builder
        .build()
        .map_err(|e| format!("DE config error: {e:?}"))?;
    let report =
        differential_evolution(f, bounds, de_config).map_err(|e| format!("DE failed: {e:?}"))?;

    Ok(ScalarOptimResult {
        evaluation_count: 0,
        evaluation_limit: None,
        elapsed_ms: 0.0,
        x: report.x.to_vec(),
        fun: report.fun,
        algorithm: canonical.to_string(),
        success: report.success,
        message: report.message,
    })
}

fn optimize_cobra<F>(
    canonical: &str,
    bounds: &[(f64, f64)],
    config: &ScalarOptimConfig,
    f: &F,
    callback: Option<super::OptimProgressCallback>,
) -> Result<ScalarOptimResult, String>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    if bounds
        .iter()
        .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite())
    {
        return Err("COBRA requires finite bounds".to_string());
    }
    let report = cobra(
        f,
        &[],
        CobraConfig {
            bounds: bounds.to_vec(),
            maxeval: config.max_iter.max(1),
            seed: Some(config.seed.unwrap_or(crate::DEFAULT_SEED)),
            // The native initial design does not accept an incoming candidate.
            // Disable callback-free polish and its estimated evaluation count.
            polish_fraction: 0.0,
            callback: callback.map(|mut callback| {
                Box::new(move |progress: &CobraIntermediate| {
                    callback(progress.iter, progress.fun, None)
                }) as math_audio_optimisation::cobra::CobraCallback
            }),
            ..Default::default()
        },
    )
    .map_err(|error| format!("COBRA failed: {error:?}"))?;
    if !report.fun.is_finite() || report.x.iter().any(|x| !x.is_finite()) {
        return Err("COBRA returned no finite scalar candidate".to_string());
    }
    Ok(ScalarOptimResult {
        evaluation_count: 0,
        evaluation_limit: None,
        elapsed_ms: 0.0,
        x: report.x.to_vec(),
        fun: report.fun,
        algorithm: canonical.to_string(),
        // Budget exhaustion or callback stop does not establish convergence.
        success: false,
        message: format!("{} (not converged, nfev={})", report.message, report.nfev),
    })
}

fn optimize_cobyla<F>(
    canonical: &str,
    bounds: &[(f64, f64)],
    x0: Array1<f64>,
    config: &ScalarOptimConfig,
    f: &F,
) -> Result<ScalarOptimResult, String>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    let report = cobyla(
        f,
        &[],
        CobylaConfig {
            x0,
            bounds: bounds.to_vec(),
            rho_begin: CobylaRhoBegin::All(0.2),
            maxeval: config.max_iter,
            stop_tol: CobylaStopTols {
                ftol_abs: config.atolerance,
                ftol_rel: config.tolerance,
                ..Default::default()
            },
        },
    )
    .map_err(|e| format!("COBYLA failed: {e:?}"))?;

    Ok(ScalarOptimResult {
        evaluation_count: 0,
        evaluation_limit: None,
        elapsed_ms: 0.0,
        x: report.x.to_vec(),
        fun: report.fun,
        algorithm: canonical.to_string(),
        success: report.success,
        message: report.message,
    })
}

fn optimize_isres<F>(
    canonical: &str,
    bounds: &[(f64, f64)],
    x0: Array1<f64>,
    config: &ScalarOptimConfig,
    f: &F,
) -> Result<ScalarOptimResult, String>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    let mu = config.population.max(2);
    let report = isres(
        f,
        &[],
        IsresConfig {
            bounds: bounds.to_vec(),
            x0: Some(x0),
            mu,
            lambda: 0,
            maxeval: config.max_iter.max(mu * 7),
            seed: Some(config.seed.unwrap_or(crate::DEFAULT_SEED)),
            f_tol: config.atolerance.max(1e-12),
            ..Default::default()
        },
    )
    .map_err(|e| format!("ISRES failed: {e:?}"))?;

    Ok(ScalarOptimResult {
        evaluation_count: 0,
        evaluation_limit: None,
        elapsed_ms: 0.0,
        x: report.x.to_vec(),
        fun: report.fun,
        algorithm: canonical.to_string(),
        success: report.success,
        message: report.message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_scalar_evaluation_caps_are_independent_of_backend_population_floors() {
        use super::super::run_control::OptimizerRunControl;
        use std::sync::atomic::{AtomicUsize, Ordering};
        for algorithm in ["autoeq:de", "autoeq:cmaes", "autoeq:cobyla", "autoeq:isres"] {
            let control = OptimizerRunControl::new(std::num::NonZeroUsize::new(3).unwrap());
            let calls = AtomicUsize::new(0);
            let config = ScalarOptimConfig {
                algorithm: algorithm.into(),
                max_iter: 20,
                population: 20,
                ..Default::default()
            };
            let result = optimize_bounded_scalar_with_control(
                &[(-1.0, 1.0)],
                &[0.8],
                &config,
                |candidate| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    (candidate[0] - 0.25).powi(2)
                },
                None,
                Some(&control),
            )
            .unwrap();
            assert!(calls.load(Ordering::Relaxed) <= 3, "{algorithm}");
            assert_eq!(result.evaluation_count, calls.load(Ordering::Relaxed));
            assert_eq!(result.evaluation_limit, Some(3));
            assert_eq!(
                control.snapshot().evaluations_completed,
                result.evaluation_count
            );
            assert!(result.fun.is_finite());
            assert!(!result.success);
        }
    }

    #[test]
    fn scalar_telemetry_reports_remaining_stage_quota() {
        use super::super::run_control::{EvaluationStage, OptimizerRunControl};
        let root = OptimizerRunControl::new(std::num::NonZeroUsize::new(100).unwrap());
        let stage = root.with_stage_budget(std::num::NonZeroUsize::new(5).unwrap());
        for _ in 0..2 {
            stage.evaluate(EvaluationStage::Search, 1, || 0.0).unwrap();
        }
        let result = optimize_bounded_scalar_with_control(
            &[(-1.0, 1.0)],
            &[0.5],
            &ScalarOptimConfig {
                algorithm: "autoeq:de".into(),
                max_iter: 20,
                ..Default::default()
            },
            |x| x[0].powi(2),
            None,
            Some(&stage),
        )
        .unwrap();
        assert_eq!(result.evaluation_count, 3);
        assert_eq!(result.evaluation_limit, Some(3));
        assert_eq!(root.snapshot().evaluations_completed, 5);
    }

    #[test]
    fn controlled_scalar_cancellation_emits_no_candidate() {
        use super::super::run_control::OptimizerRunControl;
        let control = OptimizerRunControl::new(std::num::NonZeroUsize::new(100).unwrap());
        control.request_cancel();
        let error = optimize_bounded_scalar_with_control(
            &[(-1.0, 1.0)],
            &[0.8],
            &ScalarOptimConfig {
                algorithm: "autoeq:de".into(),
                max_iter: 20,
                ..Default::default()
            },
            |_| panic!("cancelled run must not score"),
            None,
            Some(&control),
        )
        .unwrap_err();
        assert!(error.contains("cancelled"), "{error}");
        assert_eq!(control.snapshot().evaluations_completed, 0);
    }

    #[test]
    fn controlled_cobyla_observer_stops_before_another_objective() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let error = optimize_bounded_scalar_with_control(
            &[(-1.0, 1.0)],
            &[0.8],
            &ScalarOptimConfig {
                algorithm: "autoeq:cobyla".into(),
                max_iter: 100,
                ..Default::default()
            },
            |candidate| {
                calls.fetch_add(1, Ordering::Relaxed);
                candidate[0].powi(2)
            },
            Some(Box::new(|_, _, _| crate::de::CallbackAction::Stop)),
            None,
        )
        .unwrap_err();
        assert!(error.contains("stopped by user"));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn native_scalar_stop_is_not_a_successful_candidate() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        for algorithm in ["autoeq:de", "cma-es", "cobra"] {
            let evaluations = AtomicUsize::new(0);
            let calls = Arc::new(AtomicUsize::new(0));
            let callback_calls = calls.clone();
            let result = optimize_bounded_scalar_with_callback(
                &[(-2.0, 2.0), (-2.0, 2.0)],
                &[1.5, 1.5],
                &ScalarOptimConfig {
                    algorithm: algorithm.into(),
                    max_iter: 1000,
                    population: 8,
                    seed: Some(7),
                    ..Default::default()
                },
                |x| {
                    evaluations.fetch_add(1, Ordering::Relaxed);
                    quadratic(x)
                },
                Some(Box::new(move |_, loss, preference| {
                    assert!(loss.is_finite());
                    assert_eq!(preference, None);
                    callback_calls.fetch_add(1, Ordering::Relaxed);
                    crate::de::CallbackAction::Stop
                })),
            );
            assert!(
                result.unwrap_err().contains("stopped by user"),
                "{algorithm}"
            );
            assert_eq!(calls.load(Ordering::Relaxed), 1, "{algorithm}");
            assert!(evaluations.load(Ordering::Relaxed) < 100, "{algorithm}");
        }
    }

    #[test]
    fn unsupported_native_scalar_stop_is_rejected_before_scoring() {
        for algorithm in ["autoeq:cobyla", "autoeq:isres"] {
            let result = optimize_bounded_scalar_with_callback(
                &[(-1.0, 1.0)],
                &[0.0],
                &ScalarOptimConfig {
                    algorithm: algorithm.into(),
                    ..Default::default()
                },
                |_| panic!("unsupported cancellation must not launch search"),
                Some(Box::new(|_, _, _| crate::de::CallbackAction::Stop)),
            );
            assert!(result.unwrap_err().contains("not supported"));
        }
    }

    #[test]
    fn continuing_scalar_callback_preserves_seeded_search() {
        for algorithm in ["autoeq:de", "autoeq:cmaes", "autoeq:cobra"] {
            let config = ScalarOptimConfig {
                algorithm: algorithm.into(),
                max_iter: 80,
                population: 8,
                seed: Some(7),
                strategy: "rand1bin".into(),
                ..Default::default()
            };
            let bounds = &[(-2.0, 2.0), (-2.0, 2.0)];
            let initial = &[1.5, 1.5];
            let plain = optimize_bounded_scalar(bounds, initial, &config, quadratic).unwrap();
            let repeated = optimize_bounded_scalar(bounds, initial, &config, quadratic).unwrap();
            assert_eq!(plain.x, repeated.x, "plain repeat {algorithm}");
            let observed = optimize_bounded_scalar_with_callback(
                bounds,
                initial,
                &config,
                quadratic,
                Some(Box::new(|_, _, _| crate::de::CallbackAction::Continue)),
            )
            .unwrap();
            assert_eq!(plain.x, observed.x, "{algorithm}");
            assert_eq!(plain.fun, observed.fun, "{algorithm}");
            assert_eq!(plain.success, observed.success, "{algorithm}");
            assert_eq!(plain.message, observed.message, "{algorithm}");
        }
    }

    #[test]
    fn seeded_lshade_scalar_search_is_reproducible() {
        // The pinned archive uses an unseeded RNG when replacing full entries.
        // Keep this regression active until the dependency honours its seed.
        let config = ScalarOptimConfig {
            algorithm: "autoeq:de".into(),
            strategy: "lshade".into(),
            max_iter: 80,
            population: 8,
            seed: Some(7),
            ..Default::default()
        };
        let run = || {
            optimize_bounded_scalar(&[(-2.0, 2.0), (-2.0, 2.0)], &[1.5, 1.5], &config, quadratic)
                .unwrap()
        };
        let first = run();
        for _ in 0..3 {
            let repeated = run();
            assert_eq!(
                first.x, repeated.x,
                "seeded L-SHADE parameter reproducibility"
            );
            assert_eq!(
                first.fun, repeated.fun,
                "seeded L-SHADE loss reproducibility"
            );
        }
    }

    #[test]
    fn cobra_rejects_nonfinite_bounds_before_scoring_or_clamping() {
        for bounds in [
            [(f64::NAN, 1.0)],
            [(-1.0, f64::INFINITY)],
            [(f64::NEG_INFINITY, 1.0)],
        ] {
            let result = optimize_bounded_scalar(
                &bounds,
                &[0.0],
                &ScalarOptimConfig {
                    algorithm: "cobra".into(),
                    ..Default::default()
                },
                |_| panic!("nonfinite COBRA bounds must be refused before scoring"),
            );
            assert!(result.unwrap_err().contains("finite bounds"));
        }
    }

    #[test]
    fn cobra_solves_bounded_scalar_quadratic_with_an_evaluation_cap() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let result = optimize_bounded_scalar(
            &[(-2.0, 2.0), (-2.0, 2.0)],
            &[1.5, 1.5],
            &ScalarOptimConfig {
                algorithm: "cobra".into(),
                max_iter: 40,
                seed: Some(7),
                ..Default::default()
            },
            |x| {
                calls.fetch_add(1, Ordering::Relaxed);
                quadratic(x)
            },
        )
        .unwrap();
        assert_eq!(result.algorithm, "autoeq:cobra");
        assert!(result.fun < 1e-2, "fun={}", result.fun);
        assert!(result.x.iter().all(|x| (-2.0..=2.0).contains(x)));
        assert_eq!(calls.load(Ordering::Relaxed), 40);
        assert!(!result.success);
    }

    fn quadratic(x: &[f64]) -> f64 {
        (x[0] - 0.25).powi(2) + (x[1] + 0.5).powi(2)
    }

    fn assert_solves_quadratic(algo: &str) {
        let result = optimize_bounded_scalar(
            &[(-2.0, 2.0), (-2.0, 2.0)],
            &[1.5, 1.5],
            &ScalarOptimConfig {
                algorithm: algo.to_string(),
                max_iter: 400,
                population: 12,
                seed: Some(7),
                ..Default::default()
            },
            quadratic,
        )
        .expect("optimizer should run");

        assert!(result.fun < 1e-2, "{algo} fun={}", result.fun);
        assert!(result.algorithm.starts_with("autoeq:"));
    }

    #[test]
    fn cmaes_solves_bounded_scalar_quadratic() {
        assert_solves_quadratic("autoeq:cmaes");
    }

    #[test]
    fn cmaes_scores_and_returns_exact_parameter_rails() {
        let lower = -4.199999999999999_f64;
        let upper = 0.7_f64;
        let result = optimize_bounded_scalar(
            &[(lower, upper)],
            &[upper],
            &ScalarOptimConfig {
                algorithm: "autoeq:cmaes".into(),
                max_iter: 32,
                population: 4,
                seed: Some(7),
                ..Default::default()
            },
            |point| {
                assert!(point[0] >= lower && point[0] <= upper);
                -point[0]
            },
        )
        .expect("a bounded optimum on the upper rail must remain usable");
        assert_eq!(result.x, [upper]);
        assert_eq!(result.fun, -upper);
    }

    #[test]
    fn cmaes_alias_resolves_for_bounded_scalar() {
        let result = optimize_bounded_scalar(
            &[(-1.0, 1.0)],
            &[0.9],
            &ScalarOptimConfig {
                algorithm: "cma-es".to_string(),
                max_iter: 200,
                population: 8,
                seed: Some(3),
                ..Default::default()
            },
            |x| (x[0] + 0.2).powi(2),
        )
        .expect("optimizer should run");

        assert_eq!(result.algorithm, "autoeq:cmaes");
        assert!(result.fun < 1e-2, "fun={}", result.fun);
    }

    #[test]
    fn de_solves_bounded_scalar_quadratic() {
        assert_solves_quadratic("autoeq:de");
    }

    #[test]
    fn cobyla_solves_bounded_scalar_quadratic() {
        assert_solves_quadratic("autoeq:cobyla");
    }

    #[test]
    fn isres_solves_bounded_scalar_quadratic() {
        assert_solves_quadratic("autoeq:isres");
    }

    #[test]
    fn invalid_algorithm_returns_error() {
        let result = optimize_bounded_scalar(
            &[(0.0, 1.0)],
            &[0.5],
            &ScalarOptimConfig {
                algorithm: "autoeq:nsga2".to_string(),
                ..Default::default()
            },
            |x| x[0],
        );
        assert!(result.is_err(), "NSGA2 should not be supported for scalar");
        assert!(result.unwrap_err().contains("not supported"));
    }

    #[test]
    fn unknown_algorithm_returns_error() {
        let result = optimize_bounded_scalar(
            &[(0.0, 1.0)],
            &[0.5],
            &ScalarOptimConfig {
                algorithm: "no-such-algo".to_string(),
                ..Default::default()
            },
            |x| x[0],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown algorithm"));
    }

    #[test]
    fn validate_problem_rejects_empty_bounds() {
        let result = optimize_bounded_scalar(
            &[],
            &[],
            &ScalarOptimConfig {
                algorithm: "autoeq:de".to_string(),
                ..Default::default()
            },
            |x| x[0],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one parameter"));
    }

    #[test]
    fn validate_problem_rejects_dimension_mismatch() {
        let result = optimize_bounded_scalar(
            &[(0.0, 1.0)],
            &[0.5, 0.5],
            &ScalarOptimConfig {
                algorithm: "autoeq:de".to_string(),
                ..Default::default()
            },
            |x| x[0],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dimension mismatch"));
    }

    #[test]
    fn validate_problem_rejects_reversed_bounds() {
        let result = optimize_bounded_scalar(
            &[(1.0, 0.0)],
            &[0.5],
            &ScalarOptimConfig {
                algorithm: "autoeq:de".to_string(),
                ..Default::default()
            },
            |x| x[0],
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("invalid scalar optimizer bounds")
        );
    }

    #[test]
    fn clamp_initial_clamps_out_of_bounds() {
        let result = optimize_bounded_scalar(
            &[(0.0, 1.0), (-1.0, 0.0)],
            &[-5.0, 5.0],
            &ScalarOptimConfig {
                algorithm: "autoeq:cobyla".to_string(),
                max_iter: 100,
                ..Default::default()
            },
            |x| (x[0] - 0.5).powi(2) + (x[1] + 0.5).powi(2),
        )
        .expect("optimizer should run");
        assert!(result.fun < 1e-2);
    }

    #[test]
    fn unseeded_scalar_search_is_reproducible() {
        let config = ScalarOptimConfig {
            algorithm: "autoeq:de".to_string(),
            max_iter: 200,
            population: 8,
            seed: None,
            ..Default::default()
        };
        let run = || {
            optimize_bounded_scalar(&[(-2.0, 2.0), (-2.0, 2.0)], &[1.5, 1.5], &config, quadratic)
                .expect("unseeded search should run")
        };
        let first = run();
        let second = run();
        assert_eq!(first.x, second.x);
        assert_eq!(first.fun, second.fun);
    }
}
