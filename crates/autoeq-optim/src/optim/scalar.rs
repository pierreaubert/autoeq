//! Shared bounded scalar-objective optimizer dispatch.
//!
//! RoomEQ has a few optimisation problems that are not PEQ filter fitting:
//! GD-Opt v2, multi-sub all-pass, and other small bounded black-box searches.
//! They should still honor the user-selected optimizer algorithm without each
//! caller wiring algorithm-specific code.

use crate::optim::registry;
use math_audio_optimisation::cobyla::{CobylaRhoBegin, cobyla};
use math_audio_optimisation::{
    CmaEsConfig, CobylaConfig, CobylaStopTols, DEConfigBuilder, Init, IsresConfig, Mutation,
    Strategy, cma_es, differential_evolution, isres,
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
    /// Optional deterministic seed.
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

/// Native generation-boundary cancellation for DE and CMA-ES.
/// A stopped run returns an error, never a deliverable best-effort candidate.
/// Backends without native callbacks reject callback requests before scoring.
/// This does not interrupt an in-flight objective evaluation or initial population.
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
    validate_problem(bounds, initial)?;

    let backend = registry::resolve(&config.algorithm)
        .ok_or_else(|| format!("Unknown algorithm: {}", config.algorithm))?;
    let canonical = backend.name().to_string();

    if callback.is_some() && !matches!(canonical.as_str(), "autoeq:cmaes" | "autoeq:de") {
        return Err(format!(
            "native scalar cancellation is not supported by {canonical}"
        ));
    }
    let stopped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let callback = callback.map(|mut callback| {
        let stopped = stopped.clone();
        Box::new(move |iteration, loss, preference| {
            let action = callback(iteration, loss, preference);
            if matches!(action, crate::de::CallbackAction::Stop) {
                stopped.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            action
        }) as super::OptimProgressCallback
    });

    let f = |x: &Array1<f64>| objective(x.as_slice().unwrap());
    let x0 = clamp_initial(initial, bounds);

    let result = match canonical.as_str() {
        "autoeq:cmaes" => optimize_cmaes(&canonical, bounds, x0, config, &f, callback),
        "autoeq:de" => optimize_de(&canonical, bounds, x0, config, &f, callback),
        "autoeq:cobyla" => optimize_cobyla(&canonical, bounds, x0, config, &f),
        "autoeq:isres" => optimize_isres(&canonical, bounds, x0, config, &f),
        other => Err(format!(
            "Algorithm '{}' is registered for PEQ filter optimization but is not supported for bounded scalar RoomEQ objectives",
            other
        )),
    };
    if stopped.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(format!("{canonical} scalar optimization stopped by user"));
    }
    result
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
    let report = cma_es(
        f,
        CmaEsConfig {
            bounds: bounds.to_vec(),
            x0: Some(x0),
            sigma0: Some(0.20),
            lambda,
            mu: 0,
            maxeval: config.max_iter.max(lambda + 1),
            seed: config.seed,
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
    if let Some(seed) = config.seed {
        builder = builder.seed(seed);
    }
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
        x: report.x.to_vec(),
        fun: report.fun,
        algorithm: canonical.to_string(),
        success: report.success,
        message: report.message,
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
            seed: config.seed,
            f_tol: config.atolerance.max(1e-12),
            ..Default::default()
        },
    )
    .map_err(|e| format!("ISRES failed: {e:?}"))?;

    Ok(ScalarOptimResult {
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
    fn native_scalar_stop_is_not_a_successful_candidate() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        for algorithm in ["autoeq:de", "cma-es"] {
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
        for algorithm in ["autoeq:de", "autoeq:cmaes"] {
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
}
