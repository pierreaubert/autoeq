use super::registry;

#[cfg(test)]
mod outcome_evidence_tests {
    use super::super::{OptimizerConfidence, OptimizerRunEvidence, OptimizerTermination};

    #[test]
    fn ok_status_marked_not_converged_is_best_effort_not_success() {
        let evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:bo",
            Ok((
                "AutoEQ BO: maximum evaluations reached (not converged, nfev=42)".to_string(),
                1.25,
            )),
            &[0.5],
            &[0.0],
            &[1.0],
            100,
            Some(7),
        );

        assert!(!evidence.converged);
        assert!(evidence.best_effort);
        assert_eq!(evidence.termination, OptimizerTermination::EvaluationLimit);
        assert_eq!(evidence.evaluation_count, Some(42));
        assert_eq!(evidence.evaluation_limit, 100);
        assert_eq!(evidence.seed, Some(7));
        assert_eq!(evidence.confidence, OptimizerConfidence::Low);
    }

    #[test]
    fn finite_error_result_preserves_best_vector_but_reports_backend_failure() {
        let evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Err(("line search failed".to_string(), 2.0)),
            &[0.5],
            &[0.0],
            &[1.0],
            50,
            None,
        );

        assert!(!evidence.converged);
        assert!(evidence.best_effort);
        assert_eq!(evidence.termination, OptimizerTermination::BackendFailure);
        assert_eq!(evidence.confidence, OptimizerConfidence::Low);
    }

    #[test]
    fn bound_constraint_violation_makes_outcome_unusable() {
        let evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:cobyla",
            Ok(("converged".to_string(), 1.0)),
            &[1.5],
            &[0.0],
            &[1.0],
            50,
            Some(11),
        );

        assert_eq!(evidence.max_constraint_violation, 0.5);
        assert_eq!(evidence.confidence, OptimizerConfidence::Unusable);
        assert!(!evidence.converged);
    }

    #[test]
    fn clean_success_is_high_confidence_and_records_empty_restart_history() {
        let evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Ok(("relative tolerance reached".to_string(), 0.5)),
            &[0.5],
            &[0.0],
            &[1.0],
            200,
            Some(3),
        );

        assert!(evidence.converged);
        assert!(!evidence.best_effort);
        assert_eq!(evidence.termination, OptimizerTermination::Converged);
        assert_eq!(evidence.confidence, OptimizerConfidence::High);
        assert!(evidence.restart_history.is_empty());
    }
}

#[cfg(test)]
mod dispatch_tests {

    /// Bug C reproducer: `optimize_filters_with_callback` previously
    /// dispatched on `backend.library() == "AutoEQ"`, which now matches
    /// all pure-Rust AutoEQ backends — silently
    /// routing non-DE backends through the DE EPA wrapper instead of
    /// running the requested algorithm. Verify each `autoeq:*` backend
    /// resolves to its OWN registry entry, not DE.
    #[test]
    fn autoeq_cobyla_and_isres_have_own_names() {
        let cobyla = super::registry::resolve("autoeq:cobyla").expect("autoeq:cobyla missing");
        assert_eq!(cobyla.name(), "autoeq:cobyla");
        assert_eq!(cobyla.library(), "AutoEQ");

        let isres = super::registry::resolve("autoeq:isres").expect("autoeq:isres missing");
        assert_eq!(isres.name(), "autoeq:isres");
        assert_eq!(isres.library(), "AutoEQ");

        let cmaes = super::registry::resolve("autoeq:cmaes").expect("autoeq:cmaes missing");
        assert_eq!(cmaes.name(), "autoeq:cmaes");
        assert_eq!(cmaes.library(), "AutoEQ");
        let cmaes_alias = super::registry::resolve("cma-es").expect("cma-es alias missing");
        assert_eq!(cmaes_alias.name(), "autoeq:cmaes");

        let nsga2 = super::registry::resolve("autoeq:nsga2").expect("autoeq:nsga2 missing");
        assert_eq!(nsga2.name(), "autoeq:nsga2");
        assert_eq!(nsga2.library(), "AutoEQ");
        let nsga2_alias = super::registry::resolve("nsga-ii").expect("nsga-ii alias missing");
        assert_eq!(nsga2_alias.name(), "autoeq:nsga2");

        let nsga3 = super::registry::resolve("autoeq:nsga3").expect("autoeq:nsga3 missing");
        assert_eq!(nsga3.name(), "autoeq:nsga3");
        assert_eq!(nsga3.library(), "AutoEQ");
        let nsga3_alias = super::registry::resolve("nsga-iii").expect("nsga-iii alias missing");
        assert_eq!(nsga3_alias.name(), "autoeq:nsga3");

        let de = super::registry::resolve("autoeq:de").expect("autoeq:de missing");
        assert_eq!(de.name(), "autoeq:de");
        assert_eq!(de.library(), "AutoEQ");

        // The dispatcher must distinguish them by NAME, not library — the
        // EPA wrapper is DE-specific.
        assert_ne!(cobyla.name(), de.name());
        assert_ne!(isres.name(), de.name());
        assert_ne!(cmaes.name(), de.name());
        assert_ne!(nsga2.name(), de.name());
        assert_ne!(nsga3.name(), de.name());
    }
}

#[cfg(test)]
mod smoothness_penalty_tests {

    use super::super::{SmoothnessPenaltyConfig, compute_smoothness_penalty};
    use ndarray::Array1;

    fn log_grid(n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|i| 20.0 * 10f64.powf(i as f64 * 3.0 / (n as f64 - 1.0))))
    }

    #[test]
    fn smoothness_penalty_zero_for_flat_curve() {
        let freqs = log_grid(200);
        let y = Array1::zeros(200);
        let cfg = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            ..Default::default()
        };
        let p = compute_smoothness_penalty(&y, &freqs, 20.0, 20_000.0, &cfg);
        assert!(p < 1e-12, "flat curve must have zero curvature, got {p}");
    }

    #[test]
    fn smoothness_penalty_zero_for_linear_log_tilt() {
        let freqs = log_grid(200);
        let y = freqs.mapv(|f| -0.8 * f.log10());
        let cfg = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            ..Default::default()
        };
        let p = compute_smoothness_penalty(&y, &freqs, 20.0, 20_000.0, &cfg);
        assert!(
            p < 1e-9,
            "linear log-freq tilt must have ~zero second derivative, got {p}"
        );
    }

    #[test]
    fn smoothness_penalty_punishes_oscillation() {
        let freqs = log_grid(200);
        let y_osc = freqs.mapv(|f| 3.0 * (f.log10() * 20.0).sin());
        let y_flat = Array1::zeros(200);
        let cfg = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            ..Default::default()
        };
        let p_osc = compute_smoothness_penalty(&y_osc, &freqs, 20.0, 20_000.0, &cfg);
        let p_flat = compute_smoothness_penalty(&y_flat, &freqs, 20.0, 20_000.0, &cfg);
        assert!(p_osc > 1000.0 * (p_flat + 1e-12));
    }

    #[test]
    fn smoothness_penalty_modal_region_relaxed() {
        let freqs = log_grid(200);
        let y = freqs.mapv(|f| {
            let s50 = (-((f - 50.0).powi(2) / 25.0)).exp();
            let s5k = (-((f - 5000.0).powi(2) / 250_000.0)).exp();
            -6.0 * (s50 + s5k)
        });
        let cfg_relaxed = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            schroeder_hz: Some(300.0),
            modal_weight_scale: 0.0,
            exponent: 1.0,
        };
        let cfg_strict = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            schroeder_hz: None,
            modal_weight_scale: 1.0,
            exponent: 1.0,
        };
        let p_relaxed = compute_smoothness_penalty(&y, &freqs, 20.0, 20_000.0, &cfg_relaxed);
        let p_strict = compute_smoothness_penalty(&y, &freqs, 20.0, 20_000.0, &cfg_strict);
        assert!(
            p_relaxed < 0.6 * p_strict,
            "modal exemption must reduce penalty: relaxed={p_relaxed}, strict={p_strict}"
        );
    }

    #[test]
    fn smoothness_penalty_disabled_returns_zero() {
        let freqs = log_grid(200);
        let y = freqs.mapv(|f| 5.0 * (f.log10() * 30.0).sin());
        let cfg = SmoothnessPenaltyConfig {
            tv2_weight: 0.0,
            ..Default::default()
        };
        assert_eq!(
            compute_smoothness_penalty(&y, &freqs, 20.0, 20_000.0, &cfg),
            0.0
        );
    }
}

#[cfg(test)]
mod backend_tests {
    use ndarray::Array1;

    use super::super::ObjectiveData;
    use super::super::backend::FilterOptimizer;
    use super::super::bo::AutoeqBoBackend;
    use super::super::isres::AutoeqIsresBackend;
    use super::super::mh::MhBackend;
    use super::super::nsga::AutoeqNsgaBackend;
    use super::super::params::OptimParams;
    use super::super::pareto::{
        ParetoFilter, extract_non_dominated, pareto_optimization, print_pareto_front,
    };
    use super::super::setup::{
        ProgressCallbackConfig, initial_guess, perform_optimization,
        perform_optimization_with_callback, perform_optimization_with_progress, setup_bounds,
        setup_objective_data,
    };
    use super::super::types::MultiObjectiveData;
    use crate::Curve;
    use crate::cli::Args;
    use clap::Parser;

    fn small_args() -> Args {
        let mut args = Args::parse_from(["autoeq"]);
        args.num_filters = 1;
        args.population = 6;
        args.maxeval = 60;
        args.seed = Some(1);
        args.min_freq = 20.0;
        args.max_freq = 20000.0;
        args.min_db = -12.0;
        args.max_db = 12.0;
        args
    }

    fn scalar_objective() -> (ObjectiveData, OptimParams, Vec<f64>, Vec<f64>, Vec<f64>) {
        let freqs = Array1::from(vec![
            20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 10240.0,
        ]);
        let input_curve = Curve {
            freq: freqs.clone(),
            spl: Array1::from_elem(freqs.len(), 5.0),
            phase: None,
            ..Default::default()
        };
        let target_curve = Curve {
            freq: freqs.clone(),
            spl: Array1::zeros(freqs.len()),
            phase: None,
            ..Default::default()
        };
        let deviation_curve = Curve {
            freq: freqs.clone(),
            spl: Array1::from_elem(freqs.len(), 5.0),
            phase: None,
            ..Default::default()
        };
        let args = small_args();
        let params = OptimParams::from(&args);
        let (obj, _use_cea) = setup_objective_data(
            &params,
            &input_curve,
            &target_curve,
            &deviation_curve,
            &None,
        )
        .unwrap();
        let (lower, upper) = setup_bounds(&params);
        let x = initial_guess(&params, &lower, &upper);
        (obj, params, lower, upper, x)
    }

    fn multi_objective() -> (ObjectiveData, OptimParams, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut obj, params, lower, upper, x) = scalar_objective();
        let obj2 = obj.clone();
        obj.multi_objective = Some(MultiObjectiveData {
            objectives: vec![obj.clone(), obj2],
            strategy: crate::roomeq::MultiMeasurementStrategy::WeightedSum,
            weights: vec![0.5, 0.5],
            variance_lambda: 0.0,
            uncertainty_cvar_alpha: None,
        });
        (obj, params, lower, upper, x)
    }

    #[test]
    fn isres_backend_optimizes_scalar() {
        let (obj, params, lower, upper, mut x) = scalar_objective();
        let backend = AutoeqIsresBackend::new("autoeq:isres");
        let result = backend.optimize(&mut x, &lower, &upper, obj, &params, None);
        assert!(result.is_ok(), "ISRES should converge: {:?}", result);
        let (status, loss) = result.unwrap();
        assert!(loss.is_finite(), "loss must be finite, got {}", loss);
        assert!(
            status.contains("ISRES"),
            "status should mention ISRES: {}",
            status
        );
    }

    #[test]
    fn mh_backends_optimizes_scalar() {
        let configs = vec![
            MhBackend::new_de("mh:de"),
            MhBackend::new_pso("mh:pso"),
            MhBackend::new_rga("mh:rga"),
            MhBackend::new_tlbo("mh:tlbo"),
            MhBackend::new_firefly("mh:firefly"),
        ];
        for backend in configs {
            let (obj, params, lower, upper, mut x) = scalar_objective();
            let result = backend.optimize(&mut x, &lower, &upper, obj, &params, None);
            assert!(
                result.is_ok(),
                "{} should converge: {:?}",
                backend.name(),
                result
            );
            let (status, loss) = result.unwrap();
            assert!(loss.is_finite(), "{} loss must be finite", backend.name());
            assert!(
                status.contains("Metaheuristics"),
                "{} status should mention Metaheuristics: {}",
                backend.name(),
                status
            );
        }
    }

    #[test]
    fn bo_backend_optimizes_scalar() {
        let (obj, mut params, lower, upper, mut x) = scalar_objective();
        params.bo_ehvi = false;
        params.maxeval = 20;
        params.bo_initial_samples = 5;
        let backend = AutoeqBoBackend::new("autoeq:bo");
        let result = backend.optimize(&mut x, &lower, &upper, obj, &params, None);
        assert!(result.is_ok(), "BO scalar should run: {:?}", result);
        let (status, loss) = result.unwrap();
        assert!(loss.is_finite(), "BO loss must be finite");
        assert!(
            status.contains("BO"),
            "status should mention BO: {}",
            status
        );
    }

    #[test]
    fn bo_backend_optimizes_multi_objective() {
        let (obj, mut params, lower, upper, mut x) = multi_objective();
        params.bo_ehvi = true;
        params.maxeval = 20;
        params.bo_initial_samples = 5;
        let backend = AutoeqBoBackend::new("autoeq:bo");
        let result = backend.optimize(&mut x, &lower, &upper, obj, &params, None);
        assert!(result.is_ok(), "BO-EHVI should run: {:?}", result);
        let (status, loss) = result.unwrap();
        assert!(loss.is_finite(), "BO-EHVI loss must be finite");
        assert!(
            status.contains("EHVI"),
            "status should mention EHVI: {}",
            status
        );
    }

    #[test]
    fn bo_backend_refine_path_runs() {
        let (obj, mut params, lower, upper, mut x) = scalar_objective();
        params.refine = true;
        params.local_algo = "autoeq:cobyla".to_string();
        params.maxeval = 20;
        params.bo_initial_samples = 5;
        let backend = AutoeqBoBackend::new("autoeq:bo");
        let result = backend.optimize(&mut x, &lower, &upper, obj, &params, None);
        assert!(result.is_ok(), "BO refine should run: {:?}", result);
        let (status, loss) = result.unwrap();
        assert!(loss.is_finite(), "BO refine loss must be finite");
        assert!(
            status.contains("refine") || status.contains("BO"),
            "status: {}",
            status
        );
    }

    #[test]
    fn nsga_backends_optimizes_scalar() {
        for backend in [
            AutoeqNsgaBackend::new_nsga2("autoeq:nsga2"),
            AutoeqNsgaBackend::new_nsga3("autoeq:nsga3"),
        ] {
            let (obj, params, lower, upper, mut x) = scalar_objective();
            let result = backend.optimize(&mut x, &lower, &upper, obj, &params, None);
            assert!(
                result.is_ok(),
                "{} should converge: {:?}",
                backend.name(),
                result
            );
            let (status, loss) = result.unwrap();
            assert!(loss.is_finite(), "{} loss must be finite", backend.name());
            assert!(
                status.contains("NSGA"),
                "{} status should mention NSGA: {}",
                backend.name(),
                status
            );
        }
    }

    #[test]
    fn nsga_backends_optimizes_multi_objective() {
        for backend in [
            AutoeqNsgaBackend::new_nsga2("autoeq:nsga2"),
            AutoeqNsgaBackend::new_nsga3("autoeq:nsga3"),
        ] {
            let (obj, mut params, lower, upper, mut x) = multi_objective();
            params.population = 16;
            params.maxeval = 64;
            let result = backend.optimize(&mut x, &lower, &upper, obj, &params, None);
            assert!(
                result.is_ok(),
                "{} multi-objective should run: {:?}",
                backend.name(),
                result
            );
            let (_status, loss) = result.unwrap();
            assert!(
                loss.is_finite(),
                "{} multi loss must be finite",
                backend.name()
            );
        }
    }

    #[test]
    fn pareto_helpers_and_integration() {
        let filters = vec![
            ParetoFilter {
                params: vec![1.0, 2.0, 3.0],
                flatness_loss: 10.0,
                score_loss: None,
                num_filters: 1,
                converged: true,
            },
            ParetoFilter {
                params: vec![1.0, 2.0, 3.0],
                flatness_loss: 5.0,
                score_loss: None,
                num_filters: 2,
                converged: true,
            },
            ParetoFilter {
                params: vec![1.0, 2.0, 3.0],
                flatness_loss: 20.0,
                score_loss: None,
                num_filters: 3,
                converged: false,
            },
        ];
        let non_dominated = extract_non_dominated(&filters);
        assert!(
            !non_dominated.is_empty(),
            "non-dominated set should not be empty"
        );
        // Just exercise the printer; it logs, should not panic.
        print_pareto_front(&filters);

        let (obj, _params, _lower, _upper, _x) = scalar_objective();
        let mut args = small_args();
        args.num_filters = 1;
        args.population = 6;
        args.maxeval = 60;
        args.algo = "autoeq:de".to_string();
        let front = pareto_optimization(&obj, &crate::OptimParams::from(&args), vec![1, 2]);
        assert_eq!(
            front.len(),
            2,
            "pareto_optimization should return one entry per filter count"
        );
    }

    #[test]
    fn perform_optimization_non_de_backend() {
        let (obj, _params, _lower, _upper, _x) = scalar_objective();
        let mut args = small_args();
        args.algo = "autoeq:cobyla".to_string();
        args.maxeval = 200;
        let result = perform_optimization(&crate::OptimParams::from(&args), &obj);
        assert!(
            result.is_ok(),
            "perform_optimization cobyla should run: {:?}",
            result
        );
        assert!(
            !result.unwrap().is_empty(),
            "should return parameter vector"
        );
    }

    #[test]
    fn perform_optimization_with_callback_de() {
        let (obj, _params, _lower, _upper, _x) = scalar_objective();
        let mut args = small_args();
        args.algo = "autoeq:de".to_string();
        args.maxeval = 60;
        let mut iterations = Vec::new();
        let result = perform_optimization_with_callback(
            &crate::OptimParams::from(&args),
            &obj,
            Box::new(move |im: &crate::de::DEIntermediate| {
                iterations.push(im.iter);
                crate::de::CallbackAction::Continue
            }),
        );
        assert!(result.is_ok(), "DE callback path should run: {:?}", result);
    }

    #[test]
    fn perform_optimization_with_progress_runs() {
        let (obj, _params, _lower, _upper, _x) = scalar_objective();
        let mut args = small_args();
        args.algo = "autoeq:de".to_string();
        args.maxeval = 60;
        let config = ProgressCallbackConfig {
            interval: 10,
            include_biquads: true,
            include_filter_response: true,
            frequencies: vec![100.0, 1000.0],
        };
        let result = perform_optimization_with_progress(
            &crate::OptimParams::from(&args),
            &obj,
            config,
            |_update| crate::de::CallbackAction::Continue,
        );
        assert!(result.is_ok(), "progress path should run: {:?}", result);
    }

    #[test]
    fn perform_optimization_refine_runs() {
        let (obj, _params, _lower, _upper, _x) = scalar_objective();
        let mut args = small_args();
        args.algo = "autoeq:de".to_string();
        args.refine = true;
        args.local_algo = "autoeq:cobyla".to_string();
        args.maxeval = 60;
        let result = perform_optimization(&crate::OptimParams::from(&args), &obj);
        assert!(
            result.is_ok(),
            "DE + cobyla refine should run: {:?}",
            result
        );
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn perform_optimization_with_progress_minimal_config_runs() {
        let (obj, _params, _lower, _upper, _x) = scalar_objective();
        let mut args = small_args();
        args.algo = "autoeq:de".to_string();
        args.maxeval = 60;
        let config = ProgressCallbackConfig {
            interval: 5,
            include_biquads: false,
            include_filter_response: false,
            frequencies: vec![],
        };
        let result = perform_optimization_with_progress(
            &crate::OptimParams::from(&args),
            &obj,
            config,
            |_update| crate::de::CallbackAction::Continue,
        );
        assert!(
            result.is_ok(),
            "minimal progress config should run: {:?}",
            result
        );
    }

    #[test]
    fn perform_optimization_with_callback_non_de_runs() {
        let (obj, _params, _lower, _upper, _x) = scalar_objective();
        let mut args = small_args();
        args.algo = "autoeq:cmaes".to_string();
        args.maxeval = 200;
        let result = perform_optimization_with_callback(
            &crate::OptimParams::from(&args),
            &obj,
            Box::new(|_intermediate| crate::de::CallbackAction::Continue),
        );
        assert!(
            result.is_ok(),
            "CMAES callback path should run: {:?}",
            result
        );
    }

    #[test]
    fn compute_fitness_penalties_wrapper_matches_ref() {
        let (mut obj, _params, _lower, _upper, x) = scalar_objective();
        let ref_val = super::super::compute_fitness_penalties_ref(&x, &obj);
        let wrapped_val = super::super::compute_fitness_penalties(&x, None, &mut obj);
        assert_eq!(ref_val, wrapped_val);
    }
}
