use super::registry;

#[cfg(test)]
mod outcome_evidence_tests {
    use super::super::run_control::{EvaluationStage, OptimizerRunControl};
    use super::super::{
        OptimizerBackendCompletion, OptimizerConfidence, OptimizerRunEvidence, OptimizerTermination,
    };
    use std::num::NonZeroUsize;

    #[test]
    fn legacy_stopped_status_is_not_proof_of_user_cancellation() {
        let evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Ok(("optimization stopped by callback (nfev=3)".into(), 0.25)),
            &[0.5],
            &[0.0],
            &[1.0],
            40,
            Some(42),
        );
        assert_eq!(evidence.termination, OptimizerTermination::NonConverged);
        assert!(!evidence.converged);
        assert!(evidence.best_effort);
        assert_eq!(evidence.confidence, OptimizerConfidence::Low);
        assert_eq!(evidence.objective, Some(0.25));
        assert_eq!(evidence.evaluation_count, Some(3));
        assert_eq!(evidence.status, "optimization stopped by callback (nfev=3)");
    }

    #[test]
    fn native_budget_status_is_not_convergence_without_extra_qualifier() {
        for status in [
            "AutoEQ COBYLA: MaxevalReached",
            "maximum evaluations reached",
            "maximum iterations reached",
            "evaluation budget exhausted",
        ] {
            let evidence = OptimizerRunEvidence::from_backend_result(
                "autoeq:cobyla",
                Ok((status.into(), 0.25)),
                &[0.5],
                &[0.0],
                &[1.0],
                40,
                Some(42),
            );
            assert_eq!(
                evidence.termination,
                OptimizerTermination::EvaluationLimit,
                "{status}"
            );
            assert!(!evidence.converged, "{status}");
            assert!(
                evidence.best_effort,
                "finite bounded candidate should remain usable: {status}"
            );
            assert_eq!(evidence.confidence, OptimizerConfidence::Low);
        }
    }

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
    fn typed_convergence_is_high_confidence_and_records_empty_restart_history() {
        let mut evidence = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Ok(("relative tolerance reached".to_string(), 0.5)),
            &[0.5],
            &[0.0],
            &[1.0],
            200,
            Some(3),
        );
        assert_eq!(evidence.termination, OptimizerTermination::NonConverged);
        evidence.apply_backend_completion(OptimizerBackendCompletion::Converged);

        assert!(evidence.converged);
        assert!(!evidence.best_effort);
        assert_eq!(evidence.termination, OptimizerTermination::Converged);
        assert_eq!(evidence.confidence, OptimizerConfidence::High);
        assert!(evidence.restart_history.is_empty());
    }

    #[test]
    fn fixed_generation_success_without_typed_completion_is_best_effort() {
        let evidence = OptimizerRunEvidence::from_backend_result(
            "mh:firefly",
            Ok(("Metaheuristics(Firefly)".into(), 0.5)),
            &[0.5],
            &[0.0],
            &[1.0],
            128,
            Some(7),
        );
        assert_eq!(evidence.termination, OptimizerTermination::NonConverged);
        assert!(!evidence.converged);
        assert!(evidence.best_effort);
        assert_eq!(evidence.status, "Metaheuristics(Firefly)");
    }

    #[test]
    fn run_control_maps_budget_user_cancel_and_deadline_without_losing_finite_result() {
        let make_evidence = || {
            OptimizerRunEvidence::from_backend_result(
                "autoeq:de",
                Ok(("legacy callback status".into(), 0.5)),
                &[0.5],
                &[0.0],
                &[1.0],
                1,
                Some(11),
            )
        };

        let budget = OptimizerRunControl::new(NonZeroUsize::new(1).unwrap());
        drop(budget.begin_evaluation(EvaluationStage::Search, 1).unwrap());
        let mut budget_evidence = make_evidence();
        budget_evidence.apply_backend_completion(OptimizerBackendCompletion::Converged);
        budget_evidence.apply_run_control(&budget.snapshot());
        budget_evidence.apply_backend_completion(OptimizerBackendCompletion::Converged);
        assert_eq!(
            budget_evidence.termination,
            OptimizerTermination::EvaluationLimit
        );
        assert_eq!(budget_evidence.objective, Some(0.5));
        assert!(budget_evidence.best_effort);

        let user_stop = OptimizerRunControl::new(NonZeroUsize::new(4).unwrap());
        user_stop.request_cancel();
        let mut user_evidence = make_evidence();
        user_evidence.apply_backend_completion(OptimizerBackendCompletion::Converged);
        user_evidence.apply_run_control(&user_stop.snapshot());
        user_evidence.apply_backend_completion(OptimizerBackendCompletion::Converged);
        assert_eq!(user_evidence.termination, OptimizerTermination::UserStopped);
        assert_eq!(user_evidence.objective, Some(0.5));
        assert!(!user_evidence.best_effort);
        assert!(user_evidence.has_valid_candidate());

        let deadline = OptimizerRunControl::new(NonZeroUsize::new(4).unwrap());
        deadline.request_deadline();
        let mut timed_evidence = make_evidence();
        timed_evidence.apply_backend_completion(OptimizerBackendCompletion::Converged);
        timed_evidence.apply_run_control(&deadline.snapshot());
        timed_evidence.apply_backend_completion(OptimizerBackendCompletion::Converged);
        assert_eq!(timed_evidence.termination, OptimizerTermination::TimedOut);
        assert_eq!(timed_evidence.objective, Some(0.5));
        assert!(timed_evidence.best_effort);
        assert!(timed_evidence.has_valid_candidate());
    }

    #[test]
    fn refusal_without_callback_stop_reports_budget_limit() {
        let control = OptimizerRunControl::new(NonZeroUsize::new(4).unwrap());
        let mut snapshot = control.snapshot();
        snapshot.evaluations_refused = 1;
        let mut evidence = OptimizerRunEvidence::from_backend_result(
            "nsga-ii",
            Ok(("ordinary backend status".into(), 0.25)),
            &[0.5],
            &[0.0],
            &[1.0],
            4,
            Some(9),
        );
        evidence.apply_run_control(&snapshot);
        assert_eq!(evidence.termination, OptimizerTermination::EvaluationLimit);
        assert!(evidence.best_effort);
    }

    #[test]
    fn backend_failure_and_invalid_result_outweigh_stop_requests() {
        let control = OptimizerRunControl::new(NonZeroUsize::new(1).unwrap());
        drop(
            control
                .begin_evaluation(EvaluationStage::Search, 1)
                .unwrap(),
        );
        let _ = control
            .begin_evaluation(EvaluationStage::Search, 1)
            .is_none();
        control.request_cancel();
        control.request_deadline();
        let snapshot = control.snapshot();

        let mut failed = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Err((
                "solver failed independently of the stop request".into(),
                f64::INFINITY,
            )),
            &[0.5],
            &[0.0],
            &[1.0],
            1,
            Some(1),
        );
        failed.apply_run_control(&snapshot);
        assert_eq!(failed.termination, OptimizerTermination::BackendFailure);
        assert_eq!(
            failed.status,
            "solver failed independently of the stop request"
        );

        let mut invalid = OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Ok(("solver returned a candidate".into(), f64::NAN)),
            &[0.5],
            &[0.0],
            &[1.0],
            1,
            Some(1),
        );
        invalid.apply_run_control(&snapshot);
        assert_eq!(invalid.termination, OptimizerTermination::InvalidResult);
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
        perform_optimization_with_callback, perform_optimization_with_progress,
        q_max_for_frequency_range, setup_bounds, setup_objective_data,
    };
    use super::super::types::MultiObjectiveData;
    use crate::Curve;
    use crate::FrequencyQPolicy;
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

    #[test]
    fn frequency_q_policy_keeps_modal_freedom_but_caps_guarded_ranges() {
        let mut params = OptimParams::from(&small_args());
        params.num_filters = 3;
        params.min_freq = 20.0;
        params.max_freq = 20_000.0;
        params.min_q = 0.5;
        params.max_q = 12.0;
        params.frequency_q_policy = Some(FrequencyQPolicy {
            schroeder_hz: Some(500.0),
            low_max_q: Some(12.0),
            high_start_hz: Some(1_600.0),
            high_max_q: Some(0.8),
        });

        assert_eq!(q_max_for_frequency_range(&params, 80.0, 300.0), 12.0);
        let crossing = q_max_for_frequency_range(&params, 300.0, 800.0);
        assert!(crossing > 0.8 && crossing < 12.0);
        assert_eq!(q_max_for_frequency_range(&params, 2_000.0, 8_000.0), 0.8);

        let (_, upper) = setup_bounds(&params);
        for filter in 0..params.num_filters {
            let offset = filter * 3;
            let f_high = 10.0_f64.powf(upper[offset]);
            if f_high >= 1_600.0 {
                assert!(upper[offset + 1] <= 0.8);
            }
        }
    }

    #[test]
    fn frequency_q_policy_is_continuous_at_schroeder_boundary() {
        let mut params = OptimParams::from(&small_args());
        params.min_freq = 20.0;
        params.max_freq = 20_000.0;
        params.min_q = 0.5;
        params.max_q = 12.0;
        params.frequency_q_policy = Some(FrequencyQPolicy {
            schroeder_hz: Some(500.0),
            low_max_q: Some(12.0),
            high_start_hz: Some(1_600.0),
            high_max_q: Some(0.8),
        });
        let below = q_max_for_frequency_range(&params, 499.0, 499.0);
        let above = q_max_for_frequency_range(&params, 501.0, 501.0);
        assert_eq!(below, 12.0);
        assert!(above < below);
        assert!(above > 0.8);
        assert!(
            (below - above) < 1.0,
            "Schroeder boundary Q jump is too large"
        );
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

    #[test]
    fn cobra_dispatch_is_seeded_bounded_and_counted() {
        use super::super::optimize::optimize_filters_with_run_control_detailed;
        use super::super::run_control::OptimizerRunControl;
        use std::num::NonZeroUsize;

        let run_once = || {
            let (objective, mut params, lower, upper, mut x) = scalar_objective();
            params.algo = "autoeq:cobra".into();
            params.maxeval = 20;
            params.seed = Some(42);
            let control = OptimizerRunControl::new(NonZeroUsize::new(20).unwrap());
            let result = optimize_filters_with_run_control_detailed(
                &mut x, &lower, &upper, objective, &params, &control,
            );
            let (_, loss) = result
                .result
                .as_ref()
                .expect("finite feasible COBRA winner");
            assert!(loss.is_finite());
            assert!(
                x.iter()
                    .zip(&lower)
                    .zip(&upper)
                    .all(|((&v, &lo), &hi)| v >= lo && v <= hi)
            );
            assert_eq!(result.snapshot.evaluations_started, 20);
            assert_eq!(result.snapshot.evaluations_completed, 20);
            assert_eq!(result.snapshot.evaluations_in_flight, 0);
            assert_eq!(result.snapshot.evaluations_refused, 0);
            assert!(!result.evidence.converged);
            (x, loss.to_bits())
        };
        assert_eq!(run_once(), run_once());
    }

    #[test]
    fn cobra_callback_stop_returns_after_first_infill_without_polish() {
        use super::super::cobra::AutoeqCobraBackend;
        let (objective, mut params, lower, upper, mut x) = scalar_objective();
        params.maxeval = 100;
        params.seed = Some(42);
        let initial = (3 * x.len() + 1).min(params.maxeval);
        let (status, loss) = AutoeqCobraBackend::new("autoeq:cobra")
            .optimize(
                &mut x,
                &lower,
                &upper,
                objective,
                &params,
                Some(Box::new(|iteration, loss, epa| {
                    assert_eq!(iteration, 1);
                    assert!(loss.is_finite());
                    assert!(epa.is_none());
                    crate::de::CallbackAction::Stop
                })),
            )
            .expect("callback stop retains a feasible candidate");
        assert!(status.contains("stopped by callback"), "{status}");
        assert!(
            status.contains(&format!("nfev={}", initial + 1)),
            "{status}"
        );
        assert!(loss.is_finite());
    }

    #[test]
    fn detailed_controlled_dispatch_identifies_budget_refusal_before_backend_start() {
        use super::super::optimize::{
            OptimizerDispatchOutcome, OptimizerTermination,
            optimize_filters_with_run_control_detailed,
        };
        use super::super::run_control::OptimizerRunControl;
        use std::num::NonZeroUsize;

        let (objective, mut params, lower, upper, mut x) = scalar_objective();
        params.algo = "autoeq:de".to_string();
        params.population = 6;
        params.maxeval = 1;
        let control = OptimizerRunControl::new(NonZeroUsize::new(1).unwrap());

        let run = optimize_filters_with_run_control_detailed(
            &mut x, &lower, &upper, objective, &params, &control,
        );

        let OptimizerDispatchOutcome::NotStartedBudgetRefusal(refusal) = run.dispatch else {
            panic!("expected typed preflight refusal, got {:?}", run.dispatch);
        };
        assert_eq!(refusal.requested_evaluations, 1);
        assert!(refusal.required_evaluations > refusal.requested_evaluations);
        assert_eq!(run.snapshot.evaluations_started, 0);
        assert_eq!(run.evidence.evaluation_count, Some(0));
        assert_eq!(
            run.evidence.termination,
            OptimizerTermination::EvaluationLimit
        );
        assert!(run.result.is_err());
    }

    #[test]
    fn detailed_controlled_dispatch_distinguishes_unresolved_backend() {
        use super::super::optimize::{
            OptimizerDispatchOutcome, OptimizerTermination,
            optimize_filters_with_run_control_detailed,
        };
        use super::super::run_control::OptimizerRunControl;
        use std::num::NonZeroUsize;

        let (objective, mut params, lower, upper, mut x) = scalar_objective();
        params.algo = "not-a-registered-backend".to_string();
        let control = OptimizerRunControl::new(NonZeroUsize::new(100).unwrap());

        let run = optimize_filters_with_run_control_detailed(
            &mut x, &lower, &upper, objective, &params, &control,
        );

        assert_eq!(
            run.dispatch,
            OptimizerDispatchOutcome::NotStartedDispatchFailure
        );
        assert_eq!(run.snapshot.evaluations_started, 0);
        assert_eq!(
            run.evidence.termination,
            OptimizerTermination::BackendFailure
        );
        assert!(run.result.is_err());
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
    fn registered_rga_reports_actual_search_limit_without_claiming_convergence() {
        use super::super::backend::BackendSearchStopCause;
        use super::super::optimize::{
            OptimizerDispatchOutcome, OptimizerTermination,
            optimize_filters_with_run_control_detailed,
        };
        use super::super::run_control::OptimizerRunControl;
        use std::num::NonZeroUsize;

        let (objective, mut params, lower, upper, mut x) = scalar_objective();
        params.algo = "mh:rga".into();
        params.maxeval = 60;
        params.seed = Some(7);
        let root_control = OptimizerRunControl::new(NonZeroUsize::new(120).unwrap());
        // Controlled dispatch uses the stage cap as the backend search cap.
        // Final selected-candidate validation is accounted separately.
        let control = root_control.with_stage_budget(NonZeroUsize::new(120).unwrap());
        let run = optimize_filters_with_run_control_detailed(
            &mut x, &lower, &upper, objective, &params, &control,
        );
        assert_eq!(run.dispatch, OptimizerDispatchOutcome::BackendInvoked);
        assert!(
            run.result.is_ok(),
            "RGA must return a finalized candidate: {:?}",
            run.result
        );
        assert_eq!(
            run.evidence.termination,
            OptimizerTermination::EvaluationLimit
        );
        assert_eq!(
            run.evidence.backend_stop_cause,
            Some(BackendSearchStopCause::ObjectiveBudgetLimit),
        );
        assert!(!run.evidence.converged);
        assert!(run.evidence.backend_evaluation_count.unwrap() >= params.population);
        assert!(run.evidence.generation_count.unwrap() > 0);
        assert!(run.evidence.task_callback_count.unwrap() < run.evidence.generation_limit.unwrap());
        assert_eq!(run.evidence.backend_evaluation_count, Some(120));
        assert!(run.evidence.backend_denied_evaluation_count.is_some());
        let stage = run.stage_snapshot.expect("RGA stage search counts");
        assert!(stage.evaluations_started <= 120);
        assert!(stage.evaluations_started >= run.evidence.backend_evaluation_count.unwrap());
        assert!(root_control.snapshot().evaluations_started <= 120);
        assert_eq!(
            run.evidence.task_callback_count,
            run.evidence
                .generation_count
                .map(|generations| generations + 1),
        );
        assert!(
            run.evidence.population_fitness_mean.is_some()
                == run.evidence.population_fitness_stddev.is_some()
        );
        assert!(
            run.evidence
                .population_fitness_stddev
                .is_none_or(f64::is_finite)
        );
    }

    #[test]
    fn registered_rga_stage_budget_refusal_keeps_run_control_priority() {
        use super::super::optimize::{
            OptimizerDispatchOutcome, OptimizerTermination,
            optimize_filters_with_run_control_detailed,
        };
        use super::super::run_control::OptimizerRunControl;
        use std::num::NonZeroUsize;

        let (objective, mut params, lower, upper, mut x) = scalar_objective();
        params.algo = "mh:rga".into();
        params.maxeval = 60;
        params.seed = Some(7);
        let root_control = OptimizerRunControl::new(NonZeroUsize::new(120).unwrap());
        let control = root_control.with_stage_budget(NonZeroUsize::new(60).unwrap());
        let run = optimize_filters_with_run_control_detailed(
            &mut x, &lower, &upper, objective, &params, &control,
        );
        assert_eq!(run.dispatch, OptimizerDispatchOutcome::BackendInvoked);
        assert_eq!(
            run.evidence.termination,
            OptimizerTermination::EvaluationLimit
        );
        assert!(!run.evidence.converged);
        assert!(
            run.stage_snapshot
                .expect("RGA stage budget")
                .evaluations_started
                <= 60
        );
        assert!(root_control.snapshot().evaluations_started <= 120);
    }

    #[test]
    fn parallel_mh_objective_reservations_never_score_denied_attempts() {
        use super::super::mh::{CallbackState, MHObjective};
        use metaheuristics_nature::ObjFunc;
        use std::sync::{Arc, Mutex};

        let (data, _, lower, upper, x) = scalar_objective();
        let state = Arc::new(Mutex::new(CallbackState {
            best_fitness: f64::INFINITY,
            best_params: Vec::new(),
            eval_count: 0,
            denied_evaluations: 0,
            last_report_eval: 0,
            iterations: 0,
            generations: 0,
            population_mean: None,
            population_stddev: None,
            callback_stopped: false,
        }));
        let objective = MHObjective {
            data,
            bounds: lower
                .into_iter()
                .zip(upper)
                .map(|(lo, hi)| [lo, hi])
                .collect(),
            callback_state: Some(Arc::clone(&state)),
            max_evaluations: 10,
        };
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let objective = objective.clone();
                    let x = x.clone();
                    scope.spawn(move || (0..32).map(|_| objective.fitness(&x)).collect::<Vec<_>>())
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("MH worker"))
                .collect::<Vec<_>>()
        });
        let state = state.lock().expect("MH objective state");
        assert_eq!(state.eval_count, 10);
        assert_eq!(state.denied_evaluations, 246);
        assert_eq!(results.iter().filter(|loss| loss.is_finite()).count(), 10);
        assert_eq!(
            results
                .iter()
                .filter(|loss| **loss == f64::INFINITY)
                .count(),
            246
        );
        assert!(state.best_fitness.is_finite());
        assert_eq!(state.best_params.len(), x.len());
    }

    #[test]
    fn registered_rga_completion_evidence_preserves_legacy_candidate() {
        use super::super::optimize::{
            optimize_filters_with_completion_evidence, optimize_filters_with_de_completion,
        };

        let (objective, mut params, lower, upper, x) = scalar_objective();
        params.algo = "mh:rga".into();
        params.maxeval = 60;
        params.seed = Some(7);
        let mut legacy_x = x.clone();
        let mut evidence_x = x;
        let (legacy_result, legacy_de) = optimize_filters_with_de_completion(
            &mut legacy_x,
            &lower,
            &upper,
            objective.clone(),
            &params,
        );
        let (evidence_result, evidence_de, search) = optimize_filters_with_completion_evidence(
            &mut evidence_x,
            &lower,
            &upper,
            objective,
            &params,
        );

        assert_eq!(evidence_result, legacy_result);
        assert_eq!(evidence_x, legacy_x);
        assert!(legacy_de.is_none());
        assert!(evidence_de.is_none());
        let search = search.expect("registered RGA search evidence");
        assert_eq!(search.evaluations, params.maxeval);
        // The backend can stop at the exact cap without attempting a denied call.
        assert_eq!(
            search.stop_cause,
            super::super::backend::BackendSearchStopCause::ObjectiveBudgetLimit
        );
    }

    #[test]
    fn registered_rga_callback_stop_preserves_run_control_priority() {
        use super::super::backend::BackendSearchStopCause;
        use super::super::optimize::{
            OptimizerTermination, optimize_filters_with_run_control_and_algo_override_detailed,
        };
        use super::super::run_control::OptimizerRunControl;
        use crate::de::CallbackAction;
        use std::num::NonZeroUsize;

        let (objective, mut params, lower, upper, mut x) = scalar_objective();
        params.algo = "mh:rga".into();
        params.maxeval = 500;
        params.seed = Some(7);
        let control = OptimizerRunControl::new(NonZeroUsize::new(500).unwrap());
        let run = optimize_filters_with_run_control_and_algo_override_detailed(
            &mut x,
            &lower,
            &upper,
            objective,
            &params,
            None,
            Some(Box::new(|_, _, _| CallbackAction::Stop)),
            &control,
        );
        assert_eq!(
            run.evidence.backend_stop_cause,
            Some(BackendSearchStopCause::ProgressCallbackStop),
        );
        assert_eq!(run.evidence.termination, OptimizerTermination::UserStopped);
        assert!(!run.evidence.converged);
        assert!(run.evidence.backend_evaluation_count.unwrap() >= 100);
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
    fn nsga_front_report_seed_reproducibility_feasibility_and_baseline() {
        use super::super::compute_pareto_objectives;
        use super::super::nsga::{build_nsga_front_report, nsga_front_report_json};
        use math_audio_optimisation::{NsgaConfig, NsgaVariant, ParetoSolution, nsga};
        use std::sync::Arc;

        fn run_front(seed: u64) -> Vec<ParetoSolution> {
            let (obj, _params, lower, upper, x0) = multi_objective();
            let objective = Arc::new(obj);
            let obj_for_call = objective.clone();
            let f = move |x: &Array1<f64>| -> Vec<f64> {
                compute_pareto_objectives(x.as_slice().unwrap(), &obj_for_call)
            };
            let bounds: Vec<(f64, f64)> = lower
                .iter()
                .zip(upper.iter())
                .map(|(&lo, &hi)| (lo, hi))
                .collect();
            let cfg = NsgaConfig {
                bounds,
                x0: Some(Array1::from(x0)),
                population_size: 8,
                maxeval: 32,
                variant: NsgaVariant::Nsga2,
                seed: Some(seed),
                ..Default::default()
            };
            let mut front = nsga(&f, cfg).expect("nsga runs").pareto_front;
            front.sort_by(|a, b| {
                a.objectives
                    .iter()
                    .zip(b.objectives.iter())
                    .map(|(x, y)| x.total_cmp(y))
                    .find(|o| *o != std::cmp::Ordering::Equal)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            front
        }

        // Same seed reproduces the same front through the real optimizer.
        let front_a = run_front(7);
        let front_b = run_front(7);
        assert!(!front_a.is_empty(), "nsga should return a front");
        assert_eq!(front_a.len(), front_b.len());
        for (a, b) in front_a.iter().zip(front_b.iter()) {
            assert_eq!(a.objectives.len(), b.objectives.len());
            for (x, y) in a.objectives.iter().zip(b.objectives.iter()) {
                assert!(
                    (x - y).abs() < 1e-12,
                    "same seed must reproduce the front: {x} vs {y}"
                );
            }
        }

        // Report over a real front: feasibility flags consistent, scalar-best
        // is the argmin baseline, compromise quality reported against it.
        let (obj, _params, lower, upper, _x) = multi_objective();
        let cfg = NsgaConfig {
            bounds: lower
                .iter()
                .zip(upper.iter())
                .map(|(&lo, &hi)| (lo, hi))
                .collect(),
            x0: None,
            population_size: 8,
            maxeval: 32,
            variant: NsgaVariant::Nsga2,
            seed: Some(7),
            ..Default::default()
        };
        let report =
            build_nsga_front_report("autoeq:nsga2", &cfg, &front_a, &obj, 32, 1).expect("report");
        assert_eq!(report.points.len(), front_a.len());
        assert_eq!(report.objectives.len(), front_a[0].objectives.len());
        let selected = &report.points[report.selection.selected_index];
        let scalar_best = &report.points[report.selection.scalar_best_index];
        assert!(selected.constraint.scalar_loss.is_finite());
        assert!(scalar_best.constraint.scalar_loss.is_finite());
        assert!(
            scalar_best.constraint.scalar_loss <= selected.constraint.scalar_loss + 1e-12,
            "scalar-best must be the scalar argmin baseline"
        );
        for point in &report.points {
            let ev = &point.constraint;
            assert!(
                ev.feasible
                    == (ev.ceiling_violation == 0.0
                        && ev.spacing_violation == 0.0
                        && ev.min_gain_violation == 0.0),
                "feasibility must match the reported violations"
            );
        }
        let json = nsga_front_report_json(&report);
        assert!(json.contains("autoeq.nsga_front/v1"));
        assert!(json.contains("normalized_compromise"));
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
