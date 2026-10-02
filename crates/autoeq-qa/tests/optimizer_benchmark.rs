use autoeq_optim::optim::registry::all_algorithms;
use autoeq_optim::optim::setup::setup_bounds;
use autoeq_optim::{OptimParams, cli::Args};
use autoeq_qa::optimizer_benchmark::{
    BenchmarkCancellation, BenchmarkRunOptions, benchmark_manifest, run_optimizer_benchmark,
    run_optimizer_benchmark_cell, run_optimizer_benchmark_cell_with_time_budget,
};
use clap::Parser;

#[test]
fn fixed_manifest_and_measurement_sources_are_identified() {
    let manifest = benchmark_manifest().expect("fixed matrix manifest");
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.cases.len(), 3);
    assert_eq!(manifest.default_evaluation_budget, 128);
    assert_eq!(manifest.default_time_budget_millis, 10_000);
    assert!(manifest.seed_set.len() >= 3);

    let report = run_optimizer_benchmark(
        BenchmarkRunOptions {
            evaluation_budget: Some(64),
            time_budget_millis: Some(10_000),
            seeds: Some(vec![1]),
            limit_cells: Some(0),
        },
        BenchmarkCancellation::default(),
    )
    .expect("fixed fixtures load and hash");
    assert!(!report.complete_matrix);
    assert_eq!(report.cells.len(), 0);
    assert_eq!(report.fixtures.len(), 3);
    assert!(report.fixtures.iter().all(|fixture| {
        fixture.declaration_sha256.is_some() || !fixture.source_sha256.is_empty()
    }));
    let measured = report
        .fixtures
        .iter()
        .find(|fixture| fixture.declaration.id == "measured_stereo_8361a")
        .expect("measured stereo case");
    assert!(measured.declaration.capture_devices.iter().all(|device| {
        device.clock_domain == "not recorded" && device.calibration == "not recorded"
    }));
}

#[test]
fn bounded_cell_reports_resolved_backend_and_actual_evaluations() {
    let cell = run_optimizer_benchmark_cell("analytic_headphone_peq", "autoeq:cobyla", 7, 64)
        .expect("one fixed optimizer cell");

    assert_eq!(cell.resolved_backend, "autoeq:cobyla");
    assert_eq!(cell.seed, 7);
    assert_eq!(cell.declared_evaluation_budget, 64);
    assert!(cell.budget_profile.is_some());
    assert!(cell.evaluation_counts.search_started <= 64);
    assert_eq!(
        cell.evaluation_counts.total_completed_candidate_evaluations,
        cell.evaluation_counts.search_completed
            + cell.evaluation_counts.finalization_completed
            + cell.evaluation_counts.metric_candidate_evaluations
    );
    assert_eq!(cell.training.len(), 2);
    assert!(
        cell.training
            .iter()
            .all(|metric| metric.baseline_loss.is_some())
    );
    if cell.feasible {
        assert!(cell.parameters.is_some());
        assert!(cell.realized.is_some());
        assert!(cell.comparison_available);
        assert_eq!(cell.comparison_measurements_expected, 2);
        assert_eq!(cell.comparison_measurements_available, 2);
        assert!(
            cell.training
                .iter()
                .all(|metric| metric.final_loss.is_some())
        );
    } else {
        assert!(cell.refusal.is_some());
    }
}

#[test]
fn analytic_de_fit_improves_the_realized_objective_over_identity() {
    let cell = run_optimizer_benchmark_cell("analytic_headphone_peq", "autoeq:de", 7, 97)
        .expect("one deterministic DE fit");

    assert!(cell.feasible, "DE result was refused: {:?}", cell.refusal);
    assert_eq!(cell.evaluation_counts.search_started, 97);
    assert_eq!(cell.evaluation_counts.search_completed, 97);
    assert_eq!(
        cell.termination,
        Some(autoeq_optim::optim::OptimizerTermination::EvaluationLimit)
    );
    let worst_baseline = cell
        .training
        .iter()
        .filter_map(|metric| metric.baseline_loss)
        .reduce(f64::max)
        .expect("two baseline measurement scores");
    let worst_final = cell
        .training
        .iter()
        .filter_map(|metric| metric.final_loss)
        .reduce(f64::max)
        .expect("two final measurement scores");
    assert!(
        worst_final < worst_baseline - 0.1,
        "expected an actual correction: baseline={worst_baseline}, final={worst_final}, filters={:?}",
        cell.parameters
    );
    let realized = cell.realized.expect("accepted candidate realization");
    assert!(realized.active_filter_count > 0);
    assert!(realized.rms_transfer_db > 0.1);
}

#[test]
fn de_budget_below_one_complete_fresh_search_unit_is_refused_before_scoring() {
    let cell = run_optimizer_benchmark_cell("analytic_headphone_peq", "autoeq:de", 7, 96)
        .expect("preflight refusal is a reported cell");

    assert_eq!(cell.evaluation_counts.search_started, 0);
    assert!(!cell.feasible);
    assert_eq!(
        cell.termination,
        Some(autoeq_optim::optim::OptimizerTermination::EvaluationLimit)
    );
    assert!(
        cell.refusal
            .as_deref()
            .is_some_and(|reason| reason.contains("requires at least 97"))
    );
}

#[test]
fn cell_deadline_closes_search_and_reports_timeout() {
    let cell = run_optimizer_benchmark_cell_with_time_budget(
        "analytic_headphone_peq",
        "autoeq:de",
        7,
        128,
        1,
    )
    .expect("one cell with a short deadline");

    assert!(cell.timed_out, "optimizer completed inside a 1 ms cutoff");
    assert_eq!(
        cell.termination,
        Some(autoeq_optim::optim::OptimizerTermination::TimedOut)
    );
    assert!(
        cell.feasible,
        "timeout should retain a finite finalized candidate: {cell:?}"
    );
    assert!(
        cell.parameters
            .as_ref()
            .is_some_and(|values| { values.iter().all(|value| value.is_finite()) })
    );
    assert!(cell.realized.is_some());
    assert!(!cell.user_cancelled);
    let serialized = serde_json::to_value(&cell).expect("cell report serializes");
    assert_eq!(serialized["timed_out"], true);
    assert_eq!(serialized["user_cancelled"], false);
    assert!(cell.evaluation_counts.search_started <= 128);
    assert!(
        cell.refusal
            .as_deref()
            .is_some_and(|reason| reason.contains("time budget"))
    );
}

#[test]
fn pre_cancelled_matrix_returns_an_explicit_partial_report() {
    let cancellation = BenchmarkCancellation::default();
    cancellation.request_and_wait_for_scores();
    let report = run_optimizer_benchmark(
        BenchmarkRunOptions {
            evaluation_budget: Some(64),
            time_budget_millis: Some(10_000),
            seeds: Some(vec![1]),
            limit_cells: None,
        },
        cancellation,
    )
    .expect("cancelled run has a partial report");
    assert!(report.cancelled);
    assert!(!report.complete_matrix);
    assert!(report.cells.is_empty());
}

#[test]
fn every_registered_backend_reports_a_complete_batch_profile() {
    let mut args = Args::parse_from(["autoeq"]);
    args.num_filters = 4;
    args.sample_rate = 48_000.0;
    args.min_freq = 20.0;
    args.max_freq = 20_000.0;
    args.min_q = 0.5;
    args.max_q = 6.0;
    args.min_db = -9.0;
    args.max_db = 6.0;
    args.population = 8;
    args.maxeval = 128;
    args.no_parallel = true;
    args.parallel_threads = 1;
    let params = OptimParams::from(&args);
    let (lower, upper) = setup_bounds(&params);

    let backends = all_algorithms();
    assert!(!backends.is_empty());
    for backend in backends {
        let profile = backend
            .evaluation_budget_profile(&lower, &upper, &params)
            .unwrap_or_else(|| {
                panic!("{} does not report matched-budget settings", backend.name())
            });
        assert_eq!(profile.requested_evaluations, 128, "{}", backend.name());
        assert!(profile.minimum_complete_batch > 0, "{}", backend.name());
        assert!(
            profile.minimum_complete_batch <= 128,
            "{} requires {} evaluations",
            backend.name(),
            profile.minimum_complete_batch
        );
        assert!(profile.initial_batch_size > 0, "{}", backend.name());
        let first_complete_unit = profile
            .generation_batch_size
            .unwrap_or(0)
            .saturating_add(profile.initial_batch_size);
        assert!(
            profile.minimum_complete_batch <= first_complete_unit,
            "{} declares minimum {} but initial/generation batches total only {}",
            backend.name(),
            profile.minimum_complete_batch,
            first_complete_unit
        );
    }
}

#[test]
fn fresh_de_profile_counts_x0_and_refuses_incomplete_search_units() {
    use autoeq_optim::optim::registry;

    let backend = registry::resolve("autoeq:de").expect("DE backend is registered");
    let mut args = Args::parse_from(["autoeq"]);
    args.num_filters = 4;
    args.sample_rate = 48_000.0;
    args.min_freq = 20.0;
    args.max_freq = 20_000.0;
    args.min_q = 0.5;
    args.max_q = 6.0;
    args.min_db = -9.0;
    args.max_db = 6.0;
    args.population = 48;
    args.no_parallel = true;
    args.parallel_threads = 1;

    for (budget, expected_generations) in [(48, 1), (49, 1), (96, 1), (97, 1), (128, 1)] {
        args.maxeval = budget;
        let params = OptimParams::from(&args);
        let lower = vec![-1.0; 12];
        let upper = vec![1.0; 12];
        let profile = backend
            .evaluation_budget_profile(&lower, &upper, &params)
            .expect("DE profile is available");

        assert_eq!(profile.requested_evaluations, budget);
        assert_eq!(profile.population_size, Some(48));
        assert_eq!(profile.initial_batch_size, 49, "budget {budget}");
        assert_eq!(profile.generation_batch_size, Some(48), "budget {budget}");
        assert_eq!(profile.generation_limit, Some(expected_generations));
        assert_eq!(profile.solver_evaluation_limit, Some(97));
        assert_eq!(profile.minimum_complete_batch, 97);
        assert_eq!(
            profile.minimum_complete_batch <= budget,
            budget >= 97,
            "budget {budget} must be refused unless one complete fresh DE unit fits"
        );
    }
}
