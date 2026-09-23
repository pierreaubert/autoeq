//! Engine-facing adapter contract for bounded optimization (O2).
//!
//! Uses only the public `autoeq_optim` API surface, the way the engine
//! worker consumes it: a backend run, backend-result evidence, the shared
//! envelope choke-point, and outcome classification. The engine owns the
//! consumer; this test pins the producer side of the contract.

use autoeq_optim::PeqModel;
use autoeq_optim::loss::LossType;
use autoeq_optim::optim::pareto::ParetoFilter;
use autoeq_optim::optim::{
    ConstraintSpec, MockOptimizerBackend, ObjectiveDataBuilder, OptimizationOutcomeKind,
    OptimizerBackend, OptimizerRunEvidence, OptimizerTermination, check_pareto_feasibility,
    classify_outcome, constrain_candidate,
};
use ndarray::Array1;

fn objective_with_envelope() -> autoeq_optim::ObjectiveData {
    let freqs: Vec<f64> =
        Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 200).to_vec();
    let freqs = Array1::from_vec(freqs);
    let n = freqs.len();
    ObjectiveDataBuilder::new(
        freqs,
        Array1::zeros(n),
        Array1::from_elem(n, 5.0),
        48_000.0,
        PeqModel::Pk,
        LossType::SpeakerFlat,
    )
    .max_db(12.0)
    .min_db(0.0)
    .freq_range(20.0, 20_000.0)
    .max_boost_envelope(vec![(20.0, 6.0), (20_000.0, 6.0)])
    .build()
    .expect("valid test objective")
}

#[test]
fn engine_adapter_reports_infeasible_despite_converged_backend() {
    let data = objective_with_envelope();
    // Two stacked +5 dB filters: each respects the per-filter bound, the
    // composite does not.
    let x = vec![1000.0_f64.log10(), 1.0, 5.0, 1000.0_f64.log10(), 1.0, 5.0];
    let backend = MockOptimizerBackend::ok("converged", 0.5);
    let params = autoeq_optim::OptimParams {
        num_filters: 2,
        peq_model: PeqModel::Pk,
        sample_rate: 48_000.0,
        min_freq: 20.0,
        max_freq: 20_000.0,
        min_q: 0.5,
        max_q: 12.0,
        min_db: -12.0,
        max_db: 12.0,
        loss: LossType::SpeakerFlat,
        smooth: false,
        smooth_n: 1,
        min_spacing_oct: 0.0,
        spacing_weight: 0.0,
        smoothness_penalty: None,
        audibility_deadband: None,
        frequency_q_policy: None,
        algo: String::from("mock"),
        population: 6,
        maxeval: 60,
        refine: false,
        local_algo: String::from("autoeq:cobyla"),
        bo_initial_samples: 0,
        bo_batch_size: 0,
        bo_posterior_std_threshold: 0.0,
        bo_acquisition: String::new(),
        bo_ehvi: false,
        strategy: String::new(),
        tolerance: 0.0,
        atolerance: 0.0,
        recombination: 0.0,
        adaptive_weight_f: 0.0,
        adaptive_weight_cr: 0.0,
        no_parallel: true,
        parallel_threads: 0,
        seed: Some(3),
        quiet: true,
    };
    // Tight bounds around the candidate: the termination under test comes
    // from the backend status text, never from a bound violation.
    let lower = x.iter().map(|v| v - 0.5).collect::<Vec<_>>();
    let upper = x.iter().map(|v| v + 0.5).collect::<Vec<_>>();
    let mut x_run = x.clone();
    let result = backend.optimize_filters(&mut x_run, &lower, &upper, data.clone(), &params);
    assert!(result.is_ok());
    let evidence = OptimizerRunEvidence::from_backend_result(
        "mock",
        result,
        &x_run,
        &lower,
        &upper,
        params.maxeval,
        params.seed,
    );
    assert_eq!(evidence.termination, OptimizerTermination::Converged);

    let candidate =
        constrain_candidate("engine-seed-0", &x, &data, &ConstraintSpec::unconstrained())
            .expect("valid candidate");
    assert!(!candidate.feasible);
    let outcome = classify_outcome(&evidence, &candidate);
    // The engine must see infeasibility, not the backend's converged status.
    assert_eq!(outcome.kind, OptimizationOutcomeKind::NoFeasibleCandidate);
    assert_eq!(outcome.candidate_id, "engine-seed-0");
    assert!(!outcome.diagnostics.is_empty());
    assert!(!outcome.detail.is_empty());
    assert!(!outcome.implies_physical_impossibility());

    // Pareto members flow through the same rules with position identities.
    let front = vec![
        ParetoFilter {
            params: x.clone(),
            flatness_loss: 0.5,
            score_loss: None,
            num_filters: 2,
            converged: true,
        },
        ParetoFilter {
            params: vec![500.0_f64.log10(), 1.0, 0.0],
            flatness_loss: 4.0,
            score_loss: None,
            num_filters: 1,
            converged: true,
        },
    ];
    let feasibility = check_pareto_feasibility(&front, &data, &ConstraintSpec::unconstrained())
        .expect("valid front");
    assert_eq!(feasibility.len(), 2);
    assert_eq!(feasibility[0].candidate_id, "pareto-0");
    assert!(!feasibility[0].feasible);
    assert_eq!(feasibility[1].candidate_id, "pareto-1");
    assert!(feasibility[1].feasible);
}

fn dispatcher_params(num_filters: usize) -> autoeq_optim::OptimParams {
    autoeq_optim::OptimParams {
        num_filters,
        peq_model: PeqModel::Pk,
        sample_rate: 48_000.0,
        min_freq: 20.0,
        max_freq: 20_000.0,
        min_q: 0.5,
        max_q: 12.0,
        min_db: -12.0,
        max_db: 12.0,
        loss: LossType::SpeakerFlat,
        smooth: false,
        smooth_n: 1,
        min_spacing_oct: 0.0,
        spacing_weight: 0.0,
        smoothness_penalty: None,
        audibility_deadband: None,
        frequency_q_policy: None,
        algo: String::from("autoeq:cobyla"),
        population: 6,
        maxeval: 30,
        refine: false,
        local_algo: String::from("autoeq:cobyla"),
        bo_initial_samples: 0,
        bo_batch_size: 0,
        bo_posterior_std_threshold: 0.0,
        bo_acquisition: String::new(),
        bo_ehvi: false,
        strategy: String::new(),
        tolerance: 0.0,
        atolerance: 0.0,
        recombination: 0.0,
        adaptive_weight_f: 0.0,
        adaptive_weight_cr: 0.0,
        no_parallel: true,
        parallel_threads: 0,
        seed: Some(3),
        quiet: true,
    }
}

/// A3 dispatcher invariant: stacked breaches cannot escape through a real
/// backend. Tight bounds pin the winner inside the breach so the shared
/// finalization must refuse with composite evidence.
#[test]
fn roadmap_correction_dispatcher_refuses_stacked_breach() {
    use autoeq_optim::optim::optimize_filters;
    let data = objective_with_envelope();
    let x0 = vec![1000.0_f64.log10(), 1.0, 10.0, 1000.0_f64.log10(), 1.0, 10.0];
    let lower = x0.iter().map(|v| v - 0.5).collect::<Vec<_>>();
    let upper = x0.iter().map(|v| v + 0.5).collect::<Vec<_>>();
    let params = dispatcher_params(2);
    let mut x = x0.clone();
    let error = optimize_filters(&mut x, &lower, &upper, data, &params)
        .expect_err("stacked breach refused through cobyla");
    assert!(error.0.contains("composite"), "{}", error.0);
}

/// A violating refinement result is refused through the algo-override
/// (local-refine) dispatcher, not only the global path. Tight bounds pin
/// the refined winner inside the breach so the shared finalization must
/// refuse with composite evidence.
#[test]
fn roadmap_correction_refine_dispatcher_refuses_stacked_breach() {
    use autoeq_optim::optim::optimize_filters_with_algo_override;
    let data = objective_with_envelope();
    let x0 = vec![1000.0_f64.log10(), 1.0, 10.0, 1000.0_f64.log10(), 1.0, 10.0];
    let lower = x0.iter().map(|v| v - 0.5).collect::<Vec<_>>();
    let upper = x0.iter().map(|v| v + 0.5).collect::<Vec<_>>();
    let params = dispatcher_params(2);
    let mut x = x0.clone();
    let error = optimize_filters_with_algo_override(
        &mut x,
        &lower,
        &upper,
        data,
        &params,
        Some("autoeq:cobyla"),
    )
    .expect_err("stacked breach refused through the refine dispatcher");
    assert!(error.0.contains("composite"), "{}", error.0);
}

/// A repaired single breach comes back feasible with a re-verified loss.
#[test]
fn roadmap_correction_dispatcher_repairs_single_breach() {
    use autoeq_optim::optim::{
        ConstraintSpec, compute_fitness_penalties_ref, constrain_candidate, optimize_filters,
    };
    let data = objective_with_envelope();
    let x0 = vec![1000.0_f64.log10(), 1.0, 10.0];
    let lower = x0.iter().map(|v| v - 0.5).collect::<Vec<_>>();
    let upper = x0.iter().map(|v| v + 0.5).collect::<Vec<_>>();
    let params = dispatcher_params(1);
    let mut x = x0.clone();
    let (_, loss) = optimize_filters(&mut x, &lower, &upper, data.clone(), &params)
        .expect("repaired breach finalizes");
    let check = constrain_candidate("dispatched", &x, &data, &ConstraintSpec::unconstrained())
        .expect("valid re-check");
    assert!(check.feasible, "emitted params feasible");
    let fresh = compute_fitness_penalties_ref(&x, &data);
    assert!(
        (loss - fresh).abs() < 1e-9,
        "returned loss re-verified at emitted params: {loss} vs {fresh}"
    );
}

/// An infeasible Pareto member is refused before selection: only the
/// feasible member survives judging, with objectives re-evaluated at the
/// repaired parameters. A front with no feasible member is an explicit
/// refusal, never a silent fallback.
#[test]
fn pareto_judging_refuses_infeasible_member_before_selection() {
    use autoeq_optim::optim::{compute_pareto_objectives, judge_pareto_members};
    let data = objective_with_envelope();
    // Member 0: two stacked +5 dB filters, each within per-filter bounds,
    // composite over the +6 dB envelope. Member 1: flat, feasible.
    let breaching = vec![1000.0_f64.log10(), 1.0, 5.0, 1000.0_f64.log10(), 1.0, 5.0];
    let flat = vec![500.0_f64.log10(), 1.0, 0.0];
    let judged = judge_pareto_members(
        "test",
        &[breaching.clone(), flat.clone()],
        &data,
        &ConstraintSpec::unconstrained(),
    )
    .expect("one feasible member survives");
    assert_eq!(judged.submitted, 2);
    assert_eq!(judged.refused, 1);
    assert_eq!(judged.members.len(), 1, "breaching member not selected");
    assert_eq!(judged.members[0].index, 1);
    assert_eq!(judged.members[0].params, flat);
    let fresh = compute_pareto_objectives(&judged.members[0].params, &data);
    assert_eq!(
        judged.members[0].objectives, fresh,
        "selection judges the re-evaluated score"
    );

    let error = judge_pareto_members(
        "test",
        &[breaching.clone(), breaching],
        &data,
        &ConstraintSpec::unconstrained(),
    )
    .expect_err("all-infeasible front refused");
    assert!(error.contains("refused all 2"), "{error}");
}

/// Joint/gain candidates honor their search budgets at acceptance: boundary
/// values pass exactly, over-budget, non-finite, and length-mismatched
/// vectors refuse with identifying detail.
#[test]
fn joint_budgets_refuse_out_of_budget_controls() {
    use autoeq_optim::optim::verify_joint_budgets;
    let lower = vec![-12.0, -12.0, 0.0, 0.0];
    let upper = vec![12.0, 12.0, 20.0, 20.0];
    assert!(
        verify_joint_budgets("test", &[0.0, -3.0, 0.0, 10.0], &lower, &upper).is_ok(),
        "in-budget gains pass"
    );
    assert!(
        verify_joint_budgets("test", &[-12.0, 12.0, 0.0, 20.0], &lower, &upper).is_ok(),
        "boundary values pass exactly"
    );
    let error = verify_joint_budgets("test", &[0.0, 13.0, 0.0, 10.0], &lower, &upper)
        .expect_err("over-budget gain refused");
    assert!(error.contains("[1]"), "{error}");
    assert!(error.contains("joint budget breach"), "{error}");
    verify_joint_budgets("test", &[0.0, f64::NAN, 0.0, 10.0], &lower, &upper)
        .expect_err("non-finite control refused");
    verify_joint_budgets("test", &[0.0], &lower, &upper).expect_err("length mismatch refused");
}
