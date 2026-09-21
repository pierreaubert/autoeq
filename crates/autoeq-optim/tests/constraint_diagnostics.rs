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
