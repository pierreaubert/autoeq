use autoeq_optim::PeqModel;
use autoeq_optim::loss::{AsymmetricLossConfig, LossType};
use autoeq_optim::optim::{
    MultiObjectiveData, ObjectiveDataBuilder, SmoothnessPenaltyConfig,
    compute_fitness_penalties_ref,
};
use autoeq_optim::roomeq::{AudibilityDeadbandConfig, MultiMeasurementStrategy};
use math_audio_optimisation::{
    CmaEsConfig, DEConfigBuilder, ParallelConfig, cma_es, differential_evolution,
};
use ndarray::Array1;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;

#[global_allocator]
static ALLOCATOR: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn objective(loss_type: LossType) -> autoeq_optim::ObjectiveData {
    let frequencies = Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 200);
    let target = Array1::zeros(frequencies.len());
    let deviation = frequencies.mapv(|frequency| (frequency.log10() * 2.3).sin() * 3.0);
    ObjectiveDataBuilder::new(
        frequencies,
        target,
        deviation,
        48_000.0,
        PeqModel::Pk,
        loss_type,
    )
    .freq_range(20.0, 20_000.0)
    .smoothing(true, 3)
    .max_boost_envelope(vec![(20.0, 12.0), (20_000.0, 3.0)])
    .min_cut_envelope(vec![(20.0, -12.0), (20_000.0, -3.0)])
    .asymmetric_loss_config(AsymmetricLossConfig::default())
    .audibility_deadband(AudibilityDeadbandConfig {
        enabled: true,
        ..Default::default()
    })
    .smoothness_penalty(SmoothnessPenaltyConfig {
        tv2_weight: 0.1,
        ..Default::default()
    })
    .build()
    .expect("valid allocation-test objective")
}

#[test]
fn flat_and_asymmetric_candidates_allocate_nothing_after_warmup() {
    let parameters = vec![
        80.0_f64.log10(),
        1.2,
        -2.0,
        300.0_f64.log10(),
        2.0,
        3.0,
        1_000.0_f64.log10(),
        0.8,
        -1.5,
    ];
    for loss_type in [LossType::SpeakerFlat, LossType::SpeakerFlatAsymmetric] {
        let mut objective = objective(loss_type);
        objective.penalty_w_ceiling = 1.0;
        assert!(compute_fitness_penalties_ref(&parameters, &objective).is_finite());

        let region = Region::new(&ALLOCATOR);
        let fitness = compute_fitness_penalties_ref(&parameters, &objective);
        let allocations = region.change().allocations;

        assert!(fitness.is_finite());
        assert_eq!(allocations, 0, "{loss_type:?} allocated after warmup");
    }

    assert_multi_measurement_candidates_allocate_nothing_after_warmup();
    profile_generation_level_solver_allocations();
}

fn assert_multi_measurement_candidates_allocate_nothing_after_warmup() {
    let parameters = vec![
        80.0_f64.log10(),
        1.2,
        -2.0,
        300.0_f64.log10(),
        2.0,
        3.0,
        1_000.0_f64.log10(),
        0.8,
        -1.5,
    ];
    let strategies = [
        MultiMeasurementStrategy::WeightedSum,
        MultiMeasurementStrategy::Minimax,
        MultiMeasurementStrategy::VariancePenalized,
        MultiMeasurementStrategy::MinimaxUncertainty,
    ];

    for positions in [1, 3, 5, 9] {
        for strategy in strategies {
            let mut objectives: Vec<_> = (0..positions)
                .map(|_| objective(LossType::SpeakerFlat))
                .collect();
            let shared_freqs = objectives[0].freqs.clone();
            let shared_target = objectives[0].target.clone();
            for objective in objectives.iter_mut().skip(1) {
                objective.freqs = shared_freqs.clone();
                objective.target = shared_target.clone();
            }

            let mut combined = objectives[0].clone();
            combined.multi_objective = Some(MultiObjectiveData {
                objectives,
                strategy,
                weights: vec![1.0 / positions as f64; positions],
                variance_lambda: 0.5,
                uncertainty_cvar_alpha: Some(0.4),
            });

            assert!(compute_fitness_penalties_ref(&parameters, &combined).is_finite());
            let region = Region::new(&ALLOCATOR);
            let fitness = compute_fitness_penalties_ref(&parameters, &combined);
            let allocations = region.change().allocations;

            assert!(fitness.is_finite());
            assert_eq!(
                allocations, 0,
                "{strategy:?}/{positions} positions allocated after warmup"
            );
        }
    }
}

fn profile_generation_level_solver_allocations() {
    let bounds = vec![(-5.0, 5.0); 30];
    let noop = |x: &Array1<f64>| x[0] * 0.0;
    let de_config = || {
        DEConfigBuilder::new()
            .maxiter(1)
            .popsize(4)
            .tol(0.0)
            .atol(0.0)
            .seed(42)
            .parallel(ParallelConfig {
                enabled: false,
                num_threads: None,
            })
            .build()
            .expect("valid allocation-profile DE config")
    };
    let cma_config = || CmaEsConfig {
        bounds: bounds.clone(),
        lambda: 32,
        maxeval: 32,
        seed: Some(42),
        stagnation_window: usize::MAX,
        f_tol: -1.0,
        target_f: f64::NEG_INFINITY,
        parallel: ParallelConfig {
            enabled: false,
            num_threads: None,
        },
        ..Default::default()
    };

    let _ = differential_evolution(&noop, &bounds, de_config()).expect("DE warm-up");
    let de_region = Region::new(&ALLOCATOR);
    let de_report = differential_evolution(&noop, &bounds, de_config()).expect("profiled DE run");
    let de_stats = de_region.change();
    std::hint::black_box(&de_report);

    let _ = cma_es(&noop, cma_config()).expect("CMA-ES warm-up");
    let cma_region = Region::new(&ALLOCATOR);
    let cma_report = cma_es(&noop, cma_config()).expect("profiled CMA-ES run");
    let cma_stats = cma_region.change();
    std::hint::black_box(&cma_report);

    eprintln!(
        "one-generation solver allocations: DE={} allocations/{} bytes, CMA-ES={} allocations/{} bytes",
        de_stats.allocations,
        de_stats.bytes_allocated,
        cma_stats.allocations,
        cma_stats.bytes_allocated,
    );
    assert!(de_stats.bytes_allocated < 10_000_000);
    assert!(cma_stats.bytes_allocated < 10_000_000);
}
