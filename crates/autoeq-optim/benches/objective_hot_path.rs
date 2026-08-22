use autoeq_optim::PeqModel;
use autoeq_optim::loss::{AsymmetricLossConfig, LossType};
use autoeq_optim::optim::{
    MultiObjectiveData, ObjectiveData, ObjectiveDataBuilder, SmoothnessPenaltyConfig,
    compute_fitness_penalties_ref,
};
use autoeq_optim::roomeq::{AudibilityDeadbandConfig, MultiMeasurementStrategy};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use math_audio_optimisation::{
    CmaEsConfig, DEConfigBuilder, ParallelConfig, cma_es, differential_evolution,
};
use ndarray::Array1;
use rayon::prelude::*;
use std::hint::black_box;

fn parameters() -> Vec<f64> {
    (0..10)
        .flat_map(|index| {
            let frequency = 40.0 * 2.0_f64.powf(index as f64 * 0.85);
            [
                frequency.log10(),
                0.7 + index as f64 * 0.1,
                (index % 3) as f64 - 1.0,
            ]
        })
        .collect()
}

fn objective(loss_type: LossType, offset: f64) -> ObjectiveData {
    let frequencies = Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 200);
    let target = Array1::zeros(frequencies.len());
    let deviation = frequencies.mapv(|frequency| {
        (frequency.log10() * 3.1 + offset).sin() * 4.0 + (frequency.log10() * 9.0).cos()
    });
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
    .expect("benchmark objective")
}

fn benchmark_scalar_objectives(criterion: &mut Criterion) {
    let parameters = parameters();
    let mut group = criterion.benchmark_group("objective_200_points_10_filters");
    for (name, loss_type, constrained) in [
        ("flat", LossType::SpeakerFlat, false),
        ("flat_ceiling", LossType::SpeakerFlat, true),
        ("asymmetric", LossType::SpeakerFlatAsymmetric, false),
        ("asymmetric_ceiling", LossType::SpeakerFlatAsymmetric, true),
    ] {
        let mut objective = objective(loss_type, 0.0);
        objective.penalty_w_ceiling = if constrained { 1.0 } else { 0.0 };
        black_box(compute_fitness_penalties_ref(&parameters, &objective));
        group.bench_function(name, |bencher| {
            bencher.iter(|| {
                compute_fitness_penalties_ref(black_box(&parameters), black_box(&objective))
            })
        });
    }
    group.finish();
}

fn benchmark_multi_measurement(criterion: &mut Criterion) {
    let parameters = parameters();
    let mut group = criterion.benchmark_group("multi_measurement_200_points_10_filters");
    for positions in [1usize, 3, 5, 9] {
        for strategy in [
            MultiMeasurementStrategy::WeightedSum,
            MultiMeasurementStrategy::Minimax,
            MultiMeasurementStrategy::VariancePenalized,
            MultiMeasurementStrategy::MinimaxUncertainty,
        ] {
            let mut objectives: Vec<_> = (0..positions)
                .map(|index| objective(LossType::SpeakerFlat, index as f64 * 0.17))
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
            black_box(compute_fitness_penalties_ref(&parameters, &combined));
            group.bench_with_input(
                BenchmarkId::new(format!("{strategy:?}"), positions),
                &combined,
                |bencher, objective| {
                    bencher.iter(|| {
                        compute_fitness_penalties_ref(black_box(&parameters), black_box(objective))
                    })
                },
            );
        }
    }
    group.finish();
}

fn benchmark_shared_pool_scaling(criterion: &mut Criterion) {
    let base_parameters = parameters();
    let candidates: Vec<Vec<f64>> = (0..64)
        .map(|candidate| {
            base_parameters
                .iter()
                .enumerate()
                .map(|(index, value)| value + (candidate * (index + 1)) as f64 * 1e-7)
                .collect()
        })
        .collect();
    let channels: Vec<_> = (0..10)
        .map(|channel| objective(LossType::SpeakerFlatAsymmetric, channel as f64 * 0.11))
        .collect();
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let mut thread_counts = vec![1, 2, 4, 8, available];
    thread_counts.retain(|threads| *threads <= available);
    thread_counts.sort_unstable();
    thread_counts.dedup();
    let mut group = criterion.benchmark_group("shared_pool_10_channels_64_candidates");
    for threads in thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("benchmark pool");
        pool.install(|| {
            channels.par_iter().for_each(|objective| {
                candidates.par_iter().for_each(|candidate| {
                    black_box(compute_fitness_penalties_ref(candidate, objective));
                });
            });
        });
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |bencher, _| {
                bencher.iter(|| {
                    pool.install(|| {
                        channels.par_iter().for_each(|objective| {
                            candidates.par_iter().for_each(|candidate| {
                                black_box(compute_fitness_penalties_ref(candidate, objective));
                            });
                        });
                    })
                })
            },
        );
    }
    group.finish();
}

fn solver_bounds() -> Vec<(f64, f64)> {
    (0..10)
        .flat_map(|_| {
            [
                (20.0_f64.log10(), 20_000.0_f64.log10()),
                (0.1, 10.0),
                (-12.0, 12.0),
            ]
        })
        .collect()
}

fn benchmark_solver_generation(criterion: &mut Criterion) {
    let bounds = solver_bounds();
    let objective = objective(LossType::SpeakerFlatAsymmetric, 0.0);
    let noop = |x: &Array1<f64>| black_box(x[0]) * 0.0;
    let room = |x: &Array1<f64>| {
        compute_fitness_penalties_ref(
            x.as_slice().expect("solver candidate must be contiguous"),
            &objective,
        )
    };

    // Warm worker-local RoomEQ scratch before measuring solver generation work.
    black_box(room(&Array1::from(parameters())));

    let mut group = criterion.benchmark_group("solver_one_generation_30_parameters");
    macro_rules! add_solver_benchmarks {
        ($name:literal, $evaluate:expr) => {{
            let evaluate = &$evaluate;
            group.bench_function(format!("de/{}", $name), |bencher| {
                bencher.iter_batched(
                    || {
                        DEConfigBuilder::new()
                            .maxiter(1)
                            .popsize(4)
                            .tol(0.0)
                            .atol(0.0)
                            .seed(42)
                            .parallel(ParallelConfig {
                                enabled: true,
                                num_threads: None,
                            })
                            .build()
                            .expect("valid one-generation DE config")
                    },
                    |config| {
                        black_box(
                            differential_evolution(evaluate, black_box(&bounds), config)
                                .expect("one-generation DE benchmark must run"),
                        )
                    },
                    BatchSize::SmallInput,
                )
            });

            group.bench_function(format!("cmaes/{}", $name), |bencher| {
                bencher.iter_batched(
                    || CmaEsConfig {
                        bounds: bounds.clone(),
                        lambda: 32,
                        maxeval: 32,
                        seed: Some(42),
                        stagnation_window: usize::MAX,
                        f_tol: -1.0,
                        target_f: f64::NEG_INFINITY,
                        parallel: ParallelConfig {
                            enabled: true,
                            num_threads: None,
                        },
                        ..Default::default()
                    },
                    |config| {
                        black_box(
                            cma_es(evaluate, config)
                                .expect("one-generation CMA benchmark must run"),
                        )
                    },
                    BatchSize::SmallInput,
                )
            });
        }};
    }

    add_solver_benchmarks!("noop", noop);
    add_solver_benchmarks!("roomeq", room);
    group.finish();
}

criterion_group!(
    benches,
    benchmark_scalar_objectives,
    benchmark_multi_measurement,
    benchmark_shared_pool_scaling,
    benchmark_solver_generation
);
criterion_main!(benches);
