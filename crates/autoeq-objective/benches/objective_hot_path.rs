//! Objective evaluation benchmarks, independent of optimizer backends.

mod fixtures;

use autoeq_objective::loss::LossType;
use autoeq_objective::optim::{MultiObjectiveData, compute_fitness_penalties_ref};
use autoeq_objective::roomeq_types::MultiMeasurementStrategy;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fixtures::{objective, parameters};
use rayon::prelude::*;
use std::hint::black_box;

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

criterion_group!(
    benches,
    benchmark_scalar_objectives,
    benchmark_multi_measurement,
    benchmark_shared_pool_scaling
);
criterion_main!(benches);
