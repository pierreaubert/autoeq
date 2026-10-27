//! Solver-generation benchmarks using the shared evaluator fixtures.

#[path = "../../autoeq-objective/benches/fixtures/mod.rs"]
mod fixtures;

use autoeq_objective::loss::LossType;
use autoeq_objective::optim::compute_fitness_penalties_ref;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use fixtures::{objective, parameters};
use math_audio_optimisation::{
    CmaEsConfig, DEConfigBuilder, ParallelConfig, cma_es, differential_evolution,
};
use ndarray::Array1;
use std::hint::black_box;

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

criterion_group!(benches, benchmark_solver_generation);
criterion_main!(benches);
