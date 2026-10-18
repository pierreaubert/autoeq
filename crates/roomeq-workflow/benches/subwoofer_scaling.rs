//! Fixed-score array scaling and verified phase-aware routed workloads.

// Rust guideline compliant 2026-02-21
#[path = "fixtures/subwoofers.rs"]
mod fixtures;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::{hint::black_box, time::Duration};

fn benchmark_joint_arrays(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("joint_sub_array_500_scores");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for (subs, seats, bins) in [(2, 1, 128), (4, 4, 512), (8, 8, 1024)] {
        let request = fixtures::joint_request(subs, seats, bins);
        let config = roomeq_model::OptimizerConfig {
            min_freq: 20.0,
            max_freq: 160.0,
            min_db: -3.0,
            max_db: 3.0,
            algorithm: "autoeq:de".into(),
            seed: Some(7),
            max_iter: 100,
            population: 20,
            tolerance: 0.0,
            atolerance: 0.0,
            joint_sub_evaluation_limit: std::num::NonZeroUsize::new(500),
            ..Default::default()
        };
        group.bench_function(
            BenchmarkId::new(format!("{subs}subs_{seats}seats"), bins),
            |bencher| {
                bencher.iter(|| {
                    let result = roomeq_engine::multisub::optimize_joint_sub_array(
                        black_box(&request),
                        black_box(&config),
                        48_000.0,
                    )
                    .unwrap();
                    assert_eq!(
                        result.search.evaluation_count, 500,
                        "fixture must execute the measured search work"
                    );
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn benchmark_verified_home_cinema(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("verified_joint_home_cinema");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for (subs, seats, fs) in [(2, 2, 44_100.0), (4, 3, 48_000.0), (8, 4, 96_000.0)] {
        let config = fixtures::home_cinema_joint(subs, seats);
        group.bench_function(
            format!("{subs}subs_{seats}seats_{}hz", fs as usize),
            |bencher| {
                bencher.iter(|| {
                    let result =
                        roomeq_workflow::optimize_room(black_box(&config), fs, None, None).unwrap();
                    let array = result
                        .channels
                        .values()
                        .find_map(|chain| chain.joint_sub.as_ref())
                        .expect("joint search must run");
                    assert!(array.search_evaluations > 0);
                    assert_eq!(array.seats.len(), seats);
                    assert_eq!(
                        result
                            .metadata
                            .bass_management
                            .as_ref()
                            .unwrap()
                            .physical_sub_outputs
                            .len(),
                        subs
                    );
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn benchmark_dba_and_fir(criterion: &mut Criterion) {
    let mut config = fixtures::home_cinema_joint(4, 1);
    config.optimizer.multi_seat = None;
    config.speakers.insert(
        "array".into(),
        roomeq_model::SpeakerConfig::Dba(roomeq_model::DBAConfig {
            name: "array".into(),
            speaker_name: None,
            front: vec![fixtures::stationary_source(1); 2],
            rear: vec![fixtures::stationary_source(1); 2],
        }),
    );
    config
        .system
        .as_mut()
        .unwrap()
        .subwoofers
        .as_mut()
        .unwrap()
        .config = roomeq_model::SubwooferStrategy::Dba;
    let mut group = criterion.benchmark_group("dba_and_residual_fir");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("verified_2front_2rear_home_cinema_48000hz", |bencher| {
        bencher.iter(|| {
            let result =
                roomeq_workflow::optimize_room(black_box(&config), 48_000.0, None, None).unwrap();
            let bass = result.metadata.bass_management.as_ref().unwrap();
            assert_eq!(bass.physical_sub_outputs.len(), 4);
            assert_eq!(
                bass.sub_outputs
                    .iter()
                    .filter(|output| output.strategy_source == "dba_front")
                    .count(),
                2
            );
            assert_eq!(
                bass.sub_outputs
                    .iter()
                    .filter(|output| output.strategy_source == "dba_rear")
                    .count(),
                2
            );
            black_box(result)
        })
    });
    let frequency = ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 512);
    let measurement = roomeq_model::Curve {
        freq: frequency.clone(),
        spl: frequency.mapv(|frequency| {
            80.0 + 6.0 * (-0.5 * ((frequency / 65.0).log2() / 0.2).powi(2)).exp()
        }),
        ..Default::default()
    };
    let target = roomeq_model::Curve {
        freq: frequency,
        spl: ndarray::Array1::from_elem(512, 80.0),
        ..Default::default()
    };
    for (taps, fs, phase) in [
        (8192, 44_100.0, "minimum"),
        (8192, 48_000.0, "linear"),
        (16384, 96_000.0, "minimum"),
    ] {
        let config = roomeq_model::OptimizerConfig {
            min_freq: 20.0,
            max_freq: 20_000.0,
            fir: Some(roomeq_model::FirConfig {
                taps,
                phase: phase.into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        group.bench_function(
            format!("residual_fir_{phase}_{taps}taps_{}hz", fs as usize),
            |bencher| {
                bencher.iter(|| {
                    let coefficients = roomeq_engine::fir::generate_fir_correction_prepared(
                        black_box(&measurement),
                        black_box(&config),
                        black_box(&target),
                        fs,
                    )
                    .unwrap();
                    assert_eq!(coefficients.len(), taps);
                    assert!(coefficients.iter().all(|value| value.is_finite()));
                    assert!(
                        coefficients
                            .iter()
                            .filter(|value| value.abs() > 1e-8)
                            .count()
                            > 1,
                        "nonflat fixture must design a correction"
                    );
                    black_box(coefficients)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_joint_arrays,
    benchmark_verified_home_cinema,
    benchmark_dba_and_fir
);
criterion_main!(benches);
