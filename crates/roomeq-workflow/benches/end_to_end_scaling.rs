use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::Array1;
use roomeq_model::{
    CrossoverConfig, MeasurementSource, OptimizerConfig, ProcessingMode, RoomConfig, SpeakerConfig,
    SubwooferStrategy, SubwooferSystemConfig, SystemConfig, SystemModel,
};
use roomeq_workflow::optimize_room;
use std::hint::black_box;
use std::time::Duration;

const CHANNELS_5_1_4: [(&str, &str); 10] = [
    ("L", "left"),
    ("R", "right"),
    ("C", "center"),
    ("LFE", "lfe"),
    ("SL", "surround_left"),
    ("SR", "surround_right"),
    ("TFL", "top_front_left"),
    ("TFR", "top_front_right"),
    ("TRL", "top_rear_left"),
    ("TRR", "top_rear_right"),
];

/// Index of the full-range LFE measurement in [`CHANNELS_5_1_4`].
/// Satellite acoustic rolloff does not attenuate redirected electrical inputs.
const LFE_CHANNEL_INDEX: usize = 3;
// Bound the sum of ten simultaneous programme inputs to unit peak before
// route/filter gain. Final electrical checks still enforce the actual graph's
// output ceiling and the unchanged default maximum safety attenuation.
const INPUT_PEAK: f64 = 1.0 / CHANNELS_5_1_4.len() as f64;
/// Satellites carry no useful content below this corner in the fixture.
const SATELLITE_HIGHPASS_HZ: f64 = 80.0;
/// Rolloff applied below the satellite corner (24 dB per octave).
const SATELLITE_HIGHPASS_SLOPE_DB_PER_OCTAVE: f64 = 24.0;

fn response(channel_index: usize) -> roomeq_model::Curve {
    let freq = Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 200);
    let phase = channel_index as f64 * 0.19;
    let spl = freq.mapv(|frequency| {
        let base = 80.0
            + (frequency.log10() * 5.7 + phase).sin() * 4.0
            + (frequency.log10() * 13.0 - phase).cos() * 1.5;
        if channel_index == LFE_CHANNEL_INDEX || frequency >= SATELLITE_HIGHPASS_HZ {
            base
        } else {
            base - SATELLITE_HIGHPASS_SLOPE_DB_PER_OCTAVE
                * (SATELLITE_HIGHPASS_HZ / frequency).log2()
        }
    });
    roomeq_model::Curve {
        freq: freq.clone(),
        spl,
        // This benchmark is synthetic; declare its deterministic branch phase
        // so final coherent replay tests the workload rather than missing data.
        phase: Some(Array1::from_elem(freq.len(), phase)),
        ..Default::default()
    }
}

fn config_5_1_4(threads: usize) -> RoomConfig {
    let speakers = CHANNELS_5_1_4
        .iter()
        .enumerate()
        .map(|(index, (_, channel))| {
            (
                (*channel).to_string(),
                SpeakerConfig::Single(MeasurementSource::InMemory(response(index))),
            )
        })
        .collect();
    let system_speakers = CHANNELS_5_1_4
        .iter()
        .filter(|(role, _)| *role != "LFE")
        .map(|(role, channel)| ((*role).to_string(), (*channel).to_string()))
        .collect();

    RoomConfig {
        version: roomeq_model::default_config_version(),
        system: Some(SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: system_speakers,
            // Keep the scaling fixture's LFE as an explicitly mapped output;
            // crossover attenuation is tested by topology fixtures, not by
            // this throughput benchmark.
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                crossover: Some(roomeq_model::SubwooferCrossoverRef::PerSub(vec![
                    "main".to_string(),
                ])),
                routing: Default::default(),
                outputs: vec![roomeq_model::SubwooferOutput {
                    id: "lfe".to_string(),
                    speaker: "lfe".to_string(),
                }],
            }),
            bass_management: None,
            ..Default::default()
        }),
        speakers,
        crossovers: Some(
            [(
                "main".to_string(),
                CrossoverConfig {
                    crossover_type: "LR24".to_string(),
                    frequency: Some(20.0),
                    frequencies: None,
                    frequency_range: None,
                },
            )]
            .into(),
        ),
        target_curve: None,
        optimizer: OptimizerConfig {
            processing_mode: ProcessingMode::LowLatency,
            algorithm: "autoeq:de".to_string(),
            num_filters: 10,
            max_iter: 10_000,
            population: 30,
            min_freq: 20.0,
            max_freq: 2_000.0,
            psychoacoustic: false,
            asymmetric_loss: true,
            refine: false,
            seed: Some(42),
            parallel_threads: Some(threads),
            finalization: roomeq_model::FinalizationConfig {
                default_input_peak: INPUT_PEAK,
                ..Default::default()
            },
            ..Default::default()
        },
        provenance: Default::default(),
        recording_config: None,
        measured_impulse_responses: Default::default(),
        ctc: None,
        reporting: None,
        cea2034_cache: None,
    }
}

fn config_5_0_4(threads: usize) -> RoomConfig {
    let mut config = config_5_1_4(threads);
    config.speakers.remove("lfe");
    config.crossovers = None;
    if let Some(system) = config.system.as_mut() {
        system.subwoofers = None;
    }
    config
}

fn scaling_thread_counts() -> Vec<usize> {
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let mut thread_counts = vec![1, 2, 4, 8, available.min(CHANNELS_5_1_4.len()), available];
    thread_counts.retain(|threads| *threads <= available);
    thread_counts.sort_unstable();
    thread_counts.dedup();
    thread_counts
}

fn benchmark_5_1_4_scaling(criterion: &mut Criterion) {
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let mut thread_counts = vec![1, 2, 4, 8, available.min(CHANNELS_5_1_4.len()), available];
    thread_counts.retain(|threads| *threads <= available);
    thread_counts.sort_unstable();
    thread_counts.dedup();

    let mut group = criterion.benchmark_group("roomeq_5_1_4_fixed_seed");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    for threads in thread_counts {
        let config = config_5_1_4(threads);
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &config,
            |bencher, config| {
                bencher.iter(|| {
                    black_box(
                        optimize_room(black_box(config), 48_000.0, None, None)
                            .expect("5.1.4 scaling benchmark must succeed"),
                    )
                })
            },
        );
    }
    group.finish();
}

fn benchmark_5_0_4_scaling(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("roomeq_5_0_4_independent_channels_fixed_seed");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    for threads in scaling_thread_counts() {
        let config = config_5_0_4(threads);
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &config,
            |bencher, config| {
                bencher.iter(|| {
                    black_box(
                        optimize_room(black_box(config), 48_000.0, None, None)
                            .expect("5.0.4 scaling benchmark must succeed"),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_5_1_4_scaling, benchmark_5_0_4_scaling);
criterion_main!(benches);
