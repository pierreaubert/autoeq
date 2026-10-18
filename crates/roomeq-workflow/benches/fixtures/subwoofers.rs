//! Deterministic complex subwoofer fixtures shared by benchmarks and integration tests.

// Rust guideline compliant 2026-02-21
use ndarray::Array1;
use roomeq_engine::multisub::JointSubRequest;
use roomeq_model::{Curve, MeasurementSource, RoomConfig};

pub fn joint_request(subs: usize, seats: usize, bins: usize) -> JointSubRequest {
    let freq =
        Array1::from_iter((0..bins).map(|bin| 20.0 * 8.0_f64.powf(bin as f64 / (bins - 1) as f64)));
    JointSubRequest {
        measurements: (0..subs)
            .map(|sub| {
                (0..seats)
                    .map(|seat| Curve {
                        freq: freq.clone(),
                        spl: Array1::from_elem(bins, 80.0 + seat as f64 * 0.1),
                        phase: Some(freq.mapv(|frequency| {
                            -360.0 * frequency * (sub as f64 * 0.5 + seat as f64 * 0.01) / 1000.0
                        })),
                        ..Default::default()
                    })
                    .collect()
            })
            .collect(),
        target_db: vec![80.0 + 20.0 * (subs as f64).log10(); bins],
        weights: Default::default(),
        lfe_gain_db: vec![0.0; subs],
        redirected_gain_db: vec![0.0; subs],
        physical_outputs: (0..subs).map(|index| format!("Sub{}", index + 1)).collect(),
        reference_scope: Some(vec![vec!["benchmark-clock".into(); seats]; subs]),
    }
}

pub fn stationary_source(seats: usize) -> MeasurementSource {
    let freq = Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 64);
    let mut value = serde_json::json!({
        "measurements": (0..seats).map(|seat| serde_json::json!({
            "name": format!("seat-{seat}"), "frequencies": freq.to_vec(),
            "magnitude_db": vec![80.0 + seat as f64; freq.len()],
            "phase_deg": vec![seat as f64 * 45.0; freq.len()]
        })).collect::<Vec<_>>(),
        "provenance": {"capture_kind":"stationary_ir", "timing_reference_id":"fixture-common-clock",
            "capture": {"geometry":"spread", "takes": (0..seats).map(|seat|serde_json::json!({
                "microphone_id":format!("seat-{seat}"),"seat_id":format!("seat-{seat}"),"device_id":"synthetic-device",
                "offset_samples":0.0,"skew_ppm":0.0,"residual_uncertainty_us":1.0,
                "correction_applied":"resampled","timing_reference_id":"fixture-common-clock",
                "calibration_id":"fixture-calibration","gain_db":0.0,"calibration_orientation":"on_axis",
                "position_m":[seat as f64,0.0,0.0],"position_uncertainty_mm":1.0,
                "preserves_acoustic_delay":true,"quality_passed":true
            })).collect::<Vec<_>>()}
        }
    });
    if seats == 1 {
        let inline = value["measurements"][0].clone();
        let mut provenance = value["provenance"].take();
        provenance.as_object_mut().unwrap().remove("capture");
        value = serde_json::json!({"inline":inline,"provenance":provenance});
    }
    serde_json::from_value(value).expect("stationary inline benchmark captures")
}

pub fn home_cinema_joint(subs: usize, seats: usize) -> RoomConfig {
    use roomeq_model::{
        CrossoverConfig, MultiSeatConfig, MultiSubGroup, OptimizerConfig, SpeakerConfig,
        SubwooferCrossoverRef, SubwooferOutput, SubwooferStrategy, SubwooferSystemConfig,
        SystemConfig, SystemModel,
    };
    RoomConfig {
        speakers: [
            (
                "left".into(),
                SpeakerConfig::Single(stationary_source(seats)),
            ),
            (
                "right".into(),
                SpeakerConfig::Single(stationary_source(seats)),
            ),
            (
                "array".into(),
                SpeakerConfig::MultiSub(MultiSubGroup {
                    name: "array".into(),
                    speaker_name: None,
                    subwoofers: vec![stationary_source(seats); subs],
                    allpass_optimization: false,
                    joint_optimization: true,
                }),
            ),
        ]
        .into(),
        system: Some(SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: [("L".into(), "left".into()), ("R".into(), "right".into())].into(),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Mso,
                crossover: Some(SubwooferCrossoverRef::PerSub(vec!["bass".into(); subs])),
                routing: Default::default(),
                outputs: (0..subs)
                    .map(|index| SubwooferOutput {
                        id: format!("Sub{}", index + 1),
                        speaker: "array".into(),
                    })
                    .collect(),
            }),
            ..Default::default()
        }),
        crossovers: Some(
            [(
                "bass".into(),
                CrossoverConfig {
                    crossover_type: "LR24".into(),
                    frequency: Some(80.0),
                    frequencies: None,
                    frequency_range: None,
                },
            )]
            .into(),
        ),
        optimizer: OptimizerConfig {
            min_freq: 20.0,
            max_freq: 200.0,
            num_filters: 0,
            max_iter: 20,
            population: 10,
            algorithm: "autoeq:de".into(),
            seed: Some(7),
            parallel_threads: Some(1),
            joint_sub_evaluation_limit: std::num::NonZeroUsize::new(200),
            bass_management_evaluation_limit: std::num::NonZeroUsize::new(120),
            multi_seat: Some(MultiSeatConfig {
                enabled: true,
                primary_seat: usize::from(seats > 1),
                seat_identity: Some(roomeq_model::SeatIdentityMap {
                    ids: (0..seats).map(|seat| format!("seat-{seat}")).collect(),
                }),
                ..Default::default()
            }),
            finalization: roomeq_model::FinalizationConfig {
                default_input_peak: 0.01,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    }
}
