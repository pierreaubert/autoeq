//! Physical output identities survive DBA optimization and serialized export.

// Rust guideline compliant 2026-02-21
use ndarray::Array1;
use roomeq_model::{
    CrossoverConfig, DBAConfig, MeasurementSource, OptimizerConfig, RoomConfig, SpeakerConfig,
    SubwooferCrossoverRef, SubwooferOutput, SubwooferStrategy, SubwooferSystemConfig, SystemConfig,
    SystemModel,
};

#[test]
fn home_cinema_retains_every_dba_driver_in_graph_and_export() {
    for (front_count, rear_count, sample_rate) in [(2, 2, 48_000.0), (2, 3, 96_000.0)] {
        let freq = Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 64);
        let main = roomeq_model::Curve {
            spl: freq.mapv(|f| {
                80.0 - if f < 80.0 {
                    24.0 * (80.0 / f).log2()
                } else {
                    0.0
                }
            }),
            phase: Some(Array1::zeros(freq.len())),
            freq: freq.clone(),
            ..Default::default()
        };
        let sub = roomeq_model::Curve {
            freq: freq.clone(),
            spl: Array1::from_elem(freq.len(), 80.0),
            phase: Some(Array1::zeros(freq.len())),
            ..Default::default()
        };
        let output_ids: Vec<_> = (0..front_count + rear_count)
            .map(|index| format!("Sub{}", index + 1))
            .collect();
        let config = RoomConfig {
            speakers: [
                (
                    "left".into(),
                    SpeakerConfig::Single(MeasurementSource::InMemory(main.clone())),
                ),
                (
                    "right".into(),
                    SpeakerConfig::Single(MeasurementSource::InMemory(main)),
                ),
                (
                    "array".into(),
                    SpeakerConfig::Dba(DBAConfig {
                        name: "array".into(),
                        speaker_name: None,
                        front: vec![MeasurementSource::InMemory(sub.clone()); front_count],
                        rear: vec![MeasurementSource::InMemory(sub); rear_count],
                    }),
                ),
            ]
            .into(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: [("L".into(), "left".into()), ("R".into(), "right".into())].into(),
                subwoofers: Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Dba,
                    crossover: Some(SubwooferCrossoverRef::PerSub(vec![
                        "bass".into();
                        output_ids.len()
                    ])),
                    routing: Default::default(),
                    outputs: output_ids
                        .iter()
                        .map(|id| SubwooferOutput {
                            id: id.clone(),
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
                max_iter: 8,
                algorithm: "autoeq:cobyla".into(),
                seed: Some(7),
                finalization: roomeq_model::FinalizationConfig {
                    default_input_peak: 0.01,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let result = roomeq_workflow::optimize_room(&config, sample_rate, None, None).unwrap();
        let bass = result.metadata.bass_management.as_ref().unwrap();
        assert_eq!(bass.physical_sub_outputs, output_ids);
        let routing = bass.routing_graph.as_ref().unwrap();
        assert_eq!(routing.physical_sub_outputs, output_ids);
        assert_eq!(routing.output_channels.len(), output_ids.len() + 2);
        for id in &output_ids {
            assert!(
                routing
                    .routes
                    .iter()
                    .any(|route| route.source_channel == "L" && route.destination == *id)
            );
            assert!(
                routing
                    .routes
                    .iter()
                    .any(|route| route.source_channel == "R" && route.destination == *id)
            );
        }
        let graph = result.to_dsp_chain_output();
        let roundtrip: roomeq_model::DspGraph =
            serde_json::from_slice(&serde_json::to_vec(&graph).unwrap()).unwrap();
        roundtrip.validate().unwrap();
        let physical: Vec<_> = roundtrip
            .channels
            .values()
            .flat_map(|chain| {
                chain
                    .drivers
                    .iter()
                    .flatten()
                    .map(|driver| driver.name.clone())
            })
            .collect();
        for id in &output_ids {
            assert_eq!(physical.iter().filter(|name| *name == id).count(), 1);
        }
        let yaml = roomeq_export::render_dsp_graph(
            &roundtrip,
            roomeq_export::ExportFormat::CamillaDsp,
            sample_rate,
        )
        .unwrap();
        assert!(yaml.contains(&format!(
            "playback:\n    type: Stdout\n    channels: {}",
            output_ids.len() + 2
        )));
        for id in &output_ids {
            assert!(
                yaml.contains(&format!("_to_{id}")),
                "missing exported branch {id}"
            );
        }
    }
}
