use super::channel_layout::ChannelLayout;
use super::consts::KAUTZ_REFERENCE_MODES;
use super::consts::QA_MAXEVAL;
use super::consts::SEED;
use super::types::DifficultyLevel;
use super::types::SubTopology;
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use roomeq_model::{
    CardioidConfig, CrossoverConfig, Curve, DBAConfig, FirConfig, MeasurementSource,
    MixedPhaseSerdeConfig, MultiSubGroup, OptimizerConfig, ProcessingMode, RoomConfig,
    SpeakerConfig, SubwooferStrategy, SubwooferSystemConfig, SystemConfig, default_config_version,
};
use roomeq_synthetic::{
    generate_cardioid_scenario, generate_channel_curve, generate_dba_scenario,
    generate_multisub_scenario, generate_subwoofer_rolloff_curve,
};
use std::collections::HashMap;

pub(super) fn build_config(degraded: &Curve, mode: ProcessingMode) -> RoomConfig {
    let mut speakers = HashMap::new();
    speakers.insert(
        "Left".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(degraded.clone())),
    );
    speakers.insert(
        "Right".to_string(),
        SpeakerConfig::Single(MeasurementSource::InMemory(degraded.clone())),
    );

    let mut config = RoomConfig {
        version: default_config_version(),
        system: None,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: Default::default(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    config.optimizer.algorithm = "autoeq:cmaes".to_string();
    config.optimizer.max_iter = QA_MAXEVAL;
    config.optimizer.population = 50;
    config.optimizer.refine = false;
    config.optimizer.seed = Some(SEED);
    config.optimizer.num_filters = 3;
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 20000.0;
    configure_processing_mode(&mut config.optimizer, mode);

    config
}

pub(super) fn configure_processing_mode(optimizer: &mut OptimizerConfig, mode: ProcessingMode) {
    optimizer.processing_mode = mode;

    match optimizer.processing_mode {
        ProcessingMode::PhaseLinear | ProcessingMode::Hybrid => {
            optimizer.max_freq = optimizer.max_freq.min(1500.0);
            optimizer.fir.get_or_insert_with(default_qa_fir_config);
        }
        ProcessingMode::MixedPhase => {
            optimizer.fir.get_or_insert_with(default_qa_fir_config);
            optimizer
                .mixed_phase
                .get_or_insert_with(default_qa_mixed_phase_config);
        }
        ProcessingMode::LowLatency | ProcessingMode::WarpedIir | ProcessingMode::KautzModal => {}
    }
}

fn default_qa_fir_config() -> FirConfig {
    FirConfig {
        taps: 2048,
        phase: "kirkeby".to_string(),
        correct_excess_phase: false,
        phase_smoothing: 0.167,
        pre_ringing: None,
        max_boost_db: None,
    }
}

fn default_qa_mixed_phase_config() -> MixedPhaseSerdeConfig {
    MixedPhaseSerdeConfig {
        max_fir_length_ms: 10.0,
        pre_ringing_threshold_db: -30.0,
        min_spatial_depth: 0.5,
        phase_smoothing_octaves: 0.167,
    }
}

pub(super) fn build_multisub_config(sub_curves: &[Curve], allpass: bool) -> RoomConfig {
    let mut speakers = HashMap::new();
    let subwoofers: Vec<MeasurementSource> = sub_curves
        .iter()
        .map(|c| MeasurementSource::InMemory(c.clone()))
        .collect();

    speakers.insert(
        "LFE".to_string(),
        SpeakerConfig::MultiSub(MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers,
            allpass_optimization: allpass,
        }),
    );

    let mut config = RoomConfig {
        version: default_config_version(),
        system: None,
        speakers,
        crossovers: None,
        target_curve: None,
        optimizer: Default::default(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    config.optimizer.algorithm = "autoeq:cmaes".to_string();
    config.optimizer.max_iter = QA_MAXEVAL;
    config.optimizer.population = 50;
    config.optimizer.refine = false;
    config.optimizer.seed = Some(SEED);
    config.optimizer.processing_mode = ProcessingMode::LowLatency;
    config.optimizer.num_filters = 3;
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 200.0;

    config
}

/// Build a RoomConfig for a multi-channel layout with a given sub topology.
///
/// Creates synthetic per-channel curves using the difficulty's room modes with
/// per-channel noise variation (different seed per channel).
pub(super) fn build_multichannel_config(
    layout: &ChannelLayout,
    sub_topo: Option<&SubTopology>,
    difficulty: &DifficultyLevel,
    base_curve: &Curve,
    processing_mode: ProcessingMode,
    sample_rate: f64,
) -> RoomConfig {
    let mut speakers = HashMap::new();
    let mut sys_speakers = HashMap::new();

    let modes_biquad: Vec<Biquad> = difficulty
        .modes
        .iter()
        .copied()
        .chain(
            (processing_mode == ProcessingMode::KautzModal)
                .then_some(KAUTZ_REFERENCE_MODES)
                .into_iter()
                .flatten()
                .copied(),
        )
        .map(|(freq, q, gain)| Biquad::new(BiquadFilterType::Peak, freq, sample_rate, q, gain))
        .collect();

    // Generate per-main-channel curves
    for (i, &role) in layout.mains.iter().enumerate() {
        let key = role.to_lowercase();
        let delay = i as f64 * 0.3; // slight per-channel delay variation
        let curve = generate_channel_curve(
            base_curve,
            &modes_biquad,
            delay,
            difficulty.noise_rms * 0.5,
            SEED.wrapping_add(i as u64 * 100),
            sample_rate,
        );
        speakers.insert(
            key.clone(),
            SpeakerConfig::Single(MeasurementSource::InMemory(curve)),
        );
        sys_speakers.insert(role.to_string(), key);
    }

    // Height channels (same treatment as mains)
    for (i, &role) in layout.heights.iter().enumerate() {
        let key = role.to_lowercase();
        let delay = (layout.mains.len() + i) as f64 * 0.3;
        let curve = generate_channel_curve(
            base_curve,
            &modes_biquad,
            delay,
            difficulty.noise_rms * 0.5,
            SEED.wrapping_add((layout.mains.len() + i) as u64 * 100),
            sample_rate,
        );
        speakers.insert(
            key.clone(),
            SpeakerConfig::Single(MeasurementSource::InMemory(curve)),
        );
        sys_speakers.insert(role.to_string(), key);
    }

    // LFE / sub topology
    let mut sub_config = if layout.has_lfe {
        let sub_topo = sub_topo.expect("layout has LFE but no sub topology");
        let bass_modes: Vec<Biquad> = difficulty
            .modes
            .iter()
            .filter(|(f, _, _)| *f < 200.0)
            .map(|&(freq, q, gain)| Biquad::new(BiquadFilterType::Peak, freq, sample_rate, q, gain))
            .collect();

        match sub_topo.name {
            "single_sub" => {
                let sub_curve = generate_channel_curve(
                    &generate_subwoofer_rolloff_curve(20.0, 200.0, 100, 80.0, -6.0),
                    &bass_modes,
                    0.0,
                    difficulty.noise_rms * 0.3,
                    SEED.wrapping_add(9000),
                    sample_rate,
                );
                speakers.insert(
                    "lfe".to_string(),
                    SpeakerConfig::Single(MeasurementSource::InMemory(sub_curve)),
                );
                sys_speakers.insert("LFE".to_string(), "lfe".to_string());
                Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Single,
                    crossover: None,
                    mapping: HashMap::new(),
                })
            }
            "mso_2sub" | "mso_2sub_allpass" => {
                let ms = generate_multisub_scenario(
                    "lfe",
                    2,
                    &bass_modes,
                    &[],
                    &[0.0, 2.0],
                    difficulty.noise_rms * 0.3,
                    SEED.wrapping_add(9000),
                    sample_rate,
                );
                let allpass = sub_topo.name == "mso_2sub_allpass";
                let subwoofers: Vec<MeasurementSource> = ms
                    .sub_curves
                    .into_iter()
                    .map(MeasurementSource::InMemory)
                    .collect();
                speakers.insert(
                    "lfe".to_string(),
                    SpeakerConfig::MultiSub(MultiSubGroup {
                        name: "subs".to_string(),
                        speaker_name: None,
                        subwoofers,
                        allpass_optimization: allpass,
                    }),
                );
                sys_speakers.insert("LFE".to_string(), "lfe".to_string());
                Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Mso,
                    crossover: None,
                    mapping: HashMap::new(),
                })
            }
            "mso_4sub" | "mso_8sub" => {
                let sub_count = if sub_topo.name == "mso_8sub" { 8 } else { 4 };
                let delays: Vec<f64> = (0..sub_count).map(|index| index as f64 * 2.0).collect();
                let ms = generate_multisub_scenario(
                    "lfe",
                    sub_count,
                    &bass_modes,
                    &[],
                    &delays,
                    difficulty.noise_rms * 0.3,
                    SEED.wrapping_add(9000),
                    sample_rate,
                );
                speakers.insert(
                    "lfe".to_string(),
                    SpeakerConfig::MultiSub(MultiSubGroup {
                        name: "subs".to_string(),
                        speaker_name: None,
                        subwoofers: ms
                            .sub_curves
                            .into_iter()
                            .map(MeasurementSource::InMemory)
                            .collect(),
                        allpass_optimization: false,
                    }),
                );
                sys_speakers.insert("LFE".to_string(), "lfe".to_string());
                Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Mso,
                    crossover: None,
                    mapping: HashMap::new(),
                })
            }
            "cardioid" => {
                let card = generate_cardioid_scenario(
                    "lfe",
                    &bass_modes,
                    1.0,
                    difficulty.noise_rms * 0.3,
                    SEED.wrapping_add(9000),
                    sample_rate,
                );
                speakers.insert(
                    "lfe".to_string(),
                    SpeakerConfig::Cardioid(Box::new(CardioidConfig {
                        name: "cardioid_sub".to_string(),
                        speaker_name: None,
                        front: MeasurementSource::InMemory(card.front_curve),
                        rear: MeasurementSource::InMemory(card.rear_curve),
                        separation_meters: card.separation_meters,
                    })),
                );
                sys_speakers.insert("LFE".to_string(), "lfe".to_string());
                Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Single, // cardioid routes via SpeakerConfig dispatch
                    crossover: None,
                    mapping: HashMap::new(),
                })
            }
            "dba" => {
                let dba = generate_dba_scenario(
                    "lfe",
                    1,
                    1,
                    &bass_modes,
                    8.0,
                    difficulty.noise_rms * 0.3,
                    SEED.wrapping_add(9000),
                    sample_rate,
                );
                let front: Vec<MeasurementSource> = dba
                    .front_curves
                    .into_iter()
                    .map(MeasurementSource::InMemory)
                    .collect();
                let rear: Vec<MeasurementSource> = dba
                    .rear_curves
                    .into_iter()
                    .map(MeasurementSource::InMemory)
                    .collect();
                speakers.insert(
                    "lfe".to_string(),
                    SpeakerConfig::Dba(DBAConfig {
                        name: "dba_sub".to_string(),
                        speaker_name: None,
                        front,
                        rear,
                    }),
                );
                sys_speakers.insert("LFE".to_string(), "lfe".to_string());
                Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Dba,
                    crossover: None,
                    mapping: HashMap::new(),
                })
            }
            _ => panic!("Unknown sub topology: {}", sub_topo.name),
        }
    } else {
        None
    };

    // Add crossover config if sub is present (required by 2.1 and home cinema workflows)
    let mut crossovers_map = None;
    if let Some(ref mut sc) = sub_config {
        sc.crossover = Some("lfe_xover".to_string());
        let mut xovers = HashMap::new();
        xovers.insert(
            "lfe_xover".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: Some(80.0),
                frequencies: None,
                frequency_range: None,
            },
        );
        crossovers_map = Some(xovers);
    }

    let system = SystemConfig {
        model: layout.system_model(),
        speakers: sys_speakers,
        subwoofers: sub_config,
        bass_management: None,
        supporting_source_outputs: None,
    };

    let mut config = RoomConfig {
        version: default_config_version(),
        system: Some(system),
        speakers,
        crossovers: crossovers_map,
        target_curve: None,
        optimizer: Default::default(),
        provenance: Default::default(),
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
    };

    config.optimizer.algorithm = "autoeq:cmaes".to_string();
    config.optimizer.max_iter = QA_MAXEVAL;
    config.optimizer.population = 50;
    config.optimizer.refine = false;
    config.optimizer.seed = Some(SEED);
    config.optimizer.processing_mode = ProcessingMode::LowLatency;
    config.optimizer.num_filters = 3;
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 20000.0;

    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::consts::{EASY, LAYOUT_2_0, LAYOUT_7_1_6, SUB_MSO_8};
    use roomeq_synthetic::generate_flat_curve;

    #[test]
    fn processing_modes_install_required_qa_configuration() {
        for mode in [ProcessingMode::PhaseLinear, ProcessingMode::Hybrid] {
            let mut optimizer = OptimizerConfig::default();
            configure_processing_mode(&mut optimizer, mode.clone());
            assert_eq!(optimizer.processing_mode, mode);
            assert!(optimizer.fir.is_some());
            assert_eq!(optimizer.max_freq, 1500.0);
        }

        let mut optimizer = OptimizerConfig::default();
        configure_processing_mode(&mut optimizer, ProcessingMode::MixedPhase);
        assert!(optimizer.fir.is_some());
        assert!(optimizer.mixed_phase.is_some());
    }

    #[test]
    fn extended_cinema_layout_builds_eight_physical_subs() {
        let base = generate_flat_curve(20.0, 20_000.0, 100);
        let config = build_multichannel_config(
            &LAYOUT_7_1_6,
            Some(&SUB_MSO_8),
            &EASY,
            &base,
            ProcessingMode::LowLatency,
            48_000.0,
        );

        assert_eq!(config.system.as_ref().unwrap().speakers.len(), 14);
        match config.speakers.get("lfe").unwrap() {
            SpeakerConfig::MultiSub(group) => assert_eq!(group.subwoofers.len(), 8),
            other => panic!("expected eight-sub MSO group, got {other:?}"),
        }
    }

    #[test]
    fn easy_fixture_contains_a_detectable_kautz_mode() {
        let base = generate_flat_curve(20.0, 20_000.0, 200);
        let config = build_multichannel_config(
            &LAYOUT_2_0,
            None,
            &EASY,
            &base,
            ProcessingMode::KautzModal,
            48_000.0,
        );
        let SpeakerConfig::Single(MeasurementSource::InMemory(curve)) =
            config.speakers.get("l").unwrap()
        else {
            panic!("expected an in-memory left-channel fixture");
        };
        let modes = roomeq_analysis::impulse_analysis::detect_room_modes(
            &curve.freq,
            &curve.spl,
            &roomeq_analysis::impulse_analysis::DecomposedCorrectionConfig::default(),
        );
        assert!(
            modes.iter().any(|mode| (mode.frequency - 28.0).abs() < 5.0),
            "expected the easy fixture's 28 Hz reference resonance to be detected: {modes:?}"
        );
    }
}
