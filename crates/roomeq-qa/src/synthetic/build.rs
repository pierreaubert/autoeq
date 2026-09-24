use super::channel_layout::ChannelLayout;
use super::consts::KAUTZ_REFERENCE_MODES;
use super::consts::QA_MAXEVAL;
use super::consts::SEED;
use super::types::DifficultyLevel;
use super::types::SubTopology;
use autoeq_core::{MeasurementProvenance, ProvenanceCaptureKind};
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use roomeq_model::{
    CardioidConfig, CrossoverConfig, Curve, DBAConfig, FirConfig, InlineMeasurement,
    MeasurementRef, MeasurementSingle, MeasurementSource, MixedPhaseSerdeConfig, MultiSubGroup,
    OptimizerConfig, ProcessingMode, RoomConfig, SpeakerConfig, SubwooferStrategy,
    SubwooferSystemConfig, SystemConfig, default_config_version,
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
        reporting: None,
        cea2034_cache: None,
    };

    config.optimizer.algorithm = "autoeq:cmaes".to_string();
    config.optimizer.max_iter = QA_MAXEVAL;
    config.optimizer.population = 20;
    config.optimizer.refine = false;
    config.optimizer.seed = Some(SEED);
    config.optimizer.num_filters = 3;
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 20000.0;
    configure_processing_mode(&mut config.optimizer, mode);

    config
}

/// Materialize the measurement/phase/FIR axes rather than merely recording
/// requested labels. Topology and crossover mapping are handled separately.
pub(super) fn apply_parameter_signal_axes(
    config: &mut RoomConfig,
    row: &crate::parameter_matrix::ParameterRow,
    sample_rate: f64,
) {
    let requested_points = [100, 200, 400][row.grid_size as usize];
    let mut names: Vec<_> = config.speakers.keys().cloned().collect();
    names.sort();
    let sub_name = config
        .system
        .as_ref()
        .and_then(|system| system.subwoofers.as_ref())
        .and_then(|subwoofers| subwoofers.outputs.first())
        .map(|output| &output.speaker)
        .cloned();
    for (index, name) in names.iter().enumerate() {
        let (minimum, maximum, points) = match row.measurement_shape {
            0 => (20.0, sample_rate / 2.0 - 100.0, requested_points),
            1 => (
                20.0,
                sample_rate / 2.0 - 100.0,
                requested_points + index * 17,
            ),
            2 => (
                100.0,
                6000.0_f64.min(sample_rate / 2.0 - 100.0),
                requested_points / 4,
            ),
            _ => unreachable!("validated ternary matrix axis"),
        };
        let mut curve = roomeq_synthetic::generate_flat_curve(minimum, maximum, points);
        // A flat input lets runtime safety discard identity-like FIRs, so it
        // cannot witness FIR-duration execution. Use the same smooth, bounded
        // 4 dB correction opportunity on every independently sampled grid.
        for (frequency, level) in curve.freq.iter().zip(curve.spl.iter_mut()) {
            *level +=
                4.0 * (-0.5 * (frequency.ln() - 300.0_f64.ln()).powi(2) / 0.7_f64.powi(2)).exp();
        }
        if sub_name.as_ref() == Some(name) {
            for (frequency, level) in curve.freq.iter().zip(curve.spl.iter_mut()) {
                *level -= 12.0 * (frequency / 200.0).max(1.0).log2();
            }
        }
        if row.phase == 1 {
            curve.phase = Some(curve.freq.mapv(|frequency| -360.0 * frequency * 0.001));
        } else {
            curve.phase = None;
        }
        let source = if row.phase == 1 {
            // These analytic responses model stationary, same-clock IRs. Declare
            // that synthetic timing reference so crossover admission tests the
            // optimizer instead of stopping at unknown acquisition provenance.
            serde_json::from_value(serde_json::json!({
                "inline": {
                    "frequencies": curve.freq.to_vec(),
                    "magnitude_db": curve.spl.to_vec(),
                    "phase_deg": curve.phase.as_ref().unwrap().to_vec(),
                },
                "provenance": {
                    "capture_kind": "stationary_ir",
                    "timing_reference_id": "qa-analytic-common-clock",
                },
            }))
            .expect("valid synthetic stationary measurement")
        } else {
            MeasurementSource::InMemory(curve)
        };
        config
            .speakers
            .insert(name.clone(), SpeakerConfig::Single(source));
    }
    if let Some(fir) = &mut config.optimizer.fir {
        let duration_ms = [5.0, 10.0, 20.0][row.fir_duration as usize];
        fir.taps = (duration_ms * sample_rate / 1000.0).round() as usize;
    }
}

pub(super) fn parameter_fixture_curve(speaker: &SpeakerConfig) -> Option<Curve> {
    let SpeakerConfig::Single(source) = speaker else {
        return None;
    };
    match source {
        MeasurementSource::InMemory(curve) => Some(curve.clone()),
        MeasurementSource::Single(single) => match &single.measurement {
            MeasurementRef::Inline(inline) => Some(Curve {
                freq: inline.frequencies.clone().into(),
                spl: inline.magnitude_db.clone().into(),
                phase: inline.phase_deg.clone().map(Into::into),
                ..Curve::default()
            }),
            _ => None,
        },
        _ => None,
    }
}

/// Build the requested physical topology before applying independently sampled measurements.
pub(super) fn build_parameter_config(
    curve: &Curve,
    row: &crate::parameter_matrix::ParameterRow,
    mode: ProcessingMode,
    sample_rate: f64,
) -> RoomConfig {
    use super::consts::{EASY, LAYOUT_2_1, LAYOUT_5_1, SUB_SINGLE};
    let mut config = match row.topology {
        0 => build_config(curve, mode.clone()),
        1 | 2 => build_multichannel_config(
            if row.topology == 1 {
                &LAYOUT_2_1
            } else {
                &LAYOUT_5_1
            },
            Some(&SUB_SINGLE),
            &EASY,
            curve,
            mode.clone(),
            sample_rate,
        ),
        _ => unreachable!("validated ternary matrix axis"),
    };
    if let Some(crossovers) = &mut config.crossovers {
        for crossover in crossovers.values_mut() {
            // Three distinct policies: fixed LR24, automatic LR24, fixed LR48.
            crossover.crossover_type = if row.crossover == 2 { "LR48" } else { "LR24" }.into();
            // Keep the search inside even the sparse fixture's measured support.
            crossover.frequency = if row.crossover == 1 {
                None
            } else {
                Some(160.0)
            };
            crossover.frequency_range = if row.crossover == 1 {
                Some((120.0, 220.0))
            } else {
                None
            };
        }
    }
    configure_processing_mode(&mut config.optimizer, mode);
    apply_parameter_signal_axes(&mut config, row, sample_rate);
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
        placement: Default::default(),
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
        assessment: Default::default(),
        max_correction_latency_ms: None,
    }
}

pub(super) fn build_multisub_config(sub_curves: &[Curve], allpass: bool) -> RoomConfig {
    let mut speakers = HashMap::new();
    let subwoofers: Vec<MeasurementSource> = sub_curves
        .iter()
        .enumerate()
        .map(|(index, curve)| {
            if !allpass {
                return MeasurementSource::InMemory(curve.clone());
            }
            let phase = curve
                .phase
                .as_ref()
                .expect("analytic all-pass fixture needs phase");
            assert_eq!(phase.len(), curve.freq.len());
            // The analytic generator uses one time origin for every sub. Declare
            // that synthetic reference explicitly so the production provenance
            // gate can exercise the all-pass path without accepting unknown data.
            MeasurementSource::Single(MeasurementSingle {
                measurement: MeasurementRef::Inline(InlineMeasurement {
                    frequencies: curve.freq.to_vec(),
                    magnitude_db: curve.spl.to_vec(),
                    phase_deg: Some(phase.to_vec()),
                    name: Some("synthetic-seat-0".to_string()),
                    wav_path: None,
                    csv_path: None,
                }),
                speaker_name: Some(format!("synthetic-sub-{}", index + 1)),
                provenance: MeasurementProvenance {
                    capture_kind: ProvenanceCaptureKind::StationaryIr,
                    timing_reference_id: Some("synthetic-shared-clock".to_string()),
                    ..Default::default()
                },
            })
        })
        .collect();

    speakers.insert(
        "LFE".to_string(),
        SpeakerConfig::MultiSub(MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers,
            allpass_optimization: allpass,
            joint_optimization: false,
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
        reporting: None,
        cea2034_cache: None,
    };

    config.optimizer.algorithm = "autoeq:cmaes".to_string();
    config.optimizer.max_iter = QA_MAXEVAL;
    config.optimizer.population = 20;
    config.optimizer.refine = false;
    config.optimizer.seed = Some(SEED);
    config.optimizer.processing_mode = ProcessingMode::LowLatency;
    config.optimizer.num_filters = 3;
    config.optimizer.finalization.default_input_peak = super::consts::POSITIVE_ROUTED_INPUT_PEAK;
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
                    routing: Default::default(),
                    outputs: Vec::new(),
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
                        joint_optimization: false,
                    }),
                );
                sys_speakers.insert("LFE".to_string(), "lfe".to_string());
                Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Mso,
                    crossover: None,
                    routing: Default::default(),
                    outputs: Vec::new(),
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
                        joint_optimization: false,
                    }),
                );
                sys_speakers.insert("LFE".to_string(), "lfe".to_string());
                Some(SubwooferSystemConfig {
                    config: SubwooferStrategy::Mso,
                    crossover: None,
                    routing: Default::default(),
                    outputs: Vec::new(),
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
                    routing: Default::default(),
                    outputs: Vec::new(),
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
                    routing: Default::default(),
                    outputs: Vec::new(),
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
        // Logical LFE is implicit in v3. Explicit outputs identify each
        // measured physical branch, including branches within a sub group.
        sys_speakers.remove("LFE");
        let output_count = match &speakers["lfe"] {
            SpeakerConfig::Single(_) => 1,
            SpeakerConfig::MultiSub(group) => group.subwoofers.len(),
            SpeakerConfig::Cardioid(_) => 2,
            SpeakerConfig::Dba(group) => group.front.len() + group.rear.len(),
            _ => unreachable!("synthetic sub topology must expose physical branches"),
        };
        sc.outputs = (0..output_count)
            .map(|index| roomeq_model::SubwooferOutput {
                id: format!("Sub{}", index + 1),
                speaker: String::from("lfe"),
            })
            .collect();
        sc.crossover = Some(roomeq_model::SubwooferCrossoverRef::PerSub(vec![
            String::from("lfe_xover"); output_count
        ]));
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
        // Stereo routing supports at most two physical subs. Larger arrays
        // use the home-cinema routing contract even with only two mains.
        model: if sub_config
            .as_ref()
            .is_some_and(|subs| subs.outputs.len() > 2)
        {
            roomeq_model::SystemModel::HomeCinema
        } else {
            layout.system_model()
        },
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
        reporting: None,
        cea2034_cache: None,
    };

    config.optimizer.algorithm = "autoeq:cmaes".to_string();
    config.optimizer.max_iter = QA_MAXEVAL;
    config.optimizer.population = 20;
    config.optimizer.refine = false;
    config.optimizer.seed = Some(SEED);
    config.optimizer.processing_mode = ProcessingMode::LowLatency;
    config.optimizer.num_filters = 3;
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 20000.0;
    if sub_topo.is_some() {
        config.optimizer.finalization.default_input_peak =
            super::consts::POSITIVE_ROUTED_INPUT_PEAK;
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_multichannel_topologies_use_valid_v3_outputs() {
        use crate::synthetic::consts::{ALL_LAYOUTS, ALL_SUB_TOPOS};
        let base = roomeq_synthetic::generate_flat_curve(20.0, 20_000.0, 100);
        for layout in ALL_LAYOUTS.iter().filter(|layout| layout.has_lfe) {
            for topology in ALL_SUB_TOPOS {
                let config = build_multichannel_config(
                    layout,
                    Some(topology),
                    &EASY,
                    &base,
                    ProcessingMode::LowLatency,
                    48_000.0,
                );
                config.validate_structure().unwrap();
                assert_eq!(
                    config.optimizer.finalization.default_input_peak,
                    crate::synthetic::consts::POSITIVE_ROUTED_INPUT_PEAK
                );
                let system = config.system.as_ref().unwrap();
                assert!(!system.speakers.contains_key("LFE"));
                let subs = system.subwoofers.as_ref().unwrap();
                let Some(roomeq_model::SubwooferCrossoverRef::PerSub(crossovers)) = &subs.crossover
                else {
                    panic!("v3 requires per-output crossovers");
                };
                assert_eq!(crossovers.len(), subs.outputs.len());
                assert!(!subs.outputs.is_empty());
            }
        }
        let stereo = ALL_LAYOUTS
            .iter()
            .find(|layout| layout.name == "2.0")
            .unwrap();
        let no_routing = build_multichannel_config(
            stereo,
            None,
            &EASY,
            &base,
            ProcessingMode::LowLatency,
            48_000.0,
        );
        assert_eq!(no_routing.optimizer.finalization.default_input_peak, 1.0);
    }

    #[test]
    #[ignore = "full DBA optimization and branch replay diagnostic"]
    fn synthetic_dba_output_branch_mapping() {
        let layout = super::super::consts::ALL_LAYOUTS
            .iter()
            .find(|layout| layout.name == "2.1")
            .unwrap();
        let base =
            roomeq_synthetic::generate_speaker_rolloff_curve(20.0, 20_000.0, 200, 80.0, -6.0);
        let config = build_multichannel_config(
            layout,
            Some(&super::super::consts::SUB_DBA),
            &EASY,
            &base,
            ProcessingMode::LowLatency,
            48_000.0,
        );
        let result = super::super::run::run_optimization(&config).unwrap();
        let outputs = &config
            .system
            .as_ref()
            .unwrap()
            .subwoofers
            .as_ref()
            .unwrap()
            .outputs;
        let drivers = result
            .channels
            .get("Sub1")
            .unwrap()
            .drivers
            .as_ref()
            .unwrap();
        assert_eq!(drivers.len(), outputs.len());
        for (index, (driver, output)) in drivers.iter().zip(outputs).enumerate() {
            assert_eq!(driver.index, index);
            assert_eq!(driver.name, output.id);
        }
    }

    #[test]
    fn parameter_signal_axes_materialize_distinct_grids_phase_and_fir_lengths() {
        for row in crate::parameter_matrix::generate_pr_matrix() {
            let rate = [44100.0, 48000.0, 96000.0][row.sample_rate as usize];
            let mode = [
                ProcessingMode::LowLatency,
                ProcessingMode::PhaseLinear,
                ProcessingMode::Hybrid,
            ][row.mode as usize]
                .clone();
            let curve = roomeq_synthetic::generate_flat_curve(20.0, 20000.0, 100);
            let mut config = build_config(&curve, mode);
            apply_parameter_signal_axes(&mut config, &row, rate);
            let curves: Vec<_> = ["Left", "Right"]
                .iter()
                .map(|name| {
                    parameter_fixture_curve(&config.speakers[*name])
                        .expect("single analytic measurement")
                })
                .collect();
            assert_eq!(curves[0].phase.is_some(), row.phase == 1);
            assert_eq!(curves[1].phase.is_some(), row.phase == 1);
            if row.measurement_shape == 1 {
                assert_ne!(curves[0].freq.len(), curves[1].freq.len());
            } else {
                assert_eq!(curves[0].freq, curves[1].freq);
            }
            if row.measurement_shape == 2 {
                assert!((curves[0].freq[0] - 100.0).abs() < 1e-9);
                assert!((curves[0].freq[curves[0].freq.len() - 1] - 6000.0).abs() < 1e-9);
                assert_eq!(
                    curves[0].freq.len(),
                    [100, 200, 400][row.grid_size as usize] / 4
                );
            }
            if row.mode == 0 {
                assert!(config.optimizer.fir.is_none());
            } else {
                let requested = [5.0, 10.0, 20.0][row.fir_duration as usize];
                let taps = config.optimizer.fir.unwrap().taps;
                assert!((taps as f64 * 1000.0 / rate - requested).abs() <= 500.0 / rate + 1e-9);
            }
        }
    }
    #[test]
    fn parameter_topology_axes_preserve_roles_and_crossover_policy() {
        for row in crate::parameter_matrix::generate_pr_matrix() {
            let rate = [44_100.0, 48_000.0, 96_000.0][row.sample_rate as usize];
            let base = generate_flat_curve(20.0, 20_000.0, 100);
            let mode = match row.mode {
                0 => ProcessingMode::LowLatency,
                1 => ProcessingMode::PhaseLinear,
                _ => ProcessingMode::Hybrid,
            };
            let config = build_parameter_config(&base, &row, mode.clone(), rate);
            assert_eq!(config.optimizer.processing_mode, mode);
            assert_eq!(config.optimizer.fir.is_some(), row.mode != 0);
            let expected = [2, 3, 6][row.topology as usize];
            assert_eq!(config.speakers.len(), expected);
            if row.topology == 0 {
                assert!(config.system.is_none());
                assert!(config.crossovers.is_none());
            } else {
                let system = config.system.as_ref().unwrap();
                assert_eq!(system.speakers.len(), expected - 1);
                assert!(!system.speakers.contains_key("LFE"));
                let subs = system.subwoofers.as_ref().unwrap();
                assert_eq!(subs.outputs.len(), 1);
                assert_eq!(subs.outputs[0].speaker, "lfe");
                assert!(
                    system
                        .speakers
                        .values()
                        .all(|name| config.speakers.contains_key(name))
                );
                for crossover in config.crossovers.as_ref().unwrap().values() {
                    assert_eq!(
                        crossover.crossover_type,
                        if row.crossover == 2 { "LR48" } else { "LR24" }
                    );
                    assert_eq!(crossover.frequency, (row.crossover != 1).then_some(160.0));
                    assert_eq!(
                        crossover.frequency_range,
                        (row.crossover == 1).then_some((120.0, 220.0))
                    );
                }
            }
            for speaker in config.speakers.values() {
                let curve = parameter_fixture_curve(speaker).expect("single analytic measurement");
                assert_eq!(curve.phase.is_some(), row.phase == 1);
                assert!(curve.spl.iter().all(|value| value.is_finite()));
            }
        }
    }

    use crate::synthetic::consts::{EASY, LAYOUT_2_0, LAYOUT_7_1_6, SUB_MSO_8};
    use roomeq_synthetic::{generate_flat_curve, generate_sub_curve_with_phase};

    #[test]
    fn allpass_fixture_declares_shared_synthetic_timing_and_seat() {
        let curves = [
            generate_sub_curve_with_phase(20.0, 200.0, 100, 1.0),
            generate_sub_curve_with_phase(20.0, 200.0, 100, 2.0),
        ];
        let config = build_multisub_config(&curves, true);
        let SpeakerConfig::MultiSub(group) = config.speakers.get("LFE").unwrap() else {
            panic!("expected a multi-sub group");
        };
        assert!(group.allpass_optimization);
        for source in &group.subwoofers {
            let MeasurementSource::Single(single) = source else {
                panic!("expected a provenance-bearing synthetic measurement");
            };
            assert_eq!(
                single.provenance.capture_kind,
                ProvenanceCaptureKind::StationaryIr
            );
            assert_eq!(
                single.provenance.timing_reference_id.as_deref(),
                Some("synthetic-shared-clock")
            );
            let MeasurementRef::Inline(inline) = &single.measurement else {
                panic!("expected inline synthetic samples");
            };
            assert_eq!(inline.name.as_deref(), Some("synthetic-seat-0"));
            assert_eq!(
                inline.phase_deg.as_ref().unwrap().len(),
                inline.frequencies.len()
            );
        }

        let config = build_multisub_config(&curves, false);
        assert_eq!(
            config.optimizer.finalization.default_input_peak,
            crate::synthetic::consts::POSITIVE_ROUTED_INPUT_PEAK
        );
        let SpeakerConfig::MultiSub(group) = config.speakers.get("LFE").unwrap() else {
            panic!("expected a multi-sub group");
        };
        assert!(
            group
                .subwoofers
                .iter()
                .all(|source| matches!(source, MeasurementSource::InMemory(_)))
        );
    }

    #[test]
    #[ignore = "runs the five-seed room optimizer"]
    fn allpass_fixture_passes_multisub_qa_case() {
        use crate::synthetic::consts::{MS_EASY, SAMPLE_RATE, SEED};
        use crate::synthetic::run::run_multisub_test;
        use crate::synthetic::types::MultiSubTopology;

        let shared = MS_EASY
            .shared_modes
            .iter()
            .map(|&(freq, q, gain)| Biquad::new(BiquadFilterType::Peak, freq, SAMPLE_RATE, q, gain))
            .collect::<Vec<_>>();
        let per_sub = MS_EASY
            .per_sub_modes
            .iter()
            .map(|modes| {
                modes
                    .iter()
                    .map(|&(freq, q, gain)| {
                        Biquad::new(BiquadFilterType::Peak, freq, SAMPLE_RATE, q, gain)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scenario = generate_multisub_scenario(
            "multisub/easy",
            MS_EASY.n_subs,
            &shared,
            &per_sub,
            MS_EASY.delays_ms,
            MS_EASY.noise_rms,
            SEED,
            SAMPLE_RATE,
        );
        let result = run_multisub_test(
            &scenario.sub_curves,
            &MultiSubTopology {
                name: "allpass",
                allpass: true,
            },
            &[],
            &MS_EASY,
        );
        assert!(result.passed, "{}: {}", result.name, result.reason);
    }

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

        let system = config.system.as_ref().unwrap();
        assert_eq!(system.speakers.len(), 13);
        assert_eq!(system.subwoofers.as_ref().unwrap().outputs.len(), 8);
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
