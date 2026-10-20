//! Joint-array dispatch, primary seat, and serialized home-cinema routing compose.

// Rust guideline compliant 2026-02-21
#[path = "../benches/fixtures/subwoofers.rs"]
mod fixtures;

#[test]
fn verified_multiseat_joint_home_cinema_retains_assessed_array_and_routes() {
    let config = fixtures::correctable_home_cinema(48_000.0);
    let result = roomeq_workflow::optimize_room(&config, 48_000.0, None, None).unwrap();
    assert_delivered(&result, 48_000.0);
}

#[test]
fn joint_cinema_replays_unequal_arrivals_at_the_nonzero_primary_seat() {
    let fs = 48_000.0;
    let arrivals_ms = [0.0, 1.0];
    let mut value = serde_json::to_value(fixtures::correctable_home_cinema(fs)).unwrap();
    for (sub, arrival) in value["speakers"]["array"]["subwoofers"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .zip(arrivals_ms)
    {
        for measurement in sub["measurements"].as_array_mut().unwrap() {
            let frequencies = measurement["frequencies"].as_array().unwrap().clone();
            for (phase, frequency) in measurement["phase_deg"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .zip(frequencies)
            {
                *phase = serde_json::json!(
                    phase.as_f64().unwrap()
                        - 360.0 * frequency.as_f64().unwrap() * arrival / 1000.0
                );
            }
        }
    }
    let config = serde_json::from_value(value).unwrap();
    let result = roomeq_workflow::optimize_room(&config, fs, None, None).unwrap();
    // The complete delivered transfer must match an independent physical sum.
    // Exact alignment of flat arrays is covered by the engine's permutation
    // oracle; a resonant room instead trades target error and output efficiency.
    assert_delivered_with_arrivals(&result, fs, arrivals_ms, 200);
}

#[test]
fn long_fir_joint_cinema_delivers_correction_on_mains_and_physical_subs() {
    for (mode, taps, fs) in [
        (roomeq_model::ProcessingMode::Hybrid, 8192, 48_000.0),
        (roomeq_model::ProcessingMode::PhaseLinear, 16384, 96_000.0),
    ] {
        let mut config = fixtures::correctable_home_cinema(fs);
        config.optimizer.fir = Some(roomeq_model::FirConfig {
            taps,
            phase: "linear".into(),
            ..Default::default()
        });
        if mode == roomeq_model::ProcessingMode::Hybrid {
            config.optimizer.mixed_config = Some(roomeq_model::MixedModeConfig {
                crossover_freq: 300.0,
                fir_band: "high".into(),
                ..Default::default()
            });
        }
        config.optimizer.processing_mode = mode;
        let output = tempfile::tempdir().unwrap();
        let result =
            roomeq_workflow::optimize_room(&config, fs, None, Some(output.path())).unwrap();
        assert_delivered(&result, fs);
        if config.optimizer.processing_mode == roomeq_model::ProcessingMode::Hybrid {
            let selection = result
                .metadata
                .stage_outcomes
                .iter()
                .find(|stage| stage.stage == "hybrid_best_part_selection")
                .unwrap();
            assert_eq!(selection.status, roomeq_model::StageStatus::Applied);
            assert!(selection.advisories.contains(&"selected=hybrid".into()));
            assert_eq!(selection.checks.len(), 2);
            assert!(selection.checks.iter().all(|check| check.passed));
            for name in ["L", "R"] {
                assert!(
                    result.channels[name]
                        .plugins
                        .iter()
                        .any(|plugin| plugin.plugin_type == "band_split")
                );
                assert!(
                    result.channels[name]
                        .plugins
                        .iter()
                        .any(|plugin| plugin.plugin_type == "band_merge")
                );
            }
        }
        assert!(
            result
                .metadata
                .stage_outcomes
                .iter()
                .any(|stage| stage.stage == "routed_pre_eq_fir_Sub1"
                    && stage.status == roomeq_model::StageStatus::Applied)
        );
        for name in ["L", "R", "Sub1"] {
            let convolution = result.channels[name]
                .plugins
                .iter()
                .find(|plugin| plugin.plugin_type == "convolution")
                .unwrap_or_else(|| panic!("{name} must retain its FIR"));
            let path = convolution.parameters["ir_file"].as_str().unwrap();
            let reader = hound::WavReader::open(output.path().join(path)).unwrap();
            assert_eq!(reader.duration() as usize, taps);
            assert!(
                convolution.parameters["correction_design_delay_ms"]
                    .as_f64()
                    .is_some()
            );
        }
    }
}

fn assert_delivered(result: &roomeq_workflow::RoomOptimizationResult, sample_rate: f64) {
    assert_delivered_with_arrivals(result, sample_rate, [0.0, 0.0], 200);
}

fn assert_delivered_with_arrivals(
    result: &roomeq_workflow::RoomOptimizationResult,
    sample_rate: f64,
    arrivals_ms: [f64; 2],
    evaluation_limit: usize,
) {
    let chain = result
        .channels
        .values()
        .find(|chain| chain.joint_sub.is_some())
        .expect("joint stage must survive finalization");
    let joint = chain.joint_sub.as_ref().unwrap();
    assert_eq!(joint.seats.len(), 2);
    assert!(joint.search_evaluations > 0 && joint.search_evaluations <= evaluation_limit);
    assert!(joint.array_rejection_reason.is_none());
    for seat in &joint.seats {
        // Independent control transfer: two identical measured subs share the
        // resonance, but each emitted physical control acts on its own branch.
        for (bin, &frequency) in seat.before.freq.iter().enumerate() {
            let (mut re, mut im, mut pre_re, mut pre_im) = (0.0, 0.0, 0.0, 0.0);
            for ((&gain, &delay), arrival) in joint
                .array_gains_db
                .iter()
                .zip(&joint.array_delays_ms)
                .zip(arrivals_ms)
            {
                let raw_angle = -2.0 * std::f64::consts::PI * frequency * arrival / 1000.0;
                pre_re += raw_angle.cos();
                pre_im += raw_angle.sin();
                let angle = raw_angle - 2.0 * std::f64::consts::PI * frequency * delay / 1000.0;
                let magnitude = 10.0_f64.powf(gain / 20.0);
                re += magnitude * angle.cos();
                im += magnitude * angle.sin();
            }
            let expected_level =
                seat.before.spl[bin] + 20.0 * (re.hypot(im) / pre_re.hypot(pre_im)).log10();
            let expected_phase = seat.before.phase.as_ref().unwrap()[bin]
                + im.atan2(re).to_degrees()
                - pre_im.atan2(pre_re).to_degrees();
            assert!((seat.after_array.spl[bin] - expected_level).abs() < 1e-7);
            let phase_error = (seat.after_array.phase.as_ref().unwrap()[bin] - expected_phase
                + 180.0)
                .rem_euclid(360.0)
                - 180.0;
            assert!(phase_error.abs() < 1e-7);
        }
    }
    assert!(
        joint.seats[0]
            .before
            .spl
            .iter()
            .zip(&joint.seats[1].before.spl)
            .all(|(first, second)| (second - first - 1.0).abs() < 1e-7)
    );
    let bass = result.metadata.bass_management.as_ref().unwrap();
    assert_eq!(bass.physical_sub_outputs, ["Sub1", "Sub2"]);
    let acceptance = result
        .metadata
        .stage_outcomes
        .iter()
        .find(|stage| stage.stage == "final_correlated_bass_acceptance")
        .unwrap_or_else(|| {
            panic!(
                "correlated bass must be assessed; stages={:?}; acceptance={:?}",
                result
                    .metadata
                    .stage_outcomes
                    .iter()
                    .map(|stage| (&stage.stage, &stage.status, &stage.advisories))
                    .collect::<Vec<_>>(),
                result
                    .metadata
                    .correction_acceptance
                    .as_ref()
                    .map(|report| (&report.decision, &report.violations))
            )
        });
    assert!(!acceptance.checks.is_empty());
    assert!(acceptance.checks.iter().all(|check| check.passed));
    let decision = result.metadata.correction_acceptance.as_ref().unwrap();
    assert!(
        decision.accepted && decision.decision == roomeq_model::CorrectionDecision::Accepted,
        "known-solvable fixture at {sample_rate} Hz must be accepted: {:?} {:?}; selection={:?}",
        decision.decision,
        decision.violations,
        result
            .metadata
            .stage_outcomes
            .iter()
            .filter(|stage| stage.stage == "final_correction_selection")
            .flat_map(|stage| stage.checks.iter().take(2).map(|check| &check.diagnostic))
            .collect::<Vec<_>>()
    );
    let seats = &decision.acoustic_quality.as_ref().unwrap().final_seats;
    assert!(
        seats
            .iter()
            .any(|seat| seat.partition == "training" && seat.logical_input == "LFE")
    );
    assert!(
        seats
            .iter()
            .filter(|seat| seat.partition == "training")
            .all(|seat| seat.improvement_lower_bound_db > 0.0)
    );
    let graph = result.to_dsp_chain_output();
    let roundtrip: roomeq_model::DspGraph =
        serde_json::from_slice(&serde_json::to_vec(&graph).unwrap()).unwrap();
    roundtrip.validate().unwrap();
    let exported = roomeq_export::render_dsp_graph(
        &roundtrip,
        roomeq_export::ExportFormat::CamillaDsp,
        sample_rate,
    );
    if roundtrip.channels.values().any(|chain| {
        chain
            .plugins
            .iter()
            .any(|plugin| plugin.plugin_type == "band_split")
    }) {
        let error = exported
            .expect_err("CamillaDSP explicitly rejects split-band graphs")
            .to_string();
        assert!(error.contains("does not support") && error.contains("band_split"));
    } else {
        let yaml = exported.unwrap();
        assert!(yaml.contains("_to_Sub1") && yaml.contains("_to_Sub2"));
    }
    let serialized = serde_json::to_value(&roundtrip).unwrap();
    assert_eq!(
        serialized["metadata"]["bass_management"]["routing_graph"]["physical_sub_outputs"],
        serde_json::json!(["Sub1", "Sub2"])
    );
    // Keep the engine-only scaling fixture checked by this target as well.
    assert_eq!(fixtures::joint_request(4, 3, 128).measurements.len(), 4);
}
