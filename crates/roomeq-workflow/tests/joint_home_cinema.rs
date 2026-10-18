//! Joint-array dispatch, primary seat, and serialized home-cinema routing compose.

// Rust guideline compliant 2026-02-21
#[path = "../benches/fixtures/subwoofers.rs"]
mod fixtures;

#[test]
fn verified_multiseat_joint_home_cinema_retains_assessed_array_and_routes() {
    // Coherent subs and mains have exactly invertible resonances. Every
    // training input, including LFE, must demonstrate delivered improvement.
    let mut value = serde_json::to_value(fixtures::home_cinema_joint(2, 2)).unwrap();
    for name in ["left", "right"] {
        for (seat, measurement) in value["speakers"][name]["measurements"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            let frequencies = ndarray::Array1::from_iter(
                measurement["frequencies"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|frequency| frequency.as_f64().unwrap()),
            );
            let peak = math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                700.0,
                48_000.0,
                0.7,
                12.0,
            );
            let response = autoeq_core::response::compute_peq_complex_response(
                &[peak],
                &frequencies,
                48_000.0,
            );
            measurement["magnitude_db"] = serde_json::json!(
                response
                    .iter()
                    .map(|value| 80.0
                        + 20.0 * 2.0_f64.log10()
                        + seat as f64
                        + 20.0 * value.norm().log10())
                    .collect::<Vec<_>>()
            );
            measurement["phase_deg"] = serde_json::json!(
                response
                    .iter()
                    .map(|value| seat as f64 * 45.0 + value.arg().to_degrees())
                    .collect::<Vec<_>>()
            );
        }
    }
    for sub in value["speakers"]["array"]["subwoofers"]
        .as_array_mut()
        .unwrap()
    {
        for measurement in sub["measurements"].as_array_mut().unwrap() {
            let frequencies = ndarray::Array1::from_iter(
                measurement["frequencies"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|frequency| frequency.as_f64().unwrap()),
            );
            let peak = math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                55.0,
                48_000.0,
                1.0,
                6.0,
            );
            let response = autoeq_core::response::compute_peq_complex_response(
                &[peak],
                &frequencies,
                48_000.0,
            );
            for (level, h) in measurement["magnitude_db"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .zip(response.iter())
            {
                *level = serde_json::json!(level.as_f64().unwrap() + 20.0 * h.norm().log10());
            }
            for (phase, h) in measurement["phase_deg"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .zip(response.iter())
            {
                *phase = serde_json::json!(phase.as_f64().unwrap() + h.arg().to_degrees());
            }
        }
    }
    value["optimizer"]["num_filters"] = serde_json::json!(2);
    value["optimizer"]["max_iter"] = serde_json::json!(1000);
    value["optimizer"]["max_freq"] = serde_json::json!(4000.0);
    value["optimizer"]["population"] = serde_json::json!(20);
    value["optimizer"]["bass_management_evaluation_limit"] = serde_json::json!(300);
    value["system"]["bass_management"] = serde_json::json!({"max_sub_boost_db":0.0});
    let config: roomeq_model::RoomConfig = serde_json::from_value(value).unwrap();
    let result = roomeq_workflow::optimize_room(&config, 48_000.0, None, None).unwrap();
    let chain = result
        .channels
        .values()
        .find(|chain| chain.joint_sub.is_some())
        .expect("joint stage must survive finalization");
    let joint = chain.joint_sub.as_ref().unwrap();
    assert_eq!(joint.seats.len(), 2);
    assert!(joint.search_evaluations > 0 && joint.search_evaluations <= 200);
    assert!(joint.array_rejection_reason.is_none());
    for seat in &joint.seats {
        // Independent control transfer: two identical measured subs share the
        // resonance, but each emitted physical control acts on its own branch.
        for (bin, &frequency) in seat.before.freq.iter().enumerate() {
            let (mut re, mut im) = (0.0, 0.0);
            for (&gain, &delay) in joint.array_gains_db.iter().zip(&joint.array_delays_ms) {
                let angle = -2.0 * std::f64::consts::PI * frequency * delay / 1000.0;
                let magnitude = 10.0_f64.powf(gain / 20.0);
                re += magnitude * angle.cos();
                im += magnitude * angle.sin();
            }
            let expected_level = seat.before.spl[bin] + 20.0 * (re.hypot(im) / 2.0).log10();
            let expected_phase =
                seat.before.phase.as_ref().unwrap()[bin] + im.atan2(re).to_degrees();
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
        decision.accepted,
        "known-solvable fixture must be accepted: {:?} {:?}; selection={:?}",
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
    let yaml = roomeq_export::render_dsp_graph(
        &roundtrip,
        roomeq_export::ExportFormat::CamillaDsp,
        48_000.0,
    )
    .unwrap();
    assert!(yaml.contains("_to_Sub1") && yaml.contains("_to_Sub2"));
    // Keep the engine-only scaling fixture checked by this target as well.
    assert_eq!(fixtures::joint_request(4, 3, 128).measurements.len(), 4);
}
