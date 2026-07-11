use super::*;
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use num_complex::Complex64;

#[test]
fn acoustic_qa_pr_analytic_oracles_are_valid_and_self_consistent() {
    let thresholds = AcceptanceThresholds::default();
    let suite = analytic_oracle_suite();
    assert!(suite.len() >= 10);

    for oracle in suite {
        oracle.validate().unwrap();
        let response = oracle.expected_transfer.to_vec();
        let report = evaluate_oracle(
            &oracle,
            CandidateTransfer {
                transfer: &response,
                impulse: None,
            },
            &thresholds,
        )
        .unwrap();
        assert!(
            report.accepted,
            "oracle '{}' rejected its own ground truth: {:?}",
            oracle.name, report.violations
        );
    }
}

#[test]
fn acoustic_qa_pr_delay_and_allpass_match_temporal_oracles() {
    // Keep adjacent phase increments below π so ordinary unwrap has an
    // unambiguous delay solution on this log grid.
    let frequencies = log_frequency_grid(513, 20.0, 2_000.0);
    let delay = delay_oracle(frequencies.clone(), 3.25);
    let group_delay = group_delay_ms(
        frequencies.as_slice().unwrap(),
        delay.expected_transfer.as_slice().unwrap(),
    );
    let mean_delay = group_delay.iter().sum::<f64>() / group_delay.len() as f64;
    assert!((mean_delay - 3.25).abs() < 1e-6);

    let allpass = allpass_oracle(frequencies, 120.0, 1.1);
    assert!(
        allpass
            .expected_transfer
            .iter()
            .all(|value| { (20.0 * value.norm().log10()).abs() < 1e-9 && value.arg().is_finite() })
    );
    assert!(
        allpass
            .expected_transfer
            .iter()
            .any(|value| value.arg().abs() > 0.2)
    );
}

#[test]
fn acoustic_qa_pr_crossover_and_parallel_components_sum_complexly() {
    let crossover = linkwitz_riley4_oracle(log_frequency_grid(513, 20.0, 20_000.0), 1_000.0);
    assert_eq!(crossover.components.len(), 2);
    for index in 0..crossover.frequencies_hz.len() {
        let sum = crossover.components[0][index] + crossover.components[1][index];
        assert!((sum - crossover.expected_transfer[index]).norm() < 1e-12);
        assert!((crossover.expected_transfer[index].norm() - 1.0).abs() < 1e-10);
    }

    let parallel = parallel_woofers_oracle(
        log_frequency_grid(193, 20.0, 300.0),
        vec![
            ParallelSourceParameters {
                gain_db: 0.0,
                delay_ms: 0.0,
                inverted: false,
            },
            ParallelSourceParameters {
                gain_db: -2.0,
                delay_ms: 1.5,
                inverted: false,
            },
            ParallelSourceParameters {
                gain_db: -3.0,
                delay_ms: 2.5,
                inverted: true,
            },
        ],
    );
    assert_eq!(parallel.components.len(), 3);
    for index in 0..parallel.frequencies_hz.len() {
        let sum = parallel
            .components
            .iter()
            .map(|component| component[index])
            .sum::<Complex64>();
        assert!((sum - parallel.expected_transfer[index]).norm() < 1e-12);
    }
}

#[test]
fn acoustic_qa_pr_generated_biquad_is_checked_by_complex_transfer() {
    let sample_rate = 48_000.0;
    let frequencies = log_frequency_grid(257, 20.0, 20_000.0);
    let filters = vec![
        Biquad::new(BiquadFilterType::Peak, 80.0, sample_rate, 1.2, -4.0),
        Biquad::new(BiquadFilterType::AllPass, 140.0, sample_rate, 0.9, 0.0),
    ];
    let oracle = biquad_cascade_oracle(
        "generated_biquad_chain",
        frequencies.clone(),
        &filters,
        sample_rate,
    );
    let generated =
        crate::response::compute_peq_complex_response(&filters, &frequencies, sample_rate);
    compare_complex_transfers(
        frequencies.as_slice().unwrap(),
        oracle.expected_transfer.as_slice().unwrap(),
        &generated,
        1e-9,
        1e-9,
    )
    .unwrap();
}

#[test]
fn acoustic_qa_pr_safety_gate_rejects_null_boost_and_pre_ringing() {
    let oracle = comb_null_oracle(log_frequency_grid(513, 20.0, 20_000.0), 0.98, 5.0);
    let mut boosted = oracle.expected_transfer.to_vec();
    for (frequency, value) in oracle.frequencies_hz.iter().zip(boosted.iter_mut()) {
        if (90.0..=110.0).contains(frequency) {
            *value *= 10.0_f64.powf(12.0 / 20.0);
        }
    }
    let report = evaluate_oracle(
        &oracle,
        CandidateTransfer {
            transfer: &boosted,
            impulse: None,
        },
        &AcceptanceThresholds {
            max_weighted_rms_db: 20.0,
            max_p95_residual_db: 20.0,
            max_worst_residual_db: 20.0,
            max_correction_energy_db2: 1_000.0,
            max_group_delay_residual_rms_ms: 100.0,
        },
    )
    .unwrap();
    assert!(!report.accepted);
    assert!(
        report
            .violations
            .iter()
            .any(|violation| violation.metric == "boost_into_null_db")
    );

    let excess = excess_phase_oracle(log_frequency_grid(257, 20.0, 20_000.0), vec![(100.0, 1.0)]);
    let impulse = [0.2, 0.2, 1.0, 0.0];
    let transfer = excess.expected_transfer.to_vec();
    let report = evaluate_oracle(
        &excess,
        CandidateTransfer {
            transfer: &transfer,
            impulse: Some(ImpulseEvidence {
                samples: &impulse,
                sample_rate: 48_000.0,
            }),
        },
        &AcceptanceThresholds::default(),
    )
    .unwrap();
    assert!(!report.accepted);
    assert!(
        report
            .violations
            .iter()
            .any(|violation| violation.metric == "pre_ringing_energy_db")
    );
}

#[test]
fn acoustic_qa_pr_schroeder_fixture_has_known_transition() {
    let oracle = room_transition_oracle(log_frequency_grid(257, 20.0, 20_000.0), 30.0, 0.4, 1.0);
    let expected = 2000.0 * (0.4_f64 / 30.0).sqrt();
    match oracle.generating_parameters {
        OracleParameters::RoomTransition { schroeder_hz, .. } => {
            assert!((schroeder_hz - expected).abs() < 1e-12);
            assert!(oracle.valid_correction_region_hz.0 < schroeder_hz);
            assert!(oracle.valid_correction_region_hz.1 > schroeder_hz);
        }
        other => panic!("unexpected parameters: {other:?}"),
    }
}

#[test]
fn acoustic_qa_pr_scenario_matrix_covers_required_axes_deterministically() {
    let first = scenario_matrix(QaTier::Pr);
    let second = scenario_matrix(QaTier::Pr);
    assert_eq!(first, second);
    assert!(first.iter().any(|scenario| scenario.seats > 1));
    assert!(
        first
            .iter()
            .any(|scenario| scenario.grid == GridProfile::Mismatched)
    );
    assert!(
        first
            .iter()
            .any(|scenario| scenario.phase == PhaseAvailability::Missing)
    );
    assert!(
        first
            .iter()
            .any(|scenario| { matches!(scenario.topology, SpeakerTopology::ParallelWoofers(4)) })
    );
    assert!(
        first
            .iter()
            .any(|scenario| { matches!(scenario.topology, SpeakerTopology::MultiSub(4)) })
    );
    assert!(
        first
            .iter()
            .any(|scenario| { matches!(scenario.topology, SpeakerTopology::HeightLayout) })
    );

    for scenario in &first {
        oracle_for_scenario(scenario).validate().unwrap();
        assert!(scenario.held_out_measurements > 0);
        assert_eq!(
            scenario.environment.training_positions_m.len(),
            scenario.seats
        );
        assert_eq!(
            scenario.environment.held_out_positions_m.len(),
            scenario.held_out_measurements
        );
        assert_eq!(
            scenario.comparison_set,
            [
                CandidateKind::Identity,
                CandidateKind::AnalyticCorrection,
                CandidateKind::CurrentMain,
                CandidateKind::Candidate,
            ]
        );
        assert!(scenario.environment.rt60_seconds > 0.0);
        assert!(scenario.environment.crossover_hz > 0.0);
        for position in scenario
            .environment
            .training_positions_m
            .iter()
            .chain(&scenario.environment.held_out_positions_m)
        {
            for (coordinate, dimension) in
                position.iter().zip(scenario.environment.room_dimensions_m)
            {
                assert!((0.0..dimension).contains(coordinate));
            }
        }
    }

    assert!(
        first
            .windows(2)
            .any(|pair| pair[0].environment.room_dimensions_m
                != pair[1].environment.room_dimensions_m)
    );
    assert!(
        first
            .windows(2)
            .any(|pair| pair[0].environment.rt60_seconds != pair[1].environment.rt60_seconds)
    );
    assert!(
        first
            .windows(2)
            .any(|pair| pair[0].environment.crossover_hz != pair[1].environment.crossover_hz)
    );
}

#[test]
fn acoustic_qa_pr_distribution_and_timbre_metrics_track_intended_quantity() {
    let before = vec![vec![0.0, 0.0, 0.0, 0.0], vec![2.0, 1.0, -1.0, -2.0]];
    let after = vec![vec![0.0, 0.0, 0.0, 0.0], vec![0.2, 0.1, -0.1, -0.2]];
    let before_spread = normalized_timbre_spread_db(&before).unwrap();
    let after_spread = normalized_timbre_spread_db(&after).unwrap();
    assert!(after_spread < before_spread);

    assert_eq!(worst_tail_mean(&[1.0, 2.0, 3.0, 4.0], 0.5), 3.5);
    let oracle = identity_oracle(log_frequency_grid(33, 20.0, 20_000.0));
    let transfer = oracle.expected_transfer.to_vec();
    let reports = (0..4)
        .map(|_| {
            evaluate_oracle(
                &oracle,
                CandidateTransfer {
                    transfer: &transfer,
                    impulse: None,
                },
                &AcceptanceThresholds::default(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let summary = summarize_distribution(&reports);
    assert_eq!(summary.count, 4);
    assert_eq!(summary.accepted_fraction, 1.0);
    assert_eq!(summary.p95_weighted_rms_db, 0.0);
}

#[test]
#[ignore = "nightly acoustic scenario matrix"]
fn acoustic_qa_nightly_matrix_is_deterministic_and_finite() {
    let scenarios = scenario_matrix(QaTier::Nightly);
    assert!(scenarios.len() >= 300);
    for scenario in scenarios {
        let oracle = oracle_for_scenario(&scenario);
        oracle.validate().unwrap();
        let first = perturb_transfer(
            oracle.expected_transfer.as_slice().unwrap(),
            scenario.noise,
            scenario.seed,
        );
        let replay = perturb_transfer(
            oracle.expected_transfer.as_slice().unwrap(),
            scenario.noise,
            scenario.seed,
        );
        assert_eq!(first, replay, "non-deterministic replay: {}", scenario.name);
        assert!(
            first
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
    }
}
