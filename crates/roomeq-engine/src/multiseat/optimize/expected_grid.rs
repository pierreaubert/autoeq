//! Finite candidate-lattice diagnostic, not a global optimality certificate.
use super::*;
use autoeq_core::Curve;
use math_audio_optimisation::continuous_area::Prior;
use roomeq_analysis::listening_area::{ListeningArea, ListeningAreaInterpolatorConfig};

#[test]
fn canonical_expected_candidate_grid_diagnostic() {
    // Mirror QA's canonical_fixture; pinned baseline equality below detects
    // drift in this fixture/evaluator combination. No QA threshold is changed.
    let freq = Array1::from_iter((0..64).map(|i| 20.0 * 10.0_f64.powf(i as f64 / 63.0)));
    let curve = |low_db: f64, offset: f64| Curve {
        freq: freq.clone(),
        spl: freq.mapv(|f| if f < 60.0 { low_db } else { 90.0 }),
        phase: Some(freq.mapv(|f| -180.0 * f / 100.0 + offset)),
        ..Default::default()
    };
    let measurements = MultiSeatMeasurements::new(vec![
        vec![curve(90.0, 0.0), curve(84.0, 12.0)],
        vec![curve(96.0, 24.0), curve(90.0, 36.0)],
    ])
    .unwrap();
    let area = ListeningArea::<1>::new(
        vec![[0.0], [1.0]],
        measurements.measurements.clone(),
        ListeningAreaInterpolatorConfig {
            idw_power: 2.0,
            ..Default::default()
        },
    )
    .unwrap();
    let prior = Prior::Uniform {
        bounds: [(0.0, 1.0)],
    };
    let (points, weights) = sobol_quadrature_points(&prior, 16, 771001).unwrap();
    let freqs = create_eval_frequency_grid(&measurements, 20.0, 120.0);
    let complex = points
        .iter()
        .map(|&p| {
            let evidence = area.interpolate_with_evidence(p).unwrap();
            evidence_response_to_complex(&evidence, &freqs).unwrap().0
        })
        .collect();
    let mut evaluator = AreaEvaluator::new(2, complex, weights, freqs, 48_000.0, 20.0, 120.0, 1);
    let mut evaluate = |gain: f64, delay: f64| {
        evaluator.evaluate_expected(
            &[0.0, gain],
            &[0.0, delay],
            &[false, false],
            &[vec![], vec![]],
        ) + 0.01 * gain * gain
    };
    let baseline = evaluate(0.0, 0.0);
    assert!(
        (baseline - 1.0380915129758224).abs() < 1e-10,
        "canonical QA baseline drift: {baseline}"
    );
    let options = MsoSearchOptions::from_config(&MultiSeatConfig::default(), 20.0, 120.0);
    let (lower, upper) = mso_bounds(2, options);
    assert_eq!(lower, [-6.0, 0.0]);
    assert_eq!(upper, [6.0, 20.0]);
    let mut best = (baseline, 0.0, 0.0);
    let mut evaluations = 0;
    for gain_index in 0..=120 {
        for delay_index in 0..=200 {
            let gain = -6.0 + gain_index as f64 * 0.1;
            let delay = delay_index as f64 * 0.1;
            let loss = evaluate(gain, delay);
            assert!(loss.is_finite());
            evaluations += 1;
            if loss < best.0 {
                best = (loss, gain, delay);
            }
        }
    }
    let coarse = best;
    for gain_index in -100..=100 {
        for delay_index in -100..=100 {
            let gain = coarse.1 + gain_index as f64 * 0.001;
            let delay = coarse.2 + delay_index as f64 * 0.001;
            if gain < lower[0] || gain > upper[0] || delay < lower[1] || delay > upper[1] {
                continue;
            }
            let loss = evaluate(gain, delay);
            assert!(loss.is_finite());
            evaluations += 1;
            if loss < best.0 {
                best = (loss, gain, delay);
            }
        }
    }
    // Resolve the tiny near-zero gain displacement found by the production
    // seed diagnostic without presenting a coarse lattice miss as exact identity.
    let local = best;
    for index in -100..=100 {
        let gain = local.1 + index as f64 * 0.00001;
        if gain < lower[0] || gain > upper[0] {
            continue;
        }
        let loss = evaluate(gain, local.2);
        evaluations += 1;
        if loss < best.0 {
            best = (loss, gain, local.2);
        }
    }
    let improvement = 100.0 * (baseline - best.0) / baseline;
    let output = serde_json::json!({
        "scope": "finite_gain_delay_lattice_diagnostic_not_global_optimality_or_quality_acceptance",
        "canonical_fixture_baseline": baseline, "best_objective": best.0,
        "best_gain_db": best.1, "best_delay_ms": best.2,
        "improvement_pct": improvement, "required_improvement_pct": 1.0,
        "meets_registry_improvement": improvement >= 1.0,
        "candidate_evaluations": evaluations, "gain_bounds_db": [lower[0], upper[0]],
        "delay_bounds_ms": [lower[1], upper[1]], "coarse_step": 0.1, "local_step": 0.001,
        "final_gain_only_step_db": 0.00001,
        "quadrature_seed": 771001, "quadrature_points": points,
        "fixed_first_output": true, "polarity_optimization": false, "allpass_count": 0,
        "sample_rate": 48000, "evaluation_band_hz": [20.0, 120.0],
    });
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/qa");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("continuous-expected-candidate-grid.json"),
        serde_json::to_vec_pretty(&output).unwrap(),
    )
    .unwrap();
    eprintln!("{output}");
}
