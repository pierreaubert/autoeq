//! Tests for multi-measurement optimization strategies.

mod common;

use autoeq::PeqModel;
use autoeq::optim::{
    MultiObjectiveData, ObjectiveData, ObjectiveDataBuilder, compute_base_fitness,
    compute_pareto_objectives,
};
use autoeq::roomeq::MultiMeasurementStrategy;
use ndarray::{Array1, array};

/// Create a minimal ObjectiveData for testing with a given deviation curve.
fn make_objective(deviation: Array1<f64>) -> ObjectiveData {
    let n = deviation.len();
    let freqs = Array1::logspace(10.0, 1.3, 3.3, n); // ~20 Hz to ~2000 Hz
    ObjectiveDataBuilder::speaker_flat(freqs, Array1::zeros(n), deviation, 48000.0, PeqModel::Pk)
        .max_db(6.0)
        .min_db(-12.0)
        .freq_range(20.0, 2000.0)
        .smoothing(false, 3)
        .build()
        .expect("valid test objective data")
}

/// Zero-gain filters: x = [log10(freq), Q, gain_db] with gain=0 → no correction.
fn zero_filters(n_filters: usize) -> Vec<f64> {
    let mut x = Vec::new();
    for i in 0..n_filters {
        let freq_log10 = 2.0 + i as f64 * 0.3; // 100, 200, 400 Hz etc.
        x.push(freq_log10);
        x.push(1.0); // Q
        x.push(0.0); // gain = 0 → no correction
    }
    x
}

#[test]
fn weighted_sum_equal_weights_equals_mean() {
    let obj1 = make_objective(array![1.0, 2.0, 3.0]);
    let obj2 = make_objective(array![3.0, 2.0, 1.0]);

    let x = zero_filters(1);

    // Get individual losses
    let loss1 = compute_base_fitness(&x, &obj1);
    let loss2 = compute_base_fitness(&x, &obj2);

    // Multi-objective with equal weights should give mean
    let mut primary = obj1.clone();
    primary.multi_objective = Some(MultiObjectiveData {
        objectives: vec![obj1, obj2],
        strategy: MultiMeasurementStrategy::WeightedSum,
        weights: vec![0.5, 0.5],
        variance_lambda: 1.0,
        uncertainty_cvar_alpha: None,
    });

    let multi_loss = compute_base_fitness(&x, &primary);
    let expected = (loss1 + loss2) / 2.0;
    assert!(
        (multi_loss - expected).abs() < 1e-10,
        "WeightedSum with equal weights should equal mean: got {}, expected {}",
        multi_loss,
        expected
    );
}

#[test]
fn weighted_sum_respects_weights() {
    let obj1 = make_objective(array![1.0, 2.0, 3.0]);
    let obj2 = make_objective(array![10.0, 10.0, 10.0]); // Much worse

    let x = zero_filters(1);

    let loss1 = compute_base_fitness(&x, &obj1);
    let loss2 = compute_base_fitness(&x, &obj2);

    // Weight heavily toward obj1
    let mut primary = obj1.clone();
    primary.multi_objective = Some(MultiObjectiveData {
        objectives: vec![obj1, obj2],
        strategy: MultiMeasurementStrategy::WeightedSum,
        weights: vec![0.9, 0.1],
        variance_lambda: 1.0,
        uncertainty_cvar_alpha: None,
    });

    let multi_loss = compute_base_fitness(&x, &primary);
    let expected = 0.9 * loss1 + 0.1 * loss2;
    assert!(
        (multi_loss - expected).abs() < 1e-10,
        "WeightedSum should respect weights: got {}, expected {}",
        multi_loss,
        expected
    );
}

#[test]
fn minimax_returns_worst_case() {
    let obj1 = make_objective(array![1.0, 1.0, 1.0]); // Good
    let obj2 = make_objective(array![5.0, 5.0, 5.0]); // Bad
    let obj3 = make_objective(array![3.0, 3.0, 3.0]); // Medium

    let x = zero_filters(1);

    let loss2 = compute_base_fitness(&x, &obj2);

    let mut primary = obj1.clone();
    primary.multi_objective = Some(MultiObjectiveData {
        objectives: vec![obj1, obj2, obj3],
        strategy: MultiMeasurementStrategy::Minimax,
        weights: vec![1.0 / 3.0; 3],
        variance_lambda: 1.0,
        uncertainty_cvar_alpha: None,
    });

    let multi_loss = compute_base_fitness(&x, &primary);
    assert!(
        (multi_loss - loss2).abs() < 1e-10,
        "Minimax should return worst case loss: got {}, expected {}",
        multi_loss,
        loss2
    );
}

#[test]
fn pareto_objectives_return_per_measurement_losses() {
    let obj1 = make_objective(array![1.0, 1.0, 1.0]);
    let obj2 = make_objective(array![5.0, 5.0, 5.0]);
    let x = zero_filters(1);
    let loss1 = compute_base_fitness(&x, &obj1);
    let loss2 = compute_base_fitness(&x, &obj2);

    let mut primary = obj1.clone();
    primary.multi_objective = Some(MultiObjectiveData {
        objectives: vec![obj1, obj2],
        strategy: MultiMeasurementStrategy::WeightedSum,
        weights: vec![0.5, 0.5],
        variance_lambda: 1.0,
        uncertainty_cvar_alpha: None,
    });

    let objectives = compute_pareto_objectives(&x, &primary);
    assert_eq!(objectives.len(), 2);
    assert!((objectives[0] - loss1).abs() < 1e-10);
    assert!((objectives[1] - loss2).abs() < 1e-10);
}

#[test]
fn variance_penalized_penalizes_inconsistency() {
    // Two identical curves: variance = 0
    let obj_same1 = make_objective(array![3.0, 3.0, 3.0]);
    let obj_same2 = make_objective(array![3.0, 3.0, 3.0]);

    // Two different curves: variance > 0
    let obj_diff1 = make_objective(array![1.0, 1.0, 1.0]);
    let obj_diff2 = make_objective(array![5.0, 5.0, 5.0]);

    let x = zero_filters(1);

    let loss_same = compute_base_fitness(&x, &obj_same1);
    let loss_diff1 = compute_base_fitness(&x, &obj_diff1);
    let loss_diff2 = compute_base_fitness(&x, &obj_diff2);

    // Consistent (same) curves
    let mut primary_same = obj_same1.clone();
    primary_same.multi_objective = Some(MultiObjectiveData {
        objectives: vec![obj_same1, obj_same2],
        strategy: MultiMeasurementStrategy::VariancePenalized,
        weights: vec![0.5, 0.5],
        variance_lambda: 1.0,
        uncertainty_cvar_alpha: None,
    });
    let consistent_loss = compute_base_fitness(&x, &primary_same);

    // Inconsistent (different) curves
    let mut primary_diff = obj_diff1.clone();
    primary_diff.multi_objective = Some(MultiObjectiveData {
        objectives: vec![obj_diff1, obj_diff2],
        strategy: MultiMeasurementStrategy::VariancePenalized,
        weights: vec![0.5, 0.5],
        variance_lambda: 1.0,
        uncertainty_cvar_alpha: None,
    });
    let inconsistent_loss = compute_base_fitness(&x, &primary_diff);

    // Consistent case: variance = 0, so loss = mean(losses)
    assert!(
        (consistent_loss - loss_same).abs() < 1e-10,
        "Consistent curves should have zero variance penalty"
    );

    // Inconsistent case should have higher loss due to variance penalty
    let mean_loss = (loss_diff1 + loss_diff2) / 2.0;
    assert!(
        inconsistent_loss > mean_loss,
        "Inconsistent curves should have variance penalty: {} > {}",
        inconsistent_loss,
        mean_loss
    );
}

#[test]
fn variance_lambda_scales_penalty() {
    let obj1 = make_objective(array![1.0, 1.0, 1.0]);
    let obj2 = make_objective(array![5.0, 5.0, 5.0]);
    let x = zero_filters(1);

    let make_multi = |lambda: f64| -> ObjectiveData {
        let mut primary = obj1.clone();
        primary.multi_objective = Some(MultiObjectiveData {
            objectives: vec![obj1.clone(), obj2.clone()],
            strategy: MultiMeasurementStrategy::VariancePenalized,
            weights: vec![0.5, 0.5],
            variance_lambda: lambda,
            uncertainty_cvar_alpha: None,
        });
        primary
    };

    let loss_low_lambda = compute_base_fitness(&x, &make_multi(0.1));
    let loss_high_lambda = compute_base_fitness(&x, &make_multi(10.0));

    assert!(
        loss_high_lambda > loss_low_lambda,
        "Higher lambda should give higher loss: {} > {}",
        loss_high_lambda,
        loss_low_lambda
    );
}

#[test]
fn config_serialization_roundtrip() {
    use autoeq::roomeq::{MultiMeasurementConfig, MultiMeasurementStrategy};

    let config = MultiMeasurementConfig {
        strategy: MultiMeasurementStrategy::Minimax,
        weights: None,
        variance_lambda: 2.5,
        spatial_robustness: None,
        bootstrap_uncertainty: None,
        rir_prototype: None,
    };

    let json = serde_json::to_string(&config).unwrap();
    let parsed: MultiMeasurementConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.strategy, MultiMeasurementStrategy::Minimax);
    assert!(parsed.weights.is_none());
    assert!((parsed.variance_lambda - 2.5).abs() < 1e-10);
}

#[test]
fn config_default_is_average() {
    let config = autoeq::roomeq::MultiMeasurementConfig::default();
    assert_eq!(config.strategy, MultiMeasurementStrategy::Average);
    assert!((config.variance_lambda - 1.0).abs() < 1e-10);
}

#[test]
fn load_source_individual_single() {
    use autoeq::read::load_source_individual;
    use autoeq::{Curve, MeasurementSource};
    use ndarray::Array1;

    let curve = Curve {
        freq: Array1::from(vec![20.0, 200.0, 2000.0]),
        spl: Array1::from(vec![80.0, 85.0, 82.0]),
        phase: None,
        ..Default::default()
    };

    let source = MeasurementSource::InMemory(curve.clone());
    let curves = load_source_individual(&source).unwrap();

    assert_eq!(curves.len(), 1);
    assert_eq!(curves[0].freq.len(), 3);
}

#[test]
fn rir_prototype_config_dry_run_succeeds() {
    use common::binary_runner::run_roomeq;
    use std::fs;
    use std::path::PathBuf;

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let config_path = temp_dir.path().join("rir_prototype_config.json");
    let output_path = temp_dir.path().join("output.json");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let left = manifest_dir.join("tests/data/roomeq/test_speaker_left.csv");
    let right = manifest_dir.join("tests/data/roomeq/test_speaker_right.csv");
    let tweeter = manifest_dir.join("tests/data/roomeq/test_tweeter.csv");

    let config = serde_json::json!({
        "version": "2.1.0",
        "speakers": {
            "left": {
                "measurements": [
                    left.to_str().unwrap(),
                    right.to_str().unwrap(),
                    tweeter.to_str().unwrap()
                ]
            }
        },
        "optimizer": {
            "num_filters": 3,
            "max_iter": 5,
            "population": 8,
            "multi_measurement": {
                "strategy": "weighted_sum",
                "rir_prototype": {
                    "reference_position": [0.0, 0.0, 0.0],
                    "source_position": [0.0, 2.5, 0.0],
                    "microphone_positions": [
                        [0.0, 0.0, 0.0],
                        [0.15, 0.0, 0.0],
                        [-0.15, 0.0, 0.0]
                    ],
                    "distance_mode": "inverse_square",
                    "directivity": "omnidirectional",
                    "frequency_dependent_directivity": false
                }
            }
        }
    });

    fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap())
        .expect("Failed to write config");

    let output = run_roomeq(&[
        "--config",
        config_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
        "--dry-run",
    ]);

    if !output.status.success() {
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("roomeq --dry-run failed with status: {}", output.status);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Configuration: VALID"),
        "dry-run should report a valid config; got:\n{}",
        stdout
    );
    assert!(
        stdout.contains("All checks passed"),
        "dry-run should complete successfully; got:\n{}",
        stdout
    );
}
