use crate::Curve;
use crate::curve_transforms::*;
use ndarray::Array1;

#[test]
fn test_interpolate_log_space_zero_freq() {
    let curve = Curve {
        freq: Array1::from_vec(vec![0.0, 100.0]),
        spl: Array1::from_vec(vec![0.0, 100.0]),
        phase: Some(Array1::from_vec(vec![0.0, 50.0])),
        ..Default::default()
    };
    let freq_out = Array1::from_vec(vec![0.0, 50.0, 100.0]);
    let result = interpolate_log_space(&freq_out, &curve);
    // DC cannot participate in logarithmic interpolation. The segment
    // from DC to the first positive bin must therefore be linear.
    assert_eq!(result.spl.to_vec(), vec![0.0, 50.0, 100.0]);
    assert_eq!(result.phase.unwrap().to_vec(), vec![0.0, 25.0, 50.0]);
}

#[test]
fn interpolate_log_space_values_exact_match() {
    let log_freq_in = vec![1.0_f64.ln(), 10.0_f64.ln(), 100.0_f64.ln()];
    let vals_in = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let log_freq_out = vec![1.0_f64.ln(), 10.0_f64.ln(), 100.0_f64.ln()];
    let result = interpolate_log_space_values(&log_freq_out, &log_freq_in, &vals_in);
    assert_eq!(result.len(), 3);
    assert!((result[0] - 0.0).abs() < 1e-12);
    assert!((result[1] - 1.0).abs() < 1e-12);
    assert!((result[2] - 2.0).abs() < 1e-12);
}

#[test]
fn interpolate_log_space_values_interior() {
    let log_freq_in = vec![1.0_f64.ln(), 100.0_f64.ln()];
    let vals_in = Array1::from_vec(vec![0.0, 2.0]);
    let log_freq_out = vec![10.0_f64.ln()];
    let result = interpolate_log_space_values(&log_freq_out, &log_freq_in, &vals_in);
    assert_eq!(result.len(), 1);
    // log10(10)=1 is midpoint between log10(1)=0 and log10(100)=2
    assert!((result[0] - 1.0).abs() < 1e-12);
}

#[test]
fn log_frequency_mean_is_invariant_to_samples_on_linear_segments() {
    let sparse_freq = Array1::from_vec(vec![100.0, 200.0, 400.0]);
    let sparse_values = Array1::from_vec(vec![0.0, 10.0, 0.0]);
    let dense_freq = Array1::from_vec(vec![100.0, 150.0, 200.0, 300.0, 400.0]);
    let dense_values = Array1::from_iter(dense_freq.iter().map(|&freq| {
        if freq <= 200.0 {
            10.0 * (freq / 100.0_f64).ln() / 2.0_f64.ln()
        } else {
            10.0 * (1.0 - (freq / 200.0_f64).ln() / 2.0_f64.ln())
        }
    }));

    let sparse = mean_over_log_frequency(&sparse_freq, &sparse_values, 100.0, 400.0).unwrap();
    let dense = mean_over_log_frequency(&dense_freq, &dense_values, 100.0, 400.0).unwrap();
    assert!((sparse - dense).abs() < 1e-12);
}

#[test]
fn interpolate_log_space_values_extrapolate_below() {
    let log_freq_in = vec![10.0_f64.ln(), 100.0_f64.ln()];
    let vals_in = Array1::from_vec(vec![1.0, 2.0]);
    let log_freq_out = vec![1.0_f64.ln()];
    let result = interpolate_log_space_values(&log_freq_out, &log_freq_in, &vals_in);
    assert_eq!(result.len(), 1);
    // slope = 1/ln(10); result = 1 + slope*(ln(1)-ln(10)) = 0
    assert!((result[0] - 0.0).abs() < 1e-12);
}

#[test]
fn interpolate_log_space_values_extrapolate_above() {
    let log_freq_in = vec![1.0_f64.ln(), 10.0_f64.ln()];
    let vals_in = Array1::from_vec(vec![0.0, 1.0]);
    let log_freq_out = vec![100.0_f64.ln()];
    let result = interpolate_log_space_values(&log_freq_out, &log_freq_in, &vals_in);
    assert_eq!(result.len(), 1);
    // slope = 1/ln(10); result = 1 + slope*(ln(100)-ln(10)) = 2
    assert!((result[0] - 2.0).abs() < 1e-12);
}

#[test]
fn interpolate_log_space_values_single_point() {
    let log_freq_in = vec![100.0_f64.ln()];
    let vals_in = Array1::from_vec(vec![5.0]);
    let log_freq_out = vec![1.0_f64.ln(), 100.0_f64.ln(), 1000.0_f64.ln()];
    let result = interpolate_log_space_values(&log_freq_out, &log_freq_in, &vals_in);
    assert_eq!(result.len(), 3);
    for v in result.iter() {
        assert!((v - 5.0).abs() < 1e-12);
    }
}

#[test]
fn interpolate_log_space_values_zero_denom() {
    // Equal log frequencies should not panic
    let log_freq_in = vec![1.0, 1.0, 2.0];
    let vals_in = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let log_freq_out = vec![1.5];
    let result = interpolate_log_space_values(&log_freq_out, &log_freq_in, &vals_in);
    assert!(result[0].is_finite());
}

#[test]
fn interpolate_log_space_values_empty_out() {
    let log_freq_in = vec![1.0_f64.ln(), 10.0_f64.ln()];
    let vals_in = Array1::from_vec(vec![0.0, 1.0]);
    let log_freq_out: Vec<f64> = vec![];
    let result = interpolate_log_space_values(&log_freq_out, &log_freq_in, &vals_in);
    assert!(result.is_empty());
}

#[test]
fn interpolate_log_space_values_preserves_phase() {
    let freq_in = Array1::from_vec(vec![100.0, 1000.0, 10000.0]);
    let curve = Curve {
        freq: freq_in.clone(),
        spl: Array1::from_vec(vec![0.0, 1.0, 2.0]),
        phase: Some(Array1::from_vec(vec![10.0, 20.0, 30.0])),
        ..Default::default()
    };
    let freq_out = Array1::from_vec(vec![100.0, 1000.0, 10000.0]);
    let result = interpolate_log_space(&freq_out, &curve);
    assert!(result.phase.is_some());
    let phase = result.phase.unwrap();
    assert!((phase[0] - 10.0).abs() < 1e-9);
    assert!((phase[1] - 20.0).abs() < 1e-9);
    assert!((phase[2] - 30.0).abs() < 1e-9);
}

#[test]
fn interpolate_log_space_unwraps_phase_across_branch_cut() {
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 10_000.0]),
        spl: Array1::from_vec(vec![0.0, 0.0]),
        phase: Some(Array1::from_vec(vec![170.0, -170.0])),
        ..Default::default()
    };

    let result = interpolate_log_space(&Array1::from_vec(vec![1_000.0]), &curve);
    let phase = result.phase.expect("phase should be preserved");

    assert!((phase[0] - 180.0).abs() < 1e-9, "phase was {}", phase[0]);
}

#[test]
fn create_log_frequency_grid_bounds_and_length() {
    let grid = create_log_frequency_grid(50, 20.0, 20000.0);
    assert_eq!(grid.len(), 50);
    assert!((grid[0] - 20.0).abs() < 1e-9, "first point should be f_min");
    assert!(
        (grid[49] - 20000.0).abs() < 1e-9,
        "last point should be f_max"
    );
    // Log-spaced means ratio between consecutive points is constant
    let ratio0 = grid[1] / grid[0];
    let ratio1 = grid[2] / grid[1];
    assert!((ratio0 - ratio1).abs() < 1e-6, "grid should be log-spaced");
}

#[test]
fn interpolate_linear_exact_match() {
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 1000.0, 10000.0]),
        spl: Array1::from_vec(vec![0.0, 10.0, 5.0]),
        phase: None,
        ..Default::default()
    };
    let freq_out = Array1::from_vec(vec![100.0, 1000.0, 10000.0]);
    let result = interpolate(&freq_out, &curve);
    assert!((result.spl[0] - 0.0).abs() < 1e-12);
    assert!((result.spl[1] - 10.0).abs() < 1e-12);
    assert!((result.spl[2] - 5.0).abs() < 1e-12);
}

#[test]
fn interpolate_linear_unwraps_phase_across_branch_cut() {
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 300.0]),
        spl: Array1::from_vec(vec![0.0, 0.0]),
        phase: Some(Array1::from_vec(vec![170.0, -170.0])),
        ..Default::default()
    };

    let result = interpolate(&Array1::from_vec(vec![200.0]), &curve);
    let phase = result.phase.expect("phase should be preserved");

    assert!((phase[0] - 180.0).abs() < 1e-9, "phase was {}", phase[0]);
}

#[test]
fn interpolate_linear_interior() {
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 200.0]),
        spl: Array1::from_vec(vec![0.0, 10.0]),
        phase: None,
        ..Default::default()
    };
    let freq_out = Array1::from_vec(vec![150.0]);
    let result = interpolate(&freq_out, &curve);
    assert!((result.spl[0] - 5.0).abs() < 1e-12);
}

#[test]
fn interpolate_linear_extrapolates_below() {
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 200.0]),
        spl: Array1::from_vec(vec![0.0, 10.0]),
        phase: None,
        ..Default::default()
    };
    let freq_out = Array1::from_vec(vec![50.0]);
    let result = interpolate(&freq_out, &curve);
    // Below range uses first point
    assert!((result.spl[0] - 0.0).abs() < 1e-12);
}

#[test]
fn interpolate_linear_extrapolates_above() {
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 200.0]),
        spl: Array1::from_vec(vec![0.0, 10.0]),
        phase: None,
        ..Default::default()
    };
    let freq_out = Array1::from_vec(vec![300.0]);
    let result = interpolate(&freq_out, &curve);
    // Above range uses last point
    assert!((result.spl[0] - 10.0).abs() < 1e-12);
}

#[test]
fn interpolate_linear_preserves_phase() {
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 200.0]),
        spl: Array1::from_vec(vec![0.0, 10.0]),
        phase: Some(Array1::from_vec(vec![0.0, 90.0])),
        ..Default::default()
    };
    let freq_out = Array1::from_vec(vec![150.0]);
    let result = interpolate(&freq_out, &curve);
    assert!(result.phase.is_some());
    let phase = result.phase.unwrap();
    assert!((phase[0] - 45.0).abs() < 1e-12);
}
