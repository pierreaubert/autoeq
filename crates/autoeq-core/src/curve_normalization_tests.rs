use crate::Curve;
use crate::curve_transforms::*;
use ndarray::Array1;

#[test]
fn normalize_response_subtracts_log_frequency_mean_in_range() {
    let curve = Curve {
        freq: Array1::from_vec(vec![500.0, 1000.0, 1500.0, 2000.0, 2500.0]),
        spl: Array1::from_vec(vec![0.0, 2.0, 4.0, 6.0, 8.0]),
        phase: None,
        ..Default::default()
    };
    let mean = mean_over_log_frequency(&curve.freq, &curve.spl, 1000.0, 2000.0).unwrap();
    let result = normalize_response(&curve, 1000.0, 2000.0);
    for ((&actual, &original), index) in result.iter().zip(curve.spl.iter()).zip(0..) {
        assert!(
            (actual - (original - mean)).abs() < 1e-12,
            "unexpected normalized value at {index}"
        );
    }
}

#[test]
fn normalize_response_no_points_in_range_returns_unchanged() {
    let curve = Curve {
        freq: Array1::from_vec(vec![10.0, 20.0, 30.0]),
        spl: Array1::from_vec(vec![5.0, 6.0, 7.0]),
        phase: None,
        ..Default::default()
    };
    let result = normalize_response(&curve, 100.0, 200.0);
    assert_eq!(result.to_vec(), vec![5.0, 6.0, 7.0]);
}

#[test]
fn normalize_and_interpolate_response_preserves_shape() {
    let standard_freq = Array1::logspace(10.0, 2.0, 4.0, 10);
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 1000.0, 10000.0]),
        spl: Array1::from_vec(vec![0.0, 5.0, 0.0]),
        phase: None,
        ..Default::default()
    };
    let result = normalize_and_interpolate_response(&standard_freq, &curve);
    assert_eq!(result.freq.len(), standard_freq.len());
    assert_eq!(result.spl.len(), standard_freq.len());
    // All output values should be finite
    for &v in result.spl.iter() {
        assert!(v.is_finite(), "spl must be finite");
    }
}

#[test]
fn interpolate_response_preserves_levels() {
    let standard_freq = Array1::from_vec(vec![100.0, 1000.0, 10000.0]);
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 1000.0, 10000.0]),
        spl: Array1::from_vec(vec![80.0, 85.0, 82.0]),
        phase: None,
        ..Default::default()
    };
    let result = interpolate_response(&standard_freq, &curve);
    // Exact match when grids align
    assert!((result.spl[0] - 80.0).abs() < 1e-9);
    assert!((result.spl[1] - 85.0).abs() < 1e-9);
    assert!((result.spl[2] - 82.0).abs() < 1e-9);
}

#[test]
fn normalize_and_interpolate_response_with_range_uses_custom_range() {
    let standard_freq = Array1::from_vec(vec![100.0, 1000.0, 10000.0]);
    let curve = Curve {
        freq: Array1::from_vec(vec![100.0, 500.0, 10000.0]),
        spl: Array1::from_vec(vec![0.0, 10.0, 0.0]),
        phase: None,
        ..Default::default()
    };
    // Normalize using 100-500 Hz range (mean = 5.0)
    let result =
        normalize_and_interpolate_response_with_range(&standard_freq, &curve, 100.0, 500.0);
    assert_eq!(result.freq.len(), standard_freq.len());
    for &v in result.spl.iter() {
        assert!(v.is_finite());
    }
}

#[test]
fn normalize_and_interpolate_is_invariant_to_source_grid_density() {
    fn response_at(freq: f64) -> f64 {
        if freq <= 1000.0 {
            0.0
        } else if freq >= 2000.0 {
            6.0
        } else {
            6.0 * (freq / 1000.0).ln() / 2.0_f64.ln()
        }
    }

    let sparse_freq = vec![500.0, 1000.0, 2000.0, 4000.0];
    let dense_freq = vec![
        500.0, 1000.0, 1050.0, 1100.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 4000.0,
    ];
    let curve = |freq: Vec<f64>| Curve {
        spl: Array1::from_iter(freq.iter().copied().map(response_at)),
        freq: Array1::from_vec(freq),
        ..Default::default()
    };
    let standard_freq = Array1::logspace(10.0, 500.0_f64.log10(), 4000.0_f64.log10(), 31);

    let sparse = normalize_and_interpolate_response(&standard_freq, &curve(sparse_freq));
    let dense = normalize_and_interpolate_response(&standard_freq, &curve(dense_freq));

    for (index, (&a, &b)) in sparse.spl.iter().zip(dense.spl.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "normalized responses differ at {index}: {a} versus {b}"
        );
    }
}
