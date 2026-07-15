use crate::Curve;
use ndarray::Array1;

use super::interpolate::*;

/// Low frequency bound for normalization (1000 Hz).
///
/// SPL values are normalized by subtracting the mean in the range
/// from `NORMALIZE_LOW_FREQ` to `NORMALIZE_HIGH_FREQ`.
pub const NORMALIZE_LOW_FREQ: f64 = 1000.0;

/// High frequency bound for normalization (2000 Hz).
///
/// SPL values are normalized by subtracting the mean in the range
/// from `NORMALIZE_LOW_FREQ` to `NORMALIZE_HIGH_FREQ`.
pub const NORMALIZE_HIGH_FREQ: f64 = 2000.0;

/// Normalize a response by subtracting its log-frequency-weighted mean.
fn normalize_response(input: &Curve, f_min: f64, f_max: f64) -> Array1<f64> {
    if let Some(mean) = mean_over_log_frequency(&input.freq, &input.spl, f_min, f_max) {
        input.spl.clone() - mean // Subtract mean from all values
    } else {
        input.spl.clone() // Return unchanged if no points in range
    }
}

/// Normalize and interpolate a frequency response curve.
///
/// Interpolates to the standard frequency grid, then normalizes the SPL by
/// subtracting the log-frequency-weighted mean in the 1000-2000 Hz range.
///
/// # Arguments
///
/// * `standard_freq` - Target frequency grid for interpolation
/// * `curve` - Input frequency response curve
///
/// # Returns
///
/// Normalized and interpolated curve on the standard frequency grid.
pub fn normalize_and_interpolate_response(
    standard_freq: &ndarray::Array1<f64>,
    curve: &Curve,
) -> Curve {
    let mut interpolated = interpolate_log_space(standard_freq, curve);
    interpolated.spl = normalize_response(&interpolated, NORMALIZE_LOW_FREQ, NORMALIZE_HIGH_FREQ);
    interpolated
}

/// Interpolate a frequency response curve WITHOUT normalizing.
///
/// This preserves the original dB levels of the curve, only resampling
/// to the standard frequency grid. Useful for CEA2034 visualization
/// where curves should maintain their relative levels.
///
/// # Arguments
///
/// * `standard_freq` - Target frequency grid for interpolation
/// * `curve` - Input frequency response curve
///
/// # Returns
///
/// Interpolated curve on the standard frequency grid (no normalization applied).
pub fn interpolate_response(standard_freq: &ndarray::Array1<f64>, curve: &Curve) -> Curve {
    interpolate_log_space(standard_freq, curve)
}

/// Normalize and interpolate response with custom normalization frequency range
///
/// This is useful for multi-driver systems where each driver should be normalized
/// by its own passband rather than a fixed 1000-2000 Hz range.
pub fn normalize_and_interpolate_response_with_range(
    standard_freq: &ndarray::Array1<f64>,
    curve: &Curve,
    norm_freq_min: f64,
    norm_freq_max: f64,
) -> Curve {
    let mut interpolated = interpolate_log_space(standard_freq, curve);
    interpolated.spl = normalize_response(&interpolated, norm_freq_min, norm_freq_max);
    interpolated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cea2034::Curve;
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
}
