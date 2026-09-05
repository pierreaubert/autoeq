//! Versioned auditory-frequency integration helpers.

use ndarray::Array1;

/// Identifier for the frequency measure used by optimization and acceptance.
pub const AUDITORY_FREQUENCY_MEASURE_VERSION: &str = "glasberg-moore-erb-rate-1990-v1";

/// Glasberg-Moore ERB-rate number for a frequency in hertz.
pub fn erb_rate(frequency: f64) -> f64 {
    21.4 * (1.0 + 0.00437 * frequency).log10()
}

/// Checked trapezoidal cell widths on the ERB-rate axis.
///
/// Returns `None` when the axis is not a valid strictly increasing grid of
/// finite positive frequencies: fewer than two points, any non-finite or
/// non-positive frequency, or any non-finite/non-positive trapezoidal cell
/// (unsorted, duplicated, or non-monotonic frequencies).
pub fn try_erb_rate_cell_widths(frequencies: &Array1<f64>) -> Option<Array1<f64>> {
    let count = frequencies.len();
    if count < 2 {
        return None;
    }
    if frequencies
        .iter()
        .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return None;
    }

    let rates: Vec<f64> = frequencies
        .iter()
        .map(|frequency| erb_rate(*frequency))
        .collect();
    let mut weights = Array1::zeros(count);
    weights[0] = 0.5 * (rates[1] - rates[0]);
    weights[count - 1] = 0.5 * (rates[count - 1] - rates[count - 2]);
    for index in 1..count - 1 {
        weights[index] = 0.5 * (rates[index + 1] - rates[index - 1]);
    }

    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        None
    } else {
        Some(weights)
    }
}

/// Trapezoidal cell widths on the ERB-rate axis.
///
/// Uniform-fallback policy (explicit, legacy): degenerate axes that
/// [`try_erb_rate_cell_widths`] rejects (fewer than two points, non-finite or
/// non-positive frequencies, unsorted/duplicated grids) yield uniform weights
/// of `1.0` per point (empty for a zero-length axis) instead of an error.
/// This preserves the historical behavior consumed by `autoeq-optim` loss
/// weighting. Callers that must not silently accept an invalid axis should use
/// [`try_erb_rate_cell_widths`] or [`erb_rate_weighted_rms`], which return
/// `None` for such axes.
pub fn erb_rate_cell_widths(frequencies: &Array1<f64>) -> Array1<f64> {
    let count = frequencies.len();
    if count == 0 {
        return Array1::zeros(0);
    }
    try_erb_rate_cell_widths(frequencies).unwrap_or_else(|| Array1::from_elem(count, 1.0))
}

/// RMS integrated on the ERB-rate axis.
///
/// Returns `None` when the axis is invalid (see [`try_erb_rate_cell_widths`]),
/// when the lengths mismatch, when there are no values, or when the weighted
/// result is non-finite. In particular an unsorted or non-finite axis never
/// falls back to uniform weights here.
pub fn erb_rate_weighted_rms(frequencies: &Array1<f64>, values: &[f64]) -> Option<f64> {
    if frequencies.len() != values.len() || values.is_empty() {
        return None;
    }
    let weights = try_erb_rate_cell_widths(frequencies)?;
    let total_weight: f64 = weights.iter().sum();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return None;
    }
    let weighted_square_sum: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| value * value * weight)
        .sum();
    weighted_square_sum
        .is_finite()
        .then(|| (weighted_square_sum / total_weight).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_error_is_grid_invariant() {
        for frequencies in [
            Array1::linspace(20.0, 20_000.0, 100),
            Array1::from_iter((0..100).map(|i| 20.0 * 1000.0_f64.powf(i as f64 / 99.0))),
            Array1::from(vec![20.0, 37.0, 91.0, 410.0, 2_300.0, 20_000.0]),
            Array1::linspace(20.0, 20_000.0, 10_000),
        ] {
            let values = vec![4.25; frequencies.len()];
            let rms = erb_rate_weighted_rms(&frequencies, &values).unwrap();
            assert!((rms - 4.25).abs() < 1e-12);
        }
    }

    #[test]
    fn invalid_axes_are_rejected_by_checked_api() {
        // The reported bug: [100, NaN, 1000] with constant residual 1 must not
        // yield a finite uniform-weighted RMS.
        let nan_axis = Array1::from(vec![100.0, f64::NAN, 1000.0]);
        assert!(try_erb_rate_cell_widths(&nan_axis).is_none());
        assert!(erb_rate_weighted_rms(&nan_axis, &[1.0, 1.0, 1.0]).is_none());

        let invalid_axes = vec![
            Array1::from(vec![100.0, 50.0, 1000.0]), // unsorted
            Array1::from(vec![1000.0, 100.0, 10.0]), // descending
            Array1::from(vec![100.0, 100.0, 1000.0]), // duplicated -> zero cell
            Array1::from(vec![100.0, f64::INFINITY, 1000.0]),
            Array1::from(vec![100.0, -50.0, 1000.0]),
            Array1::from(vec![0.0, 100.0, 1000.0]),
            Array1::from(vec![100.0]),
            Array1::zeros(0),
        ];
        for axis in &invalid_axes {
            assert!(try_erb_rate_cell_widths(axis).is_none(), "{axis:?}");
            let values = vec![1.0; axis.len()];
            assert!(
                erb_rate_weighted_rms(axis, &values).is_none(),
                "{axis:?}"
            );
        }
    }

    #[test]
    fn legacy_uniform_fallback_policy_is_explicit() {
        // Documented fallback: degenerate axes still yield uniform weights via
        // the legacy entry point, while the checked API rejects them.
        for axis in [
            Array1::from(vec![100.0, f64::NAN, 1000.0]),
            Array1::from(vec![100.0, 50.0, 1000.0]),
            Array1::from(vec![100.0]),
        ] {
            let weights = erb_rate_cell_widths(&axis);
            assert_eq!(weights.len(), axis.len());
            assert!(weights.iter().all(|weight| (*weight - 1.0).abs() < 1e-12));
            assert!(try_erb_rate_cell_widths(&axis).is_none());
        }
        assert_eq!(erb_rate_cell_widths(&Array1::zeros(0)).len(), 0);
    }

    #[test]
    fn nonconstant_residual_is_density_invariant() {
        // Residual varies with ERB rate; dense vs coarse log grids must agree.
        let residual = |frequency: f64| erb_rate(frequency) - erb_rate(1000.0);
        let dense = Array1::from_iter(
            (0..2000).map(|i| 20.0 * 1000.0_f64.powf(i as f64 / 1999.0)),
        );
        let coarse = Array1::from_iter(
            (0..25).map(|i| 20.0 * 1000.0_f64.powf(i as f64 / 24.0)),
        );
        let dense_values: Vec<f64> =
            dense.iter().map(|frequency| residual(*frequency)).collect();
        let coarse_values: Vec<f64> =
            coarse.iter().map(|frequency| residual(*frequency)).collect();
        let dense_rms = erb_rate_weighted_rms(&dense, &dense_values).unwrap();
        let coarse_rms = erb_rate_weighted_rms(&coarse, &coarse_values).unwrap();
        let relative = ((coarse_rms - dense_rms) / dense_rms).abs();
        assert!(relative < 0.02, "dense={dense_rms} coarse={coarse_rms}");
    }

    #[test]
    fn shuffled_grid_is_rejected_and_shifted_grid_is_stable() {
        let base =
            Array1::from_iter((0..50).map(|i| 20.0 * 1000.0_f64.powf(i as f64 / 49.0)));
        let residual = |frequency: f64| (erb_rate(frequency) * 0.5).sin();
        let base_values: Vec<f64> =
            base.iter().map(|frequency| residual(*frequency)).collect();
        let base_rms = erb_rate_weighted_rms(&base, &base_values).unwrap();

        // Shuffled (reversed) axis has equal length but is not a valid grid.
        let mut shuffled = base.to_vec();
        shuffled.reverse();
        let shuffled_axis = Array1::from(shuffled);
        assert!(try_erb_rate_cell_widths(&shuffled_axis).is_none());
        assert!(
            erb_rate_weighted_rms(&shuffled_axis, &base_values).is_none()
        );

        // Equal-length but slightly shifted grid must give a nearby RMS for a
        // smooth residual.
        let shifted =
            Array1::from_iter(base.iter().map(|frequency| frequency * 1.001));
        let shifted_values: Vec<f64> =
            shifted.iter().map(|frequency| residual(*frequency)).collect();
        let shifted_rms = erb_rate_weighted_rms(&shifted, &shifted_values).unwrap();
        let relative = ((shifted_rms - base_rms) / base_rms).abs();
        assert!(relative < 0.05, "base={base_rms} shifted={shifted_rms}");

        // Length mismatch is rejected.
        assert!(erb_rate_weighted_rms(&base, &base_values[1..]).is_none());
    }
}
