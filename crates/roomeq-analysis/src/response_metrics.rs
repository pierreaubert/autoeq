//! Scalar response metrics shared by RoomEQ preparation and reporting.

use math_audio_dsp::analysis::compute_average_response;

use crate::Curve;

/// Log-frequency-weighted mean response inside an optional frequency band.
pub fn mean_response_in_range(curve: &Curve, min_freq: f64, max_freq: f64) -> f64 {
    let frequencies: Vec<f32> = curve
        .freq
        .iter()
        .map(|&frequency| frequency as f32)
        .collect();
    let levels: Vec<f32> = curve.spl.iter().map(|&level| level as f32).collect();
    compute_average_response(
        &frequencies,
        &levels,
        Some((min_freq as f32, max_freq as f32)),
    ) as f64
}

/// Detect the reproducible response band and its unsmoothed mean level.
pub fn detect_passband_and_mean(curve: &Curve) -> (Option<(f64, f64)>, f64) {
    let frequencies: Vec<f32> = curve
        .freq
        .iter()
        .map(|&frequency| frequency as f32)
        .collect();
    let levels: Vec<f32> = curve.spl.iter().map(|&level| level as f32).collect();
    if levels.len() < 2 {
        return (None, 0.0);
    }

    let smoothed = autoeq_core::smooth_one_over_n_octave(curve, 1);
    let smoothed_levels: Vec<f32> = smoothed.spl.iter().map(|&level| level as f32).collect();
    let reference_level = compute_average_response(&frequencies, &smoothed_levels, None);
    if !reference_level.is_finite() || reference_level < -100.0 {
        return (None, 0.0);
    }
    let threshold = reference_level - 10.0;
    let first_above = smoothed_levels.iter().position(|&level| level >= threshold);
    let last_above = smoothed_levels
        .iter()
        .rposition(|&level| level >= threshold);
    let (start_index, end_index) = match (first_above, last_above) {
        (Some(start), Some(end)) if end > start => (start, end),
        _ => return (None, 0.0),
    };

    let low = if start_index > 0 {
        interpolate_threshold_crossing(
            frequencies[start_index - 1],
            frequencies[start_index],
            smoothed_levels[start_index - 1],
            smoothed_levels[start_index],
            threshold,
        )
    } else {
        frequencies[start_index]
    };
    let high = if end_index + 1 < smoothed_levels.len() {
        interpolate_threshold_crossing(
            frequencies[end_index],
            frequencies[end_index + 1],
            smoothed_levels[end_index],
            smoothed_levels[end_index + 1],
            threshold,
        )
    } else {
        frequencies[end_index]
    };
    let mean = compute_average_response(&frequencies, &levels, Some((low, high))) as f64;
    (Some((f64::from(low), f64::from(high))), mean)
}

fn interpolate_threshold_crossing(
    f0: f32,
    f1: f32,
    level0: f32,
    level1: f32,
    threshold: f32,
) -> f32 {
    let denominator = level1 - level0;
    if denominator.abs() < 1e-9 {
        return f0;
    }
    let position = ((threshold - level0) / denominator).clamp(0.0, 1.0);
    f0 + position * (f1 - f0)
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;

    #[test]
    fn range_mean_uses_only_the_requested_band() {
        let curve = Curve {
            freq: Array1::from_vec(vec![20.0, 100.0, 1_000.0, 10_000.0]),
            spl: Array1::from_vec(vec![40.0, 80.0, 80.0, 20.0]),
            ..Curve::default()
        };

        assert!((mean_response_in_range(&curve, 100.0, 1_000.0) - 80.0).abs() < 1e-6);
    }

    #[test]
    fn passband_detection_ignores_interior_dips() {
        let curve = Curve {
            freq: Array1::from_vec(vec![20.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1_000.0]),
            spl: Array1::from_vec(vec![40.0, 70.0, 80.0, 60.0, 80.0, 70.0, 40.0]),
            ..Curve::default()
        };

        let (range, mean) = detect_passband_and_mean(&curve);
        let (low, high) = range.expect("passband");
        assert!(low < 100.0);
        assert!(high > 200.0);
        assert!(mean.is_finite());
    }

    #[test]
    fn threshold_interpolation_is_clamped_and_handles_flat_segments() {
        assert!(
            (interpolate_threshold_crossing(100.0, 200.0, 0.0, 10.0, 5.0) - 150.0).abs() < 1e-3
        );
        assert_eq!(
            interpolate_threshold_crossing(100.0, 200.0, 0.0, 10.0, -5.0),
            100.0
        );
        assert_eq!(
            interpolate_threshold_crossing(100.0, 200.0, 0.0, 10.0, 15.0),
            200.0
        );
        assert_eq!(
            interpolate_threshold_crossing(100.0, 200.0, 5.0, 5.0, 5.0),
            100.0
        );
    }
}
