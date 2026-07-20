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

/// Detect a subwoofer's natural low and high -3 dB passband limits.
pub fn detect_sub_passband_3db(curve: &Curve) -> Option<(f64, f64)> {
    if curve.freq.len() < 4 || curve.spl.len() != curve.freq.len() {
        return None;
    }

    let smoothed = autoeq_core::smooth_one_over_n_octave(curve, 1);
    if smoothed.freq.len() < 4 {
        return None;
    }
    const SUB_SEARCH_LO_HZ: f64 = 10.0;
    const SUB_SEARCH_HI_HZ: f64 = 500.0;
    let mut peak_idx = None;
    let mut peak_spl = f64::NEG_INFINITY;
    for index in 0..smoothed.freq.len() {
        let frequency = smoothed.freq[index];
        if !(SUB_SEARCH_LO_HZ..=SUB_SEARCH_HI_HZ).contains(&frequency) {
            continue;
        }
        let level = smoothed.spl[index];
        if level.is_finite() && level > peak_spl {
            peak_spl = level;
            peak_idx = Some(index);
        }
    }
    let peak_idx = peak_idx?;

    const IN_BAND_TOLERANCE_DB: f64 = 2.0;
    let in_band_threshold = peak_spl - IN_BAND_TOLERANCE_DB;
    let mut in_lo = peak_idx;
    while in_lo > 0 && smoothed.spl[in_lo - 1] >= in_band_threshold {
        in_lo -= 1;
    }
    let mut in_hi = peak_idx;
    while in_hi + 1 < smoothed.spl.len() && smoothed.spl[in_hi + 1] >= in_band_threshold {
        in_hi += 1;
    }
    let frequencies = smoothed
        .freq
        .iter()
        .map(|&frequency| frequency as f32)
        .collect::<Vec<_>>();
    let levels = smoothed
        .spl
        .iter()
        .map(|&level| level as f32)
        .collect::<Vec<_>>();
    let band_average = compute_average_response(
        &frequencies,
        &levels,
        Some((smoothed.freq[in_lo] as f32, smoothed.freq[in_hi] as f32)),
    ) as f64;
    if !band_average.is_finite() {
        return None;
    }
    let threshold = band_average - 3.0;

    let mut low = smoothed.freq[0];
    let mut found_low = false;
    for index in (0..peak_idx).rev() {
        let level = smoothed.spl[index];
        if level <= threshold {
            let low_frequency = smoothed.freq[index];
            let high_frequency = smoothed.freq[index + 1];
            let high_level = smoothed.spl[index + 1];
            let denominator = high_level - level;
            low = if denominator.abs() > f64::EPSILON {
                let position = ((threshold - level) / denominator).clamp(0.0, 1.0);
                (low_frequency.ln() + position * (high_frequency.ln() - low_frequency.ln())).exp()
            } else {
                low_frequency
            };
            found_low = true;
            break;
        }
    }

    let mut high = smoothed.freq[smoothed.freq.len() - 1];
    let mut found_high = false;
    for index in (peak_idx + 1)..smoothed.spl.len() {
        let level = smoothed.spl[index];
        if level <= threshold {
            let low_frequency = smoothed.freq[index - 1];
            let high_frequency = smoothed.freq[index];
            let low_level = smoothed.spl[index - 1];
            let denominator = level - low_level;
            high = if denominator.abs() > f64::EPSILON {
                let position = ((threshold - low_level) / denominator).clamp(0.0, 1.0);
                (low_frequency.ln() + position * (high_frequency.ln() - low_frequency.ln())).exp()
            } else {
                high_frequency
            };
            found_high = true;
            break;
        }
    }

    ((found_low || found_high) && high > low).then_some((low, high))
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
    use math_audio_iir_fir::{Biquad, BiquadFilterType};
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

    #[test]
    fn sub_passband_detection_uses_the_unmodified_measurement_curve() {
        let freq = Array1::logspace(10.0, f64::log10(20.0), f64::log10(500.0), 64);
        let spl = freq
            .iter()
            .map(|&frequency| {
                if frequency < 200.0 {
                    80.0
                } else {
                    80.0 - 20.0 * ((frequency / 200.0).log2().max(0.0))
                }
            })
            .collect::<Vec<_>>();
        let raw_curve = Curve {
            freq,
            spl: Array1::from(spl),
            ..Curve::default()
        };
        let highpass = Biquad::new(BiquadFilterType::Highpass, 80.0, 48_000.0, 0.707, 0.0);
        let highpass_response = autoeq_core::response::compute_peq_complex_response(
            &[highpass],
            &raw_curve.freq,
            48_000.0,
        );
        let highpass_curve =
            autoeq_core::response::apply_complex_response(&raw_curve, &highpass_response);

        let raw_band = detect_sub_passband_3db(&raw_curve).unwrap();
        let highpass_band = detect_sub_passband_3db(&highpass_curve).unwrap();
        assert!(raw_band.0 < 40.0);
        assert!(highpass_band.0 > 50.0);
        assert!((raw_band.1 - highpass_band.1).abs() < 30.0);
    }
}
