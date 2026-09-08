use autoeq_core::{Curve, interpolate_log_space, response};
use log::info;
use math_audio_dsp::analysis::compute_average_response;
use math_audio_iir_fir::Biquad;
use std::collections::HashMap;

pub fn compute_crossover_complex_response(
    type_str: &str,
    freq: f64,
    sample_rate: f64,
    is_lowpass: bool,
    freqs: &ndarray::Array1<f64>,
) -> Vec<num_complex::Complex64> {
    if is_linear_phase_crossover_type(type_str) {
        let coeffs = linear_phase_crossover_coefficients(freq, sample_rate, is_lowpass);
        response::compute_fir_complex_response(&coeffs, freqs, sample_rate)
    } else {
        let filters = create_crossover_filters(type_str, freq, sample_rate, is_lowpass);
        response::compute_peq_complex_response(&filters, freqs, sample_rate)
    }
}

/// Align channel levels by normalizing down to the lowest level.
pub fn align_channels_to_lowest(
    channels: &HashMap<String, Curve>,
    ranges: &HashMap<String, (f64, f64)>,
) -> HashMap<String, f64> {
    let mut means = HashMap::new();
    let mut min_mean = f64::INFINITY;

    for (name, curve) in channels {
        let (min_f, max_f) = ranges.get(name).cloned().unwrap_or((100.0, 2000.0));

        let freqs_f32: Vec<f32> = curve.freq.iter().map(|&f| f as f32).collect();
        let spl_f32: Vec<f32> = curve.spl.iter().map(|&s| s as f32).collect();

        let mean =
            compute_average_response(&freqs_f32, &spl_f32, Some((min_f as f32, max_f as f32)))
                as f64;

        means.insert(name.clone(), mean);
        if mean < min_mean {
            min_mean = mean;
        }
    }

    let mut gains = HashMap::new();
    for (name, mean) in means {
        let diff = min_mean - mean;
        gains.insert(name.clone(), diff);
        info!(
            "  Level alignment for '{}': {:.2} dB (mean {:.2} -> {:.2})",
            name, diff, mean, min_mean
        );
    }
    gains
}

/// Coherent (complex) sum of N main channels, used by the stereo-2.1 and
/// home-cinema-with-sub crossover optimizers.
///
/// The previous per-bin SPL average with a discarded/averaged phase hid
/// inter-channel phase mismatches from the crossover / group-delay loss
/// (B8). Using the complex sum preserves phase coherence the same way
/// `preprocess_cardioid` does for the front/rear sub pair.
///
/// Callers must only use this for curves with measured/current phase. Bass
/// management is phase-critical, so the workflows skip delay/polarity
/// optimization instead of inventing 0 deg phase.
///
/// Expects every input curve to share the same frequency grid. Empty input
/// panics — callers always supply ≥ 1 main.
pub fn complex_sum_mains(curves: &[&Curve]) -> Curve {
    use num_complex::Complex;
    assert!(!curves.is_empty(), "complex_sum_mains needs ≥ 1 curve");
    let freq = curves[0].freq.clone();
    let n = freq.len();
    let aligned = curves
        .iter()
        .map(|curve| autoeq_core::curve_transforms::interpolate_log_space(&freq, curve))
        .collect::<Vec<_>>();

    let mut spl = ndarray::Array1::<f64>::zeros(n);
    let mut phase = ndarray::Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = Complex::new(0.0_f64, 0.0);
        for c in &aligned {
            let mag = 10.0_f64.powf(c.spl[i] / 20.0);
            let phi = c.phase.as_ref().expect("phase checked by caller")[i].to_radians();
            sum += Complex::from_polar(mag, phi);
        }
        spl[i] = 20.0 * sum.norm().max(1e-12).log10();
        phase[i] = sum.arg().to_degrees();
    }
    // Unwrap so downstream processing (e.g., GD computation, delay estimation)
    // sees a continuous phase curve rather than [-180, 180] discontinuities.
    phase = autoeq_core::phase_utils::unwrap_phase_degrees(&phase);

    Curve {
        freq,
        spl,
        phase: Some(phase),
        ..Default::default()
    }
}

pub fn average_mains_magnitude(curves: &[&Curve]) -> Curve {
    assert!(
        !curves.is_empty(),
        "average_mains_magnitude needs >= 1 curve"
    );
    let ref_freq = curves[0].freq.clone();
    let mut spl = ndarray::Array1::<f64>::zeros(ref_freq.len());

    for curve in curves {
        let interpolated = interpolate_log_space(&ref_freq, curve);
        spl += &interpolated.spl;
    }
    spl.mapv_inplace(|v| v / curves.len() as f64);

    Curve {
        freq: ref_freq,
        spl,
        phase: None,
        ..Default::default()
    }
}

pub fn curve_has_usable_phase(curve: &Curve) -> bool {
    curve
        .phase
        .as_ref()
        .map(|phase| phase.len() >= curve.freq.len() && phase.iter().all(|v| v.is_finite()))
        .unwrap_or(false)
}

pub fn normalize_crossover_delays(main_delay_ms: f64, sub_delay_ms: f64) -> (f64, f64) {
    let common_delay_ms = main_delay_ms.min(sub_delay_ms);
    (
        main_delay_ms - common_delay_ms,
        sub_delay_ms - common_delay_ms,
    )
}

pub fn create_crossover_filters(
    type_str: &str,
    freq: f64,
    sample_rate: f64,
    is_lowpass: bool,
) -> Vec<Biquad> {
    use math_audio_iir_fir::*;
    if is_linear_phase_crossover_type(type_str) {
        return Vec::new();
    }
    let crossover_type = type_str.parse::<roomeq_model::CrossoverType>();
    let peq = match crossover_type {
        Ok(roomeq_model::CrossoverType::LinkwitzRiley2) => {
            if is_lowpass {
                peq_linkwitzriley_lowpass(2, freq, sample_rate)
            } else {
                peq_linkwitzriley_highpass(2, freq, sample_rate)
            }
        }
        Ok(roomeq_model::CrossoverType::LinkwitzRiley4) => {
            if is_lowpass {
                peq_linkwitzriley_lowpass(4, freq, sample_rate)
            } else {
                peq_linkwitzriley_highpass(4, freq, sample_rate)
            }
        }
        Ok(roomeq_model::CrossoverType::LinkwitzRiley8) => {
            if is_lowpass {
                peq_linkwitzriley_lowpass(8, freq, sample_rate)
            } else {
                peq_linkwitzriley_highpass(8, freq, sample_rate)
            }
        }
        Ok(roomeq_model::CrossoverType::Butterworth2) => {
            if is_lowpass {
                peq_butterworth_lowpass(2, freq, sample_rate)
            } else {
                peq_butterworth_highpass(2, freq, sample_rate)
            }
        }
        Ok(roomeq_model::CrossoverType::Butterworth4) => {
            if is_lowpass {
                peq_butterworth_lowpass(4, freq, sample_rate)
            } else {
                peq_butterworth_highpass(4, freq, sample_rate)
            }
        }
        Ok(roomeq_model::CrossoverType::LinearPhase | roomeq_model::CrossoverType::None) => {
            Vec::new()
        }
        Err(_) => {
            log::warn!("Unknown crossover type '{}', defaulting to LR24", type_str);
            if is_lowpass {
                peq_linkwitzriley_lowpass(4, freq, sample_rate)
            } else {
                peq_linkwitzriley_highpass(4, freq, sample_rate)
            }
        }
    };
    peq.into_iter().map(|(_, b)| b).collect()
}

pub fn is_linear_phase_crossover_type(type_str: &str) -> bool {
    matches!(
        type_str.to_ascii_lowercase().as_str(),
        "linearphase" | "linear_phase" | "linear-phase" | "linearphasefir" | "fir" | "lpfir"
    )
}

pub fn linear_phase_crossover_coefficients(
    freq: f64,
    sample_rate: f64,
    is_lowpass: bool,
) -> Vec<f64> {
    let crossover = math_audio_iir_fir::FirCrossover::new(
        freq,
        sample_rate,
        1,
        math_audio_iir_fir::DEFAULT_FIR_CROSSOVER_TAPS,
    );
    if is_lowpass {
        crossover.lowpass_coefficients().to_vec()
    } else {
        crossover.highpass_coefficients()
    }
}

#[cfg(test)]
mod complex_sum_tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn aligns_mismatched_frequency_grids_before_complex_sum() {
        let make_curve = |count| Curve {
            freq: Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), count),
            spl: Array1::zeros(count),
            phase: Some(Array1::zeros(count)),
            ..Default::default()
        };
        let main = make_curve(100);
        let bass = make_curve(333);

        let sum = complex_sum_mains(&[&main, &bass]);

        assert_eq!(sum.freq, main.freq);
        assert_eq!(sum.spl.len(), 100);
        assert!(sum.spl.iter().all(|value| (*value - 6.0206).abs() < 1e-3));
    }
}

/// Interpolate a bass measurement without clipping the receiving main's grid.
/// Outside a demonstrably rolled-off tail, continue only a falling envelope.
/// An energetic endpoint is held conservatively; it is never assumed silent.
pub fn interpolate_bass_response(frequencies: &ndarray::Array1<f64>, curve: &Curve) -> Curve {
    let mut result = autoeq_core::interpolate_log_space(frequencies, curve);
    if curve.freq.len() < 2 || curve.spl.len() != curve.freq.len() {
        return result;
    }
    let last = curve.freq.len() - 1;
    let high = curve.freq[last];
    let peak = curve.spl.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let previous = curve
        .freq
        .iter()
        .rposition(|&frequency| frequency <= high / 2.0_f64.sqrt())
        .unwrap_or(0);
    let octaves = (high / curve.freq[previous]).log2();
    if octaves <= 0.0 {
        return result;
    }
    let slope = (curve.spl[last] - curve.spl[previous]) / octaves;
    let falling_tail = curve.spl[last] <= peak - 24.0 && slope <= -12.0;
    for (index, &frequency) in frequencies.iter().enumerate() {
        if frequency > high {
            result.spl[index] = if falling_tail {
                (curve.spl[last] + slope.max(-48.0) * (frequency / high).log2()).max(-240.0)
            } else {
                curve.spl[last]
            };
            if let (Some(phase), Some(original)) = (result.phase.as_mut(), curve.phase.as_ref()) {
                phase[index] = original[last];
            }
        }
    }
    result
}

#[cfg(test)]
mod limited_bass_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn short_rolled_off_sub_keeps_full_range_main_analysis() {
        let main = Curve {
            freq: array![50.0, 100.0, 200.0, 1000.0, 16000.0],
            spl: ndarray::Array1::from_elem(5, 80.0),
            phase: Some(ndarray::Array1::from_elem(5, 0.0)),
            ..Default::default()
        };
        let sub = Curve {
            freq: array![50.0, 100.0, 140.0, 200.0],
            spl: array![80.0, 75.0, 66.0, 50.0],
            phase: Some(ndarray::Array1::from_elem(4, 0.0)),
            ..Default::default()
        };
        let extended = interpolate_bass_response(&main.freq, &sub);
        assert_eq!(extended.spl[2], 50.0);
        assert!(extended.spl[4] < -100.0);
        let sum = complex_sum_mains(&[&main, &extended]);
        assert_eq!(sum.freq, main.freq);
        assert!((sum.spl[4] - 80.0).abs() < 0.001);
    }

    #[test]
    fn energetic_sub_endpoint_is_not_invented_as_silent() {
        let sub = Curve {
            freq: array![50.0, 100.0, 200.0],
            spl: ndarray::Array1::from_elem(3, 80.0),
            ..Default::default()
        };
        let extended = interpolate_bass_response(&array![200.0, 1000.0, 16000.0], &sub);
        assert!(extended.spl.iter().all(|&level| level == 80.0));
    }
}

/// Preserve every measured frequency in the common supported span before
/// summing a physical array. Source order must not choose its resolution.
pub fn shared_measurement_grid(curves: &[&Curve]) -> Option<ndarray::Array1<f64>> {
    let (low, high) =
        roomeq_analysis::frequency_grid::common_frequency_range(curves.iter().copied())?;
    let mut frequencies: Vec<_> = curves
        .iter()
        .flat_map(|curve| curve.freq.iter().copied())
        .filter(|&frequency| frequency >= low && frequency <= high)
        .collect();
    frequencies.sort_by(f64::total_cmp);
    frequencies.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    Some(ndarray::Array1::from_vec(frequencies))
}

/// Keep the main's full measured span and every available bass sample within it.
/// A limited-band sub must neither truncate the main nor set its resolution.
pub fn bass_management_measurement_grid(main: &Curve, sources: &[&Curve]) -> ndarray::Array1<f64> {
    let (Some(&low), Some(&high)) = (main.freq.first(), main.freq.last()) else {
        return main.freq.clone();
    };
    let mut frequencies = main.freq.to_vec();
    frequencies.extend(
        sources
            .iter()
            .flat_map(|source| source.freq.iter().copied())
            .filter(|frequency| frequency.is_finite() && *frequency >= low && *frequency <= high),
    );
    frequencies.sort_by(f64::total_cmp);
    frequencies.dedup();
    ndarray::Array1::from_vec(frequencies)
}

#[cfg(test)]
mod shared_grid_tests {
    use super::*;
    #[test]
    fn crossover_grid_retains_fine_sub_notch_and_full_range_main() {
        let main = Curve {
            freq: ndarray::array![20.0, 80.0, 200.0, 1000.0, 16000.0],
            ..Default::default()
        };
        let sub = Curve {
            freq: ndarray::array![10.0, 20.0, 49.9, 50.0, 50.1, 200.0],
            ..Default::default()
        };
        assert_eq!(
            bass_management_measurement_grid(&main, &[&sub]),
            ndarray::array![20.0, 49.9, 50.0, 50.1, 80.0, 200.0, 1000.0, 16000.0]
        );
    }
    #[test]
    fn source_order_does_not_discard_finer_sub_samples() {
        let coarse = Curve {
            freq: ndarray::array![20.0, 80.0, 200.0],
            ..Default::default()
        };
        let fine = Curve {
            freq: ndarray::array![20.0, 49.9, 50.0, 50.1, 200.0],
            ..Default::default()
        };
        let expected = ndarray::array![20.0, 49.9, 50.0, 50.1, 80.0, 200.0];
        assert_eq!(
            shared_measurement_grid(&[&coarse, &fine]).unwrap(),
            expected
        );
        assert_eq!(
            shared_measurement_grid(&[&fine, &coarse]).unwrap(),
            expected
        );
    }
}
