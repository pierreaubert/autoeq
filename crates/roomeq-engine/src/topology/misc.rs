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
    let n = curves.iter().map(|c| c.spl.len()).min().unwrap();
    let freq = ndarray::Array1::from_iter(curves[0].freq.iter().take(n).copied());

    let mut spl = ndarray::Array1::<f64>::zeros(n);
    let mut phase = ndarray::Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = Complex::new(0.0_f64, 0.0);
        for c in curves {
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
