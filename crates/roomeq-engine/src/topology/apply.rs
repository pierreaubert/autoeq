use super::compute_crossover_complex_response;
use autoeq_core::{Curve, interpolate_log_space, response};

pub fn apply_delay_and_polarity_to_curve(curve: &Curve, delay_ms: f64, invert: bool) -> Curve {
    let mut adjusted = curve.clone();
    let Some(phase) = adjusted.phase.as_mut() else {
        return adjusted;
    };
    let delay_s = delay_ms / 1000.0;
    for (phase_deg, &freq_hz) in phase.iter_mut().zip(adjusted.freq.iter()) {
        *phase_deg -= 360.0 * freq_hz * delay_s;
        if invert {
            *phase_deg += 180.0;
        }
    }
    adjusted
}

pub fn apply_curve_delta_to_reference_curve(
    reference_curve: &Curve,
    initial_curve: &Curve,
    final_curve: &Curve,
) -> Curve {
    let initial_on_reference = interpolate_log_space(&reference_curve.freq, initial_curve);
    let final_on_reference = interpolate_log_space(&reference_curve.freq, final_curve);
    let phase = match (
        reference_curve.phase.as_ref(),
        initial_on_reference.phase.as_ref(),
        final_on_reference.phase.as_ref(),
    ) {
        (Some(reference_phase), Some(initial_phase), Some(final_phase)) => {
            Some(reference_phase + &(final_phase - initial_phase))
        }
        _ => reference_curve.phase.clone(),
    };
    Curve {
        freq: reference_curve.freq.clone(),
        spl: &reference_curve.spl + &(&final_on_reference.spl - &initial_on_reference.spl),
        phase,
        ..Default::default()
    }
}

pub fn apply_crossover_response_to_curve(
    curve: &Curve,
    type_str: &str,
    freq: f64,
    sample_rate: f64,
    is_lowpass: bool,
) -> Curve {
    let resp =
        compute_crossover_complex_response(type_str, freq, sample_rate, is_lowpass, &curve.freq);
    response::apply_complex_response_with_min_db(
        curve,
        &resp,
        response::MIN_REALIZATION_RESPONSE_DB,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn delay_handles_phase_vectors_longer_than_frequency_grid() {
        let curve = Curve {
            freq: Array1::from(vec![100.0, 200.0]),
            spl: Array1::zeros(2),
            phase: Some(Array1::from(vec![0.0, 0.0, 123.0])),
            ..Default::default()
        };

        let adjusted = apply_delay_and_polarity_to_curve(&curve, 1.0, false);
        let phase = adjusted.phase.unwrap();
        assert_eq!(phase[0], -36.0);
        assert_eq!(phase[1], -72.0);
        assert_eq!(phase[2], 123.0);
    }
}
