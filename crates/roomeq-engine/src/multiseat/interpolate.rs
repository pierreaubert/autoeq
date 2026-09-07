use super::misc::find_bracket_indices;
use super::multi_seat_measurements::MultiSeatMeasurements;
use crate::Curve;
use crate::error::{AutoeqError, Result};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_analysis::listening_area::InterpolatedResponse;

/// Convert evidence-interpolated curves to complex pressure on `freqs`.
///
/// Bins with low phasor resultant (directional phase disagreement between
/// calibration positions) are downweighted by that resultant instead of
/// contributing a full-level phasor at the circular-mean angle (F06). A
/// midpoint between opposed 0°/180° measurements therefore contributes ~zero
/// pressure rather than a precise-looking full-level 90° transfer the
/// optimizer could exploit.
///
/// Returns the per-sub complex responses plus the fraction of reference-grid
/// bins flagged phase-ambiguous across subs, for coverage disclosure.
pub(super) fn evidence_response_to_complex(
    response: &InterpolatedResponse,
    freqs: &Array1<f64>,
) -> Result<(Vec<Vec<Complex64>>, f64)> {
    let mut out = Vec::with_capacity(response.curves.len());
    let mut flagged = 0usize;
    let mut total = 0usize;
    for (sub_idx, curve) in response.curves.iter().enumerate() {
        let mut complex = interpolate_curve_to_grid(curve, freqs)?;
        let confidence =
            response
                .confidence
                .get(sub_idx)
                .ok_or_else(|| AutoeqError::InvalidMeasurement {
                    message: "continuous_area interpolation evidence is missing confidence for a sub"
                        .to_string(),
                })?;
        for (freq_idx, &frequency) in freqs.iter().enumerate() {
            let resultant = interpolate_resultant(&curve.freq, confidence, frequency);
            complex[freq_idx] *= resultant.clamp(0.0, 1.0);
        }
        if let Some(flags) = response.phase_ambiguous.get(sub_idx) {
            flagged += flags.iter().filter(|flag| **flag).count();
            total += flags.len();
        }
        out.push(complex);
    }
    let ambiguous_fraction = if total > 0 {
        flagged as f64 / total as f64
    } else {
        0.0
    };
    Ok((out, ambiguous_fraction))
}

/// Log-frequency interpolation of a phasor resultant onto one eval frequency.
fn interpolate_resultant(
    reference: &Array1<f64>,
    confidence: &Array1<f64>,
    frequency: f64,
) -> f64 {
    if reference.len() != confidence.len() || reference.is_empty() {
        return 0.0;
    }
    let (lower_idx, upper_idx) = find_bracket_indices(reference, frequency);
    let f_low = reference[lower_idx];
    let f_high = reference[upper_idx];
    let t = if f_high > f_low && f_low > 0.0 && frequency > 0.0 {
        ((frequency.ln() - f_low.ln()) / (f_high.ln() - f_low.ln())).clamp(0.0, 1.0)
    } else if f_high > f_low {
        ((frequency - f_low) / (f_high - f_low)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    confidence[lower_idx] + t * (confidence[upper_idx] - confidence[lower_idx])
}

/// Interpolate all measurements to a common frequency grid
pub(super) fn interpolate_all_measurements(
    measurements: &MultiSeatMeasurements,
    freqs: &Array1<f64>,
) -> Result<Vec<Vec<Vec<Complex64>>>> {
    let mut result = Vec::new();

    for sub_measurements in &measurements.measurements {
        let mut sub_interp = Vec::new();
        for curve in sub_measurements {
            let interp = interpolate_curve_to_grid(curve, freqs)?;
            sub_interp.push(interp);
        }
        result.push(sub_interp);
    }

    Ok(result)
}

/// Interpolate a single curve to the common frequency grid
pub(super) fn interpolate_curve_to_grid(
    curve: &Curve,
    freqs: &Array1<f64>,
) -> Result<Vec<Complex64>> {
    let phase = curve
        .phase
        .as_ref()
        .ok_or_else(|| AutoeqError::InvalidMeasurement {
            message: "Multi-seat subwoofer optimization requires phase data for every sub/seat measurement; refusing to assume 0° phase for complex summation".to_string(),
        })?;

    let mut result = Vec::with_capacity(freqs.len());

    for &f in freqs.iter() {
        // Find bracketing indices
        let (lower_idx, upper_idx) = find_bracket_indices(&curve.freq, f);

        // Log-frequency interpolation for SPL and phase. Measurement grids are
        // commonly log-spaced, and this keeps low-frequency midpoints centered
        // perceptually and numerically.
        let f_low = curve.freq[lower_idx];
        let f_high = curve.freq[upper_idx];
        let t = if f_high > f_low && f_low > 0.0 && f > 0.0 {
            (f.ln() - f_low.ln()) / (f_high.ln() - f_low.ln())
        } else if f_high > f_low {
            (f - f_low) / (f_high - f_low)
        } else {
            0.0
        };

        let spl_interp = curve.spl[lower_idx] + t * (curve.spl[upper_idx] - curve.spl[lower_idx]);

        // Interpolate phase with wrap handling (shortest arc through ±180°)
        let mut diff = phase[upper_idx] - phase[lower_idx];
        diff -= 360.0 * (diff / 360.0).round();
        let phase_rad = (phase[lower_idx] + t * diff).to_radians();

        let magnitude = 10.0_f64.powf(spl_interp / 20.0);
        result.push(Complex64::from_polar(magnitude, phase_rad));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_analysis::listening_area::{ListeningArea, ListeningAreaInterpolatorConfig};

    fn opposed_area() -> ListeningArea<1> {
        let freq = Array1::from(vec![40.0, 80.0, 160.0]);
        let mk = |phase_deg: f64| Curve {
            freq: freq.clone(),
            spl: Array1::from_elem(3, 80.0),
            phase: Some(Array1::from_elem(3, phase_deg)),
            ..Curve::default()
        };
        ListeningArea::new(
            vec![[0.0], [1.0]],
            vec![vec![mk(0.0), mk(180.0)]],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("opposed calibration area")
    }

    #[test]
    fn opposed_phases_yield_near_zero_confident_pressure() {
        // F06: midpoint between 0°/180° positions must not present a
        // full-level phantom transfer to the optimizer.
        let area = opposed_area();
        let evidence = area
            .interpolate_with_evidence([0.5])
            .expect("midpoint is in support");
        assert!(
            evidence.confidence[0].iter().all(|&r| r < 1e-6),
            "opposed phases must report ~zero resultant, got {:?}",
            evidence.confidence[0]
        );
        assert!(evidence.phase_ambiguous[0].iter().all(|&b| b));

        let legacy = &area.interpolate_at([0.5])[0];
        let legacy_level =
            interpolate_curve_to_grid(legacy, &Array1::from(vec![80.0])).expect("legacy grid")[0]
                .norm();
        assert!(
            (20.0 * legacy_level.log10() - 80.0).abs() < 1.0,
            "legacy path should return a full-level (~80 dB) phantom, got {} dB",
            20.0 * legacy_level.log10()
        );

        let (complex, fraction) =
            evidence_response_to_complex(&evidence, &Array1::from(vec![80.0]))
                .expect("evidence conversion");
        assert!(
            complex[0][0].norm() < 1e-6,
            "ambiguous midpoint must contribute ~zero pressure, got {}",
            complex[0][0].norm()
        );
        assert!(
            (fraction - 1.0).abs() < 1e-12,
            "all bins ambiguous, fraction={fraction}"
        );
    }

    #[test]
    fn agreeing_phases_pass_through_unattenuated() {
        let freq = Array1::from(vec![40.0, 80.0, 160.0]);
        let mk = || Curve {
            freq: freq.clone(),
            spl: Array1::from_elem(3, 80.0),
            phase: Some(Array1::from_elem(3, 0.0)),
            ..Curve::default()
        };
        let area = ListeningArea::new(
            vec![[0.0], [1.0]],
            vec![vec![mk(), mk()]],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("agreeing calibration area");
        let evidence = area
            .interpolate_with_evidence([0.5])
            .expect("midpoint is in support");
        let (complex, fraction) =
            evidence_response_to_complex(&evidence, &Array1::from(vec![80.0]))
                .expect("evidence conversion");
        let expected = 10.0_f64.powf(80.0 / 20.0);
        assert!(
            (complex[0][0].norm() - expected).abs() / expected < 1e-9,
            "agreeing midpoint must be unattenuated, got {}",
            complex[0][0].norm()
        );
        assert!(fraction.abs() < 1e-12, "no bins ambiguous, fraction={fraction}");
    }
}
