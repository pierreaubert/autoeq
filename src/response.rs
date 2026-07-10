//! Frequency response calculation for filters

use crate::Curve;
use crate::iir::Biquad;
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Lowest response magnitude exposed to optimization and reporting.
pub const MIN_FILTER_RESPONSE_DB: f64 = -40.0;

/// Compute complex frequency response of a list of Biquad filters
pub fn compute_peq_complex_response(
    filters: &[Biquad],
    freqs: &Array1<f64>,
    sample_rate: f64,
) -> Vec<Complex64> {
    // `Biquad` owns the canonical coefficient generation for every filter
    // type. Cache those normalized coefficients once instead of maintaining a
    // second set of formulas here (and recomputing them for every frequency).
    let coefficients: Vec<_> = filters.iter().map(Biquad::constants).collect();

    freqs
        .iter()
        .map(|&f| {
            let w = 2.0 * PI * f / sample_rate;
            let z_inv = Complex64::from_polar(1.0, -w);
            let z_inv_2 = z_inv * z_inv;

            let mut total_h = Complex64::new(1.0, 0.0);

            for &(a1, a2, b0, b1, b2) in &coefficients {
                let num = b0 + b1 * z_inv + b2 * z_inv_2;
                let den = 1.0 + a1 * z_inv + a2 * z_inv_2;

                if den.norm_sqr() > 1e-12 {
                    total_h *= num / den;
                }
            }
            total_h
        })
        .collect()
}

/// Compute complex frequency response of FIR coefficients
pub fn compute_fir_complex_response(
    coeffs: &[f64],
    freqs: &Array1<f64>,
    sample_rate: f64,
) -> Vec<Complex64> {
    // Direct DFT calculation (O(N*M))
    // Appropriate for evaluation at specific log-spaced frequencies
    freqs
        .iter()
        .map(|&f| {
            let w = 2.0 * PI * f / sample_rate;
            let mut h = Complex64::new(0.0, 0.0);

            for (n, &val) in coeffs.iter().enumerate() {
                let angle = -w * n as f64;
                h += Complex64::from_polar(val, angle);
            }
            h
        })
        .collect()
}

/// Apply complex filter response (magnitude and phase) to a curve
pub fn apply_complex_response(curve: &Curve, response: &[Complex64]) -> Curve {
    if response.len() != curve.freq.len() {
        log::warn!(
            "Complex response length {} does not match curve length {}; unmatched bins are preserved",
            response.len(),
            curve.freq.len()
        );
    }

    let mut new_spl = curve.spl.clone();
    let old_phase = curve.phase.as_ref();
    let mut new_phase = old_phase
        .cloned()
        .unwrap_or_else(|| Array1::zeros(curve.freq.len()));

    for (i, &h) in response.iter().take(curve.freq.len()).enumerate() {
        let h_mag_db = (20.0 * h.norm().log10()).max(MIN_FILTER_RESPONSE_DB);
        let h_phase_deg = h.arg().to_degrees();

        new_spl[i] = curve.spl[i] + h_mag_db;
        let p_in = old_phase.map(|p| p[i]).unwrap_or(0.0);
        new_phase[i] = p_in + h_phase_deg;
    }

    Curve {
        freq: curve.freq.clone(),
        spl: new_spl,
        phase: Some(new_phase),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iir::BiquadFilterType;
    use ndarray::Array1;
    use num_complex::Complex64;
    use std::f64::consts::PI;

    #[test]
    fn test_fir_response_impulse() {
        // Impulse response [1.0] should have flat magnitude (0 dB) and 0 phase
        let coeffs = vec![1.0];
        let freqs = Array1::from(vec![100.0, 1000.0, 10000.0]);
        let sample_rate = 48000.0;

        let response = compute_fir_complex_response(&coeffs, &freqs, sample_rate);

        for h in response {
            assert!((h.norm() - 1.0).abs() < 1e-10);
            assert!(h.arg().abs() < 1e-10);
        }
    }

    #[test]
    fn test_fir_response_delay() {
        // Delayed impulse [0.0, 1.0] should have flat magnitude and linear phase
        let coeffs = vec![0.0, 1.0];
        let freqs = Array1::from(vec![100.0, 1000.0, 10000.0]);
        let sample_rate = 48000.0;

        let response = compute_fir_complex_response(&coeffs, &freqs, sample_rate);

        for (i, h) in response.iter().enumerate() {
            assert!((h.norm() - 1.0).abs() < 1e-10);

            // Phase should be -w * delay
            // delay = 1 sample
            let w = 2.0 * PI * freqs[i] / sample_rate;
            let expected_phase = -w;

            // Normalize phase to [-pi, pi]
            let phase = h.arg();
            let mut diff = (phase - expected_phase).abs();
            while diff > PI {
                diff -= 2.0 * PI;
            }
            assert!(diff.abs() < 1e-10);
        }
    }

    #[test]
    fn test_peq_empty_filters_identity() {
        let filters: Vec<crate::iir::Biquad> = vec![];
        let freqs = Array1::from(vec![100.0, 1000.0, 10000.0]);
        let sr = 48000.0;
        let resp = compute_peq_complex_response(&filters, &freqs, sr);
        assert_eq!(resp.len(), freqs.len());
        for h in &resp {
            assert!((h.norm() - 1.0).abs() < 1e-12);
            assert!(h.arg().abs() < 1e-12);
        }
    }

    fn response_from_stored_coefficients(
        biquad: &Biquad,
        frequency: f64,
        sample_rate: f64,
    ) -> Complex64 {
        let (a1, a2, b0, b1, b2) = biquad.constants();
        let z_inv = Complex64::from_polar(1.0, -2.0 * PI * frequency / sample_rate);
        let z_inv_2 = z_inv * z_inv;
        (b0 + b1 * z_inv + b2 * z_inv_2) / (1.0 + a1 * z_inv + a2 * z_inv_2)
    }

    #[test]
    fn test_peq_response_uses_coefficients_for_every_biquad_type() {
        let sample_rate = 48_000.0;
        let frequencies = Array1::from(vec![500.0, 1_000.0, 2_000.0]);
        let filter_types = [
            BiquadFilterType::Bandpass,
            BiquadFilterType::Notch,
            BiquadFilterType::AllPass,
            BiquadFilterType::LowshelfOrf,
            BiquadFilterType::HighshelfOrf,
            BiquadFilterType::PeakMatched,
            BiquadFilterType::HighpassVariableQ,
        ];

        for filter_type in filter_types {
            let biquad = Biquad::new(filter_type, 1_000.0, sample_rate, 1.1, 6.0);
            let actual = compute_peq_complex_response(
                std::slice::from_ref(&biquad),
                &frequencies,
                sample_rate,
            );

            for (&frequency, actual) in frequencies.iter().zip(actual) {
                let expected = response_from_stored_coefficients(&biquad, frequency, sample_rate);
                assert!(
                    (actual - expected).norm() < 1e-12,
                    "{filter_type:?} response mismatch at {frequency} Hz: actual={actual:?}, expected={expected:?}"
                );
            }
        }
    }

    #[test]
    fn test_apply_complex_response_identity() {
        let curve = Curve {
            freq: Array1::from(vec![100.0, 1000.0]),
            spl: Array1::from(vec![0.0, 10.0]),
            phase: None,
            ..Default::default()
        };
        let identity = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let out = apply_complex_response(&curve, &identity);
        assert!((out.spl[0] - 0.0).abs() < 1e-12);
        assert!((out.spl[1] - 10.0).abs() < 1e-12);
        assert!(out.phase.is_some());
        assert!(out.phase.unwrap()[0].abs() < 1e-12);
    }

    #[test]
    fn test_apply_complex_response_boost() {
        let curve = Curve {
            freq: Array1::from(vec![1000.0]),
            spl: Array1::from(vec![0.0]),
            phase: None,
            ..Default::default()
        };
        let boost = vec![Complex64::new(2.0, 0.0)];
        let out = apply_complex_response(&curve, &boost);
        let expected_db = 20.0 * 2.0f64.log10();
        assert!(
            (out.spl[0] - expected_db).abs() < 1e-9,
            "got {}",
            out.spl[0]
        );
    }

    #[test]
    fn audit_zero_complex_response_is_limited_to_minus_40_db() {
        let curve = Curve {
            freq: Array1::from(vec![1000.0]),
            spl: Array1::from(vec![0.0]),
            phase: None,
            ..Default::default()
        };
        let out = apply_complex_response(&curve, &[Complex64::new(0.0, 0.0)]);

        assert!(out.spl[0].is_finite());
        assert_eq!(out.spl[0], -40.0);
    }

    #[test]
    fn audit_mismatched_complex_response_preserves_unmatched_bins() {
        let curve = Curve {
            freq: Array1::from(vec![100.0, 1000.0]),
            spl: Array1::from(vec![1.0, 2.0]),
            phase: None,
            ..Default::default()
        };
        let outcome = std::panic::catch_unwind(|| {
            apply_complex_response(&curve, &[Complex64::new(1.0, 0.0)])
        });

        assert!(outcome.is_ok(), "mismatched response must not panic");
        let out = outcome.unwrap();
        assert_eq!(out.spl.to_vec(), vec![1.0, 2.0]);
    }
}
