//! DSP convention checks for core filter and phase primitives.
//!
//! Pins the sign conventions every higher layer relies on: positive PEQ gain
//! boosts, the `z^-1` recurrence behind normalized biquad coefficients, FIR
//! DC gain, phase-unwrap continuity, minimum-phase identity of flat spectra,
//! and fail-fast length agreement when applying complex responses. Method
//! repairs belong here; threshold relaxation does not.

// Rust guideline compliant 2026-02-21

use crate::iir::Biquad;
use crate::response::compute_peq_complex_response;
use ndarray::Array1;

/// Magnitude in dB of one PEQ filter at its center frequency.
pub fn peq_center_gain_db(filter: &Biquad, center_hz: f64, sample_rate_hz: f64) -> f64 {
    let freqs = Array1::from_vec(vec![center_hz]);
    let response =
        compute_peq_complex_response(std::slice::from_ref(filter), &freqs, sample_rate_hz);
    20.0 * response[0].norm().max(1e-12).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Curve;
    use crate::iir::BiquadFilterType;
    use crate::phase_utils::{reconstruct_minimum_phase, unwrap_phase_degrees};
    use crate::response::{apply_complex_response, compute_fir_complex_response};
    use num_complex::Complex64;

    fn peak_6db_at_1k() -> Biquad {
        Biquad::new(BiquadFilterType::Peak, 1_000.0, 48_000.0, 1.0, 6.0)
    }

    /// Positive PEQ gain boosts: +6 dB at center reads back +6 dB.
    #[test]
    fn peq_positive_gain_boosts_at_center() {
        let gain = peq_center_gain_db(&peak_6db_at_1k(), 1_000.0, 48_000.0);
        assert!((gain - 6.0).abs() < 1e-9, "center gain was {gain} dB");
        let cut = Biquad::new(BiquadFilterType::Peak, 1_000.0, 48_000.0, 1.0, -6.0);
        let cut_gain = peq_center_gain_db(&cut, 1_000.0, 48_000.0);
        assert!(
            (cut_gain + 6.0).abs() < 1e-9,
            "center cut was {cut_gain} dB"
        );
    }

    /// Coefficient signs follow the `z^-1` recurrence: the denominator
    /// evaluates as `1 + a1 z^-1 + a2 z^-2` with `z^-1 = e^{-jw}`, matching
    /// the response kernel. A DC evaluation must equal the coefficient sums
    /// ratio exactly.
    #[test]
    fn biquad_dc_matches_coefficient_sums() {
        let filter = peak_6db_at_1k();
        let (a1, a2, b0, b1, b2) = filter.constants();
        let dc_expected = (b0 + b1 + b2) / (1.0 + a1 + a2);
        let freqs = Array1::from_vec(vec![0.0]);
        let response =
            compute_peq_complex_response(std::slice::from_ref(&filter), &freqs, 48_000.0);
        assert!((response[0].re - dc_expected).abs() < 1e-9);
        assert!(response[0].im.abs() < 1e-9);
    }

    /// FIR evaluation is the direct DFT: response at DC equals the tap sum,
    /// and a unit delay reads back the exact delay phase.
    #[test]
    fn fir_dc_gain_is_tap_sum() {
        let coeffs = vec![0.25, 0.5, 1.0, 0.5, 0.25];
        let freqs = Array1::from_vec(vec![0.0, 1_000.0]);
        let response = compute_fir_complex_response(&coeffs, &freqs, 48_000.0);
        let expected: f64 = coeffs.iter().sum();
        assert!((response[0].re - expected).abs() < 1e-12);
        assert!(response[0].im.abs() < 1e-12);
        let delay = compute_fir_complex_response(&[0.0, 1.0], &freqs, 48_000.0);
        let expected_phase = -2.0 * std::f64::consts::PI * 1_000.0 / 48_000.0;
        assert!((delay[1].arg() - expected_phase).abs() < 1e-12);
    }

    /// Unwrap removes `2 pi` jumps so group-delay differencing sees a
    /// continuous phase curve.
    #[test]
    fn unwrap_removes_branch_jumps() {
        let wrapped = Array1::from_vec(vec![170.0, 175.0, -179.0, -174.0, -169.0]);
        let unwrapped = unwrap_phase_degrees(&wrapped);
        for pair in unwrapped.as_slice().unwrap().windows(2) {
            assert!((pair[1] - pair[0]).abs() < 30.0);
        }
        assert!(unwrapped[2] > 175.0);
    }

    /// Minimum-phase reconstruction of a flat spectrum is zero phase: the
    /// transform adds no delay or all-pass of its own.
    #[test]
    fn flat_spectrum_reconstructs_zero_phase() {
        let freq = Array1::from_vec(vec![20.0, 100.0, 1_000.0, 10_000.0]);
        let spl = Array1::from_elem(4, 80.0);
        let phase = reconstruct_minimum_phase(&freq, &spl);
        for value in phase.iter() {
            assert!(value.abs() < 1e-6, "flat phase was {value}");
        }
    }

    /// Length disagreement between curve and response fails fast instead of
    /// truncating into a silently corrupt correction.
    #[test]
    #[should_panic(expected = "does not match curve length")]
    fn response_length_mismatch_panics() {
        let curve = Curve {
            freq: Array1::from_vec(vec![100.0, 200.0]),
            spl: Array1::from_vec(vec![80.0, 81.0]),
            phase: None,
            ..Default::default()
        };
        apply_complex_response(&curve, &[Complex64::new(1.0, 0.0)]);
    }
}
