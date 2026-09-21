//! Analytic fixture controls F01-F04 (Q2 standalone part).
//!
//! Pure algebraic expectations from the global plan. These run without
//! synthetic S1-S3 fixtures or producer APIs, so they stand alone while
//! F05-F15 wait on the G2/G3 gates. Tolerances follow the global plan:
//! absolute 1e-9 in stated units for pure algebra.

// Rust guideline compliant 2026-02-21

/// Absolute tolerance for pure-algebra fixtures in stated units.
pub const ALGEBRA_ABS_TOL: f64 = 1e-9;

/// F01: accumulated clock offset from a fractional rate error.
///
/// `rate_error_ppm` is parts-per-million (signed); the sign convention is
/// preserved and raw offsets are never rectified.
pub fn clock_offset_seconds(rate_error_ppm: f64, duration_seconds: f64) -> f64 {
    rate_error_ppm * 1e-6 * duration_seconds
}

/// F02: phase uncertainty in degrees from a timing uncertainty.
///
/// `phase = 360 * frequency_hz * timing_uncertainty_seconds`.
pub fn phase_uncertainty_degrees(frequency_hz: f64, timing_uncertainty_seconds: f64) -> f64 {
    360.0 * frequency_hz * timing_uncertainty_seconds
}

/// Outcome of combining coherent sources.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoherentCombination {
    /// Constructive sum with the combined gain in dB.
    GainDb(f64),
    /// Destructive interference: flagged, never clamped into a good score.
    Cancellation,
}

/// F03: coherent combination of equal-magnitude sources.
///
/// `amplitudes` carries signed linear amplitudes (+1/-1 for equal sources
/// with a polarity flip). A zero-magnitude sum is cancellation, not 0 dB.
pub fn coherent_combination_db(amplitudes: &[f64]) -> CoherentCombination {
    let sum: f64 = amplitudes.iter().sum();
    if sum.abs() <= f64::EPSILON {
        return CoherentCombination::Cancellation;
    }
    CoherentCombination::GainDb(20.0 * sum.abs().log10())
}

/// F04: applying one common EQ preserves relative seat differences.
///
/// Responses are in dB; `common_eq` is added to both seats. Returns the
/// maximum absolute drift of the seat-to-seat difference where both
/// responses are defined (`None` entries are skipped, never interpolated).
pub fn common_eq_relative_drift_db(
    seat_a_db: &[Option<f64>],
    seat_b_db: &[Option<f64>],
    common_eq_db: &[Option<f64>],
) -> f64 {
    seat_a_db
        .iter()
        .zip(seat_b_db.iter())
        .zip(common_eq_db.iter())
        .filter_map(|((a, b), eq)| match (a, b, eq) {
            (Some(a), Some(b), Some(eq)) => Some(((a + eq) - (b + eq)) - (a - b)),
            _ => None,
        })
        .fold(0.0_f64, |worst, drift| worst.max(drift.abs()))
}

#[cfg(test)]
mod analytic_tests {
    use super::*;

    #[test]
    fn f01_clock_rate_error_accumulates_signed_offset() {
        // 50 ppm over 20 s is exactly 1 ms.
        let offset = clock_offset_seconds(50.0, 20.0);
        assert!((offset - 1e-3).abs() <= ALGEBRA_ABS_TOL, "{offset}");
        // Sign convention is preserved.
        let negative = clock_offset_seconds(-50.0, 20.0);
        assert!((negative + 1e-3).abs() <= ALGEBRA_ABS_TOL, "{negative}");
    }

    #[test]
    fn f02_timing_uncertainty_maps_to_phase() {
        // 0.5 ms is 18 degrees at 100 Hz and 180 degrees at 1 kHz.
        let at_100 = phase_uncertainty_degrees(100.0, 0.5e-3);
        assert!((at_100 - 18.0).abs() <= ALGEBRA_ABS_TOL, "{at_100}");
        let at_1000 = phase_uncertainty_degrees(1000.0, 0.5e-3);
        assert!((at_1000 - 180.0).abs() <= ALGEBRA_ABS_TOL, "{at_1000}");
    }

    #[test]
    fn f03_equal_sources_sum_and_opposite_polarity_cancels() {
        // Two equal coherent sources sum to 20 log10(2) dB.
        let expected = 20.0 * 2.0_f64.log10();
        assert!((expected - 6.020_599_913_279_624).abs() <= ALGEBRA_ABS_TOL);
        assert_eq!(
            coherent_combination_db(&[1.0, 1.0]),
            CoherentCombination::GainDb(expected)
        );
        // Opposite polarity cancels: flagged, not scored.
        assert_eq!(
            coherent_combination_db(&[1.0, -1.0]),
            CoherentCombination::Cancellation
        );
    }

    #[test]
    fn f04_common_eq_preserves_relative_seat_difference() {
        let seat_a = [Some(70.0), Some(72.0), None, Some(68.0)];
        let seat_b = [Some(66.0), Some(71.0), Some(69.0), None];
        let common_eq = [Some(-3.0), Some(2.0), Some(1.0), Some(-1.0)];
        // Where both seats are defined the relative difference is unchanged.
        assert!(common_eq_relative_drift_db(&seat_a, &seat_b, &common_eq) <= ALGEBRA_ABS_TOL);
        // A seat-specific (non-common) change is detected, not hidden.
        let seat_a_changed = [Some(65.0), Some(72.0), None, Some(68.0)];
        let drift = seat_a_changed
            .iter()
            .zip(seat_b.iter())
            .filter_map(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => Some(a - b),
                _ => None,
            })
            .zip(
                seat_a
                    .iter()
                    .zip(seat_b.iter())
                    .filter_map(|(a, b)| match (a, b) {
                        (Some(a), Some(b)) => Some(a - b),
                        _ => None,
                    }),
            )
            .fold(0.0_f64, |worst, (now, before)| {
                worst.max((now - before).abs())
            });
        assert!(drift > 1.0, "{drift}");
    }
}
