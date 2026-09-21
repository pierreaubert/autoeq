//! Band-local support alignment and timing-to-phase conversion.
//!
//! Alignment reports which requested grid bins carry measured evidence.
//! Support travels separately from sample values: gaps are never filled
//! and confidence never extends past measured support.

// Rust guideline compliant 2026-02-21

use crate::error::{AutoeqError, Result};
use crate::evidence::EvidenceBand;
use ndarray::Array1;

/// Phase uncertainty in degrees from timing uncertainty.
///
/// Converts units with `360 * freq_hz * timing_uncertainty_s`; it does
/// not choose any eligibility threshold.
///
/// # Errors
/// Returns [`AutoeqError::InvalidMeasurement`] for a non-finite or
/// non-positive frequency, or a non-finite or negative timing error.
pub fn timing_uncertainty_to_phase_deg(freq_hz: f64, timing_uncertainty_s: f64) -> Result<f64> {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!("phase conversion needs a finite positive frequency, got {freq_hz}"),
        });
    }
    if !timing_uncertainty_s.is_finite() || timing_uncertainty_s < 0.0 {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!(
                "phase conversion needs a finite nonnegative timing uncertainty, got {timing_uncertainty_s}"
            ),
        });
    }
    Ok(360.0 * freq_hz * timing_uncertainty_s)
}

/// Check that a requested grid is finite, positive, strictly increasing.
///
/// # Errors
/// Returns [`AutoeqError::InvalidMeasurement`] for any non-finite or
/// non-positive bin, or any bin that does not strictly increase.
pub fn validate_alignment_grid(grid_hz: &Array1<f64>, context: &str) -> Result<()> {
    if let Some((index, value)) = grid_hz
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value <= 0.0)
    {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!("{context} grid bin {index} must be finite and positive, got {value}"),
        });
    }
    if grid_hz
        .windows(2)
        .into_iter()
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!("{context} grid frequencies must be strictly increasing"),
        });
    }
    Ok(())
}

/// Align evidence bands to a requested grid, returning per-bin support.
///
/// A bin is supported when its frequency lies inside a validated band
/// (`low_hz <= f <= high_hz`). Bins in coverage gaps or outside every
/// band report `false`: disjoint bands stay disjoint and no gap is
/// filled or extrapolated. Dimensions and ordering are checked before
/// any arithmetic; an empty grid yields empty support.
///
/// # Errors
/// Returns [`AutoeqError::InvalidMeasurement`] for an invalid grid or
/// any invalid band.
pub fn align_evidence_support(bands: &[EvidenceBand], grid_hz: &Array1<f64>) -> Result<Vec<bool>> {
    validate_alignment_grid(grid_hz, "evidence alignment")?;
    for band in bands {
        band.validate("evidence alignment")?;
    }
    Ok(grid_hz
        .iter()
        .map(|freq| {
            bands
                .iter()
                .any(|band| *freq >= band.low_hz && *freq <= band.high_hz)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{CaptureKind, EvidenceEnvelope, Uncertainty, UncertaintyKind};

    fn band(id: &str, low_hz: f64, high_hz: f64) -> EvidenceBand {
        EvidenceBand {
            id: id.to_string(),
            low_hz,
            high_hz,
            snr_db: Some(30.0),
            coherence: Some(0.95),
            spread: Some(Uncertainty {
                kind: UncertaintyKind::ConfidenceInterval,
                magnitude_db: 0.5,
            }),
            timing_uncertainty_s: None,
            reasons: Vec::new(),
            references: Vec::new(),
        }
    }

    fn gap_bands() -> Vec<EvidenceBand> {
        vec![band("low", 20.0, 200.0), band("high", 2000.0, 20_000.0)]
    }

    #[test]
    fn core_evidence_alignment_preserves_gap() {
        let bands = gap_bands();
        // F05: two grids share a length but carry different values, with a
        // coverage gap between 200 Hz and 2 kHz.
        let grid = Array1::from_vec(vec![50.0, 100.0, 500.0, 1000.0, 5000.0]);
        assert_eq!(
            align_evidence_support(&bands, &grid).unwrap(),
            vec![true, true, false, false, true]
        );

        let shifted = Array1::from_vec(vec![30.0, 150.0, 250.0, 3000.0, 10_000.0]);
        assert_eq!(shifted.len(), grid.len());
        assert_eq!(
            align_evidence_support(&bands, &shifted).unwrap(),
            vec![true, true, false, true, true]
        );

        // Disjoint valid bands remain disjoint: nothing bridges the gap.
        let dense = Array1::from_vec(vec![150.0, 199.0, 201.0, 1999.0, 2001.0]);
        assert_eq!(
            align_evidence_support(&bands, &dense).unwrap(),
            vec![true, true, false, false, true]
        );

        // Band edges are inclusive support.
        let edges = Array1::from_vec(vec![20.0, 200.0, 2000.0, 20_000.0]);
        assert_eq!(
            align_evidence_support(&bands, &edges).unwrap(),
            vec![true, true, true, true]
        );
    }

    #[test]
    fn core_timing_uncertainty_phase_units() {
        // F02: 0.5 ms of timing uncertainty.
        let timing_s = 0.000_5;
        let at_100 = timing_uncertainty_to_phase_deg(100.0, timing_s).unwrap();
        let at_1000 = timing_uncertainty_to_phase_deg(1000.0, timing_s).unwrap();
        assert!(
            (at_100 - 18.0).abs() <= 1e-9,
            "expected 18 deg at 100 Hz, got {at_100}"
        );
        assert!(
            (at_1000 - 180.0).abs() <= 1e-9,
            "expected 180 deg at 1 kHz, got {at_1000}"
        );

        // Exact zero timing carries zero phase uncertainty.
        assert_eq!(timing_uncertainty_to_phase_deg(1000.0, 0.0).unwrap(), 0.0);

        // Band timing uncertainty flows through the same conversion.
        let mut envelope = EvidenceEnvelope::new("timing-1");
        envelope.capture = CaptureKind::StationaryIr;
        let mut timed = band("timed", 20.0, 20_000.0);
        timed.timing_uncertainty_s = Some(timing_s);
        envelope.bands.push(timed);
        envelope.validate("timed envelope").unwrap();
        let converted =
            timing_uncertainty_to_phase_deg(100.0, envelope.bands[0].timing_uncertainty_s.unwrap())
                .unwrap();
        assert!((converted - 18.0).abs() <= 1e-9);

        assert!(timing_uncertainty_to_phase_deg(0.0, timing_s).is_err());
        assert!(timing_uncertainty_to_phase_deg(100.0, -1e-9).is_err());
        assert!(timing_uncertainty_to_phase_deg(f64::NAN, timing_s).is_err());
    }

    #[test]
    fn core_evidence_no_confidence_extrapolation() {
        let bands = gap_bands();
        // Out-of-band bins stay unknown even adjacent to valid support.
        let outside = Array1::from_vec(vec![10.0, 19.9, 20_000.1, 30_000.0]);
        assert_eq!(
            align_evidence_support(&bands, &outside).unwrap(),
            vec![false, false, false, false]
        );

        // A single band never lends confidence past its own edges.
        let single = vec![band("mid", 100.0, 1000.0)];
        let grid = Array1::from_vec(vec![50.0, 100.0, 500.0, 1000.0, 2000.0]);
        assert_eq!(
            align_evidence_support(&single, &grid).unwrap(),
            vec![false, true, true, true, false]
        );

        // Invalid grids and bands fail before any arithmetic.
        let unsorted = Array1::from_vec(vec![100.0, 50.0]);
        assert!(align_evidence_support(&bands, &unsorted).is_err());
        let nonfinite = Array1::from_vec(vec![100.0, f64::NAN]);
        assert!(align_evidence_support(&bands, &nonfinite).is_err());
        let bad_band = vec![band("reversed", 1000.0, 100.0)];
        let grid = Array1::from_vec(vec![200.0]);
        assert!(align_evidence_support(&bad_band, &grid).is_err());
    }
}
