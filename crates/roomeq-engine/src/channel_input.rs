//! Complete path-free input prepared for one RoomEQ channel.

use autoeq_core::{Curve, Result, SpinoramaBundle};
use std::borrow::Cow;

use crate::PreparedChannelMeasurements;
use crate::eq::EqResources;

/// CEA-2034 resource selection resolved before channel processing.
#[derive(Clone, Debug, Default)]
pub struct PreparedCea2034 {
    speaker_name: Option<String>,
    data: Option<Box<SpinoramaBundle>>,
}

impl PreparedCea2034 {
    pub fn new(speaker_name: Option<String>, data: Option<Box<SpinoramaBundle>>) -> Self {
        Self { speaker_name, data }
    }

    pub fn speaker_name(&self) -> Option<&str> {
        self.speaker_name.as_deref()
    }

    pub fn data(&self) -> Option<&SpinoramaBundle> {
        self.data.as_deref()
    }
}

/// Curves and external resources resolved by workflow before engine execution.
///
/// No filesystem paths or measurement-source descriptors cross this boundary.
#[derive(Clone, Debug)]
pub struct PreparedChannelInput {
    measurements: PreparedChannelMeasurements,
    valid_bands_hz: Vec<[f64; 2]>,
    arrival_time_ms: Option<f64>,
    cea2034: PreparedCea2034,
    eq_resources: EqResources,
}

impl PreparedChannelInput {
    pub fn new(
        measurements: PreparedChannelMeasurements,
        arrival_time_ms: Option<f64>,
        cea2034: PreparedCea2034,
        eq_resources: EqResources,
    ) -> Self {
        Self {
            measurements,
            valid_bands_hz: Vec::new(),
            arrival_time_ms,
            cea2034,
            eq_resources,
        }
    }

    pub fn from_measurements(measurements: PreparedChannelMeasurements) -> Self {
        Self::new(
            measurements,
            None,
            PreparedCea2034::default(),
            EqResources::default(),
        )
    }

    pub fn measurements(&self) -> &PreparedChannelMeasurements {
        &self.measurements
    }

    /// Restrict correction to the declared usable measurement band.
    ///
    /// This declaration does not authorize direct-sound or phase correction.
    /// Raw measurement curves remain unchanged.
    ///
    /// # Errors
    /// Returns an error unless both edges are finite and `0 < low < high`.
    pub fn with_valid_band_hz(self, band: [f64; 2]) -> autoeq_core::Result<Self> {
        self.with_valid_bands_hz(&[band])
    }

    /// Restrict correction to declared usable measurement segments, preserving
    /// internal coverage gaps.
    ///
    /// Bands must be finite with `0 < low < high`, ordered, and disjoint,
    /// mirroring the `MeasurementProvenance::declared_support_bands` contract.
    /// This declaration does not authorize direct-sound or phase correction.
    /// Raw measurement curves remain unchanged.
    ///
    /// # Errors
    /// Returns an error for empty, invalid, unordered, or overlapping bands.
    pub fn with_valid_bands_hz(mut self, bands: &[[f64; 2]]) -> autoeq_core::Result<Self> {
        if bands.is_empty() {
            return Err(autoeq_core::AutoeqError::InvalidMeasurement {
                message: "Declared valid_bands_hz must not be empty".to_owned(),
            });
        }
        for (index, [low, high]) in bands.iter().copied().enumerate() {
            if !low.is_finite() || !high.is_finite() || low <= 0.0 || low >= high {
                return Err(autoeq_core::AutoeqError::InvalidMeasurement {
                    message: format!(
                        "Declared valid_bands_hz band {index} must have finite edges with 0 < low < high"
                    ),
                });
            }
            if index > 0 && low <= bands[index - 1][1] {
                return Err(autoeq_core::AutoeqError::InvalidMeasurement {
                    message: "Declared valid_bands_hz must be ordered and disjoint".to_owned(),
                });
            }
        }
        self.valid_bands_hz = bands.to_vec();
        Ok(self)
    }

    /// Return the declared usable band when exactly one segment is declared.
    ///
    /// Multi-segment support has no single-band view; use
    /// [`Self::valid_bands_hz`] so internal gaps are never flattened away.
    pub fn valid_band_hz(&self) -> Option<[f64; 2]> {
        if self.valid_bands_hz.len() == 1 {
            Some(self.valid_bands_hz[0])
        } else {
            None
        }
    }

    /// Return the declared usable segments (empty when unrestricted).
    pub fn valid_bands_hz(&self) -> &[[f64; 2]] {
        &self.valid_bands_hz
    }

    /// Whether any declared usable support restricts this channel.
    pub fn has_valid_bands(&self) -> bool {
        !self.valid_bands_hz.is_empty()
    }

    /// Internal coverage gaps between consecutive declared segments.
    ///
    /// No measurement exists inside these intervals: correction filters may
    /// not center there and scoring must not consume gap samples.
    pub fn gap_intervals_hz(&self) -> Vec<[f64; 2]> {
        self.valid_bands_hz
            .windows(2)
            .map(|pair| [pair[0][1], pair[1][0]])
            .collect()
    }

    /// Whether a frequency lies inside declared usable support.
    ///
    /// Unrestricted channels support every finite positive frequency;
    /// non-finite or non-positive frequencies are never supported.
    pub fn supports_frequency(&self, frequency_hz: f64) -> bool {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return false;
        }
        self.valid_bands_hz.is_empty()
            || self
                .valid_bands_hz
                .iter()
                .any(|[low, high]| frequency_hz >= *low && frequency_hz <= *high)
    }

    /// Select usable loaded samples without extrapolating or changing raw measurements.
    ///
    /// Multi-segment support selects the union of declared segments; gap
    /// samples are dropped, never interpolated.
    pub(crate) fn usable_curve<'a>(&self, curve: &'a Curve) -> Result<Cow<'a, Curve>> {
        if self.valid_bands_hz.is_empty() {
            return Ok(Cow::Borrowed(curve));
        }
        if self.valid_bands_hz.len() == 1 {
            return curve
                .select_frequency_band(self.valid_bands_hz[0])
                .map(Cow::Owned);
        }
        curve
            .select_frequency_bands(&self.valid_bands_hz)
            .map(Cow::Owned)
    }

    pub fn arrival_time_ms(&self) -> Option<f64> {
        self.arrival_time_ms
    }

    pub fn cea2034(&self) -> &PreparedCea2034 {
        &self.cea2034
    }

    pub fn eq_resources(&self) -> &EqResources {
        &self.eq_resources
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;
    use crate::Curve;

    #[test]
    fn defaults_from_measurements_are_path_free_and_optional() {
        let curve = Curve {
            freq: Array1::from_vec(vec![100.0, 1_000.0]),
            spl: Array1::from_vec(vec![80.0, 80.0]),
            ..Curve::default()
        };
        let input = PreparedChannelInput::from_measurements(PreparedChannelMeasurements::new(
            curve.clone(),
            vec![curve],
            false,
        ));

        assert!(input.arrival_time_ms().is_none());
        assert!(input.valid_band_hz().is_none());
        assert!(input.cea2034().speaker_name().is_none());
        assert!(input.eq_resources().impulse_response.is_none());
    }

    #[test]
    fn usable_curve_retains_sample_metadata_and_refuses_sparse_support() {
        let curve = Curve {
            freq: ndarray::array![20.0, 50.0, 100.0, 200.0, 500.0],
            spl: ndarray::array![60.0, 80.0, 85.0, 80.0, 100.0],
            phase: Some(ndarray::array![0.0, 1.0, 2.0, 3.0, 4.0]),
            coherence: Some(ndarray::array![0.1, 0.7, 0.8, 0.9, 0.2]),
            noise_floor_db: Some(ndarray::array![40.0, 30.0, 31.0, 32.0, 50.0]),
            min_phase: Some(ndarray::Array1::zeros(5)),
            excess_phase: Some(ndarray::Array1::zeros(5)),
            excess_delay_ms: Some(2.0),
        };
        let input = PreparedChannelInput::from_measurements(PreparedChannelMeasurements::new(
            curve.clone(),
            vec![curve.clone()],
            false,
        ));
        assert!(matches!(
            input.usable_curve(&curve).unwrap(),
            Cow::Borrowed(_)
        ));
        let bounded = input.clone().with_valid_band_hz([50.0, 200.0]).unwrap();
        let usable = bounded.usable_curve(&curve).unwrap();
        assert_eq!(usable.freq, ndarray::array![50.0, 100.0, 200.0]);
        assert_eq!(usable.spl, ndarray::array![80.0, 85.0, 80.0]);
        assert_eq!(usable.phase, Some(ndarray::array![1.0, 2.0, 3.0]));
        assert_eq!(usable.coherence, Some(ndarray::array![0.7, 0.8, 0.9]));
        assert_eq!(
            usable.noise_floor_db,
            Some(ndarray::array![30.0, 31.0, 32.0])
        );
        assert!(
            usable.min_phase.is_none()
                && usable.excess_phase.is_none()
                && usable.excess_delay_ms.is_none()
        );
        assert_eq!(bounded.measurements().representative().freq.len(), 5);
        assert_eq!(
            bounded.measurements().representative().excess_delay_ms,
            Some(2.0)
        );
        for band in [[51.0, 99.0], [50.0, 99.0], [600.0, 700.0]] {
            assert!(
                input
                    .clone()
                    .with_valid_band_hz(band)
                    .unwrap()
                    .usable_curve(&curve)
                    .is_err()
            );
        }
    }

    #[test]
    fn declared_band_rejects_invalid_edges_and_preserves_measurements() {
        let curve = Curve {
            freq: Array1::from_vec(vec![100.0, 1_000.0]),
            spl: Array1::from_vec(vec![80.0, 80.0]),
            ..Curve::default()
        };
        let input = PreparedChannelInput::from_measurements(PreparedChannelMeasurements::new(
            curve.clone(),
            vec![curve],
            false,
        ));
        for band in [
            [0.0, 800.0],
            [-1.0, 800.0],
            [800.0, 200.0],
            [200.0, 200.0],
            [f64::NAN, 800.0],
            [200.0, f64::NAN],
            [f64::NEG_INFINITY, 800.0],
            [200.0, f64::INFINITY],
        ] {
            assert!(input.clone().with_valid_band_hz(band).is_err(), "{band:?}");
        }
        let bounded = input.with_valid_band_hz([200.0, 800.0]).unwrap();
        assert_eq!(bounded.valid_band_hz(), Some([200.0, 800.0]));
        assert_eq!(bounded.measurements().representative().freq[0], 100.0);
    }

    #[test]
    fn disjoint_bands_select_union_and_expose_gaps() {
        let curve = Curve {
            freq: ndarray::array![20.0, 100.0, 1_000.0, 8_000.0, 20_000.0],
            spl: ndarray::array![80.0, 81.0, 99.0, 79.0, 78.0],
            ..Curve::default()
        };
        let input = PreparedChannelInput::from_measurements(PreparedChannelMeasurements::new(
            curve.clone(),
            vec![curve.clone()],
            false,
        ));
        let bounded = input
            .with_valid_bands_hz(&[[20.0, 100.0], [8_000.0, 20_000.0]])
            .unwrap();
        // Multi-segment support has no single-band view: callers must use
        // the segment list so the internal gap is never flattened away.
        assert_eq!(bounded.valid_band_hz(), None);
        assert_eq!(
            bounded.valid_bands_hz(),
            &[[20.0, 100.0], [8_000.0, 20_000.0]]
        );
        assert!(bounded.has_valid_bands());
        assert_eq!(bounded.gap_intervals_hz(), vec![[100.0, 8_000.0]]);
        assert!(bounded.supports_frequency(50.0));
        assert!(bounded.supports_frequency(10_000.0));
        assert!(!bounded.supports_frequency(1_000.0));
        assert!(!bounded.supports_frequency(f64::NAN));
        let usable = bounded.usable_curve(&curve).unwrap();
        assert_eq!(usable.freq, ndarray::array![20.0, 100.0, 8_000.0, 20_000.0]);
        // Empty, unordered, and overlapping declarations are rejected.
        for bands in [
            vec![],
            vec![[8_000.0, 20_000.0], [20.0, 100.0]],
            vec![[20.0, 8_000.0], [8_000.0, 20_000.0]],
            vec![[0.0, 100.0], [8_000.0, 20_000.0]],
        ] {
            assert!(
                PreparedChannelInput::from_measurements(PreparedChannelMeasurements::new(
                    curve.clone(),
                    vec![curve.clone()],
                    false,
                ))
                .with_valid_bands_hz(&bands)
                .is_err(),
                "{bands:?}"
            );
        }
    }
}
