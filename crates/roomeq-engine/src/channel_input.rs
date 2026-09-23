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
    valid_band_hz: Option<[f64; 2]>,
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
            valid_band_hz: None,
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
    pub fn with_valid_band_hz(mut self, band: [f64; 2]) -> autoeq_core::Result<Self> {
        let [low, high] = band;
        if !low.is_finite() || !high.is_finite() || low <= 0.0 || low >= high {
            return Err(autoeq_core::AutoeqError::InvalidMeasurement {
                message: "Declared valid_band_hz must have finite edges with 0 < low < high"
                    .to_owned(),
            });
        }
        self.valid_band_hz = Some(band);
        Ok(self)
    }

    /// Return the declared usable band, if supplied by measurement provenance.
    pub fn valid_band_hz(&self) -> Option<[f64; 2]> {
        self.valid_band_hz
    }

    /// Select usable loaded samples without extrapolating or changing raw measurements.
    pub(crate) fn usable_curve<'a>(&self, curve: &'a Curve) -> Result<Cow<'a, Curve>> {
        let Some(band) = self.valid_band_hz else {
            return Ok(Cow::Borrowed(curve));
        };
        curve.select_frequency_band(band).map(Cow::Owned)
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
}
