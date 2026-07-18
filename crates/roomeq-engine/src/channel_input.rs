//! Complete path-free input prepared for one RoomEQ channel.

use autoeq_core::SpinoramaBundle;

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
        assert!(input.cea2034().speaker_name().is_none());
        assert!(input.eq_resources().impulse_response.is_none());
    }
}
