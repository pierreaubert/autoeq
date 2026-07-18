//! Prepared, in-memory measurement inputs for one RoomEQ channel.

use autoeq_core::Curve;

/// Measurement curves resolved by the workflow before channel processing.
///
/// This contract deliberately contains no source descriptors or filesystem
/// paths. The representative response is used by the normal single-channel
/// stages, while `individual` retains the aligned position responses needed by
/// multi-measurement optimization and spatial-robustness analysis.
#[derive(Clone, Debug)]
pub struct PreparedChannelMeasurements {
    representative: Curve,
    individual: Vec<Curve>,
    multi_measurement_source: bool,
}

impl PreparedChannelMeasurements {
    /// Build a prepared channel input from already-loaded curves.
    pub fn new(
        representative: Curve,
        individual: Vec<Curve>,
        multi_measurement_source: bool,
    ) -> Self {
        Self {
            representative,
            individual,
            multi_measurement_source,
        }
    }

    /// Power-domain representative response for the channel.
    pub fn representative(&self) -> &Curve {
        &self.representative
    }

    /// Aligned responses for each measurement position.
    pub fn individual(&self) -> &[Curve] {
        &self.individual
    }

    /// Whether the source was explicitly configured as multi-measurement.
    ///
    /// This remains distinct from `individual().len() > 1` so that a configured
    /// multi-measurement source containing one curve preserves its existing
    /// optimizer dispatch semantics.
    pub fn is_multi_measurement_source(&self) -> bool {
        self.multi_measurement_source
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;

    fn curve(spl: f64) -> Curve {
        Curve {
            freq: Array1::from_vec(vec![100.0, 1_000.0]),
            spl: Array1::from_vec(vec![spl, spl]),
            ..Curve::default()
        }
    }

    #[test]
    fn retains_representative_and_individual_curves() {
        let prepared =
            PreparedChannelMeasurements::new(curve(81.0), vec![curve(80.0), curve(82.0)], true);

        assert_eq!(prepared.representative().spl[0], 81.0);
        assert_eq!(prepared.individual().len(), 2);
        assert!(prepared.is_multi_measurement_source());
    }
}
