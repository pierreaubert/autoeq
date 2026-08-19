//! Resolve channel measurement sources before entering the RoomEQ engine.

use crate::measurement::load_source_with_individual_with_frequency_samples;
use autoeq_measurements::MeasurementSource;
use roomeq_engine::PreparedChannelMeasurements;

/// Load a channel's source once and retain both its representative response and
/// its aligned position responses for downstream in-memory processing.
pub fn prepare_channel_measurements(
    source: &MeasurementSource,
) -> Result<PreparedChannelMeasurements, Box<dyn std::error::Error>> {
    prepare_channel_measurements_with_frequency_samples(source, crate::DEFAULT_FREQUENCY_SAMPLES)
}

/// Prepare channel measurements with a configurable frequency grid.
pub fn prepare_channel_measurements_with_frequency_samples(
    source: &MeasurementSource,
    frequency_samples: usize,
) -> Result<PreparedChannelMeasurements, Box<dyn std::error::Error>> {
    let multi_measurement_source = matches!(
        source,
        MeasurementSource::Multiple(_) | MeasurementSource::InMemoryMultiple(_)
    );
    let (representative, individual) =
        load_source_with_individual_with_frequency_samples(source, frequency_samples)?;

    Ok(PreparedChannelMeasurements::new(
        representative,
        individual,
        multi_measurement_source,
    ))
}

#[cfg(test)]
mod tests {
    use autoeq_measurements::Curve;
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
    fn prepares_representative_and_individual_curves() {
        let source = MeasurementSource::InMemoryMultiple(vec![curve(80.0), curve(83.0)]);

        let prepared = prepare_channel_measurements(&source).expect("prepare measurements");

        assert_eq!(prepared.individual().len(), 2);
        assert!(prepared.is_multi_measurement_source());
        let expected =
            10.0 * ((10.0_f64.powf(80.0 / 10.0) + 10.0_f64.powf(83.0 / 10.0)) / 2.0).log10();
        assert!((prepared.representative().spl[0] - expected).abs() < 1e-6);
    }
}
