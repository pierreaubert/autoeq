//! Resolve channel measurement sources before entering the RoomEQ engine.

use crate::measurement::load_source_with_conditioning;
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
    let loaded = load_source_with_conditioning(source, frequency_samples)?;

    Ok(PreparedChannelMeasurements::new(
        loaded.representative,
        loaded.individual,
        multi_measurement_source,
    )
    .with_conditioning(loaded.conditioning)
    .with_native_identities(loaded.native_identities))
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
        assert_eq!(prepared.conditioning().len(), 1);
        assert_eq!(
            prepared.conditioning()[0].operation,
            "source_spatial_power_rms"
        );
    }

    #[test]
    fn prepared_conditioning_links_native_curves_to_dense_optimizer_inputs() {
        let native: Vec<_> = [0.0, 1.0]
            .into_iter()
            .map(|offset| {
                let freq = Array1::from_iter((1..=1000).map(|i| i as f64 * 10.0 + offset));
                Curve {
                    spl: freq.mapv(|f| 80.0 + (f / 100.0).sin()),
                    freq,
                    ..Default::default()
                }
            })
            .collect();
        let mut known: std::collections::HashSet<_> = native
            .iter()
            .map(|curve| curve.content_hash().unwrap())
            .collect();
        let source = MeasurementSource::InMemoryMultiple(native);
        let prepared = prepare_channel_measurements_with_frequency_samples(&source, 64).unwrap();
        let receipt = prepared.conditioning_receipt().unwrap();
        let MeasurementSource::InMemoryMultiple(native) = &source else {
            unreachable!()
        };
        assert_eq!(
            receipt.native_identities,
            native
                .iter()
                .map(|curve| curve.content_hash().unwrap())
                .collect::<Vec<_>>()
        );
        assert!(
            prepared
                .conditioning()
                .iter()
                .any(|entry| entry.operation == "source_overlap_alignment")
        );
        assert!(
            prepared
                .conditioning()
                .iter()
                .any(|entry| entry.operation == "roomeq_dense_grid_conditioning")
        );
        for entry in prepared.conditioning() {
            assert!(!entry.input_hashes.is_empty());
            assert!(
                entry.input_hashes.iter().all(|hash| known.contains(hash)),
                "unbound inputs for {}",
                entry.operation
            );
            known.insert(entry.output_hash.clone());
        }
        assert!(known.contains(&prepared.representative().content_hash().unwrap()));
        for curve in prepared.individual() {
            assert!(known.contains(&curve.content_hash().unwrap()));
        }
        let identity = prepare_channel_measurements_with_frequency_samples(
            &MeasurementSource::InMemory(curve(80.0)),
            64,
        )
        .unwrap();
        assert!(
            identity.conditioning().is_empty(),
            "unchanged loading must not invent conditioning"
        );
    }
}
