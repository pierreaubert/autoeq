//! Prepared, in-memory measurement inputs for one RoomEQ channel.

use autoeq_core::Curve;

/// Producer-recorded loading operations bound to prepared response content.
///
/// This transports the canonical measurement ledger; it is not acquisition
/// authentication or a claim that later engine conditioning is fully recorded.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct MeasurementConditioningReceipt {
    /// Producer-recorded loaded-curve roots; absent in legacy receipts.
    #[serde(default)]
    pub native_identities: Vec<String>,
    pub entries: Vec<autoeq_measurements::LedgerEntry>,
    pub representative_identity: String,
    pub individual_identities: Vec<String>,
}

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
    conditioning: Vec<autoeq_measurements::LedgerEntry>,
    native_identities: Vec<String>,
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
            conditioning: Vec::new(),
            native_identities: Vec::new(),
        }
    }

    /// Attach producer-recorded measurement conditioning operations.
    pub fn with_conditioning(
        mut self,
        conditioning: Vec<autoeq_measurements::LedgerEntry>,
    ) -> Self {
        self.conditioning = conditioning;
        self
    }

    /// Attach loaded-curve identities recorded before source alignment.
    ///
    /// Roots must come from the loader, not be inferred from ledger inputs.
    /// They do not authenticate acquisition or physical seat attribution.
    pub fn with_native_identities(mut self, native_identities: Vec<String>) -> Self {
        self.native_identities = native_identities;
        self
    }

    /// Recorded source loading operations, excluding unrecorded downstream processing.
    pub fn conditioning(&self) -> &[autoeq_measurements::LedgerEntry] {
        &self.conditioning
    }

    /// Bind recorded loading operations to the prepared numerical responses.
    ///
    /// # Errors
    /// Rejects curves that cannot provide a valid canonical content identity.
    pub fn conditioning_receipt(&self) -> autoeq_core::Result<MeasurementConditioningReceipt> {
        Ok(MeasurementConditioningReceipt {
            native_identities: self.native_identities.clone(),
            entries: self.conditioning.clone(),
            representative_identity: self.representative.content_hash()?,
            individual_identities: self
                .individual
                .iter()
                .map(Curve::content_hash)
                .collect::<autoeq_core::Result<_>>()?,
        })
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
