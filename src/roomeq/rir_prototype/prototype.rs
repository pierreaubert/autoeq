//! Weighted RIR prototype builder.

use super::config::RirPrototypeConfig;

/// A distance- and directivity-weighted RIR prototype.
#[derive(Debug, Clone)]
pub struct WeightedPrototype {
    // TODO: populate with weighted impulse responses once Task 2 is implemented.
}

/// Build a weighted RIR prototype from the given configuration.
pub fn build_weighted_prototype(_config: &RirPrototypeConfig) -> WeightedPrototype {
    WeightedPrototype {}
}
