//! Measurement source handling (single file or averaging)

// Private path shims keep loader/tests source-compatible while canonical
// descriptor ownership lives in autoeq-core.
#[cfg(test)]
mod inline_measurement {
    pub use autoeq_core::InlineMeasurement;
}
#[cfg(test)]
mod measurement_ref {
    pub use autoeq_core::MeasurementRef;
}
#[cfg(test)]
mod measurement_single {
    pub use autoeq_core::MeasurementSingle;
}
#[cfg(test)]
mod measurement_source {
    pub use autoeq_core::MeasurementSource;
}
#[cfg(test)]
mod types {
    pub use autoeq_core::MeasurementMultiple;
}

mod load;
#[cfg(test)]
mod tests;

pub use autoeq_core::{
    InlineMeasurement, MeasurementMultiple, MeasurementRef, MeasurementSingle, MeasurementSource,
};
pub use load::*;
