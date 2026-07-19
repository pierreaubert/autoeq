//! Measurement resource loading for RoomEQ application workflows.

use anyhow::{Context, Result, anyhow};
use roomeq_model::{Curve, MeasurementRef, MeasurementSource};
use std::path::Path;

/// Load one CSV measurement curve with a workflow-level diagnostic.
pub fn load_curve_from_csv(path: &Path) -> Result<Curve> {
    autoeq_measurements::read::read_curve_from_csv(&path.to_path_buf())
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| format!("failed to load measurement curve {}", path.display()))
}

/// Load one measurement descriptor with a workflow-level diagnostic.
pub fn load_measurement(measurement: &MeasurementRef) -> Result<Curve> {
    autoeq_measurements::read::load_measurement(measurement)
        .map_err(|error| anyhow!(error.to_string()))
        .context("failed to load measurement")
}

/// Load and combine a RoomEQ measurement source.
pub fn load_source(source: &MeasurementSource) -> Result<Curve> {
    autoeq_measurements::read::load_source(source)
        .map_err(|error| anyhow!(error.to_string()))
        .context("failed to load measurement source")
}
