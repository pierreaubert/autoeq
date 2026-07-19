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

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::MeasurementSingle;

    fn write_measurement(directory: &Path) -> std::path::PathBuf {
        let path = directory.join("measurement.csv");
        std::fs::write(&path, "frequency,spl\n20,70\n100,71\n1000,69\n").unwrap();
        path
    }

    #[test]
    fn workflow_measurement_adapters_load_csv_ref_and_source() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_measurement(directory.path());

        let direct = load_curve_from_csv(&path).unwrap();
        let measurement = MeasurementRef::Path(path);
        let referenced = load_measurement(&measurement).unwrap();
        let source = MeasurementSource::Single(MeasurementSingle {
            measurement,
            speaker_name: Some("left".to_string()),
        });
        let combined = load_source(&source).unwrap();

        assert_eq!(direct.freq, referenced.freq);
        assert_eq!(referenced.spl, combined.spl);
    }

    #[test]
    fn workflow_measurement_adapters_preserve_operation_context() {
        let missing = Path::new("missing-measurement.csv");
        assert!(
            load_curve_from_csv(missing)
                .unwrap_err()
                .to_string()
                .contains("failed to load measurement curve")
        );
        assert!(
            load_measurement(&MeasurementRef::Path(missing.to_path_buf()))
                .unwrap_err()
                .to_string()
                .contains("failed to load measurement")
        );
    }
}
