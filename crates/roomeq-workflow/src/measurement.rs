//! Measurement resource loading for RoomEQ application workflows.

use anyhow::{Context, Result, anyhow};
use autoeq_measurements::read::{
    create_log_frequency_grid, interpolate_log_space, smooth_one_over_n_octave,
};
use roomeq_model::{Curve, MeasurementRef, MeasurementSource};
use std::path::Path;

pub const DEFAULT_FREQUENCY_SAMPLES: usize = 200;
const ROOM_EQ_RESAMPLE_MIN_FREQ_HZ: f64 = 20.0;
const ROOM_EQ_RESAMPLE_MAX_FREQ_HZ: f64 = 20_000.0;
const ROOM_EQ_RESAMPLE_SMOOTHING_BANDS_PER_OCTAVE: usize = 2;

/// Reduce dense RoomEQ measurements to the grid used by the optimizer.
///
/// Generic AutoEQ readers preserve the source grid. RoomEQ intentionally
/// limits oversized inputs because its optimization objective may smooth the
/// response for every candidate filter. Resample first, then smooth the small
/// fixed grid so dense source files do not make that objective quadratic in
/// the original sample count.
fn cap_measurement_curve(curve: Curve, frequency_samples: usize) -> Curve {
    if frequency_samples == 0 || curve.freq.len() <= frequency_samples {
        return curve;
    }

    let frequency_grid = create_log_frequency_grid(
        frequency_samples,
        ROOM_EQ_RESAMPLE_MIN_FREQ_HZ,
        ROOM_EQ_RESAMPLE_MAX_FREQ_HZ,
    );
    let resampled = interpolate_log_space(&frequency_grid, &curve);
    smooth_one_over_n_octave(&resampled, ROOM_EQ_RESAMPLE_SMOOTHING_BANDS_PER_OCTAVE)
}

/// Load one CSV measurement curve with a workflow-level diagnostic.
pub fn load_curve_from_csv(path: &Path) -> Result<Curve> {
    load_curve_from_csv_with_frequency_samples(path, DEFAULT_FREQUENCY_SAMPLES)
}

/// Load one CSV measurement curve using a configurable RoomEQ frequency grid.
pub fn load_curve_from_csv_with_frequency_samples(
    path: &Path,
    frequency_samples: usize,
) -> Result<Curve> {
    autoeq_measurements::read::read_curve_from_csv(&path.to_path_buf())
        .map(|curve| cap_measurement_curve(curve, frequency_samples))
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| format!("failed to load measurement curve {}", path.display()))
}

/// Load one measurement descriptor with a workflow-level diagnostic.
pub fn load_measurement(measurement: &MeasurementRef) -> Result<Curve> {
    load_measurement_with_frequency_samples(measurement, DEFAULT_FREQUENCY_SAMPLES)
}

/// Load one measurement descriptor using a configurable RoomEQ frequency grid.
pub fn load_measurement_with_frequency_samples(
    measurement: &MeasurementRef,
    frequency_samples: usize,
) -> Result<Curve> {
    autoeq_measurements::read::load_measurement(measurement)
        .map(|curve| cap_measurement_curve(curve, frequency_samples))
        .map_err(|error| anyhow!(error.to_string()))
        .context("failed to load measurement")
}

/// Load individual measurements from a RoomEQ source, applying the RoomEQ
/// dense-curve cap to every measurement before aggregation.
pub fn load_source_individual(source: &MeasurementSource) -> Result<Vec<Curve>> {
    load_source_individual_with_frequency_samples(source, DEFAULT_FREQUENCY_SAMPLES)
}

/// Load individual measurements using a configurable RoomEQ frequency grid.
pub fn load_source_individual_with_frequency_samples(
    source: &MeasurementSource,
    frequency_samples: usize,
) -> Result<Vec<Curve>> {
    autoeq_measurements::read::load_source_individual(source)
        .map(|curves| {
            curves
                .into_iter()
                .map(|curve| cap_measurement_curve(curve, frequency_samples))
                .collect()
        })
        .map_err(|error| anyhow!(error.to_string()))
        .context("failed to load individual measurement source")
}

/// Load a source's representative and individual curves through the RoomEQ
/// dense-curve cap.
pub fn load_source_with_individual(source: &MeasurementSource) -> Result<(Curve, Vec<Curve>)> {
    load_source_with_individual_with_frequency_samples(source, DEFAULT_FREQUENCY_SAMPLES)
}

/// Load representative and individual curves using a configurable RoomEQ
/// frequency grid.
pub fn load_source_with_individual_with_frequency_samples(
    source: &MeasurementSource,
    frequency_samples: usize,
) -> Result<(Curve, Vec<Curve>)> {
    autoeq_measurements::load_source_with_individual(source)
        .map(|(representative, curves)| {
            (
                cap_measurement_curve(representative, frequency_samples),
                curves
                    .into_iter()
                    .map(|curve| cap_measurement_curve(curve, frequency_samples))
                    .collect(),
            )
        })
        .map_err(|error| anyhow!(error.to_string()))
        .context("failed to load measurement source with individual curves")
}

/// Load and combine a RoomEQ measurement source.
pub fn load_source(source: &MeasurementSource) -> Result<Curve> {
    load_source_with_frequency_samples(source, DEFAULT_FREQUENCY_SAMPLES)
}

/// Load and combine a source using a configurable RoomEQ frequency grid.
pub fn load_source_with_frequency_samples(
    source: &MeasurementSource,
    frequency_samples: usize,
) -> Result<Curve> {
    autoeq_measurements::read::load_source(source)
        .map(|curve| cap_measurement_curve(curve, frequency_samples))
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
    fn oversized_measurements_are_smoothed_to_room_eq_grid() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dense-measurement.csv");
        let mut csv = String::from("frequency,spl,phase\n");
        for index in 0..=400 {
            let fraction = index as f64 / 400.0;
            let frequency = 20.0 * 1000.0_f64.powf(fraction);
            let spl = 80.0 + 3.0 * (frequency / 1_000.0).log10();
            let phase = 15.0 * fraction;
            csv.push_str(&format!("{frequency},{spl},{phase}\n"));
        }
        std::fs::write(&path, csv).unwrap();

        let curve = load_curve_from_csv(&path).unwrap();

        assert_eq!(curve.freq.len(), DEFAULT_FREQUENCY_SAMPLES);
        assert_eq!(curve.spl.len(), DEFAULT_FREQUENCY_SAMPLES);
        assert_eq!(
            curve.phase.as_ref().unwrap().len(),
            DEFAULT_FREQUENCY_SAMPLES
        );
        assert!((curve.freq[0] - ROOM_EQ_RESAMPLE_MIN_FREQ_HZ).abs() < 1e-12);
        assert!((curve.freq[199] - ROOM_EQ_RESAMPLE_MAX_FREQ_HZ).abs() < 1e-9);

        let first_ratio = curve.freq[1] / curve.freq[0];
        for pair in curve.freq.windows(2) {
            assert!((pair[1] / pair[0] - first_ratio).abs() < 1e-12);
        }
    }

    #[test]
    fn oversized_measurements_use_custom_room_eq_grid() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dense-measurement.csv");
        let mut csv = String::from("frequency,spl\n");
        for index in 0..=400 {
            let fraction = index as f64 / 400.0;
            let frequency = 20.0 * 1000.0_f64.powf(fraction);
            let spl = 80.0 + 3.0 * (frequency / 1_000.0).log10();
            csv.push_str(&format!("{frequency},{spl}\n"));
        }
        std::fs::write(&path, csv).unwrap();

        let curve = load_curve_from_csv_with_frequency_samples(&path, 64).unwrap();

        assert_eq!(curve.freq.len(), 64);
        assert_eq!(curve.spl.len(), 64);
        assert!((curve.freq[0] - ROOM_EQ_RESAMPLE_MIN_FREQ_HZ).abs() < 1e-12);
        assert!((curve.freq[63] - ROOM_EQ_RESAMPLE_MAX_FREQ_HZ).abs() < 1e-9);
    }

    #[test]
    fn small_measurements_keep_their_original_grid() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_measurement(directory.path());

        let curve = load_curve_from_csv(&path).unwrap();

        assert_eq!(curve.freq.to_vec(), vec![20.0, 100.0, 1000.0]);
        assert_eq!(curve.spl.to_vec(), vec![70.0, 71.0, 69.0]);
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
