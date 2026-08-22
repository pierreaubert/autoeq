//! Measurement resource loading for RoomEQ application workflows.

use anyhow::{Context, Result, anyhow};
use autoeq_core::phase_utils::{
    compute_excess_phase, estimate_delay_from_excess_phase, reconstruct_minimum_phase,
    unwrap_phase_degrees,
};
use autoeq_measurements::read::{
    create_log_frequency_grid, interpolate_log_space, smooth_one_over_n_octave,
};
use ndarray::Array1;
use roomeq_model::{Curve, MeasurementRef, MeasurementSource};
use std::path::Path;

pub const DEFAULT_FREQUENCY_SAMPLES: usize = 200;
const ROOM_EQ_RESAMPLE_MIN_FREQ_HZ: f64 = 20.0;
const ROOM_EQ_RESAMPLE_MAX_FREQ_HZ: f64 = 20_000.0;
const ROOM_EQ_RESAMPLE_LOW_FREQ_MAX_HZ: f64 = 1_000.0;
const ROOM_EQ_RESAMPLE_LOW_FREQ_STEP_HZ: f64 = 4.0;
const ROOM_EQ_RESAMPLE_SMOOTHING_BANDS_PER_OCTAVE: usize = 2;

/// Reduce dense RoomEQ measurements to the grid used by the optimizer.
///
/// Generic AutoEQ readers preserve the source grid. RoomEQ intentionally
/// limits oversized inputs because its optimization objective may smooth the
/// response for every candidate filter. Resample first, then smooth the small
/// hybrid grid so dense source files do not make that objective quadratic in
/// the original sample count.
fn hybrid_frequency_grid(frequency_samples: usize) -> Array1<f64> {
    let high_frequency_samples = frequency_samples.max(2);
    let log_step = (ROOM_EQ_RESAMPLE_MAX_FREQ_HZ / ROOM_EQ_RESAMPLE_MIN_FREQ_HZ).ln()
        / (high_frequency_samples - 1) as f64;
    let high_frequency_intervals =
        ((ROOM_EQ_RESAMPLE_MAX_FREQ_HZ / ROOM_EQ_RESAMPLE_LOW_FREQ_MAX_HZ).ln() / log_step).ceil()
            as usize;
    let high_frequency_grid = create_log_frequency_grid(
        high_frequency_intervals + 1,
        ROOM_EQ_RESAMPLE_LOW_FREQ_MAX_HZ,
        ROOM_EQ_RESAMPLE_MAX_FREQ_HZ,
    );

    let low_frequency_samples = ((ROOM_EQ_RESAMPLE_LOW_FREQ_MAX_HZ - ROOM_EQ_RESAMPLE_MIN_FREQ_HZ)
        / ROOM_EQ_RESAMPLE_LOW_FREQ_STEP_HZ)
        .round() as usize
        + 1;
    let mut frequencies = Vec::with_capacity(low_frequency_samples + high_frequency_intervals);
    frequencies.extend((0..low_frequency_samples).map(|index| {
        ROOM_EQ_RESAMPLE_MIN_FREQ_HZ + index as f64 * ROOM_EQ_RESAMPLE_LOW_FREQ_STEP_HZ
    }));
    frequencies.extend(high_frequency_grid.iter().skip(1).copied());
    Array1::from_vec(frequencies)
}

fn clipped_hybrid_frequency_grid(curve: &Curve, frequency_samples: usize) -> Option<Array1<f64>> {
    let source_min = curve.freq.first().copied()?;
    let source_max = curve.freq.last().copied()?;
    let mut frequencies: Vec<f64> = hybrid_frequency_grid(frequency_samples)
        .iter()
        .copied()
        .filter(|frequency| *frequency >= source_min && *frequency <= source_max)
        .collect();

    if frequencies.is_empty() {
        return None;
    }
    if frequencies[0] > source_min {
        frequencies.insert(0, source_min);
    }
    if *frequencies.last().unwrap_or(&source_max) < source_max {
        frequencies.push(source_max);
    }
    Some(Array1::from_vec(frequencies))
}

fn interpolate_linear_values(
    output_frequencies: &Array1<f64>,
    input_frequencies: &Array1<f64>,
    input_values: &Array1<f64>,
) -> Array1<f64> {
    if input_values.is_empty() {
        return Array1::zeros(output_frequencies.len());
    }
    if input_values.len() == 1 {
        return Array1::from_elem(output_frequencies.len(), input_values[0]);
    }

    Array1::from_iter(output_frequencies.iter().map(|&frequency| {
        if frequency <= input_frequencies[0] {
            return input_values[0];
        }
        let last = input_frequencies.len() - 1;
        if frequency >= input_frequencies[last] {
            return input_values[last];
        }
        let right = input_frequencies
            .as_slice()
            .expect("frequency array is contiguous")
            .partition_point(|&value| value < frequency);
        let left = right - 1;
        let denominator = input_frequencies[right] - input_frequencies[left];
        if denominator.abs() <= f64::EPSILON {
            input_values[left]
        } else {
            let fraction = (frequency - input_frequencies[left]) / denominator;
            input_values[left] + fraction * (input_values[right] - input_values[left])
        }
    }))
}

fn reconstruct_resampled_phase(original: &Curve, resampled: &mut Curve) {
    let Some(measured_phase) = original
        .phase
        .as_ref()
        .filter(|phase| phase.len() == original.freq.len())
    else {
        return;
    };
    if original.freq.len() < 2 || original.spl.len() != original.freq.len() {
        return;
    }

    let original_min_phase = reconstruct_minimum_phase(&original.freq, &original.spl);
    let low_frequency_indices: Vec<usize> = original
        .freq
        .iter()
        .enumerate()
        .filter_map(|(index, &frequency)| {
            (frequency <= ROOM_EQ_RESAMPLE_LOW_FREQ_MAX_HZ).then_some(index)
        })
        .collect();
    let (delay_ms, _) = if low_frequency_indices.len() >= 2 {
        let low_freq = Array1::from_iter(
            low_frequency_indices
                .iter()
                .map(|&index| original.freq[index]),
        );
        let low_phase = Array1::from_iter(
            low_frequency_indices
                .iter()
                .map(|&index| measured_phase[index]),
        );
        let low_min_phase = Array1::from_iter(
            low_frequency_indices
                .iter()
                .map(|&index| original_min_phase[index]),
        );
        let low_excess_phase =
            compute_excess_phase(&unwrap_phase_degrees(&low_phase), &low_min_phase);
        estimate_delay_from_excess_phase(&low_freq, &low_excess_phase)
    } else {
        let unwrapped_phase = unwrap_phase_degrees(measured_phase);
        let original_excess_phase = compute_excess_phase(&unwrapped_phase, &original_min_phase);
        estimate_delay_from_excess_phase(&original.freq, &original_excess_phase)
    };
    let delay_seconds = delay_ms / 1_000.0;
    let corrected_phase = Array1::from_iter(measured_phase.iter().zip(original.freq.iter()).map(
        |(&phase, &frequency)| {
            let corrected = phase + 360.0 * frequency * delay_seconds;
            ((corrected + 180.0).rem_euclid(360.0)) - 180.0
        },
    ));
    let corrected_excess_phase =
        compute_excess_phase(&unwrap_phase_degrees(&corrected_phase), &original_min_phase);
    let (_, residual_excess_phase) =
        estimate_delay_from_excess_phase(&original.freq, &corrected_excess_phase);
    let resampled_min_phase = reconstruct_minimum_phase(&resampled.freq, &resampled.spl);
    let resampled_excess_phase =
        interpolate_linear_values(&resampled.freq, &original.freq, &residual_excess_phase);
    let phase = Array1::from_iter(
        resampled
            .freq
            .iter()
            .zip(resampled_min_phase.iter())
            .zip(resampled_excess_phase.iter())
            .map(|((&frequency, &min_phase), &excess_phase)| {
                min_phase + excess_phase - 360.0 * frequency * delay_seconds
            }),
    );
    resampled.phase = Some(phase);
    resampled.min_phase = Some(resampled_min_phase);
    resampled.excess_phase = Some(resampled_excess_phase);
    resampled.excess_delay_ms = Some(delay_ms);
}

fn cap_measurement_curve(curve: Curve, frequency_samples: usize) -> Curve {
    if frequency_samples == 0 || curve.freq.len() <= frequency_samples {
        return curve;
    }

    let Some(frequency_grid) = clipped_hybrid_frequency_grid(&curve, frequency_samples) else {
        return curve;
    };
    let resampled = interpolate_log_space(&frequency_grid, &curve);
    let mut smoothed =
        smooth_one_over_n_octave(&resampled, ROOM_EQ_RESAMPLE_SMOOTHING_BANDS_PER_OCTAVE);
    reconstruct_resampled_phase(&curve, &mut smoothed);
    smoothed
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

        let expected_len = hybrid_frequency_grid(DEFAULT_FREQUENCY_SAMPLES).len();
        assert_eq!(curve.freq.len(), expected_len);
        assert_eq!(curve.spl.len(), expected_len);
        assert_eq!(curve.phase.as_ref().unwrap().len(), expected_len);
        assert!((curve.freq[0] - ROOM_EQ_RESAMPLE_MIN_FREQ_HZ).abs() < 1e-12);
        assert!((curve.freq.last().copied().unwrap() - ROOM_EQ_RESAMPLE_MAX_FREQ_HZ).abs() < 1e-9);

        assert!(
            curve
                .freq
                .windows(2)
                .into_iter()
                .all(|pair| pair[1] > pair[0])
        );
        assert!(
            curve.freq.as_slice().unwrap()[1..=245]
                .windows(2)
                .all(|pair| (pair[1] - pair[0] - ROOM_EQ_RESAMPLE_LOW_FREQ_STEP_HZ).abs() < 1e-12)
        );
        let high_ratio = curve.freq[247] / curve.freq[246];
        let default_ratio = (ROOM_EQ_RESAMPLE_MAX_FREQ_HZ / ROOM_EQ_RESAMPLE_MIN_FREQ_HZ)
            .powf(1.0 / (DEFAULT_FREQUENCY_SAMPLES - 1) as f64);
        assert!((high_ratio - default_ratio).abs() < 0.002);
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

        let expected_len = hybrid_frequency_grid(64).len();
        assert_eq!(curve.freq.len(), expected_len);
        assert_eq!(curve.spl.len(), expected_len);
        assert!((curve.freq[0] - ROOM_EQ_RESAMPLE_MIN_FREQ_HZ).abs() < 1e-12);
        assert!((curve.freq.last().copied().unwrap() - ROOM_EQ_RESAMPLE_MAX_FREQ_HZ).abs() < 1e-9);
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

    #[test]
    fn phase_reconstruction_preserves_delay_and_excess_phase() {
        let frequencies =
            Array1::from_iter((0..=400).map(|index| 20.0 * 1000.0_f64.powf(index as f64 / 400.0)));
        let delay_ms = 2.5;
        let spl = Array1::from_elem(401, 80.0);
        let minimum_phase = reconstruct_minimum_phase(&frequencies, &spl);
        let base_excess = frequencies.mapv(|frequency| (frequency / 300.0).ln().sin() * 4.0);
        let sum_frequency = frequencies.sum();
        let sum_excess = base_excess.sum();
        let sum_frequency_squared = frequencies.mapv(|frequency| frequency * frequency).sum();
        let sum_frequency_excess = frequencies
            .iter()
            .zip(base_excess.iter())
            .map(|(&frequency, &excess)| frequency * excess)
            .sum::<f64>();
        let count = frequencies.len() as f64;
        let slope = (count * sum_frequency_excess - sum_frequency * sum_excess)
            / (count * sum_frequency_squared - sum_frequency * sum_frequency);
        let intercept = (sum_excess - slope * sum_frequency) / count;
        let excess = Array1::from_iter(
            frequencies
                .iter()
                .zip(base_excess.iter())
                .map(|(&frequency, &value)| value - slope * frequency - intercept),
        );
        let total_phase = &minimum_phase
            + &frequencies.mapv(|frequency| -360.0 * frequency * delay_ms / 1_000.0)
            + &excess;
        let phase = total_phase.mapv(|value| ((value + 180.0).rem_euclid(360.0)) - 180.0);
        let curve = Curve {
            freq: frequencies,
            spl,
            phase: Some(phase),
            ..Default::default()
        };
        let resampled = cap_measurement_curve(curve.clone(), DEFAULT_FREQUENCY_SAMPLES);

        let estimated_delay = resampled.excess_delay_ms.unwrap();
        assert!(
            (estimated_delay - delay_ms).abs() < 0.05,
            "expected {delay_ms} ms, got {estimated_delay} ms"
        );
        let reconstructed_excess = resampled.excess_phase.unwrap();
        let expected_excess = interpolate_linear_values(&resampled.freq, &curve.freq, &excess);
        for (&actual, &expected) in reconstructed_excess.iter().zip(expected_excess.iter()) {
            assert!((actual - expected).abs() < 0.1);
        }
    }

    #[test]
    fn phase_wrapping_is_unwrapped_before_resampling() {
        let frequencies =
            Array1::from_iter((0..=400).map(|index| 20.0 * 1000.0_f64.powf(index as f64 / 400.0)));
        let phase = frequencies.mapv(|frequency| {
            let unwrapped = -360.0 * frequency * 0.001;
            ((unwrapped + 180.0).rem_euclid(360.0)) - 180.0
        });
        let curve = Curve {
            freq: frequencies,
            spl: Array1::from_elem(401, 80.0),
            phase: Some(phase),
            ..Default::default()
        };
        let resampled = cap_measurement_curve(curve, DEFAULT_FREQUENCY_SAMPLES);
        let phase = resampled.phase.unwrap();
        let max_delta = phase
            .as_slice()
            .unwrap()
            .get(..=245)
            .unwrap()
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .fold(0.0, f64::max);
        assert!(max_delta < 20.0, "phase contains a {max_delta} degree jump");
    }

    #[test]
    fn sparse_measurements_are_not_extrapolated() {
        let curve = Curve {
            freq: Array1::from_vec(vec![80.0, 200.0, 1_000.0, 8_000.0]),
            spl: Array1::from_vec(vec![80.0, 81.0, 79.0, 78.0]),
            ..Default::default()
        };
        let capped = cap_measurement_curve(curve, 2);
        assert_eq!(capped.freq[0], 80.0);
        assert_eq!(capped.freq.last().copied(), Some(8_000.0));
        assert!(
            capped
                .freq
                .iter()
                .all(|&frequency| (80.0..=8_000.0).contains(&frequency))
        );
    }

    #[test]
    fn missing_phase_remains_magnitude_only() {
        let curve = Curve {
            freq: Array1::from_iter(
                (0..=400).map(|index| 20.0 * 1000.0_f64.powf(index as f64 / 400.0)),
            ),
            spl: Array1::from_elem(401, 80.0),
            phase: None,
            ..Default::default()
        };
        let capped = cap_measurement_curve(curve, DEFAULT_FREQUENCY_SAMPLES);
        assert!(capped.phase.is_none());
        assert!(capped.min_phase.is_none());
        assert!(capped.excess_phase.is_none());
        assert!(capped.excess_delay_ms.is_none());
    }
}
