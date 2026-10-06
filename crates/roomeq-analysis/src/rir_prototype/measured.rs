//! Measured arrival-energy mixtures for magnitude-only prototype weights.

use super::config::RirPrototypeConfig;
use super::weights::{directivity_weight, distance_weight, normalized_weights};
use autoeq_core::capture_provenance::{CaptureArrival, CaptureGeometry, CaptureProvenance};
use ndarray::{Array1, Array2};

fn arrival_angle(
    event: &CaptureArrival,
    frequency: f64,
    axis: [f64; 3],
    relative_us: f64,
) -> Option<f64> {
    let direction = event.direction?;
    let band = event.band_hz?;
    let residual = event.residual_samples?;
    if event.mirror_ambiguous
        || !residual.is_finite()
        || !(0.0..=0.5).contains(&residual)
        || direction.iter().any(|value| !value.is_finite())
        || (direction.iter().map(|value| value * value).sum::<f64>() - 1.0).abs() > 1e-6
        || band.iter().any(|value| !value.is_finite())
        || band[0] <= 0.0
        || band[1] <= band[0]
        || band[1] > 3000.0
        || frequency < band[0]
        || frequency > band[1]
        || band[1] > 25_000.0 / relative_us
    {
        return None;
    }
    Some(
        direction
            .iter()
            .zip(axis)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .clamp(-1.0, 1.0)
            .acos(),
    )
}

pub(super) fn weights_with_capture(
    distances: &[f64],
    angles: &[f64],
    frequencies: &Array1<f64>,
    config: &RirPrototypeConfig,
    capture: Option<&CaptureProvenance>,
) -> (Array2<f64>, Vec<usize>) {
    let mut weights = normalized_weights(distances, angles, frequencies, config);
    let mut measured_bins = vec![0; distances.len()];
    let Some(capture) = capture else {
        return (weights, measured_bins);
    };
    if distances.len() < 4
        || capture.geometry != CaptureGeometry::Compact
        || capture.coherent_reference(distances.len()).is_err()
        || capture
            .takes
            .iter()
            .zip(&config.microphone_positions)
            .any(|(take, position)| {
                let distance = take
                    .position_m
                    .iter()
                    .zip(position)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                !distance.is_finite() || distance > take.position_uncertainty_mm / 1000.0 + 1e-9
            })
    {
        return (weights, measured_bins);
    }
    let origin = capture.takes[0].position_m;
    let edge = |mic: usize| {
        std::array::from_fn::<_, 3, _>(|axis| capture.takes[mic].position_m[axis] - origin[axis])
    };
    let a = edge(1);
    let b = edge(2);
    let c = edge(3);
    let normal = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let normal_norm = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
    let height = normal.iter().zip(c).map(|(a, b)| a * b).sum::<f64>().abs() / normal_norm;
    let survey_m = capture
        .takes
        .iter()
        .map(|take| take.position_uncertainty_mm / 1000.0)
        .fold(0.0_f64, f64::max);
    // Do not let a claimed resolved direction override planar or survey-limited geometry.
    if !height.is_finite() || height <= (4.0 * survey_m).max(1e-6) {
        return (weights, measured_bins);
    }
    let Some(report) = capture
        .reflection_report
        .as_ref()
        .filter(|report| report.issues.is_empty())
    else {
        return (weights, measured_bins);
    };
    let Some(direct) = &report.direct_sound else {
        return (weights, measured_bins);
    };
    if report.early_reflections.len() > 64 {
        return (weights, measured_bins);
    }
    let events: Vec<_> = std::iter::once(direct)
        .chain(&report.early_reflections)
        .collect();
    if events.iter().any(|event| {
        event.microphone_energy_db.len() != distances.len()
            || event
                .microphone_energy_db
                .iter()
                .any(|level| !level.is_finite())
    }) {
        return (weights, measured_bins);
    }
    let mut bounds: Vec<_> = capture
        .takes
        .iter()
        .filter_map(|take| take.residual_uncertainty_us)
        .collect();
    if bounds.iter().any(|bound| *bound <= 0.0) {
        return (weights, measured_bins);
    }
    bounds.sort_by(|a, b| b.total_cmp(a));
    let relative_us = bounds[0] + bounds[1];
    // Head-forward axis points from the reference listener toward the source.
    // Measured DOA vectors also point toward the arriving image source.
    let mut axis = std::array::from_fn::<_, 3, _>(|i| {
        config.source_position[i] - config.reference_position[i]
    });
    let norm = axis.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return (weights, measured_bins);
    }
    axis.iter_mut().for_each(|value| *value /= norm);
    for (bin, &frequency) in frequencies.iter().enumerate() {
        let event_angles: Vec<_> = events
            .iter()
            .map(|event| arrival_angle(event, frequency, axis, relative_us))
            .collect();
        if event_angles.iter().all(Option::is_none) {
            continue;
        }
        let model_frequency = if config.frequency_dependent_directivity {
            frequency
        } else {
            1000.0
        };
        let mut raw = vec![0.0; distances.len()];
        for mic in 0..distances.len() {
            // Shift logarithmic energies before exponentiating to keep extreme
            // but finite level declarations from overflowing the mixture.
            let maximum = events
                .iter()
                .map(|event| event.microphone_energy_db[mic])
                .fold(f64::NEG_INFINITY, f64::max);
            let mut energy_sum = 0.0;
            let mut directional_sum = 0.0;
            for (event, measured_angle) in events.iter().zip(&event_angles) {
                let energy = 10.0_f64.powf((event.microphone_energy_db[mic] - maximum) / 10.0);
                energy_sum += energy;
                directional_sum += energy
                    * directivity_weight(
                        model_frequency,
                        measured_angle.unwrap_or(angles[mic]),
                        config.directivity,
                    );
            }
            raw[mic] = distance_weight(distances[mic], config.distance_mode) * directional_sum
                / energy_sum;
        }
        let total = raw.iter().sum::<f64>();
        if total.is_finite() && total > 0.0 {
            for mic in 0..distances.len() {
                weights[[mic, bin]] = raw[mic] / total;
                measured_bins[mic] += 1;
            }
        }
    }
    (weights, measured_bins)
}

#[cfg(test)]
mod tests {
    use super::super::config::{DirectivityModel, DistanceWeightMode};
    use super::super::{build_weighted_prototype, build_weighted_prototype_with_capture};
    use super::*;
    use crate::Curve;
    use ndarray::array;

    fn fixture() -> (Vec<Curve>, RirPrototypeConfig, CaptureProvenance) {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.0, 0.06, 0.0],
            [0.0, 0.0, 0.06],
        ];
        let curves = [0.0, 10.0, 0.0, 10.0]
            .into_iter()
            .map(|level| Curve {
                freq: array![100.0, 1000.0, 10000.0],
                spl: array![level, level, level],
                ..Default::default()
            })
            .collect();
        let config = RirPrototypeConfig {
            reference_position: [0.0, 0.0, 0.0],
            source_position: [1.0, 0.0, 0.0],
            microphone_positions: positions.clone(),
            distance_mode: DistanceWeightMode::Uniform,
            directivity: DirectivityModel::SphericalHead { radius_m: 0.0875 },
            frequency_dependent_directivity: true,
        };
        use autoeq_core::capture_provenance::{
            CaptureCorrection, CaptureReflectionReport, CaptureTakeProvenance,
        };
        let takes = positions
            .iter()
            .enumerate()
            .map(|(index, position)| CaptureTakeProvenance {
                seat_id: None,
                microphone_id: format!("mic-{index}"),
                device_id: "aggregate".into(),
                offset_samples: Some(100.0),
                skew_ppm: Some(0.0),
                residual_uncertainty_us: Some(10.0),
                correction_applied: CaptureCorrection::Resampled,
                timing_reference_id: Some("fixed".into()),
                calibration_id: "frozen".into(),
                gain_db: 0.0,
                calibration_orientation: "on_axis".into(),
                position_m: *position,
                position_uncertainty_mm: 0.5,
                preserves_acoustic_delay: true,
                quality_passed: true,
            })
            .collect();
        let event = |direction, energies: [f64; 4]| CaptureArrival {
            arrival_ms: 10.0,
            relative_ms: 5.0,
            level_db: -6.0,
            microphone_energy_db: energies.to_vec(),
            direction: Some(direction),
            mirror_ambiguous: false,
            residual_samples: Some(0.1),
            band_hz: Some([300.0, 1200.0]),
            issues: Vec::new(),
        };
        let capture = CaptureProvenance {
            geometry: CaptureGeometry::Compact,
            takes,
            reflection_report: Some(CaptureReflectionReport {
                source_id: "left".into(),
                issues: Vec::new(),
                direct_sound: Some(event([1.0, 0.0, 0.0], [0.0; 4])),
                early_reflections: vec![event([-1.0, 0.0, 0.0], [-20.0, 0.0, -20.0, 0.0])],
            }),
        };
        (curves, config, capture)
    }

    #[test]
    fn measured_reflection_energy_changes_only_supported_bins() {
        let (curves, config, capture) = fixture();
        let baseline = build_weighted_prototype(&curves, &config).unwrap();
        let measured =
            build_weighted_prototype_with_capture(&curves, &config, Some(&capture)).unwrap();
        assert_eq!(measured.measured_direction_bins, vec![1; 4]);
        assert!(measured.weights[[0, 1]] > baseline.weights[[0, 1]]);
        assert!(measured.curve.spl[1] < baseline.curve.spl[1]);
        for bin in [0, 2] {
            assert_eq!(measured.curve.spl[bin], baseline.curve.spl[bin]);
        }
        assert!(measured.curve.phase.is_none());
    }

    #[test]
    fn planar_geometry_cannot_claim_a_resolved_measured_direction() {
        let (curves, mut config, mut capture) = fixture();
        config.microphone_positions[3] = [0.06, 0.06, 0.0];
        capture.takes[3].position_m = config.microphone_positions[3];
        let baseline = build_weighted_prototype(&curves, &config).unwrap();
        let result =
            build_weighted_prototype_with_capture(&curves, &config, Some(&capture)).unwrap();
        assert_eq!(result.measured_direction_bins, vec![0; 4]);
        assert_eq!(result.weights, baseline.weights);
    }

    #[test]
    fn unknown_and_ambiguous_evidence_retains_geometric_weights() {
        let (curves, config, capture) = fixture();
        let baseline = build_weighted_prototype(&curves, &config).unwrap();
        let mut cases = Vec::new();
        let mut limited = capture.clone();
        for take in &mut limited.takes {
            take.residual_uncertainty_us = Some(20.0);
        }
        cases.push(limited);
        let mut unknown = capture.clone();
        unknown.takes[0].residual_uncertainty_us = None;
        cases.push(unknown);
        let mut unknown = capture.clone();
        unknown
            .reflection_report
            .as_mut()
            .unwrap()
            .direct_sound
            .as_mut()
            .unwrap()
            .microphone_energy_db
            .clear();
        cases.push(unknown);
        let mut ambiguous = capture;
        let report = ambiguous.reflection_report.as_mut().unwrap();
        report.direct_sound.as_mut().unwrap().mirror_ambiguous = true;
        report.early_reflections[0].mirror_ambiguous = true;
        cases.push(ambiguous);
        for capture in cases {
            let result =
                build_weighted_prototype_with_capture(&curves, &config, Some(&capture)).unwrap();
            assert_eq!(result.measured_direction_bins, vec![0; 4]);
            assert_eq!(result.weights, baseline.weights);
        }
    }
}
