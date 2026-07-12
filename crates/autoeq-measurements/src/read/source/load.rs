use super::measurement_ref::MeasurementRef;
use super::measurement_source::MeasurementSource;
use crate::Curve;
use crate::read::{interpolate_log_space, read_curve_from_csv};
use ndarray::Array1;
use std::error::Error;
use std::path::PathBuf;

/// Load a single measurement from a file or inline data
pub fn load_measurement(measurement: &MeasurementRef) -> Result<Curve, Box<dyn Error>> {
    let curve = match measurement {
        MeasurementRef::Path(path) => {
            read_curve_from_csv(path).map_err(|error| -> Box<dyn Error> {
                format!("Failed to load measurement '{}': {error}", path.display()).into()
            })?
        }
        MeasurementRef::Named { path, .. } => {
            read_curve_from_csv(path).map_err(|error| -> Box<dyn Error> {
                format!("Failed to load measurement '{}': {error}", path.display()).into()
            })?
        }
        MeasurementRef::Inline(inline) => {
            // If inline data is empty but csv_path is provided, load from CSV
            if inline.frequencies.is_empty() || inline.magnitude_db.is_empty() {
                if let Some(ref csv_path) = inline.csv_path {
                    read_curve_from_csv(&PathBuf::from(csv_path))?
                } else {
                    return Err(format!(
                        "Inline measurement has empty data and no csv_path to fall back to (name: {:?})",
                        inline.name
                    )
                    .into());
                }
            } else {
                if inline.frequencies.len() != inline.magnitude_db.len() {
                    return Err(format!(
                        "Inline measurement has mismatched lengths: {} frequencies, {} magnitude values",
                        inline.frequencies.len(),
                        inline.magnitude_db.len()
                    )
                    .into());
                }

                let phase = inline.phase_deg.as_ref().and_then(|p| {
                    if p.len() != inline.frequencies.len() {
                        log::debug!(
                            "Warning: phase array length ({}) doesn't match frequencies ({}), ignoring phase",
                            p.len(),
                            inline.frequencies.len()
                        );
                        None
                    } else {
                        Some(Array1::from(p.clone()))
                    }
                });

                Curve {
                    freq: Array1::from(inline.frequencies.clone()),
                    spl: Array1::from(inline.magnitude_db.clone()),
                    phase,
                    ..Default::default()
                }
            }
        }
    };
    curve.validate("measurement")?;
    Ok(curve)
}

/// Load individual measurement curves from a source without averaging.
///
/// - `Single` → returns `vec![curve]`
/// - `Multiple` → loads all curves, interpolates to first curve's frequency grid
/// - `InMemory` → returns `vec![curve]`
pub fn load_source_individual(source: &MeasurementSource) -> Result<Vec<Curve>, Box<dyn Error>> {
    match source {
        MeasurementSource::Single(s) => {
            let curve = load_measurement(&s.measurement)?;
            Ok(vec![curve])
        }
        MeasurementSource::InMemory(curve) => {
            curve.validate("in-memory measurement")?;
            Ok(vec![curve.clone()])
        }
        MeasurementSource::InMemoryMultiple(curves) => {
            if curves.is_empty() {
                return Err("In-memory measurement list is empty".into());
            }
            for (index, curve) in curves.iter().enumerate() {
                curve.validate(&format!("in-memory measurement {index}"))?;
            }
            Ok(curves.clone())
        }
        MeasurementSource::Multiple(m) => {
            if m.measurements.is_empty() {
                return Err("Measurement list is empty".into());
            }

            let mut curves = Vec::new();
            for r in &m.measurements {
                match load_measurement(r) {
                    Ok(c) => curves.push(c),
                    Err(e) => {
                        let name = r
                            .path()
                            .map(|p| p.display().to_string())
                            .or_else(|| r.name().map(String::from))
                            .unwrap_or_else(|| "inline".to_string());
                        log::debug!("Warning: failed to load measurement {}: {}", name, e)
                    }
                }
            }

            if curves.is_empty() {
                return Err("No valid measurements loaded".into());
            }

            // Interpolate all curves to the first curve's frequency grid
            let ref_freqs = curves[0].freq.clone();
            let mut result = vec![curves[0].clone()];
            for curve in &curves[1..] {
                result.push(interpolate_log_space(&ref_freqs, curve));
            }
            Ok(result)
        }
    }
}

/// Average a set of curves in the power domain after interpolating to the
/// first curve's frequency grid.
///
/// All curves must be non-empty; callers are responsible for checking.
fn average_curves_power_domain(curves: &[Curve]) -> Curve {
    let ref_curve = &curves[0];
    let freqs = ref_curve.freq.clone();

    let mut power_sum = Array1::<f64>::zeros(freqs.len());
    let mut coherence_sum = curves
        .iter()
        .all(|curve| curve.coherence.is_some())
        .then(|| Array1::<f64>::zeros(freqs.len()));

    for curve in curves {
        let interpolated = interpolate_log_space(&freqs, curve);
        // Convert SPL to power (proportional to pressure squared)
        // Power = 10^(SPL/10)
        let p = interpolated.spl.mapv(|spl| 10.0_f64.powf(spl / 10.0));
        power_sum = power_sum + p;
        if let (Some(sum), Some(coherence)) =
            (coherence_sum.as_mut(), interpolated.coherence.as_ref())
        {
            *sum = sum.clone() + coherence;
        }
    }

    let avg_power = power_sum / (curves.len() as f64);
    let avg_spl = avg_power.mapv(|p| 10.0 * p.log10());
    let phase = ref_curve.phase.clone();
    let coherence = coherence_sum.map(|sum| sum / curves.len() as f64);

    Curve {
        freq: freqs,
        spl: avg_spl,
        phase,
        coherence,
        ..Default::default()
    }
}

/// Load measurement(s) from a source and average if necessary
pub fn load_source(source: &MeasurementSource) -> Result<Curve, Box<dyn Error>> {
    match source {
        MeasurementSource::Single(s) => load_measurement(&s.measurement),
        MeasurementSource::InMemory(curve) => Ok(curve.clone()),
        MeasurementSource::InMemoryMultiple(curves) => {
            if curves.is_empty() {
                return Err("InMemoryMultiple has no curves".into());
            }
            if curves.len() == 1 {
                return Ok(curves[0].clone());
            }
            Ok(average_curves_power_domain(curves))
        }
        MeasurementSource::Multiple(m) => {
            if m.measurements.is_empty() {
                return Err("Measurement list is empty".into());
            }

            // Load all curves
            let mut curves = Vec::new();
            for r in &m.measurements {
                match load_measurement(r) {
                    Ok(c) => curves.push(c),
                    Err(e) => {
                        let name = r
                            .path()
                            .map(|p| p.display().to_string())
                            .or_else(|| r.name().map(String::from))
                            .unwrap_or_else(|| "inline".to_string());
                        log::debug!("Warning: failed to load measurement {}: {}", name, e)
                    }
                }
            }

            if curves.is_empty() {
                return Err("No valid measurements loaded".into());
            }

            Ok(average_curves_power_domain(&curves))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::inline_measurement::InlineMeasurement;
    use super::super::measurement_ref::MeasurementRef;
    use super::super::measurement_single::MeasurementSingle;
    use super::super::measurement_source::MeasurementSource;
    use super::super::types::MeasurementMultiple;
    use super::*;
    use ndarray::Array1;

    fn sample_inline() -> InlineMeasurement {
        InlineMeasurement {
            frequencies: vec![100.0, 1000.0, 10000.0],
            magnitude_db: vec![80.0, 75.0, 70.0],
            phase_deg: Some(vec![0.0, 45.0, 90.0]),
            name: Some("inline".to_string()),
            wav_path: None,
            csv_path: None,
        }
    }

    fn sample_curve(spl_offset: f64) -> Curve {
        Curve {
            freq: Array1::from(vec![100.0, 1000.0, 10000.0]),
            spl: Array1::from(vec![
                80.0 + spl_offset,
                75.0 + spl_offset,
                70.0 + spl_offset,
            ]),
            phase: None,
            ..Default::default()
        }
    }

    #[test]
    fn load_measurement_inline_ok() {
        let m = MeasurementRef::Inline(sample_inline());
        let curve = load_measurement(&m).unwrap();
        assert_eq!(curve.freq.len(), 3);
        assert_eq!(curve.spl[0], 80.0);
        assert!(curve.phase.is_some());
    }

    #[test]
    fn load_measurement_inline_ignores_mismatched_phase() {
        let mut inline = sample_inline();
        inline.phase_deg = Some(vec![0.0, 45.0]);
        let curve = load_measurement(&MeasurementRef::Inline(inline)).unwrap();
        assert!(curve.phase.is_none());
    }

    #[test]
    fn load_measurement_inline_rejects_mismatched_lengths() {
        let mut inline = sample_inline();
        inline.magnitude_db.push(65.0);
        let result = load_measurement(&MeasurementRef::Inline(inline));
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("mismatched lengths")
        );
    }

    #[test]
    fn load_measurement_rejects_single_bin_and_non_finite_inline_data() {
        for (frequencies, magnitude_db) in [
            (vec![100.0], vec![80.0]),
            (vec![100.0, f64::NAN], vec![80.0, 81.0]),
            (vec![100.0, f64::INFINITY], vec![80.0, 81.0]),
            (vec![100.0, 1_000.0], vec![80.0, f64::NEG_INFINITY]),
        ] {
            let inline = InlineMeasurement {
                frequencies,
                magnitude_db,
                phase_deg: None,
                name: Some("invalid".to_string()),
                wav_path: None,
                csv_path: None,
            };
            assert!(load_measurement(&MeasurementRef::Inline(inline)).is_err());
        }
    }

    #[test]
    fn load_source_rejects_invalid_in_memory_curve() {
        let source = MeasurementSource::InMemory(Curve {
            freq: Array1::from_vec(vec![100.0, 1_000.0]),
            spl: Array1::from_vec(vec![80.0]),
            ..Default::default()
        });
        assert!(load_source_individual(&source).is_err());
    }

    #[test]
    fn load_measurement_inline_empty_rejects_without_csv_path() {
        let inline = InlineMeasurement {
            frequencies: vec![],
            magnitude_db: vec![],
            phase_deg: None,
            name: Some("empty".to_string()),
            wav_path: None,
            csv_path: None,
        };
        let result = load_measurement(&MeasurementRef::Inline(inline));
        assert!(result.is_err());
    }

    #[test]
    fn load_measurement_named_missing_file_returns_error() {
        let path = std::path::PathBuf::from("/tmp/does_not_exist_abc123.csv");
        let m = MeasurementRef::Named {
            path: path.clone(),
            name: Some("missing".to_string()),
        };
        let error = load_measurement(&m).expect_err("missing measurement must fail");
        assert!(
            error.to_string().contains(&path.display().to_string()),
            "missing path was not preserved in error: {error}"
        );
    }

    #[test]
    fn load_source_single_inline() {
        let source = MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Inline(sample_inline()),
            speaker_name: Some("L".to_string()),
        });
        let curve = load_source(&source).unwrap();
        assert_eq!(curve.freq.len(), 3);
        assert_eq!(source.speaker_name(), Some("L"));
    }

    #[test]
    fn load_source_in_memory_returns_clone() {
        let curve = sample_curve(0.0);
        let source = MeasurementSource::InMemory(curve.clone());
        let loaded = load_source(&source).unwrap();
        assert_eq!(loaded.spl[0], curve.spl[0]);
    }

    #[test]
    fn load_source_in_memory_multiple_averages() {
        let c1 = sample_curve(0.0);
        let c2 = sample_curve(3.0);
        let source = MeasurementSource::InMemoryMultiple(vec![c1, c2]);
        let avg = load_source(&source).unwrap();
        assert_eq!(avg.freq.len(), 3);
        // Averaging in power domain: 3 dB difference => average ~81.76 dB at first point
        let expected =
            10.0 * ((10.0_f64.powf(80.0 / 10.0) + 10.0_f64.powf(83.0 / 10.0)) / 2.0).log10();
        assert!((avg.spl[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn load_source_individual_multiple_interpolates_to_first_grid() {
        let c1 = sample_curve(0.0);
        let mut c2 = sample_curve(3.0);
        // Different grid to exercise interpolation path
        c2.freq = Array1::from(vec![120.0, 1100.0, 9000.0]);
        let source = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![
                MeasurementRef::Inline(InlineMeasurement {
                    frequencies: c1.freq.to_vec(),
                    magnitude_db: c1.spl.to_vec(),
                    phase_deg: None,
                    name: None,
                    wav_path: None,
                    csv_path: None,
                }),
                MeasurementRef::Inline(InlineMeasurement {
                    frequencies: c2.freq.to_vec(),
                    magnitude_db: c2.spl.to_vec(),
                    phase_deg: None,
                    name: None,
                    wav_path: None,
                    csv_path: None,
                }),
            ],
            speaker_name: None,
        });
        let curves = load_source_individual(&source).unwrap();
        assert_eq!(curves.len(), 2);
        assert_eq!(curves[0].freq[0], 100.0);
    }

    #[test]
    fn load_source_individual_empty_multiple_errors() {
        let source = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![],
            speaker_name: None,
        });
        assert!(load_source_individual(&source).is_err());
    }

    #[test]
    fn load_source_multiple_skips_failed_measurements() {
        let source = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![
                MeasurementRef::Inline(sample_inline()),
                MeasurementRef::Path(std::path::PathBuf::from("/tmp/missing.csv")),
            ],
            speaker_name: None,
        });
        let curve = load_source(&source).unwrap();
        assert_eq!(curve.freq.len(), 3);
    }
}
