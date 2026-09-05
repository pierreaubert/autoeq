use super::{MeasurementRef, MeasurementSource};
use crate::Curve;
use crate::read::{interpolate_log_space, read_curve_from_csv};
use ndarray::Array1;
use std::error::Error;
use std::path::PathBuf;

fn measurement_identity(measurement: &MeasurementRef) -> String {
    measurement
        .path()
        .map(|path| path.display().to_string())
        .or_else(|| measurement.name().map(String::from))
        .unwrap_or_else(|| "inline".to_string())
}

fn load_measurements_strict(measurements: &[MeasurementRef]) -> Result<Vec<Curve>, Box<dyn Error>> {
    let mut curves = Vec::with_capacity(measurements.len());
    let mut failures = Vec::new();
    for measurement in measurements {
        match load_measurement(measurement) {
            Ok(curve) => curves.push(curve),
            Err(error) => failures.push(format!("{}: {error}", measurement_identity(measurement))),
        }
    }
    if failures.is_empty() {
        Ok(curves)
    } else {
        Err(format!(
            "Failed to load {} of {} measurements: {}",
            failures.len(),
            measurements.len(),
            failures.join("; ")
        )
        .into())
    }
}

fn grids_match(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| (a - b).abs() <= 1e-9)
}

/// Original measured support of one input curve, in Hz.
///
/// Curves are validated strictly increasing before this is read, so
/// `min_hz`/`max_hz` are the first/last grid points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveSupport {
    pub min_hz: f64,
    pub max_hz: f64,
}

/// Return the validated support of one curve.
pub fn curve_support(curve: &Curve) -> CurveSupport {
    CurveSupport {
        min_hz: curve.freq[0],
        max_hz: curve.freq[curve.freq.len() - 1],
    }
}

/// Curves aligned to a shared grid plus the evidence that grid is real.
///
/// The grid is always a subset of the physical intersection of all inputs,
/// so no output bin is extrapolated. `validity_mask[curve][bin]` is
/// therefore all-`true` by construction; it is retained so downstream
/// consumers (e.g. the optimiser) can mask without recomputing support.
#[derive(Debug, Clone)]
pub struct AlignedCurves {
    pub curves: Vec<Curve>,
    /// Physical intersection of all input supports, in Hz.
    pub overlap_hz: (f64, f64),
    /// Original per-curve support, in input order.
    pub support_hz: Vec<CurveSupport>,
    /// Per-curve, per-bin validity on the shared grid.
    pub validity_mask: Vec<Vec<bool>>,
}

/// Physical intersection of all curve supports.
///
/// Returns `Err` when the intersection is empty: disjoint measurements
/// must fail here instead of being extrapolated into fake agreement.
fn overlap_range(curves: &[Curve], context: &str) -> Result<(f64, f64), Box<dyn Error>> {
    let mut lower = f64::NEG_INFINITY;
    let mut upper = f64::INFINITY;
    for curve in curves {
        lower = lower.max(curve.freq[0]);
        upper = upper.min(curve.freq[curve.freq.len() - 1]);
    }
    if lower < upper {
        Ok((lower, upper))
    } else {
        Err(format!(
            "{context} has no overlapping frequency support \
             (intersection [{lower}, {upper}] Hz is empty); \
             refusing to extrapolate disjoint measurements",
        )
        .into())
    }
}

/// Order-independent shared grid: the sorted union of all input grid points
/// clipped to the physical overlap.
///
/// Union + sort + dedup is a pure function of the input *set*, so curve
/// order cannot change the accepted range or the resampled values.
fn common_overlap_grid(
    curves: &[Curve],
    overlap: (f64, f64),
    context: &str,
) -> Result<Array1<f64>, Box<dyn Error>> {
    let (lower, upper) = overlap;
    let mut grid: Vec<f64> = curves
        .iter()
        .flat_map(|curve| curve.freq.iter().copied())
        .filter(|freq| *freq >= lower - 1e-9 && *freq <= upper + 1e-9)
        .map(|freq| freq.clamp(lower, upper))
        .collect();
    grid.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("validated frequencies are finite")
    });
    grid.dedup_by(|a, b| (*a - *b).abs() <= 1e-9);
    if grid.len() < 2 {
        return Err(format!(
            "{context} has insufficient overlapping support \
             ([{lower}, {upper}] Hz yields {} shared grid point(s)); \
             need at least two",
            grid.len()
        )
        .into());
    }
    Ok(Array1::from(grid))
}

/// Validate, intersect, and resample a set of curves onto their shared
/// physical support. See [`AlignedCurves`].
fn load_aligned_curves(
    curves: &[Curve],
    context: &str,
) -> Result<AlignedCurves, Box<dyn Error>> {
    let Some(_) = curves.first() else {
        return Err(format!("{context} is empty").into());
    };
    for (index, curve) in curves.iter().enumerate() {
        curve.validate(&format!("{context} {index}"))?;
    }

    let overlap = overlap_range(curves, context)?;
    let grid = common_overlap_grid(curves, overlap, context)?;
    // Every grid point lies inside every curve's support by construction,
    // so `interpolate_log_space` below only interpolates — the endpoint-slope
    // extrapolation in the core transform is never reached via this path.
    let aligned: Vec<Curve> = curves
        .iter()
        .map(|curve| {
            if grids_match(&grid, &curve.freq) {
                curve.clone()
            } else {
                interpolate_log_space(&grid, curve)
            }
        })
        .collect();
    let validity_mask = vec![vec![true; grid.len()]; aligned.len()];
    let support_hz = curves.iter().map(curve_support).collect();
    Ok(AlignedCurves {
        curves: aligned,
        overlap_hz: overlap,
        support_hz,
        validity_mask,
    })
}

/// Load individual measurement curves with their overlap evidence.
///
/// Additive companion to [`load_source_individual`]: same alignment, plus
/// the physical overlap, the original per-curve supports, and the validity
/// mask. Other crates should prefer this when they need to constrain
/// downstream output (e.g. optimiser grids) to real support.
pub fn load_source_individual_with_support(
    source: &MeasurementSource,
) -> Result<AlignedCurves, Box<dyn Error>> {
    match source {
        MeasurementSource::Single(s) => {
            let curve = load_measurement(&s.measurement)?;
            load_aligned_curves(std::slice::from_ref(&curve), "measurement")
        }
        MeasurementSource::InMemory(curve) => {
            curve.validate("in-memory measurement")?;
            load_aligned_curves(std::slice::from_ref(curve), "in-memory measurement")
        }
        MeasurementSource::InMemoryMultiple(curves) => {
            load_aligned_curves(curves, "in-memory measurement")
        }
        MeasurementSource::Multiple(m) => {
            if m.measurements.is_empty() {
                return Err("Measurement list is empty".into());
            }
            let curves = load_measurements_strict(&m.measurements)?;
            load_aligned_curves(&curves, "measurement")
        }
    }
}

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
    load_source_individual_with_support(source).map(|aligned| aligned.curves)
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
    let preserve_phase = curves.iter().all(|curve| {
        curve
            .phase
            .as_ref()
            .is_some_and(|phase| phase.len() == curve.freq.len())
    });
    let mut phase_real_sum = preserve_phase.then(|| Array1::<f64>::zeros(freqs.len()));
    let mut phase_imag_sum = preserve_phase.then(|| Array1::<f64>::zeros(freqs.len()));

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
        if let (Some(real_sum), Some(imag_sum), Some(phase)) = (
            phase_real_sum.as_mut(),
            phase_imag_sum.as_mut(),
            interpolated.phase.as_ref(),
        ) {
            for (((real, imag), &spl), &phase_deg) in real_sum
                .iter_mut()
                .zip(imag_sum.iter_mut())
                .zip(interpolated.spl.iter())
                .zip(phase.iter())
            {
                let amplitude = 10.0_f64.powf(spl / 20.0);
                let phase_rad = phase_deg.to_radians();
                *real += amplitude * phase_rad.cos();
                *imag += amplitude * phase_rad.sin();
            }
        }
    }

    let avg_power = power_sum / (curves.len() as f64);
    let avg_spl = avg_power.mapv(|p| 10.0 * p.log10());
    let coherence = coherence_sum.map(|sum| sum / curves.len() as f64);
    let phase = phase_real_sum.zip(phase_imag_sum).map(|(real, imag)| {
        Array1::from_iter(
            real.iter()
                .zip(imag.iter())
                .map(|(&real, &imag)| imag.atan2(real).to_degrees()),
        )
    });

    Curve {
        freq: freqs,
        spl: avg_spl,
        phase,
        coherence,
        ..Default::default()
    }
}

/// Load measurement(s) once and return both the representative response and
/// the aligned individual responses.
pub fn load_source_with_individual(
    source: &MeasurementSource,
) -> Result<(Curve, Vec<Curve>), Box<dyn Error>> {
    let curves = load_source_individual(source)?;
    let representative = if curves.len() == 1 {
        curves[0].clone()
    } else {
        average_curves_power_domain(&curves)
    };
    Ok((representative, curves))
}

/// Load measurement(s) from a source and average if necessary.
pub fn load_source(source: &MeasurementSource) -> Result<Curve, Box<dyn Error>> {
    load_source_with_individual(source).map(|(representative, _)| representative)
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
        assert!(load_source(&source).is_err());
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
    fn load_source_with_individual_reuses_aligned_curves() {
        let first = sample_curve(0.0);
        let second = sample_curve(3.0);
        let source = MeasurementSource::InMemoryMultiple(vec![first, second]);

        let (representative, individual) = load_source_with_individual(&source).unwrap();

        assert_eq!(individual.len(), 2);
        assert_eq!(representative.freq, individual[0].freq);
        let expected =
            10.0 * ((10.0_f64.powf(80.0 / 10.0) + 10.0_f64.powf(83.0 / 10.0)) / 2.0).log10();
        assert!((representative.spl[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn load_source_power_average_invalidates_position_specific_phase() {
        let mut first = sample_curve(0.0);
        first.phase = Some(Array1::from_vec(vec![10.0, 20.0, 30.0]));
        first.min_phase = Some(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        first.excess_phase = Some(Array1::from_vec(vec![9.0, 18.0, 27.0]));
        first.excess_delay_ms = Some(1.0);
        let source = MeasurementSource::InMemoryMultiple(vec![first, sample_curve(3.0)]);

        let average = load_source(&source).unwrap();

        assert!(average.phase.is_none());
        assert!(average.min_phase.is_none());
        assert!(average.excess_phase.is_none());
        assert!(average.excess_delay_ms.is_none());
    }

    #[test]
    fn load_source_power_average_preserves_circular_phase_when_all_positions_have_phase() {
        let mut first = sample_curve(0.0);
        first.phase = Some(Array1::from_vec(vec![170.0, 20.0, -45.0]));
        let mut second = sample_curve(0.0);
        second.phase = Some(Array1::from_vec(vec![-170.0, 40.0, -15.0]));

        let source = MeasurementSource::InMemoryMultiple(vec![first, second]);
        let average = load_source(&source).unwrap();
        let phase = average
            .phase
            .expect("phase must survive all-phase averaging");

        assert!((phase[0].abs() - 180.0).abs() < 1e-9);
        assert!((phase[1] - 30.0).abs() < 1e-9);
        assert!((phase[2] + 30.0).abs() < 1e-9);
    }

    #[test]
    fn load_source_individual_multiple_uses_overlap_grid() {
        let c1 = sample_curve(0.0);
        let mut c2 = sample_curve(3.0);
        // Different grid to exercise interpolation path
        c2.freq = Array1::from(vec![120.0, 1100.0, 9000.0]);
        let inline = |freq: &Array1<f64>, spl: &Array1<f64>| {
            MeasurementRef::Inline(InlineMeasurement {
                frequencies: freq.to_vec(),
                magnitude_db: spl.to_vec(),
                phase_deg: None,
                name: None,
                wav_path: None,
                csv_path: None,
            })
        };
        let forward = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![inline(&c1.freq, &c1.spl), inline(&c2.freq, &c2.spl)],
            speaker_name: None,
        });
        let backward = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![inline(&c2.freq, &c2.spl), inline(&c1.freq, &c1.spl)],
            speaker_name: None,
        });
        // Physical intersection is [120, 9000]; the shared grid is the
        // sorted union clipped to it — identical for both curve orders.
        let expected = vec![120.0, 1000.0, 1100.0, 9000.0];
        for source in [forward, backward] {
            let curves = load_source_individual(&source).unwrap();
            assert_eq!(curves.len(), 2);
            assert_eq!(curves[0].freq.to_vec(), expected);
            assert_eq!(curves[1].freq.to_vec(), expected);
        }
    }

    #[test]
    fn load_source_individual_in_memory_multiple_uses_overlap_grid() {
        let first = sample_curve(0.0);
        let second = Curve {
            freq: Array1::from_vec(vec![120.0, 1100.0, 9000.0]),
            spl: Array1::from_vec(vec![83.0, 78.0, 73.0]),
            ..Default::default()
        };
        let expected = vec![120.0, 1000.0, 1100.0, 9000.0];
        for curves in [
            vec![first.clone(), second.clone()],
            vec![second.clone(), first.clone()],
        ] {
            let source = MeasurementSource::InMemoryMultiple(curves);
            let loaded = load_source_individual(&source).unwrap();
            assert_eq!(loaded.len(), 2);
            assert_eq!(loaded[0].freq.to_vec(), expected);
            assert_eq!(loaded[1].freq.to_vec(), expected);
        }
    }

    fn probe_curves() -> (Curve, Curve) {
        // BUG1 repro: probe A spans [20, 200], probe B spans [100, 200].
        let probe_a = Curve {
            freq: Array1::from_vec(vec![20.0, 100.0, 200.0]),
            spl: Array1::from_vec(vec![70.0, 80.0, 90.0]),
            ..Default::default()
        };
        let probe_b = Curve {
            freq: Array1::from_vec(vec![100.0, 200.0]),
            spl: Array1::from_vec(vec![80.0, 86.0]),
            ..Default::default()
        };
        (probe_a, probe_b)
    }

    #[test]
    fn partial_overlap_never_extrapolates_outside_support() {
        let (probe_a, probe_b) = probe_curves();
        for curves in [
            vec![probe_a.clone(), probe_b.clone()],
            vec![probe_b.clone(), probe_a.clone()],
        ] {
            let source = MeasurementSource::InMemoryMultiple(curves);
            let loaded = load_source_individual(&source).unwrap();
            assert_eq!(loaded.len(), 2);
            // Shared grid is exactly the physical intersection [100, 200]:
            // no invented 20 Hz bin for probe B in either order.
            for curve in &loaded {
                assert_eq!(curve.freq.to_vec(), vec![100.0, 200.0]);
            }
            // Probe B keeps its measured values; probe A is interpolated.
            // (Both probes read 80 dB at 100 Hz, so identify B by its
            // full [80, 86] dB signature.)
            let b = loaded
                .iter()
                .find(|curve| {
                    (curve.spl[0] - 80.0).abs() < 1e-9 && (curve.spl[1] - 86.0).abs() < 1e-9
                })
                .expect("probe B must keep its measured [80, 86] dB values");
            assert_eq!(b.freq.to_vec(), vec![100.0, 200.0]);
        }
    }

    #[test]
    fn partial_overlap_representative_stays_on_real_support() {
        let (probe_a, probe_b) = probe_curves();
        for curves in [
            vec![probe_a.clone(), probe_b.clone()],
            vec![probe_b.clone(), probe_a.clone()],
        ] {
            let source = MeasurementSource::InMemoryMultiple(curves);
            let representative = load_source(&source).unwrap();
            assert!(representative.freq[0] >= 100.0 - 1e-9);
            assert!(representative.freq[representative.freq.len() - 1] <= 200.0 + 1e-9);
        }
    }

    #[test]
    fn disjoint_spans_are_rejected_in_both_orders() {
        let low = Curve {
            freq: Array1::from_vec(vec![20.0, 40.0]),
            spl: Array1::from_vec(vec![80.0, 81.0]),
            ..Default::default()
        };
        let high = Curve {
            freq: Array1::from_vec(vec![100.0, 200.0]),
            spl: Array1::from_vec(vec![80.0, 81.0]),
            ..Default::default()
        };
        for curves in [vec![low.clone(), high.clone()], vec![high.clone(), low.clone()]] {
            let source = MeasurementSource::InMemoryMultiple(curves);
            for error in [
                load_source_individual(&source).unwrap_err(),
                load_source(&source).unwrap_err(),
            ] {
                let message = error.to_string();
                assert!(
                    message.contains("no overlapping frequency support"),
                    "unexpected error: {message}"
                );
            }
        }
    }

    #[test]
    fn touching_spans_with_one_shared_point_are_rejected() {
        let low = Curve {
            freq: Array1::from_vec(vec![20.0, 100.0]),
            spl: Array1::from_vec(vec![80.0, 81.0]),
            ..Default::default()
        };
        let high = Curve {
            freq: Array1::from_vec(vec![100.0, 200.0]),
            spl: Array1::from_vec(vec![81.0, 82.0]),
            ..Default::default()
        };
        let source = MeasurementSource::InMemoryMultiple(vec![low, high]);
        let error = load_source_individual(&source).unwrap_err();
        assert!(
            error.to_string().contains("overlapping"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn with_support_reports_overlap_and_original_supports() {
        let (probe_a, probe_b) = probe_curves();
        let source = MeasurementSource::InMemoryMultiple(vec![probe_a, probe_b]);
        let aligned = load_source_individual_with_support(&source).unwrap();
        assert_eq!(aligned.overlap_hz, (100.0, 200.0));
        assert_eq!(
            aligned.support_hz,
            vec![
                CurveSupport {
                    min_hz: 20.0,
                    max_hz: 200.0
                },
                CurveSupport {
                    min_hz: 100.0,
                    max_hz: 200.0
                },
            ]
        );
        assert_eq!(aligned.curves.len(), 2);
        for mask in &aligned.validity_mask {
            assert_eq!(mask, &vec![true, true]);
        }
    }

    fn write_csv(dir: &std::path::Path, name: &str, rows: &[(f64, f64)]) -> PathBuf {
        let path = dir.join(name);
        let mut contents = String::from("frequency,spl\n");
        for (freq, spl) in rows {
            contents.push_str(&format!("{freq},{spl}\n"));
        }
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn file_backed_partial_overlap_matches_in_memory() {
        let dir = std::env::temp_dir().join(format!(
            "autoeq_overlap_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path_a = write_csv(&dir, "a.csv", &[(20.0, 70.0), (100.0, 80.0), (200.0, 90.0)]);
        let path_b = write_csv(&dir, "b.csv", &[(100.0, 80.0), (200.0, 86.0)]);
        let source = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![
                MeasurementRef::Path(path_a.clone()),
                MeasurementRef::Path(path_b.clone()),
            ],
            speaker_name: None,
        });
        let reversed = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![
                MeasurementRef::Path(path_b),
                MeasurementRef::Path(path_a),
            ],
            speaker_name: None,
        });
        for source in [source, reversed] {
            let loaded = load_source_individual(&source).unwrap();
            assert_eq!(loaded[0].freq.to_vec(), vec![100.0, 200.0]);
            assert_eq!(loaded[1].freq.to_vec(), vec![100.0, 200.0]);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn file_backed_disjoint_spans_are_rejected() {
        let dir = std::env::temp_dir().join(format!(
            "autoeq_disjoint_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let low = write_csv(&dir, "low.csv", &[(20.0, 80.0), (40.0, 81.0)]);
        let high = write_csv(&dir, "high.csv", &[(100.0, 80.0), (200.0, 81.0)]);
        let source = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![MeasurementRef::Path(low), MeasurementRef::Path(high)],
            speaker_name: None,
        });
        let error = load_source_individual(&source).unwrap_err();
        assert!(
            error.to_string().contains("no overlapping frequency support"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_source_rejects_invalid_curve_inside_in_memory_multiple() {
        let invalid = Curve {
            freq: Array1::from_vec(vec![100.0, 1000.0]),
            spl: Array1::from_vec(vec![80.0]),
            ..Default::default()
        };
        let source = MeasurementSource::InMemoryMultiple(vec![sample_curve(0.0), invalid]);

        assert!(load_source_individual(&source).is_err());
        assert!(load_source(&source).is_err());
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
    fn load_source_multiple_rejects_partial_measurement_failure() {
        let source = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![
                MeasurementRef::Inline(sample_inline()),
                MeasurementRef::Path(std::path::PathBuf::from("/tmp/missing.csv")),
            ],
            speaker_name: None,
        });
        for error in [
            load_source(&source).unwrap_err(),
            load_source_individual(&source).unwrap_err(),
        ] {
            let message = error.to_string();
            assert!(message.contains("1 of 2"), "unexpected error: {message}");
            assert!(
                message.contains("missing.csv"),
                "unexpected error: {message}"
            );
        }
    }
}
