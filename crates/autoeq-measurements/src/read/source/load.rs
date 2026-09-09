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

/// Load a single measurement from a file or inline data.
///
/// Lenient default: a phase array whose length does not match the frequency
/// grid is dropped with a debug log (historical behavior). Phase-critical
/// workflows should use [`load_measurement_strict`] instead.
pub fn load_measurement(measurement: &MeasurementRef) -> Result<Curve, Box<dyn Error>> {
    load_measurement_with_policy(measurement, false)
}

/// Strict single-measurement loader for phase-critical workflows.
///
/// Unlike [`load_measurement`], a phase array whose length does not match
/// the frequency grid is an error here instead of a warn-and-drop: silently
/// discarding phase evidence is unacceptable when downstream phase handling
/// (delay estimation, coherent averaging) depends on its presence.
pub fn load_measurement_strict(
    measurement: &MeasurementRef,
) -> Result<Curve, Box<dyn Error>> {
    load_measurement_with_policy(measurement, true)
}

/// Single-measurement loader with an explicit phase-mismatch policy.
pub fn load_measurement_with_policy(
    measurement: &MeasurementRef,
    strict_phase: bool,
) -> Result<Curve, Box<dyn Error>> {
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

                let phase = match inline.phase_deg.as_ref() {
                    Some(p) if p.len() != inline.frequencies.len() => {
                        if strict_phase {
                            return Err(format!(
                                "Inline measurement phase array length ({}) doesn't match \
                                 frequencies ({}); strict phase mode refuses to silently drop it \
                                 (name: {:?})",
                                p.len(),
                                inline.frequencies.len(),
                                inline.name
                            )
                            .into());
                        }
                        log::debug!(
                            "Warning: phase array length ({}) doesn't match frequencies ({}), ignoring phase",
                            p.len(),
                            inline.frequencies.len()
                        );
                        None
                    }
                    Some(p) => Some(Array1::from(p.clone())),
                    None => None,
                };

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

/// Per-seat provenance carried through loading (additive, all optional
/// except the seat identity).
///
/// `load_source_detailed` fills in what loading actually knows — the seat id
/// (measurement name, else `seat-{index}`), the source path when file-backed,
/// and a phase-confidence estimate from mean coherence when present.
/// `calibration_id` and `delay_ms` stay `None` here: populating them is the
/// measurement contract of calibration-aware callers (mic calibration
/// tables, arrival/delay estimation). [`CoherentAverageContract`] gates on
/// them, so a coherent average can never silently run on uncalibrated seats.
#[derive(Debug, Clone, Default)]
pub struct SeatProvenance {
    /// Seat identity: measurement name, else `seat-{index}` in input order.
    pub seat_id: String,
    /// Source file path for file-backed measurements, if any.
    pub source_path: Option<String>,
    /// Microphone/chain calibration identity, when the caller provides one.
    pub calibration_id: Option<String>,
    /// Estimated propagation delay in milliseconds, when known.
    pub delay_ms: Option<f64>,
    /// Phase trust in `[0, 1]` (mean coherence when the curve carries one).
    pub phase_confidence: Option<f64>,
}

/// Explicit measurement contract for a coherent (complex-mean) average.
///
/// A coherent average is only meaningful for calibrated, time-aligned seats
/// with trustworthy phase. The loader enforces that here instead of
/// exposing an always-on hybrid: `seats` must identify every input curve in
/// order, each seat's `phase_confidence` must clear `min_phase_confidence`,
/// and — unless the caller opts out — every seat must carry a
/// `calibration_id`.
#[derive(Debug, Clone)]
pub struct CoherentAverageContract {
    /// One entry per input curve, in input order.
    pub seats: Vec<SeatProvenance>,
    /// Minimum accepted `phase_confidence` per seat (missing counts as 0).
    pub min_phase_confidence: f64,
    /// When true (default), every seat must carry a `calibration_id`.
    pub require_calibration: bool,
}

impl Default for CoherentAverageContract {
    fn default() -> Self {
        Self {
            seats: Vec::new(),
            min_phase_confidence: 0.8,
            require_calibration: true,
        }
    }
}

/// Fully attributed multi-seat load: spatial magnitude, primary seat, and
/// (only under an explicit contract) a coherent average.
///
/// - `spatial_rms`: power-domain (RMS) magnitude average. `phase` is always
///   `None` — an RMS magnitude must never be paired with an averaged angle.
/// - `primary_seat`: the first input curve, untouched, carrying that seat's
///   measured complex response for phase-sensitive work (delay estimation,
///   GD optimisation).
/// - `coherent`: complex-mean magnitude *and* angle, present only when the
///   caller passes a [`CoherentAverageContract`] that all seats satisfy.
#[derive(Debug, Clone)]
pub struct DetailedLoad {
    pub spatial_rms: Curve,
    pub primary_seat: Curve,
    pub coherent: Option<Curve>,
    pub individual: Vec<Curve>,
    pub seats: Vec<SeatProvenance>,
    /// Physical intersection all outputs are constrained to, in Hz.
    pub overlap_hz: (f64, f64),
    /// Original per-curve support, in input order.
    pub support_hz: Vec<CurveSupport>,
    /// Per-bin validity of the shared grid (all true by construction).
    pub validity_mask: Vec<bool>,
}

/// Spatial (power-domain RMS) magnitude average.
///
/// The output carries **no phase**: combining an RMS magnitude with an
/// averaged angle produces a hybrid that looks like an ordinary
/// phase-bearing `Curve` (equal 80 dB SPL at 0° and 180° would keep 80 dB
/// with a numerically unstable angle), so phase presence can never be used
/// as proof of coherence. Curves must already share a grid — callers use
/// the overlap-aligned output; the interpolation here is an identity for
/// aligned inputs and never extrapolates for them.
fn average_curves_power_domain(curves: &[Curve]) -> Curve {
    let ref_curve = &curves[0];
    let freqs = ref_curve.freq.clone();
    debug_assert!(curves.iter().all(|curve| grids_match(&freqs, &curve.freq)));

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
    let coherence = coherence_sum.map(|sum| sum / curves.len() as f64);

    Curve {
        freq: freqs,
        spl: avg_spl,
        phase: None,
        coherence,
        ..Default::default()
    }
}

/// Coherent (complex-pressure mean) average under an explicit contract.
///
/// Magnitude **and** angle both come from the mean complex pressure, so the
/// result is a genuine single complex response — never an RMS magnitude
/// with a grafted-on angle. Fails when any curve lacks phase, when the
/// contract's seat list does not cover every curve, when a seat's phase
/// confidence is below the contract minimum, when calibration is required
/// but missing, or when seats cancel completely (non-finite magnitude).
pub fn coherent_average_measurement(
    curves: &[Curve],
    contract: &CoherentAverageContract,
) -> Result<Curve, Box<dyn Error>> {
    if !contract.min_phase_confidence.is_finite()
        || !(0.0..=1.0).contains(&contract.min_phase_confidence)
    {
        return Err(
            "coherent average minimum phase confidence must be finite and in [0, 1]".into(),
        );
    }
    if curves.is_empty() {
        return Err("coherent average needs at least one curve".into());
    }
    if contract.seats.len() != curves.len() {
        return Err(format!(
            "coherent average needs one contracted seat per curve ({} seats, {} curves)",
            contract.seats.len(),
            curves.len()
        )
        .into());
    }
    let freqs = curves[0].freq.clone();
    if !curves.iter().all(|curve| grids_match(&freqs, &curve.freq)) {
        return Err("coherent average needs overlap-aligned curves".into());
    }
    for (index, curve) in curves.iter().enumerate() {
        if curve
            .phase
            .as_ref()
            .is_none_or(|phase| phase.len() != curve.freq.len())
        {
            return Err(format!(
                "coherent average requires measured phase on every seat (seat {} has none)",
                contract.seats[index].seat_id
            )
            .into());
        }
    }
    for seat in &contract.seats {
        if contract.require_calibration
            && seat
                .calibration_id
                .as_ref()
                .is_none_or(|id| id.trim().is_empty())
        {
            return Err(format!(
                "coherent average requires a calibration identity for seat '{}'",
                seat.seat_id
            )
            .into());
        }
        if seat
            .phase_confidence
            .is_some_and(|confidence| !confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
        {
            return Err(format!("seat '{}' has invalid phase confidence", seat.seat_id).into());
        }
        if seat.phase_confidence.unwrap_or(0.0) < contract.min_phase_confidence {
            return Err(format!(
                "seat '{}' phase confidence ({:?}) is below the coherent-average minimum {}",
                seat.seat_id, seat.phase_confidence, contract.min_phase_confidence
            )
            .into());
        }
    }

    let mut real_sum = Array1::<f64>::zeros(freqs.len());
    let mut imag_sum = Array1::<f64>::zeros(freqs.len());
    let mut coherence_sum = curves
        .iter()
        .all(|curve| curve.coherence.is_some())
        .then(|| Array1::<f64>::zeros(freqs.len()));
    for curve in curves {
        let phase = curve.phase.as_ref().expect("phase checked above");
        for (bin, (&spl, &phase_deg)) in curve.spl.iter().zip(phase.iter()).enumerate() {
            let amplitude = 10.0_f64.powf(spl / 20.0);
            let phase_rad = phase_deg.to_radians();
            real_sum[bin] += amplitude * phase_rad.cos();
            imag_sum[bin] += amplitude * phase_rad.sin();
        }
        if let (Some(sum), Some(coherence)) = (coherence_sum.as_mut(), curve.coherence.as_ref())
        {
            *sum = sum.clone() + coherence;
        }
    }
    let count = curves.len() as f64;
    let mut spl = Array1::<f64>::zeros(freqs.len());
    let mut phase = Array1::<f64>::zeros(freqs.len());
    for bin in 0..freqs.len() {
        let magnitude = (real_sum[bin] / count).hypot(imag_sum[bin] / count);
        if !magnitude.is_finite() || magnitude <= 0.0 {
            return Err(format!(
                "coherent average has non-finite magnitude at {} Hz (seats cancel)",
                freqs[bin]
            )
            .into());
        }
        spl[bin] = 20.0 * magnitude.log10();
        phase[bin] = (imag_sum[bin]).atan2(real_sum[bin]).to_degrees();
    }
    let coherence = coherence_sum.map(|sum| sum / count);
    let averaged = Curve {
        freq: freqs,
        spl,
        phase: Some(phase),
        coherence,
        ..Default::default()
    };
    averaged.validate("coherent average")?;
    Ok(averaged)
}

fn seat_for_ref(measurement: &MeasurementRef, index: usize) -> SeatProvenance {
    SeatProvenance {
        seat_id: measurement
            .name()
            .map(String::from)
            .unwrap_or_else(|| format!("seat-{index}")),
        source_path: measurement.path().map(|path| path.display().to_string()),
        ..Default::default()
    }
}

fn phase_confidence_of(curve: &Curve) -> Option<f64> {
    curve
        .coherence
        .as_ref()
        .map(|coherence| coherence.iter().sum::<f64>() / coherence.len() as f64)
}

fn seats_for_aligned(source: &MeasurementSource, curves: &[Curve]) -> Vec<SeatProvenance> {
    let mut seats: Vec<SeatProvenance> = match source {
        MeasurementSource::Single(s) => vec![seat_for_ref(&s.measurement, 0)],
        MeasurementSource::Multiple(m) => m
            .measurements
            .iter()
            .enumerate()
            .map(|(index, measurement)| seat_for_ref(measurement, index))
            .collect(),
        MeasurementSource::InMemory(_) => vec![SeatProvenance {
            seat_id: "seat-0".to_string(),
            ..Default::default()
        }],
        MeasurementSource::InMemoryMultiple(_) => (0..curves.len())
            .map(|index| SeatProvenance {
                seat_id: format!("seat-{index}"),
                ..Default::default()
            })
            .collect(),
    };
    for (seat, curve) in seats.iter_mut().zip(curves) {
        seat.phase_confidence = phase_confidence_of(curve);
    }
    seats
}

/// Load a source with full seat attribution.
///
/// Additive companion to [`load_source_with_individual`]: the same
/// overlap-aligned individuals and spatial-RMS representative, plus the
/// separately identified primary-seat curve, per-seat provenance
/// (seat ids, source paths, phase confidence), and — only when
/// `coherent_contract` is `Some` and every seat satisfies it — a coherent
/// complex average. Other crates doing phase-sensitive work should consume
/// `primary_seat` or `coherent`, never `spatial_rms.phase` (always `None`
/// for multi-seat loads).
pub fn load_source_detailed(
    source: &MeasurementSource,
    coherent_contract: Option<&CoherentAverageContract>,
) -> Result<DetailedLoad, Box<dyn Error>> {
    let aligned = load_source_individual_with_support(source)?;
    let seats = seats_for_aligned(source, &aligned.curves);
    let spatial_rms = if aligned.curves.len() == 1 {
        aligned.curves[0].clone()
    } else {
        average_curves_power_domain(&aligned.curves)
    };
    let primary_seat = aligned.curves[0].clone();
    let coherent = coherent_contract
        .map(|contract| coherent_average_measurement(&aligned.curves, contract))
        .transpose()?;
    let validity_mask = aligned.validity_mask.first().cloned().unwrap_or_default();
    Ok(DetailedLoad {
        spatial_rms,
        primary_seat,
        coherent,
        individual: aligned.curves,
        seats,
        overlap_hz: aligned.overlap_hz,
        support_hz: aligned.support_hz,
        validity_mask,
    })
}

/// Load measurement(s) once and return both the representative response and
/// the aligned individual responses.
///
/// The representative is the spatial (power-domain RMS) magnitude on the
/// physical overlap grid; for multi-seat loads its `phase` is always `None`
/// (see [`load_source_detailed`] for the separately identified primary-seat
/// curve and the contracted coherent average).
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
    fn load_source_spatial_average_never_carries_averaged_angle() {
        let mut first = sample_curve(0.0);
        first.phase = Some(Array1::from_vec(vec![170.0, 20.0, -45.0]));
        let mut second = sample_curve(0.0);
        second.phase = Some(Array1::from_vec(vec![-170.0, 40.0, -15.0]));

        let source = MeasurementSource::InMemoryMultiple(vec![first.clone(), second.clone()]);
        let average = load_source(&source).unwrap();
        // An RMS magnitude must never be paired with an averaged angle:
        // phase presence on the spatial average cannot prove coherence.
        assert!(average.phase.is_none());

        // The same seat pair under an explicit coherent contract yields a
        // genuine complex-mean response with circularly averaged phase.
        let individuals = load_source_individual(&source).unwrap();
        let seats = vec![
            SeatProvenance {
                seat_id: "a".to_string(),
                calibration_id: Some("mic-1".to_string()),
                phase_confidence: Some(1.0),
                ..Default::default()
            },
            SeatProvenance {
                seat_id: "b".to_string(),
                calibration_id: Some("mic-1".to_string()),
                phase_confidence: Some(1.0),
                ..Default::default()
            },
        ];
        let contract = CoherentAverageContract {
            seats,
            min_phase_confidence: 0.0,
            require_calibration: true,
        };
        let coherent = coherent_average_measurement(&individuals, &contract).unwrap();
        let phase = coherent.phase.expect("coherent average carries phase");
        assert!((phase[0].abs() - 180.0).abs() < 1e-9);
        assert!((phase[1] - 30.0).abs() < 1e-9);
        assert!((phase[2] + 30.0).abs() < 1e-9);
    }

    #[test]
    fn equal_spl_opposite_phase_keeps_spatial_level_without_angle() {
        // 80 dB at 0° + 80 dB at 180°: the spatial RMS stays 80 dB with NO
        // phase (previously this kept 80 dB with a numerically unstable
        // averaged angle). The coherent mean instead cancels (≈ −238 dB).
        let freq = Array1::from_vec(vec![100.0, 1000.0, 10000.0]);
        let first = Curve {
            freq: freq.clone(),
            spl: Array1::from_vec(vec![80.0, 80.0, 80.0]),
            phase: Some(Array1::from_vec(vec![0.0, 0.0, 0.0])),
            ..Default::default()
        };
        let second = Curve {
            freq: freq.clone(),
            spl: Array1::from_vec(vec![80.0, 80.0, 80.0]),
            phase: Some(Array1::from_vec(vec![180.0, 180.0, 180.0])),
            ..Default::default()
        };
        let source = MeasurementSource::InMemoryMultiple(vec![first, second]);
        let average = load_source(&source).unwrap();
        for spl in average.spl.iter() {
            assert!((spl - 80.0).abs() < 1e-9);
        }
        assert!(average.phase.is_none());

        let individuals = load_source_individual(&source).unwrap();
        let seats = (0..2)
            .map(|index| SeatProvenance {
                seat_id: format!("seat-{index}"),
                phase_confidence: Some(1.0),
                ..Default::default()
            })
            .collect();
        let contract = CoherentAverageContract {
            seats,
            min_phase_confidence: 0.0,
            require_calibration: false,
        };
        // The seats cancel: the coherent mean collapses ~300 dB below the
        // spatial RMS instead of inheriting its 80 dB.
        let coherent = coherent_average_measurement(&individuals, &contract).unwrap();
        assert!(coherent.phase.is_some());
        assert!(coherent.spl[0] < -100.0);
    }

    #[test]
    fn coherent_average_rejects_non_finite_magnitude() {
        let freq = Array1::from_vec(vec![100.0, 1000.0]);
        let loud = |phase: f64| Curve {
            freq: freq.clone(),
            spl: Array1::from_vec(vec![1.0e308, 1.0e308]),
            phase: Some(Array1::from_vec(vec![phase, phase])),
            ..Default::default()
        };
        let individuals = vec![loud(0.0), loud(180.0)];
        let contract = CoherentAverageContract {
            seats: vec![
                SeatProvenance {
                    seat_id: "a".to_string(),
                    ..Default::default()
                },
                SeatProvenance {
                    seat_id: "b".to_string(),
                    ..Default::default()
                },
            ],
            min_phase_confidence: 0.0,
            require_calibration: false,
        };
        let error = coherent_average_measurement(&individuals, &contract).unwrap_err();
        assert!(
            error.to_string().contains("non-finite magnitude"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn coherent_average_is_gated_on_calibration_confidence_and_phase() {
        let mut first = sample_curve(0.0);
        first.phase = Some(Array1::from_vec(vec![10.0, 20.0, 30.0]));
        let mut second = sample_curve(0.0);
        second.phase = Some(Array1::from_vec(vec![12.0, 22.0, 32.0]));
        let source = MeasurementSource::InMemoryMultiple(vec![first, second]);
        let individuals = load_source_individual(&source).unwrap();

        // Default contract with no seat provenance: rejected.
        assert!(coherent_average_measurement(&individuals, &CoherentAverageContract::default())
            .is_err());

        // Uncalibrated seats under a calibration-requiring contract: rejected.
        let uncalibrated = vec![
            SeatProvenance {
                seat_id: "a".to_string(),
                phase_confidence: Some(1.0),
                ..Default::default()
            },
            SeatProvenance {
                seat_id: "b".to_string(),
                phase_confidence: Some(1.0),
                ..Default::default()
            },
        ];
        assert!(
            coherent_average_measurement(
                &individuals,
                &CoherentAverageContract {
                    seats: uncalibrated,
                    min_phase_confidence: 0.0,
                    require_calibration: true,
                }
            )
            .is_err()
        );

        // Low-confidence seat: rejected.
        let low_confidence = vec![
            SeatProvenance {
                seat_id: "a".to_string(),
                calibration_id: Some("mic-1".to_string()),
                phase_confidence: Some(0.1),
                ..Default::default()
            },
            SeatProvenance {
                seat_id: "b".to_string(),
                calibration_id: Some("mic-1".to_string()),
                phase_confidence: Some(1.0),
                ..Default::default()
            },
        ];
        assert!(
            coherent_average_measurement(
                &individuals,
                &CoherentAverageContract {
                    seats: low_confidence,
                    min_phase_confidence: 0.8,
                    require_calibration: true,
                }
            )
            .is_err()
        );

        // Missing phase on one seat: rejected even with a permissive contract.
        let no_phase = vec![sample_curve(0.0), sample_curve(0.0)];
        let permissive = CoherentAverageContract {
            seats: vec![
                SeatProvenance {
                    seat_id: "a".to_string(),
                    ..Default::default()
                },
                SeatProvenance {
                    seat_id: "b".to_string(),
                    ..Default::default()
                },
            ],
            min_phase_confidence: 0.0,
            require_calibration: false,
        };
        assert!(coherent_average_measurement(&no_phase, &permissive).is_err());
    }

    #[test]
    fn coherent_average_rejects_malformed_evidence() {
        let mut curve = sample_curve(0.0);
        curve.phase = Some(Array1::from_vec(vec![10.0, 20.0, 30.0]));
        let curves = vec![curve];
        let valid = CoherentAverageContract {
            seats: vec![SeatProvenance {
                seat_id: "primary".into(),
                calibration_id: Some("mic-1".into()),
                phase_confidence: Some(1.0),
                ..Default::default()
            }],
            min_phase_confidence: 0.8,
            require_calibration: true,
        };
        assert!(coherent_average_measurement(&curves, &valid).is_ok());
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
            let mut contract = valid.clone();
            contract.min_phase_confidence = value;
            assert!(coherent_average_measurement(&curves, &contract).is_err());
            let mut contract = valid.clone();
            contract.seats[0].phase_confidence = Some(value);
            assert!(coherent_average_measurement(&curves, &contract).is_err());
        }
        for id in [None, Some("".into()), Some(" \t\n".into())] {
            let mut contract = valid.clone();
            contract.seats[0].calibration_id = id;
            assert!(coherent_average_measurement(&curves, &contract).is_err());
        }
    }

    #[test]
    fn detailed_load_separates_spatial_primary_and_coherent() {
        let mut first = sample_curve(0.0);
        first.phase = Some(Array1::from_vec(vec![10.0, 20.0, 30.0]));
        let mut second = sample_curve(3.0);
        second.phase = Some(Array1::from_vec(vec![12.0, 22.0, 32.0]));
        let source = MeasurementSource::InMemoryMultiple(vec![first.clone(), second.clone()]);

        // Without a contract: spatial RMS (no phase) + identified primary.
        let detailed = load_source_detailed(&source, None).unwrap();
        assert!(detailed.spatial_rms.phase.is_none());
        assert_eq!(detailed.primary_seat.spl.to_vec(), first.spl.to_vec());
        assert!(detailed.primary_seat.phase.is_some());
        assert!(detailed.coherent.is_none());
        assert_eq!(
            detailed.seats.iter().map(|s| s.seat_id.clone()).collect::<Vec<_>>(),
            vec!["seat-0".to_string(), "seat-1".to_string()]
        );
        assert_eq!(detailed.individual.len(), 2);

        // With a satisfied contract: coherent average appears alongside.
        let seats = vec![
            SeatProvenance {
                seat_id: "seat-0".to_string(),
                calibration_id: Some("mic-1".to_string()),
                phase_confidence: Some(1.0),
                ..Default::default()
            },
            SeatProvenance {
                seat_id: "seat-1".to_string(),
                calibration_id: Some("mic-1".to_string()),
                phase_confidence: Some(1.0),
                ..Default::default()
            },
        ];
        let contract = CoherentAverageContract {
            seats,
            min_phase_confidence: 0.0,
            require_calibration: true,
        };
        let detailed = load_source_detailed(&source, Some(&contract)).unwrap();
        let coherent = detailed.coherent.expect("contract satisfied");
        assert!(coherent.phase.is_some());
        // Nearly aligned but distinct seat angles: the coherent magnitude is
        // strictly below the RMS magnitude (triangle inequality).
        assert!(coherent.spl[0] < detailed.spatial_rms.spl[0]);
    }

    #[test]
    fn load_measurement_strict_rejects_mismatched_phase() {
        let mut inline = sample_inline();
        inline.phase_deg = Some(vec![0.0, 45.0]);
        // Lenient default keeps warn-and-drop behavior.
        let lenient =
            load_measurement(&MeasurementRef::Inline(inline.clone())).unwrap();
        assert!(lenient.phase.is_none());
        // Strict mode errors instead.
        let error =
            load_measurement_strict(&MeasurementRef::Inline(inline)).unwrap_err();
        assert!(
            error.to_string().contains("strict phase mode"),
            "unexpected error: {error}"
        );
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
