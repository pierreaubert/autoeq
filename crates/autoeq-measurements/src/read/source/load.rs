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
    grid.sort_by(|a, b| a.partial_cmp(b).expect("validated frequencies are finite"));
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
fn load_aligned_curves(curves: &[Curve], context: &str) -> Result<AlignedCurves, Box<dyn Error>> {
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
/// Declared usable bands are aligned independently before aggregation. Outer
/// loaded samples remain available but do not establish correction eligibility.
///
/// # Errors
/// Rejects invalid measurements or bands, insufficient usable samples, and
/// disjoint measured or usable support.
pub fn load_source_individual_with_support(
    source: &MeasurementSource,
) -> Result<AlignedCurves, Box<dyn Error>> {
    let curves = load_source_unaligned(source)?;
    align_source_curves(&curves, source.provenance().valid_band_hz)
}

fn align_source_curves(
    curves: &[Curve],
    valid_band: Option<[f64; 2]>,
) -> Result<AlignedCurves, Box<dyn Error>> {
    let Some(band) = valid_band else {
        return load_aligned_curves(curves, "measurement");
    };
    let usable: Vec<_> = curves
        .iter()
        .map(|curve| curve.select_frequency_band(band))
        .collect::<Result<_, _>>()?;
    let usable = load_aligned_curves(&usable, "usable measurement")?;
    let mut full = load_aligned_curves(curves, "measurement")?;
    let [low, high] = band;
    // Both aligned sets share a grid internally. Merge by the same index map
    // for every seat and every measured field, never interpolating across the
    // declared boundary. Bins inside the declaration but outside common usable
    // sample support are omitted, not synthesized from excluded neighbors.
    let mut indices: Vec<_> = full.curves[0]
        .freq
        .iter()
        .enumerate()
        .filter_map(|(i, f)| (*f < low || *f > high).then_some((false, i)))
        .chain((0..usable.curves[0].freq.len()).map(|i| (true, i)))
        .collect();
    let frequency = |(inside, i): &(bool, usize)| {
        if *inside {
            usable.curves[0].freq[*i]
        } else {
            full.curves[0].freq[*i]
        }
    };
    indices.sort_by(|a, b| frequency(a).total_cmp(&frequency(b)));
    for (outer, inner) in full.curves.iter_mut().zip(&usable.curves) {
        let merge = |outside: &Array1<f64>, inside: &Array1<f64>| {
            Array1::from_iter(indices.iter().map(
                |(usable, i)| {
                    if *usable { inside[*i] } else { outside[*i] }
                },
            ))
        };
        let merge_optional = |outside: Option<&Array1<f64>>,
                              inside: Option<&Array1<f64>>|
         -> Result<Option<Array1<f64>>, Box<dyn Error>> {
            match (outside, inside) {
                (Some(outside), Some(inside)) => Ok(Some(merge(outside, inside))),
                (None, None) => Ok(None),
                _ => Err("usable alignment changed measurement metadata availability".into()),
            }
        };
        *outer = Curve {
            freq: merge(&outer.freq, &inner.freq),
            spl: merge(&outer.spl, &inner.spl),
            phase: merge_optional(outer.phase.as_ref(), inner.phase.as_ref())?,
            coherence: merge_optional(outer.coherence.as_ref(), inner.coherence.as_ref())?,
            noise_floor_db: merge_optional(
                outer.noise_floor_db.as_ref(),
                inner.noise_floor_db.as_ref(),
            )?,
            // A global decomposition is not valid for this piecewise view.
            ..Default::default()
        };
        outer.validate("usable-band aligned measurement")?;
    }
    full.validity_mask = vec![vec![true; indices.len()]; full.curves.len()];
    Ok(full)
}

/// Load validated individual responses without changing their native grids or levels.
///
/// Unlike [`load_source_individual_with_support`], this does not intersect
/// support, interpolate, average, or normalize the individual responses.
/// Parsed response data does not establish raw recording provenance.
///
/// # Errors
/// Rejects unreadable or malformed measurements and empty measurement sets.
pub fn load_source_unaligned(source: &MeasurementSource) -> Result<Vec<Curve>, Box<dyn Error>> {
    let curves = match source {
        MeasurementSource::Single(s) => {
            let curve = load_measurement(&s.measurement)?;
            vec![curve]
        }
        MeasurementSource::InMemory(curve) => {
            vec![curve.clone()]
        }
        MeasurementSource::InMemoryMultiple(curves) => curves.clone(),
        MeasurementSource::Multiple(m) => {
            if m.measurements.is_empty() {
                return Err("Measurement list is empty".into());
            }
            load_measurements_strict(&m.measurements)?
        }
    };
    if curves.is_empty() {
        return Err("Measurement list is empty".into());
    }
    for (index, curve) in curves.iter().enumerate() {
        curve.validate(&format!("measurement {index}"))?;
    }
    Ok(curves)
}

/// Freeze parsed source responses while preserving original metadata and provenance.
///
/// Numerical loaders use the frozen full curves, not the original paths. This
/// does not freeze associated recording WAVs or authenticate acquisition facts.
/// In-memory sources are already snapshots and retain their unknown provenance.
///
/// # Errors
/// Rejects invalid responses, including malformed inline phase arrays, unreadable
/// source files, and empty sources. No partially frozen source is returned.
pub fn snapshot_source(source: &MeasurementSource) -> Result<MeasurementSource, Box<dyn Error>> {
    let freeze = |reference: &MeasurementRef| -> Result<MeasurementRef, Box<dyn Error>> {
        let loaded_response = load_measurement_strict(reference)?;
        Ok(MeasurementRef::Loaded {
            original: Box::new(reference.original().clone()),
            loaded_response: Box::new(loaded_response),
        })
    };
    let mut snapshot = source.clone();
    match &mut snapshot {
        MeasurementSource::Single(single) => single.measurement = freeze(&single.measurement)?,
        MeasurementSource::Multiple(multiple) => {
            for reference in &mut multiple.measurements {
                *reference = freeze(reference)?;
            }
        }
        MeasurementSource::InMemory(_) | MeasurementSource::InMemoryMultiple(_) => {}
    }
    load_source_unaligned(&snapshot)?;
    Ok(snapshot)
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
pub fn load_measurement_strict(measurement: &MeasurementRef) -> Result<Curve, Box<dyn Error>> {
    load_measurement_with_policy(measurement, true)
}

/// Single-measurement loader with an explicit phase-mismatch policy.
pub fn load_measurement_with_policy(
    measurement: &MeasurementRef,
    strict_phase: bool,
) -> Result<Curve, Box<dyn Error>> {
    let curve = match measurement {
        MeasurementRef::Loaded {
            loaded_response, ..
        } => loaded_response.as_ref().clone(),
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
                    read_curve_from_csv(&PathBuf::from(csv_path)).map_err(
                        |error| -> Box<dyn Error> {
                            format!("Failed to load measurement '{csv_path}': {error}").into()
                        },
                    )?
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
    /// Canonical loaded-curve identities before source alignment, in input order.
    /// These identify parsed numerical content, not authenticated capture bytes.
    pub native_identities: Vec<String>,
    /// Executed source conditioning, hash-linked to native and returned curves.
    ///
    /// These records cover source alignment and aggregation only, not acquisition,
    /// calibration, workflow dense-grid conditioning, or subsequent optimization.
    pub conditioning: Vec<crate::LedgerEntry>,
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
        if let (Some(sum), Some(coherence)) = (coherence_sum.as_mut(), curve.coherence.as_ref()) {
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
    let native = load_source_unaligned(source)?;
    let band = source.provenance().valid_band_hz;
    let aligned = align_source_curves(&native, band)?;
    let native_hashes = native
        .iter()
        .map(Curve::content_hash)
        .collect::<Result<Vec<_>, _>>()?;
    let mut conditioning = Vec::new();
    for (index, curve) in aligned.curves.iter().enumerate() {
        let output_hash = curve.content_hash()?;
        if output_hash != native_hashes[index] {
            conditioning.push(source_conditioning_entry(
                "source_overlap_alignment",
                native_hashes.clone(),
                output_hash,
                serde_json::json!({
                    "source_index": index,
                    "declared_valid_band_hz": band,
                    "grid_policy": "sorted_union_inside_common_support",
                    "declared_band_policy": "align_retained_native_samples_independently_then_merge_outer_response",
                    "interpolation": "autoeq_core::interpolate_log_space",
                    "input_bins": native[index].freq.len(),
                    "output_bins": curve.freq.len(),
                    "acquisition_validated": false
                }),
            )?);
        }
    }
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
    let aligned_hashes = aligned
        .curves
        .iter()
        .map(Curve::content_hash)
        .collect::<Result<Vec<_>, _>>()?;
    if aligned.curves.len() > 1 {
        conditioning.push(source_conditioning_entry(
            "source_spatial_power_rms",
            aligned_hashes.clone(),
            spatial_rms.content_hash()?,
            serde_json::json!({"weights": "equal", "phase": "absent", "normalization": "none"}),
        )?);
    }
    if let (Some(curve), Some(contract)) = (&coherent, coherent_contract) {
        conditioning.push(source_conditioning_entry(
            "source_coherent_pressure_mean",
            aligned_hashes,
            curve.content_hash()?,
            serde_json::json!({
                "weights": "equal",
                "normalization": "none",
                "min_phase_confidence": contract.min_phase_confidence,
                "require_calibration": contract.require_calibration,
                "contract_seats": contract.seats.iter().map(|seat| serde_json::json!({
                    "seat_id": seat.seat_id,
                    "calibration_id": seat.calibration_id,
                    "delay_ms": seat.delay_ms,
                    "phase_confidence": seat.phase_confidence
                })).collect::<Vec<_>>(),
                "contract_claims_authenticated": false
            }),
        )?);
    }
    let validity_mask = aligned.validity_mask.first().cloned().unwrap_or_default();
    Ok(DetailedLoad {
        native_identities: native_hashes,
        conditioning,
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

fn source_conditioning_entry(
    operation: &str,
    input_hashes: Vec<String>,
    output_hash: String,
    parameters: serde_json::Value,
) -> Result<crate::LedgerEntry, Box<dyn Error>> {
    Ok(crate::LedgerEntry {
        operation: operation.into(),
        version: 1,
        parameters: serde_json::from_value(parameters)?,
        input_hashes,
        output_hash,
        lossy: true,
        executed_at: None,
        tool: Some(crate::ToolIdentity {
            application: Some("autoeq-measurements".into()),
            version: Some(env!("CARGO_PKG_VERSION").into()),
            ..Default::default()
        }),
        determinism: Some(crate::Determinism::PlatformSensitive),
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
    fn snapshot_source_preserves_full_curve_metadata_and_roundtrip_without_reopening() {
        let mut curve = Curve {
            freq: vec![40.0, 80.0, 160.0].into(),
            spl: vec![80.0, 84.0, 80.0].into(),
            phase: Some(vec![0.0, -10.0, -20.0].into()),
            coherence: Some(vec![0.9; 3].into()),
            noise_floor_db: Some(vec![20.0; 3].into()),
            ..Default::default()
        };
        let provenance = autoeq_core::MeasurementProvenance {
            capture_kind: autoeq_core::ProvenanceCaptureKind::StationaryIr,
            calibration_id: Some("declared-cal".into()),
            timing_reference_id: Some("declared-time".into()),
            ..Default::default()
        };
        for multiple in [false, true] {
            let reference = MeasurementRef::Loaded {
                original: Box::new(MeasurementRef::Named {
                    path: "/Volumes/home_tmp/tmp/absent-snapshot-source.csv".into(),
                    name: Some("seat-label".into()),
                }),
                loaded_response: Box::new(curve.clone()),
            };
            let source = if multiple {
                MeasurementSource::Multiple(MeasurementMultiple {
                    measurements: vec![reference],
                    speaker_name: Some("speaker".into()),
                    provenance: provenance.clone(),
                })
            } else {
                MeasurementSource::Single(MeasurementSingle {
                    measurement: reference,
                    speaker_name: Some("speaker".into()),
                    provenance: provenance.clone(),
                })
            };
            let frozen = snapshot_source(&source).unwrap();
            let encoded = serde_json::to_value(&frozen).unwrap();
            let decoded: MeasurementSource = serde_json::from_value(encoded.clone()).unwrap();
            assert_eq!(decoded.provenance(), provenance);
            assert_eq!(decoded.speaker_name(), Some("speaker"));
            assert_eq!(
                serde_json::to_value(load_source_unaligned(&decoded).unwrap()[0].clone()).unwrap(),
                serde_json::to_value(&curve).unwrap()
            );
            assert_eq!(
                serde_json::to_value(snapshot_source(&decoded).unwrap()).unwrap(),
                encoded
            );
            curve.spl[1] += 1.0;
        }
    }

    #[test]
    fn snapshot_source_refuses_malformed_phase_before_freezing() {
        let mut inline = sample_inline();
        inline.phase_deg = Some(vec![0.0]);
        let source = MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Inline(inline),
            speaker_name: None,
            provenance: Default::default(),
        });
        assert!(snapshot_source(&source).is_err());
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
            provenance: Default::default(),
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
    fn declared_band_alignment_cannot_interpolate_from_excluded_samples() {
        let mut outputs = Vec::new();
        for outside in [40.0, 120.0] {
            let measurements: Vec<_> = [
                vec![500.0, 1000.0, 1500.0, 2000.0, 2500.0],
                vec![500.0, 990.0, 1100.0, 1600.0, 1900.0, 2010.0, 2500.0],
            ]
            .into_iter()
            .map(|frequencies| {
                let magnitude_db: Vec<_> = frequencies
                    .iter()
                    .map(|f| {
                        if (1000.0..=2000.0).contains(f) {
                            80.0
                        } else {
                            outside
                        }
                    })
                    .collect();
                serde_json::json!({"frequencies": frequencies, "magnitude_db": magnitude_db})
            })
            .collect();
            let source: MeasurementSource = serde_json::from_value(serde_json::json!({
                "measurements": measurements,
                "provenance": {"valid_band_hz": [1000.0, 2000.0]}
            }))
            .unwrap();
            let native = load_source_unaligned(&source).unwrap();
            assert_eq!(native[0].freq.len(), 5);
            assert_eq!(native[1].freq.len(), 7);
            let loaded = load_source_detailed(&source, None).unwrap();
            let mut known: std::collections::HashSet<_> = native
                .iter()
                .map(|curve| curve.content_hash().unwrap())
                .collect();
            assert!(
                loaded
                    .conditioning
                    .iter()
                    .any(|entry| entry.operation == "source_overlap_alignment")
            );
            for entry in &loaded.conditioning {
                assert!(entry.input_hashes.iter().all(|hash| known.contains(hash)));
                known.insert(entry.output_hash.clone());
            }
            assert_eq!(
                loaded.conditioning.last().unwrap().output_hash,
                loaded.spatial_rms.content_hash().unwrap()
            );
            let encoded = serde_json::to_value(&loaded.conditioning).unwrap();
            let restored: Vec<crate::LedgerEntry> = serde_json::from_value(encoded).unwrap();
            assert_eq!(restored, loaded.conditioning);
            assert_eq!(
                loaded.conditioning,
                load_source_detailed(&source, None).unwrap().conditioning
            );
            let (representative, individual) = load_source_with_individual(&source).unwrap();
            assert_eq!(representative.freq, loaded.spatial_rms.freq);
            assert_eq!(representative.spl, loaded.spatial_rms.spl);
            assert_eq!(individual.len(), native.len());
            for (actual, expected) in individual.iter().zip(&loaded.individual) {
                assert_eq!(actual.freq, expected.freq);
                assert_eq!(actual.spl, expected.spl);
            }
            assert!(loaded.spatial_rms.freq[0] < 1000.0);
            assert!(*loaded.spatial_rms.freq.last().unwrap() > 2000.0);
            let usable = loaded
                .spatial_rms
                .select_frequency_band([1000.0, 2000.0])
                .unwrap();
            assert!(
                usable.spl.iter().all(|spl| (spl - 80.0).abs() < 1e-9),
                "excluded samples contaminated usable magnitudes: {:?}",
                usable.spl
            );
            assert_eq!(usable.freq[0], 1100.0);
            assert_eq!(*usable.freq.last().unwrap(), 1900.0);
            outputs.push(usable);
        }
        assert_eq!(outputs[0].freq, outputs[1].freq);
        assert_eq!(outputs[0].spl, outputs[1].spl);
    }

    fn source_with_declared_band(curves: &[Curve], band: [f64; 2]) -> MeasurementSource {
        MeasurementSource::Multiple(MeasurementMultiple {
            measurements: curves
                .iter()
                .enumerate()
                .map(|(index, curve)| MeasurementRef::Loaded {
                    original: Box::new(MeasurementRef::Named {
                        path: format!("seat-{index}.csv").into(),
                        name: Some(format!("seat-{index}")),
                    }),
                    loaded_response: Box::new(curve.clone()),
                })
                .collect(),
            speaker_name: None,
            provenance: autoeq_core::MeasurementProvenance {
                valid_band_hz: Some(band),
                ..Default::default()
            },
        })
    }

    #[test]
    fn declared_band_alignment_preserves_measured_fields_and_native_snapshots() {
        let mut outputs = Vec::new();
        for outside in [40.0, 120.0] {
            let curves: Vec<_> = [
                vec![500.0, 1000.0, 1500.0, 2000.0, 2500.0],
                vec![500.0, 990.0, 1100.0, 1600.0, 1900.0, 2010.0, 2500.0],
            ]
            .into_iter()
            .map(|frequencies| {
                let freq = Array1::from_vec(frequencies);
                Curve {
                    spl: freq.mapv(|f| {
                        if (1000.0..=2000.0).contains(&f) {
                            80.0
                        } else {
                            outside
                        }
                    }),
                    phase: Some(freq.mapv(|f| {
                        if (1000.0..=2000.0).contains(&f) {
                            -30.0
                        } else {
                            outside
                        }
                    })),
                    coherence: Some(freq.mapv(|f| {
                        if (1000.0..=2000.0).contains(&f) {
                            0.9
                        } else {
                            outside / 200.0
                        }
                    })),
                    noise_floor_db: Some(freq.mapv(|f| {
                        if (1000.0..=2000.0).contains(&f) {
                            -60.0
                        } else {
                            -outside
                        }
                    })),
                    freq,
                    ..Default::default()
                }
            })
            .collect();
            let source = source_with_declared_band(&curves, [1000.0, 2000.0]);
            let before = serde_json::to_value(&source).unwrap();
            let aligned = load_source_individual_with_support(&source).unwrap();
            assert_eq!(serde_json::to_value(&source).unwrap(), before);
            assert_eq!(
                serde_json::to_value(load_source_unaligned(&source).unwrap()).unwrap(),
                serde_json::to_value(&curves).unwrap()
            );
            for (curve, original) in aligned.curves.iter().zip(&curves) {
                curve.validate("piecewise alignment test").unwrap();
                assert_eq!(curve.freq[0], original.freq[0]);
                assert_eq!(curve.spl[0], original.spl[0]);
                assert!(curve.min_phase.is_none() && curve.excess_phase.is_none());
                let usable = curve.select_frequency_band([1000.0, 2000.0]).unwrap();
                assert!(
                    usable
                        .phase
                        .as_ref()
                        .unwrap()
                        .iter()
                        .all(|p| (p + 30.0).abs() < 1e-9)
                );
                assert!(
                    usable
                        .coherence
                        .as_ref()
                        .unwrap()
                        .iter()
                        .all(|c| (c - 0.9).abs() < 1e-9)
                );
                assert!(
                    usable
                        .noise_floor_db
                        .as_ref()
                        .unwrap()
                        .iter()
                        .all(|n| (n + 60.0).abs() < 1e-9)
                );
                outputs.push(usable);
            }
            assert!(
                aligned
                    .validity_mask
                    .iter()
                    .all(|mask| mask.len() == aligned.curves[0].freq.len())
            );
        }
        assert_eq!(outputs[0].phase, outputs[2].phase);
        assert_eq!(outputs[1].phase, outputs[3].phase);
    }

    #[test]
    fn declared_band_alignment_rejects_sparse_or_disjoint_usable_support() {
        let curve = |frequencies: Vec<f64>| Curve {
            spl: Array1::from_elem(frequencies.len(), 80.0),
            freq: Array1::from_vec(frequencies),
            ..Default::default()
        };
        for second in [
            vec![500.0, 1500.0, 2500.0],
            vec![500.0, 1800.0, 1900.0, 2500.0],
        ] {
            let source = source_with_declared_band(
                &[curve(vec![500.0, 1100.0, 1200.0, 2500.0]), curve(second)],
                [1000.0, 2000.0],
            );
            assert!(load_source_unaligned(&source).is_ok());
            assert!(load_source_individual_with_support(&source).is_err());
            assert!(load_source_detailed(&source, None).is_err());
        }
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
        assert!(
            coherent_average_measurement(&individuals, &CoherentAverageContract::default())
                .is_err()
        );

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
            detailed
                .seats
                .iter()
                .map(|s| s.seat_id.clone())
                .collect::<Vec<_>>(),
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
        let receipt = detailed.conditioning.last().unwrap();
        assert_eq!(receipt.operation, "source_coherent_pressure_mean");
        assert_eq!(
            receipt.output_hash,
            detailed.coherent.as_ref().unwrap().content_hash().unwrap()
        );
        assert_eq!(receipt.parameters["contract_claims_authenticated"], false);
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
        let lenient = load_measurement(&MeasurementRef::Inline(inline.clone())).unwrap();
        assert!(lenient.phase.is_none());
        // Strict mode errors instead.
        let error = load_measurement_strict(&MeasurementRef::Inline(inline)).unwrap_err();
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
            provenance: Default::default(),
        });
        let backward = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![inline(&c2.freq, &c2.spl), inline(&c1.freq, &c1.spl)],
            speaker_name: None,
            provenance: Default::default(),
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
        for curves in [
            vec![low.clone(), high.clone()],
            vec![high.clone(), low.clone()],
        ] {
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
            provenance: Default::default(),
        });
        let reversed = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![MeasurementRef::Path(path_b), MeasurementRef::Path(path_a)],
            speaker_name: None,
            provenance: Default::default(),
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
            provenance: Default::default(),
        });
        let error = load_source_individual(&source).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no overlapping frequency support"),
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
            provenance: Default::default(),
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
            provenance: Default::default(),
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
