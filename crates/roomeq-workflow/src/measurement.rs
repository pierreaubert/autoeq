//! Measurement resource loading for RoomEQ application workflows.

use anyhow::{Context, Result, anyhow};
use autoeq_core::phase_utils::{
    compute_excess_phase, estimate_delay_from_excess_phase, reconstruct_minimum_phase,
    unwrap_phase_degrees,
};
use autoeq_measurements::read::{interpolate_log_space, smooth_one_over_n_octave};
use ndarray::Array1;
use roomeq_engine::analysis::frequency_grid::{
    DEFAULT_ROOM_EQ_FREQUENCY_SAMPLES, ROOM_EQ_RESAMPLE_LOW_FREQ_MAX_HZ,
    clipped_room_eq_frequency_grid,
};
#[cfg(test)]
use roomeq_engine::analysis::frequency_grid::{
    ROOM_EQ_RESAMPLE_MAX_FREQ_HZ, ROOM_EQ_RESAMPLE_MIN_FREQ_HZ, room_eq_hybrid_frequency_grid,
};
use roomeq_model::{Curve, MeasurementRef, MeasurementSource};
use std::path::Path;

pub const DEFAULT_FREQUENCY_SAMPLES: usize = DEFAULT_ROOM_EQ_FREQUENCY_SAMPLES;
const ROOM_EQ_RESAMPLE_SMOOTHING_BANDS_PER_OCTAVE: usize = 2;

/// Reduce dense RoomEQ measurements to the grid used by the optimizer.
///
/// Generic AutoEQ readers preserve the source grid. RoomEQ intentionally
/// limits oversized inputs because its optimization objective may smooth the
/// response for every candidate filter. Resample first, then smooth the small
/// hybrid grid so dense source files do not make that objective quadratic in
/// the original sample count.
fn clipped_hybrid_frequency_grid(curve: &Curve, frequency_samples: usize) -> Option<Array1<f64>> {
    clipped_room_eq_frequency_grid(curve, frequency_samples)
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

/// Minimum prominence for a local extremum to survive dense-curve reduction.
///
/// Narrow resonances and cancellations narrower than the 4 Hz bass grid would
/// otherwise vanish when interpolating onto the hybrid grid (F01). Extrema
/// standing out by at least this amount from both neighbours are retained as
/// explicit grid points and restored after smoothing.
const PRESERVED_EXTREMUM_PROMINENCE_DB: f64 = 1.0;
/// Upper bound on retained extrema so pathological noisy inputs cannot make
/// the reduced grid quadratic in the source size again.
const MAX_PRESERVED_EXTREMA: usize = 512;

/// Collect significant local maxima/minima as (frequency, spl, is_max).
fn significant_extrema(curve: &Curve) -> Vec<(f64, f64, bool)> {
    if curve.freq.len() != curve.spl.len() || curve.freq.len() < 3 {
        return Vec::new();
    }
    let mut extrema: Vec<(f64, f64, f64, bool)> = Vec::new();
    for index in 1..curve.freq.len() - 1 {
        let previous = curve.spl[index - 1];
        let current = curve.spl[index];
        let next = curve.spl[index + 1];
        if !previous.is_finite() || !current.is_finite() || !next.is_finite() {
            continue;
        }
        let frequency = curve.freq[index];
        if !frequency.is_finite() || frequency <= 0.0 {
            continue;
        }
        if current > previous && current >= next {
            let prominence = current - previous.max(next);
            if prominence >= PRESERVED_EXTREMUM_PROMINENCE_DB {
                extrema.push((frequency, current, prominence, true));
            }
        } else if current < previous && current <= next {
            let prominence = previous.min(next) - current;
            if prominence >= PRESERVED_EXTREMUM_PROMINENCE_DB {
                extrema.push((frequency, current, prominence, false));
            }
        }
    }
    extrema.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    extrema.truncate(MAX_PRESERVED_EXTREMA);
    extrema
        .into_iter()
        .map(|(frequency, spl, _, is_max)| (frequency, spl, is_max))
        .collect()
}

fn cap_measurement_curve(curve: Curve, frequency_samples: usize) -> Curve {
    if frequency_samples == 0
        || curve.freq.len() <= frequency_samples
        || curve
            .freq
            .last()
            .is_some_and(|&frequency| frequency <= 500.0)
    {
        return curve;
    }

    let Some(base_grid) = clipped_hybrid_frequency_grid(&curve, frequency_samples) else {
        return curve;
    };
    // Preserve significant narrow extrema that fall between hybrid-grid bins.
    let preserved = significant_extrema(&curve);
    let mut frequencies = base_grid.to_vec();
    // Phase-critical bass summation needs every measured bin, including bins
    // which are not extrema in either individual sub's magnitude response.
    frequencies.extend(
        curve
            .freq
            .iter()
            .copied()
            .filter(|&frequency| frequency <= 500.0),
    );
    for (frequency, _, _) in &preserved {
        let already_close = frequencies
            .iter()
            .any(|&existing| (existing - *frequency).abs() <= f64::EPSILON * 8.0);
        if !already_close {
            frequencies.push(*frequency);
        }
    }
    frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    frequencies.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    let frequency_grid = Array1::from_vec(frequencies);
    let resampled = interpolate_log_space(&frequency_grid, &curve);
    let mut smoothed =
        smooth_one_over_n_octave(&resampled, ROOM_EQ_RESAMPLE_SMOOTHING_BANDS_PER_OCTAVE);
    // Smoothing dilutes single-bin extrema back toward the local mean; restore
    // the measured peak/null values so loading does not erase them.
    for (frequency, spl, is_max) in &preserved {
        if let Some(index) = smoothed
            .freq
            .iter()
            .position(|&candidate| candidate == *frequency)
        {
            if *is_max {
                if smoothed.spl[index] < *spl {
                    smoothed.spl[index] = *spl;
                }
            } else if smoothed.spl[index] > *spl {
                smoothed.spl[index] = *spl;
            }
        }
    }
    reconstruct_resampled_phase(&curve, &mut smoothed);
    for (index, &frequency) in smoothed.freq.iter().enumerate() {
        if frequency <= 500.0 {
            smoothed.spl[index] = resampled.spl[index];
            if let (Some(phase), Some(original)) =
                (smoothed.phase.as_mut(), resampled.phase.as_ref())
            {
                phase[index] = original[index];
            }
        }
    }
    smoothed
}

fn cap_source_curve(
    curve: Curve,
    frequency_samples: usize,
    valid_band: Option<[f64; 2]>,
) -> Result<Curve> {
    cap_source_curve_bands(
        curve,
        frequency_samples,
        &valid_band.into_iter().collect::<Vec<_>>(),
    )
}

/// Condition a source curve on declared usable segments, preserving internal
/// coverage gaps.
///
/// Each segment is selected and dense-grid conditioned independently, so
/// resampling, smoothing, and extrema selection never cross invalid support.
/// Gap bins are retained from the globally conditioned curve as display
/// samples only; the engine drops them before optimization and scoring.
/// An empty band list conditions the whole grid (legacy behavior).
fn cap_source_curve_bands(
    curve: Curve,
    frequency_samples: usize,
    bands: &[[f64; 2]],
) -> Result<Curve> {
    if bands.is_empty() {
        return Ok(cap_measurement_curve(curve, frequency_samples));
    }
    // Condition usable evidence independently. Smoothing, extrema selection,
    // and minimum-phase reconstruction must not see excluded samples.
    let mut usable_segments = Vec::with_capacity(bands.len());
    for band in bands {
        let selected = curve.select_frequency_band(*band)?;
        usable_segments.push(cap_measurement_curve(selected, frequency_samples));
    }
    if usable_segments.len() == 1 {
        let [low, high] = bands[0];
        return merge_conditioned_segments(curve, frequency_samples, usable_segments, |f| {
            *f < low || *f > high
        });
    }
    let is_gap = |f: &f64| !bands.iter().any(|[low, high]| *f >= *low && *f <= *high);
    merge_conditioned_segments(curve, frequency_samples, usable_segments, is_gap)
}

/// Merge independently conditioned segments with retained gap display samples.
fn merge_conditioned_segments(
    curve: Curve,
    frequency_samples: usize,
    usable_segments: Vec<Curve>,
    is_gap: impl Fn(&f64) -> bool,
) -> Result<Curve> {
    let full = cap_measurement_curve(curve, frequency_samples);
    // (segment index or gap, sample index): gap samples come from the
    // globally conditioned curve, segment samples from their independently
    // conditioned segment.
    let mut indices: Vec<(Option<usize>, usize)> = full
        .freq
        .iter()
        .enumerate()
        .filter_map(|(i, f)| is_gap(f).then_some((None, i)))
        .collect();
    for (segment, usable) in usable_segments.iter().enumerate() {
        indices.extend((0..usable.freq.len()).map(|i| (Some(segment), i)));
    }
    let frequency = |(segment, i): &(Option<usize>, usize)| match segment {
        Some(segment) => usable_segments[*segment].freq[*i],
        None => full.freq[*i],
    };
    indices.sort_by(|a, b| frequency(a).total_cmp(&frequency(b)));
    let at = |(segment, i): &(Option<usize>, usize), values: &[&Array1<f64>]| match segment {
        Some(segment) => values[*segment + 1][*i],
        None => values[0][*i],
    };
    let merge = |outer: &Array1<f64>, inners: &[&Array1<f64>]| {
        let values: Vec<&Array1<f64>> = std::iter::once(outer)
            .chain(inners.iter().copied())
            .collect();
        Array1::from_iter(indices.iter().map(|key| at(key, &values)))
    };
    let merge_optional = |outer: Option<&Array1<f64>>,
                          inners: Vec<Option<&Array1<f64>>>|
     -> Result<Option<Array1<f64>>> {
        match (outer, inners.iter().all(|inner| inner.is_none())) {
            (Some(outer), false) => {
                let inners: Option<Vec<&Array1<f64>>> = inners.into_iter().collect();
                match inners {
                    Some(inners) => Ok(Some(merge(outer, &inners))),
                    None => Err(anyhow!(
                        "measurement conditioning changed optional sample metadata availability"
                    )),
                }
            }
            (None, true) => Ok(None),
            _ => Err(anyhow!(
                "measurement conditioning changed optional sample metadata availability"
            )),
        }
    };
    let inners: Vec<&Curve> = usable_segments.iter().collect();
    let spl: Vec<&Array1<f64>> = inners.iter().map(|curve| &curve.spl).collect();
    let result = Curve {
        freq: merge(
            &full.freq,
            &inners.iter().map(|curve| &curve.freq).collect::<Vec<_>>(),
        ),
        spl: merge(&full.spl, &spl),
        phase: merge_optional(
            full.phase.as_ref(),
            inners.iter().map(|curve| curve.phase.as_ref()).collect(),
        )?,
        coherence: merge_optional(
            full.coherence.as_ref(),
            inners
                .iter()
                .map(|curve| curve.coherence.as_ref())
                .collect(),
        )?,
        noise_floor_db: merge_optional(
            full.noise_floor_db.as_ref(),
            inners
                .iter()
                .map(|curve| curve.noise_floor_db.as_ref())
                .collect(),
        )?,
        // No global phase decomposition is valid for this piecewise view.
        min_phase: None,
        excess_phase: None,
        excess_delay_ms: None,
    };
    result.validate("band-isolated conditioned measurement")?;
    Ok(result)
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
    let band = require_contiguous_source_support(source)?;
    autoeq_measurements::read::load_source_individual(source)
        .map_err(|error| anyhow!(error.to_string()))
        .and_then(|curves| {
            curves
                .into_iter()
                .map(|curve| cap_source_curve(curve, frequency_samples, band))
                .collect()
        })
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
    load_source_with_conditioning(source, frequency_samples)
        .map(|loaded| (loaded.representative, loaded.individual))
}

pub(crate) struct ConditionedSource {
    pub representative: Curve,
    pub individual: Vec<Curve>,
    pub conditioning: Vec<autoeq_measurements::LedgerEntry>,
    pub native_identities: Vec<String>,
}

pub(crate) fn load_source_with_conditioning(
    source: &MeasurementSource,
    frequency_samples: usize,
) -> Result<ConditionedSource> {
    // The channel path is support-aware: each declared segment is
    // conditioned independently and internal gaps are retained as display
    // samples. Curve-only single-curve loaders keep refusing disjoint
    // support (see `require_contiguous_source_support`).
    let bands: Vec<[f64; 2]> = source
        .provenance()
        .declared_support_bands()
        .map_err(|error| anyhow!(error))?
        .unwrap_or_default();
    let loaded = autoeq_measurements::read::load_source_detailed(source, None)
        .map_err(|error| anyhow!(error.to_string()))
        .context("failed to load measurement source with conditioning")?;
    let mut conditioning = loaded.conditioning;
    let mut condition = |curve: Curve, source_index: Option<usize>| -> Result<Curve> {
        let input_hash = curve
            .content_hash()
            .map_err(|error| anyhow!(error.to_string()))?;
        let input_bins = curve.freq.len();
        let result = cap_source_curve_bands(curve, frequency_samples, &bands)?;
        let output_hash = result
            .content_hash()
            .map_err(|error| anyhow!(error.to_string()))?;
        if input_hash != output_hash {
            // Legacy receipts stay byte-identical: the multi-band field is
            // recorded only when disjoint support was actually conditioned.
            let mut parameters = serde_json::json!({
                "source_index": source_index,
                "role": if source_index.is_some() { "individual" } else { "representative" },
                "frequency_samples": frequency_samples,
                "declared_valid_band_hz": bands.first(),
                "implementation": "cap_source_curve/v1",
                "input_bins": input_bins,
                "output_bins": result.freq.len(),
                "acquisition_validated": false
            });
            if bands.len() > 1 {
                parameters["declared_valid_bands_hz"] = serde_json::json!(bands);
            }
            conditioning.push(autoeq_measurements::LedgerEntry {
                operation: "roomeq_dense_grid_conditioning".into(),
                version: 1,
                parameters: serde_json::from_value(parameters)?,
                input_hashes: vec![input_hash],
                output_hash,
                lossy: true,
                executed_at: None,
                tool: Some(autoeq_measurements::ToolIdentity {
                    application: Some("roomeq-workflow".into()),
                    version: Some(env!("CARGO_PKG_VERSION").into()),
                    ..Default::default()
                }),
                determinism: Some(autoeq_measurements::Determinism::PlatformSensitive),
            });
        }
        Ok(result)
    };
    let representative = condition(loaded.spatial_rms, None)?;
    let individual = loaded
        .individual
        .into_iter()
        .enumerate()
        .map(|(index, curve)| condition(curve, Some(index)))
        .collect::<Result<_>>()?;
    Ok(ConditionedSource {
        representative,
        individual,
        conditioning,
        native_identities: loaded.native_identities,
    })
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
    let band = require_contiguous_source_support(source)?;
    autoeq_measurements::read::load_source(source)
        .map_err(|error| anyhow!(error.to_string()))
        .and_then(|curve| cap_source_curve(curve, frequency_samples, band))
        .context("failed to load measurement source")
}

fn require_contiguous_source_support(source: &MeasurementSource) -> Result<Option<[f64; 2]>> {
    match source
        .provenance()
        .declared_support_bands()
        .map_err(|error| anyhow!(error))?
    {
        None => Ok(None),
        Some(bands) if bands.len() == 1 => Ok(Some(bands[0])),
        Some(_) => Err(anyhow!(
            "disjoint measurement support requires a support-aware correction path; curve-only loading cannot preserve the internal gap"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::MeasurementSingle;

    #[test]
    fn curve_only_workflow_load_refuses_disjoint_support() {
        let source = MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Inline(autoeq_core::InlineMeasurement {
                frequencies: vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                magnitude_db: vec![80.0; 6],
                phase_deg: None,
                name: None,
                wav_path: None,
                csv_path: None,
            }),
            speaker_name: None,
            provenance: autoeq_core::MeasurementProvenance {
                valid_bands_hz: vec![[100.0, 200.0], [500.0, 600.0]],
                ..Default::default()
            },
        });
        assert!(autoeq_measurements::read::load_source_individual_with_support(&source).is_ok());
        // Single-curve and averaging loaders cannot carry the internal gap:
        // a bare Curve has no validity mask, so they keep refusing.
        for result in [
            load_source_with_frequency_samples(&source, 64).map(|_| ()),
            load_source_individual_with_frequency_samples(&source, 64).map(|_| ()),
        ] {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("support-aware correction path")
            );
        }
        // The support-aware conditioning path accepts: each segment is
        // conditioned independently and gap bins are retained as display
        // samples only.
        let (representative, individual) =
            load_source_with_individual_with_frequency_samples(&source, 64).unwrap();
        assert!(!individual.is_empty());
        let gap_bins: Vec<_> = representative
            .freq
            .iter()
            .copied()
            .filter(|f| *f > 200.0 && *f < 500.0)
            .collect();
        assert_eq!(gap_bins, vec![300.0, 400.0]);
        for curve in std::iter::once(&representative).chain(individual.iter()) {
            assert!(curve.freq.iter().any(|f| *f == 100.0));
            assert!(curve.freq.iter().any(|f| *f == 600.0));
        }
    }

    #[test]
    fn roadmap_dense_loading_does_not_smooth_unusable_samples_into_valid_band() {
        let mut loaded = Vec::new();
        for outside in [40.0, 120.0] {
            let frequencies: Vec<_> = (1..=1000).map(|i| i as f64 * 10.0).collect();
            let magnitude_db = frequencies
                .iter()
                .map(|f| {
                    if (1000.0..=2000.0).contains(f) {
                        80.0
                    } else {
                        outside
                    }
                })
                .collect();
            let source = MeasurementSource::Single(MeasurementSingle {
                measurement: MeasurementRef::Inline(autoeq_core::InlineMeasurement {
                    frequencies,
                    magnitude_db,
                    phase_deg: None,
                    name: None,
                    wav_path: None,
                    csv_path: None,
                }),
                speaker_name: None,
                provenance: autoeq_core::MeasurementProvenance {
                    valid_band_hz: Some([1000.0, 2000.0]),
                    ..Default::default()
                },
            });
            let curve = load_source_with_frequency_samples(&source, 64).unwrap();
            let individuals = load_source_individual_with_frequency_samples(&source, 64).unwrap();
            let (representative, combined_individuals) =
                load_source_with_individual_with_frequency_samples(&source, 64).unwrap();
            assert_eq!(individuals.len(), 1);
            assert_eq!(combined_individuals.len(), 1);
            for other in [&individuals[0], &representative, &combined_individuals[0]] {
                assert_eq!(other.freq, curve.freq);
                assert_eq!(other.spl, curve.spl);
                assert_eq!(other.phase, curve.phase);
            }
            assert!(curve.freq[0] < 1000.0 && *curve.freq.last().unwrap() > 2000.0);
            let usable = curve.select_frequency_band([1000.0, 2000.0]).unwrap();
            loaded.push(usable);
        }
        assert_eq!(loaded[0].freq, loaded[1].freq);
        let worst_difference = loaded[0]
            .spl
            .iter()
            .zip(loaded[1].spl.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            worst_difference < 1e-9,
            "unusable data contaminated dense loading by {worst_difference} dB"
        );
    }

    #[test]
    fn roadmap_dense_loading_preserves_metadata_and_isolates_phase() {
        let mut results = Vec::new();
        for outside in [40.0, 120.0] {
            let freq = Array1::from_iter((1..=1000).map(|i| i as f64 * 10.0));
            let curve = Curve {
                spl: freq.mapv(|f| {
                    if (1000.0..=2000.0).contains(&f) {
                        80.0
                    } else {
                        outside
                    }
                }),
                phase: Some(freq.mapv(|f| -f * 0.36)),
                coherence: Some(freq.mapv(|f| 0.5 + f / 20000.0)),
                noise_floor_db: Some(freq.mapv(|f| -80.0 + f / 1000.0)),
                freq,
                ..Default::default()
            };
            let expected =
                cap_measurement_curve(curve.select_frequency_band([1000.0, 2000.0]).unwrap(), 64);
            let merged = cap_source_curve(curve, 64, Some([1000.0, 2000.0])).unwrap();
            merged.validate("test merged metadata").unwrap();
            assert!(merged.min_phase.is_none());
            assert!(merged.excess_phase.is_none());
            assert!(merged.excess_delay_ms.is_none());
            let usable = merged.select_frequency_band([1000.0, 2000.0]).unwrap();
            assert_eq!(usable.freq, expected.freq);
            assert_eq!(usable.spl, expected.spl);
            assert_eq!(usable.phase, expected.phase);
            assert_eq!(usable.coherence, expected.coherence);
            assert_eq!(usable.noise_floor_db, expected.noise_floor_db);
            results.push(usable);
        }
        assert_eq!(results[0].phase, results[1].phase);
        assert_eq!(results[0].coherence, results[1].coherence);
        assert_eq!(results[0].noise_floor_db, results[1].noise_floor_db);
    }

    #[test]
    fn roadmap_dense_loading_requires_usable_samples_and_preserves_legacy_path() {
        let curve = Curve {
            freq: Array1::from_vec(vec![100.0, 1000.0, 2000.0]),
            spl: Array1::from_vec(vec![70.0, 80.0, 75.0]),
            ..Default::default()
        };
        for band in [[3000.0, 4000.0], [900.0, 1100.0]] {
            assert!(cap_source_curve(curve.clone(), 2, Some(band)).is_err());
        }
        let expected = cap_measurement_curve(curve.clone(), 2);
        let actual = cap_source_curve(curve, 2, None).unwrap();
        assert_eq!(actual.freq, expected.freq);
        assert_eq!(actual.spl, expected.spl);
        assert_eq!(actual.phase, expected.phase);
    }

    fn write_measurement(directory: &Path) -> std::path::PathBuf {
        let path = directory.join("measurement.csv");
        std::fs::write(&path, "frequency,spl\n20,70\n100,71\n1000,69\n").unwrap();
        path
    }

    #[test]
    fn sub_measurement_keeps_every_native_bin_and_measured_phase() {
        let frequencies = Array1::logspace(10.0, 10.0_f64.log10(), 250.0_f64.log10(), 1000);
        let curve = Curve {
            spl: frequencies.mapv(|frequency| 80.0 + (frequency * 0.7).sin()),
            phase: Some(frequencies.mapv(|frequency| -frequency * 7.2)),
            freq: frequencies,
            ..Default::default()
        };
        let capped = cap_measurement_curve(curve.clone(), 32);
        assert_eq!(capped.freq, curve.freq);
        assert_eq!(capped.spl, curve.spl);
        assert_eq!(capped.phase, curve.phase);
    }

    #[test]
    fn dense_and_sparse_sampling_preserve_narrow_resonance() {
        // F01: the same continuous transfer sampled at 1 Hz and 0.5 Hz must
        // yield comparable loaded maxima. Gaussian peak: 82 Hz, 12 dB, sigma 0.5 Hz.
        let directory = tempfile::tempdir().unwrap();
        let sparse_path = directory.path().join("sparse.csv");
        let dense_path = directory.path().join("dense.csv");
        let mut sparse_csv = String::from("frequency,spl\n");
        for i in 0..=180 {
            let f = 20.0 + i as f64;
            let spl = 80.0 + 12.0 * (-0.5 * ((f - 82.0) / 0.5).powi(2)).exp();
            sparse_csv.push_str(&format!("{f},{spl}\n"));
        }
        let mut dense_csv = String::from("frequency,spl\n");
        for i in 0..=360 {
            let f = 20.0 + i as f64 / 2.0;
            let spl = 80.0 + 12.0 * (-0.5 * ((f - 82.0) / 0.5).powi(2)).exp();
            dense_csv.push_str(&format!("{f},{spl}\n"));
        }
        std::fs::write(&sparse_path, sparse_csv).unwrap();
        std::fs::write(&dense_path, dense_csv).unwrap();

        let sparse = load_curve_from_csv(&sparse_path).unwrap();
        let dense = load_curve_from_csv(&dense_path).unwrap();
        let sparse_max = sparse.spl.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let dense_max = dense.spl.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            sparse_max > 88.0,
            "sparse input should preserve the 12 dB peak, got {sparse_max}"
        );
        assert!(
            dense_max > 88.0,
            "dense input erased the narrow resonance, got {dense_max}"
        );
        assert!(
            (sparse_max - dense_max).abs() < 3.0,
            "density-dependent loss: sparse {sparse_max} vs dense {dense_max}"
        );
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
            provenance: Default::default(),
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

        let expected_len = room_eq_hybrid_frequency_grid(DEFAULT_FREQUENCY_SAMPLES).len();
        assert!(curve.freq.len() >= expected_len);
        assert_eq!(curve.spl.len(), curve.freq.len());
        assert_eq!(curve.phase.as_ref().unwrap().len(), curve.freq.len());
        assert!((curve.freq[0] - ROOM_EQ_RESAMPLE_MIN_FREQ_HZ).abs() < 1e-12);
        assert!((curve.freq.last().copied().unwrap() - ROOM_EQ_RESAMPLE_MAX_FREQ_HZ).abs() < 1e-9);

        assert!(
            curve
                .freq
                .windows(2)
                .into_iter()
                .all(|pair| pair[1] > pair[0])
        );
        for index in 0..=400 {
            let native = 20.0 * 1000.0_f64.powf(index as f64 / 400.0);
            if native <= 500.0 {
                assert!(
                    curve
                        .freq
                        .iter()
                        .any(|&frequency| (frequency - native).abs() < 1e-8)
                );
            }
        }
        let high: Vec<_> = curve
            .freq
            .iter()
            .copied()
            .filter(|&frequency| frequency > 2000.0)
            .collect();
        let high_ratio = high[1] / high[0];
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

        let expected_len = room_eq_hybrid_frequency_grid(64).len();
        assert!(curve.freq.len() >= expected_len);
        assert_eq!(curve.spl.len(), curve.freq.len());
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
