//! Shared frequency-grid construction and validation helpers for RoomEQ.

use autoeq_core::{Curve, create_log_frequency_grid};
use ndarray::Array1;

/// Default density of the logarithmic portion of the RoomEQ optimizer grid.
pub const DEFAULT_ROOM_EQ_FREQUENCY_SAMPLES: usize = 200;

/// Lower bound of the canonical RoomEQ optimizer grid.
pub const ROOM_EQ_RESAMPLE_MIN_FREQ_HZ: f64 = 20.0;
/// Upper bound of the canonical RoomEQ optimizer grid.
pub const ROOM_EQ_RESAMPLE_MAX_FREQ_HZ: f64 = 20_000.0;
/// Boundary between linearly and logarithmically spaced samples.
pub const ROOM_EQ_RESAMPLE_LOW_FREQ_MAX_HZ: f64 = 1_000.0;
/// Bass-grid spacing below [`ROOM_EQ_RESAMPLE_LOW_FREQ_MAX_HZ`].
pub const ROOM_EQ_RESAMPLE_LOW_FREQ_STEP_HZ: f64 = 4.0;

/// Construct the canonical RoomEQ optimizer grid.
///
/// Bass frequencies use a fixed 4 Hz spacing so room modes and crossovers keep
/// adequate resolution. Frequencies above 1 kHz use logarithmic spacing with a
/// density derived from `frequency_samples` across the full audible range.
pub fn room_eq_hybrid_frequency_grid(frequency_samples: usize) -> Array1<f64> {
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

/// Clip the canonical RoomEQ optimizer grid to a curve's measured span.
///
/// Exact source endpoints are retained when they do not already occur on the
/// canonical grid, ensuring that interpolation never extrapolates.
pub fn clipped_room_eq_frequency_grid(
    curve: &Curve,
    frequency_samples: usize,
) -> Option<Array1<f64>> {
    if !is_valid_frequency_grid(&curve.freq) {
        return None;
    }
    let source_min = curve.freq[0];
    let source_max = curve.freq[curve.freq.len() - 1];
    let mut frequencies: Vec<f64> = room_eq_hybrid_frequency_grid(frequency_samples)
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

/// Return true when two frequency axes are equivalent by value.
///
/// A tiny relative tolerance allows harmless floating-point serialization
/// differences, but rejects genuinely shifted measurement grids.
pub fn same_frequency_grid(reference: &Array1<f64>, candidate: &Array1<f64>) -> bool {
    if reference.len() != candidate.len() {
        return false;
    }

    reference.iter().zip(candidate.iter()).all(|(&a, &b)| {
        let scale = a.abs().max(b.abs()).max(1.0);
        (a - b).abs() <= scale * 1e-6
    })
}

/// Return true when a frequency grid is finite and strictly increasing.
pub fn is_valid_frequency_grid(freq: &Array1<f64>) -> bool {
    freq.len() >= 2
        && freq.iter().all(|f| f.is_finite() && *f > 0.0)
        && freq.windows(2).into_iter().all(|w| w[1] > w[0])
}

/// Compute the common frequency span shared by a set of curves.
pub fn common_frequency_range<'a>(
    curves: impl IntoIterator<Item = &'a Curve>,
) -> Option<(f64, f64)> {
    let mut min_freq = f64::NEG_INFINITY;
    let mut max_freq = f64::INFINITY;
    let mut saw_curve = false;

    for curve in curves {
        if !is_valid_frequency_grid(&curve.freq) {
            return None;
        }
        min_freq = min_freq.max(curve.freq[0]);
        max_freq = max_freq.min(curve.freq[curve.freq.len() - 1]);
        saw_curve = true;
    }

    if saw_curve && min_freq < max_freq {
        Some((min_freq, max_freq))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_grid_has_linear_bass_and_logarithmic_treble() {
        let grid = room_eq_hybrid_frequency_grid(DEFAULT_ROOM_EQ_FREQUENCY_SAMPLES);
        assert_eq!(grid[0], 20.0);
        assert_eq!(grid[1], 24.0);
        assert!(grid.iter().any(|frequency| *frequency == 1_000.0));
        assert!((grid[grid.len() - 1] - 20_000.0).abs() < 1e-9);
        assert!(is_valid_frequency_grid(&grid));
    }

    #[test]
    fn clipped_grid_preserves_noncanonical_source_endpoints() {
        let curve = Curve {
            freq: Array1::from_vec(vec![23.0, 100.0, 19_000.0]),
            spl: Array1::zeros(3),
            ..Default::default()
        };
        let grid = clipped_room_eq_frequency_grid(&curve, DEFAULT_ROOM_EQ_FREQUENCY_SAMPLES)
            .expect("valid source span should produce a grid");
        assert_eq!(grid[0], 23.0);
        assert_eq!(grid[grid.len() - 1], 19_000.0);
        assert!(is_valid_frequency_grid(&grid));
    }
}
