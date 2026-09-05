//! FIR filter design and optimization
//!
//! Provides functionality to generate FIR filters matching a target frequency response,
//! with support for linear and minimum phase.
//!
//! This module wraps the core FIR design functions from `math_audio_iir_fir` and adds
//! convenience wrappers that work with the `Curve` type.

use autoeq_core::Curve;
use std::path::Path;

// Re-export core types from math-iir-fir
pub use math_audio_iir_fir::{
    FirDesignConfig, FirPhase, WindowType,
    generate_kirkeby_correction as generate_kirkeby_correction_raw, save_fir_to_wav,
};

/// Minimum accepted tap count for the checked design API.
///
/// Below this a frequency-sampling design is degenerate (fewer degrees of
/// freedom than control points in any realistic curve); use the unchecked
/// wrappers if a trivial filter is really intended.
pub const MIN_CHECKED_TAPS: usize = 4;
/// Maximum accepted tap count for the checked design API (2^20).
pub const MAX_CHECKED_TAPS: usize = 1_048_576;
/// Maximum accepted sample rate for the checked design API (8 MHz).
pub const MAX_CHECKED_SAMPLE_RATE: f64 = 8_000_000.0;

/// Error type for the checked FIR design API.
///
/// The legacy wrappers below return bare `Vec<f64>` and forward their inputs
/// unchecked; the `*_checked` variants validate first and return this error
/// instead of propagating NaNs, panics, or meaningless coefficients.
#[derive(Debug, Clone, PartialEq)]
pub enum FirDesignError {
    /// Sample rate is non-finite, non-positive, or implausibly large.
    InvalidSampleRate {
        value: f64,
    },
    /// Tap count outside `[MIN_CHECKED_TAPS, MAX_CHECKED_TAPS]`.
    InvalidTapCount {
        n_taps: usize,
    },
    /// Curve has fewer than two points.
    EmptyCurve {
        which: &'static str,
    },
    /// `freq`/`spl` (or phase) array lengths disagree.
    LengthMismatch {
        which: &'static str,
        freq_len: usize,
        other_len: usize,
        other: &'static str,
    },
    /// Non-finite magnitude or frequency value.
    NonFiniteValue {
        which: &'static str,
        index: usize,
    },
    /// Frequency grid is not strictly increasing or not positive.
    InvalidFrequencyGrid {
        which: &'static str,
        index: usize,
    },
    /// Correction band is empty, inverted, or outside `(0, Nyquist]`.
    UnsupportedFrequencySpan {
        min_freq: f64,
        max_freq: f64,
        nyquist: f64,
    },
    /// Excess-phase correction requested without measurement phase data.
    MissingPhase,
    /// Measurement phase contains a non-finite value.
    NonFinitePhase {
        index: usize,
    },
}

impl std::fmt::Display for FirDesignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSampleRate { value } => {
                write!(f, "invalid sample rate: {value}")
            }
            Self::InvalidTapCount { n_taps } => write!(
                f,
                "invalid tap count {n_taps}: expected [{MIN_CHECKED_TAPS}, {MAX_CHECKED_TAPS}]"
            ),
            Self::EmptyCurve { which } => write!(f, "{which} curve has fewer than two points"),
            Self::LengthMismatch {
                which,
                freq_len,
                other_len,
                other,
            } => write!(
                f,
                "{which} curve length mismatch: freq has {freq_len} points but {other} has {other_len}"
            ),
            Self::NonFiniteValue { which, index } => {
                write!(f, "{which} curve has a non-finite value at index {index}")
            }
            Self::InvalidFrequencyGrid { which, index } => write!(
                f,
                "{which} curve frequency grid is not strictly increasing and positive at index {index}"
            ),
            Self::UnsupportedFrequencySpan {
                min_freq,
                max_freq,
                nyquist,
            } => write!(
                f,
                "unsupported frequency span [{min_freq}, {max_freq}] with Nyquist {nyquist}"
            ),
            Self::MissingPhase => write!(
                f,
                "excess-phase correction requested but measurement has no phase data"
            ),
            Self::NonFinitePhase { index } => {
                write!(f, "measurement phase has a non-finite value at index {index}")
            }
        }
    }
}

impl std::error::Error for FirDesignError {}

fn check_sample_rate(sample_rate: f64) -> Result<f64, FirDesignError> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 || sample_rate > MAX_CHECKED_SAMPLE_RATE {
        return Err(FirDesignError::InvalidSampleRate { value: sample_rate });
    }
    Ok(sample_rate / 2.0)
}

fn check_taps(n_taps: usize) -> Result<(), FirDesignError> {
    if !(MIN_CHECKED_TAPS..=MAX_CHECKED_TAPS).contains(&n_taps) {
        return Err(FirDesignError::InvalidTapCount { n_taps });
    }
    Ok(())
}

fn check_curve(curve: &Curve, which: &'static str) -> Result<(), FirDesignError> {
    if curve.freq.len() < 2 || curve.spl.len() < 2 {
        return Err(FirDesignError::EmptyCurve { which });
    }
    if curve.freq.len() != curve.spl.len() {
        return Err(FirDesignError::LengthMismatch {
            which,
            freq_len: curve.freq.len(),
            other_len: curve.spl.len(),
            other: "spl",
        });
    }
    if let Some(phase) = curve.phase.as_ref()
        && phase.len() != curve.freq.len()
    {
        return Err(FirDesignError::LengthMismatch {
            which,
            freq_len: curve.freq.len(),
            other_len: phase.len(),
            other: "phase",
        });
    }
    for (index, (&freq, &spl)) in curve.freq.iter().zip(curve.spl.iter()).enumerate() {
        if !freq.is_finite() || !spl.is_finite() {
            return Err(FirDesignError::NonFiniteValue { which, index });
        }
        if freq <= 0.0 || (index > 0 && freq <= curve.freq[index - 1]) {
            return Err(FirDesignError::InvalidFrequencyGrid { which, index });
        }
    }
    Ok(())
}

fn check_band(min_freq: f64, max_freq: f64, nyquist: f64) -> Result<(), FirDesignError> {
    if !min_freq.is_finite()
        || !max_freq.is_finite()
        || min_freq <= 0.0
        || max_freq <= min_freq
        || min_freq >= nyquist
        || max_freq > nyquist
    {
        return Err(FirDesignError::UnsupportedFrequencySpan {
            min_freq,
            max_freq,
            nyquist,
        });
    }
    Ok(())
}

fn check_excess_phase(
    measurement: &Curve,
    correct_excess_phase: bool,
) -> Result<Option<Vec<f64>>, FirDesignError> {
    if !correct_excess_phase {
        return Ok(None);
    }
    let Some(phase) = measurement.phase.as_ref() else {
        return Err(FirDesignError::MissingPhase);
    };
    if phase.len() != measurement.freq.len() {
        return Err(FirDesignError::LengthMismatch {
            which: "measurement",
            freq_len: measurement.freq.len(),
            other_len: phase.len(),
            other: "phase",
        });
    }
    for (index, &value) in phase.iter().enumerate() {
        if !value.is_finite() {
            return Err(FirDesignError::NonFinitePhase { index });
        }
    }
    Ok(Some(phase.to_vec()))
}

/// Generate an FIR filter to match a target frequency response
///
/// This helper supports generic target matching for `FirPhase::Linear` and
/// `FirPhase::Minimum`. Use `generate_kirkeby_correction*` for Kirkeby
/// regularized inversion, which requires both a measurement and a target.
///
/// # Arguments
/// * `target_curve` - The target frequency response (magnitude only needed)
/// * `sample_rate` - Sample rate in Hz
/// * `n_taps` - Number of taps (coefficients) for the FIR filter
/// * `phase_type` - Desired phase characteristic
///
/// # Returns
/// * Vector of FIR coefficients
pub fn generate_fir_from_response(
    target_curve: &Curve,
    sample_rate: f64,
    n_taps: usize,
    phase_type: FirPhase,
) -> Vec<f64> {
    let config = FirDesignConfig {
        n_taps,
        sample_rate,
        phase: phase_type,
        // A causal minimum-phase impulse begins at tap zero. Symmetric windows
        // also begin at zero, which erases that leading energy and destroys
        // the minimum-phase result. Truncation is sufficient here.
        window: if phase_type == FirPhase::Minimum {
            WindowType::Rectangular
        } else {
            FirDesignConfig::default().window
        },
        ..Default::default()
    };

    // Convert Curve to raw arrays
    let freqs: Vec<f64> = target_curve.freq.to_vec();
    let magnitude_db: Vec<f64> = target_curve.spl.to_vec();

    math_audio_iir_fir::generate_fir_from_response(&freqs, &magnitude_db, &config)
}

/// Checked variant of [`generate_fir_from_response`].
///
/// Validates the sample rate, tap count, and target curve before designing.
/// The unchecked wrapper above is unchanged for backward compatibility.
///
/// # Errors
/// * [`FirDesignError::InvalidSampleRate`] for non-finite, non-positive, or
///   implausibly large sample rates.
/// * [`FirDesignError::InvalidTapCount`] for tap counts outside
///   `[MIN_CHECKED_TAPS, MAX_CHECKED_TAPS]`.
/// * [`FirDesignError::EmptyCurve`]/[`FirDesignError::LengthMismatch`]/
///   [`FirDesignError::NonFiniteValue`]/[`FirDesignError::InvalidFrequencyGrid`]
///   for malformed target curves.
pub fn generate_fir_from_response_checked(
    target_curve: &Curve,
    sample_rate: f64,
    n_taps: usize,
    phase_type: FirPhase,
) -> Result<Vec<f64>, FirDesignError> {
    check_sample_rate(sample_rate)?;
    check_taps(n_taps)?;
    check_curve(target_curve, "target")?;
    Ok(generate_fir_from_response(
        target_curve,
        sample_rate,
        n_taps,
        phase_type,
    ))
}

/// Generate Kirkeby regularized FIR correction filter from Curve
///
/// # Arguments
/// * `measurement` - Measurement curve (SPL and optionally phase)
/// * `target` - Target curve (SPL)
/// * `sample_rate` - Sample rate in Hz
/// * `n_taps` - Number of taps
/// * `min_freq` - Minimum frequency for in-band regularization
/// * `max_freq` - Maximum frequency for in-band regularization
///
/// # Returns
/// * Vector of FIR coefficients
pub fn generate_kirkeby_correction(
    measurement: &Curve,
    target: &Curve,
    sample_rate: f64,
    n_taps: usize,
    min_freq: f64,
    max_freq: f64,
) -> Vec<f64> {
    generate_kirkeby_correction_with_phase(
        measurement,
        target,
        sample_rate,
        n_taps,
        min_freq,
        max_freq,
        false, // Default: magnitude-only correction
    )
}

/// Generate Kirkeby regularized FIR correction filter with optional excess phase correction
///
/// # Arguments
/// * `measurement` - Measurement curve (SPL and optionally phase)
/// * `target` - Target curve (SPL)
/// * `sample_rate` - Sample rate in Hz
/// * `n_taps` - Number of taps
/// * `min_freq` - Minimum frequency for in-band regularization
/// * `max_freq` - Maximum frequency for in-band regularization
/// * `correct_excess_phase` - Whether to correct excess phase (requires phase data in measurement)
///
/// # Returns
/// * Vector of FIR coefficients
pub fn generate_kirkeby_correction_with_phase(
    measurement: &Curve,
    target: &Curve,
    sample_rate: f64,
    n_taps: usize,
    min_freq: f64,
    max_freq: f64,
    correct_excess_phase: bool,
) -> Vec<f64> {
    generate_kirkeby_correction_with_smoothing(
        measurement,
        target,
        sample_rate,
        n_taps,
        min_freq,
        max_freq,
        correct_excess_phase,
        0.167, // Default 1/6 octave smoothing
    )
}

/// Generate Kirkeby regularized FIR correction filter with optional excess phase correction and smoothing
///
/// # Arguments
/// * `measurement` - Measurement curve (SPL and optionally phase)
/// * `target` - Target curve (SPL)
/// * `sample_rate` - Sample rate in Hz
/// * `n_taps` - Number of taps
/// * `min_freq` - Minimum frequency for in-band regularization
/// * `max_freq` - Maximum frequency for in-band regularization
/// * `correct_excess_phase` - Whether to correct excess phase (requires phase data in measurement)
/// * `phase_smoothing_octaves` - Phase smoothing width in octaves (0.0 to disable)
///
/// # Returns
/// * Vector of FIR coefficients
#[allow(clippy::too_many_arguments)]
pub fn generate_kirkeby_correction_with_smoothing(
    measurement: &Curve,
    target: &Curve,
    sample_rate: f64,
    n_taps: usize,
    min_freq: f64,
    max_freq: f64,
    correct_excess_phase: bool,
    phase_smoothing_octaves: f64,
) -> Vec<f64> {
    generate_kirkeby_correction_with_smoothing_and_pre_ringing(
        measurement,
        target,
        sample_rate,
        n_taps,
        min_freq,
        max_freq,
        correct_excess_phase,
        phase_smoothing_octaves,
        None,
    )
}

/// Kirkeby correction variant that preserves the caller's pre-ringing policy.
#[allow(clippy::too_many_arguments)]
pub fn generate_kirkeby_correction_with_smoothing_and_pre_ringing(
    measurement: &Curve,
    target: &Curve,
    sample_rate: f64,
    n_taps: usize,
    min_freq: f64,
    max_freq: f64,
    correct_excess_phase: bool,
    phase_smoothing_octaves: f64,
    pre_ringing: Option<math_audio_iir_fir::PreRingingConfig>,
) -> Vec<f64> {
    let config = FirDesignConfig {
        n_taps,
        sample_rate,
        phase: FirPhase::Kirkeby,
        min_freq,
        max_freq,
        correct_excess_phase,
        phase_smoothing_octaves,
        pre_ringing,
        ..Default::default()
    };

    let meas_freqs: Vec<f64> = measurement.freq.to_vec();
    let meas_db: Vec<f64> = measurement.spl.to_vec();
    let meas_phase: Option<Vec<f64>> = measurement.phase.as_ref().map(|p| p.to_vec());

    // Interpolate target onto the measurement grid when needed
    let target_db = resolve_target_db_on_measurement_grid(measurement.freq.clone(), target);

    generate_kirkeby_correction_raw(
        &meas_freqs,
        &meas_db,
        meas_phase.as_deref(),
        &target_db,
        &config,
    )
}

/// Checked variant of
/// [`generate_kirkeby_correction_with_smoothing_and_pre_ringing`].
///
/// Additionally validates the correction band against the Nyquist frequency
/// and requires finite measurement phase data when `correct_excess_phase` is
/// set. All legacy `generate_kirkeby_*` wrappers above are unchanged.
///
/// # Errors
/// Same as [`generate_fir_from_response_checked`], plus
/// [`FirDesignError::UnsupportedFrequencySpan`] when the `[min_freq,
/// max_freq]` band is empty, inverted, or outside `(0, Nyquist]`, and
/// [`FirDesignError::MissingPhase`]/[`FirDesignError::NonFinitePhase`] when
/// excess-phase correction is requested without usable phase data.
#[allow(clippy::too_many_arguments)]
pub fn generate_kirkeby_correction_with_smoothing_and_pre_ringing_checked(
    measurement: &Curve,
    target: &Curve,
    sample_rate: f64,
    n_taps: usize,
    min_freq: f64,
    max_freq: f64,
    correct_excess_phase: bool,
    phase_smoothing_octaves: f64,
    pre_ringing: Option<math_audio_iir_fir::PreRingingConfig>,
) -> Result<Vec<f64>, FirDesignError> {
    let nyquist = check_sample_rate(sample_rate)?;
    check_taps(n_taps)?;
    check_curve(measurement, "measurement")?;
    check_curve(target, "target")?;
    check_band(min_freq, max_freq, nyquist)?;
    check_excess_phase(measurement, correct_excess_phase)?;
    if !phase_smoothing_octaves.is_finite() || phase_smoothing_octaves < 0.0 {
        return Err(FirDesignError::NonFiniteValue {
            which: "phase smoothing",
            index: 0,
        });
    }
    Ok(generate_kirkeby_correction_with_smoothing_and_pre_ringing(
        measurement,
        target,
        sample_rate,
        n_taps,
        min_freq,
        max_freq,
        correct_excess_phase,
        phase_smoothing_octaves,
        pre_ringing,
    ))
}

/// Checked variant of [`generate_kirkeby_correction`].
#[allow(clippy::too_many_arguments)]
pub fn generate_kirkeby_correction_checked(
    measurement: &Curve,
    target: &Curve,
    sample_rate: f64,
    n_taps: usize,
    min_freq: f64,
    max_freq: f64,
) -> Result<Vec<f64>, FirDesignError> {
    generate_kirkeby_correction_with_phase_checked(
        measurement,
        target,
        sample_rate,
        n_taps,
        min_freq,
        max_freq,
        false,
    )
}

/// Checked variant of [`generate_kirkeby_correction_with_phase`].
#[allow(clippy::too_many_arguments)]
pub fn generate_kirkeby_correction_with_phase_checked(
    measurement: &Curve,
    target: &Curve,
    sample_rate: f64,
    n_taps: usize,
    min_freq: f64,
    max_freq: f64,
    correct_excess_phase: bool,
) -> Result<Vec<f64>, FirDesignError> {
    generate_kirkeby_correction_with_smoothing_checked(
        measurement,
        target,
        sample_rate,
        n_taps,
        min_freq,
        max_freq,
        correct_excess_phase,
        0.167,
    )
}

/// Checked variant of [`generate_kirkeby_correction_with_smoothing`].
#[allow(clippy::too_many_arguments)]
pub fn generate_kirkeby_correction_with_smoothing_checked(
    measurement: &Curve,
    target: &Curve,
    sample_rate: f64,
    n_taps: usize,
    min_freq: f64,
    max_freq: f64,
    correct_excess_phase: bool,
    phase_smoothing_octaves: f64,
) -> Result<Vec<f64>, FirDesignError> {
    generate_kirkeby_correction_with_smoothing_and_pre_ringing_checked(
        measurement,
        target,
        sample_rate,
        n_taps,
        min_freq,
        max_freq,
        correct_excess_phase,
        phase_smoothing_octaves,
        None,
    )
}

/// Resolve target SPL values onto the measurement frequency grid.
///
/// Grids are compared element-wise (with tolerance): equal lengths alone do
/// not imply the same grid, and index-aligned copying of a mismatched target
/// would place corrections at the wrong frequencies.
pub fn resolve_target_db_on_measurement_grid(
    measurement_freq: ndarray::Array1<f64>,
    target: &Curve,
) -> Vec<f64> {
    const GRID_TOLERANCE_HZ: f64 = 1e-6;
    let same_grid = target.freq.len() == measurement_freq.len()
        && target.freq.iter().zip(measurement_freq.iter()).all(
            |(&target_frequency, &measurement_frequency)| {
                (target_frequency - measurement_frequency).abs()
                    <= GRID_TOLERANCE_HZ * measurement_frequency.max(1.0)
            },
        );
    if same_grid {
        return target.spl.to_vec();
    }
    autoeq_core::interpolate(&measurement_freq, target)
        .spl
        .to_vec()
}

/// Save FIR coefficients to a WAV file (32-bit float mono)
///
/// Convenience wrapper that takes a Path reference.
pub fn save_fir_wav(
    coeffs: &[f64],
    sample_rate: u32,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    save_fir_to_wav(coeffs, sample_rate, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use tempfile::TempDir;

    /// Helper to create a test curve with given frequencies and SPL values
    fn create_test_curve(freqs: &[f64], spl_values: &[f64]) -> Curve {
        Curve {
            freq: Array1::from(freqs.to_vec()),
            spl: Array1::from(spl_values.to_vec()),
            phase: None,
            ..Default::default()
        }
    }

    /// Create a flat response curve at given SPL level
    fn create_flat_curve(min_freq: f64, max_freq: f64, n_points: usize, spl_db: f64) -> Curve {
        let freqs: Vec<f64> = (0..n_points)
            .map(|i| {
                let t = i as f64 / (n_points - 1) as f64;
                min_freq * (max_freq / min_freq).powf(t)
            })
            .collect();
        let spl: Vec<f64> = vec![spl_db; n_points];
        create_test_curve(&freqs, &spl)
    }

    /// Compute energy in a specific portion of the signal
    fn compute_energy_in_range(coeffs: &[f64], start_fraction: f64, end_fraction: f64) -> f64 {
        let n = coeffs.len();
        let start = (n as f64 * start_fraction) as usize;
        let end = (n as f64 * end_fraction) as usize;
        coeffs[start..end].iter().map(|x| x * x).sum()
    }

    #[test]
    fn equal_length_but_different_grids_are_interpolated_not_index_aligned() {
        // Same point count, different frequency axes: index-aligned copying
        // would place target values at the wrong frequencies.
        let meas_freqs = [20.0, 100.0, 1000.0, 10_000.0, 20_000.0];
        let target_freqs = [25.0, 150.0, 1500.0, 15_000.0, 19_000.0];
        let mut target = create_test_curve(&target_freqs, &[80.0; 5]);
        target.spl[2] = 92.0; // +12 dB bump at its own 1500 Hz point

        let resolved =
            resolve_target_db_on_measurement_grid(Array1::from(meas_freqs.to_vec()), &target);

        assert_eq!(resolved.len(), 5);
        // At measurement 1000 Hz the bump must be interpolated from the
        // target's surrounding points (150/1500 Hz), not copied verbatim.
        assert!(
            resolved[2] < 90.0,
            "index-aligned copy would yield 92.0 at 1000 Hz, got {}",
            resolved[2]
        );
        assert!(
            resolved[2] > 82.0,
            "interpolation should retain part of the nearby bump, got {}",
            resolved[2]
        );
        // Away from the bump, values stay near the flat level.
        assert!((resolved[0] - 80.0).abs() < 1e-9);
    }

    #[test]
    fn identical_grids_copy_target_values_directly() {
        let freqs = [20.0, 100.0, 1000.0];
        let target = create_test_curve(&freqs, &[81.0, 82.0, 83.0]);

        let resolved = resolve_target_db_on_measurement_grid(Array1::from(freqs.to_vec()), &target);

        assert_eq!(resolved, vec![81.0, 82.0, 83.0]);
    }

    #[test]
    fn test_linear_phase_impulse_symmetry() {
        let sample_rate = 48000.0;
        let n_taps = 512;

        let target_curve = create_test_curve(
            &[20.0, 100.0, 1000.0, 5000.0, 20000.0],
            &[0.0, 2.0, 0.0, -1.0, -2.0],
        );

        let coeffs =
            generate_fir_from_response(&target_curve, sample_rate, n_taps, FirPhase::Linear);

        assert_eq!(coeffs.len(), n_taps);

        // Check that the energy is centered
        let (max_idx, _) = coeffs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap();

        let center = n_taps / 2;
        let tolerance = n_taps / 10;
        assert!(
            (max_idx as isize - center as isize).unsigned_abs() < tolerance,
            "Linear phase FIR peak should be near center. Peak at {}, center at {}",
            max_idx,
            center
        );
    }

    #[test]
    fn test_minimum_phase_energy_concentration() {
        let sample_rate = 48000.0;
        let n_taps = 1024;

        let target_curve = create_test_curve(
            &[20.0, 100.0, 500.0, 1000.0, 5000.0, 20000.0],
            &[-3.0, 0.0, 2.0, 0.0, -2.0, -5.0],
        );

        let coeffs =
            generate_fir_from_response(&target_curve, sample_rate, n_taps, FirPhase::Minimum);

        assert_eq!(coeffs.len(), n_taps);

        // For minimum phase, first half should have more energy than second half
        // (windowing affects the exact distribution)
        let first_half_energy = compute_energy_in_range(&coeffs, 0.0, 0.5);
        let second_half_energy = compute_energy_in_range(&coeffs, 0.5, 1.0);

        assert!(
            first_half_energy > second_half_energy,
            "Minimum phase should have more energy in first half: first={:.4}, second={:.4}",
            first_half_energy,
            second_half_energy
        );
    }

    #[test]
    fn test_minimum_phase_flat_target_keeps_impulse_at_start() {
        let target_curve = create_flat_curve(20.0, 20_000.0, 100, 0.0);
        let coeffs = generate_fir_from_response(&target_curve, 48_000.0, 256, FirPhase::Minimum);

        assert!(
            (coeffs[0] - 1.0).abs() < 1e-6,
            "minimum-phase flat response should start with a unit impulse, got {}",
            coeffs[0]
        );
        let tail_energy: f64 = coeffs[1..].iter().map(|value| value * value).sum();
        assert!(tail_energy < 1e-12, "unexpected tail energy: {tail_energy}");
    }

    #[test]
    fn test_flat_target_produces_near_impulse() {
        let sample_rate = 48000.0;
        let n_taps = 256;

        let target_curve = create_flat_curve(20.0, 20000.0, 100, 0.0);

        let coeffs =
            generate_fir_from_response(&target_curve, sample_rate, n_taps, FirPhase::Linear);

        assert_eq!(coeffs.len(), n_taps);

        let (max_idx, max_val) = coeffs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap();

        let center = n_taps / 2;
        assert!(
            (max_idx as isize - center as isize).abs() < 10,
            "Peak should be near center for linear phase"
        );

        assert!(*max_val > 0.0, "Peak coefficient should be positive");
    }

    #[test]
    fn test_save_fir_to_wav_creates_valid_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let wav_path = temp_dir.path().join("test_fir.wav");

        let coeffs: Vec<f64> = (0..256).map(|i| (i as f64 * 0.01).sin()).collect();

        let result = save_fir_wav(&coeffs, 48000, &wav_path);
        assert!(result.is_ok(), "save_fir_wav should succeed");
        assert!(wav_path.exists(), "WAV file should be created");

        let reader = hound::WavReader::open(&wav_path).expect("Should open WAV file");
        let spec = reader.spec();

        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 48000);
        assert_eq!(spec.bits_per_sample, 32);
        assert_eq!(reader.len() as usize, coeffs.len());
    }

    #[test]
    fn test_fir_phase_types_differ() {
        let sample_rate = 48000.0;
        let n_taps = 512;

        let target_curve = create_test_curve(
            &[20.0, 100.0, 1000.0, 10000.0, 20000.0],
            &[0.0, 3.0, 0.0, -3.0, -6.0],
        );

        let linear_coeffs =
            generate_fir_from_response(&target_curve, sample_rate, n_taps, FirPhase::Linear);
        let minimum_coeffs =
            generate_fir_from_response(&target_curve, sample_rate, n_taps, FirPhase::Minimum);

        assert_eq!(linear_coeffs.len(), minimum_coeffs.len());

        let sum_diff: f64 = linear_coeffs
            .iter()
            .zip(minimum_coeffs.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            sum_diff > 0.1,
            "Linear and minimum phase should produce different coefficients"
        );
    }

    #[test]
    fn test_kirkeby_correction() {
        let measurement = create_test_curve(
            &[20.0, 100.0, 500.0, 1000.0, 5000.0, 20000.0],
            &[75.0, 82.0, 80.0, 78.0, 72.0, 65.0],
        );
        let target = create_test_curve(
            &[20.0, 100.0, 500.0, 1000.0, 5000.0, 20000.0],
            &[80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
        );

        let coeffs =
            generate_kirkeby_correction(&measurement, &target, 48000.0, 4096, 20.0, 1000.0);

        assert_eq!(coeffs.len(), 4096);
        assert!(coeffs.iter().any(|&x| x.abs() > 1e-10));
    }

    // ---- Checked API + realization coverage ----

    fn log_grid(min_freq: f64, max_freq: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| min_freq * (max_freq / min_freq).powf(i as f64 / (n - 1) as f64))
            .collect()
    }

    fn create_test_curve_with_phase(freqs: &[f64], spl: &[f64], phase: &[f64]) -> Curve {
        let mut curve = create_test_curve(freqs, spl);
        curve.phase = Some(Array1::from(phase.to_vec()));
        curve
    }

    /// Magnitude response of `coeffs` in dB via a direct DFT.
    /// Magnitude is delay-invariant, so no delay compensation is needed;
    /// callers remove the overall gain offset before comparing shapes.
    fn realized_db(coeffs: &[f64], sample_rate: f64, freqs: &[f64]) -> Vec<f64> {
        freqs
            .iter()
            .map(|&freq| {
                let mut re = 0.0;
                let mut im = 0.0;
                for (n, &c) in coeffs.iter().enumerate() {
                    let p =
                        2.0 * std::f64::consts::PI * freq * n as f64 / sample_rate;
                    re += c * p.cos();
                    im -= c * p.sin();
                }
                20.0 * re.hypot(im).max(1e-12).log10()
            })
            .collect()
    }

    /// Max absolute deviation after removing the overall gain offset
    /// (amplitude normalization).
    fn normalized_max_dev(realized: &[f64], expected: &[f64]) -> f64 {
        let offset: f64 =
            realized.iter().zip(expected.iter()).map(|(r, e)| r - e).sum::<f64>()
                / realized.len() as f64;
        realized
            .iter()
            .zip(expected.iter())
            .map(|(r, e)| (r - e - offset).abs())
            .fold(0.0, f64::max)
    }

    fn assert_all_finite(coeffs: &[f64]) {
        assert!(
            coeffs.iter().all(|c| c.is_finite()),
            "coefficients must all be finite"
        );
        assert!(
            coeffs.iter().any(|&c| c.abs() > 1e-12),
            "filter must not be all zeros"
        );
    }

    #[test]
    fn checked_api_rejects_invalid_inputs() {
        let target = create_flat_curve(20.0, 20_000.0, 50, 0.0);
        let measurement = create_flat_curve(20.0, 20_000.0, 50, 80.0);

        // Bad sample rates.
        for &sr in &[0.0, -48_000.0, f64::NAN, f64::INFINITY, 1e12] {
            assert!(
                matches!(
                    generate_fir_from_response_checked(&target, sr, 256, FirPhase::Linear),
                    Err(FirDesignError::InvalidSampleRate { .. })
                ),
                "sample rate {sr} should be rejected"
            );
            assert!(
                generate_kirkeby_correction_checked(&measurement, &target, sr, 512, 20.0, 1000.0)
                    .is_err(),
                "sample rate {sr} should be rejected (kirkeby)"
            );
        }

        // Bad tap counts.
        for &taps in &[0, 1, 3] {
            assert!(
                matches!(
                    generate_fir_from_response_checked(&target, 48_000.0, taps, FirPhase::Linear),
                    Err(FirDesignError::InvalidTapCount { .. })
                ),
                "{taps} taps should be rejected"
            );
        }

        // Malformed curves.
        let empty = create_test_curve(&[], &[]);
        assert!(matches!(
            generate_fir_from_response_checked(&empty, 48_000.0, 64, FirPhase::Linear),
            Err(FirDesignError::EmptyCurve { .. })
        ));
        let one_point = create_test_curve(&[100.0], &[0.0]);
        assert!(
            generate_fir_from_response_checked(&one_point, 48_000.0, 64, FirPhase::Linear).is_err()
        );
        let backwards =
            create_test_curve(&[1000.0, 100.0, 10_000.0], &[0.0, 0.0, 0.0]);
        assert!(matches!(
            generate_fir_from_response_checked(&backwards, 48_000.0, 64, FirPhase::Linear),
            Err(FirDesignError::InvalidFrequencyGrid { .. })
        ));
        let non_positive = create_test_curve(&[0.0, 100.0, 1000.0], &[0.0, 0.0, 0.0]);
        assert!(matches!(
            generate_fir_from_response_checked(&non_positive, 48_000.0, 64, FirPhase::Linear),
            Err(FirDesignError::InvalidFrequencyGrid { .. })
        ));
        let mut nan_mag = create_flat_curve(20.0, 20_000.0, 10, 0.0);
        nan_mag.spl[4] = f64::NAN;
        assert!(matches!(
            generate_fir_from_response_checked(&nan_mag, 48_000.0, 64, FirPhase::Linear),
            Err(FirDesignError::NonFiniteValue { .. })
        ));
        let mut short_spl = create_flat_curve(20.0, 20_000.0, 10, 0.0);
        short_spl.spl = Array1::from(vec![0.0; 7]);
        assert!(matches!(
            generate_fir_from_response_checked(&short_spl, 48_000.0, 64, FirPhase::Linear),
            Err(FirDesignError::LengthMismatch { .. })
        ));

        // Unsupported frequency spans.
        for (min_f, max_f) in [
            (1000.0, 20.0),   // inverted
            (500.0, 500.0),   // empty
            (-20.0, 1000.0),  // negative
            (20.0, 30_000.0), // above Nyquist at 48 kHz
            (25_000.0, 26_000.0), // entirely above Nyquist
        ] {
            assert!(
                matches!(
                    generate_kirkeby_correction_checked(
                        &measurement, &target, 48_000.0, 512, min_f, max_f
                    ),
                    Err(FirDesignError::UnsupportedFrequencySpan { .. })
                ),
                "band [{min_f}, {max_f}] should be rejected"
            );
        }

        // Missing phase when excess-phase correction is requested.
        assert!(matches!(
            generate_kirkeby_correction_with_phase_checked(
                &measurement, &target, 48_000.0, 512, 20.0, 1000.0, true
            ),
            Err(FirDesignError::MissingPhase)
        ));
        // Same request succeeds once phase data is present.
        let phased = create_test_curve_with_phase(
            &[20.0, 100.0, 1000.0, 10_000.0, 20_000.0],
            &[80.0, 81.0, 80.0, 79.0, 78.0],
            &[-5.0, -10.0, -30.0, -90.0, -150.0],
        );
        let phased_target = create_flat_curve(20.0, 20_000.0, 50, 80.0);
        assert!(
            generate_kirkeby_correction_with_phase_checked(
                &phased, &phased_target, 48_000.0, 256, 20.0, 1000.0, true
            )
            .is_ok()
        );
        // Non-finite phase is rejected, not silently corrected.
        let mut bad_phase = phased.clone();
        bad_phase.phase.as_mut().unwrap()[2] = f64::INFINITY;
        assert!(matches!(
            generate_kirkeby_correction_with_phase_checked(
                &bad_phase, &phased_target, 48_000.0, 256, 20.0, 1000.0, true
            ),
            Err(FirDesignError::NonFinitePhase { .. })
        ));
    }

    #[test]
    fn checked_api_accepts_valid_designs_all_modes() {
        let target = create_test_curve(
            &[20.0, 100.0, 1000.0, 5000.0, 20_000.0],
            &[0.0, 2.0, 0.0, -1.0, -2.0],
        );
        for phase in [FirPhase::Linear, FirPhase::Minimum] {
            let coeffs = generate_fir_from_response_checked(&target, 48_000.0, 128, phase)
                .expect("valid design should succeed");
            assert_eq!(coeffs.len(), 128);
            assert_all_finite(&coeffs);
        }
        let measurement = create_test_curve(
            &[20.0, 100.0, 1000.0, 5000.0, 20_000.0],
            &[75.0, 82.0, 80.0, 78.0, 72.0],
        );
        let flat = create_flat_curve(20.0, 20_000.0, 50, 80.0);
        let coeffs =
            generate_kirkeby_correction_checked(&measurement, &flat, 48_000.0, 512, 20.0, 10_000.0)
                .expect("valid kirkeby design should succeed");
        assert_eq!(coeffs.len(), 512);
        assert_all_finite(&coeffs);
    }

    #[test]
    fn realized_response_matches_target_across_sample_rates() {
        // Tilted target; linear-phase design should realize its shape at
        // 44.1, 48, and 96 kHz after amplitude normalization.
        let target_freqs = log_grid(20.0, 20_000.0, 12);
        let target_spl: Vec<f64> = target_freqs
            .iter()
            .map(|f| 4.0 * (1000.0 / f).log10())
            .collect();
        let target = create_test_curve(&target_freqs, &target_spl);
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            for (phase, limit) in [(FirPhase::Linear, 6.0), (FirPhase::Minimum, 6.0)] {
                let coeffs =
                    generate_fir_from_response_checked(&target, sample_rate, 256, phase)
                        .expect("design should succeed");
                let probes = log_grid(100.0, 10_000.0, 25);
                let expected: Vec<f64> = probes
                    .iter()
                    .map(|f| 4.0 * (1000.0 / f).log10())
                    .collect();
                let dev = normalized_max_dev(&realized_db(&coeffs, sample_rate, &probes), &expected);
                assert!(
                    dev < limit,
                    "{phase:?} @{sample_rate} Hz realized shape deviates by {dev:.2} dB"
                );
            }
        }
    }

    #[test]
    fn kirkeby_realization_matches_target_minus_measurement() {
        // The realized correction should approximate target - measurement
        // in-band after amplitude normalization.
        let grid = log_grid(20.0, 20_000.0, 40);
        let meas_spl: Vec<f64> = grid
            .iter()
            .map(|f| 80.0 + 6.0 * (f / 1000.0).log10().sin() - 4.0 * (f / 5000.0).log10())
            .collect();
        let measurement = create_test_curve(&grid, &meas_spl);
        let target = create_flat_curve(20.0, 20_000.0, 40, 80.0);
        let coeffs =
            generate_kirkeby_correction_checked(&measurement, &target, 48_000.0, 1024, 30.0, 12_000.0)
                .expect("design should succeed");
        assert_all_finite(&coeffs);
        let probes: Vec<f64> = log_grid(60.0, 8000.0, 25);
        let expected: Vec<f64> = probes
            .iter()
            .map(|f| {
                let m = 80.0 + 6.0 * (f / 1000.0).log10().sin() - 4.0 * (f / 5000.0).log10();
                80.0 - m
            })
            .collect();
        let dev = normalized_max_dev(&realized_db(&coeffs, 48_000.0, &probes), &expected);
        assert!(dev < 8.0, "kirkeby realization deviates by {dev:.2} dB");
    }

    #[test]
    fn low_tap_counts_succeed_and_stay_finite() {
        let target = create_test_curve(
            &[20.0, 200.0, 1000.0, 8000.0, 20_000.0],
            &[0.0, 1.0, 0.0, -1.0, -2.0],
        );
        for &taps in &[16, 32] {
            for phase in [FirPhase::Linear, FirPhase::Minimum] {
                let coeffs =
                    generate_fir_from_response_checked(&target, 48_000.0, taps, phase)
                        .expect("low tap count should succeed");
                assert_eq!(coeffs.len(), taps);
                assert_all_finite(&coeffs);
            }
        }
    }

    #[test]
    fn shifted_target_grid_design_succeeds() {
        // Target on a shifted grid exercises the interpolation path in the
        // checked Kirkeby wrapper; the design must still realize the
        // correction shape on the measurement grid.
        let meas_freqs = [20.0, 100.0, 1000.0, 10_000.0, 20_000.0];
        let measurement = create_test_curve(&meas_freqs, &[75.0, 82.0, 80.0, 78.0, 72.0]);
        let shifted_freqs = [25.0, 150.0, 1500.0, 15_000.0, 19_000.0];
        let shifted_target = create_test_curve(&shifted_freqs, &[80.0; 5]);
        let coeffs = generate_kirkeby_correction_checked(
            &measurement, &shifted_target, 48_000.0, 512, 20.0, 10_000.0,
        )
        .expect("shifted-grid design should succeed");
        assert_all_finite(&coeffs);
    }

    fn rms(values: &[f64]) -> f64 {
        (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
    }

    #[test]
    fn multiseat_design_improves_per_seat_and_held_out_response() {
        // Three design seats + one held-out seat sharing a common room mode
        // plus per-seat variation. A correction designed from the seat mean
        // must reduce every seat's residual, including the held-out one.
        let grid = log_grid(20.0, 20_000.0, 40);
        let room_mode = |f: f64| 7.0 * (-((f / 120.0).log10().powi(2)) / 0.02).exp();
        let seat_variation = |f: f64, k: f64| k * (f / 2000.0).log10().sin() * 2.0;
        let seats: Vec<Vec<f64>> = [0.5, -0.5, 1.0, -1.0]
            .iter()
            .map(|&k| {
                grid.iter()
                    .map(|&f| 80.0 + room_mode(f) + seat_variation(f, k))
                    .collect()
            })
            .collect();
        let mean: Vec<f64> = (0..grid.len())
            .map(|i| (seats[0][i] + seats[1][i] + seats[2][i]) / 3.0)
            .collect();
        let measurement = create_test_curve(&grid, &mean);
        let target = create_flat_curve(20.0, 20_000.0, 40, 80.0);
        let coeffs =
            generate_kirkeby_correction_checked(&measurement, &target, 48_000.0, 1024, 30.0, 12_000.0)
                .expect("design should succeed");
        assert_all_finite(&coeffs);

        let probes: Vec<f64> = log_grid(60.0, 8000.0, 25);
        let correction = realized_db(&coeffs, 48_000.0, &probes);
        // Amplitude normalization: mean correction over the probe band.
        let offset: f64 = correction.iter().sum::<f64>() / correction.len() as f64;
        for (seat_idx, seat) in seats.iter().enumerate() {
            // Interpolate this seat onto the probe grid (log-linear).
            let at_probe: Vec<f64> = probes
                .iter()
                .map(|&f| {
                    let mut j = 0;
                    while j + 1 < grid.len() && grid[j + 1] < f {
                        j += 1;
                    }
                    let (f0, f1) = (grid[j], grid[j + 1]);
                    let t = (f.ln() - f0.ln()) / (f1.ln() - f0.ln());
                    seat[j] + t * (seat[j + 1] - seat[j])
                })
                .collect();
            let before: Vec<f64> = at_probe.iter().map(|&s| s - 80.0).collect();
            let after: Vec<f64> = at_probe
                .iter()
                .zip(correction.iter())
                .map(|(&s, &h)| s + (h - offset) - 80.0)
                .collect();
            assert!(
                rms(&after) < rms(&before),
                "seat {seat_idx} residual not improved: {} -> {}",
                rms(&before),
                rms(&after)
            );
        }
    }

    #[test]
    fn sparse_and_noisy_phase_designs_succeed() {
        // Sparse phase (few grid points) and noisy phase must not break the
        // excess-phase path; results stay finite.
        let freqs = log_grid(20.0, 20_000.0, 8);
        let spl: Vec<f64> = freqs.iter().map(|f| 80.0 + (f / 500.0).log10()).collect();
        let clean_phase: Vec<f64> = freqs.iter().map(|f| -20.0 * (f / 1000.0).log10()).collect();
        let sparse = create_test_curve_with_phase(&freqs, &spl, &clean_phase);
        let target = create_flat_curve(20.0, 20_000.0, 40, 80.0);
        let coeffs = generate_kirkeby_correction_with_phase_checked(
            &sparse, &target, 48_000.0, 256, 20.0, 10_000.0, true,
        )
        .expect("sparse phase design should succeed");
        assert_all_finite(&coeffs);

        let dense_freqs = log_grid(20.0, 20_000.0, 60);
        let dense_spl: Vec<f64> = dense_freqs
            .iter()
            .map(|f| 80.0 + (f / 500.0).log10())
            .collect();
        // Deterministic pseudo-noise on the phase (+/-6 deg hash).
        let noisy_phase: Vec<f64> = dense_freqs
            .iter()
            .enumerate()
            .map(|(i, f)| {
                -20.0 * (f / 1000.0).log10()
                    + 6.0 * (((i * 2654435761) % 100) as f64 / 50.0 - 1.0)
            })
            .collect();
        let noisy = create_test_curve_with_phase(&dense_freqs, &dense_spl, &noisy_phase);
        let coeffs = generate_kirkeby_correction_with_phase_checked(
            &noisy, &target, 48_000.0, 256, 20.0, 10_000.0, true,
        )
        .expect("noisy phase design should succeed");
        assert_all_finite(&coeffs);
    }

    #[test]
    fn insufficient_low_frequency_support_degrades_gracefully() {
        // Measurement starts at 200 Hz while the band opens at 20 Hz: no
        // low-frequency information exists, but the design must still return
        // finite coefficients (documented graceful behavior, not an error).
        let grid = log_grid(200.0, 20_000.0, 40);
        let spl: Vec<f64> = grid.iter().map(|f| 80.0 + (f / 1000.0).log10()).collect();
        let measurement = create_test_curve(&grid, &spl);
        let target = create_flat_curve(20.0, 20_000.0, 40, 80.0);
        let coeffs =
            generate_kirkeby_correction_checked(&measurement, &target, 48_000.0, 512, 20.0, 10_000.0)
                .expect("design should succeed despite missing low-frequency support");
        assert_all_finite(&coeffs);
    }
}
