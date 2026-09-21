//! Fast psychoacoustic-principle coverage, section A (core half).
//!
//! Production entry points under test:
//! - [`crate::auditory_frequency::erb_rate_weighted_rms`] (grid consistency,
//!   invalid axes stay unassessed instead of becoming zero error).
//! - [`crate::evidence::EvidenceBand::validate`] (gap/bound rejection;
//!   negative SNR stays representable as evidence).
//! - [`crate::curve_transforms::interpolate`] and
//!   [`crate::curve_transforms::create_log_frequency_grid`] (design data
//!   survives transforms; refined grids bracket narrow features).
//!
//! Oracles are independent analytic constants and separately constructed
//! grids, never the implementation's own helpers.

use ndarray::Array1;

use crate::auditory_frequency::erb_rate_weighted_rms;
use crate::curve_transforms::{create_log_frequency_grid, interpolate};
use crate::curve::Curve;
use crate::evidence::EvidenceBand;

fn band(id: &str, low_hz: f64, high_hz: f64) -> EvidenceBand {
    EvidenceBand {
        id: id.to_string(),
        low_hz,
        high_hz,
        snr_db: None,
        coherence: None,
        spread: None,
        timing_uncertainty_s: None,
        reasons: Vec::new(),
        references: Vec::new(),
    }
}

/// A04: one constant error measures the same on a uniform-log grid and on an
/// irregular sorted grid. Weighting must not depend on grid density.
#[test]
fn psycho_fast_a04_constant_weighting_grid_consistent() {
    let uniform = Array1::linspace(20.0, 20_000.0, 128);
    let irregular = Array1::from(vec![
        20.0, 25.0, 41.0, 90.0, 210.0, 500.0, 1_150.0, 2_600.0, 6_000.0, 13_000.0, 20_000.0,
    ]);
    let uniform_values = vec![2.5; uniform.len()];
    let irregular_values = vec![2.5; irregular.len()];
    let uniform_rms = erb_rate_weighted_rms(&uniform, &uniform_values).expect("valid axis");
    let irregular_rms = erb_rate_weighted_rms(&irregular, &irregular_values).expect("valid axis");
    assert!(
        (uniform_rms - 2.5).abs() <= 1e-9,
        "uniform grid rms {uniform_rms}"
    );
    assert!(
        (uniform_rms - irregular_rms).abs() <= 1e-9,
        "uniform {uniform_rms} vs irregular {irregular_rms}"
    );
}

/// A04: invalid axes and gaps never become zero error or high confidence.
/// The strict path returns `None`; callers must treat that as unassessed.
#[test]
fn psycho_fast_a04_invalid_axis_never_silent_zero() {
    let freqs = Array1::linspace(20.0, 20_000.0, 64);
    let values = vec![1.0; 64];
    // Length mismatch.
    assert!(erb_rate_weighted_rms(&freqs, &values[..63]).is_none());
    // Unsorted axis.
    let mut unsorted = freqs.to_vec();
    unsorted.swap(10, 40);
    assert!(erb_rate_weighted_rms(&Array1::from(unsorted), &values).is_none());
    // Empty input.
    assert!(
        erb_rate_weighted_rms(&Array1::from(Vec::<f64>::new()), &[]).is_none()
    );
    // Non-finite value poisons the result instead of scoring as good.
    let mut nonfinite = values.clone();
    nonfinite[30] = f64::NAN;
    assert!(erb_rate_weighted_rms(&freqs, &nonfinite).is_none());
}

/// A04/F05: bands with reversed, zero-width, or non-positive bounds are
/// rejected, so a coverage gap cannot be interpolated across as valid
/// support. Negative SNR stays representable: low SNR is evidence, not
/// malformed input.
#[test]
fn psycho_fast_a04_gap_bands_rejected_negative_snr_kept() {
    assert!(band("ok", 20.0, 400.0).validate("ctx").is_ok());
    assert!(band("reversed", 400.0, 20.0).validate("ctx").is_err());
    assert!(band("zero-width", 100.0, 100.0).validate("ctx").is_err());
    assert!(band("non-positive", 0.0, 100.0).validate("ctx").is_err());
    let mut bad_coherence = band("bad-coherence", 20.0, 400.0);
    bad_coherence.coherence = Some(1.5);
    assert!(bad_coherence.validate("ctx").is_err());
    let mut low_snr = band("low-snr", 20.0, 400.0);
    low_snr.snr_db = Some(-12.0);
    assert!(
        low_snr.validate("ctx").is_ok(),
        "negative SNR is evidence, not malformed input"
    );
}

/// A05: interpolation reproduces design data on its native grid exactly, so
/// no silent smoothing erases measured support during grid transforms.
#[test]
fn psycho_fast_a05_native_grid_interpolation_preserves_design_data() {
    let freqs = create_log_frequency_grid(256, 20.0, 20_000.0);
    let spl = freqs
        .iter()
        .map(|f| -6.0 * (-((f.ln() - 1_000.0_f64.ln()).powi(2)) / 0.002).exp())
        .collect::<Vec<_>>();
    let curve = Curve {
        freq: freqs.clone(),
        spl: Array1::from(spl),
        ..Default::default()
    };
    let back = interpolate(&freqs, &curve);
    let worst = back
        .spl
        .iter()
        .zip(curve.spl.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(worst <= 1e-9, "native-grid drift {worst}");
}

/// A05: a 1024-point refined log grid samples within half a bin of a 1 kHz
/// narrow feature and brackets it, so extremum controls actually sample
/// the feature instead of stepping over it.
#[test]
fn psycho_fast_a05_refined_grid_brackets_narrow_feature() {
    let grid = create_log_frequency_grid(1024, 20.0, 20_000.0);
    assert_eq!(grid.len(), 1024);
    let center = 1_000.0;
    let nearest = grid
        .iter()
        .min_by(|a, b| {
            (*a - center)
                .abs()
                .partial_cmp(&(*b - center).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("nonempty grid");
    let position = grid.iter().position(|f| f == nearest).expect("member");
    assert!(position > 0 && position + 1 < grid.len(), "center bracketed");
    let half_bin = 0.25 * (grid[position + 1] - grid[position - 1]);
    assert!(
        (*nearest - center).abs() <= half_bin + 1e-9,
        "nearest grid point {nearest} too far from 1 kHz"
    );
}
