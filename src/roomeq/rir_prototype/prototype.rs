//! Weighted RIR prototype builder.

use crate::Curve;
use crate::roomeq::rir_prototype::config::{
    DirectivityModel, DistanceWeightMode, RirPrototypeConfig,
};
use crate::roomeq::rir_prototype::weights::{
    compute_angles, compute_distances, normalized_weights,
};
use ndarray::Array1;

/// Tolerance for comparing frequency-grid values (relative or absolute).
const FREQ_GRID_TOLERANCE: f64 = 1e-6;

/// Errors that can occur while building a weighted RIR prototype.
#[derive(Debug, thiserror::Error)]
pub enum RirPrototypeError {
    /// No curves were provided.
    #[error("build_weighted_prototype: no curves provided")]
    NoCurves,
    /// The number of microphone positions does not match the number of curves.
    #[error("microphone_positions length ({0}) must match curves length ({1})")]
    MismatchedCounts(usize, usize),
    /// The source and reference positions are too close to define a directivity axis.
    #[error("source and reference positions must differ: {0}")]
    SourceReferenceTooClose(String),
    /// A curve's frequency grid has a different length than the first curve.
    #[error(
        "curve {index} frequency grid length ({got}) does not match expected grid length ({expected})"
    )]
    MismatchedFreqLength {
        index: usize,
        got: usize,
        expected: usize,
    },
    /// A curve's SPL array has a different length than the first curve's frequency grid.
    #[error("curve {index} SPL length ({got}) does not match expected grid length ({expected})")]
    MismatchedSplLength {
        index: usize,
        got: usize,
        expected: usize,
    },
    /// A curve's frequency grid differs from the first curve's grid at a specific bin.
    #[error(
        "curve {index} frequency value at bin {bin} differs from reference: {got} vs {expected}"
    )]
    MismatchedFreqValue {
        index: usize,
        bin: usize,
        got: f64,
        expected: f64,
    },
    /// The Gaussian distance weight mode was given a non-positive sigma.
    #[error("Gaussian distance weight requires positive sigma_m, got {0}")]
    InvalidGaussianSigma(f64),
    /// The spherical-head directivity model was given a non-positive radius.
    #[error("SphericalHead directivity requires positive radius_m, got {0}")]
    InvalidSphericalHeadRadius(f64),
}

/// Result of building a weighted RIR prototype.
#[derive(Debug, Clone)]
pub struct WeightedPrototype {
    pub curve: Curve,
    pub weights: ndarray::Array2<f64>,
}

/// Build a single prototype curve from multiple measurement curves and positions.
///
/// All curves must share the same frequency grid (same length and same values
/// within [`FREQ_GRID_TOLERANCE`]). Callers typically use
/// `read::load_source_individual` first, which interpolates all curves to the
/// first curve's grid.
///
/// The returned `WeightedPrototype.curve` is a clone of the first input curve
/// with only `freq` and `spl` overwritten. All other metadata — including
/// `coherence`, `noise_floor_db`, `phase`, and the cached phase-decomposition
/// fields — are preserved.
pub fn build_weighted_prototype(
    curves: &[Curve],
    config: &RirPrototypeConfig,
) -> Result<WeightedPrototype, RirPrototypeError> {
    if curves.is_empty() {
        return Err(RirPrototypeError::NoCurves);
    }
    if config.microphone_positions.len() != curves.len() {
        return Err(RirPrototypeError::MismatchedCounts(
            config.microphone_positions.len(),
            curves.len(),
        ));
    }

    let expected_len = curves[0].freq.len();
    let reference_freq = &curves[0].freq;
    for (i, curve) in curves.iter().enumerate() {
        if curve.freq.len() != expected_len {
            return Err(RirPrototypeError::MismatchedFreqLength {
                index: i,
                got: curve.freq.len(),
                expected: expected_len,
            });
        }
        if curve.spl.len() != expected_len {
            return Err(RirPrototypeError::MismatchedSplLength {
                index: i,
                got: curve.spl.len(),
                expected: expected_len,
            });
        }
        for (bin, (&got, &expected)) in curve.freq.iter().zip(reference_freq.iter()).enumerate() {
            // Reject a grid value only if it fails BOTH the absolute and the
            // relative tolerance checks; either check passing means the bin is
            // close enough for prototype averaging.
            if (got - expected).abs() > FREQ_GRID_TOLERANCE
                && ((got - expected).abs() / expected.abs()) > FREQ_GRID_TOLERANCE
            {
                return Err(RirPrototypeError::MismatchedFreqValue {
                    index: i,
                    bin,
                    got,
                    expected,
                });
            }
        }
    }

    if let DistanceWeightMode::Gaussian { sigma_m } = config.distance_mode
        && sigma_m <= 0.0
    {
        return Err(RirPrototypeError::InvalidGaussianSigma(sigma_m));
    }

    if let DirectivityModel::SphericalHead { radius_m } = config.directivity
        && radius_m <= 0.0
    {
        return Err(RirPrototypeError::InvalidSphericalHeadRadius(radius_m));
    }

    let distances = compute_distances(&config.reference_position, &config.microphone_positions);
    let angles = compute_angles(
        &config.source_position,
        &config.reference_position,
        &config.microphone_positions,
    )
    .map_err(|e| RirPrototypeError::SourceReferenceTooClose(e.to_string()))?;

    let freqs = curves[0].freq.clone();
    let weights = normalized_weights(&distances, &angles, &freqs, config);

    let mut power_sum = Array1::<f64>::zeros(freqs.len());
    for (i, curve) in curves.iter().enumerate() {
        let row = weights.row(i);
        let p = curve.spl.mapv(|spl| 10.0_f64.powf(spl / 10.0));
        for j in 0..freqs.len() {
            power_sum[j] += row[j] * p[j];
        }
    }

    let avg_spl = power_sum.mapv(|p| 10.0 * p.max(1e-12).log10());

    // Preserve all metadata (coherence, noise_floor_db, phase, delay cache, etc.)
    // from the first input curve; only the frequency grid and SPL are recomputed.
    let mut prototype = curves[0].clone();
    prototype.freq = freqs;
    prototype.spl = avg_spl;

    Ok(WeightedPrototype {
        curve: prototype,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomeq::rir_prototype::config::{
        DirectivityModel, DistanceWeightMode, RirPrototypeConfig,
    };
    use ndarray::Array1;

    fn flat_curve(spl: f64) -> Curve {
        Curve {
            freq: Array1::from_vec(vec![100.0, 1000.0, 10000.0]),
            spl: Array1::from_vec(vec![spl, spl, spl]),
            ..Default::default()
        }
    }

    #[test]
    fn prototype_prefers_reference_mic() {
        let config = RirPrototypeConfig {
            reference_position: [0.0, 0.0, 0.0],
            source_position: [0.0, 2.0, 0.0],
            microphone_positions: vec![[0.0, 0.0, 0.0], [0.4, 0.1, 0.0], [-0.4, 0.1, 0.0]],
            distance_mode: DistanceWeightMode::InverseSquare,
            directivity: DirectivityModel::Omnidirectional,
            frequency_dependent_directivity: false,
        };
        let reference = flat_curve(75.0);
        let far = flat_curve(80.0);
        let off_axis = flat_curve(70.0);
        let prototype = build_weighted_prototype(&[reference, far, off_axis], &config).unwrap();
        // Reference mic at distance 0 dominates; prototype should be close to 75 dB.
        for &spl in prototype.curve.spl.iter() {
            assert!((spl - 75.0).abs() < 2.5, "got {}", spl);
        }
    }

    #[test]
    fn prototype_rejects_mismatched_counts() {
        let config = RirPrototypeConfig {
            reference_position: [0.0, 0.0, 0.0],
            source_position: [0.0, 2.0, 0.0],
            microphone_positions: vec![[0.0, 0.0, 0.0]],
            distance_mode: DistanceWeightMode::Uniform,
            directivity: DirectivityModel::Omnidirectional,
            frequency_dependent_directivity: false,
        };
        let c1 = flat_curve(0.0);
        let c2 = flat_curve(0.0);
        assert!(build_weighted_prototype(&[c1, c2], &config).is_err());
    }

    #[test]
    fn uniform_weights_reproduce_plain_average() {
        let config = RirPrototypeConfig {
            reference_position: [0.0, 0.0, 0.0],
            source_position: [0.0, 2.0, 0.0],
            microphone_positions: vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
            distance_mode: DistanceWeightMode::Uniform,
            directivity: DirectivityModel::Omnidirectional,
            frequency_dependent_directivity: false,
        };
        let c1 = flat_curve(80.0);
        let c2 = flat_curve(86.0);
        let prototype = build_weighted_prototype(&[c1, c2], &config).unwrap();
        // 6 dB difference => power average ~81.76 dB.
        let expected =
            10.0 * ((10.0_f64.powf(80.0 / 10.0) + 10.0_f64.powf(86.0 / 10.0)) / 2.0).log10();
        assert!((prototype.curve.spl[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn prototype_rejects_mismatched_grid_lengths() {
        let config = RirPrototypeConfig {
            reference_position: [0.0, 0.0, 0.0],
            source_position: [0.0, 2.0, 0.0],
            microphone_positions: vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
            distance_mode: DistanceWeightMode::Uniform,
            directivity: DirectivityModel::Omnidirectional,
            frequency_dependent_directivity: false,
        };
        let c1 = flat_curve(80.0);
        let mut c2 = flat_curve(86.0);
        c2.freq = Array1::from_vec(vec![100.0, 1000.0]);
        c2.spl = Array1::from_vec(vec![86.0, 86.0]);
        assert!(build_weighted_prototype(&[c1, c2], &config).is_err());
    }

    #[test]
    fn prototype_rejects_mismatched_grid_values() {
        let config = RirPrototypeConfig {
            reference_position: [0.0, 0.0, 0.0],
            source_position: [0.0, 2.0, 0.0],
            microphone_positions: vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
            distance_mode: DistanceWeightMode::Uniform,
            directivity: DirectivityModel::Omnidirectional,
            frequency_dependent_directivity: false,
        };
        let c1 = flat_curve(80.0);
        let mut c2 = flat_curve(86.0);
        // Same length as c1, but shifted by more than the tolerance.
        c2.freq = Array1::from_vec(vec![100.01, 1000.0, 10000.0]);
        let err = build_weighted_prototype(&[c1, c2], &config).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("frequency value at bin 0 differs"),
            "expected mismatched frequency value error, got: {}",
            msg
        );
    }

    #[test]
    fn prototype_accepts_equal_grid_values() {
        let config = RirPrototypeConfig {
            reference_position: [0.0, 0.0, 0.0],
            source_position: [0.0, 2.0, 0.0],
            microphone_positions: vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
            distance_mode: DistanceWeightMode::Uniform,
            directivity: DirectivityModel::Omnidirectional,
            frequency_dependent_directivity: false,
        };
        let c1 = flat_curve(80.0);
        let mut c2 = flat_curve(86.0);
        // Tiny shift within tolerance.
        c2.freq = Array1::from_vec(vec![100.0 + 1e-7, 1000.0, 10000.0]);
        assert!(build_weighted_prototype(&[c1, c2], &config).is_ok());
    }

    #[test]
    fn prototype_rejects_invalid_gaussian_sigma() {
        let config = RirPrototypeConfig {
            reference_position: [0.0, 0.0, 0.0],
            source_position: [0.0, 2.0, 0.0],
            microphone_positions: vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
            distance_mode: DistanceWeightMode::Gaussian { sigma_m: 0.0 },
            directivity: DirectivityModel::Omnidirectional,
            frequency_dependent_directivity: false,
        };
        let c1 = flat_curve(80.0);
        let c2 = flat_curve(86.0);
        assert!(build_weighted_prototype(&[c1, c2], &config).is_err());
    }

    #[test]
    fn prototype_rejects_invalid_spherical_head_radius() {
        let config = RirPrototypeConfig {
            reference_position: [0.0, 0.0, 0.0],
            source_position: [0.0, 2.0, 0.0],
            microphone_positions: vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
            distance_mode: DistanceWeightMode::Uniform,
            directivity: DirectivityModel::SphericalHead { radius_m: -0.1 },
            frequency_dependent_directivity: false,
        };
        let c1 = flat_curve(80.0);
        let c2 = flat_curve(86.0);
        assert!(build_weighted_prototype(&[c1, c2], &config).is_err());
    }

    #[test]
    fn prototype_frequency_dependent_spherical_head_runs() {
        let config = RirPrototypeConfig {
            reference_position: [0.0, 0.0, 0.0],
            source_position: [0.0, 2.0, 0.0],
            microphone_positions: vec![[0.0, 0.0, 0.0], [0.3, 0.1, 0.0]],
            distance_mode: DistanceWeightMode::InverseSquare,
            directivity: DirectivityModel::SphericalHead { radius_m: 0.0875 },
            frequency_dependent_directivity: true,
        };
        let c1 = flat_curve(80.0);
        let c2 = flat_curve(86.0);
        let prototype = build_weighted_prototype(&[c1, c2], &config).unwrap();
        assert_eq!(prototype.curve.freq.len(), 3);
    }
}
