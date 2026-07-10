//! Weighted RIR prototype builder.

use crate::Curve;
use crate::roomeq::rir_prototype::config::RirPrototypeConfig;
use crate::roomeq::rir_prototype::weights::{
    compute_angles, compute_distances, normalized_weights,
};
use ndarray::Array1;

/// Result of building a weighted RIR prototype.
#[derive(Debug, Clone)]
pub struct WeightedPrototype {
    pub curve: Curve,
    pub weights: ndarray::Array2<f64>,
}

/// Build a single prototype curve from multiple measurement curves and positions.
///
/// All curves must share the same frequency grid. Callers typically use
/// `read::load_source_individual` first, which interpolates all curves to the
/// first curve's grid.
pub fn build_weighted_prototype(
    curves: &[Curve],
    config: &RirPrototypeConfig,
) -> Result<WeightedPrototype, Box<dyn std::error::Error>> {
    if curves.is_empty() {
        return Err("build_weighted_prototype: no curves provided".into());
    }
    if config.microphone_positions.len() != curves.len() {
        return Err(format!(
            "microphone_positions length ({}) must match curves length ({})",
            config.microphone_positions.len(),
            curves.len()
        )
        .into());
    }

    let distances = compute_distances(&config.reference_position, &config.microphone_positions);
    let angles = compute_angles(
        &config.source_position,
        &config.reference_position,
        &config.microphone_positions,
    )?;

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
    let phase = curves[0].phase.clone();

    Ok(WeightedPrototype {
        curve: Curve {
            freq: freqs,
            spl: avg_spl,
            phase,
            ..Default::default()
        },
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
            microphone_positions: vec![
                [0.0, 0.0, 0.0],
                [0.4, 0.1, 0.0],
                [-0.4, 0.1, 0.0],
            ],
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
        let expected = 10.0 * ((10.0_f64.powf(80.0 / 10.0) + 10.0_f64.powf(86.0 / 10.0)) / 2.0).log10();
        assert!((prototype.curve.spl[0] - expected).abs() < 1e-6);
    }
}
