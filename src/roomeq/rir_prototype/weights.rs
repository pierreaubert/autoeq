//! Distance and directivity weight calculations.

use crate::roomeq::rir_prototype::config::{
    DirectivityModel, DistanceWeightMode, RirPrototypeConfig,
};
use ndarray::{Array1, Array2};

const SOUND_SPEED_MPS: f64 = 343.0;
const MIN_DISTANCE_M: f64 = 1e-6;

/// Compute a scalar distance weight.
pub fn distance_weight(distance_m: f64, mode: DistanceWeightMode) -> f64 {
    match mode {
        DistanceWeightMode::InverseSquare => {
            let d = distance_m.max(MIN_DISTANCE_M);
            1.0 / (d * d)
        }
        DistanceWeightMode::Gaussian { sigma_m } => {
            if sigma_m <= 0.0 {
                return 1.0;
            }
            (-distance_m * distance_m / (2.0 * sigma_m * sigma_m)).exp()
        }
        DistanceWeightMode::Uniform => 1.0,
    }
}

/// Compute a scalar directivity weight for a single frequency and angle.
///
/// `angle_rad` is the angle between the source axis and the microphone direction,
/// as seen from the source. 0 = on-axis, π = directly behind the source.
pub fn directivity_weight(freq_hz: f64, angle_rad: f64, model: DirectivityModel) -> f64 {
    match model {
        DirectivityModel::Omnidirectional => 1.0,
        DirectivityModel::SphericalHead { radius_m } => {
            if radius_m <= 0.0 {
                return 1.0;
            }
            let ka = 2.0 * std::f64::consts::PI * freq_hz * radius_m / SOUND_SPEED_MPS;
            // Directionality grows from 0 (omnidirectional) toward 1 (dipole-like) as ka increases.
            let directionality = ka * ka / (1.0 + ka * ka);
            let cos_theta = angle_rad.cos();
            let off_axis = 0.5 * (1.0 + cos_theta);
            (1.0 - directionality) + off_axis * directionality
        }
    }
}

/// Compute Euclidean distance from each microphone to the reference position.
pub fn compute_distances(reference: &[f64; 3], microphones: &[[f64; 3]]) -> Vec<f64> {
    microphones
        .iter()
        .map(|m| {
            let dx = m[0] - reference[0];
            let dy = m[1] - reference[1];
            let dz = m[2] - reference[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect()
}

/// Compute the angle between source→reference and source→microphone for each mic.
pub fn compute_angles(
    source: &[f64; 3],
    reference: &[f64; 3],
    microphones: &[[f64; 3]],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut axis = [
        reference[0] - source[0],
        reference[1] - source[1],
        reference[2] - source[2],
    ];
    let axis_norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if axis_norm < 1e-9 {
        return Err("source and reference positions must differ".into());
    }
    for a in &mut axis {
        *a /= axis_norm;
    }

    let mut angles = Vec::with_capacity(microphones.len());
    for mic in microphones {
        let mut v = [mic[0] - source[0], mic[1] - source[1], mic[2] - source[2]];
        let v_norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if v_norm < 1e-9 {
            angles.push(0.0);
            continue;
        }
        for x in &mut v {
            *x /= v_norm;
        }
        let dot = (axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2]).clamp(-1.0, 1.0);
        angles.push(dot.acos());
    }
    Ok(angles)
}

/// Build a [measurement × frequency] weight matrix and normalize per frequency.
pub fn normalized_weights(
    distances: &[f64],
    angles_rad: &[f64],
    freq_hz: &Array1<f64>,
    config: &RirPrototypeConfig,
) -> Array2<f64> {
    let n = distances.len();
    let m = freq_hz.len();
    let mut weights = Array2::<f64>::zeros((n, m));

    for i in 0..n {
        let dw = distance_weight(distances[i], config.distance_mode);
        for (j, &f) in freq_hz.iter().enumerate() {
            let fw = if config.frequency_dependent_directivity {
                f
            } else {
                1000.0
            };
            let dir_w = directivity_weight(fw, angles_rad[i], config.directivity);
            weights[[i, j]] = dw * dir_w;
        }
    }

    for j in 0..m {
        let sum = weights.column(j).sum();
        if sum > 0.0 {
            for i in 0..n {
                weights[[i, j]] /= sum;
            }
        } else {
            for i in 0..n {
                weights[[i, j]] = 1.0 / n as f64;
            }
        }
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomeq::rir_prototype::config::{
        DirectivityModel, DistanceWeightMode, RirPrototypeConfig,
    };
    use ndarray::Array1;

    #[test]
    fn distance_weight_inverse_square_clips_near_zero() {
        let w0 = distance_weight(1e-9, DistanceWeightMode::InverseSquare);
        let w1 = distance_weight(1e-6, DistanceWeightMode::InverseSquare);
        assert!(w0.is_finite());
        assert_eq!(w0, w1); // both clipped to 1e-6
        let w2 = distance_weight(2.0, DistanceWeightMode::InverseSquare);
        assert!(w2 < w1);
    }

    #[test]
    fn distance_weight_gaussian_decreases_with_distance() {
        let near = distance_weight(0.1, DistanceWeightMode::Gaussian { sigma_m: 0.5 });
        let far = distance_weight(1.0, DistanceWeightMode::Gaussian { sigma_m: 0.5 });
        assert!(far < near);
        assert!(near <= 1.0);
    }

    #[test]
    fn directivity_weight_omnidirectional_is_one() {
        let w = directivity_weight(1000.0, 1.0, DirectivityModel::Omnidirectional);
        assert_eq!(w, 1.0);
    }

    #[test]
    fn directivity_weight_spherical_head_low_freq_omnidirectional() {
        let on_axis = directivity_weight(
            20.0,
            0.0,
            DirectivityModel::SphericalHead { radius_m: 0.0875 },
        );
        let off_axis = directivity_weight(
            20.0,
            std::f64::consts::PI,
            DirectivityModel::SphericalHead { radius_m: 0.0875 },
        );
        assert!((on_axis - 1.0).abs() < 1e-3);
        assert!((off_axis - 1.0).abs() < 2e-3);
    }

    #[test]
    fn directivity_weight_spherical_head_high_freq_off_axis_attenuates() {
        let on_axis = directivity_weight(
            10000.0,
            0.0,
            DirectivityModel::SphericalHead { radius_m: 0.0875 },
        );
        let off_axis = directivity_weight(
            10000.0,
            std::f64::consts::PI,
            DirectivityModel::SphericalHead { radius_m: 0.0875 },
        );
        assert!(on_axis > off_axis);
        assert!(on_axis <= 1.0);
    }

    #[test]
    fn normalized_weights_sum_to_one_per_frequency() {
        let config = RirPrototypeConfig {
            reference_position: [0.0, 0.0, 0.0],
            source_position: [0.0, 2.0, 0.0],
            microphone_positions: vec![[0.0, 0.0, 0.0], [0.3, 0.1, 0.0], [-0.3, 0.1, 0.0]],
            distance_mode: DistanceWeightMode::InverseSquare,
            directivity: DirectivityModel::Omnidirectional,
            frequency_dependent_directivity: false,
        };
        let distances = compute_distances(&config.reference_position, &config.microphone_positions);
        let angles = compute_angles(
            &config.source_position,
            &config.reference_position,
            &config.microphone_positions,
        )
        .unwrap();
        let freq = Array1::from_vec(vec![100.0, 1000.0, 10000.0]);
        let weights = normalized_weights(&distances, &angles, &freq, &config);
        for j in 0..freq.len() {
            let sum = weights.column(j).sum();
            assert!((sum - 1.0).abs() < 1e-9, "column {} sums to {}", j, sum);
        }
    }
}
