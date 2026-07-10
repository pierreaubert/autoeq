//! Distance and directivity weight calculations.

use super::config::{DirectivityModel, DistanceWeightMode};

/// Compute the Euclidean distance from the reference position to each microphone.
pub fn compute_distances(reference: [f64; 3], microphones: &[[f64; 3]]) -> Vec<f64> {
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

/// Compute the angle between the source-to-reference axis and each microphone.
pub fn compute_angles(
    _reference: [f64; 3],
    _source: [f64; 3],
    microphones: &[[f64; 3]],
) -> Vec<f64> {
    microphones.iter().map(|_| 0.0).collect()
}

/// Compute a per-microphone distance weight according to the chosen mode.
pub fn distance_weight(distances: &[f64], mode: DistanceWeightMode) -> Vec<f64> {
    match mode {
        DistanceWeightMode::InverseSquare => distances
            .iter()
            .map(|d| {
                let clamped = d.max(0.01);
                1.0 / (clamped * clamped)
            })
            .collect(),
        DistanceWeightMode::Gaussian { sigma_m } => distances
            .iter()
            .map(|d| (-d * d / (2.0 * sigma_m * sigma_m)).exp())
            .collect(),
        DistanceWeightMode::Uniform => distances.iter().map(|_| 1.0).collect(),
    }
}

/// Compute a per-microphone directivity weight for the given frequency.
pub fn directivity_weight(
    angles: &[f64],
    _frequency_hz: f64,
    model: DirectivityModel,
    _frequency_dependent: bool,
) -> Vec<f64> {
    match model {
        // TODO: implement proper head-shadow approximation for SphericalHead in Task 3.
        DirectivityModel::Omnidirectional | DirectivityModel::SphericalHead { .. } => {
            angles.iter().map(|_| 1.0).collect()
        }
    }
}

/// Normalize a set of weights so they sum to one.
pub fn normalized_weights(weights: &[f64]) -> Vec<f64> {
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        weights.iter().map(|w| w / sum).collect()
    } else {
        weights.to_vec()
    }
}
