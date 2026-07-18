//! Scalar response metrics shared by RoomEQ preparation and reporting.

use math_audio_dsp::analysis::compute_average_response;

use crate::Curve;

/// Log-frequency-weighted mean response inside an optional frequency band.
pub fn mean_response_in_range(curve: &Curve, min_freq: f64, max_freq: f64) -> f64 {
    let frequencies: Vec<f32> = curve
        .freq
        .iter()
        .map(|&frequency| frequency as f32)
        .collect();
    let levels: Vec<f32> = curve.spl.iter().map(|&level| level as f32).collect();
    compute_average_response(
        &frequencies,
        &levels,
        Some((min_freq as f32, max_freq as f32)),
    ) as f64
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;

    #[test]
    fn range_mean_uses_only_the_requested_band() {
        let curve = Curve {
            freq: Array1::from_vec(vec![20.0, 100.0, 1_000.0, 10_000.0]),
            spl: Array1::from_vec(vec![40.0, 80.0, 80.0, 20.0]),
            ..Curve::default()
        };

        assert!((mean_response_in_range(&curve, 100.0, 1_000.0) - 80.0).abs() < 1e-6);
    }
}
