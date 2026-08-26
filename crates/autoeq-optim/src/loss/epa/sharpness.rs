use super::loudness::total_loudness;

/// Sharpness weighting function `g(z)` sampled at the 24 integer Bark-band
/// centres, following DIN 45692:
///
/// `g(z) = 1` for `z <= 15.8 Bark`, otherwise
/// `g(z) = 0.15 * exp(0.42 * (z - 15.8)) + 0.85`.
///
/// The values are written out so this remains a public constant and so the
/// standard's numerical samples can be regression-tested directly.
pub const SHARPNESS_WEIGHT: [f64; 24] = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.013_144_334_071_324,
    1.098_299_404_473_558,
    1.227_902_147_891_757,
    1.425_152_540_876_961,
    1.725_360_055_831_110,
    2.182_264_352_299_130,
    2.877_655_126_120_436,
    3.936_013_150_008_355,
    5.546_793_374_446_277,
];

/// Compute Zwicker sharpness (in acum) from specific loudness values.
///
/// S = 0.11 * sum(N'(z) * g(z) * z) / max(N_total, 0.001)
///
/// where z is the band index (1-based), g(z) is the sharpness weight,
/// N'(z) is specific loudness, and 0.11 is the calibration constant
/// (1 acum = narrowband noise at 1 kHz, 60 dB).
pub fn sharpness(specific_loudness: &[f64; 24]) -> f64 {
    let mut numerator = 0.0_f64;

    for z in 0..24 {
        let z_center = (z + 1) as f64; // 1-based band index
        numerator += specific_loudness[z] * SHARPNESS_WEIGHT[z] * z_center;
    }

    let denominator = total_loudness(specific_loudness).max(0.001);
    0.11 * numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::epa::loudness::specific_loudness;

    fn make_bandlimited_response(lo_hz: f64, hi_hz: f64, level_db: f64) -> (Vec<f64>, Vec<f64>) {
        let n = 1000;
        let freqs: Vec<f64> = (0..n)
            .map(|i| 20.0 + (16000.0 - 20.0) * i as f64 / n as f64)
            .collect();
        let spl: Vec<f64> = freqs
            .iter()
            .map(|&f| {
                if f >= lo_hz && f <= hi_hz {
                    level_db
                } else {
                    -40.0
                }
            })
            .collect();
        (freqs, spl)
    }

    #[test]
    fn test_sharpness_low_freq_only() {
        let (freqs, spl) = make_bandlimited_response(20.0, 1000.0, 70.0);
        let spec = specific_loudness(&freqs, &spl, 70.0);
        let s = sharpness(&spec);
        assert!(
            s < 1.0,
            "Low-frequency-only signal should have low sharpness, got {s} acum"
        );
    }

    #[test]
    fn test_sharpness_high_freq_only() {
        let (freqs, spl) = make_bandlimited_response(5000.0, 15000.0, 70.0);
        let spec = specific_loudness(&freqs, &spl, 70.0);
        let s = sharpness(&spec);
        assert!(
            s > 2.0,
            "High-frequency-only signal should have high sharpness, got {s} acum"
        );
    }

    #[test]
    fn test_sharpness_broadband() {
        let (freqs, spl) = make_bandlimited_response(20.0, 15000.0, 70.0);
        let spec = specific_loudness(&freqs, &spl, 70.0);
        let s = sharpness(&spec);
        assert!(
            (0.5..=2.5).contains(&s),
            "Broadband signal should have moderate sharpness, got {s} acum"
        );
    }

    #[test]
    fn one_khz_narrowband_reference_is_one_acum() {
        let mut specific = [0.0; 24];
        specific[8] = 1.0; // Ninth Bark band is centered at 1 kHz.
        let value = sharpness(&specific);

        assert!(
            (value - 1.0).abs() <= 0.05,
            "1 kHz narrowband reference should be 1 acum, got {value}"
        );
    }

    #[test]
    fn din_45692_integer_bark_weight_samples_match_reference_formula() {
        for (index, &actual) in SHARPNESS_WEIGHT.iter().enumerate() {
            let z_bark = (index + 1) as f64;
            let expected = if z_bark <= 15.8 {
                1.0
            } else {
                0.15 * f64::exp(0.42 * (z_bark - 15.8)) + 0.85
            };
            assert!(
                (actual - expected).abs() <= 1e-14,
                "DIN 45692 g({z_bark}) expected {expected}, got {actual}"
            );
        }

        assert!((SHARPNESS_WEIGHT[15] - 1.013_144_334_071_324).abs() <= 1e-14);
        assert!((SHARPNESS_WEIGHT[19] - 1.725_360_055_831_110).abs() <= 1e-14);
        assert!((SHARPNESS_WEIGHT[23] - 5.546_793_374_446_277).abs() <= 1e-14);
    }

    #[test]
    fn top_bark_band_uses_din_exponential_weight() {
        let mut specific = [0.0; 24];
        specific[23] = 1.0;

        let value = sharpness(&specific);
        let expected = 0.11 * 24.0 * 5.546_793_374_446_277;
        assert!(
            (value - expected).abs() <= 1e-12,
            "expected {expected}, got {value}"
        );
    }
}
