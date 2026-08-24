pub(super) fn interpolate_log_frequency(
    freqs: &[f64],
    values: &[f64],
    target_hz: f64,
) -> Option<f64> {
    if freqs.is_empty() || freqs.len() != values.len() || target_hz <= 0.0 {
        return None;
    }
    if target_hz <= freqs[0] {
        return Some(values[0]);
    }
    for i in 0..freqs.len().saturating_sub(1) {
        let f0 = freqs[i];
        let f1 = freqs[i + 1];
        if target_hz <= f1 {
            if f0 <= 0.0 || f1 <= 0.0 {
                return Some(values[i]);
            }
            let denom = f1.ln() - f0.ln();
            if denom.abs() < 1e-12 {
                return Some(values[i]);
            }
            let t = ((target_hz.ln() - f0.ln()) / denom).clamp(0.0, 1.0);
            return Some(values[i] + t * (values[i + 1] - values[i]));
        }
    }
    values.last().copied()
}

/// Shift a level-relative response to a configured diagnostic level reference.
///
/// Measurement curves in the autoeq/roomeq pipeline are typically
/// mean-subtracted around 1–2 kHz so they hover near 0 dB. The diagnostic
/// loudness formula in [`crate::loss::epa::loudness`] expects an SPL-like
/// reference, so this helper applies the configured presentation offset.
///
/// Adding `listening_level_phon` to a curve normalized at 1 kHz yields a
/// consistently shifted diagnostic curve. It does not provide microphone
/// calibration, programme spectrum, or the frequency-dependent phon-to-SPL
/// conversion required for a validated perceptual prediction, and these
/// descriptors do not enter `epa_loss`.
pub(super) fn denormalize_spl(spl_rel: &[f64], listening_level_phon: f64) -> Vec<f64> {
    spl_rel.iter().map(|v| v + listening_level_phon).collect()
}

pub(super) fn masking_weight(time_ms: f64, window_ms: f64) -> f64 {
    if window_ms <= 0.0 {
        1.0
    } else {
        (time_ms / window_ms).clamp(0.0, 1.0).powi(2)
    }
}
