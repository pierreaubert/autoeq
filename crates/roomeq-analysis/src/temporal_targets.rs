//! Perceptual thresholds for modal decay time.
//!
//! This module belongs to the reusable measurement-analysis layer.
//!
//! Based on psychometric studies showing frequency-dependent sensitivity
//! to room mode ringing. These thresholds define the maximum acceptable
//! decay time at each frequency — below threshold, ringing is inaudible;
//! above, it degrades perceived bass articulation.
//!
//! `impulse_analysis::detect_room_modes` evaluates these thresholds for each
//! detected modal peak and stores the resulting severity on `RoomMode` for
//! optimizer/reporting diagnostics.

/// Perceptual decay thresholds for artificial test stimuli.
/// Returns the maximum acceptable T60-like decay time in seconds.
const ARTIFICIAL_THRESHOLDS: &[(f64, f64)] = &[
    (32.0, 0.90),
    (50.0, 0.60),
    (63.0, 0.45),
    (100.0, 0.25),
    (200.0, 0.17),
    (250.0, 0.15),
];

/// Perceptual decay thresholds for music stimuli (slightly different due to temporal masking).
const MUSIC_THRESHOLDS: &[(f64, f64)] = &[
    (32.0, 0.85),
    (50.0, 0.70),
    (63.0, 0.51),
    (100.0, 0.30),
    (200.0, 0.15),
    (250.0, 0.12),
];

/// Validated domain of the threshold tables in Hz (F17).
///
/// The underlying studies measured 32–250 Hz. Outside this range the tables
/// are silent: endpoint clamping would invent limits the studies never
/// measured (a 20 Hz mode is not a 32 Hz mode; a 252 Hz mode is not a
/// 250 Hz mode).
pub const DECAY_THRESHOLD_MIN_HZ: f64 = 32.0;
/// Validated domain of the threshold tables in Hz (F17).
pub const DECAY_THRESHOLD_MAX_HZ: f64 = 250.0;

/// Whether a frequency lies inside the validated threshold domain.
pub fn decay_threshold_domain_contains(freq_hz: f64) -> bool {
    freq_hz.is_finite() && freq_hz >= DECAY_THRESHOLD_MIN_HZ && freq_hz <= DECAY_THRESHOLD_MAX_HZ
}

/// Get the maximum acceptable decay time at a given frequency, or `None`
/// outside the validated 32–250 Hz domain.
///
/// Prefer this over [`max_acceptable_decay_time`]: unknown audibility must
/// not become a numeric limit.
pub fn max_acceptable_decay_time_checked(
    freq_hz: f64,
    use_music_thresholds: bool,
) -> Option<f64> {
    decay_threshold_domain_contains(freq_hz)
        .then(|| max_acceptable_decay_time(freq_hz, use_music_thresholds))
}

/// Get the maximum acceptable decay time at a given frequency.
/// Interpolates between threshold points in log-frequency.
/// Below 32Hz: uses 32Hz value. Above 250Hz: uses 250Hz value.
///
/// Legacy endpoint-clamped lookup. Do not use this to judge modes outside
/// 32–250 Hz; use [`max_acceptable_decay_time_checked`] so out-of-domain
/// modes report unknown instead of inheriting an unmeasured limit.
pub fn max_acceptable_decay_time(freq_hz: f64, use_music_thresholds: bool) -> f64 {
    let table = if use_music_thresholds {
        MUSIC_THRESHOLDS
    } else {
        ARTIFICIAL_THRESHOLDS
    };

    let freq = freq_hz.max(table[0].0);
    let last = table.len() - 1;
    if freq >= table[last].0 {
        return table[last].1;
    }

    // Find the bracketing entries and interpolate in log-frequency
    for i in 0..last {
        let (f0, t0) = table[i];
        let (f1, t1) = table[i + 1];
        if freq >= f0 && freq <= f1 {
            let alpha = (freq.ln() - f0.ln()) / (f1.ln() - f0.ln());
            return t0 + alpha * (t1 - t0);
        }
    }

    // Shouldn't reach here, but return last value as fallback
    table[last].1
}

/// Compute the Q factor that would produce a given decay time at a frequency.
///
/// decay_time ≈ ln(1000) / (π * BW) where BW = freq / Q
/// Therefore Q = freq * decay_time * π / ln(1000)
pub fn q_for_decay_time(freq_hz: f64, decay_time_s: f64) -> f64 {
    freq_hz * decay_time_s * std::f64::consts::PI / 1000.0_f64.ln()
}

/// Compute the maximum acceptable Q for a room mode at a given frequency.
/// A mode with Q higher than this will have audible ringing.
pub fn max_acceptable_q(freq_hz: f64, use_music_thresholds: bool) -> f64 {
    let max_decay = max_acceptable_decay_time(freq_hz, use_music_thresholds);
    q_for_decay_time(freq_hz, max_decay)
}

/// Compute a "temporal severity" weight for a detected room mode.
/// Returns 0.0 if the mode's decay is below threshold (inaudible),
/// higher values for modes that exceed threshold more severely.
///
/// Heuristic magnitude-shaping weight only, never a reverberation-time
/// verdict: it judges an indirect magnitude Q against study tables, not a
/// measured decay. Outside the validated 32–250 Hz domain it returns 0.0
/// (unknown → no modal action), so unmeasured limits cannot trigger
/// modal treatment.
pub fn temporal_severity(mode_freq_hz: f64, mode_q: f64, use_music: bool) -> f64 {
    if !decay_threshold_domain_contains(mode_freq_hz) || !mode_q.is_finite() {
        return 0.0;
    }
    let max_q = max_acceptable_q(mode_freq_hz, use_music);
    if mode_q <= max_q {
        0.0
    } else {
        20.0 * (mode_q / max_q).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_threshold_32hz_tolerant() {
        let t = max_acceptable_decay_time(32.0, false);
        assert!((t - 0.90).abs() < 0.01, "32Hz should allow 0.90s decay");
    }

    #[test]
    fn test_decay_threshold_100hz_sensitive() {
        let t = max_acceptable_decay_time(100.0, false);
        assert!(
            (t - 0.25).abs() < 0.01,
            "100Hz should allow only 0.25s decay"
        );
    }

    #[test]
    fn test_decay_interpolation() {
        // 75Hz should interpolate between 63Hz (0.45s) and 100Hz (0.25s)
        let t = max_acceptable_decay_time(75.0, false);
        assert!(
            t > 0.25 && t < 0.45,
            "75Hz should be between 63Hz and 100Hz thresholds, got {}",
            t
        );
    }

    #[test]
    fn test_music_thresholds_different() {
        let art = max_acceptable_decay_time(63.0, false);
        let mus = max_acceptable_decay_time(63.0, true);
        assert!(
            art != mus,
            "Music and artificial thresholds should differ at 63Hz"
        );
    }

    #[test]
    fn test_q_for_decay() {
        let q = q_for_decay_time(100.0, 0.25);
        // Q = 100 * 0.25 * π / ln(1000) = 25π/6.9078 ≈ 11.4
        assert!((q - 11.4).abs() < 0.5, "Expected Q ~11.4, got {}", q);
    }

    #[test]
    fn test_max_acceptable_q_positive() {
        let q_32 = max_acceptable_q(32.0, false);
        let q_200 = max_acceptable_q(200.0, false);
        assert!(q_32 > 0.0 && q_200 > 0.0);
    }

    #[test]
    fn test_temporal_severity_below_threshold() {
        // A gentle mode (low Q) should have zero severity
        let severity = temporal_severity(100.0, 3.0, false);
        assert_eq!(severity, 0.0);
    }

    #[test]
    fn test_temporal_severity_above_threshold() {
        // A sharp mode (high Q) should have positive severity
        let severity = temporal_severity(100.0, 30.0, false);
        assert!(severity > 0.0, "High-Q mode at 100Hz should be severe");
    }

    #[test]
    fn test_below_32hz_clamps() {
        let t_20 = max_acceptable_decay_time(20.0, false);
        let t_32 = max_acceptable_decay_time(32.0, false);
        assert!(
            (t_20 - t_32).abs() < 1e-10,
            "Below 32Hz should clamp to 32Hz value"
        );
    }

    #[test]
    fn test_above_250hz_clamps() {
        let t_300 = max_acceptable_decay_time(300.0, false);
        let t_250 = max_acceptable_decay_time(250.0, false);
        assert!(
            (t_300 - t_250).abs() < 1e-10,
            "Above 250Hz should clamp to 250Hz value"
        );
    }

    #[test]
    fn out_of_domain_modes_report_unknown_not_a_limit() {
        // F17: the studies measured 32–250 Hz. Outside, the checked lookup
        // is silent and severity stays at zero (no modal action on unknown
        // audibility) instead of inheriting an unmeasured endpoint limit.
        assert!(max_acceptable_decay_time_checked(20.0, false).is_none());
        assert!(max_acceptable_decay_time_checked(252.0, false).is_none());
        assert!(max_acceptable_decay_time_checked(100.0, false).is_some());
        assert_eq!(temporal_severity(20.0, 30.0, false), 0.0);
        assert_eq!(temporal_severity(300.0, 30.0, false), 0.0);
        // In-domain behavior is unchanged: sharp modes stay severe.
        assert!(temporal_severity(100.0, 30.0, false) > 0.0);
    }
}
