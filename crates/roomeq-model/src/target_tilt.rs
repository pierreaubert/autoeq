//! Target curve generation for room correction.
//!
//! Builds the single target curve consumed by both broadband pre-correction
//! and the EQ optimiser from a [`TargetResponseConfig`]: a base tilt (flat /
//! Harman / custom slope / file / derived-from-measurement) plus optional
//! user preference shelves (bass / treble).

use crate::home_cinema::{HomeCinemaRole, role_for_channel};
use crate::{Curve, TargetResponseConfig, TargetShape};
use ndarray::Array1;

/// Build a complete target curve from `TargetResponseConfig`.
///
/// Computes the target shape (flat/harman/custom slope) then layers user
/// preference adjustments (bass shelf, treble shelf) on top.
/// This is the single source of truth used by both broadband pre-correction
/// and the EQ optimizer.
pub fn build_complete_target_curve(freqs: &Array1<f64>, config: &TargetResponseConfig) -> Curve {
    let slope = match config.shape {
        TargetShape::Flat => 0.0,
        TargetShape::Harman => -0.8,
        TargetShape::Custom => config.slope_db_per_octave,
        TargetShape::File => {
            // Workflow-prepared file curves bypass this procedural builder.
            // Reaching this branch means the resource was not supplied.
            log::warn!(
                "build_complete_target_curve called with File shape but no curve provided; falling back to flat"
            );
            0.0
        }
        TargetShape::FromMeasurement => {
            // Should have been resolved to a concrete slope by the caller.
            log::warn!(
                "build_complete_target_curve called with FromMeasurement; falling back to flat"
            );
            0.0
        }
    };

    let ref_freq = config.reference_freq.max(1.0);
    let pref = &config.preference;

    let spl = Array1::from_shape_fn(freqs.len(), |i| {
        let f = freqs[i].max(1.0);

        // Base tilt: slope * log2(f / ref_freq)
        let tilt_db = slope * (f / ref_freq).log2();

        // Bass shelf preference (smooth 2nd-order transition)
        let bass_adj = if pref.bass_shelf_db.abs() > 0.001
            && pref.bass_shelf_freq > 0.0
            && f < pref.bass_shelf_freq * 2.0
        {
            let ratio = f / pref.bass_shelf_freq;
            let transition = 1.0 / (1.0 + ratio.powi(2));
            pref.bass_shelf_db * transition
        } else {
            0.0
        };

        // Treble shelf preference (smooth 2nd-order transition)
        // Uses 4th-order for steeper onset: ~90% at 2x shelf_freq, ~50% at shelf_freq
        let treble_adj = if pref.treble_shelf_db.abs() > 0.001
            && pref.treble_shelf_freq > 0.0
            && f > pref.treble_shelf_freq * 0.25
        {
            let ratio = f / pref.treble_shelf_freq;
            let transition = 1.0 / (1.0 + (1.0 / ratio).powi(4));
            pref.treble_shelf_db * transition
        } else {
            0.0
        };

        tilt_db + bass_adj + treble_adj
    });

    Curve {
        freq: freqs.clone(),
        spl,
        phase: None,
        ..Default::default()
    }
}

/// Apply enabled home-cinema role shaping to an already-built target curve.
pub fn apply_role_target_curve_shape(
    channel_name: &str,
    target_curve: &mut Curve,
    target: &TargetResponseConfig,
) {
    let Some(role_targets) = target.role_targets.as_ref().filter(|config| config.enabled) else {
        return;
    };
    let role = role_for_channel(channel_name);

    if role == HomeCinemaRole::Center && role_targets.center_dialog_boost_db.abs() > 0.001 {
        apply_log_band_emphasis(
            target_curve,
            role_targets.center_dialog_low_hz,
            role_targets.center_dialog_high_hz,
            role_targets.center_dialog_boost_db,
        );
    }

    if role_targets.cinema_x_curve_enabled
        && role_targets.cinema_x_curve_db_per_octave.abs() > 0.001
    {
        apply_high_frequency_slope(
            target_curve,
            role_targets.cinema_x_curve_start_hz,
            role_targets.cinema_x_curve_db_per_octave,
        );
    }

    if let Some(distance_m) = role_targets.listening_distance_m {
        let reference_m = role_targets.cinema_reference_distance_m;
        if distance_m > reference_m
            && reference_m > 0.0
            && role_targets.distance_treble_rolloff_db_per_doubling.abs() > 0.001
        {
            let distance_doublings = (distance_m / reference_m).log2();
            apply_high_frequency_slope(
                target_curve,
                role_targets.cinema_x_curve_start_hz,
                -role_targets.distance_treble_rolloff_db_per_doubling.abs() * distance_doublings,
            );
        }
    }
}

fn apply_log_band_emphasis(target_curve: &mut Curve, low_hz: f64, high_hz: f64, gain_db: f64) {
    if !(low_hz > 0.0 && high_hz > low_hz) {
        return;
    }
    let center_hz = (low_hz * high_hz).sqrt();
    let half_width_oct = (high_hz / low_hz).log2() / 2.0;
    if half_width_oct <= 0.0 {
        return;
    }

    for (freq, spl) in target_curve.freq.iter().zip(target_curve.spl.iter_mut()) {
        let distance_oct = (*freq / center_hz).max(1e-9).log2().abs();
        if distance_oct <= half_width_oct {
            let weight = 0.5 * (1.0 + (std::f64::consts::PI * distance_oct / half_width_oct).cos());
            *spl += gain_db * weight;
        }
    }
}

fn apply_high_frequency_slope(target_curve: &mut Curve, start_hz: f64, slope_db_per_octave: f64) {
    if start_hz <= 0.0 {
        return;
    }
    for (freq, spl) in target_curve.freq.iter().zip(target_curve.spl.iter_mut()) {
        if *freq > start_hz {
            *spl += slope_db_per_octave * (*freq / start_hz).log2();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frequencies() -> Array1<f64> {
        Array1::from(vec![
            20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0,
        ])
    }

    #[test]
    fn test_complete_target_flat() {
        let freqs = test_frequencies();
        let config = TargetResponseConfig::default(); // flat, no preference
        let curve = build_complete_target_curve(&freqs, &config);
        for &spl in curve.spl.iter() {
            assert!(
                (spl).abs() < 1e-10,
                "Flat target should be all zeros, got {}",
                spl
            );
        }
    }

    #[test]
    fn test_complete_target_harman() {
        let freqs = test_frequencies();
        let config = TargetResponseConfig {
            shape: TargetShape::Harman,
            ..Default::default()
        };
        let curve = build_complete_target_curve(&freqs, &config);

        // At 1 kHz reference → 0 dB
        let idx_1k = freqs
            .iter()
            .position(|&f| (f - 1000.0).abs() < 1.0)
            .unwrap();
        assert!((curve.spl[idx_1k]).abs() < 1e-10);

        // At 2 kHz → -0.8 dB
        let idx_2k = freqs
            .iter()
            .position(|&f| (f - 2000.0).abs() < 1.0)
            .unwrap();
        assert!((curve.spl[idx_2k] - (-0.8)).abs() < 1e-10);
    }

    #[test]
    fn test_complete_target_with_treble_shelf() {
        let freqs = test_frequencies();
        let config = TargetResponseConfig {
            shape: TargetShape::Flat,
            preference: crate::UserPreference {
                treble_shelf_db: -2.0,
                treble_shelf_freq: 8000.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let curve = build_complete_target_curve(&freqs, &config);

        // Well above treble shelf → close to -2 dB
        let idx_20k = freqs
            .iter()
            .position(|&f| (f - 20000.0).abs() < 1.0)
            .unwrap();
        assert!(
            curve.spl[idx_20k] < -1.5,
            "At 20kHz should have treble cut, got {:.2}",
            curve.spl[idx_20k]
        );

        // Well below treble shelf → near 0 dB
        let idx_1k = freqs
            .iter()
            .position(|&f| (f - 1000.0).abs() < 1.0)
            .unwrap();
        assert!(
            curve.spl[idx_1k].abs() < 0.1,
            "At 1kHz should be near 0, got {:.2}",
            curve.spl[idx_1k]
        );
    }

    #[test]
    fn test_complete_target_harman_plus_bass_boost() {
        let freqs = test_frequencies();
        let config = TargetResponseConfig {
            shape: TargetShape::Harman,
            preference: crate::UserPreference {
                bass_shelf_db: 3.0,
                bass_shelf_freq: 200.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let curve = build_complete_target_curve(&freqs, &config);

        // At 20 Hz: Harman tilt (+4.5 dB) + bass boost (~3 dB) > 5 dB
        assert!(
            curve.spl[0] > 5.0,
            "20Hz should have tilt + bass boost, got {:.2}",
            curve.spl[0]
        );

        // At 10 kHz: Harman tilt only (~-2.6 dB), no bass effect
        let idx_10k = freqs
            .iter()
            .position(|&f| (f - 10000.0).abs() < 1.0)
            .unwrap();
        assert!(
            curve.spl[idx_10k] < -2.0,
            "10kHz should be tilted down, got {:.2}",
            curve.spl[idx_10k]
        );
    }

    #[test]
    fn role_shape_applies_center_dialog_emphasis() {
        let frequencies = test_frequencies();
        let mut curve = build_complete_target_curve(&frequencies, &TargetResponseConfig::default());
        let target = TargetResponseConfig {
            role_targets: Some(crate::RoleTargetConfig {
                enabled: true,
                center_dialog_boost_db: 3.0,
                ..crate::RoleTargetConfig::default()
            }),
            ..TargetResponseConfig::default()
        };

        apply_role_target_curve_shape("Center", &mut curve, &target);

        let index = frequencies
            .iter()
            .position(|frequency| (*frequency - 2_000.0).abs() < 1.0)
            .unwrap();
        assert!(curve.spl[index] > 0.0);
    }
}
