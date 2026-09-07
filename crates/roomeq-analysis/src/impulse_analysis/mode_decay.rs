use super::types::RoomMode;
use math_audio_iir_fir::{Biquad, BiquadFilterType};

/// Noise-aware, band-limited decay estimate for one detected mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeDecayEstimate {
    pub frequency_hz: f64,
    pub rt60_seconds: f64,
    pub rt60_lower_seconds: f64,
    pub rt60_upper_seconds: f64,
    /// Regression confidence in `[0, 1]`, combining fit R² and usable range.
    pub confidence: f64,
}

/// Minimum confidence required before measured decay replaces magnitude-Q fallback.
pub const MIN_MODE_DECAY_CONFIDENCE: f64 = 0.5;

/// Convert a confident measured decay estimate into temporal severity.
///
/// `None` preserves the explicitly labelled magnitude-Q fallback.
pub fn measured_temporal_severity_db(
    mode_frequency_hz: f64,
    estimate: &ModeDecayEstimate,
    use_music_threshold: bool,
) -> Option<f64> {
    if estimate.confidence < MIN_MODE_DECAY_CONFIDENCE {
        return None;
    }
    let threshold =
        crate::temporal_targets::max_acceptable_decay_time(mode_frequency_hz, use_music_threshold);
    Some(if estimate.rt60_seconds > threshold {
        20.0 * (estimate.rt60_seconds / threshold).log10()
    } else {
        0.0
    })
}

/// Estimate mode-specific decay from an impulse response.
///
/// Each mode is band-pass filtered around its measured center/Q. A late-tail
/// noise estimate is removed from the reverse-integrated energy curve before
/// fitting the -5 to -35 dB decay region. Missing/low-dynamic-range fits return
/// `None`; callers should then retain the magnitude-Q fallback.
pub fn estimate_mode_decays(
    modes: &[RoomMode],
    impulse: &[f32],
    sample_rate: f64,
) -> Vec<Option<ModeDecayEstimate>> {
    modes
        .iter()
        .map(|mode| estimate_mode_decay(mode, impulse, sample_rate))
        .collect()
}

/// Relative agreement required between the nominal and wide-band fits.
///
/// A genuine room decay is a property of the impulse, so both bandwidths fit
/// the same tail. An analysis-filter ring scales with Q instead: halving Q
/// halves the reported RT60 (relative disagreement ~0.5).
const BANDWIDTH_CONSISTENCY_TOLERANCE: f64 = 0.35;
/// Confidence ceiling for filter-limited estimates, below
/// [`MIN_MODE_DECAY_CONFIDENCE`] so callers retain the magnitude-Q fallback.
const FILTER_LIMITED_CONFIDENCE: f64 = 0.25;

fn estimate_mode_decay(
    mode: &RoomMode,
    impulse: &[f32],
    sample_rate: f64,
) -> Option<ModeDecayEstimate> {
    if impulse.len() < 256
        || !sample_rate.is_finite()
        || sample_rate <= 0.0
        || !mode.frequency.is_finite()
        || mode.frequency <= 0.0
        || mode.frequency >= 0.45 * sample_rate
    {
        return None;
    }

    let q = mode.q.clamp(1.0, 20.0);
    let narrow = fit_mode_decay(mode, impulse, sample_rate, q)?;
    // F07: the bandpass itself rings with T60 ~ ln(1000) Q / (pi f). A room
    // decay well below that floor is indistinguishable from the filter's own
    // decay, so changing analysis Q manufactures a "confident" room decay
    // that tracks Q. Re-fit one octave wider: agreement means the tail is the
    // room's; disagreement (or no wide fit) means filter-limited.
    let wide_q = (q / 2.0).max(1.0);
    if (wide_q - q).abs() < f64::EPSILON {
        return Some(narrow);
    }
    let consistent = fit_mode_decay(mode, impulse, sample_rate, wide_q).is_some_and(|wide| {
        wide.rt60_seconds.is_finite()
            && narrow.rt60_seconds.is_finite()
            && (narrow.rt60_seconds - wide.rt60_seconds).abs()
                / narrow.rt60_seconds.max(wide.rt60_seconds)
                <= BANDWIDTH_CONSISTENCY_TOLERANCE
    });
    if consistent {
        Some(narrow)
    } else {
        Some(ModeDecayEstimate {
            confidence: narrow.confidence.min(FILTER_LIMITED_CONFIDENCE),
            ..narrow
        })
    }
}

fn fit_mode_decay(
    mode: &RoomMode,
    impulse: &[f32],
    sample_rate: f64,
    q: f64,
) -> Option<ModeDecayEstimate> {
    let mut filtered: Vec<f64> = impulse.iter().map(|sample| *sample as f64).collect();
    let mut bandpass = Biquad::new(
        BiquadFilterType::Bandpass,
        mode.frequency,
        sample_rate,
        q,
        0.0,
    );
    bandpass.process_block(&mut filtered);

    let peak_index = filtered
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))?
        .0;
    let decay = &filtered[peak_index..];
    if decay.len() < 128 {
        return None;
    }

    let tail_count = (decay.len() / 10).max(64).min(decay.len());
    let noise_power = decay[decay.len() - tail_count..]
        .iter()
        .map(|sample| sample * sample)
        .sum::<f64>()
        / tail_count as f64;

    let mut integrated = vec![0.0; decay.len()];
    let mut accumulated = 0.0;
    for index in (0..decay.len()).rev() {
        accumulated += decay[index] * decay[index];
        let remaining = decay.len() - index;
        integrated[index] = (accumulated - noise_power * remaining as f64).max(0.0);
    }
    let reference = integrated[0];
    if !reference.is_finite() || reference <= 0.0 {
        return None;
    }

    let points: Vec<(f64, f64)> = integrated
        .iter()
        .enumerate()
        .filter_map(|(index, energy)| {
            if *energy <= 0.0 {
                return None;
            }
            let level_db = 10.0 * (energy / reference).log10();
            (-35.0..=-5.0)
                .contains(&level_db)
                .then_some((index as f64 / sample_rate, level_db))
        })
        .collect();
    if points.len() < 16 {
        return None;
    }

    let count = points.len() as f64;
    let mean_time = points.iter().map(|point| point.0).sum::<f64>() / count;
    let mean_level = points.iter().map(|point| point.1).sum::<f64>() / count;
    let time_variance = points
        .iter()
        .map(|point| (point.0 - mean_time).powi(2))
        .sum::<f64>();
    if time_variance <= f64::EPSILON {
        return None;
    }
    let slope = points
        .iter()
        .map(|point| (point.0 - mean_time) * (point.1 - mean_level))
        .sum::<f64>()
        / time_variance;
    if !slope.is_finite() || slope >= -1.0 {
        return None;
    }
    let intercept = mean_level - slope * mean_time;
    let residual_sum = points
        .iter()
        .map(|point| (point.1 - (intercept + slope * point.0)).powi(2))
        .sum::<f64>();
    let total_sum = points
        .iter()
        .map(|point| (point.1 - mean_level).powi(2))
        .sum::<f64>();
    let r_squared = if total_sum > 0.0 {
        (1.0 - residual_sum / total_sum).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let slope_error = if points.len() > 2 {
        ((residual_sum / (count - 2.0)) / time_variance).sqrt()
    } else {
        f64::INFINITY
    };
    let rt60_seconds = -60.0 / slope;
    if !rt60_seconds.is_finite() || !(0.01..=10.0).contains(&rt60_seconds) {
        return None;
    }

    let steep_slope = slope - 1.96 * slope_error;
    let shallow_slope = slope + 1.96 * slope_error;
    let lower = if steep_slope < 0.0 {
        -60.0 / steep_slope
    } else {
        rt60_seconds
    };
    let upper = if shallow_slope < -f64::EPSILON {
        -60.0 / shallow_slope
    } else {
        10.0
    };
    let covered_range = (points.first()?.1 - points.last()?.1).abs();
    let confidence = (r_squared * (covered_range / 30.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);

    Some(ModeDecayEstimate {
        frequency_hz: mode.frequency,
        rt60_seconds,
        rt60_lower_seconds: lower.min(rt60_seconds),
        rt60_upper_seconds: upper.max(rt60_seconds),
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode() -> RoomMode {
        RoomMode {
            frequency: 80.0,
            q: 8.0,
            temporal_severity_db: 0.0,
            prominence_db: 10.0,
            index: 0,
        }
    }

    fn decaying_mode(rt60_seconds: f64) -> Vec<f32> {
        let sample_rate = 48_000.0;
        let decay = 3.0 * std::f64::consts::LN_10 / rt60_seconds;
        (0..96_000)
            .map(|index| {
                let time = index as f64 / sample_rate;
                ((-decay * time).exp() * (2.0 * std::f64::consts::PI * 80.0 * time).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn delta_impulse_reports_no_confident_room_decay() {
        // F07: a delta contains no room tail. The bandpass ring alone must
        // not become a confident ~0.55 s room decay at any analysis Q.
        let sample_rate = 48_000.0;
        let mut impulse = vec![0.0_f32; 240_000];
        impulse[0] = 1.0;
        for q in [2.0, 10.0, 20.0] {
            let mode = RoomMode {
                frequency: 80.0,
                q,
                temporal_severity_db: 0.0,
                prominence_db: 8.0,
                index: 0,
            };
            match estimate_mode_decays(&[mode], &impulse, sample_rate)[0] {
                None => {}
                Some(estimate) => assert!(
                    estimate.confidence < MIN_MODE_DECAY_CONFIDENCE,
                    "delta impulse produced a confident room decay at Q={q}: \
                     RT60={:.6} s confidence={:.5}",
                    estimate.rt60_seconds,
                    estimate.confidence
                ),
            }
        }
    }

    #[test]
    fn true_decay_agrees_across_analysis_bandwidth() {
        // A genuine room decay well above both analysis-filter floors must
        // stay confident and consistent when the analysis Q changes.
        let sample_rate = 48_000.0;
        let impulse = decaying_mode(0.9);
        let mut estimates = Vec::new();
        for q in [8.0, 20.0] {
            let mode = RoomMode {
                frequency: 80.0,
                q,
                temporal_severity_db: 0.0,
                prominence_db: 10.0,
                index: 0,
            };
            let estimate = estimate_mode_decays(&[mode], &impulse, sample_rate)[0]
                .expect("true 0.9 s decay should fit");
            assert!(
                estimate.confidence >= MIN_MODE_DECAY_CONFIDENCE,
                "true decay lost confidence at Q={q}: {}",
                estimate.confidence
            );
            estimates.push(estimate.rt60_seconds);
        }
        let (a, b) = (estimates[0], estimates[1]);
        assert!(
            (a - b).abs() / a.max(b) < 0.35,
            "true decay disagrees across bandwidth: {a:.3} s vs {b:.3} s"
        );
    }

    #[test]
    fn same_mode_frequency_with_different_decay_is_distinguished() {
        let short = estimate_mode_decays(&[mode()], &decaying_mode(0.25), 48_000.0)[0].unwrap();
        let long = estimate_mode_decays(&[mode()], &decaying_mode(0.9), 48_000.0)[0].unwrap();
        assert!(long.rt60_seconds > short.rt60_seconds * 2.0);
        let short_severity = measured_temporal_severity_db(80.0, &short, false).unwrap();
        let long_severity = measured_temporal_severity_db(80.0, &long, false).unwrap();
        assert_eq!(short_severity, 0.0);
        assert!(long_severity > short_severity);
        assert!(short.confidence > 0.7);
        assert!(long.confidence > 0.7);
        assert!(short.rt60_lower_seconds <= short.rt60_seconds);
        assert!(short.rt60_upper_seconds >= short.rt60_seconds);
    }
}
