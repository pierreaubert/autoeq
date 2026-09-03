use super::consts::MIN_CORRECTION_DB;
use super::grid::align_curves_to_common_grid;
use super::misc::estimate_correction_q;
use super::misc::smooth_for_peak_finding;
use super::types::ChannelMatchingResult;
use log::warn;
use math_audio_iir_fir::Biquad;
use std::collections::HashMap;

/// Role-specific channel matching correction limits.
#[derive(Debug, Clone, Copy)]
pub struct ChannelMatchingCorrectionProfile {
    /// Deviation left uncorrected so matching does not over-tighten this role.
    pub peak_tolerance_db: f64,
    /// Fraction of the deviation beyond tolerance to correct.
    pub correction_weight: f64,
    /// Lower matching band edge; combined with measured F3 at call time.
    pub min_freq_hz: f64,
    /// Upper matching band edge for this role.
    pub max_freq_hz: f64,
}

impl Default for ChannelMatchingCorrectionProfile {
    fn default() -> Self {
        Self {
            peak_tolerance_db: 1.0,
            correction_weight: 1.0,
            min_freq_hz: 0.0,
            max_freq_hz: 10_000.0,
        }
    }
}

impl ChannelMatchingCorrectionProfile {
    pub fn matching_band(self, f3_hz: f64) -> (f64, f64) {
        let min_freq = f3_hz.max(self.min_freq_hz);
        let max_freq = self.max_freq_hz.max(min_freq);
        (min_freq, max_freq)
    }

    /// Clamp fields into well-defined ranges. Negative tolerances/weights
    /// would invert the correction sign, NaN values would propagate through
    /// every gain computation, and swapped min/max would silently disable
    /// the correction band — so we normalise once at the entry point.
    pub fn sanitized(self) -> Self {
        fn finite_or_zero(v: f64) -> f64 {
            if v.is_finite() { v } else { 0.0 }
        }
        let peak_tolerance_db = finite_or_zero(self.peak_tolerance_db).max(0.0);
        let correction_weight = finite_or_zero(self.correction_weight).max(0.0);
        let min = finite_or_zero(self.min_freq_hz).max(0.0);
        let max = finite_or_zero(self.max_freq_hz).max(0.0);
        let (min_freq_hz, max_freq_hz) = if min <= max { (min, max) } else { (max, min) };
        Self {
            peak_tolerance_db,
            correction_weight,
            min_freq_hz,
            max_freq_hz,
        }
    }
}

pub fn reliable_upper_passband_hz(curve: &autoeq_core::Curve) -> Option<f64> {
    let upper_hz = roomeq_analysis::response_metrics::detect_passband_and_mean(curve)
        .0?
        .1;
    (upper_hz.is_finite() && upper_hz > 0.0).then_some(upper_hz)
}

pub fn constrain_channel_matching_profile_to_passbands(
    mut profile: ChannelMatchingCorrectionProfile,
    curves: &HashMap<String, autoeq_core::Curve>,
) -> ChannelMatchingCorrectionProfile {
    if let Some(reliable_upper_hz) = curves
        .values()
        .filter_map(reliable_upper_passband_hz)
        .reduce(f64::min)
    {
        profile.max_freq_hz = profile.max_freq_hz.min(reliable_upper_hz);
    }
    profile
}

pub fn correct_inter_channel_deviation_with_profile(
    final_curves: &HashMap<String, crate::Curve>,
    f3_hz: f64,
    max_filters: usize,
    sample_rate: f64,
    profile: ChannelMatchingCorrectionProfile,
) -> Vec<ChannelMatchingResult> {
    let profile =
        constrain_channel_matching_profile_to_passbands(profile.sanitized(), final_curves);
    if final_curves.len() <= 1 || max_filters == 0 {
        return Vec::new();
    }

    let first_curve = match final_curves.values().next() {
        Some(c) => c,
        None => return Vec::new(),
    };
    let aligned_curves;
    let final_curves = if final_curves.values().any(|curve| {
        !roomeq_analysis::frequency_grid::same_frequency_grid(&first_curve.freq, &curve.freq)
    }) {
        let Some(aligned) = align_curves_to_common_grid(final_curves) else {
            warn!(
                "Channel matching correction skipped: channels have no valid shared frequency span"
            );
            return Vec::new();
        };
        aligned_curves = aligned;
        &aligned_curves
    } else {
        final_curves
    };
    let freq = &final_curves.values().next().unwrap().freq;
    let n = freq.len();

    let (matching_min_hz, matching_max_hz) = profile.matching_band(f3_hz);

    // Compute pointwise average (reference) — normalize each to its own passband mean first
    let passband_means: HashMap<String, f64> = final_curves
        .iter()
        .map(|(name, curve)| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for i in 0..curve.spl.len().min(n) {
                if freq[i] >= f3_hz && freq[i] <= 10000.0 {
                    sum += curve.spl[i];
                    count += 1;
                }
            }
            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            (name.clone(), mean)
        })
        .collect();

    let mut reference = vec![0.0; n];
    for (name, curve) in final_curves {
        let mean = passband_means[name];
        for (i, ref_val) in reference
            .iter_mut()
            .enumerate()
            .take(n.min(curve.spl.len()))
        {
            *ref_val += (curve.spl[i] - mean) / final_curves.len() as f64;
        }
    }

    let mut results = Vec::new();

    for (name, curve) in final_curves {
        let mean = passband_means[name];
        // diff = channel (normalized) - reference → positive means channel is louder
        let diff: Vec<f64> = (0..n.min(curve.spl.len()))
            .map(|i| (curve.spl[i] - mean) - reference[i])
            .collect();

        // Find the N largest deviation peaks in the midrange (f3..10kHz)
        // Use 1/3 octave smoothing to avoid chasing noise
        let smoothed_diff = smooth_for_peak_finding(&diff, freq, n);

        let mut peaks: Vec<(usize, f64)> = Vec::new(); // (index, signed_deviation)
        for i in 1..smoothed_diff.len().saturating_sub(1) {
            let f = freq[i];
            if f < matching_min_hz || f > matching_max_hz {
                continue;
            }
            let abs_val = smoothed_diff[i].abs();
            if abs_val <= profile.peak_tolerance_db {
                continue; // Leave role-appropriate deviations uncorrected.
            }
            // Local extremum (peak or dip in deviation)
            let is_peak = smoothed_diff[i].abs() >= smoothed_diff[i - 1].abs()
                && smoothed_diff[i].abs() >= smoothed_diff[i + 1].abs();
            if is_peak {
                peaks.push((i, smoothed_diff[i]));
            }
        }

        // Sort by absolute deviation (largest first), take up to max_filters
        peaks.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        peaks.truncate(max_filters);

        // Enforce minimum 1/3 octave spacing between selected peaks
        let mut selected: Vec<(usize, f64)> = Vec::new();
        for &(idx, dev) in &peaks {
            let f = freq[idx];
            let too_close = selected.iter().any(|&(sidx, _)| {
                let sf = freq[sidx];
                (f / sf).abs().log2().abs() < 1.0 / 3.0
            });
            if !too_close {
                selected.push((idx, dev));
            }
        }

        // Create PEQ filters to correct the deviations
        let mut filters = Vec::new();
        for &(idx, dev) in &selected {
            let f = freq[idx];
            let gain_db = channel_matching_correction_gain(dev, profile);
            if gain_db.abs() < MIN_CORRECTION_DB {
                continue;
            }
            // Q based on deviation width: narrow for sharp peaks, broader for gentle humps
            let q = estimate_correction_q(&smoothed_diff, freq, idx);

            filters.push(Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                f,
                sample_rate,
                q,
                gain_db,
            ));
        }

        results.push(ChannelMatchingResult {
            channel_name: name.clone(),
            filters,
        });
    }

    results
}

pub(super) fn channel_matching_correction_gain(
    deviation_db: f64,
    profile: ChannelMatchingCorrectionProfile,
) -> f64 {
    let excess = deviation_db.abs() - profile.peak_tolerance_db;
    if excess <= 0.0 {
        0.0
    } else {
        -deviation_db.signum() * excess * profile.correction_weight
    }
}
