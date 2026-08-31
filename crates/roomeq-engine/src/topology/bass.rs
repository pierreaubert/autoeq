use super::{compute_flat_loss, predict_bass_management_sum, same_frequency_grid};
use crate::Curve;

/// Maximum target-relative underfill accepted from an optimized routed
/// crossover after normalizing against the corrected main-only band.
pub const MAX_ACCEPTED_CROSSOVER_UNDERFILL_DB: f64 = 3.0;

pub fn bass_management_objective(curve: Option<&Curve>, xover_freq: f64) -> Option<f64> {
    let curve = curve?;
    // Use a symmetric band around the crossover in log-frequency space.
    // When a cap (20 Hz low or 2 kHz high) is hit, adjust the other side
    // to maintain equal octave span on both sides.
    let mut min_freq = xover_freq / 2.0;
    let mut max_freq = xover_freq * 2.0;
    if min_freq < 20.0 || max_freq > 2000.0 {
        let ratio = if min_freq < 20.0 {
            xover_freq / 20.0
        } else {
            2000.0 / xover_freq
        };
        min_freq = (xover_freq / ratio).max(20.0);
        max_freq = (xover_freq * ratio).min(2000.0);
    }
    max_freq = max_freq.max(min_freq + 1.0);
    Some(compute_flat_loss(curve, min_freq, max_freq))
}

/// Score the routed response against the configured target shape around the
/// crossover. A target-independent flatness score biases redirected bass low
/// whenever the requested house curve rises toward low frequencies.
pub fn bass_management_objective_with_target(
    curve: Option<&Curve>,
    target: Option<&Curve>,
    xover_freq: f64,
) -> Option<f64> {
    let Some(target) = target else {
        return bass_management_objective(curve, xover_freq);
    };
    let curve = curve?;
    let target = autoeq_core::curve_transforms::interpolate_log_space(&curve.freq, target);
    let mut error = curve.clone();
    for (level, target_level) in error.spl.iter_mut().zip(target.spl.iter()) {
        *level -= *target_level;
    }
    error.phase = None;
    let mut min_freq = xover_freq / 2.0;
    let mut max_freq = xover_freq * 2.0;
    if min_freq < 20.0 || max_freq > 2000.0 {
        let ratio = if min_freq < 20.0 {
            xover_freq / 20.0
        } else {
            2000.0 / xover_freq
        };
        min_freq = (xover_freq / ratio).max(20.0);
        max_freq = (xover_freq * ratio).min(2000.0);
    }
    max_freq = max_freq.max(min_freq + 1.0);

    // Calibrate absolute level later, but anchor the crossover region to the
    // corrected main-only band above it. Centering inside the crossover itself
    // makes a broad cancellation or underfill look deceptively flat.
    let reference_max_freq = (xover_freq * 8.0).min(2000.0);
    let reference_levels = error
        .freq
        .iter()
        .zip(error.spl.iter())
        .filter(|(frequency, level)| {
            **frequency >= max_freq
                && **frequency <= reference_max_freq
                && frequency.is_finite()
                && level.is_finite()
        })
        .map(|(_, level)| *level)
        .collect::<Vec<_>>();
    if reference_levels.is_empty() {
        return bass_management_objective(Some(&error), xover_freq);
    }
    let reference_mean = reference_levels.iter().sum::<f64>() / reference_levels.len() as f64;
    error.spl.mapv_inplace(|level| level - reference_mean);
    let base_loss = autoeq_optim::loss::flat_loss(&error.freq, &error.spl, min_freq, max_freq);
    let crossover_errors = error
        .freq
        .iter()
        .zip(error.spl.iter())
        .filter(|(frequency, level)| {
            **frequency >= min_freq
                && **frequency <= max_freq
                && frequency.is_finite()
                && level.is_finite()
        })
        .map(|(_, level)| *level)
        .collect::<Vec<_>>();
    let deepest_underfill = crossover_errors
        .into_iter()
        .fold(0.0_f64, |worst, error| worst.max((-error).max(0.0)));

    Some(base_loss + 2.0 * deepest_underfill)
}

/// Return the deepest target-relative dip across one octave centered on the
/// crossover, normalized to the corrected main-only band above that region.
pub fn bass_management_max_underfill_db_with_target(
    curve: Option<&Curve>,
    target: Option<&Curve>,
    xover_freq: f64,
) -> Option<f64> {
    let curve = curve?;
    let target = target?;
    let target = autoeq_core::curve_transforms::interpolate_log_space(&curve.freq, target);
    let mut error = curve.clone();
    for (level, target_level) in error.spl.iter_mut().zip(target.spl.iter()) {
        *level -= *target_level;
    }

    let mut min_freq = xover_freq / 2.0;
    let mut max_freq = xover_freq * 2.0;
    if min_freq < 20.0 || max_freq > 2000.0 {
        let ratio = if min_freq < 20.0 {
            xover_freq / 20.0
        } else {
            2000.0 / xover_freq
        };
        min_freq = (xover_freq / ratio).max(20.0);
        max_freq = (xover_freq * ratio).min(2000.0);
    }
    max_freq = max_freq.max(min_freq + 1.0);

    let reference_max_freq = (xover_freq * 8.0).min(2000.0);
    let reference_levels = error
        .freq
        .iter()
        .zip(error.spl.iter())
        .filter(|(frequency, level)| {
            **frequency >= max_freq
                && **frequency <= reference_max_freq
                && frequency.is_finite()
                && level.is_finite()
        })
        .map(|(_, level)| *level)
        .collect::<Vec<_>>();
    if reference_levels.is_empty() {
        return None;
    }
    let reference_mean = reference_levels.iter().sum::<f64>() / reference_levels.len() as f64;

    error
        .freq
        .iter()
        .zip(error.spl.iter())
        .filter(|(frequency, level)| {
            **frequency >= min_freq
                && **frequency <= max_freq
                && frequency.is_finite()
                && level.is_finite()
        })
        .map(|(_, level)| (reference_mean - *level).max(0.0))
        .reduce(f64::max)
}

/// Return the frequency and depth of the worst target-relative crossover dip.
///
/// The level reference is the corrected main-only band above the crossover,
/// matching [`bass_management_max_underfill_db_with_target`].
pub fn bass_management_worst_underfill_with_target(
    curve: Option<&Curve>,
    target: Option<&Curve>,
    xover_freq: f64,
) -> Option<(f64, f64)> {
    let curve = curve?;
    let target = target?;
    let target = autoeq_core::curve_transforms::interpolate_log_space(&curve.freq, target);
    let mut error = curve.clone();
    for (level, target_level) in error.spl.iter_mut().zip(target.spl.iter()) {
        *level -= *target_level;
    }

    let mut min_freq = xover_freq / 2.0;
    let mut max_freq = xover_freq * 2.0;
    if min_freq < 20.0 || max_freq > 2_000.0 {
        let ratio = if min_freq < 20.0 {
            xover_freq / 20.0
        } else {
            2_000.0 / xover_freq
        };
        min_freq = (xover_freq / ratio).max(20.0);
        max_freq = (xover_freq * ratio).min(2_000.0);
    }
    max_freq = max_freq.max(min_freq + 1.0);

    let reference_max_freq = (xover_freq * 8.0).min(2_000.0);
    let reference_levels = error
        .freq
        .iter()
        .zip(error.spl.iter())
        .filter(|(frequency, level)| {
            **frequency >= max_freq
                && **frequency <= reference_max_freq
                && frequency.is_finite()
                && level.is_finite()
        })
        .map(|(_, level)| *level)
        .collect::<Vec<_>>();
    if reference_levels.is_empty() {
        return None;
    }
    let reference_mean = reference_levels.iter().sum::<f64>() / reference_levels.len() as f64;

    error
        .freq
        .iter()
        .zip(error.spl.iter())
        .filter(|(frequency, level)| {
            **frequency >= min_freq
                && **frequency <= max_freq
                && frequency.is_finite()
                && level.is_finite()
        })
        .map(|(frequency, level)| (*frequency, (reference_mean - *level).max(0.0)))
        .max_by(|left, right| left.1.total_cmp(&right.1))
}

/// Return the deepest crossover cancellation relative to the stronger
/// realized branch at each frequency.
///
/// Unlike target error, this isolates underfill introduced by coherent
/// summation. Broad room-response or target mismatch shared by the branches
/// cannot make a valid crossover look like a cancellation failure.
pub fn bass_management_crossover_cancellation_underfill_db(
    main_branch: &Curve,
    bass_branch: &Curve,
    combined: &Curve,
    xover_freq: f64,
) -> Option<f64> {
    if !same_frequency_grid(&main_branch.freq, &bass_branch.freq)
        || !same_frequency_grid(&main_branch.freq, &combined.freq)
        || main_branch.spl.len() != bass_branch.spl.len()
        || main_branch.spl.len() != combined.spl.len()
    {
        return None;
    }
    let min_freq = (xover_freq / 2.0).max(20.0);
    let max_freq = (xover_freq * 2.0).min(2_000.0);
    main_branch
        .freq
        .iter()
        .zip(main_branch.spl.iter())
        .zip(bass_branch.spl.iter())
        .zip(combined.spl.iter())
        .filter(|(((frequency, main), bass), sum)| {
            **frequency >= min_freq
                && **frequency <= max_freq
                && frequency.is_finite()
                && main.is_finite()
                && bass.is_finite()
                && sum.is_finite()
        })
        .map(|(((_, main), bass), sum)| (main.max(*bass) - *sum).max(0.0))
        .reduce(f64::max)
}

pub fn bass_management_crossover_type_candidates(requested: &str) -> Vec<String> {
    let requested = requested.trim();
    if requested.eq_ignore_ascii_case("auto") || requested.eq_ignore_ascii_case("optimize") {
        vec![
            "LR24".to_string(),
            "LR48".to_string(),
            "BW12".to_string(),
            "BW24".to_string(),
        ]
    } else {
        vec![requested.to_string()]
    }
}

pub fn select_bass_management_crossover_type(
    requested: &str,
    main_curve: &Curve,
    sub_curve: &Curve,
    xover_freq: f64,
    sample_rate: f64,
) -> String {
    let candidates = bass_management_crossover_type_candidates(requested);
    if candidates.len() == 1 {
        return candidates[0].clone();
    }

    candidates
        .iter()
        .filter(|candidate| {
            candidate
                .parse::<autoeq_optim::loss::CrossoverType>()
                .is_ok()
        })
        .filter_map(|candidate| {
            let predicted = predict_bass_management_sum(
                main_curve,
                sub_curve,
                candidate,
                xover_freq,
                sample_rate,
                0.0,
                0.0,
                0.0,
                0.0,
                false,
            );
            bass_management_objective(predicted.as_ref(), xover_freq)
                .map(|objective| (candidate.clone(), objective))
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(candidate, _)| candidate)
        .unwrap_or_else(|| "LR24".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn curve_with_levels(levels: impl Fn(f64) -> f64) -> Curve {
        let freq = Array1::linspace(20.0, 1_000.0, 981);
        Curve {
            spl: freq.mapv(levels),
            phase: Some(Array1::zeros(freq.len())),
            freq,
            ..Curve::default()
        }
    }

    #[test]
    fn crossover_underfill_is_anchored_to_main_only_band() {
        let target = curve_with_levels(|_| 0.0);
        let response = curve_with_levels(|frequency| {
            if (40.0..=160.0).contains(&frequency) {
                -4.25
            } else {
                0.0
            }
        });
        let underfill =
            bass_management_max_underfill_db_with_target(Some(&response), Some(&target), 80.0)
                .expect("reference band");
        assert!((underfill - 4.25).abs() <= 0.02);
        assert!(underfill > MAX_ACCEPTED_CROSSOVER_UNDERFILL_DB);
    }

    #[test]
    fn common_level_offset_is_not_crossover_underfill() {
        let target = curve_with_levels(|_| 0.0);
        let response = curve_with_levels(|_| -12.0);
        let underfill =
            bass_management_max_underfill_db_with_target(Some(&response), Some(&target), 80.0)
                .expect("reference band");
        assert!(underfill <= 1.0e-12);
    }

    #[test]
    fn cancellation_underfill_is_measured_against_realized_branches() {
        let main = curve_with_levels(|_| -6.0);
        let bass = curve_with_levels(|_| -6.0);
        let combined = curve_with_levels(|frequency| {
            if (40.0..=160.0).contains(&frequency) {
                -10.25
            } else {
                0.0
            }
        });
        let underfill =
            bass_management_crossover_cancellation_underfill_db(&main, &bass, &combined, 80.0)
                .expect("shared grid");
        assert!((underfill - 4.25).abs() <= 0.02);
        assert!(underfill > MAX_ACCEPTED_CROSSOVER_UNDERFILL_DB);
    }

    #[test]
    fn branch_magnitude_mismatch_is_not_crossover_cancellation() {
        let main = curve_with_levels(|frequency| -12.0 + frequency.log10());
        let bass = curve_with_levels(|frequency| -3.0 - frequency.log10());
        let combined = curve_with_levels(|frequency| {
            let main = -12.0 + frequency.log10();
            let bass = -3.0 - frequency.log10();
            20.0 * (10.0_f64.powf(main / 20.0) + 10.0_f64.powf(bass / 20.0)).log10()
        });
        let underfill =
            bass_management_crossover_cancellation_underfill_db(&main, &bass, &combined, 80.0)
                .expect("shared grid");
        assert!(underfill <= 1.0e-12);
    }
}
