use super::{compute_flat_loss, predict_bass_management_sum, same_frequency_grid};
use crate::Curve;

/// Maximum target-relative underfill accepted from an optimized routed
/// crossover after normalizing against the corrected main-only band.
pub const MAX_ACCEPTED_CROSSOVER_UNDERFILL_DB: f64 = 3.0;
/// Tolerance for response-grid interpolation and optimizer boundary noise.
/// The objective still penalizes every amount above the nominal 3 dB limit.
pub const CROSSOVER_UNDERFILL_ACCEPTANCE_TOLERANCE_DB: f64 = 0.05;

/// Minimum relative reduction in dB shortfall for a still-imperfect Post-EQ pass.
///
/// This product policy admits useful partial correction; it is not a loudness
/// percentage or a demonstrated audibility threshold.
pub const POST_EQ_MIN_UNDERFILL_REDUCTION: f64 = 0.20;
/// Minimum absolute improvement required alongside the relative Post-EQ rule.
///
/// One dB prevents small numerical or already-negligible changes from passing
/// solely because their percentage is large.
pub const POST_EQ_MIN_UNDERFILL_IMPROVEMENT_DB: f64 = 1.0;

/// Accept a good residual or a material improvement over the immediate input.
///
/// Both depths must be finite and nonnegative. Above the absolute quality
/// target, require at least 20% and 1 dB reduction in the shortfall measured
/// in dB. This stage policy does not replace final output or seat checks.
pub fn post_eq_underfill_is_acceptable(before_db: f64, after_db: f64) -> bool {
    if !before_db.is_finite() || !after_db.is_finite() || before_db < 0.0 || after_db < 0.0 {
        return false;
    }
    bass_management_underfill_is_acceptable(after_db)
        || (before_db - after_db >= POST_EQ_MIN_UNDERFILL_IMPROVEMENT_DB
            && after_db <= before_db * (1.0 - POST_EQ_MIN_UNDERFILL_REDUCTION))
}

pub fn bass_management_underfill_is_acceptable(underfill_db: f64) -> bool {
    underfill_db
        <= MAX_ACCEPTED_CROSSOVER_UNDERFILL_DB + CROSSOVER_UNDERFILL_ACCEPTANCE_TOLERANCE_DB
}

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
    bass_management_crossover_cancellation_worst_bin(main_branch, bass_branch, combined, xover_freq)
        .map(|(_, deficit)| deficit)
}

/// Frequency and deficit of the worst cancellation bin, not the configured
/// crossover center. The deficit is relative to the louder physical branch.
pub fn bass_management_crossover_cancellation_worst_bin(
    main_branch: &Curve,
    bass_branch: &Curve,
    combined: &Curve,
    xover_freq: f64,
) -> Option<(f64, f64)> {
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
        .map(|(((frequency, main), bass), sum)| (*frequency, (main.max(*bass) - *sum).max(0.0)))
        .max_by(|left, right| left.1.total_cmp(&right.1))
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

    #[test]
    fn post_eq_underfill_accepts_absolute_or_material_relative_improvement() {
        for (before, after, accepted) in [
            (3.05, 3.05, true),
            (3.051, 3.051, false),
            (5.0, 4.0, true),
            (5.0, 4.000001, false),
            (10.0, 8.0, true),
            (10.0, 8.000001, false),
            (4.0, 3.1, false),
            (4.0, 3.2, false),
            (0.0, 0.0, true),
            (10.160898695740133, 3.950030168390911, true),
            (20.113734419323308, 6.765109022500242, true),
            (10.0, 10.0, false),
            (10.0, 11.0, false),
        ] {
            assert_eq!(
                post_eq_underfill_is_acceptable(before, after),
                accepted,
                "{before} -> {after}"
            );
        }
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            assert!(!post_eq_underfill_is_acceptable(invalid, 0.0));
            assert!(!post_eq_underfill_is_acceptable(10.0, invalid));
        }
    }
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
    fn underfill_acceptance_has_small_shared_boundary_tolerance() {
        assert!(bass_management_underfill_is_acceptable(3.0));
        assert!(bass_management_underfill_is_acceptable(3.049));
        assert!(!bass_management_underfill_is_acceptable(3.051));
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
    fn dominating_hot_bass_passes_underfill_that_balance_fails() {
        // Phase 0 pathology record (measured unknown: +12 dB bass drowns the
        // main and the route optimizer calls it optimal). Underfill is
        // max(branch) - sum, so single-branch domination shrinks it: the raw
        // metric stays truthful, but the route OBJECTIVE must add a
        // domination penalty (Phase 3a) instead of rewarding this.
        let main = curve_with_levels(|_| 0.0);
        let bass = curve_with_levels(|_| 0.0);
        let nulled = curve_with_levels(|frequency| {
            if (40.0..=160.0).contains(&frequency) {
                -40.0
            } else {
                6.0
            }
        });
        let balanced =
            bass_management_crossover_cancellation_underfill_db(&main, &bass, &nulled, 80.0)
                .expect("shared grid");
        let hot_bass = curve_with_levels(|_| 12.0);
        let dominated = curve_with_levels(|frequency| {
            if (40.0..=160.0).contains(&frequency) {
                9.5
            } else {
                12.0
            }
        });
        let drowning =
            bass_management_crossover_cancellation_underfill_db(&main, &hot_bass, &dominated, 80.0)
                .expect("shared grid");
        assert!(balanced > 20.0, "balanced null must read huge: {balanced}");
        assert!(
            drowning < MAX_ACCEPTED_CROSSOVER_UNDERFILL_DB,
            "drowning must pass the raw gate: {drowning}"
        );
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

    #[test]
    fn cancellation_worst_bin_is_not_assumed_to_be_crossover_center() {
        let main = curve_with_levels(|_| -6.0);
        let bass = main.clone();
        let combined = curve_with_levels(|frequency| if frequency == 117.0 { -18.0 } else { -6.0 });
        assert_eq!(
            bass_management_crossover_cancellation_worst_bin(&main, &bass, &combined, 80.0),
            Some((117.0, 12.0))
        );
        let mut shifted = combined.clone();
        shifted.freq += 0.5;
        assert_eq!(
            bass_management_crossover_cancellation_worst_bin(&main, &bass, &shifted, 80.0),
            None
        );
    }
}
