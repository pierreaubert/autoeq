//! Calibrate two acoustic bands before global EQ. Crossover flatness alone is
//! invariant to level and does not establish the bass/treble target relationship.

use super::*;
use crate::eq::EqResources;

#[allow(clippy::too_many_arguments)]
pub(crate) fn align_two_band_target_levels(
    drivers: &[Curve],
    gains: &mut [f64],
    delays: &mut [f64],
    frequencies: &[f64],
    inversions: &[bool],
    crossover_type: CrossoverType,
    config: &OptimizerConfig,
    resources: &EqResources,
    sample_rate: f64,
    correction_filters: Option<&[math_audio_iir_fir::Biquad]>,
) -> Option<Curve> {
    if drivers.len() != 2
        || gains.len() != 2
        || delays.len() != 2
        || inversions.len() != 2
        || frequencies.len() != 1
        || !config.max_db.is_finite()
        || config.max_db <= 0.0
    {
        return None;
    }
    let data = DriversLossData::new_ordered(
        drivers
            .iter()
            .zip(inversions)
            .map(|(curve, &inverted)| super::apply_polarity_inversion_to_driver(curve, inverted))
            .collect(),
        crossover_type,
    );
    let render = |candidate: &[f64], candidate_delays: &[f64]| {
        let response = autoeq_optim::loss::compute_drivers_combined_response_complex(
            &data,
            candidate,
            frequencies,
            Some(candidate_delays),
            sample_rate,
        );
        Curve {
            freq: data.freq_grid.clone(),
            spl: response.mapv(|z| 20.0 * z.norm().max(1e-12).log10()),
            phase: Some(response.mapv(|z| z.arg().to_degrees())),
            ..Curve::default()
        }
    };
    let initial = render(gains, delays);
    let correction_response = correction_filters.map(|filters| {
        crate::response::compute_peq_complex_response(filters, &data.freq_grid, sample_rate)
    });
    // No trustworthy upper band => retain the original crossover solution.
    crate::eq::group_upper_reference_target(&initial, config, Some(resources))?;
    let score = |curve: &Curve| {
        let corrected;
        let evaluated = if let Some(response) = &correction_response {
            corrected = crate::response::apply_complex_response(curve, response);
            &corrected
        } else {
            curve
        };
        crate::eq::group_upper_reference_target(evaluated, config, Some(resources))
            .map(|target| {
                if correction_filters.is_some() {
                    // Once shape EQ is fixed, align band LEVELS rather than
                    // letting peaks/dips bias the shared target's SPL reference.
                    let residual = Curve {
                        freq: evaluated.freq.clone(),
                        spl: &evaluated.spl - &target.spl,
                        ..Curve::default()
                    };
                    roomeq_analysis::response_metrics::mean_response_in_range(
                        &residual,
                        config.min_freq,
                        config.max_freq,
                    )
                    .abs()
                } else {
                    crate::group::target_error_score(
                        evaluated,
                        &target,
                        config.min_freq,
                        config.max_freq,
                    )
                }
            })
            .unwrap_or(f64::INFINITY)
    };
    let before = score(&initial);
    // Preserve common gain where possible, but do not let an arbitrary common
    // offset at a bound prevent relative calibration. The final shared target
    // alignment owns overall level. Keep setup_drivers_bounds' original limits.
    let center = (gains[0] + gains[1]) * 0.5;
    let radius = config.max_db;
    let candidate_gains = |difference: f64| {
        let difference = difference.clamp(-radius, radius);
        let room = radius - difference.abs();
        let common = center.clamp(-room, room);
        [common + difference, common - difference]
    };
    let search_gains = |candidate_delays: &[f64]| {
        let mut best_half_difference = (gains[0] - gains[1]) * 0.5;
        let mut best_curve = render(gains, candidate_delays);
        let mut best_score = score(&best_curve);
        // Bounded coarse-to-fine search avoids assuming convexity of complex sums.
        let mut low = -radius;
        let mut high = radius;
        for _ in 0..4 {
            let step = (high - low) / 32.0;
            for index in 0..=32 {
                let difference = low + index as f64 * step;
                let candidate = render(&candidate_gains(difference), candidate_delays);
                let error = score(&candidate);
                if error < best_score {
                    best_score = error;
                    best_half_difference = difference;
                    best_curve = candidate;
                }
            }
            low = (best_half_difference - step).max(-radius);
            high = (best_half_difference + step).min(radius);
        }
        (candidate_gains(best_half_difference), best_curve, best_score)
    };
    let (mut best_gains, mut best_curve, mut best_score) = search_gains(delays);
    let mut best_delays = [delays[0], delays[1]];
    if let Some(phase) = config.phase_alignment.as_ref().filter(|phase|
        phase.enabled && phase.max_delay_ms.is_finite() && phase.max_delay_ms > 0.0
            && drivers.iter().all(|driver| driver.phase.is_some()))
    {
        // Level-only fitting can move two nearly opposite branches to equal
        // amplitude and create a new null. Recheck phase at each calibrated
        // level, before the common FIR is designed (it cannot undo cancellation).
        let quality = |curve: &Curve, level_error: f64| {
            let corrected;
            let evaluated = if let Some(response) = &correction_response {
                corrected = crate::response::apply_complex_response(curve, response);
                &corrected
            } else { curve };
            let target = crate::eq::group_upper_reference_target(evaluated, config, Some(resources)).unwrap();
            let rms = crate::group::target_error_score(evaluated, &target, config.min_freq, config.max_freq);
            rms + if correction_filters.is_some() { 10.0 * level_error } else { 0.0 }
        };
        let mut best_quality = quality(&best_curve, best_score);
        let center_delay = (delays[0] + delays[1]) * 0.5;
        let limit = phase.max_delay_ms.min(40.0);
        let mut low = -limit;
        let mut high = limit;
        for _ in 0..2 {
            let step = (high - low) / 40.0;
            for index in 0..=40 {
                let relative = (low + index as f64 * step).clamp(-limit, limit);
                let room = 20.0 - relative.abs() * 0.5;
                let common = center_delay.clamp(-room, room);
                let trial_delays = [common + relative * 0.5, common - relative * 0.5];
                let (trial_gains, trial_curve, trial_score) = search_gains(&trial_delays);
                let trial_quality = quality(&trial_curve, trial_score);
                if trial_quality < best_quality {
                    best_quality = trial_quality;
                    best_gains = trial_gains;
                    best_curve = trial_curve;
                    best_score = trial_score;
                    best_delays = trial_delays;
                }
            }
            let relative = best_delays[0] - best_delays[1];
            low = (relative - step).max(-limit);
            high = (relative + step).min(limit);
        }
    }
    gains.copy_from_slice(&best_gains);
    delays.copy_from_slice(&best_delays);
    let metric = if correction_filters.is_some() {
        "band level error"
    } else {
        "target RMS"
    };
    info!(
        "Two-band target level calibration: gains={gains:?}, {metric} {before:.3} -> {best_score:.3} dB"
    );
    Some(best_curve)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bass_target_calibration_rechecks_phase_after_level_changes() {
        let curve = Curve {
            freq: Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 256),
            spl: Array1::from_elem(256, 80.0),
            phase: Some(Array1::zeros(256)),
            ..Curve::default()
        };
        let mut gains = [-6.0, 6.0];
        // Half a cycle of relative delay at the 80 Hz crossover produces a
        // cancellation when the gain calibration equalizes the two branches.
        let mut delays = [6.25, 0.0];
        let config = OptimizerConfig {
            min_freq: 20.0, max_freq: 200.0, max_db: 6.0,
            phase_alignment: Some(roomeq_model::PhaseAlignmentConfig {
                max_delay_ms: 10.0, ..roomeq_model::PhaseAlignmentConfig::default()
            }),
            ..OptimizerConfig::default()
        };
        let output = align_two_band_target_levels(
            &[curve.clone(), curve], &mut gains, &mut delays, &[80.0], &[false, false],
            CrossoverType::LinkwitzRiley4, &config, &EqResources::default(), 48_000.0, Some(&[]),
        ).unwrap();
        let target = crate::eq::group_upper_reference_target(&output, &config, None).unwrap();
        assert!(crate::group::target_error_score(&output, &target, 20.0, 200.0) < 0.1);
        assert!((delays[0] - delays[1]).abs() < 0.1, "{delays:?}");
        assert!(gains.iter().all(|gain| gain.abs() <= config.max_db));
    }

    #[test]
    fn bass_target_calibration_repairs_opposite_driver_gain_limits() {
        let curve = Curve {
            freq: Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 256),
            spl: Array1::from_elem(256, 80.0),
            phase: Some(Array1::zeros(256)),
            ..Curve::default()
        };
        let mut gains = [-6.0, 6.0];
        let config = OptimizerConfig {
            min_freq: 20.0,
            max_freq: 200.0,
            max_db: 6.0,
            ..OptimizerConfig::default()
        };
        let output = align_two_band_target_levels(
            &[curve.clone(), curve],
            &mut gains,
            &mut [0.0, 0.0],
            &[80.0],
            &[false, false],
            CrossoverType::LinkwitzRiley4,
            &config,
            &EqResources::default(),
            48_000.0,
            None,
        )
        .unwrap();
        assert!(gains.iter().all(|g| g.abs() < 0.02), "{gains:?}");
        let target = crate::eq::group_upper_reference_target(&output, &config, None).unwrap();
        assert!(crate::group::target_error_score(&output, &target, 20.0, 200.0) < 0.02);
    }

    #[test]
    fn bass_target_calibration_can_leave_a_common_gain_bound() {
        let main = Curve {
            freq: Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 256),
            spl: Array1::from_elem(256, 80.0),
            phase: Some(Array1::zeros(256)),
            ..Curve::default()
        };
        let sub = Curve {
            spl: &main.spl - 12.0,
            ..main.clone()
        };
        let mut gains = [6.0, 6.0];
        let config = OptimizerConfig {
            min_freq: 20.0,
            max_freq: 200.0,
            max_db: 6.0,
            ..OptimizerConfig::default()
        };
        let output = align_two_band_target_levels(
            &[sub, main],
            &mut gains,
            &mut [0.0, 0.0],
            &[80.0],
            &[false, false],
            CrossoverType::LinkwitzRiley4,
            &config,
            &EqResources::default(),
            48_000.0,
            None,
        )
        .unwrap();
        let target = crate::eq::group_upper_reference_target(&output, &config, None).unwrap();
        assert!(crate::group::target_error_score(&output, &target, 20.0, 200.0) < 0.02);
        assert!(gains[0] > 5.98 && gains[1] < -5.98, "{gains:?}");
        assert!(gains.iter().all(|gain| gain.abs() <= config.max_db));
    }

    #[test]
    fn bass_target_calibration_does_not_use_missing_upper_band() {
        let curve = Curve {
            freq: Array1::from_vec(vec![20.0, 80.0, 200.0]),
            spl: Array1::from_elem(3, 80.0),
            ..Curve::default()
        };
        let mut gains = [-6.0, 6.0];
        let config = OptimizerConfig {
            max_freq: 200.0,
            max_db: 6.0,
            ..OptimizerConfig::default()
        };
        assert!(
            align_two_band_target_levels(
                &[curve.clone(), curve],
                &mut gains,
                &mut [0.0, 0.0],
                &[80.0],
                &[false, false],
                CrossoverType::LinkwitzRiley4,
                &config,
                &EqResources::default(),
                48_000.0,
                None,
            )
            .is_none()
        );
        assert_eq!(gains, [-6.0, 6.0]);
    }

    #[test]
    fn bass_target_calibration_rechecks_level_after_shape_eq() {
        let curve = Curve {
            freq: Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 256),
            spl: Array1::from_elem(256, 80.0),
            phase: Some(Array1::zeros(256)),
            ..Curve::default()
        };
        let mut gains = [0.0, 0.0];
        let config = OptimizerConfig {
            min_freq: 20.0,
            max_freq: 200.0,
            max_db: 6.0,
            ..OptimizerConfig::default()
        };
        let filters = [math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Lowshelf,
            80.0,
            48_000.0,
            0.707,
            -4.0,
        )];
        let output = align_two_band_target_levels(
            &[curve.clone(), curve],
            &mut gains,
            &mut [0.0, 0.0],
            &[80.0],
            &[false, false],
            CrossoverType::LinkwitzRiley4,
            &config,
            &EqResources::default(),
            48_000.0,
            Some(&filters),
        )
        .unwrap();
        let response =
            crate::response::compute_peq_complex_response(&filters, &output.freq, 48_000.0);
        let corrected = crate::response::apply_complex_response(&output, &response);
        let target = crate::eq::group_upper_reference_target(&corrected, &config, None).unwrap();
        let residual = Curve {
            spl: &corrected.spl - &target.spl,
            ..corrected
        };
        assert!(
            roomeq_analysis::response_metrics::mean_response_in_range(&residual, 20.0, 200.0).abs()
                < 0.02
        );
        assert!(gains[0] > 1.0 && gains[1] < -1.0, "{gains:?}");
        assert!(gains.iter().all(|g| g.abs() <= config.max_db));
    }
}
