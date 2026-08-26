//! Pure helpers shared by prepared group and topology execution.

use autoeq_core::{AutoeqError, Curve, Result, response};
use math_audio_iir_fir::Biquad;
use ndarray::Array1;
use roomeq_model::{
    MultiMeasurementConfig, MultiSeatConfig, MultiSeatStrategy, SpatialRobustnessSerdeConfig,
};

use crate::multiseat::{self, MultiSeatMeasurements};

pub const GLOBAL_EQ_REGRESSION_TOLERANCE: f64 = 1e-6;

pub fn multiseat_peq_config(policy: &MultiSeatConfig, seat_count: usize) -> MultiMeasurementConfig {
    let mut weights = policy
        .seat_weights
        .clone()
        .unwrap_or_else(|| vec![1.0; seat_count]);
    for weight in &mut weights {
        if !weight.is_finite() || *weight < 0.0 {
            *weight = 0.0;
        }
    }
    if policy.strategy == MultiSeatStrategy::PrimaryWithConstraints
        && policy.primary_seat < weights.len()
    {
        weights[policy.primary_seat] *= policy.primary_seat_weight.max(0.0);
    }
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum <= f64::EPSILON {
        weights = vec![1.0 / seat_count.max(1) as f64; seat_count];
    } else {
        for weight in &mut weights {
            *weight /= weight_sum;
        }
    }

    MultiMeasurementConfig {
        strategy: policy.all_channel_strategy,
        weights: Some(weights),
        variance_lambda: 1.0,
        spatial_robustness: Some(SpatialRobustnessSerdeConfig {
            variance_threshold_db: 3.0,
            transition_width_db: 2.0,
            min_correction_depth: 0.1,
            mask_smoothing_octaves: 1.0 / 6.0,
        }),
        bootstrap_uncertainty: None,
        rir_prototype: None,
    }
}

pub fn apply_per_sub_filters(
    seat_measurements: &[Vec<Curve>],
    per_sub_filters: &[Vec<Biquad>],
    sample_rate: f64,
) -> Vec<Vec<Curve>> {
    seat_measurements
        .iter()
        .zip(per_sub_filters.iter())
        .map(|(sub_curves, filters)| {
            sub_curves
                .iter()
                .map(|curve| {
                    if filters.is_empty() {
                        curve.clone()
                    } else {
                        let filter_response = response::compute_peq_complex_response(
                            filters,
                            &curve.freq,
                            sample_rate,
                        );
                        response::apply_complex_response(curve, &filter_response)
                    }
                })
                .collect()
        })
        .collect()
}

/// Spatially aggregate magnitudes using energy averaging and discard phase.
pub fn average_power_curve(curves: &[Curve]) -> Result<Curve> {
    let Some(first) = curves.first() else {
        return Err(AutoeqError::InvalidMeasurement {
            message: "Cannot average an empty multi-seat curve set".to_string(),
        });
    };
    for (index, curve) in curves.iter().enumerate() {
        if curve.freq.len() != first.freq.len()
            || curve
                .freq
                .iter()
                .zip(first.freq.iter())
                .any(|(left, right)| (left - right).abs() > 1e-6 * right.abs().max(1.0))
        {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!(
                    "Cannot average multi-seat curves because seat {index} has a different frequency grid"
                ),
            });
        }
    }

    let mut power_sum = Array1::<f64>::zeros(first.freq.len());
    for curve in curves {
        power_sum = power_sum + curve.spl.mapv(|spl| 10.0_f64.powf(spl / 10.0));
    }
    let average_power = power_sum / curves.len() as f64;
    Ok(Curve {
        freq: first.freq.clone(),
        spl: average_power.mapv(|power| 10.0 * power.max(1e-12).log10()),
        phase: None,
        ..Curve::default()
    })
}

pub fn flat_loss_score(curve: &Curve, min_freq: f64, max_freq: f64) -> f64 {
    let mean = roomeq_analysis::response_metrics::mean_response_in_range(curve, min_freq, max_freq);
    let normalized = &curve.spl - mean;
    autoeq_optim::loss::flat_loss(&curve.freq, &normalized, min_freq, max_freq)
}

pub fn eq_score_regressed(pre_score: f64, post_score: f64) -> bool {
    !post_score.is_finite()
        || (pre_score.is_finite() && post_score > pre_score + GLOBAL_EQ_REGRESSION_TOLERANCE)
}

pub fn identity_multiseat_result(
    measurements: &MultiSeatMeasurements,
    policy: &MultiSeatConfig,
) -> multiseat::MultiSeatOptimizationResult {
    multiseat::MultiSeatOptimizationResult {
        gains: vec![0.0; measurements.num_subs],
        delays: vec![0.0; measurements.num_subs],
        polarities: vec![false; measurements.num_subs],
        allpass_filters: vec![Vec::new(); measurements.num_subs],
        strategy: policy.strategy.clone(),
        objective_name: "identity".to_string(),
        objective_before: 0.0,
        objective_after: 0.0,
        objective_improvement_db: 0.0,
        variance_before: 0.0,
        variance_after: 0.0,
        variance_improvement_db: 0.0,
        improvement_db: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use roomeq_model::MultiSeatStrategy;

    use super::*;

    fn flat_curve() -> Curve {
        Curve {
            freq: array![100.0, 200.0, 400.0, 800.0, 1_600.0],
            spl: array![80.0, 80.0, 80.0, 80.0, 80.0],
            ..Curve::default()
        }
    }

    #[test]
    fn flat_loss_distinguishes_flat_and_uneven_curves() {
        assert!(flat_loss_score(&flat_curve(), 100.0, 1_600.0).abs() < 1e-6);
        let mut uneven = flat_curve();
        uneven.spl = array![80.0, 85.0, 80.0, 75.0, 80.0];
        assert!(flat_loss_score(&uneven, 100.0, 1_600.0) > 0.1);
    }

    #[test]
    fn global_eq_regression_guard_rejects_worse_or_nonfinite_scores() {
        assert!(eq_score_regressed(1.0, 1.01));
        assert!(eq_score_regressed(1.0, f64::NAN));
        assert!(!eq_score_regressed(1.0, 1.0));
        assert!(!eq_score_regressed(
            1.0,
            1.0 + GLOBAL_EQ_REGRESSION_TOLERANCE
        ));
    }

    #[test]
    fn power_average_is_magnitude_only_and_does_not_cancel() {
        let first = Curve {
            freq: array![100.0],
            spl: array![80.0],
            phase: Some(array![0.0]),
            ..Curve::default()
        };
        let second = Curve {
            phase: Some(array![180.0]),
            ..first.clone()
        };
        let average = average_power_curve(&[first, second]).unwrap();
        assert!(average.phase.is_none());
        assert!((average.spl[0] - 80.0).abs() < 1e-9);
    }

    #[test]
    fn power_average_rejects_empty_and_mismatched_grids() {
        assert!(average_power_curve(&[]).is_err());
        let first = flat_curve();
        let mut second = flat_curve();
        second.freq[2] = 401.0;
        assert!(average_power_curve(&[first, second]).is_err());
    }

    #[test]
    fn multiseat_peq_normalizes_valid_and_invalid_weights() {
        let policy = MultiSeatConfig {
            strategy: MultiSeatStrategy::PrimaryWithConstraints,
            primary_seat: 1,
            primary_seat_weight: 2.0,
            seat_weights: Some(vec![1.0, 1.0]),
            ..MultiSeatConfig::default()
        };
        let config = multiseat_peq_config(&policy, 2);
        assert_eq!(config.weights.unwrap(), vec![1.0 / 3.0, 2.0 / 3.0]);

        let invalid = MultiSeatConfig {
            seat_weights: Some(vec![f64::NAN, -1.0]),
            ..MultiSeatConfig::default()
        };
        assert_eq!(
            multiseat_peq_config(&invalid, 2).weights.unwrap(),
            vec![0.5, 0.5]
        );

        let mismatched = MultiSeatConfig {
            seat_weights: Some(vec![1.0]),
            ..MultiSeatConfig::default()
        };
        assert_eq!(
            multiseat_peq_config(&mismatched, 2).weights.unwrap(),
            vec![1.0]
        );
    }

    #[test]
    fn multiseat_peq_honors_fractional_primary_seat_weight() {
        let policy = MultiSeatConfig {
            strategy: MultiSeatStrategy::PrimaryWithConstraints,
            primary_seat: 0,
            primary_seat_weight: 0.5,
            seat_weights: Some(vec![1.0, 1.0]),
            ..MultiSeatConfig::default()
        };

        let weights = multiseat_peq_config(&policy, 2).weights.unwrap();
        assert!((weights[0] - 1.0 / 3.0).abs() <= 1e-12);
        assert!((weights[1] - 2.0 / 3.0).abs() <= 1e-12);
    }

    #[test]
    fn applying_empty_per_sub_filters_preserves_measurements() {
        let curve = flat_curve();
        let measurements = vec![vec![curve.clone()], vec![curve.clone()]];
        let filters: Vec<Vec<Biquad>> = vec![Vec::new(), Vec::new()];
        let result = apply_per_sub_filters(&measurements, &filters, 48_000.0);
        assert_eq!(result[0][0].spl, curve.spl);
        assert_eq!(result[1][0].spl, curve.spl);
    }
}
