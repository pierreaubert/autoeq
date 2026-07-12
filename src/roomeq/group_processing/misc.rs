use super::super::multiseat::{self, MultiSeatMeasurements};
use super::super::types::{MultiMeasurementConfig, MultiSeatConfig, MultiSubGroup};
use crate::Curve;
use crate::error::{AutoeqError, Result};
use crate::read as load;
use crate::response;
use math_audio_dsp::analysis::compute_average_response;
use math_audio_iir_fir::Biquad;
use ndarray::Array1;

pub(super) const GLOBAL_EQ_REGRESSION_TOLERANCE: f64 = 1e-6;

// Re-export shared crossover helpers from the sibling `crossover_utils` module
// so group_processing does not duplicate them.
pub(super) use super::super::crossover_utils::{
    compute_lr24_crossover_responses, split_curve_at_frequency,
};

pub(super) fn load_multisub_seat_measurements(
    group: &MultiSubGroup,
) -> Result<Option<Vec<Vec<Curve>>>> {
    let mut per_sub = Vec::with_capacity(group.subwoofers.len());
    let mut expected_seats = None;
    let mut any_multi_seat = false;

    for (sub_idx, source) in group.subwoofers.iter().enumerate() {
        let curves =
            load::load_source_individual(source).map_err(|e| AutoeqError::InvalidMeasurement {
                message: format!(
                    "Failed to load seat measurements for sub {} in group '{}': {}",
                    sub_idx, group.name, e
                ),
            })?;
        if curves.len() > 1 {
            any_multi_seat = true;
        }
        match expected_seats {
            Some(expected) if curves.len() != expected => {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "Multi-seat multi-sub group '{}' has inconsistent seat counts: sub 0 has {}, sub {} has {}",
                        group.name,
                        expected,
                        sub_idx,
                        curves.len()
                    ),
                });
            }
            None => expected_seats = Some(curves.len()),
            _ => {}
        }
        per_sub.push(curves);
    }

    if any_multi_seat && expected_seats.unwrap_or(0) >= 2 {
        Ok(Some(per_sub))
    } else {
        Ok(None)
    }
}

pub(super) fn multiseat_peq_config(
    policy: &MultiSeatConfig,
    seat_count: usize,
) -> MultiMeasurementConfig {
    let mut weights = match policy.seat_weights.as_ref() {
        Some(weights) if weights.len() == seat_count => weights.clone(),
        _ => vec![1.0; seat_count],
    };
    for weight in &mut weights {
        if !weight.is_finite() || *weight < 0.0 {
            *weight = 0.0;
        }
    }
    if policy.strategy == super::super::types::MultiSeatStrategy::PrimaryWithConstraints
        && policy.primary_seat < weights.len()
    {
        weights[policy.primary_seat] *= policy.primary_seat_weight.max(1.0);
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
        spatial_robustness: Some(super::super::types::SpatialRobustnessSerdeConfig {
            variance_threshold_db: 3.0,
            transition_width_db: 2.0,
            min_correction_depth: 0.1,
            mask_smoothing_octaves: 1.0 / 6.0,
        }),
        bootstrap_uncertainty: None,
        rir_prototype: None,
    }
}

pub(super) fn apply_per_sub_filters(
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
                        let resp = response::compute_peq_complex_response(
                            filters,
                            &curve.freq,
                            sample_rate,
                        );
                        response::apply_complex_response(curve, &resp)
                    }
                })
                .collect()
        })
        .collect()
}

pub(super) fn average_power_curve(curves: &[Curve]) -> Result<Curve> {
    let Some(first) = curves.first() else {
        return Err(AutoeqError::InvalidMeasurement {
            message: "Cannot average an empty multi-seat curve set".to_string(),
        });
    };
    for (idx, curve) in curves.iter().enumerate() {
        if curve.freq.len() != first.freq.len()
            || curve
                .freq
                .iter()
                .zip(first.freq.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6 * b.abs().max(1.0))
        {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!(
                    "Cannot average multi-seat curves because seat {} has a different frequency grid",
                    idx
                ),
            });
        }
    }

    // Spatial aggregation is an energy average. Phase from different seats is
    // not a coherent transfer function and must never feed temporal alignment.
    let mut power_sum = Array1::<f64>::zeros(first.freq.len());
    for curve in curves {
        power_sum = power_sum + curve.spl.mapv(|spl| 10.0_f64.powf(spl / 10.0));
    }
    let avg_power = power_sum / curves.len() as f64;
    Ok(Curve {
        freq: first.freq.clone(),
        spl: avg_power.mapv(|power| 10.0 * power.max(1e-12).log10()),
        phase: None,
        ..Default::default()
    })
}

pub(super) fn flat_loss_score(curve: &Curve, min_freq: f64, max_freq: f64) -> f64 {
    let freqs_f32: Vec<f32> = curve.freq.iter().map(|&f| f as f32).collect();
    let spl_f32: Vec<f32> = curve.spl.iter().map(|&s| s as f32).collect();
    let mean = compute_average_response(
        &freqs_f32,
        &spl_f32,
        Some((min_freq as f32, max_freq as f32)),
    ) as f64;
    let normalized_spl = &curve.spl - mean;
    crate::loss::flat_loss(&curve.freq, &normalized_spl, min_freq, max_freq)
}

pub(super) fn eq_score_regressed(pre_score: f64, post_score: f64) -> bool {
    !post_score.is_finite()
        || (pre_score.is_finite() && post_score > pre_score + GLOBAL_EQ_REGRESSION_TOLERANCE)
}

pub(super) fn identity_multiseat_result(
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
