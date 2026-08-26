use super::super::constraints::{
    viol_ceiling_from_spl, viol_min_gain_from_xs, viol_spacing_from_xs,
};
use super::super::loss::LossType;
use super::super::x2peq::x2spl_into;
use super::clamp::{clamp_cuts_to_envelope, clamp_envelopes_into, clamp_gains_to_envelope};
use super::loss::ObjectiveContext;
use super::objective_data::ObjectiveData;
use super::smoothness_penalty_config::SmoothnessPenaltyConfig;
use super::types::MultiObjectiveData;
use crate::PeqModel;
use ndarray::Array1;
use std::cell::RefCell;
use std::sync::Arc;

#[cfg(test)]
thread_local! {
    static RESPONSE_COMPUTATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[inline]
fn record_response_computation() {
    #[cfg(test)]
    RESPONSE_COMPUTATIONS.with(|count| count.set(count.get() + 1));
}

struct EvaluationScratch {
    parameters: Vec<f64>,
    response_parameters: Vec<f64>,
    response_freqs: Option<Arc<Array1<f64>>>,
    response_srate: f64,
    response_peq_model: Option<PeqModel>,
    peq: crate::iir::Peq,
    response: Array1<f64>,
    filter_response: Array1<f64>,
    error: Array1<f64>,
    smoothed_error: Array1<f64>,
    multi_losses: Vec<f64>,
}

impl EvaluationScratch {
    fn new() -> Self {
        Self {
            parameters: Vec::new(),
            response_parameters: Vec::new(),
            response_freqs: None,
            response_srate: f64::NAN,
            response_peq_model: None,
            peq: crate::iir::Peq::new(),
            response: Array1::zeros(0),
            filter_response: Array1::zeros(0),
            error: Array1::zeros(0),
            smoothed_error: Array1::zeros(0),
            multi_losses: Vec::new(),
        }
    }

    fn ensure_frequency_len(&mut self, len: usize) {
        if self.response.len() != len {
            self.response = Array1::zeros(len);
            self.filter_response = Array1::zeros(len);
            self.error = Array1::zeros(len);
            self.smoothed_error = Array1::zeros(len);
        }
    }
}

thread_local! {
    static EVALUATION_SCRATCH: RefCell<EvaluationScratch> = RefCell::new(EvaluationScratch::new());
}

/// Compute multi-objective fitness across multiple measurement curves.
///
/// Each objective shares the same filter parameters `x` but evaluates against
/// a different measurement curve. The per-curve losses are combined according
/// to the configured strategy.
fn variance_penalized_loss(
    weighted_losses: impl IntoIterator<Item = (f64, f64)>,
    variance_lambda: f64,
) -> f64 {
    let mut total_weight = 0.0;
    let mut mean = 0.0;
    let mut sum_squared_deviation = 0.0;
    for (loss, weight) in weighted_losses {
        if weight <= 0.0 {
            continue;
        }
        total_weight += weight;
        let delta = loss - mean;
        mean += (weight / total_weight) * delta;
        sum_squared_deviation += weight * delta * (loss - mean);
    }
    if total_weight <= 0.0 {
        return f64::INFINITY;
    }
    let variance = sum_squared_deviation / total_weight;
    mean + variance_lambda * variance
}

fn compute_multi_objective_fitness(x: &[f64], mo: &MultiObjectiveData) -> f64 {
    use crate::roomeq::MultiMeasurementStrategy;

    match mo.strategy {
        MultiMeasurementStrategy::Average => {
            let sum: f64 = mo
                .objectives
                .iter()
                .map(|objective| compute_base_fitness_single(x, objective))
                .sum();
            sum / mo.objectives.len() as f64
        }
        MultiMeasurementStrategy::WeightedSum => mo
            .objectives
            .iter()
            .zip(&mo.weights)
            .map(|(objective, weight)| compute_base_fitness_single(x, objective) * weight)
            .sum(),
        MultiMeasurementStrategy::Minimax => mo
            .objectives
            .iter()
            .map(|objective| compute_base_fitness_single(x, objective))
            .fold(f64::NEG_INFINITY, f64::max),
        MultiMeasurementStrategy::VariancePenalized => variance_penalized_loss(
            mo.objectives
                .iter()
                .zip(&mo.weights)
                .map(|(objective, weight)| (compute_base_fitness_single(x, objective), *weight)),
            mo.variance_lambda,
        ),
        MultiMeasurementStrategy::SpatialRobustness => {
            // Engine adapters map SpatialRobustness to direct per-seat
            // VariancePenalized risk before constructing this optimizer payload.
            unreachable!("unadapted SpatialRobustness strategy reached optimizer")
        }
        MultiMeasurementStrategy::MinimaxUncertainty => {
            // The bootstrap-resampled objectives have already been materialised
            // into `mo.objectives` at setup time. Either take the max (pure
            // worst-case) or the mean of the worst α-tail (CVaR).
            match mo.uncertainty_cvar_alpha {
                None => mo
                    .objectives
                    .iter()
                    .map(|objective| compute_base_fitness_single(x, objective))
                    .fold(f64::NEG_INFINITY, f64::max),
                Some(alpha) => {
                    let alpha = alpha.clamp(f64::MIN_POSITIVE, 1.0);
                    let mut sorted = EVALUATION_SCRATCH
                        .with(|slot| std::mem::take(&mut slot.borrow_mut().multi_losses));
                    sorted.clear();
                    for objective in &mo.objectives {
                        sorted.push(compute_base_fitness_single(x, objective));
                    }
                    // Worst losses first.
                    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let n = (alpha * sorted.len() as f64).ceil() as usize;
                    let n = n.clamp(1, sorted.len());
                    let result = sorted.iter().take(n).sum::<f64>() / n as f64;
                    EVALUATION_SCRATCH.with(|slot| {
                        slot.borrow_mut().multi_losses = sorted;
                    });
                    result
                }
            }
        }
    }
}

/// Compute the objective vector used by Pareto optimizers.
///
/// For multi-measurement data this returns the per-measurement losses before
/// scalarisation, with any shared AutoEQ penalty added to each component. For
/// ordinary scalar objectives it returns a one-element vector containing the
/// penalised scalar loss.
pub fn compute_pareto_objectives(x: &[f64], data: &ObjectiveData) -> Vec<f64> {
    if let Some(ref mo) = data.multi_objective {
        let base_scalar = compute_multi_objective_fitness(x, mo);
        let penalized_scalar = compute_fitness_penalties_ref(x, data);
        let shared_penalty = (penalized_scalar - base_scalar).max(0.0);
        return mo
            .objectives
            .iter()
            .map(|obj| compute_base_fitness_single(x, obj) + shared_penalty)
            .collect();
    }

    vec![compute_fitness_penalties_ref(x, data)]
}

/// Compute second-difference L1 (or Lp) penalty on cascaded magnitude in
/// log-frequency. Returns 0.0 when disabled or under-sampled.
pub fn compute_smoothness_penalty(
    y: &Array1<f64>,
    freqs: &Array1<f64>,
    min_freq: f64,
    max_freq: f64,
    cfg: &SmoothnessPenaltyConfig,
) -> f64 {
    if cfg.tv2_weight <= 0.0 || y.len() < 3 || freqs.len() != y.len() {
        return 0.0;
    }

    let mut acc = 0.0_f64;
    let mut active_terms = 0usize;
    for i in 1..(y.len() - 1) {
        let f_c = freqs[i];
        if f_c < min_freq
            || f_c > max_freq
            || f_c <= 0.0
            || freqs[i - 1] <= 0.0
            || freqs[i + 1] <= 0.0
        {
            continue;
        }

        let lf_p = freqs[i + 1].log10();
        let lf_c = f_c.log10();
        let lf_m = freqs[i - 1].log10();
        let dx_fwd = lf_p - lf_c;
        let dx_bwd = lf_c - lf_m;
        if dx_fwd <= 0.0 || dx_bwd <= 0.0 {
            continue;
        }

        let slope_fwd = (y[i + 1] - y[i]) / dx_fwd;
        let slope_bwd = (y[i] - y[i - 1]) / dx_bwd;
        let curvature = (slope_fwd - slope_bwd) / (0.5 * (dx_fwd + dx_bwd));
        let w = match cfg.schroeder_hz {
            Some(fs) if f_c < fs => cfg.modal_weight_scale,
            _ => 1.0,
        };
        let term = if (cfg.exponent - 1.0).abs() < 1e-9 {
            curvature.abs()
        } else if (cfg.exponent - 2.0).abs() < 1e-9 {
            curvature * curvature
        } else {
            curvature.abs().powf(cfg.exponent)
        };
        acc += w * term;
        active_terms += 1;
    }

    if active_terms == 0 {
        0.0
    } else {
        cfg.tv2_weight * acc / active_terms as f64
    }
}

/// Compute the base fitness for a single ObjectiveData (no multi-objective delegation).
/// This is the inner implementation that does not check `multi_objective`.
fn compute_base_fitness_single(x: &[f64], data: &ObjectiveData) -> f64 {
    if matches!(
        data.loss_type,
        LossType::SpeakerFlat | LossType::HeadphoneFlat | LossType::SpeakerFlatAsymmetric
    ) {
        return compute_prepared_flat_fitness(x, data);
    }

    let clamped_boost;
    let clamped_cut;
    let x = {
        let skip = matches!(
            data.loss_type,
            LossType::DriversFlat | LossType::MultiSubFlat
        );
        let x = if !skip && let Some(ref envelope) = data.max_boost_envelope {
            clamped_boost = clamp_gains_to_envelope(x, envelope, data.peq_model);
            &clamped_boost
        } else {
            x
        };
        if !skip && let Some(ref envelope) = data.min_cut_envelope {
            clamped_cut = clamp_cuts_to_envelope(x, envelope, data.peq_model);
            &clamped_cut
        } else {
            x
        }
    };

    let objective = data
        .objective
        .clone()
        .unwrap_or_else(|| data.build_objective());

    let ctx = ObjectiveContext {
        freqs: &data.freqs,
        target: &data.target,
        deviation: &data.deviation,
        srate: data.srate,
        peq_model: data.peq_model,
        min_freq: data.min_freq,
        max_freq: data.max_freq,
        smooth: data.smooth,
        smooth_n: data.smooth_n,
        audibility_deadband: data.audibility_deadband.as_ref(),
        smoothness_penalty: data.smoothness_penalty.as_ref(),
    };

    objective.compute(x, &ctx)
}

fn compute_prepared_flat_fitness(x: &[f64], data: &ObjectiveData) -> f64 {
    let prepared = data.prepared();
    EVALUATION_SCRATCH.with(|slot| {
        let mut scratch = slot.borrow_mut();
        scratch.ensure_frequency_len(data.freqs.len());
        let EvaluationScratch {
            parameters,
            response_parameters,
            response_freqs,
            response_srate,
            response_peq_model,
            peq,
            response,
            filter_response,
            error,
            ..
        } = &mut *scratch;
        let x = clamp_envelopes_into(
            x,
            parameters,
            data.max_boost_envelope.as_deref(),
            data.min_cut_envelope.as_deref(),
            data.peq_model,
        );
        let response_is_cached = response_parameters.as_slice() == x
            && response_freqs
                .as_ref()
                .is_some_and(|freqs| Arc::ptr_eq(freqs, &data.freqs))
            && *response_srate == data.srate
            && *response_peq_model == Some(data.peq_model);
        if !response_is_cached {
            x2spl_into(
                &data.freqs,
                x,
                data.srate,
                data.peq_model,
                peq,
                response,
                filter_response,
            );
            record_response_computation();
            response_parameters.clear();
            response_parameters.extend_from_slice(x);
            *response_freqs = Some(data.freqs.clone());
            *response_srate = data.srate;
            *response_peq_model = Some(data.peq_model);
        }
        for ((value, &correction), &deviation) in error
            .iter_mut()
            .zip(response.iter())
            .zip(data.deviation.iter())
        {
            *value = correction - deviation;
        }
        prepared.apply_deadband(error.as_slice_mut().expect("contiguous error scratch"));
        let base = match data.loss_type {
            LossType::SpeakerFlat | LossType::HeadphoneFlat => prepared
                .flat
                .as_ref()
                .expect("prepared flat loss")
                .evaluate(error),
            LossType::SpeakerFlatAsymmetric => prepared
                .asymmetric
                .as_ref()
                .expect("prepared asymmetric loss")
                .evaluate(error),
            _ => unreachable!("prepared flat path only handles flat objectives"),
        };
        base + prepared
            .smoothness
            .as_ref()
            .map(|penalty| {
                penalty.evaluate(response.as_slice().expect("contiguous response scratch"))
            })
            .unwrap_or(0.0)
    })
}

/// Evaluate a PEQ ceiling constraint using the response left by the objective
/// on this worker, or the same reusable buffers on a cache miss.
pub(crate) fn compute_ceiling_violation_into(
    freqs: &Arc<Array1<f64>>,
    x: &[f64],
    srate: f64,
    peq_model: PeqModel,
    max_db: f64,
) -> f64 {
    EVALUATION_SCRATCH.with(|slot| {
        let mut scratch = slot.borrow_mut();
        scratch.ensure_frequency_len(freqs.len());
        let response_is_cached = scratch.response_parameters.as_slice() == x
            && scratch
                .response_freqs
                .as_ref()
                .is_some_and(|cached| Arc::ptr_eq(cached, freqs))
            && scratch.response_srate == srate
            && scratch.response_peq_model == Some(peq_model);
        if !response_is_cached {
            let EvaluationScratch {
                peq,
                response,
                filter_response,
                response_parameters,
                response_freqs,
                response_srate,
                response_peq_model,
                ..
            } = &mut *scratch;
            x2spl_into(freqs, x, srate, peq_model, peq, response, filter_response);
            record_response_computation();
            response_parameters.clear();
            response_parameters.extend_from_slice(x);
            *response_freqs = Some(freqs.clone());
            *response_srate = srate;
            *response_peq_model = Some(peq_model);
        }
        viol_ceiling_from_spl(&scratch.response, max_db, peq_model)
    })
}

/// Compute the base fitness value (without penalties) for given parameters
///
/// This is the unified fitness function used by both NLOPT and metaheuristics optimizers.
/// If `multi_objective` is set, delegates to multi-objective fitness computation.
pub fn compute_base_fitness(x: &[f64], data: &ObjectiveData) -> f64 {
    // If multi-objective data is present, delegate to multi-objective fitness
    if let Some(ref mo) = data.multi_objective {
        return compute_multi_objective_fitness(x, mo);
    }

    compute_base_fitness_single(x, data)
}

/// Compute objective function value including penalty terms for constraints
///
/// Non-mutating version used by optimizers that don't require `&mut` data
/// (e.g., metaheuristics). Avoids cloning ObjectiveData on every evaluation.
pub fn compute_fitness_penalties_ref(x: &[f64], data: &ObjectiveData) -> f64 {
    let fit = compute_base_fitness(x, data);

    // PEQ-specific penalties only apply when the parameter vector has PEQ layout
    // (freq/Q/gain triplets). DriversFlat and MultiSubFlat use a different layout
    // (gains/delays/crossovers) and these penalty functions would misinterpret the values.
    let is_peq_loss = !matches!(
        data.loss_type,
        LossType::DriversFlat | LossType::MultiSubFlat
    );

    // When penalties are enabled (weights > 0), add them to the base fit so that
    // optimizers without nonlinear constraints can still respect our limits.
    let mut penalized = fit;

    if data.penalty_w_ceiling > 0.0 && is_peq_loss {
        let viol =
            compute_ceiling_violation_into(&data.freqs, x, data.srate, data.peq_model, data.max_db);
        penalized += data.penalty_w_ceiling * viol * viol;
    }

    if data.penalty_w_spacing > 0.0 && is_peq_loss {
        let viol = viol_spacing_from_xs(x, data.peq_model, data.min_spacing_oct);
        penalized += data.penalty_w_spacing * viol * viol;
    }

    if data.penalty_w_mingain > 0.0 && data.min_db > 0.0 && is_peq_loss {
        let viol = viol_min_gain_from_xs(x, data.peq_model, data.min_db);
        penalized += data.penalty_w_mingain * viol * viol;
    }

    penalized
}

/// Compute objective function value including penalty terms for constraints
///
/// NLOPT-compatible wrapper that takes `&mut ObjectiveData` (required by NLOPT's callback
/// signature). Delegates to `compute_fitness_penalties_ref`.
///
/// # Arguments
/// * `x` - Parameter vector
/// * `_gradient` - Gradient vector (unused, for NLOPT compatibility)
/// * `data` - Objective data containing penalty weights and parameters
///
/// # Returns
/// Base fitness value plus weighted penalty terms
pub fn compute_fitness_penalties(
    x: &[f64],
    _gradient: Option<&mut [f64]>,
    data: &mut ObjectiveData,
) -> f64 {
    compute_fitness_penalties_ref(x, data)
}

/// Extract sorted center frequencies from parameter vector and compute adjacent spacings in octaves.
pub fn compute_sorted_freqs_and_adjacent_octave_spacings(
    x: &[f64],
    peq_model: PeqModel,
) -> (Vec<f64>, Vec<f64>) {
    let n = crate::param_utils::num_filters(x, peq_model);
    let mut freqs: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let params = crate::param_utils::get_filter_params(x, i, peq_model);
        freqs.push(10f64.powf(params.freq));
    }
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let spacings: Vec<f64> = if freqs.len() < 2 {
        Vec::new()
    } else {
        freqs
            .windows(2)
            .map(|w| (w[1].max(1e-9) / w[0].max(1e-9)).log2().abs())
            .collect()
    };
    (freqs, spacings)
}

#[cfg(test)]
mod spacing_diag_tests {

    use super::super::compute_sorted_freqs_and_adjacent_octave_spacings;

    #[test]
    fn adjacent_octave_spacings_basic() {
        // x: [f,q,g, f,q,g, f,q,g]
        let x = [
            100f64.log10(),
            1.0,
            0.0,
            200f64.log10(),
            1.0,
            0.0,
            400f64.log10(),
            1.0,
            0.0,
        ];
        use crate::PeqModel;
        let (_freqs, spacings) =
            compute_sorted_freqs_and_adjacent_octave_spacings(&x, PeqModel::Pk);
        assert!((spacings[0] - 1.0).abs() < 1e-12);
        assert!((spacings[1] - 1.0).abs() < 1e-12);
    }
}

#[cfg(test)]
mod smoothness_penalty_edge_tests {
    use super::{SmoothnessPenaltyConfig, compute_smoothness_penalty};
    use ndarray::Array1;

    #[test]
    fn smoothness_penalty_short_array_returns_zero() {
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let freqs = Array1::from_vec(vec![100.0, 200.0]);
        let cfg = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            ..Default::default()
        };
        assert_eq!(
            compute_smoothness_penalty(&y, &freqs, 20.0, 20000.0, &cfg),
            0.0
        );
    }

    #[test]
    fn smoothness_penalty_mismatched_lengths_returns_zero() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let freqs = Array1::from_vec(vec![100.0, 200.0]);
        let cfg = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            ..Default::default()
        };
        assert_eq!(
            compute_smoothness_penalty(&y, &freqs, 20.0, 20000.0, &cfg),
            0.0
        );
    }

    #[test]
    fn smoothness_penalty_negative_or_zero_freqs_skipped() {
        let freqs = Array1::from_vec(vec![0.0, 100.0, 200.0, 300.0]);
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0]);
        let cfg = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            ..Default::default()
        };
        let p = compute_smoothness_penalty(&y, &freqs, 20.0, 20000.0, &cfg);
        assert!(p.is_finite() && p >= 0.0);
    }

    #[test]
    fn smoothness_penalty_is_invariant_to_log_grid_density() {
        let evaluate = |count| {
            let frequencies = Array1::<f64>::logspace(10.0, 1.0, 4.0, count);
            let response = frequencies.mapv(|frequency| frequency.ln().powi(2));
            compute_smoothness_penalty(
                &response,
                &frequencies,
                10.0,
                10_000.0,
                &SmoothnessPenaltyConfig {
                    tv2_weight: 1.0,
                    exponent: 2.0,
                    ..Default::default()
                },
            )
        };

        let coarse = evaluate(17);
        let dense = evaluate(257);
        assert!(coarse > 0.0 && dense > 0.0);
        assert!(
            (coarse - dense).abs() <= 1e-10 * coarse.max(dense),
            "coarse={coarse}, dense={dense}"
        );
    }

    #[test]
    fn smoothness_penalty_custom_exponent() {
        let n = 50;
        let freqs = Array1::<f64>::logspace(10.0, 1.0, 4.0, n);
        let y = freqs.mapv(|f| 2.0 * (f.log10() * 10.0).sin());
        let cfg1 = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            exponent: 1.5,
            ..Default::default()
        };
        let cfg2 = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            exponent: 2.0,
            ..Default::default()
        };
        let p1 = compute_smoothness_penalty(&y, &freqs, 20.0, 20000.0, &cfg1);
        let p2 = compute_smoothness_penalty(&y, &freqs, 20.0, 20000.0, &cfg2);
        assert!(p1 > 0.0 && p2 > 0.0);
        assert!(
            p2 > p1 * 0.5,
            "exponent=2.0 should produce comparable or larger penalty than 1.5"
        );
    }

    #[test]
    fn smoothness_penalty_nonuniform_dx_skipped() {
        // Non-monotonic log-freq grid (dx_fwd <= 0 or dx_bwd <= 0)
        let freqs = Array1::from_vec(vec![100.0, 200.0, 150.0, 400.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 0.5, 0.0]);
        let cfg = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            ..Default::default()
        };
        let p = compute_smoothness_penalty(&y, &freqs, 20.0, 20000.0, &cfg);
        assert!(p.is_finite() && p >= 0.0);
    }

    #[test]
    fn smoothness_penalty_schroeder_boundary() {
        // Create a curve with a sharp bend at exactly 200 Hz (the Schroeder boundary)
        let n = 100;
        let freqs = Array1::logspace(10.0, 1.0, 4.0, n);
        let y = freqs.mapv(|f: f64| {
            let ratio = f / 200.0_f64;
            if f < 200.0_f64 {
                5.0_f64 * ratio.log10()
            } else {
                -5.0_f64 * ratio.log10()
            }
        });
        let cfg_modal = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            schroeder_hz: Some(200.0),
            modal_weight_scale: 0.0,
            exponent: 1.0,
        };
        let cfg_strict = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            schroeder_hz: None,
            modal_weight_scale: 1.0,
            exponent: 1.0,
        };
        let p_modal = compute_smoothness_penalty(&y, &freqs, 20.0, 20000.0, &cfg_modal);
        let p_strict = compute_smoothness_penalty(&y, &freqs, 20.0, 20000.0, &cfg_strict);
        assert!(
            p_modal <= p_strict,
            "modal exemption at schroeder boundary should reduce or equal penalty: modal={p_modal}, strict={p_strict}"
        );
    }

    #[test]
    fn smoothness_penalty_outside_range_returns_zero() {
        let freqs = Array1::from_vec(vec![100.0, 200.0, 300.0, 400.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let cfg = SmoothnessPenaltyConfig {
            tv2_weight: 1.0,
            ..Default::default()
        };
        // min_freq > max_freq in data range → no points evaluated
        let p = compute_smoothness_penalty(&y, &freqs, 500.0, 600.0, &cfg);
        assert_eq!(p, 0.0, "outside range should produce zero penalty");
    }
}

#[cfg(test)]
mod multi_objective_and_base_fitness_tests {
    use super::{
        EVALUATION_SCRATCH, ObjectiveData, RESPONSE_COMPUTATIONS, compute_base_fitness,
        compute_base_fitness_single, compute_fitness_penalties, compute_fitness_penalties_ref,
        compute_multi_objective_fitness, compute_pareto_objectives,
        compute_sorted_freqs_and_adjacent_octave_spacings, variance_penalized_loss,
    };
    use crate::MultiObjectiveData;
    use crate::ObjectiveDataBuilder;
    use crate::PeqModel;
    use crate::loss::epa::score::EpaConfig;
    use crate::loss::{HeadphoneLossData, LossType, SpeakerLossData};
    use crate::roomeq::MultiMeasurementStrategy;
    use ndarray::Array1;
    use std::sync::Arc;

    fn freqs() -> Array1<f64> {
        Array1::from_vec(vec![100.0, 200.0, 400.0, 800.0, 1600.0])
    }

    #[test]
    fn two_seat_zero_ten_db_risk_prefers_documented_midpoint() {
        let unbalanced = variance_penalized_loss([(0.0, 0.5), (10.0, 0.5)], 1.0);
        let balanced = variance_penalized_loss([(5.0, 0.5), (5.0, 0.5)], 1.0);
        assert!((unbalanced - 30.0).abs() < 1e-12);
        assert!((balanced - 5.0).abs() < 1e-12);
        assert!(balanced < unbalanced);
    }

    #[test]
    fn variance_penalized_risk_honors_seat_weights() {
        let equal = variance_penalized_loss([(0.0, 0.5), (10.0, 0.5)], 0.0);
        let primary_weighted = variance_penalized_loss([(0.0, 0.9), (10.0, 0.1)], 0.0);
        assert!((equal - 5.0).abs() < 1e-12);
        assert!((primary_weighted - 1.0).abs() < 1e-12);
    }

    #[test]
    fn prepared_flat_hot_path_does_not_smooth_signed_error_before_loss() {
        let mut alternating = base_objective(LossType::SpeakerFlat);
        alternating.smooth = true;
        alternating.smooth_n = 2;
        alternating.deviation = Arc::new(Array1::from_vec(vec![6.0, -6.0, 6.0, -6.0, 6.0]));

        let mut constant = base_objective(LossType::SpeakerFlat);
        constant.smooth = true;
        constant.smooth_n = 2;
        constant.deviation = Arc::new(Array1::from_elem(5, 6.0));

        let alternating_loss = compute_base_fitness_single(&x(), &alternating);
        let constant_loss = compute_base_fitness_single(&x(), &constant);
        assert!((alternating_loss - constant_loss).abs() < 1e-12);
    }

    fn x() -> Vec<f64> {
        // one peak filter at 500 Hz, Q=1, gain=0 (neutral)
        vec![500f64.log10(), 1.0, 0.0]
    }

    fn base_objective(loss_type: LossType) -> ObjectiveData {
        let f = freqs();
        let n = f.len();
        ObjectiveDataBuilder::new(
            f,
            Array1::from_elem(n, 80.0),
            Array1::from_elem(n, 5.0),
            48000.0,
            PeqModel::Pk,
            loss_type,
        )
        .min_spacing_oct(0.5)
        .max_db(12.0)
        .min_db(0.0)
        .freq_range(20.0, 20000.0)
        .smoothing(false, 0)
        .build()
        .expect("valid base objective")
    }

    fn multi_objective(strategy: MultiMeasurementStrategy) -> MultiObjectiveData {
        let obj = base_objective(LossType::SpeakerFlat);
        MultiObjectiveData {
            objectives: vec![obj.clone(), obj.clone()],
            strategy,
            weights: vec![0.4, 0.6],
            variance_lambda: 0.5,
            uncertainty_cvar_alpha: Some(0.5),
        }
    }

    #[test]
    fn multi_objective_strategies() {
        let mo = multi_objective(MultiMeasurementStrategy::Average);
        let avg = compute_multi_objective_fitness(&x(), &mo);
        assert!(avg.is_finite());

        let mo = multi_objective(MultiMeasurementStrategy::WeightedSum);
        let ws = compute_multi_objective_fitness(&x(), &mo);
        assert!(ws.is_finite());

        let mo = multi_objective(MultiMeasurementStrategy::Minimax);
        let mm = compute_multi_objective_fitness(&x(), &mo);
        assert!(mm.is_finite());

        let mo = multi_objective(MultiMeasurementStrategy::VariancePenalized);
        let vp = compute_multi_objective_fitness(&x(), &mo);
        assert!(vp.is_finite());

        let mo = multi_objective(MultiMeasurementStrategy::MinimaxUncertainty);
        let mmu = compute_multi_objective_fitness(&x(), &mo);
        assert!(mmu.is_finite());
    }

    #[test]
    fn multi_objective_reuses_response_for_shared_frequency_grid() {
        let objective = base_objective(LossType::SpeakerFlat);
        let mo = MultiObjectiveData {
            objectives: vec![objective.clone(), objective.clone(), objective],
            weights: vec![1.0 / 3.0; 3],
            strategy: MultiMeasurementStrategy::WeightedSum,
            variance_lambda: 0.5,
            uncertainty_cvar_alpha: Some(0.5),
        };

        EVALUATION_SCRATCH.with(|slot| {
            let mut scratch = slot.borrow_mut();
            scratch.response_parameters.clear();
            scratch.response_freqs = None;
        });
        RESPONSE_COMPUTATIONS.with(|count| count.set(0));

        let loss = compute_multi_objective_fitness(&x(), &mo);

        assert!(loss.is_finite());
        RESPONSE_COMPUTATIONS.with(|count| {
            assert_eq!(
                count.get(),
                1,
                "one candidate response should serve every aligned measurement"
            );
        });
    }

    #[test]
    fn compute_pareto_objectives_scalar_and_multi() {
        let obj = base_objective(LossType::SpeakerFlat);
        let scalar = compute_pareto_objectives(&x(), &obj);
        assert_eq!(scalar.len(), 1);

        let mut multi = obj.clone();
        multi.multi_objective = Some(multi_objective(MultiMeasurementStrategy::WeightedSum));
        let vec = compute_pareto_objectives(&x(), &multi);
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn base_fitness_speaker_flat_and_asymmetric() {
        let obj = base_objective(LossType::SpeakerFlat);
        assert!(compute_base_fitness_single(&x(), &obj).is_finite());

        let asym = ObjectiveDataBuilder::speaker_flat_asymmetric(
            freqs(),
            Array1::from_elem(freqs().len(), 80.0),
            Array1::from_elem(freqs().len(), 5.0),
            48000.0,
            PeqModel::Pk,
        )
        .max_db(12.0)
        .min_db(0.0)
        .freq_range(20.0, 20000.0)
        .smoothing(false, 0)
        .null_suppression(Array1::from_elem(freqs().len(), 1.0))
        .build()
        .expect("valid asymmetric objective");
        assert!(compute_base_fitness_single(&x(), &asym).is_finite());
    }

    #[test]
    fn base_fitness_missing_data_rejected_by_builder() {
        let f = freqs();
        let n = f.len();
        let core = |loss_type| {
            ObjectiveDataBuilder::new(
                f.clone(),
                Array1::from_elem(n, 80.0),
                Array1::from_elem(n, 5.0),
                48000.0,
                PeqModel::Pk,
                loss_type,
            )
            .max_db(12.0)
            .min_db(0.0)
            .freq_range(20.0, 20000.0)
        };

        assert!(core(LossType::DriversFlat).build().is_err());
        assert!(core(LossType::MultiSubFlat).build().is_err());
        assert!(core(LossType::SpeakerScore).build().is_err());
        assert!(core(LossType::HeadphoneScore).build().is_err());
    }

    #[test]
    fn base_fitness_speaker_score_and_headphone_score() {
        let f = Array1::from_vec(vec![100.0, 1000.0, 10000.0]);
        let n = f.len();
        let x = vec![2.0, 1.0, 0.0];

        let speaker = ObjectiveDataBuilder::speaker_score(
            f.clone(),
            Array1::from_elem(n, 80.0),
            Array1::from_elem(n, 5.0),
            48000.0,
            PeqModel::Pk,
            SpeakerLossData {
                on: Array1::from_vec(vec![80.0, 85.0, 82.0]),
                lw: Array1::from_vec(vec![81.0, 84.0, 83.0]),
                sp: Array1::from_vec(vec![78.0, 82.0, 80.0]),
                pir: Array1::from_vec(vec![80.5, 84.0, 82.5]),
            },
        )
        .max_db(12.0)
        .min_db(0.0)
        .freq_range(20.0, 20000.0)
        .smoothing(false, 0)
        .build()
        .expect("valid speaker-score objective");
        assert!(compute_base_fitness_single(&x, &speaker).is_finite());

        let headphone = ObjectiveDataBuilder::headphone_score(
            f.clone(),
            Array1::from_elem(n, 80.0),
            Array1::from_elem(n, 5.0),
            48000.0,
            PeqModel::Pk,
            HeadphoneLossData::new(false, 0),
        )
        .input_curve(crate::Curve {
            freq: f.clone(),
            spl: Array1::from_elem(n, 80.0),
            phase: None,
            ..Default::default()
        })
        .max_db(12.0)
        .min_db(0.0)
        .freq_range(20.0, 20000.0)
        .smoothing(false, 0)
        .build()
        .expect("valid headphone-score objective");
        assert!(compute_base_fitness_single(&x, &headphone).is_finite());
    }

    #[test]
    fn base_fitness_epa() {
        let mut epa = base_objective(LossType::Epa);
        epa.epa_config = Some(EpaConfig::default());
        assert!(compute_base_fitness_single(&x(), &epa).is_finite());
    }

    #[test]
    fn base_fitness_multi_objective_delegation() {
        let mut obj = base_objective(LossType::SpeakerFlat);
        obj.multi_objective = Some(multi_objective(MultiMeasurementStrategy::Minimax));
        let loss = compute_base_fitness(&x(), &obj);
        assert!(loss.is_finite());
    }

    #[test]
    fn fitness_penalties_add_penalty_terms() {
        let mut obj = base_objective(LossType::SpeakerFlat);
        obj.penalty_w_ceiling = 1.0;
        obj.penalty_w_spacing = 1.0;
        obj.min_db = 1.0;
        obj.penalty_w_mingain = 1.0;
        let penalized = compute_fitness_penalties_ref(&x(), &obj);
        let base = compute_base_fitness(&x(), &obj);
        assert!(penalized >= base);

        // Drivers loss skips PEQ-specific penalties.
        obj.loss_type = LossType::DriversFlat;
        let drivers_penalized = compute_fitness_penalties_ref(&x(), &obj);
        assert!(
            drivers_penalized.is_infinite()
                || drivers_penalized == compute_base_fitness(&x(), &obj)
        );
    }

    #[test]
    fn fitness_penalties_wrapper_matches_ref() {
        let mut obj = base_objective(LossType::SpeakerFlat);
        let ref_val = compute_fitness_penalties_ref(&x(), &obj);
        let wrapped_val = compute_fitness_penalties(&x(), None, &mut obj);
        assert_eq!(ref_val, wrapped_val);
    }

    #[test]
    fn sorted_freqs_and_spacings_one_filter() {
        let (freqs, spacings) =
            compute_sorted_freqs_and_adjacent_octave_spacings(&x(), PeqModel::Pk);
        assert_eq!(freqs.len(), 1);
        assert!(spacings.is_empty());
    }

    #[test]
    fn sorted_freqs_and_spacings_two_filters() {
        // two peak filters at 100 Hz and 400 Hz
        let params = vec![100f64.log10(), 1.0, 0.0, 400f64.log10(), 1.0, 0.0];
        let (freqs, spacings) =
            compute_sorted_freqs_and_adjacent_octave_spacings(&params, PeqModel::Pk);
        assert_eq!(freqs.len(), 2);
        assert_eq!(spacings.len(), 1);
        assert!((spacings[0] - 2.0).abs() < 1e-12);
    }
}
