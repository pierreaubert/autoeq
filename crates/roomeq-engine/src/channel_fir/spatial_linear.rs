//! Finite-tap, common-linear-phase FIR bank optimization against all seats.
//!
//! The feasible family is convex combinations of per-objective residual
//! designs, the representative design, and a neutral design. All have the
//! requested length and common phase construction. This is a bounded basis
//! search, not an unrestricted FIR optimum or a spatial/temporal guarantee.

use super::*;
use autoeq_core::Curve;
use autoeq_optim::optim::{
    compute_response_fitness,
    scalar::ScalarOptimConfig,
};
use num_complex::Complex64;

pub(super) fn optimize(
    request: &FirChannelRequest<'_>,
    optimization_curve: &Curve,
    filters: &[Biquad],
    representative: Vec<f64>,
    progress: &progress::FirProgress,
) -> Result<(Vec<f64>, OptimizerRunEvidence)> {
    let fail = |message: String| AutoeqError::OptimizationFailed { message };
    let measurements = request.prepared.measurements();
    let curves = crate::channel_optimizer::apply_representative_preprocessing_to_individuals(
        measurements.representative(),
        measurements.individual(),
        optimization_curve,
    );
    let (objective, _, effective) = crate::eq::prepare_multi_measurement_objective(
        &curves,
        request.optimizer,
        request.optimizer.multi_measurement.as_ref().unwrap(),
        Some(request.eq_resources),
        request.sample_rate,
    )
    .map_err(|error| fail(format!("Hybrid FIR objective preparation: {error}")))?;
    let bank = &objective.multi_objective.as_ref().unwrap().objectives;
    let iir: Vec<_> = bank
        .iter()
        .map(|seat| {
            response::compute_peq_complex_response(filters, &seat.freqs, request.sample_rate)
        })
        .collect();
    let mut candidates = vec![representative];
    for (seat, transfer) in bank.iter().zip(&iir) {
        let input = Curve {
            freq: seat.freqs.as_ref().clone(),
            spl: Array1::zeros(seat.freqs.len()),
            ..Default::default()
        };
        let target = Curve {
            freq: input.freq.clone(),
            spl: Array1::from_iter(
                seat.deviation
                    .iter()
                    .zip(transfer)
                    .map(|(deviation, h)| deviation - 20.0 * h.norm().max(1e-20).log10()),
            ),
            ..Default::default()
        };
        candidates.push(
            crate::fir::generate_fir_correction_prepared(
                &input,
                &effective,
                &target,
                request.sample_rate,
            )
            .map_err(|error| fail(format!("Hybrid FIR seat design: {error}")))?,
        );
    }
    let neutral = Curve {
        freq: optimization_curve.freq.clone(),
        spl: Array1::zeros(optimization_curve.freq.len()),
        ..Default::default()
    };
    candidates.push(
        crate::fir::generate_fir_correction_prepared(
            &neutral,
            &effective,
            &neutral,
            request.sample_rate,
        )
        .map_err(|error| fail(format!("Hybrid FIR neutral design: {error}")))?,
    );
    let taps = candidates[0].len();
    if candidates
        .iter()
        .any(|candidate| candidate.len() != taps || candidate.iter().any(|v| !v.is_finite()))
    {
        return Err(fail(
            "Hybrid FIR bank has inconsistent or invalid coefficients".into(),
        ));
    }
    // Exact finite-record DTFT: H(f) = sum_n h[n] exp(-j 2 pi f n / fs).
    // Cache identical grids; bootstrap objectives usually share one grid.
    let mut distinct: Vec<(Array1<f64>, Vec<Vec<Complex64>>)> = Vec::new();
    let mut grid_indices = Vec::new();
    for seat in bank {
        let index = distinct
            .iter()
            .position(|(grid, _)| grid == seat.freqs.as_ref())
            .unwrap_or_else(|| {
                let responses = candidates
                    .iter()
                    .map(|candidate| {
                        response::compute_fir_complex_response(
                            candidate,
                            &seat.freqs,
                            request.sample_rate,
                        )
                    })
                    .collect();
                distinct.push((seat.freqs.as_ref().clone(), responses));
                distinct.len() - 1
            });
        grid_indices.push(index);
    }
    let evaluations = std::sync::atomic::AtomicUsize::new(0);
    let evaluate = |weights: &[f64]| {
        evaluations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let sum: f64 = weights.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return f64::INFINITY;
        }
        // The electrical FIR is shared by every seat. Mix it once per
        // distinct grid, including when many bootstrap objectives share it.
        let mixed: Vec<Vec<Complex64>> = distinct
            .iter()
            .map(|(grid, candidates)| {
                (0..grid.len())
                    .map(|bin| {
                        candidates
                            .iter()
                            .zip(weights)
                            .map(|(candidate, weight)| candidate[bin] * (weight / sum))
                            .sum()
                    })
                    .collect()
            })
            .collect();
        let responses: Vec<_> = bank
            .iter()
            .enumerate()
            .map(|(seat_index, seat)| {
                let mixed = &mixed[grid_indices[seat_index]];
                Array1::from_iter((0..seat.freqs.len()).map(|bin| {
                    let fir = mixed[bin];
                    20.0 * (fir * iir[seat_index][bin]).norm().max(1e-20).log10()
                }))
            })
            .collect();
        compute_response_fitness(&responses, &objective).unwrap_or(f64::INFINITY)
    };
    // Explicit vertices prevent a numerical search from discarding a known
    // better candidate, including neutral correction for an already-flat seat.
    let count = candidates.len();
    let mut best = vec![0.0; count];
    let mut best_loss = f64::INFINITY;
    for index in 0..count {
        let mut vertex = vec![0.0; count];
        vertex[index] = 1.0;
        let loss = evaluate(&vertex);
        progress.check(index, loss)?;
        if loss < best_loss {
            best_loss = loss;
            best = vertex;
        }
    }
    let config = ScalarOptimConfig {
        algorithm: effective.algorithm.clone(),
        max_iter: effective.max_iter,
        population: effective.population,
        strategy: effective.strategy.clone(),
        seed: effective.seed,
        ..Default::default()
    };
    let result = progress.optimize(&vec![(0.0, 1.0); count], &best, &config, evaluate)
        .map_err(|error| fail(format!("Hybrid FIR basis optimizer: {error}")))?;
    let returned_loss = evaluate(&result.x);
    let selected_search = returned_loss.is_finite() && returned_loss < best_loss;
    if selected_search {
        best_loss = returned_loss;
        best = result.x;
    }
    if !best_loss.is_finite() {
        return Err(fail(
            "Hybrid FIR has no finite shared-objective candidate".into(),
        ));
    }
    let sum: f64 = best.iter().sum();
    let coefficients = (0..taps)
        .map(|tap| {
            candidates
                .iter()
                .zip(&best)
                .map(|(candidate, weight)| candidate[tap] * (weight / sum))
                .sum()
        })
        .collect();
    let status = format!(
        "{}hybrid linear FIR convex-bank search; {} candidates; {} taps; selected={}; backend_objective={}; recomputed_backend_objective={}; configured_iteration_or_evaluation_budget={}; {}",
        if result.success {
            ""
        } else {
            "not converged: "
        },
        count,
        taps,
        if selected_search {
            "bounded_search"
        } else {
            "bank_vertex"
        },
        result.fun,
        returned_loss,
        effective.max_iter,
        result.message
    );
    let mut evidence = OptimizerRunEvidence::from_backend_result(
        &result.algorithm,
        Ok((status, best_loss)),
        &best,
        &vec![0.0; count],
        &vec![1.0; count],
        effective.max_iter,
        effective.seed,
    );
    evidence.evaluation_count = Some(evaluations.load(std::sync::atomic::Ordering::Relaxed));
    Ok((coefficients, evidence))
}
