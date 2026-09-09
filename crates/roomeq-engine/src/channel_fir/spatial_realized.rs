//! Nonlinear FIR residual search in a convex basis of desired dB corrections.
//! Every candidate is realized before scoring; coefficients are never mixed.

use super::*;
use autoeq_core::Curve;
use autoeq_optim::optim::{
    compute_response_fitness,
    scalar::ScalarOptimConfig,
};
use num_complex::Complex64;

pub(super) fn optimize(
    request: &FirChannelRequest<'_>,
    curve: &Curve,
    filters: &[Biquad],
    progress: &progress::FirProgress,
) -> Result<(Vec<f64>, OptimizerRunEvidence)> {
    let sample_rate = request.sample_rate;
    let kirkeby = request
        .optimizer
        .fir
        .as_ref()
        .is_some_and(|fir| fir.phase.eq_ignore_ascii_case("kirkeby"));
    let phase_label = if kirkeby {
        "kirkeby reference-phase"
    } else {
        "minimum-phase"
    };
    let fail = |message: String| AutoeqError::OptimizationFailed { message };
    let measurements = request.prepared.measurements();
    let curves = crate::channel_optimizer::apply_representative_preprocessing_to_individuals(
        measurements.representative(),
        measurements.individual(),
        curve,
    );
    let (objective, _, effective) = crate::eq::prepare_multi_measurement_objective(
        &curves,
        request.optimizer,
        request.optimizer.multi_measurement.as_ref().unwrap(),
        Some(request.eq_resources),
        sample_rate,
    )
    .map_err(|error| fail(format!("Minimum-phase FIR objective: {error}")))?;
    let bank = &objective.multi_objective.as_ref().unwrap().objectives;
    let grid = bank[0].freqs.as_ref();
    if bank.iter().any(|seat| seat.freqs.as_ref() != grid) {
        return Err(fail(format!(
            "{phase_label} spatial FIR needs aligned objective grids"
        )));
    }
    let iir = response::compute_peq_complex_response(filters, grid, sample_rate);
    let mut basis: Vec<Array1<f64>> = bank
        .iter()
        .map(|seat| {
            Array1::from_iter(
                seat.deviation
                    .iter()
                    .zip(&iir)
                    .map(|(deviation, h)| deviation - 20.0 * h.norm().max(1e-20).log10()),
            )
        })
        .collect();
    basis.push(Array1::zeros(grid.len()));
    let neutral = if kirkeby {
        if &curve.freq != grid {
            return Err(fail(
                "Kirkeby spatial FIR needs its phase-reference curve on the objective grid".into(),
            ));
        }
        if request.optimizer.fir.as_ref().unwrap().correct_excess_phase && curve.phase.is_none() {
            return Err(fail(
                "Kirkeby excess-phase correction requires acoustic phase on the reference curve"
                    .into(),
            ));
        }
        let mut residual = response::apply_complex_response(curve, &iir);
        if curve.phase.is_none() {
            residual.phase = None;
        }
        residual
    } else {
        Curve {
            freq: grid.clone(),
            spl: Array1::zeros(grid.len()),
            ..Default::default()
        }
    };
    let realize = |weights: &[f64]| -> Option<Vec<f64>> {
        let sum: f64 = weights.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return None;
        }
        let target = Curve {
            freq: grid.clone(),
            spl: Array1::from_iter((0..grid.len()).map(|bin| {
                basis
                    .iter()
                    .zip(weights)
                    .map(|(target, weight)| target[bin] * (weight / sum))
                    .sum::<f64>()
                    + neutral.spl[bin]
            })),
            ..Default::default()
        };
        crate::fir::generate_fir_correction_prepared(&neutral, &effective, &target, sample_rate)
            .ok()
    };
    let evaluations = std::sync::atomic::AtomicUsize::new(0);
    let evaluate = |weights: &[f64]| {
        evaluations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Some(taps) = realize(weights) else {
            return f64::INFINITY;
        };
        let correction = Array1::from_iter(grid.iter().zip(&iir).map(|(frequency, iir)| {
            let fir = horner_response(&taps, *frequency, sample_rate);
            20.0 * (fir * iir).norm().max(1e-20).log10()
        }));
        compute_response_fitness(&vec![correction; bank.len()], &objective).unwrap_or(f64::INFINITY)
    };
    let count = basis.len();
    let mut best = vec![0.0; count];
    let mut loss = f64::INFINITY;
    for index in 0..count {
        let mut weights = vec![0.0; count];
        weights[index] = 1.0;
        let value = evaluate(&weights);
        progress.check(index, value)?;
        if value < loss {
            loss = value;
            best = weights;
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
        .map_err(|error| fail(format!("Minimum-phase FIR search: {error}")))?;
    let returned = evaluate(&result.x);
    let selected_search = returned.is_finite() && returned < loss;
    if selected_search {
        loss = returned;
        best = result.x;
    }
    let coefficients = realize(&best)
        .filter(|taps| !taps.is_empty() && taps.iter().all(|value| value.is_finite()))
        .ok_or_else(|| fail("Minimum-phase FIR search produced invalid coefficients".into()))?;
    if !loss.is_finite() {
        return Err(fail(
            "Minimum-phase FIR search has no finite candidate".into(),
        ));
    }
    let status = format!(
        "{}hybrid {} realized dB-basis search; {} templates; {} taps; selected={}; backend_objective={}; recomputed_backend_objective={}; {}",
        if result.success {
            ""
        } else {
            "not converged: "
        },
        phase_label,
        count,
        coefficients.len(),
        if selected_search {
            "bounded_search"
        } else {
            "basis_vertex"
        },
        result.fun,
        returned,
        result.message
    );
    let mut evidence = OptimizerRunEvidence::from_backend_result(
        &result.algorithm,
        Ok((status, loss)),
        &best,
        &vec![0.0; count],
        &vec![1.0; count],
        effective.max_iter,
        effective.seed,
    );
    evidence.evaluation_count = Some(evaluations.load(std::sync::atomic::Ordering::Relaxed));
    Ok((coefficients, evidence))
}

/// Evaluate H(z) by Horner's rule on the unit circle. Same finite polynomial
/// as a direct DTFT, without evaluating a transcendental function per tap.
fn horner_response(taps: &[f64], frequency: f64, sample_rate: f64) -> Complex64 {
    let z = Complex64::from_polar(1.0, -std::f64::consts::TAU * frequency / sample_rate);
    taps.iter()
        .rev()
        .fold(Complex64::new(0.0, 0.0), |acc, tap| acc * z + *tap)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn horner_matches_direct_dtft_for_long_causal_filters() {
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let mut taps: Vec<_> = (0..8192)
                .map(|i| (i as f64 * 0.37).sin() * (-0.002 * i as f64).exp() / 100.0)
                .collect();
            taps[8191] += 0.5;
            for frequency in [20.0, 120.0, 500.0, 0.46 * rate] {
                let direct: Complex64 = taps
                    .iter()
                    .enumerate()
                    .map(|(i, tap)| {
                        Complex64::from_polar(
                            *tap,
                            -std::f64::consts::TAU * frequency * i as f64 / rate,
                        )
                    })
                    .sum();
                assert!((horner_response(&taps, frequency, rate) - direct).norm() < 1e-10);
            }
        }
    }
}
