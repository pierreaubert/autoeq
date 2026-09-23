//! Nonlinear FIR residual search in a convex basis of desired dB corrections.
//! Every candidate is realized before scoring; coefficients are never mixed.

use super::*;
use autoeq_core::Curve;
use autoeq_optim::optim::{compute_response_fitness, scalar::ScalarOptimConfig};
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
    let curves = curves
        .iter()
        .map(|curve| {
            request
                .prepared
                .usable_curve(curve)
                .map(|curve| curve.into_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    let (objective, _, effective, normalization) =
        crate::eq::prepare_multi_measurement_objective_recorded(
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
        } else if request
            .optimizer
            .fir
            .as_ref()
            .is_some_and(|fir| fir.correct_excess_phase)
        {
            // A common arrival delay belongs to source alignment, not to the
            // excess-phase inverse. Remove only the fitted linear trend before
            // Kirkeby sees the reference phase; otherwise a short FIR spends
            // its support trying to cancel an acoustic delay and leaves the
            // position-dependent dispersion under-corrected.
            if let Some(phase) = residual.phase.take() {
                let unwrapped = autoeq_core::phase_utils::unwrap_phase_degrees(&phase);
                let (_, delay_free) = autoeq_core::phase_utils::estimate_delay_from_excess_phase(
                    &residual.freq,
                    &unwrapped,
                );
                residual.phase = Some(delay_free);
            }
        }
        residual
    } else {
        Curve {
            freq: grid.clone(),
            spl: Array1::zeros(grid.len()),
            ..Default::default()
        }
    };
    // A configured seat weight is a preference over the basis itself, not
    // only a scalarisation weight applied after every candidate is realised.
    // Keep zero-weight seats out of the generated target; otherwise the
    // nonlinear search can choose a correction derived entirely from a seat
    // the caller explicitly excluded.
    let configured_weights = request
        .optimizer
        .multi_measurement
        .as_ref()
        .and_then(|config| config.weights.as_ref())
        .map(|weights| {
            let sum = weights.iter().sum::<f64>();
            weights
                .iter()
                .map(|weight| *weight / sum)
                .collect::<Vec<_>>()
        });
    let realize = |weights: &[f64]| -> Option<Vec<f64>> {
        let effective_weights: Vec<f64> = weights
            .iter()
            .enumerate()
            .map(|(index, weight)| {
                let multiplier = configured_weights
                    .as_ref()
                    .and_then(|configured| configured.get(index))
                    .copied()
                    .unwrap_or(1.0);
                weight * multiplier
            })
            .collect();
        let sum: f64 = effective_weights.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return None;
        }
        let target = Curve {
            freq: grid.clone(),
            spl: Array1::from_iter((0..grid.len()).map(|bin| {
                basis
                    .iter()
                    .zip(&effective_weights)
                    .map(|(target, weight)| target[bin] * (weight / sum))
                    .sum::<f64>()
                    + neutral.spl[bin]
            })),
            ..Default::default()
        };
        let coefficients = if kirkeby
            && request
                .optimizer
                .fir
                .as_ref()
                .is_some_and(|fir| fir.correct_excess_phase)
        {
            // The Kirkeby implementation derives minimum phase from the
            // measurement magnitude.  The IIR stage has already accounted for
            // that magnitude, so use a flat phase reference here and pass the
            // desired correction as a relative dB curve.  This keeps the
            // measured (delay-free) excess phase without subtracting a second
            // minimum-phase component from the residual.
            let mut phase_measurement = neutral.clone();
            phase_measurement.spl.fill(0.0);
            let phase_target = Curve {
                freq: grid.clone(),
                spl: &target.spl - &neutral.spl,
                ..Default::default()
            };
            crate::fir::generate_fir_correction_prepared(
                &phase_measurement,
                &effective,
                &phase_target,
                sample_rate,
            )
        } else {
            crate::fir::generate_fir_correction_prepared(&neutral, &effective, &target, sample_rate)
        };
        coefficients.ok()
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
    let result = progress
        .optimize(&vec![(0.0, 1.0); count], &best, &config, evaluate)
        .map_err(|error| fail(format!("Minimum-phase FIR search: {error}")))?;
    let returned = evaluate(&result.x);
    let mut selected_search = returned.is_finite() && returned < loss;
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
    // Realized-transfer ceiling: every candidate is already realized before
    // scoring, so judge the emitted taps directly. A breaching winner
    // reverts to the neutral vertex with a recorded reason; refusal when
    // neutral breaches.
    let mut neutral_weights = vec![0.0; count];
    neutral_weights[count - 1] = 1.0;
    let neutral_taps = realize(&neutral_weights)
        .filter(|taps| !taps.is_empty() && taps.iter().all(|value| value.is_finite()))
        .ok_or_else(|| fail("Minimum-phase FIR neutral fallback is invalid".into()))?;
    let mut ceiling_note: Option<String> = None;
    let coefficients = match super::enforce_realized_fir_ceiling(
        "hybrid-realized-fir",
        coefficients,
        neutral_taps,
        std::slice::from_ref(grid),
        sample_rate,
        request.optimizer,
    ) {
        Ok((taps, note)) => {
            if note.is_some() {
                loss = evaluate(&neutral_weights);
                if !loss.is_finite() {
                    return Err(fail(
                        "Minimum-phase FIR neutral fallback has no finite objective".into(),
                    ));
                }
                best = neutral_weights;
                selected_search = false;
                ceiling_note = note;
            }
            taps
        }
        Err(error) => return Err(error),
    };
    let status = format!(
        "{}hybrid {} realized dB-basis search; {} templates; {} taps; selected={}; backend_objective={}; recomputed_backend_objective={}; {}{}",
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
        result.message,
        ceiling_note
            .map(|note| format!("; {note}"))
            .unwrap_or_default(),
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
    evidence.multi_input_normalization = Some(normalization);
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
