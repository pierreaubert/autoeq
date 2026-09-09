use super::consts::SFM_EPS;
use super::misc::allpass_complex_response;
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Minimum quadrature-point count before the point loop goes multi-threaded.
/// Below this the thread-spawn cost dominates the dot products.
const PARALLEL_MIN_POINTS: usize = 32;
/// Hard cap on evaluator worker threads so one area evaluation cannot
/// oversubscribe the machine when the outer DE also parallelizes elsewhere.
const PARALLEL_MAX_WORKERS: usize = 8;

/// Buffered continuous-area point evaluator.
///
/// The legacy evaluation rebuilt the per-sub all-pass biquads and cloned
/// every per-sub frequency vector into a one-seat nested allocation
/// (`seat_form`) for *each* quadrature point of *each* area evaluation. This
/// evaluator instead:
///
/// * builds the per-sub channel factors
///   (`gain * polarity * delay-phase * all-pass`) once per candidate over the
///   shared grid,
/// * indexes the pre-baked static per-point complex responses by slice (no
///   per-sub clones, no nested one-seat allocations),
/// * reuses the complex-sum and SPL scratch buffers across points, and
/// * fans the point loop out over an explicit shared worker budget with a
///   deterministic fixed-order reduction.
pub(super) struct AreaEvaluator {
    /// Pre-baked complex responses `[point][sub][freq]` on the shared grid.
    static_complex: Vec<Vec<Vec<Complex64>>>,
    /// Quadrature weights, one per point.
    weights: Vec<f64>,
    freqs: Array1<f64>,
    sample_rate: f64,
    eval_min: f64,
    eval_max: f64,
    workers: usize,
    /// Per-candidate channel factors `[sub][freq]`, rebuilt by
    /// [`prepare_candidate`](Self::prepare_candidate).
    factors: Vec<Vec<Complex64>>,
    sum_buf: Vec<Complex64>,
    spl_buf: Vec<f64>,
}

impl AreaEvaluator {
    pub(super) fn new(
        num_subs: usize,
        static_complex: Vec<Vec<Vec<Complex64>>>,
        weights: Vec<f64>,
        freqs: Array1<f64>,
        sample_rate: f64,
        eval_min: f64,
        eval_max: f64,
        workers: usize,
    ) -> Self {
        let num_freqs = freqs.len();
        // Worst-case search supplies positions dynamically and has no static
        // quadrature. Source count belongs to the measurement contract, not
        // to the optional pre-baked point collection.
        Self {
            static_complex,
            weights,
            freqs,
            sample_rate,
            eval_min,
            eval_max,
            workers: workers.max(1),
            factors: vec![vec![Complex64::new(0.0, 0.0); num_freqs]; num_subs],
            sum_buf: vec![Complex64::new(0.0, 0.0); num_freqs],
            spl_buf: vec![0.0; num_freqs],
        }
    }

    /// Explicit worker budget, shared by every area evaluation on this path.
    /// `Some(n)` caps threads at `n`; `None` stays sequential below
    /// [`PARALLEL_MIN_POINTS`] points and otherwise uses the available
    /// parallelism capped at [`PARALLEL_MAX_WORKERS`].
    pub(super) fn resolve_workers(requested: Option<usize>, num_points: usize) -> usize {
        match requested {
            Some(n) => n.max(1),
            None => {
                if num_points < PARALLEL_MIN_POINTS {
                    1
                } else {
                    let upper = num_points.clamp(1, PARALLEL_MAX_WORKERS);
                    std::thread::available_parallelism()
                        .map(|parallelism| parallelism.get())
                        .unwrap_or(1)
                        .min(upper)
                }
            }
        }
    }

    /// Rebuild the per-sub channel factors for one candidate. All-pass
    /// responses are evaluated once per (sub, frequency) here instead of once
    /// per (point, sub, frequency).
    pub(super) fn prepare_candidate(
        &mut self,
        gains: &[f64],
        delays: &[f64],
        polarities: &[bool],
        allpass_filters: &[Vec<(f64, f64)>],
    ) {
        // Factor rows are sized `[num_subs][num_freqs]` at construction and
        // every candidate decoded by `decode_mso_params` carries exactly
        // `num_subs` entries, so no per-call allocation happens here.
        let allpass_biquads: Vec<Vec<Biquad>> = allpass_filters
            .iter()
            .map(|filters| {
                filters
                    .iter()
                    .map(|&(freq, q)| {
                        Biquad::new(BiquadFilterType::AllPass, freq, self.sample_rate, q, 0.0)
                    })
                    .collect()
            })
            .collect();
        for (sub_idx, factors) in self.factors.iter_mut().enumerate() {
            let gain_linear = 10.0_f64.powf(gains.get(sub_idx).copied().unwrap_or(0.0) / 20.0);
            let polarity = if polarities.get(sub_idx).copied().unwrap_or(false) {
                -1.0
            } else {
                1.0
            };
            let delay_s = delays.get(sub_idx).copied().unwrap_or(0.0) / 1000.0;
            for (freq_idx, &freq) in self.freqs.iter().enumerate() {
                if freq < self.eval_min || freq > self.eval_max {
                    continue;
                }
                let omega = 2.0 * PI * freq;
                let delay_phase = Complex64::from_polar(1.0, -omega * delay_s);
                let allpass_phase = allpass_biquads
                    .get(sub_idx)
                    .map(|filters| {
                        filters
                            .iter()
                            .fold(Complex64::new(1.0, 0.0), |acc, allpass| {
                                acc * allpass_complex_response(allpass, freq)
                            })
                    })
                    .unwrap_or_else(|| Complex64::new(1.0, 0.0));
                factors[freq_idx] =
                    Complex64::new(gain_linear * polarity, 0.0) * delay_phase * allpass_phase;
            }
        }
    }

    /// SPL flatness (std of dB SPL) of one static quadrature point under the
    /// prepared candidate.
    pub(super) fn point_flatness(&mut self, point_idx: usize) -> f64 {
        let per_sub = &self.static_complex[point_idx];
        flatness_of(
            per_sub,
            &self.factors,
            &self.freqs,
            self.eval_min,
            self.eval_max,
            &mut self.sum_buf,
            &mut self.spl_buf,
        )
    }

    /// SPL flatness of an ad-hoc per-sub response (worst-case inner search)
    /// under the prepared candidate. No cloning: slices are dotted directly.
    pub(super) fn point_flatness_from_per_sub(&mut self, per_sub: &[Vec<Complex64>]) -> f64 {
        flatness_of(
            per_sub,
            &self.factors,
            &self.freqs,
            self.eval_min,
            self.eval_max,
            &mut self.sum_buf,
            &mut self.spl_buf,
        )
    }

    /// Expected-value scalarisation over the static points.
    pub(super) fn evaluate_expected(
        &mut self,
        gains: &[f64],
        delays: &[f64],
        polarities: &[bool],
        allpass_filters: &[Vec<(f64, f64)>],
    ) -> f64 {
        self.prepare_candidate(gains, delays, polarities, allpass_filters);
        if self.workers <= 1 || self.static_complex.len() < PARALLEL_MIN_POINTS {
            let mut acc = 0.0;
            for point_idx in 0..self.static_complex.len() {
                acc += self.weights[point_idx] * self.point_flatness(point_idx);
            }
            return acc;
        }
        self.point_losses_parallel()
            .iter()
            .zip(self.weights.iter())
            .map(|(loss, weight)| loss * weight)
            .sum()
    }

    /// CVaR scalarisation over the static points. `alpha` must already be
    /// validated into (0, 1] by the caller; it is never clamped here.
    pub(super) fn evaluate_cvar(
        &mut self,
        alpha: f64,
        gains: &[f64],
        delays: &[f64],
        polarities: &[bool],
        allpass_filters: &[Vec<(f64, f64)>],
    ) -> f64 {
        self.prepare_candidate(gains, delays, polarities, allpass_filters);
        let mut weighted: Vec<(f64, f64)> =
            if self.workers <= 1 || self.static_complex.len() < PARALLEL_MIN_POINTS {
                (0..self.static_complex.len())
                    .map(|point_idx| {
                        let loss = self.point_flatness(point_idx);
                        let weight = self.weights[point_idx];
                        (
                            if loss.is_finite() {
                                loss
                            } else {
                                f64::INFINITY
                            },
                            if weight.is_finite() && weight > 0.0 {
                                weight
                            } else {
                                0.0
                            },
                        )
                    })
                    .collect()
            } else {
                let losses = self.point_losses_parallel();
                losses
                    .into_iter()
                    .zip(self.weights.iter())
                    .map(|(loss, &weight)| {
                        (
                            if loss.is_finite() {
                                loss
                            } else {
                                f64::INFINITY
                            },
                            if weight.is_finite() && weight > 0.0 {
                                weight
                            } else {
                                0.0
                            },
                        )
                    })
                    .collect()
            };
        weighted.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut acc_loss = 0.0;
        let mut acc_mass = 0.0;
        for (loss, weight) in &weighted {
            // Zero probability does not end the tail: lower-loss points can
            // still carry mass needed to reach alpha.
            if *weight <= 0.0 {
                continue;
            }
            let take = (alpha - acc_mass).min(*weight);
            if take <= 0.0 {
                break;
            }
            acc_loss += take * loss;
            acc_mass += take;
            if acc_mass >= alpha {
                break;
            }
        }
        if acc_mass > 0.0 {
            acc_loss / acc_mass
        } else {
            f64::INFINITY
        }
    }

    /// Per-point flatness losses in point order, fanned over the worker
    /// budget. The reduction order is fixed (chunk order, then point order),
    /// so results are deterministic for a fixed worker count.
    fn point_losses_parallel(&self) -> Vec<f64> {
        let num_points = self.static_complex.len();
        let mut losses = vec![f64::INFINITY; num_points];
        let chunk = num_points.div_ceil(self.workers);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (chunk_idx, out) in losses.chunks_mut(chunk).enumerate() {
                let start = chunk_idx * chunk;
                let points = &self.static_complex[start..(start + out.len()).min(num_points)];
                let handle = scope.spawn(move || {
                    let mut sum = vec![Complex64::new(0.0, 0.0); self.freqs.len()];
                    let mut spl = vec![0.0; self.freqs.len()];
                    for (slot, per_sub) in out.iter_mut().zip(points.iter()) {
                        *slot = flatness_of(
                            per_sub,
                            &self.factors,
                            &self.freqs,
                            self.eval_min,
                            self.eval_max,
                            &mut sum,
                            &mut spl,
                        );
                    }
                });
                handles.push(handle);
            }
            for handle in handles {
                handle.join().expect("area point worker panicked");
            }
        });
        losses
    }
}

/// Combine one point's per-sub responses with prepared channel factors and
/// return the SPL flatness (population std of dB SPL). Writes through caller
/// scratch buffers; allocates nothing.
fn flatness_of(
    per_sub: &[Vec<Complex64>],
    factors: &[Vec<Complex64>],
    freqs: &Array1<f64>,
    eval_min: f64,
    eval_max: f64,
    sum: &mut [Complex64],
    spl: &mut [f64],
) -> f64 {
    let bins = freqs.len();
    if per_sub.is_empty()
        || factors.len() != per_sub.len()
        || sum.len() < bins
        || spl.len() < bins
        || per_sub.iter().chain(factors).any(|row| row.len() != bins)
        || !eval_min.is_finite()
        || !eval_max.is_finite()
        || eval_min > eval_max
        || freqs
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return f64::INFINITY;
    }
    let mut count = 0_usize;
    for (freq_idx, &freq) in freqs.iter().enumerate() {
        if freq < eval_min || freq > eval_max {
            continue;
        }
        let mut combined = Complex64::new(0.0, 0.0);
        for (sub_idx, sub_data) in per_sub.iter().enumerate() {
            let factor_row = &factors[sub_idx];
            let sample = sub_data[freq_idx];
            let factor = factor_row[freq_idx];
            if !sample.re.is_finite()
                || !sample.im.is_finite()
                || !factor.re.is_finite()
                || !factor.im.is_finite()
            {
                return f64::INFINITY;
            }
            combined += sub_data[freq_idx] * factor_row[freq_idx];
        }
        if !combined.re.is_finite() || !combined.im.is_finite() {
            return f64::INFINITY;
        }
        sum[freq_idx] = combined;
        spl[freq_idx] = 20.0 * combined.norm().max(SFM_EPS).log10();
        count += 1;
    }
    if count == 0 {
        return f64::INFINITY;
    }
    let mut total = 0.0;
    for (freq_idx, &freq) in freqs.iter().enumerate() {
        if freq < eval_min || freq > eval_max {
            continue;
        }
        total += spl[freq_idx];
    }
    let mean = total / count as f64;
    let mut sum_sq = 0.0;
    for (freq_idx, &freq) in freqs.iter().enumerate() {
        if freq < eval_min || freq > eval_max {
            continue;
        }
        sum_sq += (spl[freq_idx] - mean).powi(2);
    }
    // Use a local measured power reference at every quadrature/adversarial
    // point. A spatially uniform cancellation must never count as perfect EQ.
    let mut candidate = Vec::with_capacity(count);
    let mut reference = Vec::with_capacity(count);
    for (i, &frequency) in freqs.iter().enumerate() {
        if frequency >= eval_min && frequency <= eval_max {
            candidate.push(spl[i]);
            reference.push(
                10.0 * per_sub
                    .iter()
                    .map(|sub| sub[i].norm_sqr())
                    .sum::<f64>()
                    .max(1e-24)
                    .log10(),
            );
        }
    }
    let low_count = freqs
        .iter()
        .filter(|&&frequency| frequency >= eval_min && frequency <= (eval_min * 2.0).min(eval_max))
        .count();
    let extension = if low_count > 0 {
        autoeq_optim::loss::multisub::array_output_penalty(
            &candidate[..low_count],
            &reference[..low_count],
        )
    } else {
        0.0
    };
    (sum_sq / count as f64).sqrt()
        + autoeq_optim::loss::multisub::array_output_penalty(&candidate, &reference)
        + extension
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_factors(num_subs: usize, num_freqs: usize) -> Vec<Vec<Complex64>> {
        vec![vec![Complex64::new(1.0, 0.0); num_freqs]; num_subs]
    }

    #[test]
    fn every_area_point_rejects_complete_cancellation() {
        let freqs = Array1::from(vec![20.0, 40.0, 80.0, 120.0]);
        let response = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let per_sub = vec![response.clone(), response];
        let mut factors = unit_factors(2, 4);
        let mut sum = vec![Complex64::new(0.0, 0.0); 4];
        let mut spl = vec![0.0; 4];
        let baseline = flatness_of(&per_sub, &factors, &freqs, 20.0, 120.0, &mut sum, &mut spl);
        factors[1].fill(Complex64::new(-1.0, 0.0));
        let cancelled = flatness_of(&per_sub, &factors, &freqs, 20.0, 120.0, &mut sum, &mut spl);
        assert!(cancelled > baseline + 1000.0);
    }

    #[test]
    fn malformed_area_transfer_cannot_become_finite_evidence() {
        let freqs = Array1::from(vec![20.0, 40.0, 80.0, 120.0]);
        let valid = vec![vec![Complex64::new(1.0, 0.0); 4]; 2];
        let mut nan = valid.clone();
        nan[0][1].re = f64::NAN;
        let mut short = valid.clone();
        short[0].pop();
        let mut invalid_factors = unit_factors(2, 4);
        invalid_factors[0][0].im = f64::NAN;
        let cases = [
            (nan, unit_factors(2, 4)),
            (short, unit_factors(2, 4)),
            (valid.clone(), invalid_factors),
            (
                vec![vec![Complex64::new(f64::MAX, 0.0); 4]; 2],
                unit_factors(2, 4),
            ),
            (valid.clone(), unit_factors(1, 4)),
            (valid.clone(), unit_factors(2, 3)),
            (Vec::new(), Vec::new()),
        ];
        for (per_sub, factors) in cases {
            let mut sum = vec![Complex64::new(0.0, 0.0); 4];
            let mut spl = vec![0.0; 4];
            let loss = flatness_of(&per_sub, &factors, &freqs, 20.0, 120.0, &mut sum, &mut spl);
            assert!(
                !loss.is_finite(),
                "malformed response was accepted as {loss}"
            );
        }
    }

    #[test]
    fn cancelled_array_objective_has_squared_output_penalty_not_db_units() {
        let freqs = Array1::from(vec![20.0, 40.0, 80.0, 120.0]);
        for level_db in [80.0_f64, 90.0, 100.0] {
            let amplitude = 10.0_f64.powf(level_db / 20.0);
            let per_sub = vec![vec![Complex64::new(amplitude, 0.0); 4]; 2];
            let mut factors = unit_factors(2, 4);
            factors[1].fill(Complex64::new(-1.0, 0.0));
            let mut sum = vec![Complex64::new(0.0, 0.0); 4];
            let mut spl = vec![0.0; 4];
            let observed = flatness_of(&per_sub, &factors, &freqs, 20.0, 120.0, &mut sum, &mut spl);
            assert!(sum.iter().all(|value| *value == Complex64::new(0.0, 0.0)));
            // Two equal sources have power-reference level L + 10log10(2).
            // Exact cancellation uses the declared numerical -240 dB floor.
            // Zero shape variance remains, but broadband and bass-extension
            // guards each charge the mean (>3 dB) and local (>12 dB) deficit.
            let deficit = level_db + 10.0 * 2.0_f64.log10() - (-240.0);
            let expected = 2.0 * (4.0 * (deficit - 3.0).powi(2) + (deficit - 12.0).powi(2));
            assert!(
                (observed - expected).abs() < 1e-6,
                "{level_db}: {observed} vs {expected}"
            );
            assert!(observed > 1_000_000.0);
        }
    }

    #[test]
    fn flat_response_has_zero_flatness_loss() {
        let freqs = Array1::from(vec![20.0, 40.0, 80.0, 120.0]);
        let per_sub = vec![vec![Complex64::new(1.0, 0.0); 4]; 2];
        let factors = unit_factors(2, 4);
        let mut sum = vec![Complex64::new(0.0, 0.0); 4];
        let mut spl = vec![0.0; 4];
        let loss = flatness_of(&per_sub, &factors, &freqs, 20.0, 120.0, &mut sum, &mut spl);
        assert!(loss.abs() < 1e-9, "flat response must score ~0, got {loss}");
    }

    #[test]
    fn empty_band_is_nonfinite() {
        let freqs = Array1::from(vec![20.0, 40.0]);
        let per_sub = vec![vec![Complex64::new(1.0, 0.0); 2]; 1];
        let factors = unit_factors(1, 2);
        let mut sum = vec![Complex64::new(0.0, 0.0); 2];
        let mut spl = vec![0.0; 2];
        let loss = flatness_of(&per_sub, &factors, &freqs, 200.0, 300.0, &mut sum, &mut spl);
        assert!(!loss.is_finite());
    }

    #[test]
    fn resolve_workers_caps_explicit_budget() {
        assert_eq!(AreaEvaluator::resolve_workers(Some(4), 1000), 4);
        assert_eq!(AreaEvaluator::resolve_workers(Some(0), 1000), 1);
        assert_eq!(AreaEvaluator::resolve_workers(None, 8), 1);
        let auto = AreaEvaluator::resolve_workers(None, 1000);
        assert!((1..=PARALLEL_MAX_WORKERS).contains(&auto));
    }

    #[test]
    fn cvar_zero_mass_points_do_not_truncate_the_tail() {
        // Single-source responses have no cancellation/output deficit. Their
        // two-bin population standard deviations are independently 20, 10, 0.
        // The 75% upper tail contains all 50% mass at loss 20 and 25% at 0.
        for (copies, workers) in [(1, 1), (16, 4)] {
            let mut points = Vec::new();
            let mut weights = Vec::new();
            for (amplitude, mass) in [(100.0, 0.5), (10.0, 0.0), (1.0, 0.5)] {
                for _ in 0..copies {
                    points.push(vec![vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(amplitude, 0.0),
                    ]]);
                    weights.push(mass / copies as f64);
                }
            }
            let mut evaluator = AreaEvaluator::new(
                1,
                points,
                weights,
                Array1::from(vec![20.0, 40.0]),
                48000.0,
                20.0,
                40.0,
                workers,
            );
            let observed = evaluator.evaluate_cvar(0.75, &[0.0], &[0.0], &[false], &[vec![]]);
            let expected = 20.0 * 0.5 / 0.75;
            assert!(
                (observed - expected).abs() < 1e-10,
                "copies={copies}, workers={workers}: {observed} vs {expected}"
            );
        }
    }

    #[test]
    fn adversarial_points_use_candidate_factors_without_static_quadrature() {
        let mut evaluator = AreaEvaluator::new(
            2,
            Vec::new(),
            Vec::new(),
            Array1::from(vec![20.0, 40.0]),
            48000.0,
            20.0,
            40.0,
            1,
        );
        let per_sub = vec![vec![Complex64::new(10000.0, 0.0); 2]; 2];
        evaluator.prepare_candidate(&[0.0, 0.0], &[0.0, 0.0], &[false, false], &[vec![], vec![]]);
        let constructive = evaluator.point_flatness_from_per_sub(&per_sub);
        assert!(
            constructive.abs() < 1e-10,
            "constructive loss {constructive}"
        );
        evaluator.prepare_candidate(&[0.0, 0.0], &[0.0, 0.0], &[false, true], &[vec![], vec![]]);
        let cancelled = evaluator.point_flatness_from_per_sub(&per_sub);
        assert!(
            cancelled.is_finite() && cancelled > 1_000_000.0,
            "candidate polarity must affect the ad-hoc point: {cancelled}"
        );
    }
}
