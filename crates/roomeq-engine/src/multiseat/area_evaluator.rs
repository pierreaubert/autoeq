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
        static_complex: Vec<Vec<Vec<Complex64>>>,
        weights: Vec<f64>,
        freqs: Array1<f64>,
        sample_rate: f64,
        eval_min: f64,
        eval_max: f64,
        workers: usize,
    ) -> Self {
        let num_freqs = freqs.len();
        let num_subs = static_complex
            .first()
            .map(Vec::len)
            .unwrap_or_default();
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
                        filters.iter().fold(Complex64::new(1.0, 0.0), |acc, allpass| {
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
        let mut weighted: Vec<(f64, f64)> = if self.workers <= 1
            || self.static_complex.len() < PARALLEL_MIN_POINTS
        {
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
    let mut count = 0_usize;
    for (freq_idx, &freq) in freqs.iter().enumerate() {
        if freq < eval_min || freq > eval_max {
            continue;
        }
        let mut combined = Complex64::new(0.0, 0.0);
        for (sub_idx, sub_data) in per_sub.iter().enumerate() {
            let Some(factor_row) = factors.get(sub_idx) else {
                continue;
            };
            combined += sub_data[freq_idx] * factor_row[freq_idx];
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
    (sum_sq / count as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_factors(num_subs: usize, num_freqs: usize) -> Vec<Vec<Complex64>> {
        vec![vec![Complex64::new(1.0, 0.0); num_freqs]; num_subs]
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
}
