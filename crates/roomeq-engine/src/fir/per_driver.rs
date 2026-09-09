//! Conservative joint, finite-tap correction of calibrated physical branches.
//!
//! Never invert an individual branch towards the full-range acoustic target.
//! Room-null masks constrain electrical boost, not the acoustic benefit of
//! changing relative branch phase. All branches receive identical causal support.
use crate::Curve;
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_model::{OptimizerConfig, ProcessingMode};
use rustfft::FftPlanner;

pub struct PerDriverFir {
    pub coefficients: Vec<Vec<f64>>,
    pub final_curve: Curve,
    pub latency_samples: usize,
    pub protected_bins: usize,
    pub before_rms_db: f64,
    pub after_rms_db: f64,
}

/// Conservative local notch / unreliable-data detector. The half-octave
/// neighbourhood is deliberately wider than the correction smoothing. A single
/// capture cannot prove SBIR; treating ambiguous dips as unboostable is safer.
pub fn fir_null_mask(curve: &Curve) -> Vec<bool> {
    curve
        .freq
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let envelope = curve
                .freq
                .iter()
                .zip(&curve.spl)
                .filter(|(other, _)| (**other / f).log2().abs() <= 0.5)
                .map(|(_, &spl)| spl)
                .fold(curve.spl[i], f64::max);
            curve.spl[i] < envelope - 6.0
                || curve
                    .coherence
                    .as_ref()
                    .is_some_and(|c| !c[i].is_finite() || c[i] < 0.8)
                || curve
                    .noise_floor_db
                    .as_ref()
                    .is_some_and(|n| curve.spl[i] - n[i] < 10.0)
        })
        .collect()
}

fn complex(curve: &Curve, i: usize) -> Complex64 {
    Complex64::from_polar(
        10.0_f64.powf(curve.spl[i] / 20.0),
        curve.phase.as_ref().map_or(0.0, |p| p[i].to_radians()),
    )
}

fn sum(branches: &[Curve], taps: &[Vec<f64>], fs: f64) -> Curve {
    let freq = &branches[0].freq;
    let mut response = vec![Complex64::new(0.0, 0.0); freq.len()];
    for (branch, coefficients) in branches.iter().zip(taps) {
        let h = crate::response::compute_fir_complex_response(coefficients, freq, fs);
        for (i, v) in h.iter().enumerate() {
            response[i] += complex(branch, i) * v;
        }
    }
    Curve {
        freq: freq.clone(),
        spl: Array1::from_iter(response.iter().map(|v| 20.0 * v.norm().max(1e-15).log10())),
        phase: Some(Array1::from_iter(
            response.iter().map(|v| v.arg().to_degrees()),
        )),
        ..Curve::default()
    }
}

fn band_weight(f: f64, lo: f64, hi: f64) -> f64 {
    if f <= lo || f >= hi {
        return 0.0;
    }
    // Transitions are INSIDE the configured correction band.
    ((f / lo).log2() / 0.25)
        .min((hi / f).log2() / 0.25)
        .clamp(0.0, 1.0)
}

/// Real, centered Hann-windowed FIR, with inverse FFT normalization 1/N.
/// Frequency sampling is interpolated on log Hz; DC/Nyquist remain real.
fn realize(freq: &Array1<f64>, db: &[f64], phase: &[f64], taps: usize, fs: f64) -> Vec<f64> {
    let n = (taps * 4).max(65536).next_power_of_two();
    let mut spectrum = vec![Complex64::new(1.0, 0.0); n];
    for k in 1..n / 2 {
        let f = k as f64 * fs / n as f64;
        let (mut left, mut right) = (0, freq.len());
        while left < right {
            let middle = (left + right) / 2;
            if freq[middle] < f {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        let j = left;
        let (gain, angle) = if j == 0 || j == freq.len() {
            (0.0, 0.0)
        } else {
            let t = (f / freq[j - 1]).ln() / (freq[j] / freq[j - 1]).ln();
            (
                db[j - 1] + t * (db[j] - db[j - 1]),
                phase[j - 1] + t * (phase[j] - phase[j - 1]),
            )
        };
        spectrum[k] = Complex64::from_polar(10.0_f64.powf(gain / 20.0), angle.to_radians());
        spectrum[n - k] = spectrum[k].conj();
    }
    FftPlanner::<f64>::new()
        .plan_fft_inverse(n)
        .process(&mut spectrum);
    let center = taps / 2;
    (0..taps)
        .map(|i| {
            let window = 0.5
                + 0.5 * (std::f64::consts::PI * (i as f64 - center as f64) / center as f64).cos();
            spectrum[(n + i - center) % n].re / n as f64 * window
        })
        .collect()
}

/// `branches` include retained gain/delay/crossover/IIR and common DSP, on an
/// identical grid; `measurements` are their unfiltered calibrated captures.
/// Mixed phase retains phase-only semantics. A delayed identity is an explicit
/// safe fallback when no finite candidate improves the protected objective.
pub fn generate_per_driver_firs(
    branches: &[Curve],
    measurements: &[Curve],
    target: &Curve,
    config: &OptimizerConfig,
    fs: f64,
) -> Result<PerDriverFir, String> {
    let fir = config.fir.clone().unwrap_or_default();
    if branches.is_empty() || branches.len() != measurements.len() || !fs.is_finite() || fs <= 0.0 {
        return Err("invalid per-driver FIR inputs".into());
    }
    target
        .validate("per-driver FIR target")
        .map_err(|e| e.to_string())?;
    for c in branches.iter().chain(measurements) {
        c.validate("per-driver FIR branch")
            .map_err(|e| e.to_string())?;
        if c.freq != target.freq {
            return Err("per-driver FIR grids must match".into());
        }
    }
    if config.min_freq <= 0.0 || config.max_freq <= config.min_freq || config.max_freq >= fs / 2.0 {
        return Err("invalid per-driver FIR correction band".into());
    }
    let phase_only = config.processing_mode == ProcessingMode::MixedPhase;
    let mp = config.mixed_phase.as_ref();
    let taps = if phase_only {
        ((mp.map_or(10.0, |c| c.max_fir_length_ms) * fs / 1000.0).round() as usize).max(31) | 1
    } else {
        fir.taps
    };
    if !(31..=131072).contains(&taps) {
        return Err("per-driver FIR taps must be 31..=131072".into());
    }
    let minimum_phase = !phase_only && fir.phase.eq_ignore_ascii_case("minimum");
    let latency = if minimum_phase { 0 } else { taps / 2 };
    let mut identity = vec![0.0; taps];
    identity[latency] = 1.0;
    let mut best = vec![identity; branches.len()];
    let baseline = sum(branches, &best, fs);
    let masks: Vec<_> = measurements.iter().map(fir_null_mask).collect();
    let combined_mask = fir_null_mask(&baseline);
    let protected_bins = masks.iter().flatten().filter(|&&v| v).count();
    let score = |c: &Curve| {
        let mut error = 0.0;
        let mut weight = 0.0;
        for i in 1..c.freq.len() {
            if !combined_mask[i] && (config.min_freq..=config.max_freq).contains(&c.freq[i]) {
                let w = (c.freq[i] / c.freq[i - 1]).ln();
                error += w * (c.spl[i] - target.spl[i]).powi(2);
                weight += w;
            }
        }
        (error / weight.max(1e-12)).sqrt()
    };
    let before = score(&baseline);
    let mut best_score = before;
    let mut best_curve = baseline.clone();
    let boost = fir
        .max_boost_db
        .unwrap_or(config.max_db)
        .min(config.max_db)
        .max(0.0);
    let phase_enabled =
        phase_only || (fir.phase.eq_ignore_ascii_case("kirkeby") && fir.correct_excess_phase);
    let mp_config = crate::mixed_phase::MixedPhaseConfig {
        phase_smoothing_octaves: mp.map_or(fir.phase_smoothing, |c| c.phase_smoothing_octaves),
        ..Default::default()
    };
    let phases: Vec<_> = measurements
        .iter()
        .map(|c| {
            if phase_enabled && c.phase.is_some() {
                crate::mixed_phase::decompose_phase(c, &mp_config)
                    .map(|(_, _, _, residual)| residual.to_vec())
                    .unwrap_or_else(|_| vec![0.0; c.freq.len()])
            } else {
                vec![0.0; c.freq.len()]
            }
        })
        .collect();
    // Joint search: every candidate is a complete set, never a collection of
    // independently accepted full-target inverses.
    for phase_strength in [0.0, 0.25, 0.5, 1.0] {
        if !phase_enabled && phase_strength != 0.0 {
            continue;
        }
        for strength in [0.0, 0.25, 0.5, 1.0] {
            if (phase_only && strength != 1.0) || (phase_strength == 0.0 && strength == 0.0) {
                continue;
            }
            let candidate: Vec<_> = branches
                .iter()
                .enumerate()
                .map(|(driver, branch)| {
                    let mut db = vec![0.0; target.freq.len()];
                    let mut phase = db.clone();
                    for i in 0..db.len() {
                        let band = band_weight(target.freq[i], config.min_freq, config.max_freq);
                        let protected = masks[driver][i] || combined_mask[i];
                        // Do not invert crossover stopbands or supply energy into
                        // a measured null. The group residual is the only target.
                        let retained_boost = (branch.spl[i] - measurements[driver].spl[i]).max(0.0);
                        let cap = if protected {
                            0.0
                        } else {
                            (boost - retained_boost).max(0.0)
                        };
                        if !phase_only {
                            db[i] = strength
                                * band
                                * (target.spl[i] - baseline.spl[i]).clamp(-12.0, cap);
                        }
                        if !masks[driver][i] {
                            phase[i] = -phase_strength * band * phases[driver][i];
                        }
                        if branch.spl[i] < baseline.spl[i] - 30.0 {
                            db[i] = 0.0;
                            phase[i] = 0.0;
                        }
                    }
                    if minimum_phase {
                        let correction = Curve {
                            freq: target.freq.clone(),
                            spl: Array1::from_vec(db),
                            ..Curve::default()
                        };
                        autoeq_fir::generate_fir_from_response(
                            &correction,
                            fs,
                            taps,
                            autoeq_fir::FirPhase::Minimum,
                        )
                    } else {
                        realize(&target.freq, &db, &phase, taps, fs)
                    }
                })
                .collect();
            let safe = candidate.iter().enumerate().all(|(driver, coefficients)| {
                if coefficients.iter().any(|v| !v.is_finite()) {
                    return false;
                }
                if phase_only || fir.pre_ringing.is_some() {
                    let threshold = if phase_only {
                        mp.map_or(-30.0, |c| c.pre_ringing_threshold_db)
                    } else {
                        fir.pre_ringing.as_ref().unwrap().threshold_db
                    };
                    let peak = coefficients.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                    let pre = coefficients[..latency]
                        .iter()
                        .map(|v| v.abs())
                        .fold(0.0_f64, f64::max);
                    if pre > peak * 10.0_f64.powf(threshold / 20.0) {
                        return false;
                    }
                }
                let response =
                    crate::response::compute_fir_complex_response(coefficients, &target.freq, fs);
                response.iter().enumerate().all(|(i, h)| {
                    let db = 20.0 * h.norm().max(1e-15).log10();
                    let protected = masks[driver][i] || combined_mask[i];
                    let outside =
                        target.freq[i] < config.min_freq || target.freq[i] > config.max_freq;
                    let retained_boost =
                        (branches[driver].spl[i] - measurements[driver].spl[i]).max(0.0);
                    db <= (if protected {
                        0.1
                    } else {
                        (boost - retained_boost).max(0.0) + 0.1
                    }) && (!phase_only || db.abs() <= 0.5)
                        && (!outside || db.abs() <= 0.5)
                })
            });
            if !safe {
                continue;
            }
            let curve = sum(branches, &candidate, fs);
            let loss = score(&curve);
            if loss.is_finite() && loss + 1e-6 < best_score {
                best = candidate;
                best_score = loss;
                best_curve = curve;
            }
        }
    }
    log::info!(
        "Per-driver FIR: {} outputs, {} taps, {} protected bins, target RMS {:.3} -> {:.3} dB{}",
        branches.len(),
        taps,
        protected_bins,
        before,
        best_score,
        if best_score == before {
            " (delayed identity: no safe improving candidate)"
        } else {
            ""
        }
    );
    Ok(PerDriverFir {
        coefficients: best,
        final_curve: best_curve,
        latency_samples: latency,
        protected_bins,
        before_rms_db: before,
        after_rms_db: best_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn flat(level: f64) -> Curve {
        let freq = Array1::logspace(10.0, 20.0_f64.log10(), 20000.0_f64.log10(), 160);
        Curve {
            spl: Array1::from_elem(freq.len(), level),
            phase: Some(Array1::zeros(freq.len())),
            freq,
            ..Curve::default()
        }
    }
    fn config() -> OptimizerConfig {
        OptimizerConfig {
            min_freq: 20.0,
            max_freq: 200.0,
            max_db: 4.0,
            processing_mode: ProcessingMode::PhaseLinear,
            fir: Some(roomeq_model::FirConfig {
                taps: 4096,
                phase: "linear".into(),
                placement: roomeq_model::FirPlacement::PerDriver,
                ..Default::default()
            }),
            ..Default::default()
        }
    }
    #[test]
    fn per_driver_fir_joint_residual_reduces_peak_and_preserves_upper_band() {
        let target = flat(80.0);
        let mut branch = flat(80.0 - 20.0 * 2.0_f64.log10());
        for (i, &f) in branch.freq.iter().enumerate() {
            branch.spl[i] += 6.0 * (-((f / 90.0).log2() / 0.6).powi(2)).exp();
        }
        let branches = vec![branch.clone(), branch];
        let result =
            generate_per_driver_firs(&branches, &branches, &target, &config(), 48000.0).unwrap();
        assert!(
            result.after_rms_db < result.before_rms_db - 0.5,
            "{} -> {}",
            result.before_rms_db,
            result.after_rms_db
        );
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.latency_samples, 2048);
        for taps in &result.coefficients {
            assert_eq!(taps.len(), 4096);
            let h = crate::response::compute_fir_complex_response(
                taps,
                &Array1::from_vec(vec![500.0, 1000.0, 10000.0]),
                48000.0,
            );
            assert!(h.iter().all(|v| (20.0 * v.norm().log10()).abs() < 0.1));
        }
    }
    #[test]
    fn per_driver_fir_room_nulls_are_not_boosted() {
        let target = flat(80.0);
        let mut branch = flat(80.0 - 20.0 * 2.0_f64.log10());
        for (i, &f) in branch.freq.iter().enumerate() {
            branch.spl[i] -= 25.0 * (-((f / 80.0).log2() / 0.08).powi(2)).exp();
        }
        let mask = fir_null_mask(&branch);
        assert!(mask.iter().any(|&v| v));
        let branches = vec![branch.clone(), branch];
        let result =
            generate_per_driver_firs(&branches, &branches, &target, &config(), 48000.0).unwrap();
        assert!(result.protected_bins > 0);
        for taps in &result.coefficients {
            let h = crate::response::compute_fir_complex_response(taps, &target.freq, 48000.0);
            for (i, &protected) in mask.iter().enumerate() {
                if protected {
                    assert!(20.0 * h[i].norm().log10() <= 0.100001);
                }
            }
        }
    }
    #[test]
    fn per_driver_fir_rejects_mismatched_grid_and_invalid_taps() {
        let c = flat(80.0);
        let mut other = c.clone();
        other.freq[0] = 19.0;
        assert!(generate_per_driver_firs(&[other], &[c.clone()], &c, &config(), 48000.0).is_err());
        let mut config = config();
        config.fir.as_mut().unwrap().taps = 0;
        assert!(
            generate_per_driver_firs(&[c.clone()], &[c.clone()], &c, &config, 48000.0).is_err()
        );
    }
    #[test]
    fn per_driver_fir_mixed_phase_is_not_magnitude_equalization() {
        let c = flat(80.0);
        let target = flat(86.0);
        let mut config = config();
        config.processing_mode = ProcessingMode::MixedPhase;
        let result =
            generate_per_driver_firs(&[c.clone()], &[c], &target, &config, 48000.0).unwrap();
        assert_eq!(result.coefficients[0].len(), 481);
        let h = crate::response::compute_fir_complex_response(
            &result.coefficients[0],
            &target.freq,
            48000.0,
        );
        assert!(h.iter().all(|v| (20.0 * v.norm().log10()).abs() < 0.00001));
    }
    #[test]
    fn per_driver_fir_low_coherence_and_noise_are_protected() {
        let mut c = flat(80.0);
        c.coherence = Some(Array1::from_elem(c.freq.len(), 0.2));
        assert!(fir_null_mask(&c).iter().all(|&v| v));
        c.coherence = None;
        c.noise_floor_db = Some(Array1::from_elem(c.freq.len(), 75.0));
        assert!(fir_null_mask(&c).iter().all(|&v| v));
    }

    #[test]
    fn per_driver_fir_repairs_relative_phase_without_electrical_boost() {
        let target = flat(80.0);
        let first = flat(80.0 - 20.0 * 2.0_f64.log10());
        let mut second = first.clone();
        for (i, &f) in second.freq.iter().enumerate() {
            second.phase.as_mut().unwrap()[i] = 80.0 * (-((f / 85.0).log2() / 0.8).powi(2)).exp();
        }
        let branches = vec![first, second];
        let mut config = config();
        config.max_db = 0.0;
        let fir = config.fir.as_mut().unwrap();
        fir.phase = "kirkeby".into();
        fir.correct_excess_phase = true;
        fir.max_boost_db = Some(0.0);
        let result =
            generate_per_driver_firs(&branches, &branches, &target, &config, 48000.0).unwrap();
        assert!(
            result.after_rms_db < result.before_rms_db - 0.01,
            "{} -> {}",
            result.before_rms_db,
            result.after_rms_db
        );
        assert_ne!(result.coefficients[0], result.coefficients[1]);
        for taps in &result.coefficients {
            let h = crate::response::compute_fir_complex_response(taps, &target.freq, 48000.0);
            assert!(h.iter().all(|v| 20.0 * v.norm().log10() <= 0.100001));
        }
    }
}
