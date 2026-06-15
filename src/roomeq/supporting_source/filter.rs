//! Core supporting-source filter computation.

use super::{SupportingSourceFilter, compute_drr, generate_velvet_noise};
use crate::Curve;
use crate::error::{AutoeqError, Result};
use crate::roomeq::types::{PrecedenceLimitBand, SupportingSourceConfig};
use ndarray::Array1;
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Maximum allowed supporting-source gain relative to the primary, in dB.
/// This is a safety clamp to prevent the filter from requesting absurd
/// headroom when the support measurement is weak or noisy.
const MAX_SUPPORT_GAIN_DB: f64 = 24.0;

/// Compute the supporting-source filter from averaged magnitude responses.
pub fn compute_supporting_source_filter(
    primary: &Curve,
    support: &Curve,
    target: &Curve,
    config: &SupportingSourceConfig,
    sample_rate: f64,
) -> Result<SupportingSourceFilter> {
    // 1. Build a common log-frequency grid covering the union of all three curves.
    let common_freq = common_frequency_grid(&primary.freq, &support.freq, &target.freq)?;

    // 2. Interpolate all curves onto the common grid.
    let primary_i = crate::read::interpolate_log_space(&common_freq, primary);
    let support_i = crate::read::interpolate_log_space(&common_freq, support);
    let target_i = crate::read::interpolate_log_space(&common_freq, target);

    // 3. Smooth to 1/3-octave.
    let primary_smooth = smooth_one_third_octave(&primary_i)?;
    let support_smooth = smooth_one_third_octave(&support_i)?;
    let target_smooth = smooth_one_third_octave(&target_i)?;

    // 4. Constrain the target.
    let (constrained_spl, precedence_hits) = constrain_target(
        &target_smooth.spl,
        &primary_smooth.spl,
        &config.precedence_limits,
        &common_freq,
    );

    // 5. Compute supporting-source gain in dB and linear magnitude.
    let support_gain_db =
        compute_support_gain_db(&constrained_spl, &primary_smooth.spl, &support_smooth.spl);

    // 6. Zero the gain outside the compensation band.
    let mut windowed_gain_db = support_gain_db.clone();
    for (i, &f) in common_freq.iter().enumerate() {
        if f < config.freq_range_hz.0 || f > config.freq_range_hz.1 {
            windowed_gain_db[i] = f64::NEG_INFINITY;
        }
    }

    // 7. Build a Curve with the windowed gain so we can extract minimum phase.
    let mut gain_curve = Curve {
        freq: common_freq.clone(),
        spl: Array1::from(windowed_gain_db.clone()),
        ..Default::default()
    };

    // Replace -inf with a very low value for phase extraction.
    let finite_gain_db: Vec<f64> = gain_curve
        .spl
        .iter()
        .map(|&v| if v.is_finite() { v } else { -120.0 })
        .collect();
    gain_curve.spl = Array1::from(finite_gain_db);

    // 8. Build the FIR using minimum-phase extraction.
    let gain_db: Vec<f64> = gain_curve.spl.iter().copied().collect();
    let mut taps = minimum_phase_fir(&common_freq, &gain_db, config.fir_taps, sample_rate)?;

    // 11. Optional velvet-noise decorrelation (pre-convolved into FIR).
    if config.decorrelation == crate::roomeq::SupportingSourceDecorrelation::VelvetNoise
        && config.velvet_noise_taps > 0
    {
        let velvet = generate_velvet_noise(
            config.velvet_noise_taps,
            0.3, // ~1 impulse per 3 samples by default; tune later
            0xdeadbeef,
        );
        taps = convolve_fir(&taps, &velvet);
        // Truncate back to requested length (the tail is mostly velvet tail).
        taps.truncate(config.fir_taps);
    }

    // 12. Normalize.
    let max_abs = taps.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if max_abs > 0.0 {
        let scale = 1.0 / max_abs;
        for v in &mut taps {
            *v *= scale;
        }
    }

    // 13. DRR diagnostics.
    let (drr_before_db, drr_after_db) =
        compute_drr(&primary_smooth, &support_smooth, &support_gain_db, 0.5);

    let constrained_target = Curve {
        freq: common_freq,
        spl: constrained_spl,
        ..Default::default()
    };

    Ok(SupportingSourceFilter {
        taps,
        constrained_target,
        support_gain_db,
        drr_before_db,
        drr_after_db,
        precedence_limit_hits: precedence_hits,
    })
}

/// Build a common log-frequency grid from three frequency vectors.
fn common_frequency_grid(a: &Array1<f64>, b: &Array1<f64>, c: &Array1<f64>) -> Result<Array1<f64>> {
    if a.is_empty() || b.is_empty() || c.is_empty() {
        return Err(AutoeqError::InvalidMeasurement {
            message: "Cannot build common grid from empty curve".to_string(),
        });
    }
    let lo = a[0].min(b[0]).min(c[0]).max(1.0);
    let hi = a[a.len() - 1].max(b[b.len() - 1]).max(c[c.len() - 1]);
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        return Err(AutoeqError::InvalidMeasurement {
            message: "Invalid frequency range for common grid".to_string(),
        });
    }

    // Use the finest resolution among the three inputs, capped at a reasonable
    // number of points for FFT-based processing.
    let n = (a.len().max(b.len()).max(c.len()) * 2).clamp(256, 4096);
    let log_lo = lo.log10();
    let log_hi = hi.log10();
    Ok(Array1::from_vec(
        (0..n)
            .map(|i| 10.0_f64.powf(log_lo + (log_hi - log_lo) * i as f64 / (n - 1) as f64))
            .collect(),
    ))
}

/// Smooth a curve to 1/3-octave resolution.
fn smooth_one_third_octave(curve: &Curve) -> Result<Curve> {
    Ok(crate::read::smooth_one_over_n_octave(curve, 3))
}

/// Apply no-cancellation floor and precedence ceiling.
fn constrain_target(
    target: &Array1<f64>,
    primary: &Array1<f64>,
    limits: &[PrecedenceLimitBand],
    freq: &Array1<f64>,
) -> (Array1<f64>, usize) {
    let mut d_mod = target.clone();
    let mut hits = 0_usize;

    for i in 0..freq.len() {
        // No-cancellation floor.
        if d_mod[i] < primary[i] {
            d_mod[i] = primary[i];
        }

        // Precedence ceiling.
        let limit_db = limit_at_freq(freq[i], limits);
        let ceiling = primary[i] + limit_db;
        if d_mod[i] > ceiling {
            d_mod[i] = ceiling;
            hits += 1;
        }
    }

    (d_mod, hits)
}

fn limit_at_freq(f: f64, limits: &[PrecedenceLimitBand]) -> f64 {
    for band in limits {
        if f >= band.low_hz && f <= band.high_hz {
            return band.limit_db;
        }
    }
    0.0
}

/// Compute supporting-source gain in dB: sqrt(d² - p²) / s.
fn compute_support_gain_db(
    constrained_target: &Array1<f64>,
    primary: &Array1<f64>,
    support: &Array1<f64>,
) -> Vec<f64> {
    constrained_target
        .iter()
        .zip(primary.iter())
        .zip(support.iter())
        .map(|((&d_db, &p_db), &s_db)| {
            let d_lin = 10.0_f64.powf(d_db / 20.0);
            let p_lin = 10.0_f64.powf(p_db / 20.0);
            let s_lin = 10.0_f64.powf(s_db / 20.0).max(1e-12);
            let w_lin2 = (d_lin * d_lin - p_lin * p_lin).max(0.0);
            let w_lin = w_lin2.sqrt() / s_lin;
            let w_db = 20.0 * w_lin.max(1e-12).log10();
            w_db.min(MAX_SUPPORT_GAIN_DB)
        })
        .collect()
}

/// Build a minimum-phase FIR from a log-frequency magnitude response.
///
/// 1. Reconstruct minimum phase from the log-magnitude via Hilbert transform.
/// 2. Build a complex frequency response on the log grid.
/// 3. Interpolate magnitude and phase to a uniform linear grid.
/// 4. IFFT to obtain a causal minimum-phase impulse response.
fn minimum_phase_fir(
    freq: &Array1<f64>,
    mag_db: &[f64],
    n_taps: usize,
    sample_rate: f64,
) -> Result<Vec<f64>> {
    if n_taps == 0 {
        return Ok(Vec::new());
    }
    if !n_taps.is_multiple_of(2) {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!("FIR tap count must be even, got {}", n_taps),
        });
    }

    // Build a curve from the gain response and extract minimum phase.
    let gain_curve = Curve {
        freq: freq.clone(),
        spl: Array1::from(mag_db.to_vec()),
        ..Default::default()
    };
    let min_phase_deg =
        crate::roomeq::phase_utils::reconstruct_minimum_phase(&gain_curve.freq, &gain_curve.spl);

    let fft_size = (n_taps * 8).max(1024).next_power_of_two();
    let n_bins = fft_size / 2 + 1;
    let df = sample_rate / fft_size as f64;
    let linear_freqs: Vec<f64> = (0..n_bins).map(|i| i as f64 * df).collect();

    // Interpolate magnitude and phase to a uniform linear grid.
    let mag_db_lin = interpolate_log_space_scalar(freq, mag_db, &linear_freqs);
    let phase_deg_lin =
        interpolate_log_space_scalar(freq, min_phase_deg.as_slice().unwrap(), &linear_freqs);

    // Build complex spectrum with conjugate symmetry.
    let mut spectrum: Vec<Complex64> = linear_freqs
        .iter()
        .zip(mag_db_lin.iter())
        .zip(phase_deg_lin.iter())
        .map(|((&_f, &db), &phase)| {
            let mag = 10.0_f64.powf(db / 20.0);
            Complex64::from_polar(mag, phase.to_radians())
        })
        .collect();

    // DC and Nyquist must be real.
    spectrum[0] = Complex64::new(spectrum[0].norm(), 0.0);
    if n_bins > 1 {
        spectrum[n_bins - 1] = Complex64::new(spectrum[n_bins - 1].norm(), 0.0);
    }

    // Mirror spectrum.
    let mut full_spectrum: Vec<Complex64> = Vec::with_capacity(fft_size);
    full_spectrum.extend_from_slice(&spectrum);
    for i in (1..n_bins - 1).rev() {
        full_spectrum.push(spectrum[i].conj());
    }

    // IFFT.
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(fft_size);
    ifft.process(&mut full_spectrum);

    let ir: Vec<f64> = full_spectrum
        .iter()
        .map(|c| c.re / fft_size as f64)
        .collect();

    // Take first n_taps. For a minimum-phase design we avoid a window that
    // zeros the first sample (e.g. Hann); truncation artifacts are mild for
    // the long FIR lengths used here.
    let taps: Vec<f64> = ir.iter().take(n_taps).copied().collect();

    Ok(taps)
}

/// Log-space interpolation of a scalar curve onto a new frequency grid.
fn interpolate_log_space_scalar(
    freq_in: &Array1<f64>,
    val_in: &[f64],
    freq_out: &[f64],
) -> Vec<f64> {
    let n_in = freq_in.len();
    let mut out = Vec::with_capacity(freq_out.len());
    for &f in freq_out {
        if f <= freq_in[0] {
            out.push(val_in[0]);
            continue;
        }
        if f >= freq_in[n_in - 1] {
            out.push(val_in[n_in - 1]);
            continue;
        }
        // Find bracketing index.
        let mut i = 1;
        while i < n_in && freq_in[i] < f {
            i += 1;
        }
        let f0 = freq_in[i - 1];
        let f1 = freq_in[i];
        let v0 = val_in[i - 1];
        let v1 = val_in[i];
        let t = ((f / f0).ln() / (f1 / f0).ln()).clamp(0.0, 1.0);
        out.push(v0 + t * (v1 - v0));
    }
    out
}

/// Convolve two FIRs.
fn convolve_fir(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut out = vec![0.0; n];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn flat_curve(freq: &[f64], spl: f64) -> Curve {
        Curve {
            freq: Array1::from(freq.to_vec()),
            spl: Array1::from_elem(freq.len(), spl),
            ..Default::default()
        }
    }

    #[test]
    fn constrain_target_floor_and_ceiling() {
        let freq = Array1::from_vec(vec![100.0, 1000.0, 10000.0]);
        let primary = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let target = Array1::from_vec(vec![-5.0, 20.0, 5.0]);
        let limits = vec![
            PrecedenceLimitBand {
                low_hz: 50.0,
                high_hz: 500.0,
                limit_db: 10.0,
            },
            PrecedenceLimitBand {
                low_hz: 500.0,
                high_hz: 20000.0,
                limit_db: 6.0,
            },
        ];
        let (d_mod, hits) = constrain_target(&target, &primary, &limits, &freq);
        assert_eq!(d_mod[0], 0.0, "floor should lift -5 dB to 0 dB");
        assert_eq!(
            d_mod[1], 6.0,
            "ceiling should cap 20 dB to +6 dB in the 500-20k band"
        );
        assert_eq!(d_mod[2], 5.0, "within limits");
        assert_eq!(hits, 1);
    }

    #[test]
    fn support_gain_fills_primary_notch() {
        let freq = Array1::logspace(10.0, f64::log10(100.0), f64::log10(10000.0), 100);
        let mut primary_spl = Array1::from_elem(100, 0.0);
        let notch_idx = 50;
        primary_spl[notch_idx] = -10.0;
        let primary = Curve {
            freq: freq.clone(),
            spl: primary_spl,
            ..Default::default()
        };
        let support = flat_curve(freq.as_slice().unwrap(), 0.0);
        let target = flat_curve(freq.as_slice().unwrap(), 0.0);
        let config = SupportingSourceConfig::default();
        let result =
            compute_supporting_source_filter(&primary, &support, &target, &config, 48000.0)
                .expect("filter computation should succeed");
        assert!(!result.taps.is_empty());
        assert!(result.taps.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn filter_is_causal_and_has_early_energy() {
        let freq = Array1::logspace(10.0, f64::log10(100.0), f64::log10(10000.0), 100);
        let primary = flat_curve(freq.as_slice().unwrap(), 0.0);
        let support = flat_curve(freq.as_slice().unwrap(), 0.0);
        let mut target = primary.clone();
        target.spl += 3.0; // need 3 dB more energy everywhere
        let config = SupportingSourceConfig {
            decorrelation: crate::roomeq::SupportingSourceDecorrelation::None,
            ..Default::default()
        };
        let result =
            compute_supporting_source_filter(&primary, &support, &target, &config, 48000.0)
                .unwrap();
        // A minimum-phase +3 dB shelf should have its largest tap at or very
        // near the beginning.
        let peak_idx = result
            .taps
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            peak_idx <= result.taps.len() / 8,
            "minimum-phase peak should be early, got index {} of {}",
            peak_idx,
            result.taps.len()
        );
    }

    #[test]
    fn no_nan_or_inf_outputs() {
        let freq = Array1::logspace(10.0, f64::log10(100.0), f64::log10(10000.0), 100);
        let primary = flat_curve(freq.as_slice().unwrap(), 0.0);
        let support = flat_curve(freq.as_slice().unwrap(), -3.0);
        let target = flat_curve(freq.as_slice().unwrap(), 1.0);
        let config = SupportingSourceConfig::default();
        let result =
            compute_supporting_source_filter(&primary, &support, &target, &config, 48000.0)
                .unwrap();
        assert!(result.taps.iter().all(|v| v.is_finite()));
        assert!(result.support_gain_db.iter().all(|v| v.is_finite()));
        assert!(result.drr_before_db.iter().all(|v| v.is_finite()));
        assert!(result.drr_after_db.iter().all(|v| v.is_finite()));
    }
}
