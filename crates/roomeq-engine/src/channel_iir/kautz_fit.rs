//! Fixed-pole fitting of the dry-plus-bank Kautz playback transfer.
//!
//! Linear basis coefficients are searched against a logarithmic magnitude
//! objective. Bounds apply to the sampled composite transfer, not coefficients.
//! The guard grid includes DC, Nyquist, measurement bins, and pole neighborhoods;
//! it is a numerical guard, not a continuous-frequency extremum certificate.

use std::ops::RangeInclusive;

use autoeq_core::{AutoeqError, Curve, Result};
use math_audio_iir_fir::KautzFilter;
use num_complex::Complex64;
use roomeq_model::OptimizerConfig;

fn invalid(message: &str) -> AutoeqError {
    AutoeqError::OptimizationFailed {
        message: format!("Kautz playback fit: {message}"),
    }
}

/// Fit real coefficients using the same complex response as playback.
pub(super) fn fit_playback_gains(
    bank: &mut KautzFilter,
    correction: &Curve,
    band: RangeInclusive<f64>,
    config: &OptimizerConfig,
) -> Result<()> {
    if bank.sections.is_empty()
        || !bank.srate.is_finite()
        || bank.srate <= 0.0
        || correction.freq.len() < 2
        || correction.freq.len() != correction.spl.len()
        || correction
            .freq
            .iter()
            .any(|f| !f.is_finite() || *f <= 0.0 || *f >= bank.srate / 2.0)
        || correction
            .freq
            .windows(2)
            .into_iter()
            .any(|pair| pair[0] >= pair[1])
        || correction.spl.iter().any(|value| !value.is_finite())
        || !config.min_db.is_finite()
        || !config.max_db.is_finite()
        || config.min_db > 0.0
        || config.max_db < 0.0
        || config.max_iter == 0
    {
        return Err(invalid(
            "requires finite ordered data, gain bounds containing unity, and a nonzero search budget",
        ));
    }
    let min_power = 10.0_f64.powf(config.min_db / 10.0);
    let max_power = 10.0_f64.powf(config.max_db / 10.0);
    if min_power <= 0.0 || !max_power.is_finite() {
        return Err(invalid("gain bounds exceed the numerical power range"));
    }

    let mut frequencies = correction.freq.to_vec();
    let measured_count = frequencies.len();
    let mut targets = correction.spl.to_vec();
    let mut weights = Vec::with_capacity(measured_count);
    for i in 0..measured_count {
        let left = if i == 0 {
            frequencies[i]
        } else {
            frequencies[i - 1]
        };
        let right = if i + 1 == measured_count {
            frequencies[i]
        } else {
            frequencies[i + 1]
        };
        weights.push(0.5 * (right.ln() - left.ln()));
        // Outside active correction support, penalize departure from unity,
        // not the loudspeaker's natural rolloff. No response is extrapolated.
        if !band.contains(&frequencies[i]) {
            targets[i] = 0.0;
        }
    }
    let total_weight: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total_weight;
    }

    // Logarithmic guards cover the digital band including both endpoints.
    // Pole-local guards resolve narrow modes even on sparse measurement grids.
    const GUARD_INTERVALS: usize = 4096;
    frequencies.push(0.0);
    let nyquist = bank.srate / 2.0;
    let guard_low = correction.freq[0].min(1.0).min(nyquist / 2.0);
    for i in 0..=GUARD_INTERVALS {
        frequencies.push(guard_low * (nyquist / guard_low).powf(i as f64 / GUARD_INTERVALS as f64));
    }
    for section in &bank.sections {
        let bandwidth = -section.pole_radius.ln() * bank.srate / std::f64::consts::PI;
        for offset in -128..=128 {
            let frequency = section.pole_freq + offset as f64 * bandwidth / 16.0;
            if (0.0..=nyquist).contains(&frequency) {
                frequencies.push(frequency);
            }
        }
    }
    let mut columns = vec![Vec::with_capacity(frequencies.len()); bank.sections.len()];
    for frequency in frequencies {
        let mut chain = Complex64::new(1.0, 0.0);
        for (section, column) in bank.sections.iter().zip(&mut columns) {
            column.push(section.basis_response(frequency, bank.srate, chain));
            chain *= section.allpass_response(frequency, bank.srate);
        }
    }
    if columns
        .iter()
        .flatten()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(invalid("nonfinite basis response"));
    }
    let scales: Vec<f64> = columns
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|value| value.norm())
                .fold(0.0_f64, f64::max)
        })
        .collect();
    if scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(invalid("nonfinite or degenerate basis"));
    }
    for (column, scale) in columns.iter_mut().zip(&scales) {
        for value in column {
            *value /= *scale;
        }
    }
    let mut response = vec![Complex64::new(1.0, 0.0); columns[0].len()];
    let mut coefficients = vec![0.0; columns.len()];
    let loss = |response: &[Complex64], column: &[Complex64], delta: f64| -> f64 {
        let mut sum = 0.0;
        for (i, (current, basis)) in response.iter().zip(column).enumerate() {
            let power = (*current + *basis * delta).norm_sqr();
            if !power.is_finite() || power < min_power || power > max_power {
                return f64::INFINITY;
            }
            if i < measured_count {
                sum += weights[i] * (10.0 * power.log10() - targets[i]).powi(2);
            }
        }
        sum
    };
    let mut best_loss = loss(&response, &columns[0], 0.0);
    if !best_loss.is_finite() {
        return Err(invalid("nonfinite baseline objective"));
    }
    // Each normalized coordinate initially changes bank magnitude by at most
    // 0.5 on the guard grid. Halving provides finer feasible moves near bounds.
    let mut step = 0.5;
    for _ in 0..config.max_iter {
        let mut improved = false;
        for (column, coefficient) in columns.iter().zip(&mut coefficients) {
            let mut best_delta = 0.0;
            for delta in [-step, step] {
                let candidate = loss(&response, column, delta);
                if candidate < best_loss {
                    best_loss = candidate;
                    best_delta = delta;
                }
            }
            if best_delta != 0.0 {
                *coefficient += best_delta;
                for (current, basis) in response.iter_mut().zip(column) {
                    *current += *basis * best_delta;
                }
                improved = true;
            }
        }
        if !improved {
            step *= 0.5;
            // Numerical amplitude resolution, not an audibility threshold.
            if step < 1e-7 {
                break;
            }
        }
    }
    for ((section, coefficient), scale) in bank.sections.iter_mut().zip(coefficients).zip(scales) {
        section.gain = coefficient / scale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn curve_from_bank(bank: &KautzFilter) -> Curve {
        let freq = Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 601);
        let spl = freq.mapv(|f| {
            20.0 * (Complex64::new(1.0, 0.0) + bank.complex_response(f))
                .norm()
                .log10()
        });
        Curve {
            freq,
            spl,
            ..Curve::default()
        }
    }

    #[test]
    fn kautz_playback_fit_recovers_representable_complex_bank_across_rates() {
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let modes = [(75.0, 8.0), (135.0, 10.0)];
            let mut oracle = KautzFilter::from_room_modes(&modes, rate);
            oracle.sections[0].gain = -0.025;
            oracle.sections[1].gain = 0.018;
            let correction = curve_from_bank(&oracle);
            let mut fitted = KautzFilter::from_room_modes(&modes, rate);
            let config = OptimizerConfig {
                min_db: -3.0,
                max_db: 3.0,
                max_iter: 800,
                ..OptimizerConfig::default()
            };
            fit_playback_gains(&mut fitted, &correction, 20.0..=20_000.0, &config).unwrap();
            // The validation grid differs from both the data and guard grids.
            let mut max_error = 0.0_f64;
            for i in 0..=3000 {
                let f = 20.0 * 1000.0_f64.powf(i as f64 / 3000.0);
                let expected = Complex64::new(1.0, 0.0) + oracle.complex_response(f);
                let actual = Complex64::new(1.0, 0.0) + fitted.complex_response(f);
                max_error = max_error.max((actual - expected).norm());
            }
            assert!(max_error < 1e-5, "{rate}: complex error {max_error}");
        }
    }

    #[test]
    fn kautz_playback_fit_enforces_realized_limits_on_independent_dense_grid() {
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let mut bank = KautzFilter::from_room_modes(&[(70.0, 20.0), (103.0, 12.0)], rate);
            let mut correction = curve_from_bank(&bank);
            correction.spl.fill(12.0);
            let config = OptimizerConfig {
                min_db: -0.5,
                max_db: 0.2,
                max_iter: 300,
                ..OptimizerConfig::default()
            };
            fit_playback_gains(&mut bank, &correction, 20.0..=20_000.0, &config).unwrap();
            for i in 0..=32768 {
                let f = if i == 0 {
                    0.0
                } else {
                    (rate / 2.0).powf(i as f64 / 32768.0)
                };
                let gain = 20.0
                    * (Complex64::new(1.0, 0.0) + bank.complex_response(f))
                        .norm()
                        .log10();
                assert!(
                    (-0.501..=0.201).contains(&gain),
                    "{rate} Hz at {f} Hz: {gain} dB"
                );
            }
        }
    }

    #[test]
    fn kautz_playback_fit_preserves_identity_and_ignores_out_of_band_target() {
        let mut bank = KautzFilter::from_room_modes(&[(100.0, 8.0)], 48_000.0);
        let mut correction = curve_from_bank(&bank);
        for (&frequency, target) in correction.freq.iter().zip(&mut correction.spl) {
            if !(50.0..=200.0).contains(&frequency) {
                *target = 15.0;
            }
        }
        bank.sections[0].gain = 99.0;
        fit_playback_gains(
            &mut bank,
            &correction,
            50.0..=200.0,
            &OptimizerConfig::default(),
        )
        .unwrap();
        assert_eq!(bank.sections[0].gain, 0.0);
    }

    #[test]
    fn kautz_playback_fit_rejects_invalid_power_or_data() {
        let mut bank = KautzFilter::from_room_modes(&[(100.0, 8.0)], 48_000.0);
        let correction = curve_from_bank(&bank);
        for config in [
            OptimizerConfig {
                min_db: 1.0,
                ..OptimizerConfig::default()
            },
            OptimizerConfig {
                max_db: -1.0,
                ..OptimizerConfig::default()
            },
            OptimizerConfig {
                max_db: f64::NAN,
                ..OptimizerConfig::default()
            },
            OptimizerConfig {
                max_iter: 0,
                ..OptimizerConfig::default()
            },
        ] {
            assert!(fit_playback_gains(&mut bank, &correction, 20.0..=20_000.0, &config).is_err());
        }
        let mut invalid_curve = correction.clone();
        invalid_curve.freq[1] = invalid_curve.freq[0];
        assert!(
            fit_playback_gains(
                &mut bank,
                &invalid_curve,
                20.0..=20_000.0,
                &OptimizerConfig::default()
            )
            .is_err()
        );
    }
}
