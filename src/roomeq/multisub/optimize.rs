use super::super::types::{MeasurementSource, OptimizerConfig};
use super::allpass::allpass_frequency_bounds;
use super::allpass::compute_combined_with_allpass;
use super::multisub_allpass_converged;
use super::multisub_allpass_loss;
use super::types::MultiSubAllPassResult;
use crate::Curve;
use crate::loss::{CrossoverType, DriverMeasurement, DriversLossData};
use crate::optim::scalar::{ScalarOptimConfig, optimize_bounded_scalar};
use crate::read as load;
use crate::workflow::DriverOptimizationResult;
use log::{info, warn};
use std::error::Error;

/// Optimize multi-subwoofer configuration
///
/// # Arguments
/// * `measurements` - List of subwoofer measurements (sources)
/// * `config` - Optimizer configuration
/// * `sample_rate` - Sample rate
///
/// # Returns
/// * Tuple of (DriverOptimizationResult, Combined Curve)
///
/// # Note on Phase Data
/// For accurate optimization, measurements should include phase data.
/// The optimizer uses complex summation to model constructive/destructive
/// interference between subwoofers. Without phase data, the optimizer
/// assumes 0° phase for all measurements, which may result in suboptimal
/// delay settings.
pub fn optimize_multisub(
    measurements: &[MeasurementSource],
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<(DriverOptimizationResult, Curve), Box<dyn Error>> {
    // Load all measurements and check for phase data
    let mut driver_measurements = Vec::new();
    let mut missing_phase_count = 0;

    for source in measurements {
        let curve = load::load_source(source)?;
        if curve.phase.is_none() {
            missing_phase_count += 1;
        }
        driver_measurements.push(DriverMeasurement {
            freq: curve.freq,
            spl: curve.spl,
            phase: curve.phase, // Critical: use phase for accurate summation
        });
    }

    // Warn if phase data is missing
    if missing_phase_count > 0 {
        warn!(
            "Multi-sub optimization: {} of {} measurements are missing phase data. \
            This may result in inaccurate delay optimization. \
            For best results, include phase data in your measurements (e.g., export from REW with phase).",
            missing_phase_count,
            measurements.len()
        );
    }

    // Create drivers data with NO crossover filtering
    let drivers_data = DriversLossData::new(driver_measurements, CrossoverType::None);

    let result = crate::workflow::optimize_multisub(
        drivers_data.clone(),
        config.min_freq,
        config.max_freq,
        sample_rate,
        &config.algorithm,
        config.max_iter,
        config.population,
        config.min_db,
        config.max_db,
        config.seed,
    )?;

    // Compute combined response
    let combined_response = crate::loss::compute_drivers_combined_response(
        &drivers_data,
        &result.gains,
        &[], // no crossovers
        Some(&result.delays),
        sample_rate,
    );

    let combined_curve = Curve {
        freq: drivers_data.freq_grid.clone(),
        spl: combined_response,
        phase: None,
        ..Default::default()
    };

    Ok((result, combined_curve))
}

/// Optimize multi-subwoofer configuration with per-sub all-pass filters.
///
/// Extends the standard gain+delay optimization with one all-pass biquad filter
/// per subwoofer. The all-pass filter adds phase rotation without changing magnitude,
/// enabling better cancellation of room modes through improved phase alignment.
///
/// Parameter vector layout: [gains(N), delays(N), ap_freq(N), ap_q(N)]
///
/// Inspired by Brännmark, Rosencratz & Andersson.
pub fn optimize_multisub_with_allpass(
    measurements: &[MeasurementSource],
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<MultiSubAllPassResult, Box<dyn Error>> {
    // Load measurements
    let mut driver_measurements = Vec::new();
    let mut missing_phase_count = 0;

    for source in measurements {
        let curve = load::load_source(source)?;
        if curve.phase.is_none() {
            missing_phase_count += 1;
        }
        driver_measurements.push(DriverMeasurement {
            freq: curve.freq,
            spl: curve.spl,
            phase: curve.phase,
        });
    }

    if missing_phase_count > 0 {
        warn!(
            "Multi-sub all-pass optimization: {} of {} measurements are missing phase data.",
            missing_phase_count,
            measurements.len()
        );
    }

    let drivers_data = DriversLossData::new(driver_measurements, CrossoverType::None);
    let n_drivers = drivers_data.drivers.len();

    // Parameter vector: [gains(N), delays(N), ap_freq(N), ap_q(N)]
    let n_params = n_drivers * 4;

    let (allpass_min_freq, allpass_max_freq) = allpass_frequency_bounds(config)?;

    // Bounds
    let mut lower_bounds = Vec::with_capacity(n_params);
    let mut upper_bounds = Vec::with_capacity(n_params);

    // Gains: [-max_db, max_db]
    for _ in 0..n_drivers {
        lower_bounds.push(-config.max_db);
        upper_bounds.push(config.max_db);
    }
    // Delays: [0, 20] ms
    for _ in 0..n_drivers {
        lower_bounds.push(0.0);
        upper_bounds.push(20.0);
    }
    // All-pass frequencies: [min_freq, max_freq]
    for _ in 0..n_drivers {
        lower_bounds.push(allpass_min_freq);
        upper_bounds.push(allpass_max_freq); // sub range
    }
    // All-pass Q: [0.3, 5.0]
    for _ in 0..n_drivers {
        lower_bounds.push(0.3);
        upper_bounds.push(5.0);
    }

    // Initial guess: zeros for gains, zeros for delays, 60 Hz + Q=1 for allpass
    let mut x = vec![0.0; n_params];
    for i in 0..n_drivers {
        x[2 * n_drivers + i] = 60.0_f64.clamp(allpass_min_freq, allpass_max_freq); // initial AP frequency
        x[3 * n_drivers + i] = 1.0; // initial AP Q
    }

    // Pre-objective
    let pre_obj = multisub_allpass_loss(
        &drivers_data,
        &x,
        sample_rate,
        config.min_freq,
        config.max_freq,
    );

    // Use DE optimizer (global search needed for all-pass parameters)
    let drivers_data_clone = drivers_data.clone();
    let min_freq = config.min_freq;
    let max_freq = config.max_freq;

    let objective_fn = move |params: &[f64]| -> f64 {
        multisub_allpass_loss(&drivers_data_clone, params, sample_rate, min_freq, max_freq)
    };

    // Build bounds as (lower, upper) pairs
    let bounds: Vec<(f64, f64)> = lower_bounds
        .iter()
        .zip(upper_bounds.iter())
        .map(|(&l, &u)| (l, u))
        .collect();

    let opt_result = optimize_bounded_scalar(
        &bounds,
        &x,
        &ScalarOptimConfig {
            algorithm: config.algorithm.clone(),
            max_iter: config.max_iter,
            population: config.population,
            tolerance: config.tolerance,
            atolerance: config.atolerance,
            strategy: config.strategy.clone(),
            seed: config.seed,
        },
        objective_fn,
    )
    .map_err(|e| format!("all-pass optimization failed: {e}"))?;

    x = opt_result.x;
    let post_obj = opt_result.fun;
    let converged = multisub_allpass_converged(opt_result.success);

    info!(
        "Multi-sub all-pass optimization: pre={:.4}, post={:.4}, improvement={:.2} dB, converged={}",
        pre_obj,
        post_obj,
        pre_obj - post_obj,
        converged
    );
    if !converged {
        warn!(
            "Multi-sub all-pass optimizer did not report convergence ({}): {}",
            opt_result.algorithm, opt_result.message
        );
    }

    // Extract results
    let gains = x[0..n_drivers].to_vec();
    let delays = x[n_drivers..2 * n_drivers].to_vec();
    let mut allpass_filters = Vec::with_capacity(n_drivers);
    for i in 0..n_drivers {
        let freq = x[2 * n_drivers + i];
        let q = x[3 * n_drivers + i];
        allpass_filters.push((freq, q));
        info!(
            "  Sub {}: gain={:.1} dB, delay={:.1} ms, AP: {:.0} Hz Q={:.2}",
            i, gains[i], delays[i], freq, q
        );
    }

    // Compute combined response with all-pass filters applied
    let combined_spl = compute_combined_with_allpass(
        &drivers_data,
        &gains,
        &delays,
        &allpass_filters,
        sample_rate,
    );

    let combined_curve = Curve {
        freq: drivers_data.freq_grid.clone(),
        spl: combined_spl,
        phase: None,
        ..Default::default()
    };

    Ok(MultiSubAllPassResult {
        base: DriverOptimizationResult {
            gains,
            delays,
            crossover_freqs: vec![],
            pre_objective: pre_obj,
            post_objective: post_obj,
            converged,
        },
        allpass_filters,
        combined_curve,
    })
}

#[cfg(test)]
mod tests {
    use super::{optimize_multisub, optimize_multisub_with_allpass};
    use crate::{Curve, MeasurementSource, OptimizerConfig};
    use ndarray::Array1;

    fn log_freq_grid(n: usize, lo: f64, hi: f64) -> Array1<f64> {
        Array1::from_vec(
            (0..n)
                .map(|i| lo * (hi / lo).powf(i as f64 / (n - 1) as f64))
                .collect(),
        )
    }

    fn flat_sub_curve(phase: Option<Array1<f64>>) -> Curve {
        let n = 16;
        let freq = log_freq_grid(n, 20.0, 200.0);
        let spl = Array1::from_elem(n, 75.0_f64);
        Curve {
            freq,
            spl,
            phase,
            ..Default::default()
        }
    }

    fn tiny_optimizer_config() -> OptimizerConfig {
        OptimizerConfig {
            min_freq: 20.0,
            max_freq: 200.0,
            algorithm: "autoeq:de".to_string(),
            max_iter: 20,
            population: 10,
            seed: Some(1),
            min_db: -12.0,
            max_db: 12.0,
            ..Default::default()
        }
    }

    #[test]
    fn optimize_multisub_happy_path_with_phase() {
        let phase = Some(Array1::from_elem(16, 0.0_f64));
        let c1 = flat_sub_curve(phase.clone());
        let c2 = flat_sub_curve(phase);
        let sources = vec![
            MeasurementSource::InMemory(c1),
            MeasurementSource::InMemory(c2),
        ];
        let config = tiny_optimizer_config();

        let result = optimize_multisub(&sources, &config, 48000.0);
        assert!(
            result.is_ok(),
            "optimize_multisub should return Ok for in-memory curves with phase: {:?}",
            result.err()
        );

        let (_, combined) = result.unwrap();
        assert!(
            !combined.freq.is_empty(),
            "combined curve frequency grid must not be empty"
        );
        assert!(
            !combined.spl.is_empty(),
            "combined curve SPL must not be empty"
        );
        assert!(
            combined.spl.iter().all(|v| v.is_finite()),
            "combined SPL must be finite"
        );
    }

    #[test]
    fn optimize_multisub_missing_phase_returns_ok() {
        let c1 = flat_sub_curve(None);
        let c2 = flat_sub_curve(None);
        let sources = vec![
            MeasurementSource::InMemory(c1),
            MeasurementSource::InMemory(c2),
        ];
        let config = tiny_optimizer_config();

        let result = optimize_multisub(&sources, &config, 48000.0);
        assert!(
            result.is_ok(),
            "optimize_multisub should return Ok even when phase is missing: {:?}",
            result.err()
        );

        let (_, combined) = result.unwrap();
        assert!(
            !combined.freq.is_empty(),
            "combined curve frequency grid must not be empty"
        );
        assert!(
            !combined.spl.is_empty(),
            "combined curve SPL must not be empty"
        );
        assert!(
            combined.spl.iter().all(|v| v.is_finite()),
            "combined SPL must be finite"
        );
    }

    #[test]
    fn optimize_multisub_with_allpass_happy_path() {
        let phase = Some(Array1::from_elem(16, 0.0_f64));
        let c1 = flat_sub_curve(phase.clone());
        let c2 = flat_sub_curve(phase);
        let sources = vec![
            MeasurementSource::InMemory(c1),
            MeasurementSource::InMemory(c2),
        ];
        let config = tiny_optimizer_config();

        let result = optimize_multisub_with_allpass(&sources, &config, 48000.0);
        assert!(
            result.is_ok(),
            "optimize_multisub_with_allpass should return Ok: {:?}",
            result.err()
        );

        let res = result.unwrap();
        assert!(
            !res.combined_curve.freq.is_empty(),
            "combined curve frequency grid must not be empty"
        );
        assert!(
            !res.combined_curve.spl.is_empty(),
            "combined curve SPL must not be empty"
        );
        assert!(
            res.combined_curve.spl.iter().all(|v| v.is_finite()),
            "combined SPL must be finite"
        );
        assert_eq!(
            res.allpass_filters.len(),
            2,
            "should have one all-pass filter per sub"
        );
    }

    #[test]
    fn optimize_multisub_with_allpass_zero_width_range_errors() {
        let phase = Some(Array1::from_elem(16, 0.0_f64));
        let c1 = flat_sub_curve(phase.clone());
        let c2 = flat_sub_curve(phase);
        let sources = vec![
            MeasurementSource::InMemory(c1),
            MeasurementSource::InMemory(c2),
        ];
        let mut config = tiny_optimizer_config();
        config.min_freq = 200.0;
        config.max_freq = 200.0;

        let result = optimize_multisub_with_allpass(&sources, &config, 48000.0);
        assert!(
            result.is_err(),
            "zero-width allpass frequency range should produce an error"
        );

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("non-zero frequency range"),
            "error should mention non-zero frequency range: {}",
            err
        );
    }
}
