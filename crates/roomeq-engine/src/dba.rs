//! Double Bass Array (DBA) optimization
//!
//! # Phase Data Requirement
//!
//! DBA optimization relies on complex summation to model the interaction between
//! front and rear subwoofer arrays. For accurate optimization, measurements should
//! include phase data. Missing phase is rejected rather than replaced with an
//! invented 0° phase response.
//!
//! The rear array is automatically inverted (180° phase shift) to create the
//! pressure wave cancellation pattern characteristic of DBA systems.

use crate::Curve;
use crate::config_adapter::OptimizerConfigExt;
use autoeq_optim::DriverOptimizationResult;
use autoeq_optim::loss::{DriverMeasurement, DriversLossData};
use autoeq_optim::{CrossoverType, LossType};
use ndarray::Array1;
use std::error::Error;

const MIN_COMPLEX_MAGNITUDE: f64 = 1e-12;

fn complex_to_spl_phase(z: num_complex::Complex64) -> (f64, f64) {
    let magnitude = z.norm();
    if !magnitude.is_finite() || magnitude <= MIN_COMPLEX_MAGNITUDE {
        return (20.0 * MIN_COMPLEX_MAGNITUDE.log10(), 0.0);
    }
    (20.0 * magnitude.log10(), z.arg().to_degrees())
}

use roomeq_model::OptimizerConfig;

/// Prepared, in-memory input for Double Bass Array optimization.
pub struct DbaPreparedInput {
    pub front: Vec<Curve>,
    pub rear: Vec<Curve>,
}

pub struct DbaOptimizationResult {
    pub driver: DriverOptimizationResult,
    pub combined_curve: Curve,
    pub optimizer_evidence: autoeq_optim::optim::OptimizerRunEvidence,
}

/// Optimize Double Bass Array configuration
///
/// # Arguments
/// * `dba_config` - DBA configuration (front/rear sources)
/// * `config` - Optimizer configuration
/// * `sample_rate` - Sample rate
///
/// # Returns
/// * Tuple of (DriverOptimizationResult, Combined Curve)
///   Result contains 2 entries: Index 0 = Front, Index 1 = Rear
///
/// # Note on Phase Data
/// For accurate DBA optimization, measurements should include phase data.
/// The optimizer uses complex summation to model constructive/destructive
/// interference between front and rear arrays.
pub fn optimize_dba(
    input: &DbaPreparedInput,
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<(DriverOptimizationResult, Curve), Box<dyn Error>> {
    let result = optimize_dba_detailed(input, config, sample_rate)?;
    Ok((result.driver, result.combined_curve))
}

pub fn optimize_dba_detailed(
    input: &DbaPreparedInput,
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<DbaOptimizationResult, Box<dyn Error>> {
    // 1. Load and Sum Front Array
    let front_curve = sum_array_response(&input.front)?;

    // 2. Load and Sum Rear Array
    let rear_curve = sum_array_response(&input.rear)?;

    // 3. Create optimization targets
    // We have 2 "drivers": Front Aggregate and Rear Aggregate
    // Front is fixed (Gain 0, Delay 0)
    // Rear is optimized (Gain, Delay)
    // DBA implies Rear is INVERTED relative to Front.
    // We add 180 degrees to Rear phase to simulate inversion.

    let rear_curve_inverted = invert_polarity(&rear_curve);

    let driver_measurements = vec![
        DriverMeasurement {
            freq: front_curve.freq.clone(),
            spl: front_curve.spl.clone(),
            phase: front_curve.phase.clone(),
        },
        DriverMeasurement {
            freq: rear_curve_inverted.freq.clone(),
            spl: rear_curve_inverted.spl.clone(),
            phase: rear_curve_inverted.phase.clone(),
        },
    ];

    // DBA's driver order is semantic (front first, rear second).  The
    // sorting constructor reorders by frequency and can silently swap these
    // two arrays when their measurement ranges differ.
    let drivers_data = DriversLossData::new_ordered(driver_measurements, CrossoverType::None);

    // 4. Custom optimization
    // We can't use standard optimize_multisub because it optimizes ALL gains/delays.
    // We want to lock Front parameters.
    // So we'll implement a constrained optimization here or use custom bounds.

    // Custom bounds:
    // Front: Gain [-0.1, 0.1], Delay [0, 0] (Tight bounds effectively lock it)
    // Rear: Gain [-20, 0], Delay [0, 50ms] (DBA usually attenuates rear slightly)

    // We'll reuse the workflow helpers but supply custom bounds.

    let mut optim_params = config.to_optim_params(sample_rate);
    optim_params.loss = LossType::MultiSubFlat;
    let objective_data = autoeq_optim::optim::setup::setup_multisub_objective_data(
        &optim_params,
        drivers_data.clone(),
    );

    // Custom bounds: [Gain1, Gain2, Delay1, Delay2]
    // Index 0: Front Gain -> 0 (Locked)
    // Index 1: Rear Gain -> config bounds (typically attenuated)
    // Index 2: Front Delay -> 0 (Locked)
    // Index 3: Rear Delay -> 0 to 100 ms (approx 34m room)

    // DBA rear array is for cancellation — clamp rear gain to 0 dB max
    // Honour the configured attenuation floor.  Only the upper bound is
    // constrained by the DBA polarity convention (rear gain must not boost).
    let min_gain = config.min_db;
    let max_gain = 0.0;

    let lower_bounds = vec![-0.01, min_gain, 0.0, 0.0];
    let upper_bounds = vec![0.01, max_gain, 0.001, 100.0];

    // Initial guess
    // Rear delay guess: 10ms (~3.4m room)
    // Rear gain guess: -3dB
    let mut x = vec![0.0, -3.0, 0.0, 10.0];
    let pre_objective = autoeq_optim::optim::compute_base_fitness(&x, &objective_data);

    // Optimize
    let opt_result = autoeq_optim::optim::optimize_filters(
        &mut x,
        &lower_bounds,
        &upper_bounds,
        objective_data.clone(),
        &optim_params,
    );
    let post_objective = autoeq_optim::optim::compute_base_fitness(&x, &objective_data);

    let optimizer_evidence = autoeq_optim::optim::OptimizerRunEvidence::from_backend_result(
        &optim_params.algo,
        opt_result,
        &x,
        &lower_bounds,
        &upper_bounds,
        optim_params.maxeval,
        optim_params.seed,
    );
    if optimizer_evidence.confidence == autoeq_optim::optim::OptimizerConfidence::Unusable {
        return Err(format!(
            "DBA optimizer produced unusable result: {}",
            optimizer_evidence.status
        )
        .into());
    }

    // Recompute scores
    // Note: compute_base_fitness uses args.loss_type which we set to MultiSubFlat
    // and uses setup_multisub_objective_data
    // So we can assume it works.

    let gains = vec![x[0], x[1]];
    let delays = vec![x[2], x[3]];
    let crossover_freqs = vec![];

    // Compute combined response
    let combined_curve = compute_dba_combined_curve(
        &front_curve,
        &rear_curve_inverted,
        &gains,
        &delays,
        &drivers_data.freq_grid,
        sample_rate,
    )?;

    Ok(DbaOptimizationResult {
        driver: DriverOptimizationResult {
            gains,
            delays,
            crossover_freqs,
            pre_objective,
            post_objective,
            converged: optimizer_evidence.converged,
        },
        combined_curve,
        optimizer_evidence,
    })
}

/// Sum multiple measurements into a single curve (complex summation)
///
/// # Phase Data
/// This function uses complex summation to properly model interference patterns.
/// If any measurement is missing phase data, this returns an error. DBA is a
/// phase-critical feature and should not invent coherence from magnitude-only
/// data.
pub fn sum_array_response(curves: &[Curve]) -> Result<Curve, Box<dyn Error>> {
    if curves.is_empty() {
        return Err("Empty array".into());
    }

    for (index, curve) in curves.iter().enumerate() {
        if curve.phase.is_none() {
            return Err(
                format!("DBA array summation requires phase data for curve {index}").into(),
            );
        }
    }

    // Reference freq from first
    let ref_freq = curves[0].freq.clone();

    // Sum complex
    use num_complex::Complex64;
    use std::f64::consts::PI;

    let mut sum_complex = Array1::<Complex64>::zeros(ref_freq.len());

    for curve in curves {
        // Interpolate to ref grid
        let interp = autoeq_core::interpolate_log_space(&ref_freq, curve);

        for i in 0..ref_freq.len() {
            let spl = interp.spl[i];
            let phase = interp
                .phase
                .as_ref()
                .ok_or("DBA interpolation lost required phase data")?[i];
            let m = 10.0_f64.powf(spl / 20.0);
            let phi = phase * PI / 180.0;
            sum_complex[i] += Complex64::from_polar(m, phi);
        }
    }

    // Phase is undefined at a null. Pin zero/near-zero complex sums to the
    // finite magnitude floor and a deterministic 0° phase instead of exposing
    // floating-point residue as an arbitrary angle.
    let (spl, phase): (Vec<_>, Vec<_>) = sum_complex
        .iter()
        .copied()
        .map(complex_to_spl_phase)
        .unzip();

    Ok(Curve {
        freq: ref_freq,
        spl: Array1::from(spl),
        phase: Some(Array1::from(phase)),
        ..Default::default()
    })
}

/// Invert polarity of a curve (add 180 deg)
fn invert_polarity(curve: &Curve) -> Curve {
    let mut new_curve = curve.clone();
    if let Some(ref mut phase) = new_curve.phase {
        *phase = phase.mapv(|p| p + 180.0);
    }
    new_curve
}

fn compute_dba_combined_curve(
    front_curve: &Curve,
    rear_curve: &Curve,
    gains: &[f64],
    delays_ms: &[f64],
    freq_grid: &Array1<f64>,
    _sample_rate: f64,
) -> Result<Curve, Box<dyn Error>> {
    use num_complex::Complex64;
    use std::f64::consts::PI;

    let front = autoeq_core::interpolate_log_space(freq_grid, front_curve);
    let rear = autoeq_core::interpolate_log_space(freq_grid, rear_curve);
    let front_phase = front
        .phase
        .as_ref()
        .ok_or("DBA combined curve requires front phase data")?;
    let rear_phase = rear
        .phase
        .as_ref()
        .ok_or("DBA combined curve requires rear phase data")?;
    let front_gain = gains.first().copied().unwrap_or(0.0);
    let rear_gain = gains.get(1).copied().unwrap_or(0.0);
    let front_delay_s = delays_ms.first().copied().unwrap_or(0.0) / 1000.0;
    let rear_delay_s = delays_ms.get(1).copied().unwrap_or(0.0) / 1000.0;

    let mut sum_complex = Array1::<Complex64>::zeros(freq_grid.len());
    for i in 0..freq_grid.len() {
        let f = freq_grid[i];
        let front_mag = 10.0_f64.powf((front.spl[i] + front_gain) / 20.0);
        let rear_mag = 10.0_f64.powf((rear.spl[i] + rear_gain) / 20.0);
        let front_phi = front_phase[i].to_radians() - 2.0 * PI * f * front_delay_s;
        let rear_phi = rear_phase[i].to_radians() - 2.0 * PI * f * rear_delay_s;
        sum_complex[i] =
            Complex64::from_polar(front_mag, front_phi) + Complex64::from_polar(rear_mag, rear_phi);
    }

    let (spl, phase): (Vec<_>, Vec<_>) = sum_complex
        .iter()
        .copied()
        .map(complex_to_spl_phase)
        .unzip();

    Ok(Curve {
        freq: freq_grid.clone(),
        spl: Array1::from(spl),
        phase: Some(Array1::from(phase)),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invert_polarity() {
        let freq = Array1::from(vec![100.0, 1000.0]);
        let spl = Array1::from(vec![80.0, 80.0]);
        let phase = Array1::from(vec![0.0, -90.0]);

        let curve = Curve {
            freq: freq.clone(),
            spl: spl.clone(),
            phase: Some(phase.clone()),
            ..Default::default()
        };

        let inverted = invert_polarity(&curve);

        let inv_phase = inverted.phase.unwrap();
        assert!((inv_phase[0] - 180.0).abs() < 1e-6);
        assert!((inv_phase[1] - 90.0).abs() < 1e-6); // -90 + 180 = 90
    }

    #[test]
    fn dba_result_reports_the_actual_pre_and_post_objectives() {
        let freq = Array1::logspace(10.0, f64::log10(20.0), f64::log10(200.0), 24);
        let front = Curve {
            spl: freq.mapv(|frequency| 80.0 + 2.0 * (frequency / 45.0).ln().sin()),
            freq: freq.clone(),
            phase: Some(Array1::from_elem(freq.len(), 0.0)),
            ..Curve::default()
        };
        let rear = Curve {
            spl: freq.mapv(|frequency| 76.0 - 1.5 * (frequency / 70.0).ln().cos()),
            freq: freq.clone(),
            phase: Some(Array1::from_elem(freq.len(), 0.0)),
            ..Curve::default()
        };
        let input = DbaPreparedInput {
            front: vec![front.clone()],
            rear: vec![rear.clone()],
        };
        let config = OptimizerConfig {
            algorithm: "autoeq:de".to_string(),
            max_iter: 12,
            population: 8,
            refine: false,
            min_freq: 20.0,
            max_freq: 200.0,
            min_db: -12.0,
            max_db: 6.0,
            seed: Some(7),
            parallel_threads: Some(1),
            ..OptimizerConfig::default()
        };

        let result = optimize_dba_detailed(&input, &config, 48_000.0).unwrap();
        let drivers = DriversLossData::new_ordered(
            vec![
                DriverMeasurement {
                    freq: front.freq,
                    spl: front.spl,
                    phase: front.phase,
                },
                DriverMeasurement {
                    freq: rear.freq.clone(),
                    spl: rear.spl.clone(),
                    phase: invert_polarity(&rear).phase,
                },
            ],
            CrossoverType::None,
        );
        let mut params = config.to_optim_params(48_000.0);
        params.loss = LossType::MultiSubFlat;
        let objective = autoeq_optim::optim::setup::setup_multisub_objective_data(&params, drivers);
        let expected_pre =
            autoeq_optim::optim::compute_base_fitness(&[0.0, -3.0, 0.0, 10.0], &objective);
        let expected_post = autoeq_optim::optim::compute_base_fitness(
            &[
                result.driver.gains[0],
                result.driver.gains[1],
                result.driver.delays[0],
                result.driver.delays[1],
            ],
            &objective,
        );

        assert!(expected_pre > 0.0 && expected_pre.is_finite());
        assert!((result.driver.pre_objective - expected_pre).abs() <= 1e-12);
        assert!((result.driver.post_objective - expected_post).abs() <= 1e-12);
    }

    #[test]
    fn sum_array_response_rejects_missing_phase() {
        let curve = Curve {
            freq: Array1::from(vec![50.0, 100.0]),
            spl: Array1::from(vec![80.0, 80.0]),
            phase: None,
            ..Default::default()
        };

        let err = sum_array_response(&[curve]).unwrap_err();
        assert!(
            err.to_string().contains("requires phase data"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sum_array_response_preserves_complex_phase() {
        let curve_a = Curve {
            freq: Array1::from(vec![100.0, 200.0]),
            spl: Array1::from(vec![80.0, 80.0]),
            phase: Some(Array1::from(vec![0.0, 0.0])),
            ..Default::default()
        };
        let curve_b = Curve {
            freq: Array1::from(vec![100.0, 200.0]),
            spl: Array1::from(vec![80.0, 80.0]),
            phase: Some(Array1::from(vec![90.0, 90.0])),
            ..Default::default()
        };

        let summed = sum_array_response(&[curve_a, curve_b]).unwrap();

        assert!(summed.phase.is_some());
        assert!((summed.phase.as_ref().unwrap()[0] - 45.0).abs() < 1e-6);
    }

    #[test]
    fn sum_array_response_exact_cancellation_has_finite_floor_and_phase() {
        let curve_a = Curve {
            freq: Array1::from(vec![100.0, 200.0]),
            spl: Array1::from(vec![0.0, 0.0]),
            phase: Some(Array1::from(vec![0.0, 0.0])),
            ..Default::default()
        };
        let curve_b = Curve {
            freq: Array1::from(vec![100.0, 200.0]),
            spl: Array1::from(vec![0.0, 0.0]),
            phase: Some(Array1::from(vec![180.0, 180.0])),
            ..Default::default()
        };

        let summed = sum_array_response(&[curve_a, curve_b]).unwrap();

        assert_eq!(summed.spl[0], -240.0);
        assert_eq!(summed.phase.as_ref().unwrap()[0], 0.0);
    }
}
