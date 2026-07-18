//! Reusable in-memory driver and multi-sub optimization services.

use crate::loss::DriversLossData;
use crate::optim::setup::{
    drivers_initial_guess, drivers_initial_guess_fixed_freqs, multisub_initial_guess,
    setup_drivers_bounds, setup_drivers_bounds_fixed_freqs, setup_drivers_objective_data,
    setup_multisub_bounds, setup_multisub_objective_data,
};
use crate::{LossType, OptimParams, PeqModel};

/// Result of driver or multi-sub optimization.
#[derive(Debug, Clone)]
pub struct DriverOptimizationResult {
    pub gains: Vec<f64>,
    pub delays: Vec<f64>,
    pub crossover_freqs: Vec<f64>,
    pub pre_objective: f64,
    pub post_objective: f64,
    pub converged: bool,
}

/// Create minimal optimizer parameters for driver/multi-sub optimization.
#[allow(clippy::too_many_arguments)]
pub fn create_driver_optimization_params(
    min_freq: f64,
    max_freq: f64,
    sample_rate: f64,
    algorithm: &str,
    max_iter: usize,
    population: usize,
    min_db: f64,
    max_db: f64,
    seed: Option<u64>,
) -> OptimParams {
    OptimParams {
        num_filters: 0,
        peq_model: PeqModel::Pk,
        sample_rate,
        min_freq,
        max_freq,
        min_q: 0.5,
        max_q: 10.0,
        min_db,
        max_db,
        loss: LossType::DriversFlat,
        smooth: false,
        smooth_n: 1,
        min_spacing_oct: 0.0,
        spacing_weight: 0.0,
        smoothness_penalty: None,
        audibility_deadband: None,
        algo: algorithm.to_string(),
        population,
        maxeval: max_iter,
        refine: false,
        local_algo: "cobyla".to_string(),
        bo_initial_samples: 0,
        bo_batch_size: 0,
        bo_posterior_std_threshold: 0.0,
        bo_acquisition: "qei".to_string(),
        bo_ehvi: false,
        strategy: "currenttobest1bin".to_string(),
        tolerance: 1e-3,
        atolerance: 1e-4,
        recombination: 0.9,
        adaptive_weight_f: 0.9,
        adaptive_weight_cr: 0.9,
        no_parallel: false,
        parallel_threads: num_cpus::get(),
        seed,
        quiet: false,
    }
}

fn validate_sample_rate(sample_rate: f64) -> Result<(), Box<dyn std::error::Error>> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(format!("sample rate must be finite and positive, got {sample_rate}").into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn optimize_drivers_crossover(
    drivers_data: DriversLossData,
    min_freq: f64,
    max_freq: f64,
    sample_rate: f64,
    algorithm: &str,
    max_iter: usize,
    population: usize,
    min_db: f64,
    max_db: f64,
    fixed_freqs: Option<Vec<f64>>,
    seed: Option<u64>,
) -> Result<DriverOptimizationResult, Box<dyn std::error::Error>> {
    validate_sample_rate(sample_rate)?;
    let n_drivers = drivers_data.drivers.len();
    let params = create_driver_optimization_params(
        min_freq,
        max_freq,
        sample_rate,
        algorithm,
        max_iter,
        population,
        min_db,
        max_db,
        seed,
    );
    let objective_data = if let Some(ref freqs) = fixed_freqs {
        let mut data = setup_drivers_objective_data(&params, drivers_data.clone());
        data.fixed_crossover_freqs = Some(freqs.clone());
        data.objective = Some(data.build_objective());
        data
    } else {
        setup_drivers_objective_data(&params, drivers_data.clone())
    };
    let (lower_bounds, upper_bounds) = if fixed_freqs.is_some() {
        setup_drivers_bounds_fixed_freqs(&params, &drivers_data)
    } else {
        setup_drivers_bounds(&params, &drivers_data)
    };
    let mut x = if fixed_freqs.is_some() {
        drivers_initial_guess_fixed_freqs(&lower_bounds, &upper_bounds, n_drivers)
    } else {
        drivers_initial_guess(&lower_bounds, &upper_bounds, n_drivers)
    };
    let initial_x = x.clone();
    let pre_objective = crate::optim::compute_base_fitness(&x, &objective_data);
    let converged = crate::optim::optimize_filters(
        &mut x,
        &lower_bounds,
        &upper_bounds,
        objective_data.clone(),
        &params,
    )
    .is_ok();
    let mut post_objective = crate::optim::compute_base_fitness(&x, &objective_data);
    if !post_objective.is_finite() || post_objective > pre_objective {
        x = initial_x;
        post_objective = pre_objective;
    }
    let gains = x[0..n_drivers].to_vec();
    let delays = x[n_drivers..2 * n_drivers].to_vec();
    let crossover_freqs = fixed_freqs.unwrap_or_else(|| {
        x[2 * n_drivers..]
            .iter()
            .map(|value| 10_f64.powf(*value))
            .collect()
    });
    Ok(DriverOptimizationResult {
        gains,
        delays,
        crossover_freqs,
        pre_objective,
        post_objective,
        converged,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn optimize_multisub(
    drivers_data: DriversLossData,
    min_freq: f64,
    max_freq: f64,
    sample_rate: f64,
    algorithm: &str,
    max_iter: usize,
    population: usize,
    min_db: f64,
    max_db: f64,
    seed: Option<u64>,
) -> Result<DriverOptimizationResult, Box<dyn std::error::Error>> {
    validate_sample_rate(sample_rate)?;
    let n_drivers = drivers_data.drivers.len();
    let mut params = create_driver_optimization_params(
        min_freq,
        max_freq,
        sample_rate,
        algorithm,
        max_iter,
        population,
        min_db,
        max_db,
        seed,
    );
    params.loss = LossType::MultiSubFlat;
    let objective_data = setup_multisub_objective_data(&params, drivers_data);
    let (lower_bounds, upper_bounds) = setup_multisub_bounds(&params, n_drivers);
    let mut x = multisub_initial_guess(n_drivers);
    let initial_x = x.clone();
    let pre_objective = crate::optim::compute_base_fitness(&x, &objective_data);
    let converged = crate::optim::optimize_filters(
        &mut x,
        &lower_bounds,
        &upper_bounds,
        objective_data.clone(),
        &params,
    )
    .is_ok();
    let mut post_objective = crate::optim::compute_base_fitness(&x, &objective_data);
    if !post_objective.is_finite() || post_objective > pre_objective {
        x = initial_x;
        post_objective = pre_objective;
    }
    Ok(DriverOptimizationResult {
        gains: x[0..n_drivers].to_vec(),
        delays: x[n_drivers..2 * n_drivers].to_vec(),
        crossover_freqs: Vec::new(),
        pre_objective,
        post_objective,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_optimization_params_have_expected_values() {
        let params = create_driver_optimization_params(
            50.0,
            5000.0,
            48000.0,
            "autoeq:de",
            20,
            6,
            -6.0,
            6.0,
            Some(1),
        );
        assert_eq!(params.min_freq, 50.0);
        assert_eq!(params.max_freq, 5000.0);
        assert_eq!(params.sample_rate, 48000.0);
        assert_eq!(params.algo, "autoeq:de");
        assert_eq!(params.maxeval, 20);
        assert_eq!(params.population, 6);
        assert_eq!(params.min_db, -6.0);
        assert_eq!(params.max_db, 6.0);
        assert_eq!(params.seed, Some(1));
        assert_eq!(params.loss, LossType::DriversFlat);
    }

    #[test]
    fn driver_params_reject_invalid_sample_rate_at_service_boundary() {
        assert!(validate_sample_rate(0.0).is_err());
        assert!(validate_sample_rate(f64::NAN).is_err());
        assert!(validate_sample_rate(48_000.0).is_ok());
    }
}
