use super::spacing::print_freq_spacing;
use autoeq::optim::{self, ObjectiveData};
use std::error::Error;

/// Struct to hold optimization results including convergence status
pub(super) struct OptimizationResult {
    pub(super) params: Vec<f64>,
    pub(super) converged: bool,
    pub(super) pre_objective: Option<f64>,
    pub(super) post_objective: Option<f64>,
}

pub(super) fn perform_optimization(
    params: &autoeq::OptimParams,
    objective_data: &ObjectiveData,
) -> Result<OptimizationResult, Box<dyn Error>> {
    perform_optimization_with_bounds(params, objective_data, None)
}

pub(super) fn perform_optimization_with_bounds(
    params: &autoeq::OptimParams,
    objective_data: &ObjectiveData,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
) -> Result<OptimizationResult, Box<dyn Error>> {
    let (lower_bounds, upper_bounds) =
        bounds.unwrap_or_else(|| autoeq::workflow::setup_bounds(params));

    // Generate initial guess based on loss type
    let mut x = if objective_data.loss_type == autoeq::LossType::DriversFlat {
        let n_drivers = objective_data.drivers_data.as_ref().unwrap().drivers.len();
        autoeq::workflow::drivers_initial_guess(&lower_bounds, &upper_bounds, n_drivers)
    } else {
        autoeq::workflow::initial_guess(params, &lower_bounds, &upper_bounds)
    };

    // Calculate pre-optimization objective value
    let pre_objective = Some(autoeq::optim::compute_fitness_penalties_ref(
        &x,
        objective_data,
    ));

    let result = optim::optimize_filters(
        &mut x,
        &lower_bounds,
        &upper_bounds,
        objective_data.clone(),
        params,
    );

    let mut converged: bool;
    let mut post_objective: Option<f64>;

    match result {
        Ok((status, val)) => {
            if !params.quiet {
                log::debug!(
                    "✅ Global optimization completed with status: {}. Objective function value: {:.6}",
                    status,
                    val
                );
            }
            converged = true;
            post_objective = Some(val);
            if !params.quiet && objective_data.loss_type != autoeq::LossType::DriversFlat {
                print_freq_spacing(&x, params, "global");
            }
        }
        Err((e, final_value)) => {
            eprintln!("❌ Optimization failed: {:?}", e);
            eprintln!("   - Final Mean Squared Error: {:.6}", final_value);
            return Err(std::io::Error::other(e).into());
        }
    };

    if params.refine {
        let result = optim::optimize_filters_with_algo_override(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            objective_data.clone(),
            params,
            Some(&params.local_algo),
        );
        match result {
            Ok((local_status, local_val)) => {
                if !params.quiet {
                    log::debug!(
                        "✅ Running local refinement with {}... completed {} objective {:.6}",
                        params.local_algo,
                        local_status,
                        local_val
                    );
                }
                // Update convergence status based on local refinement
                converged = true;
                post_objective = Some(local_val);
                if !params.quiet && objective_data.loss_type != autoeq::LossType::DriversFlat {
                    print_freq_spacing(&x, params, "local");
                    autoeq::x2peq::peq_print_from_x(&x, params.sample_rate, params.peq_model);
                }
            }
            Err((e, final_value)) => {
                eprintln!("⚠️  Local refinement failed: {:?}", e);
                eprintln!("   - Final Mean Squared Error: {:.6}", final_value);
                return Err(std::io::Error::other(e).into());
            }
        }
    };

    Ok(OptimizationResult {
        params: x,
        converged,
        pre_objective,
        post_objective,
    })
}
