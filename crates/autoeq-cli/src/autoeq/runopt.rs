use super::spacing::print_freq_spacing;
use autoeq::optim::{
    self, ObjectiveData, OptimizerBackend, OptimizerConfidence, OptimizerRunEvidence,
    RealOptimizerBackend,
};
use std::error::Error;

/// Struct to hold optimization results including convergence status
pub(super) struct OptimizationResult {
    pub(super) params: Vec<f64>,
    pub(super) converged: bool,
    pub(super) pre_objective: Option<f64>,
    pub(super) post_objective: Option<f64>,
    /// Structured per-invocation evidence, global first and local
    /// refinement second when `refine` is enabled.
    ///
    /// Additive contract for QA tooling in other crates:
    /// `selected_for_output` marks the invocation that supplied
    /// `params`/`post_objective`. Superseded passes remain for diagnosis
    /// but must not be treated as production-acceptance inputs.
    pub(super) optimizer_evidence: Vec<OptimizerRunEvidence>,
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
    perform_optimization_with_backend(params, objective_data, bounds, &RealOptimizerBackend::new())
}

/// Backend-injectable optimization driver.
///
/// Production callers pass [`RealOptimizerBackend`]; tests inject
/// [`autoeq::optim::MockOptimizerBackend`] (or a local fake) for
/// deterministic coverage of the refinement-acceptance policy without
/// running a stochastic search.
pub(super) fn perform_optimization_with_backend(
    params: &autoeq::OptimParams,
    objective_data: &ObjectiveData,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    backend: &dyn OptimizerBackend,
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
    let pre_objective = Some(optim::compute_fitness_penalties_ref(&x, objective_data));

    let global_result = backend.optimize_filters(
        &mut x,
        &lower_bounds,
        &upper_bounds,
        objective_data.clone(),
        params,
    );
    let global_evidence = OptimizerRunEvidence::from_backend_result(
        &params.algo,
        global_result.clone(),
        &x,
        &lower_bounds,
        &upper_bounds,
        params.maxeval,
        params.seed,
    );

    match &global_result {
        Ok((status, val)) => {
            if !params.quiet {
                log::debug!(
                    "✅ Global optimization completed with status: {}. Objective function value: {:.6}",
                    status,
                    val
                );
            }
        }
        Err((e, final_value)) => {
            eprintln!("❌ Optimization failed: {:?}", e);
            eprintln!("   - Final Mean Squared Error: {:.6}", final_value);
            return Err(std::io::Error::other(e.clone()).into());
        }
    };
    let global_loss = match global_evidence.objective {
        Some(val) => val,
        None => {
            return Err(std::io::Error::other(format!(
                "global optimizer returned a non-finite objective ({})",
                global_evidence.status
            ))
            .into());
        }
    };
    // Transport-level success does not imply convergence: an `Ok` return
    // carrying a best-effort status (e.g. "... (not converged, ...)") is
    // usable but must not be reported as converged.
    if !global_evidence.converged && !global_evidence.best_effort {
        return Err(std::io::Error::other(format!(
            "global optimizer produced an unusable result ({})",
            global_evidence.status
        ))
        .into());
    }
    if !global_evidence.converged && !params.quiet {
        log::warn!(
            "Global optimization did not fully converge ({}); keeping best-effort result",
            global_evidence.status
        );
    }

    let mut converged = global_evidence.converged;
    let mut post_objective = Some(global_loss);
    let mut optimizer_evidence = vec![global_evidence];

    if !params.quiet && objective_data.loss_type != autoeq::LossType::DriversFlat {
        print_freq_spacing(&x, params, "global");
    }

    if params.refine {
        // Snapshot the global result before handing the vector to the
        // local optimizer: local methods are not guaranteed to improve
        // their input, so a regressing (or failing) refinement must roll
        // back instead of overwriting the usable global result.
        let x_before_refine = x.clone();
        let local_result = backend.optimize_filters_with_algo_override(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            objective_data.clone(),
            params,
            Some(&params.local_algo),
        );
        let mut local_evidence = OptimizerRunEvidence::from_backend_result(
            &params.local_algo,
            local_result.clone(),
            &x,
            &lower_bounds,
            &upper_bounds,
            params.maxeval,
            params.seed,
        );
        match &local_result {
            Ok((local_status, local_val)) => {
                if !params.quiet {
                    log::debug!(
                        "✅ Running local refinement with {}... completed {} objective {:.6}",
                        params.local_algo,
                        local_status,
                        local_val
                    );
                }
            }
            Err((e, final_value)) => {
                // Transport-level failure keeps the usable global result
                // instead of discarding it (previous behavior returned Err).
                eprintln!("⚠️  Local refinement failed: {:?}", e);
                eprintln!("   - Final Mean Squared Error: {:.6}", final_value);
            }
        }
        let local_loss = local_evidence.objective.unwrap_or(f64::INFINITY);
        // Selection policy (mirrors the guarded RoomEQ refinement in
        // roomeq-engine `eq::optimize`): accept only a usable (finite and
        // in-bounds, i.e. confidence better than `Unusable`) refinement
        // that strictly improves the chosen scalar objective.
        let use_local = local_evidence.confidence != OptimizerConfidence::Unusable
            && local_loss < global_loss;
        local_evidence.selected_for_output = use_local;
        if use_local {
            if !params.quiet {
                log::debug!(
                    "✅ Local refinement improved objective {:.6} -> {:.6}",
                    global_loss,
                    local_loss
                );
            }
            // Convergence follows the accepted pass: a best-effort
            // refinement that improved the objective is usable but not
            // reported as converged.
            converged = local_evidence.converged;
            post_objective = Some(local_loss);
            if !params.quiet && objective_data.loss_type != autoeq::LossType::DriversFlat {
                print_freq_spacing(&x, params, "local");
                autoeq::x2peq::peq_print_from_x(&x, params.sample_rate, params.peq_model);
            }
        } else {
            if !params.quiet {
                log::debug!(
                    "Local refinement did not improve ({:.6} -> {:.6}), keeping global result",
                    global_loss,
                    local_loss
                );
            }
            x.clone_from(&x_before_refine);
        }
        optimizer_evidence[0].selected_for_output = !use_local;
        optimizer_evidence.push(local_evidence);
    }

    Ok(OptimizationResult {
        params: x,
        converged,
        pre_objective,
        post_objective,
        optimizer_evidence,
    })
}
