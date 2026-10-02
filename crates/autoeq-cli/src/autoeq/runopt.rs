use super::spacing::print_freq_spacing;
use autoeq::optim::{
    self, ObjectiveData, OptimizerBackend, OptimizerConfidence, OptimizerRunEvidence,
    RealOptimizerBackend,
};
use std::error::Error;

pub(super) type CandidateProgressCallback =
    Box<dyn FnMut(&autoeq::optim::setup::ProgressUpdate) -> Result<(), String> + Send + 'static>;

struct OptimizationInvocationOptions {
    progress_callback: Option<CandidateProgressCallback>,
    direct_de_warm_start: bool,
    exact_checkpoint: Option<autoeq::optim::setup::ExactDECheckpointOptions>,
}

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
    /// Absolute objective gap between the optimizer parameters and their
    /// integer-Hz APO serialization (`None` for non-PEQ layouts).
    /// Keeps the reported evidence aligned with the shipped preset; see
    /// [`super::save::apo_roundtrip_objective_gap`].
    pub(super) apo_roundtrip_gap: Option<f64>,
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

pub(super) fn perform_optimization_with_candidate(
    params: &autoeq::OptimParams,
    objective_data: &ObjectiveData,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    initial_candidate: &[f64],
) -> Result<OptimizationResult, Box<dyn Error>> {
    perform_optimization_with_backend_and_candidate_and_progress_callback(
        params,
        objective_data,
        bounds,
        Some(initial_candidate),
        &RealOptimizerBackend::new(),
        OptimizationInvocationOptions {
            progress_callback: None,
            direct_de_warm_start: true,
            exact_checkpoint: None,
        },
    )
}

/// Run AutoEQ DE with candidate-bearing progress events for durable checkpoints.
/// Backends that expose only iteration/loss events cannot safely save a
/// recoverable parameter vector, so this path rejects them explicitly.
pub(super) fn perform_optimization_with_progress_callback(
    params: &autoeq::OptimParams,
    objective_data: &ObjectiveData,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    initial_candidate: Option<&[f64]>,
    callback: CandidateProgressCallback,
) -> Result<OptimizationResult, Box<dyn Error>> {
    let backend = autoeq::optim::backend::resolve(&params.algo)
        .ok_or_else(|| std::io::Error::other(format!("unknown optimizer: {}", params.algo)))?;
    if !backend.name().eq_ignore_ascii_case("autoeq:de") {
        return Err(std::io::Error::other(format!(
            "periodic warm-start checkpoints require AutoEQ DE candidate progress; {} does not expose candidate snapshots",
            backend.name()
        ))
        .into());
    }
    if matches!(
        objective_data.loss_type,
        autoeq::LossType::DriversFlat | autoeq::LossType::MultiSubFlat
    ) {
        return Err(std::io::Error::other(
            "periodic warm-start checkpoints are not supported for multi-driver optimization",
        )
        .into());
    }

    perform_optimization_with_backend_and_candidate_and_progress_callback(
        params,
        objective_data,
        bounds,
        initial_candidate,
        &RealOptimizerBackend::new(),
        OptimizationInvocationOptions {
            progress_callback: Some(callback),
            direct_de_warm_start: true,
            exact_checkpoint: None,
        },
    )
}

/// Run exact AutoEQ DE continuation with a full-state persistence callback.
///
/// This path defers baseline scoring until the math layer validates saved state.
pub(super) fn perform_optimization_with_exact_checkpoint(
    params: &autoeq::OptimParams,
    objective_data: &ObjectiveData,
    continuation: autoeq::optim::setup::ExactDECheckpointOptions,
) -> Result<OptimizationResult, Box<dyn Error>> {
    if params.refine {
        return Err(std::io::Error::other(
            "exact DE continuation does not support a follow-up local-refinement stage",
        )
        .into());
    }
    perform_optimization_with_backend_and_candidate_and_progress_callback(
        params,
        objective_data,
        None,
        None,
        &RealOptimizerBackend::new(),
        OptimizationInvocationOptions {
            progress_callback: None,
            direct_de_warm_start: false,
            exact_checkpoint: Some(continuation),
        },
    )
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
    perform_optimization_with_backend_and_candidate(params, objective_data, bounds, None, backend)
}

pub(super) fn perform_optimization_with_backend_and_candidate(
    params: &autoeq::OptimParams,
    objective_data: &ObjectiveData,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    initial_candidate: Option<&[f64]>,
    backend: &dyn OptimizerBackend,
) -> Result<OptimizationResult, Box<dyn Error>> {
    perform_optimization_with_backend_and_candidate_and_progress_callback(
        params,
        objective_data,
        bounds,
        initial_candidate,
        backend,
        OptimizationInvocationOptions {
            progress_callback: None,
            direct_de_warm_start: false,
            exact_checkpoint: None,
        },
    )
}

fn perform_optimization_with_backend_and_candidate_and_progress_callback(
    params: &autoeq::OptimParams,
    objective_data: &ObjectiveData,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    initial_candidate: Option<&[f64]>,
    backend: &dyn OptimizerBackend,
    options: OptimizationInvocationOptions,
) -> Result<OptimizationResult, Box<dyn Error>> {
    let OptimizationInvocationOptions {
        progress_callback,
        direct_de_warm_start,
        exact_checkpoint,
    } = options;
    let resolved_backend = autoeq::optim::backend::resolve(&params.algo)
        .ok_or_else(|| std::io::Error::other(format!("unknown optimizer: {}", params.algo)))?;
    if initial_candidate.is_some() && !resolved_backend.supports_initial_candidate() {
        return Err(std::io::Error::other(format!(
            "warm-start candidates are unsupported for {} because its optimizer path does not use the supplied initial candidate",
            resolved_backend.name()
        ))
        .into());
    }
    let (lower_bounds, upper_bounds) =
        bounds.unwrap_or_else(|| autoeq::workflow::setup_bounds(params));

    // Generate an initial guess or finalize the validated warm-start candidate.
    let mut x = if let Some(candidate) = initial_candidate {
        if candidate.is_empty()
            || candidate.len() != lower_bounds.len()
            || candidate.len() != upper_bounds.len()
        {
            return Err(std::io::Error::other(format!(
                "warm-start candidate has {} parameters; current bounds have {}",
                candidate.len(),
                lower_bounds.len()
            ))
            .into());
        }
        if let Some(index) = candidate.iter().position(|value| !value.is_finite()) {
            return Err(std::io::Error::other(format!(
                "warm-start candidate parameter {index} is not finite"
            ))
            .into());
        }
        for (index, (&value, (&lower, &upper))) in candidate
            .iter()
            .zip(lower_bounds.iter().zip(&upper_bounds))
            .enumerate()
        {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(std::io::Error::other(format!(
                    "warm-start bounds at parameter {index} are invalid: [{lower}, {upper}]"
                ))
                .into());
            }
            if value < lower || value > upper {
                return Err(std::io::Error::other(format!(
                    "warm-start candidate parameter {index}={value} is outside current bounds [{lower}, {upper}]"
                ))
                .into());
            }
        }
        let constraint_spec = autoeq::optim::OwnedConstraintSpec::from_params(params)
            .map_err(std::io::Error::other)?;
        autoeq::optim::finalize_candidate(
            "cli-warm-start",
            candidate,
            objective_data,
            &constraint_spec.as_spec(),
        )
        .map_err(|reason| {
            std::io::Error::other(format!("warm-start candidate is infeasible: {reason}"))
        })?
        .params
    } else if objective_data.loss_type == autoeq::LossType::DriversFlat {
        let n_drivers = objective_data.drivers_data.as_ref().unwrap().drivers.len();
        autoeq::workflow::drivers_initial_guess(&lower_bounds, &upper_bounds, n_drivers)
    } else {
        autoeq::workflow::initial_guess(params, &lower_bounds, &upper_bounds)
    };

    // Exact-resume validation must happen before scoring the requested objective.
    let pre_objective = if exact_checkpoint.is_some() {
        None
    } else {
        Some(optim::compute_fitness_penalties_ref(&x, objective_data))
    };

    let global_result = if let Some(continuation) = exact_checkpoint {
        if initial_candidate.is_some() {
            return Err(std::io::Error::other(
                "exact continuation cannot be combined with a warm-start candidate",
            )
            .into());
        }
        let mut global_params = params.clone();
        global_params.refine = false;
        let output =
            autoeq::optim::setup::perform_optimization_with_run_descriptor_and_exact_checkpoint(
                &global_params,
                objective_data,
                continuation,
                Box::new(|_| autoeq::de::CallbackAction::Continue),
            )?;
        x.clone_from_slice(&output.parameters);
        Ok((
            output.descriptor.stopping_reason,
            optim::compute_fitness_penalties_ref(&x, objective_data),
        ))
    } else if let Some(mut progress_callback) = progress_callback {
        use std::sync::{Arc, Mutex};

        let callback_error = Arc::new(Mutex::new(None));
        let callback_error_for_de = Arc::clone(&callback_error);
        let de_callback =
            move |update: &autoeq::optim::setup::ProgressUpdate| match progress_callback(update) {
                Ok(()) => autoeq::de::CallbackAction::Continue,
                Err(error) => {
                    *callback_error_for_de
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
                    autoeq::de::CallbackAction::Stop
                }
            };
        let mut global_params = params.clone();
        // Keep the CLI's existing local-refinement path and its per-pass
        // evidence contract. The candidate callback records the global pass.
        global_params.refine = false;
        let callback_config = autoeq::optim::setup::ProgressCallbackConfig {
            interval: 1,
            include_biquads: false,
            include_filter_response: false,
            frequencies: Vec::new(),
        };
        let output_result = if let Some(candidate) = initial_candidate {
            autoeq::optim::setup::perform_optimization_with_progress_and_candidate(
                &global_params,
                objective_data,
                callback_config,
                candidate,
                de_callback,
            )
        } else {
            autoeq::optim::setup::perform_optimization_with_progress(
                &global_params,
                objective_data,
                callback_config,
                de_callback,
            )
        };
        if let Some(error) = callback_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            return Err(std::io::Error::other(error).into());
        }
        let output = output_result?;
        x.clone_from_slice(&output.params);
        Ok((
            output.optimization_run.stopping_reason,
            optim::compute_fitness_penalties_ref(&x, objective_data),
        ))
    } else if direct_de_warm_start
        && resolved_backend.name().eq_ignore_ascii_case("autoeq:de")
        && let Some(candidate) = initial_candidate
    {
        let mut global_params = params.clone();
        global_params.refine = false;
        let output = autoeq::optim::setup::perform_optimization_with_run_descriptor_and_candidate(
            &global_params,
            objective_data,
            candidate,
            Box::new(|_| autoeq::de::CallbackAction::Continue),
        )?;
        x.clone_from_slice(&output.parameters);
        Ok((
            output.descriptor.stopping_reason,
            optim::compute_fitness_penalties_ref(&x, objective_data),
        ))
    } else {
        backend.optimize_filters(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            objective_data.clone(),
            params,
        )
    };
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
        let use_local =
            local_evidence.confidence != OptimizerConfidence::Unusable && local_loss < global_loss;
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

    // Measure how far the integer-Hz APO serialization drifts from the
    // optimizer response. Non-PEQ layouts (driver gains/delays) have no
    // frequency serialization, so no gap applies.
    let apo_roundtrip_gap = if objective_data.loss_type == autoeq::LossType::DriversFlat
        || objective_data.loss_type == autoeq::LossType::MultiSubFlat
    {
        None
    } else {
        super::save::apo_roundtrip_objective_gap(
            &x,
            params.sample_rate,
            params.peq_model,
            objective_data,
        )
    };

    Ok(OptimizationResult {
        params: x,
        converged,
        pre_objective,
        post_objective,
        optimizer_evidence,
        apo_roundtrip_gap,
    })
}
