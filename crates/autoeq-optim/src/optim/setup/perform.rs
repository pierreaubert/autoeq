use super::super::constraint_envelope::{
    OwnedConstraintSpec, finalize_candidate, project_gains_onto_envelopes,
};
use super::super::de::{
    DECheckpointSaveCallback, optimize_filters_autoeq_with_callback,
    optimize_filters_autoeq_with_callback_and_initial,
    optimize_filters_autoeq_with_exact_checkpoint,
};
use super::super::run_descriptor::OptimizationRunResult;
use super::super::{ObjectiveData, optimize_filters_with_algo_override};
use super::misc::initial_guess;
use super::misc::resolves_to_backend;
use super::progress_callback_config::ProgressCallbackConfig;
use super::setup_bounds;
use super::types::OptimizationOutput;
use super::types::ProgressUpdate;
use crate::iir::Biquad;
use crate::read;
use crate::x2peq::x2peq;
use ndarray::Array1;
use std::error::Error;

/// State and save callback for exact DE continuation.
pub struct ExactDECheckpointOptions {
    /// Saved state to restore, or None to start a newly checkpointed run.
    pub checkpoint: Option<crate::de::DECheckpoint>,
    /// Identity for the measurement, objective, normalization, and config.
    pub run_identity: String,
    /// Atomically saves each generation barrier; errors stop the optimizer.
    pub save_callback: DECheckpointSaveCallback,
}

impl std::fmt::Debug for ExactDECheckpointOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExactDECheckpointOptions")
            .field("checkpoint", &self.checkpoint)
            .field("run_identity", &self.run_identity)
            .field("save_callback", &"<generation-barrier callback>")
            .finish()
    }
}

/// Run global (and optional local refine) optimization and return the parameter vector.
pub fn perform_optimization(
    params: &crate::OptimParams,
    objective_data: &ObjectiveData,
) -> Result<Vec<f64>, Box<dyn Error>> {
    Ok(perform_optimization_with_run_descriptor(
        params,
        objective_data,
        Box::new(|_intermediate| crate::de::CallbackAction::Continue),
    )?
    .parameters)
}

/// Run optimization and retain the serializable descriptor required by a
/// provenance-aware workflow boundary.
pub fn perform_optimization_with_run_descriptor(
    params: &crate::OptimParams,
    objective_data: &ObjectiveData,
    callback: Box<dyn FnMut(&crate::de::DEIntermediate) -> crate::de::CallbackAction + Send>,
) -> Result<OptimizationRunResult, Box<dyn Error>> {
    perform_optimization_with_optional_candidate(params, objective_data, None, callback, None)
}

/// Run optimization from a validated warm-start candidate.
///
/// The candidate is revalidated and finalized against the run's complete
/// current constraint specification before search. This includes gain,
/// global/local-Q, and composite envelopes.
///
/// # Errors
///
/// Returns an error when the candidate has invalid dimensions, non-finite
/// values, violates current bounds, or remains infeasible after constraint
/// finalization.
pub fn perform_optimization_with_run_descriptor_and_candidate(
    params: &crate::OptimParams,
    objective_data: &ObjectiveData,
    initial_candidate: &[f64],
    callback: Box<dyn FnMut(&crate::de::DEIntermediate) -> crate::de::CallbackAction + Send>,
) -> Result<OptimizationRunResult, Box<dyn Error>> {
    perform_optimization_with_optional_candidate(
        params,
        objective_data,
        Some(initial_candidate),
        callback,
        None,
    )
}

/// Run exact DE continuation and retain its serializable run descriptor.
///
/// Exact state is supported for seeded AutoEQ DE runs without local
/// refinement or multi-driver or multi-sub objectives. Candidate warm starts
/// continue to use the separate candidate API.
///
/// # Errors
///
/// Returns an error when the selected optimizer is not AutoEQ DE, the seed is
/// absent, local refinement or a multi-driver objective is requested, the
/// saved checkpoint identity or configuration is invalid, or checkpoint
/// persistence fails.
pub fn perform_optimization_with_run_descriptor_and_exact_checkpoint(
    params: &crate::OptimParams,
    objective_data: &ObjectiveData,
    continuation: ExactDECheckpointOptions,
    callback: Box<dyn FnMut(&crate::de::DEIntermediate) -> crate::de::CallbackAction + Send>,
) -> Result<OptimizationRunResult, Box<dyn Error>> {
    if params.refine {
        return Err(std::io::Error::other(
            "exact DE continuation does not support a follow-up local-refinement stage",
        )
        .into());
    }
    if params.seed.is_none() {
        return Err(
            std::io::Error::other("exact DE continuation requires an explicit --seed").into(),
        );
    }
    let backend = super::super::backend::resolve(&params.algo)
        .ok_or_else(|| std::io::Error::other(format!("unknown optimizer: {}", params.algo)))?;
    if !backend.name().eq_ignore_ascii_case("autoeq:de") {
        return Err(std::io::Error::other(format!(
            "exact continuation is supported only for AutoEQ DE; resolved {}",
            backend.name()
        ))
        .into());
    }
    if matches!(
        objective_data.loss_type,
        crate::LossType::DriversFlat | crate::LossType::MultiSubFlat
    ) {
        return Err(std::io::Error::other(
            "exact DE continuation is not supported for multi-driver or multi-sub optimization",
        )
        .into());
    }
    perform_optimization_with_optional_candidate(
        params,
        objective_data,
        None,
        callback,
        Some(continuation),
    )
}

fn perform_optimization_with_optional_candidate(
    params: &crate::OptimParams,
    objective_data: &ObjectiveData,
    initial_candidate: Option<&[f64]>,
    callback: Box<dyn FnMut(&crate::de::DEIntermediate) -> crate::de::CallbackAction + Send>,
    exact_continuation: Option<ExactDECheckpointOptions>,
) -> Result<OptimizationRunResult, Box<dyn Error>> {
    let (lower_bounds, upper_bounds) = setup_bounds(params);
    let mut descriptor = super::super::run_descriptor::OptimizationRunDescriptor::started(
        params,
        objective_data,
        &lower_bounds,
        &upper_bounds,
    );
    if initial_candidate.is_some() {
        let backend = super::super::backend::resolve(&params.algo)
            .ok_or_else(|| std::io::Error::other(format!("unknown optimizer: {}", params.algo)))?;
        if !backend.supports_initial_candidate() {
            return Err(std::io::Error::other(format!(
                "warm-start candidates are unsupported for {} because this optimizer path does not use the supplied initial candidate",
                backend.name()
            ))
            .into());
        }
    }
    // O1: project the seed onto the configured per-filter gain envelopes so
    // the search starts inside the feasible region. Deterministic
    // post-processing that consumes no RNG; bit-identical without envelopes.
    let initial = match initial_candidate {
        Some(candidate) => {
            validate_initial_candidate(candidate, &lower_bounds, &upper_bounds)?;
            let constraint_spec =
                OwnedConstraintSpec::from_params(params).map_err(std::io::Error::other)?;
            let finalized = finalize_candidate(
                "warm-start",
                candidate,
                objective_data,
                &constraint_spec.as_spec(),
            )
            .map_err(|reason| {
                std::io::Error::other(format!("warm-start candidate is infeasible: {reason}"))
            })?;
            validate_initial_candidate(&finalized.params, &lower_bounds, &upper_bounds)?;
            finalized.params
        }
        None => initial_guess(params, &lower_bounds, &upper_bounds),
    };
    let mut x = project_gains_onto_envelopes(
        &initial,
        params.peq_model,
        objective_data.loss_type,
        objective_data.max_boost_envelope.as_deref(),
        objective_data.min_cut_envelope.as_deref(),
    )
    .0;
    validate_initial_candidate(&x, &lower_bounds, &upper_bounds)?;

    let result = if let Some(continuation) = exact_continuation {
        if initial_candidate.is_some() {
            return Err(std::io::Error::other(
                "exact continuation cannot be combined with a warm-start candidate",
            )
            .into());
        }
        optimize_filters_autoeq_with_exact_checkpoint(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            objective_data.clone(),
            &params.algo,
            params,
            callback,
            super::super::de::DEExactContinuation {
                checkpoint: continuation.checkpoint,
                run_identity: continuation.run_identity,
                save_callback: continuation.save_callback,
            },
        )
    } else if resolves_to_backend(&params.algo, "autoeq:de") {
        if initial_candidate.is_some() {
            let explicit_initial_candidate = x.clone();
            optimize_filters_autoeq_with_callback_and_initial(
                &mut x,
                &lower_bounds,
                &upper_bounds,
                objective_data.clone(),
                &params.algo,
                params,
                Some(&explicit_initial_candidate),
                callback,
            )
        } else {
            optimize_filters_autoeq_with_callback(
                &mut x,
                &lower_bounds,
                &upper_bounds,
                objective_data.clone(),
                &params.algo,
                params,
                callback,
            )
        }
    } else {
        optimize_filters_with_algo_override(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            objective_data.clone(),
            params,
            None,
        )
    };

    let (global_status, global_fun) = match result {
        Ok((status, value)) => (status, value),
        Err((error, _final_value)) => return Err(std::io::Error::other(error).into()),
    };
    let mut stopping_reason = global_status;
    let mut objective_value = global_fun;

    if params.refine && !resolves_to_backend(&params.algo, "autoeq:bo") {
        // Local solvers are not guaranteed to improve the global optimum.
        let x_pre_refine = x.clone();
        let local_result = optimize_filters_with_algo_override(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            objective_data.clone(),
            params,
            Some(&params.local_algo),
        );
        match local_result {
            Ok((local_status, local_value)) => {
                if !local_value.is_finite() || local_value > global_fun {
                    if !params.quiet {
                        log::warn!(
                            "Local refine ({}) regressed: {:.6} -> {:.6}; keeping global result.",
                            params.local_algo,
                            global_fun,
                            local_value,
                        );
                    }
                    x = x_pre_refine;
                    stopping_reason =
                        format!("{stopping_reason}; local refine rejected: {local_status}");
                } else {
                    stopping_reason = format!("{stopping_reason}; local refine: {local_status}");
                    objective_value = local_value;
                }
            }
            Err((error, _final_value)) => return Err(std::io::Error::other(error).into()),
        }
    }

    // O1 extraction: realize the envelope-projected parameters the loss was
    // evaluated on, so the delivered vector matches the scored response.
    // Bit-identical without envelopes.
    let parameters = project_gains_onto_envelopes(
        &x,
        params.peq_model,
        objective_data.loss_type,
        objective_data.max_boost_envelope.as_deref(),
        objective_data.min_cut_envelope.as_deref(),
    )
    .0;

    descriptor.finished(stopping_reason);
    Ok(OptimizationRunResult {
        parameters,
        objective_value,
        descriptor,
    })
}

fn validate_initial_candidate(
    candidate: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
) -> Result<(), Box<dyn Error>> {
    if candidate.is_empty()
        || candidate.len() != lower_bounds.len()
        || candidate.len() != upper_bounds.len()
    {
        return Err(std::io::Error::other(format!(
            "warm-start candidate has {} parameters; bounds have {} lower and {} upper values",
            candidate.len(),
            lower_bounds.len(),
            upper_bounds.len()
        ))
        .into());
    }
    for (index, ((&value, &lower), &upper)) in candidate
        .iter()
        .zip(lower_bounds)
        .zip(upper_bounds)
        .enumerate()
    {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(std::io::Error::other(format!(
                "warm-start bounds at parameter {index} are invalid: [{lower}, {upper}]"
            ))
            .into());
        }
        if !value.is_finite() {
            return Err(std::io::Error::other(format!(
                "warm-start candidate parameter {index} is not finite"
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
    Ok(())
}

/// Run optimization with a DE progress callback (only used for AutoEQ DE).
pub fn perform_optimization_with_callback(
    params: &crate::OptimParams,
    objective_data: &ObjectiveData,
    callback: Box<dyn FnMut(&crate::de::DEIntermediate) -> crate::de::CallbackAction + Send>,
) -> Result<Vec<f64>, Box<dyn Error>> {
    Ok(perform_optimization_with_run_descriptor(params, objective_data, callback)?.parameters)
}

/// Run optimization with progress callback at configurable intervals
///
/// This wraps `perform_optimization_with_callback` with:
/// - Interval-based reporting (not every iteration)
/// - Automatic biquad decoding from raw params
/// - Filter response computation
/// - Score calculation when speaker_score_data is available
///
/// # Arguments
/// * `args` - CLI arguments (will be converted to OptimParams internally)
/// * `objective_data` - Objective data from setup_objective_data
/// * `config` - Callback configuration (interval, what to include)
/// * `callback` - User callback receiving ProgressUpdate
///
/// # Returns
/// Optimization result with raw filter parameters and history
pub fn perform_optimization_with_progress<F>(
    params: &crate::OptimParams,
    objective_data: &ObjectiveData,
    config: ProgressCallbackConfig,
    callback: F,
) -> Result<OptimizationOutput, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    perform_optimization_with_progress_and_optional_candidate(
        params,
        objective_data,
        config,
        None,
        callback,
    )
}

/// Run optimization from a warm-start candidate and report progress.
///
/// The candidate is revalidated and finalized against the run's complete
/// current constraint specification before it reaches the optimizer.
///
/// # Errors
///
/// Returns an error when the candidate has invalid dimensions, non-finite
/// values, or values outside the current parameter bounds.
pub fn perform_optimization_with_progress_and_candidate<F>(
    params: &crate::OptimParams,
    objective_data: &ObjectiveData,
    config: ProgressCallbackConfig,
    initial_candidate: &[f64],
    callback: F,
) -> Result<OptimizationOutput, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    perform_optimization_with_progress_and_optional_candidate(
        params,
        objective_data,
        config,
        Some(initial_candidate),
        callback,
    )
}

fn perform_optimization_with_progress_and_optional_candidate<F>(
    params: &crate::OptimParams,
    objective_data: &ObjectiveData,
    config: ProgressCallbackConfig,
    initial_candidate: Option<&[f64]>,
    mut callback: F,
) -> Result<OptimizationOutput, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    use std::sync::{Arc, Mutex};

    let frequencies: Vec<f64> = if config.frequencies.is_empty() {
        read::create_log_frequency_grid(200, 20.0, 20000.0)
            .iter()
            .copied()
            .collect()
    } else {
        config.frequencies.clone()
    };
    let freq_array = Array1::from(frequencies.clone());
    let speaker_score_data = objective_data.speaker_score_data.clone();
    let sample_rate = params.sample_rate;
    let peq_model = params.peq_model;
    let maxeval = params.maxeval;

    let last_reported = Arc::new(Mutex::new(0usize));
    let history = Arc::new(Mutex::new(Vec::new()));

    let last_reported_clone = Arc::clone(&last_reported);
    let history_clone = Arc::clone(&history);
    let freq_array_clone = freq_array.clone();
    let frequencies_clone = frequencies.clone();

    let de_callback = move |intermediate: &crate::de::DEIntermediate| -> crate::de::CallbackAction {
        // Always record history
        {
            let mut hist = history_clone.lock().unwrap();
            hist.push((intermediate.iter, intermediate.fun));
        }

        let mut last = last_reported_clone.lock().unwrap();

        // Check if we should report
        if intermediate.iter == 0 || intermediate.iter.saturating_sub(*last) >= config.interval {
            *last = intermediate.iter;

            // Decode biquads if requested
            let biquads: Vec<Biquad> = if config.include_biquads {
                x2peq(&intermediate.x.to_vec(), sample_rate, peq_model)
                    .into_iter()
                    .map(|(_, b)| b)
                    .collect()
            } else {
                Vec::new()
            };

            // Compute filter response if requested
            let filter_response: Vec<f64> = if config.include_filter_response && !biquads.is_empty()
            {
                frequencies_clone
                    .iter()
                    .map(|&f| biquads.iter().map(|b| b.log_result(f)).sum())
                    .collect()
            } else {
                Vec::new()
            };

            // Compute score if speaker_score_data available
            let score = speaker_score_data.as_ref().map(|sd| {
                let peq_response = if !filter_response.is_empty() {
                    Array1::from(filter_response.clone())
                } else {
                    let bs = x2peq(&intermediate.x.to_vec(), sample_rate, peq_model);
                    let resp: Vec<f64> = frequencies_clone
                        .iter()
                        .map(|&f| bs.iter().map(|(_, b)| b.log_result(f)).sum())
                        .collect();
                    Array1::from(resp)
                };
                crate::loss::speaker_score_loss(sd, &freq_array_clone, &peq_response)
            });

            let update = ProgressUpdate {
                iteration: intermediate.iter,
                max_iterations: maxeval,
                loss: intermediate.fun,
                score,
                convergence: intermediate.convergence,
                params: intermediate.x.to_vec(),
                biquads,
                filter_response,
            };

            callback(&update)
        } else {
            crate::de::CallbackAction::Continue
        }
    };

    let run = perform_optimization_with_optional_candidate(
        params,
        objective_data,
        initial_candidate,
        Box::new(de_callback),
        None,
    )?;

    let final_history = Arc::try_unwrap(history)
        .map(|m| m.into_inner().unwrap())
        .unwrap_or_default();

    Ok(OptimizationOutput {
        params: run.parameters,
        history: final_history,
        optimization_run: run.descriptor,
    })
}
