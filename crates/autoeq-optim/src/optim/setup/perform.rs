use super::super::de::optimize_filters_autoeq_with_callback;
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
    let (lower_bounds, upper_bounds) = setup_bounds(params);
    let mut descriptor = super::super::run_descriptor::OptimizationRunDescriptor::started(
        params,
        objective_data,
        &lower_bounds,
        &upper_bounds,
    );
    let mut x = initial_guess(params, &lower_bounds, &upper_bounds);

    let result = if resolves_to_backend(&params.algo, "autoeq:de") {
        optimize_filters_autoeq_with_callback(
            &mut x,
            &lower_bounds,
            &upper_bounds,
            objective_data.clone(),
            &params.algo,
            params,
            callback,
        )
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

    descriptor.finished(stopping_reason);
    Ok(OptimizationRunResult {
        parameters: x,
        objective_value,
        descriptor,
    })
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

    let run =
        perform_optimization_with_run_descriptor(params, objective_data, Box::new(de_callback))?;

    let final_history = Arc::try_unwrap(history)
        .map(|m| m.into_inner().unwrap())
        .unwrap_or_default();

    Ok(OptimizationOutput {
        params: run.parameters,
        history: final_history,
        optimization_run: run.descriptor,
    })
}
