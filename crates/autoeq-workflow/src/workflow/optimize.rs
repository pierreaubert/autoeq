use super::build::build_target_curve;
use super::misc::interpolate_cea2034_data;
use super::types::DriverOptimizationResult;
use super::types::HeadphoneOptResult;
use super::types::OptimizationLineage;
use super::types::SpeakerOptResult;
use super::types::compute_visualization_curves;
use crate::Curve;
use crate::iir::Biquad;
pub use crate::optim::setup::*;
use crate::read;
use crate::workflow::resume::{
    OptimizerState, WarmStartIdentity, config_identity_digest, load_optimizer_state,
    save_optimizer_state,
};
use crate::x2peq;
use autoeq_measurements::{MeasurementOrigin, MeasurementRecord, OperationContext, ToolIdentity};
use autoeq_optim::create_driver_optimization_params as create_driver_optimization_args;
use chrono::Utc;
use serde_json::json;
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

fn workflow_lineage(
    input: MeasurementRecord,
    normalized_curve: Curve,
    corrected_curve: Curve,
    target_curve: &Curve,
    run: &crate::OptimizationRunDescriptor,
) -> Result<OptimizationLineage, Box<dyn Error>> {
    let mut normalize_parameters = std::collections::BTreeMap::new();
    normalize_parameters.insert("method".into(), json!("normalize_and_interpolate"));
    normalize_parameters.insert(
        "output_grid_hash".into(),
        json!(normalized_curve.content_hash()?),
    );
    let normalized_input = input.transformed(
        normalized_curve,
        "normalization_interpolation",
        normalize_parameters,
        true,
    )?;

    let mut optimization_parameters = std::collections::BTreeMap::new();
    optimization_parameters.insert(
        "target_curve_hash".into(),
        json!(target_curve.content_hash()?),
    );
    optimization_parameters.insert("optimizer_run".into(), serde_json::to_value(run)?);
    let corrected_output = normalized_input.transformed_with_context(
        corrected_curve,
        "optimization_filter_synthesis",
        optimization_parameters,
        false,
        Some(OperationContext {
            executed_at: Some(Utc::now().to_rfc3339()),
            tool: Some(ToolIdentity {
                application: Some("autoeq-workflow".into()),
                version: Some(env!("CARGO_PKG_VERSION").into()),
                compiler: Some(run.platform.compiler.clone()),
                os: Some(run.platform.operating_system.clone()),
                architecture: Some(run.platform.architecture.clone()),
                ..Default::default()
            }),
            determinism: Some(autoeq_measurements::Determinism::PlatformSensitive),
        }),
    )?;
    Ok(OptimizationLineage {
        input,
        normalized_input,
        corrected_output,
    })
}

fn workflow_config_identity(
    params: &crate::OptimParams,
    input_curve: &Curve,
    normalized_curve: &Curve,
    target_curve: &Curve,
    deviation_curve: &Curve,
    spin_map: Option<&HashMap<String, Curve>>,
) -> Result<String, Box<dyn Error>> {
    let mut canonical_params = params.clone();
    canonical_params.algo = crate::workflow::resume::canonical_optimizer_identity(&params.algo)
        .map_err(std::io::Error::other)?;
    let mut parts = vec![
        format!("optimizer-params:{canonical_params:#?}"),
        format!("input:{}", input_curve.content_hash()?),
        format!("normalized-input:{}", normalized_curve.content_hash()?),
        format!("target:{}", target_curve.content_hash()?),
        format!("deviation:{}", deviation_curve.content_hash()?),
    ];
    if let Some(spin_map) = spin_map {
        let mut spin_hashes = spin_map
            .iter()
            .map(|(name, curve)| Ok((name, curve.content_hash()?)))
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        spin_hashes.sort_by(|left, right| left.0.cmp(right.0));
        for (name, hash) in spin_hashes {
            parts.push(format!("spin:{name}:{hash}"));
        }
    }
    Ok(config_identity_digest(parts.iter().map(String::as_str)))
}

struct BestCheckpoint {
    loss: f64,
}

type SharedBestCheckpoint = Arc<Mutex<Option<BestCheckpoint>>>;

struct CheckpointIdentity {
    measurement_identity: String,
    config_identity: String,
    normalization_hash: String,
    sample_rate: f64,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    algorithm: String,
    algorithm_version: String,
    budget: usize,
    seed: Option<u64>,
}

struct CheckpointProgressContext {
    path: Option<PathBuf>,
    identity: CheckpointIdentity,
    objective_data: autoeq_optim::ObjectiveData,
    constraint_spec: autoeq_optim::optim::OwnedConstraintSpec,
    errors: Arc<Mutex<Option<String>>>,
    best: SharedBestCheckpoint,
}

fn checkpoint_progress_callback<F>(
    mut user_callback: Option<F>,
    context: CheckpointProgressContext,
) -> impl FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    move |update| {
        let identity_data = &context.identity;
        if let Some(path) = context.path.as_ref()
            && candidate_within_bounds(
                &update.params,
                &identity_data.lower_bounds,
                &identity_data.upper_bounds,
            )
            && let Ok(candidate) = autoeq_optim::optim::finalize_candidate(
                "checkpoint-progress",
                &update.params,
                &context.objective_data,
                &context.constraint_spec.as_spec(),
            )
            && candidate_within_bounds(
                &candidate.params,
                &identity_data.lower_bounds,
                &identity_data.upper_bounds,
            )
        {
            let identity = WarmStartIdentity {
                measurement_identity: &identity_data.measurement_identity,
                config_identity: &identity_data.config_identity,
                normalization_hash: Some(&identity_data.normalization_hash),
                sample_rate: identity_data.sample_rate,
                lower_bounds: &identity_data.lower_bounds,
                upper_bounds: &identity_data.upper_bounds,
                algorithm: &identity_data.algorithm,
                algorithm_version: &identity_data.algorithm_version,
                budget: identity_data.budget,
            };
            let iteration = update.iteration.min(identity_data.budget);
            if candidate.loss.is_finite() {
                let improved = {
                    let mut best = context
                        .best
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if best.as_ref().is_none_or(|best| candidate.loss < best.loss) {
                        *best = Some(BestCheckpoint {
                            loss: candidate.loss,
                        });
                        true
                    } else {
                        false
                    }
                };
                if improved {
                    let state = OptimizerState::from_candidate(
                        &candidate.params,
                        candidate.loss,
                        iteration,
                        identity_data.budget,
                        false,
                        identity_data.seed,
                        true,
                        &identity,
                    );
                    if let Err(error) = save_optimizer_state(&state, path) {
                        *context
                            .errors
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(format!(
                            "failed to save warm-start checkpoint {}: {error}",
                            path.display()
                        ));
                        return crate::de::CallbackAction::Stop;
                    }
                }
            }
        }
        user_callback
            .as_mut()
            .map(|callback| callback(update))
            .unwrap_or(crate::de::CallbackAction::Continue)
    }
}

fn candidate_within_bounds(candidate: &[f64], lower: &[f64], upper: &[f64]) -> bool {
    !candidate.is_empty()
        && candidate.len() == lower.len()
        && candidate.len() == upper.len()
        && candidate
            .iter()
            .zip(lower.iter().zip(upper))
            .all(|(&value, (&minimum, &maximum))| {
                value.is_finite() && value >= minimum && value <= maximum
            })
}

/// Run complete speaker optimization from spinorama data
///
/// # Arguments
/// * `speaker` - Speaker name
/// * `version` - Version (e.g., "asr")
/// * `measurement` - Measurement type (e.g., "CEA2034")
/// * `args` - Optimization arguments (use `Args::speaker_defaults()` as base)
/// * `progress_config` - Optional progress callback configuration
/// * `progress_callback` - Optional progress callback
///
/// # Returns
/// Complete optimization result with all curves
pub async fn optimize_speaker<F>(
    input: &crate::workflow::InputConfig,
    params: &crate::OptimParams,
    progress_config: Option<ProgressCallbackConfig>,
    progress_callback: Option<F>,
) -> Result<SpeakerOptResult, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    optimize_speaker_with_grid(
        input,
        params,
        &crate::workflow::VisualizationGridConfig::default(),
        progress_config,
        progress_callback,
    )
    .await
}

/// Run speaker optimization with an explicit normalization/report grid.
pub async fn optimize_speaker_with_grid<F>(
    input: &crate::workflow::InputConfig,
    params: &crate::OptimParams,
    visualization_grid: &crate::workflow::VisualizationGridConfig,
    progress_config: Option<ProgressCallbackConfig>,
    progress_callback: Option<F>,
) -> Result<SpeakerOptResult, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    optimize_speaker_with_checkpoint(
        input,
        params,
        visualization_grid,
        None,
        None,
        progress_config,
        progress_callback,
    )
    .await
}

/// Optimize a speaker with optional validated warm-start loading and checkpoint saving.
///
/// A loaded checkpoint seeds a fresh optimizer run. It does not restore
/// optimizer population, adaptation state, or random-stream state.
///
/// # Errors
///
/// Returns an error when a requested checkpoint is missing, incompatible, or
/// cannot be saved.
pub async fn optimize_speaker_with_checkpoint<F>(
    input: &crate::workflow::InputConfig,
    params: &crate::OptimParams,
    visualization_grid: &crate::workflow::VisualizationGridConfig,
    resume_path: Option<&Path>,
    checkpoint_path: Option<&Path>,
    progress_config: Option<ProgressCallbackConfig>,
    progress_callback: Option<F>,
) -> Result<SpeakerOptResult, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    // 1. Load measurement with spin data
    let speaker = input.speaker.as_deref().unwrap_or("");
    let version = input.version.as_deref().unwrap_or("");
    let measurement = input.measurement.as_deref().unwrap_or("");
    let (input_curve, spin_data) =
        read::load_spinorama_with_spin(speaker, version, measurement, &input.curve_name).await?;
    let input_record = MeasurementRecord::from_api(
        input_curve.clone(),
        format!(
            "https://api.spinorama.org/v1/speaker/{speaker}/version/{version}/measurements/{measurement}"
        ),
    )?;

    // 2. Normalize to standard frequency grid
    let standard_freq = visualization_grid.create_frequency_grid(params)?;
    let input_normalized = read::normalize_and_interpolate_response(&standard_freq, &input_curve);

    // 3. Build target curve
    let target_curve = build_target_curve(
        &crate::workflow::TargetConfig {
            target_path: None,
            curve_name: input.curve_name.clone(),
        },
        &standard_freq,
        &input_normalized,
    )?;

    // 4. Create deviation curve
    let deviation_curve = Curve {
        freq: target_curve.freq.clone(),
        spl: &target_curve.spl - &input_normalized.spl,
        phase: None,
        ..Default::default()
    };

    // 5. Setup objective - normalize spin data to same frequency grid
    let spin_map = spin_data.as_ref().map(|s| {
        s.curves
            .iter()
            .map(|(name, curve)| {
                let normalized = read::normalize_and_interpolate_response(&standard_freq, curve);
                (name.clone(), normalized)
            })
            .collect::<HashMap<String, Curve>>()
    });
    let (objective_data, _) = setup_objective_data(
        params,
        &input_normalized,
        &target_curve,
        &deviation_curve,
        &spin_map,
    )?;

    let measurement_identity = input_curve.content_hash()?;
    let normalization_hash = input_normalized.content_hash()?;
    let config_identity = workflow_config_identity(
        params,
        &input_curve,
        &input_normalized,
        &target_curve,
        &deviation_curve,
        spin_map.as_ref(),
    )?;
    let algorithm_identity = crate::workflow::resume::canonical_optimizer_identity(&params.algo)
        .map_err(std::io::Error::other)?;
    let (lower_bounds, upper_bounds) = setup_bounds(params);
    let identity = WarmStartIdentity {
        measurement_identity: &measurement_identity,
        config_identity: &config_identity,
        normalization_hash: Some(&normalization_hash),
        sample_rate: params.sample_rate,
        lower_bounds: &lower_bounds,
        upper_bounds: &upper_bounds,
        algorithm: &algorithm_identity,
        algorithm_version: autoeq_optim::optim::OPTIMIZER_IMPLEMENTATION_VERSION,
        budget: params.maxeval,
    };
    let constraint_spec = autoeq_optim::optim::OwnedConstraintSpec::from_params(params)
        .map_err(std::io::Error::other)?;
    let warm_state = match resume_path {
        Some(path) => {
            let state = load_optimizer_state(path)?.ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "requested warm-start checkpoint {} does not exist; run once with checkpoint saving enabled",
                        path.display()
                    ),
                )
            })?;
            state
                .check_warm_start_compatible(&identity)
                .map_err(|reason| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("warm-start checkpoint rejected: {reason}"),
                    )
                })?;
            Some(state)
        }
        None => None,
    };
    let warm_candidate = match warm_state.as_ref() {
        Some(state) => {
            let finalized = autoeq_optim::optim::finalize_candidate(
                "warm-start",
                &state.best_params,
                &objective_data,
                &constraint_spec.as_spec(),
            )
            .map_err(|reason| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("warm-start candidate failed current constraint validation: {reason}"),
                )
            })?;
            if !candidate_within_bounds(&finalized.params, &lower_bounds, &upper_bounds) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "warm-start candidate is outside current optimizer bounds after constraint validation",
                )
                .into());
            }
            Some(finalized.params)
        }
        None => None,
    };
    if warm_candidate.is_some() {
        let backend = autoeq_optim::optim::backend::resolve(&params.algo)
            .ok_or_else(|| std::io::Error::other(format!("unknown optimizer: {}", params.algo)))?;
        if !backend.supports_initial_candidate() {
            return Err(std::io::Error::other(format!(
                "warm-start reuse is unsupported for {} because this optimizer path does not use the supplied initial candidate",
                backend.name()
            ))
            .into());
        }
    }

    // 6. Save a feasible starting point before the potentially long run. Every
    // later replacement is conditional on a better feasible loss.
    let checkpoint_path = checkpoint_path.map(Path::to_path_buf);
    let best_checkpoint: SharedBestCheckpoint = Arc::new(Mutex::new(None));
    if let Some(path) = checkpoint_path.as_ref() {
        let starting_candidate = match warm_candidate.as_ref() {
            Some(candidate) => candidate.clone(),
            None => crate::workflow::initial_guess(params, &lower_bounds, &upper_bounds),
        };
        let finalized = autoeq_optim::optim::finalize_candidate(
            "checkpoint-start",
            &starting_candidate,
            &objective_data,
            &constraint_spec.as_spec(),
        )
        .map_err(|reason| {
            std::io::Error::other(format!(
                "cannot checkpoint a feasible starting candidate: {reason}"
            ))
        })?;
        if !candidate_within_bounds(&finalized.params, &lower_bounds, &upper_bounds) {
            return Err(std::io::Error::other(
                "starting checkpoint candidate is outside current optimizer bounds",
            )
            .into());
        }
        let starting_iteration = warm_state
            .as_ref()
            .map(|state| state.iteration.min(params.maxeval))
            .unwrap_or(0);
        let starting_state = OptimizerState::from_candidate(
            &finalized.params,
            finalized.loss,
            starting_iteration,
            params.maxeval,
            false,
            params.seed,
            true,
            &identity,
        );
        save_optimizer_state(&starting_state, path)?;
        *best_checkpoint
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(BestCheckpoint {
            loss: finalized.loss,
        });
    }

    // Checkpoint writes happen before the user callback so a user-requested
    // stop leaves the latest valid candidate on disk.
    let save_progress = checkpoint_path.is_some();
    let use_progress = save_progress || (progress_config.is_some() && progress_callback.is_some());
    let (opt_params, history, optimization_run) = if use_progress {
        let mut config = progress_config.unwrap_or_default();
        config.interval = config.interval.max(1);
        let checkpoint_errors = Arc::new(Mutex::new(None));
        let callback = checkpoint_progress_callback(
            progress_callback,
            CheckpointProgressContext {
                path: checkpoint_path.clone(),
                identity: CheckpointIdentity {
                    measurement_identity: measurement_identity.clone(),
                    config_identity: config_identity.clone(),
                    normalization_hash: normalization_hash.clone(),
                    sample_rate: params.sample_rate,
                    lower_bounds: lower_bounds.clone(),
                    upper_bounds: upper_bounds.clone(),
                    algorithm: algorithm_identity.clone(),
                    algorithm_version: autoeq_optim::optim::OPTIMIZER_IMPLEMENTATION_VERSION
                        .to_owned(),
                    budget: params.maxeval,
                    seed: params.seed,
                },
                objective_data: objective_data.clone(),
                constraint_spec: constraint_spec.clone(),
                errors: Arc::clone(&checkpoint_errors),
                best: Arc::clone(&best_checkpoint),
            },
        );
        let output_result = match warm_candidate.as_deref() {
            Some(candidate) => perform_optimization_with_progress_and_candidate(
                params,
                &objective_data,
                config,
                candidate,
                callback,
            ),
            None => perform_optimization_with_progress(params, &objective_data, config, callback),
        };
        if let Some(error) = checkpoint_errors
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            return Err(std::io::Error::other(error).into());
        }
        let output = output_result?;
        (output.params, output.history, output.optimization_run)
    } else if let Some(candidate) = warm_candidate.as_deref() {
        let output = perform_optimization_with_run_descriptor_and_candidate(
            params,
            &objective_data,
            candidate,
            Box::new(|_| crate::de::CallbackAction::Continue),
        )?;
        (output.parameters, Vec::new(), output.descriptor)
    } else {
        let output = perform_optimization_with_run_descriptor(
            params,
            &objective_data,
            Box::new(|_| crate::de::CallbackAction::Continue),
        )?;
        (output.parameters, Vec::new(), output.descriptor)
    };

    if let Some(path) = checkpoint_path.as_ref() {
        let finalized = autoeq_optim::optim::finalize_candidate(
            "checkpoint-final",
            &opt_params,
            &objective_data,
            &constraint_spec.as_spec(),
        )
        .map_err(|reason| {
            std::io::Error::other(format!(
                "final checkpoint candidate is infeasible: {reason}"
            ))
        })?;
        let final_identity = WarmStartIdentity {
            measurement_identity: &measurement_identity,
            config_identity: &config_identity,
            normalization_hash: Some(&normalization_hash),
            sample_rate: params.sample_rate,
            lower_bounds: &lower_bounds,
            upper_bounds: &upper_bounds,
            algorithm: &algorithm_identity,
            algorithm_version: autoeq_optim::optim::OPTIMIZER_IMPLEMENTATION_VERSION,
            budget: params.maxeval,
        };
        if !candidate_within_bounds(&finalized.params, &lower_bounds, &upper_bounds) {
            return Err(std::io::Error::other(
                "final checkpoint candidate is outside current optimizer bounds",
            )
            .into());
        }
        let final_iteration = history
            .last()
            .map(|(iteration, _)| *iteration)
            .unwrap_or(0)
            .min(params.maxeval);
        let final_is_better = {
            let mut best = best_checkpoint
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if best.as_ref().is_none_or(|best| finalized.loss < best.loss) {
                *best = Some(BestCheckpoint {
                    loss: finalized.loss,
                });
                true
            } else {
                false
            }
        };
        if final_is_better {
            let final_state = OptimizerState::from_candidate(
                &finalized.params,
                finalized.loss,
                final_iteration,
                params.maxeval,
                false,
                params.seed,
                true,
                &final_identity,
            );
            save_optimizer_state(&final_state, path)?;
        }
    }

    // 7. Convert to biquads
    let biquads: Vec<Biquad> = x2peq(&opt_params, params.sample_rate, params.peq_model)
        .into_iter()
        .map(|(_, b)| b)
        .collect();

    // 8. Compute visualization curves
    let frequencies: Vec<f64> = standard_freq.iter().copied().collect();
    let curves =
        compute_visualization_curves(&frequencies, &input_normalized, &target_curve, &biquads)?;
    let corrected_curve = Curve {
        freq: standard_freq.clone(),
        spl: ndarray::Array1::from_vec(curves.corrected_curve.clone()),
        ..Default::default()
    };
    let lineage = workflow_lineage(
        input_record,
        input_normalized.clone(),
        corrected_curve,
        &target_curve,
        &optimization_run,
    )?;

    let initial_loss = history.first().map(|x| x.1).unwrap_or(0.0);
    let final_loss = history.last().map(|x| x.1).unwrap_or(0.0);

    // Interpolate spin_data to standard frequency grid for consistent visualization
    // Note: Does NOT normalize - preserves original dB levels
    let interpolated_spin_data = spin_data.map(|s| interpolate_cea2034_data(&s, &standard_freq));

    Ok(SpeakerOptResult {
        biquads,
        curves,
        spin_data: interpolated_spin_data,
        history,
        initial_loss,
        final_loss,
        optimization_run,
        lineage,
    })
}

/// Run complete headphone optimization from CSV measurement
///
/// # Arguments
/// * `curve_path` - Path to headphone measurement CSV
/// * `target_curve` - Target curve (use bundled Harman curves or custom)
/// * `args` - Optimization arguments (use `Args::headphone_defaults()` as base)
/// * `progress_config` - Optional progress callback configuration
/// * `progress_callback` - Optional progress callback
///
/// # Returns
/// Complete optimization result with all curves
pub fn optimize_headphone<F>(
    curve_path: &PathBuf,
    target_curve: &Curve,
    params: &crate::OptimParams,
    progress_config: Option<ProgressCallbackConfig>,
    progress_callback: Option<F>,
) -> Result<HeadphoneOptResult, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    optimize_headphone_with_grid(
        curve_path,
        target_curve,
        params,
        &crate::workflow::VisualizationGridConfig::default(),
        progress_config,
        progress_callback,
    )
}

/// Run headphone optimization with an explicit normalization/report grid.
pub fn optimize_headphone_with_grid<F>(
    curve_path: &PathBuf,
    target_curve: &Curve,
    params: &crate::OptimParams,
    visualization_grid: &crate::workflow::VisualizationGridConfig,
    progress_config: Option<ProgressCallbackConfig>,
    progress_callback: Option<F>,
) -> Result<HeadphoneOptResult, Box<dyn Error>>
where
    F: FnMut(&ProgressUpdate) -> crate::de::CallbackAction + Send + 'static,
{
    // 1. Load measurement
    let input_record = MeasurementRecord::from_source_path(
        read::read_curve_from_csv(curve_path)?,
        MeasurementOrigin::Csv,
        curve_path,
    )?;
    let input_curve = input_record.curve.clone();

    // 2. Normalize to standard frequency grid
    let standard_freq = visualization_grid.create_frequency_grid(params)?;
    let input_normalized = read::normalize_and_interpolate_response(&standard_freq, &input_curve);
    let target_normalized = read::normalize_and_interpolate_response(&standard_freq, target_curve);

    // 3. Create deviation curve
    let deviation_curve = Curve {
        freq: target_normalized.freq.clone(),
        spl: &target_normalized.spl - &input_normalized.spl,
        phase: None,
        ..Default::default()
    };

    // 4. Setup objective
    let (objective_data, _) = setup_objective_data(
        params,
        &input_normalized,
        &target_normalized,
        &deviation_curve,
        &None,
    )?;

    // 5. Run optimization
    let (opt_params, history, optimization_run) = if let (Some(config), Some(callback)) =
        (progress_config, progress_callback)
    {
        let output = perform_optimization_with_progress(params, &objective_data, config, callback)?;
        (output.params, output.history, output.optimization_run)
    } else {
        let output = perform_optimization_with_run_descriptor(
            params,
            &objective_data,
            Box::new(|_| crate::de::CallbackAction::Continue),
        )?;
        (output.parameters, Vec::new(), output.descriptor)
    };

    // 6. Convert to biquads
    let biquads: Vec<Biquad> = x2peq(&opt_params, params.sample_rate, params.peq_model)
        .into_iter()
        .map(|(_, b)| b)
        .collect();

    // 7. Compute visualization curves
    let frequencies: Vec<f64> = standard_freq.iter().copied().collect();
    let curves = compute_visualization_curves(
        &frequencies,
        &input_normalized,
        &target_normalized,
        &biquads,
    )?;
    let corrected_curve = Curve {
        freq: standard_freq.clone(),
        spl: ndarray::Array1::from_vec(curves.corrected_curve.clone()),
        ..Default::default()
    };
    let lineage = workflow_lineage(
        input_record,
        input_normalized.clone(),
        corrected_curve,
        &target_normalized,
        &optimization_run,
    )?;

    let initial_loss = history.first().map(|x| x.1).unwrap_or(0.0);
    let final_loss = history.last().map(|x| x.1).unwrap_or(0.0);

    Ok(HeadphoneOptResult {
        biquads,
        curves,
        history,
        initial_loss,
        final_loss,
        optimization_run,
        lineage,
    })
}

/// Optimize multi-driver crossover configuration
///
/// This function orchestrates the complete driver optimization workflow:
/// 1. Sets up optimization objective data
/// 2. Computes parameter bounds
/// 3. Generates initial guess
/// 4. Runs optimization
/// 5. Extracts gains and crossover frequencies from results
///
/// # Arguments
/// * `drivers_data` - Driver measurements with crossover type
/// * `min_freq`, `max_freq` - Optimization frequency range (Hz)
/// * `sample_rate` - Sample rate for filter design (Hz)
/// * `algorithm` - Optimization algorithm (e.g., "autoeq:cobyla", "autoeq:de")
/// * `max_iter` - Maximum number of iterations/evaluations
/// * `min_db`, `max_db` - Per-driver gain bounds (dB)
///
/// # Returns
/// * `DriverOptimizationResult` containing optimal gains, crossover frequencies, and scores
///
/// # Example
/// ```ignore
/// let drivers_data = DriversLossData::new(measurements, CrossoverType::LinkwitzRiley4);
/// let result = optimize_drivers_crossover(
///     drivers_data,
///     100.0,    // min_freq
///     10000.0,  // max_freq
///     48000.0,  // sample_rate
///     "autoeq:cobyla",
///     5000,     // max_iter
///     -12.0,    // min_db
///     12.0,     // max_db
///     None,     // fixed_freqs
///     None,     // seed
/// )?;
/// log::info!("Gains: {:?}", result.gains);
/// log::info!("Crossover freqs: {:?}", result.crossover_freqs);
/// ```
fn validate_workflow_sample_rate(sample_rate: f64) -> Result<(), Box<dyn std::error::Error>> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(format!("sample rate must be finite and positive, got {sample_rate}").into());
    }
    Ok(())
}

fn with_driver_smoothness(
    mut params: crate::OptimParams,
    smoothness_penalty: Option<autoeq_optim::SmoothnessPenaltyConfig>,
) -> crate::OptimParams {
    params.smoothness_penalty = smoothness_penalty;
    params
}

#[allow(clippy::too_many_arguments)]
pub fn optimize_drivers_crossover(
    drivers_data: crate::loss::DriversLossData,
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
    optimize_drivers_crossover_with_smoothness(
        drivers_data,
        min_freq,
        max_freq,
        sample_rate,
        algorithm,
        max_iter,
        population,
        min_db,
        max_db,
        fixed_freqs,
        seed,
        None,
    )
}

/// Optimize a multi-driver crossover with an optional correction-curve
/// smoothness penalty.
#[allow(clippy::too_many_arguments)]
pub fn optimize_drivers_crossover_with_smoothness(
    drivers_data: crate::loss::DriversLossData,
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
    smoothness_penalty: Option<autoeq_optim::SmoothnessPenaltyConfig>,
) -> Result<DriverOptimizationResult, Box<dyn std::error::Error>> {
    validate_workflow_sample_rate(sample_rate)?;
    let n_drivers = drivers_data.drivers.len();

    // Create optimization parameters for driver optimization
    let params = with_driver_smoothness(
        create_driver_optimization_args(
            min_freq,
            max_freq,
            sample_rate,
            algorithm,
            max_iter,
            population,
            min_db,
            max_db,
            seed,
        ),
        smoothness_penalty,
    );

    // Setup objective data with optional fixed frequencies.
    // When fixed crossover frequencies are supplied we must rebuild the cached
    // objective strategy so it sees the fixed frequencies.
    let objective_data = if let Some(ref freqs) = fixed_freqs {
        let mut data = setup_drivers_objective_data(&params, drivers_data.clone());
        data.fixed_crossover_freqs = Some(freqs.clone());
        data.objective = Some(data.build_objective());
        data
    } else {
        setup_drivers_objective_data(&params, drivers_data.clone())
    };

    // Setup bounds (exclude crossover frequencies if fixed)
    let (lower_bounds, upper_bounds) = if fixed_freqs.is_some() {
        setup_drivers_bounds_fixed_freqs(&params, &drivers_data)
    } else {
        setup_drivers_bounds(&params, &drivers_data)
    };

    // Generate initial guess
    let mut x = if fixed_freqs.is_some() {
        drivers_initial_guess_fixed_freqs(&lower_bounds, &upper_bounds, n_drivers)
    } else {
        drivers_initial_guess(&lower_bounds, &upper_bounds, n_drivers)
    };
    let initial_x = x.clone();

    // Compute pre-optimization objective
    let pre_objective = crate::optim::compute_base_fitness(&x, &objective_data);

    // Run optimization and classify convergence from structured evidence.
    // A usable best-effort `Ok` whose status reports budget exhaustion must
    // NOT set `converged`: convergence, parameter usability (rollback guard
    // below), and identity fallback (fixed freqs) are independent decisions.
    let evidence = crate::optim::optimize_filters_detailed(
        &mut x,
        &lower_bounds,
        &upper_bounds,
        objective_data.clone(),
        &params,
    );
    let converged = evidence.converged;

    // Compute post-optimization objective
    let mut post_objective = crate::optim::compute_base_fitness(&x, &objective_data);
    if !post_objective.is_finite() || post_objective > pre_objective {
        x = initial_x;
        post_objective = pre_objective;
    }

    // Extract results from parameter vector
    let gains = x[0..n_drivers].to_vec();
    let delays = x[n_drivers..2 * n_drivers].to_vec();

    // Crossover frequencies: from optimization or fixed
    let crossover_freqs = if let Some(freqs) = fixed_freqs {
        freqs
    } else {
        // Parameter layout: [gains(N), delays(N), xovers(N-1)]
        let xover_freqs_log10 = &x[2 * n_drivers..];
        xover_freqs_log10.iter().map(|x| 10_f64.powf(*x)).collect()
    };

    Ok(DriverOptimizationResult {
        gains,
        delays,
        crossover_freqs,
        pre_objective,
        post_objective,
        converged,
    })
}

/// Timing breakdown for a multi-sub optimization run.
///
/// `setup_secs` covers building the prepared objective, bounds, and initial
/// guess (measurement-derived state). `eval_secs` covers candidate evaluation
/// only: the pre/post fitness probes and `optimize_filters_detailed`, excluding
/// setup and result extraction. Use it to attribute wall-clock cost when
/// profiling repeated parameter sweeps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriverOptimizationTiming {
    /// Seconds spent preparing objective data, bounds, and initial guess.
    pub setup_secs: f64,
    /// Seconds spent evaluating candidates (pre/post probes + optimizer).
    pub eval_secs: f64,
}

/// Multi-sub objective prepared once and reused across related searches.
///
/// Bundles the built objective data, parameter bounds, optimizer parameters,
/// driver count, and the setup cost, so repeated sweeps (different seeds, warm
/// starts) do not reload or rebuild measurement-derived state. Build with
/// [`prepare_multisub_objective`] and run with [`optimize_multisub_prepared`].
#[derive(Debug, Clone)]
pub struct PreparedMultisubObjective {
    /// Optimizer parameters (loss is forced to multi-sub-flat).
    pub params: crate::OptimParams,
    /// Built objective data shared by every search on this problem.
    pub objective_data: crate::optim::ObjectiveData,
    /// Number of subwoofers (parameter layout is `[gains(N), delays(N)]`).
    pub n_drivers: usize,
    /// Gain/delay lower bounds.
    pub lower_bounds: Vec<f64>,
    /// Gain/delay upper bounds.
    pub upper_bounds: Vec<f64>,
    /// Seconds spent building this prepared state.
    pub setup_secs: f64,
}

/// Prepare multi-sub objective data once for reuse across related searches.
///
/// Builds the optimizer parameters, objective data, and parameter bounds a
/// single time so repeated sweeps (different seeds, warm starts) share the
/// measurement-derived state instead of rebuilding it per run.
#[allow(clippy::too_many_arguments)]
pub fn prepare_multisub_objective(
    drivers_data: crate::loss::DriversLossData,
    min_freq: f64,
    max_freq: f64,
    sample_rate: f64,
    algorithm: &str,
    max_iter: usize,
    population: usize,
    min_db: f64,
    max_db: f64,
) -> Result<PreparedMultisubObjective, Box<dyn std::error::Error>> {
    validate_workflow_sample_rate(sample_rate)?;
    let setup_start = std::time::Instant::now();
    let mut params = create_driver_optimization_args(
        min_freq,
        max_freq,
        sample_rate,
        algorithm,
        max_iter,
        population,
        min_db,
        max_db,
        None,
    );
    params.loss = crate::LossType::MultiSubFlat;
    let n_drivers = drivers_data.drivers.len();
    let objective_data = setup_multisub_objective_data(&params, drivers_data);
    let (lower_bounds, upper_bounds) = setup_multisub_bounds(&params, n_drivers);
    let setup_secs = setup_start.elapsed().as_secs_f64();
    Ok(PreparedMultisubObjective {
        params,
        objective_data,
        n_drivers,
        lower_bounds,
        upper_bounds,
        setup_secs,
    })
}

/// Run a multi-sub search on a prepared objective with a chosen seed.
///
/// Reuses the prepared objective data and bounds; only the seed varies per
/// call. Returns the optimization result alongside a timing breakdown that
/// separates the (amortized) setup cost from candidate-evaluation time.
pub fn optimize_multisub_prepared(
    prepared: &PreparedMultisubObjective,
    seed: Option<u64>,
) -> Result<(DriverOptimizationResult, DriverOptimizationTiming), Box<dyn std::error::Error>> {
    let mut params = prepared.params.clone();
    params.seed = seed;
    let n_drivers = prepared.n_drivers;

    // Initial guess
    let mut x = multisub_initial_guess(n_drivers);
    let initial_x = x.clone();

    // Candidate evaluation only: pre/post probes plus the optimizer itself.
    let eval_start = std::time::Instant::now();
    let pre_objective = crate::optim::compute_base_fitness(&x, &prepared.objective_data);

    // Optimize; convergence comes from structured evidence so a usable
    // best-effort `Ok` after budget exhaustion does not report converged.
    let evidence = crate::optim::optimize_filters_detailed(
        &mut x,
        &prepared.lower_bounds,
        &prepared.upper_bounds,
        prepared.objective_data.clone(),
        &params,
    );
    let converged = evidence.converged;

    let mut post_objective = crate::optim::compute_base_fitness(&x, &prepared.objective_data);
    if !post_objective.is_finite() || post_objective > pre_objective {
        x = initial_x;
        post_objective = pre_objective;
    }
    let eval_secs = eval_start.elapsed().as_secs_f64();

    // Extract results: [gains(N), delays(N)]
    let gains = x[0..n_drivers].to_vec();
    let delays = x[n_drivers..2 * n_drivers].to_vec();

    Ok((
        DriverOptimizationResult {
            gains,
            delays,
            crossover_freqs: vec![],
            pre_objective,
            post_objective,
            converged,
        },
        DriverOptimizationTiming {
            setup_secs: prepared.setup_secs,
            eval_secs,
        },
    ))
}

/// Optimize multi-subwoofer configuration (gain, delay) to achieve flat summed response
#[allow(clippy::too_many_arguments)]
pub fn optimize_multisub(
    drivers_data: crate::loss::DriversLossData,
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
    validate_workflow_sample_rate(sample_rate)?;
    let n_drivers = drivers_data.drivers.len();

    // Create optimization parameters for multi-sub optimization
    let mut params = create_driver_optimization_args(
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
    params.loss = crate::LossType::MultiSubFlat;

    // Setup objective data
    let objective_data = setup_multisub_objective_data(&params, drivers_data.clone());

    // Setup bounds (gains + delays)
    let (lower_bounds, upper_bounds) = setup_multisub_bounds(&params, n_drivers);

    // Initial guess
    let mut x = multisub_initial_guess(n_drivers);
    let initial_x = x.clone();

    // Pre-objective
    let pre_objective = crate::optim::compute_base_fitness(&x, &objective_data);

    // Optimize; convergence comes from structured evidence so a usable
    // best-effort `Ok` after budget exhaustion does not report converged.
    let evidence = crate::optim::optimize_filters_detailed(
        &mut x,
        &lower_bounds,
        &upper_bounds,
        objective_data.clone(),
        &params,
    );
    let converged = evidence.converged;

    let mut post_objective = crate::optim::compute_base_fitness(&x, &objective_data);
    if !post_objective.is_finite() || post_objective > pre_objective {
        x = initial_x;
        post_objective = pre_objective;
    }

    // Extract results: [gains(N), delays(N)]
    let gains = x[0..n_drivers].to_vec();
    let delays = x[n_drivers..2 * n_drivers].to_vec();
    let crossover_freqs = vec![];

    Ok(DriverOptimizationResult {
        gains,
        delays,
        crossover_freqs,
        pre_objective,
        post_objective,
        converged,
    })
}

/// Split one driver-optimization outcome into independent decisions.
///
/// * `converged` follows only [`crate::optim::OptimizerRunEvidence::converged`],
///   so a usable best-effort `Ok` after budget exhaustion reports `false`.
/// * `usable` follows only the nonregression guard: the candidate parameters
///   are kept when the post objective is finite and does not regress past
///   `pre_objective`, otherwise the caller rolls back to the initial vector.
/// * identity fallback (fixed crossover frequencies) is handled separately by
///   the caller and is unaffected by either flag.
#[cfg(test)]
pub(crate) fn classify_driver_outcome(
    evidence: &crate::optim::OptimizerRunEvidence,
    post_objective: f64,
    pre_objective: f64,
) -> (bool, bool) {
    let converged = evidence.converged;
    let usable = post_objective.is_finite() && post_objective <= pre_objective;
    (converged, usable)
}

#[cfg(test)]
mod driver_smoothness_tests {
    use super::*;

    #[test]
    fn driver_workflow_preserves_smoothness_configuration() {
        let penalty = autoeq_optim::SmoothnessPenaltyConfig {
            tv2_weight: 0.25,
            schroeder_hz: Some(180.0),
            modal_weight_scale: 0.2,
            exponent: 1.5,
        };
        let params = with_driver_smoothness(
            create_driver_optimization_args(
                20.0,
                20_000.0,
                48_000.0,
                "autoeq:de",
                100,
                20,
                -12.0,
                12.0,
                Some(7),
            ),
            Some(penalty.clone()),
        );
        let actual = params.smoothness_penalty.expect("smoothness penalty");
        assert_eq!(actual.tv2_weight, penalty.tv2_weight);
        assert_eq!(actual.schroeder_hz, penalty.schroeder_hz);
        assert_eq!(actual.modal_weight_scale, penalty.modal_weight_scale);
        assert_eq!(actual.exponent, penalty.exponent);
    }

    #[test]
    fn finite_best_effort_result_is_usable_but_not_converged() {
        let evidence = crate::optim::OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Ok((
                "AutoEQ DE: maximum evaluations reached (not converged, nfev=100)".to_string(),
                1.25,
            )),
            &[0.5],
            &[0.0],
            &[1.0],
            100,
            Some(7),
        );
        assert!(evidence.best_effort);
        // Finite improved objective: parameters usable, convergence false.
        let (converged, usable) = super::classify_driver_outcome(&evidence, 1.0, 2.0);
        assert!(!converged);
        assert!(usable);
    }

    #[test]
    fn regressed_result_triggers_rollback_without_convergence() {
        let evidence = crate::optim::OptimizerRunEvidence::from_backend_result(
            "autoeq:de",
            Ok((
                "AutoEQ DE: maximum evaluations reached (not converged, nfev=100)".to_string(),
                3.0,
            )),
            &[0.5],
            &[0.0],
            &[1.0],
            100,
            Some(7),
        );
        // Regressed objective mirrors the nonregression guard: roll back.
        let (converged, usable) = super::classify_driver_outcome(&evidence, 3.0, 2.0);
        assert!(!converged);
        assert!(!usable);
        // Non-finite post objective is never usable either.
        let (_, usable) = super::classify_driver_outcome(&evidence, f64::INFINITY, 2.0);
        assert!(!usable);
    }
}

#[cfg(test)]
mod checkpoint_progress_tests {
    use super::*;
    use ndarray::Array1;

    fn constant_curve(level: f64) -> Curve {
        let freq = Array1::from_vec(vec![20.0, 1_000.0, 20_000.0]);
        Curve {
            spl: Array1::from_elem(freq.len(), level),
            freq,
            phase: None,
            ..Default::default()
        }
    }

    #[test]
    fn progress_checkpoint_records_reusable_optimizer_version() {
        let directory = tempfile::tempdir().unwrap();
        let checkpoint_path = directory.path().join("optimizer-state.json");
        let args = crate::cli::Args::speaker_defaults();
        let mut params = crate::OptimParams::from(&args);
        params.num_filters = 1;
        params.maxeval = 8;
        let input = constant_curve(80.0);
        let target = constant_curve(80.0);
        let deviation = constant_curve(0.0);
        let (objective_data, _) = setup_objective_data(&params, &input, &target, &deviation, &None)
            .expect("test objective data should be valid");
        let (lower_bounds, upper_bounds) = setup_bounds(&params);
        let candidate = initial_guess(&params, &lower_bounds, &upper_bounds);
        let constraint_spec =
            autoeq_optim::optim::OwnedConstraintSpec::from_params(&params).unwrap();
        let measurement_identity = "measurement-checkpoint-test".to_owned();
        let config_identity = "config-checkpoint-test".to_owned();
        let normalization_hash = "normalization-checkpoint-test".to_owned();
        let algorithm = "autoeq:de".to_owned();
        let best_checkpoint = Arc::new(Mutex::new(Some(BestCheckpoint { loss: f64::MAX })));
        let checkpoint_errors = Arc::new(Mutex::new(None));
        let mut callback = checkpoint_progress_callback(
            Some(|_: &ProgressUpdate| crate::de::CallbackAction::Continue),
            CheckpointProgressContext {
                path: Some(checkpoint_path.clone()),
                identity: CheckpointIdentity {
                    measurement_identity: measurement_identity.clone(),
                    config_identity: config_identity.clone(),
                    normalization_hash: normalization_hash.clone(),
                    sample_rate: params.sample_rate,
                    lower_bounds: lower_bounds.clone(),
                    upper_bounds: upper_bounds.clone(),
                    algorithm: algorithm.clone(),
                    algorithm_version: autoeq_optim::optim::OPTIMIZER_IMPLEMENTATION_VERSION
                        .to_owned(),
                    budget: params.maxeval,
                    seed: params.seed,
                },
                objective_data: objective_data.clone(),
                constraint_spec,
                errors: Arc::clone(&checkpoint_errors),
                best: Arc::clone(&best_checkpoint),
            },
        );

        let update = ProgressUpdate {
            iteration: 3,
            max_iterations: params.maxeval,
            loss: f64::MAX,
            score: None,
            convergence: 0.0,
            params: candidate,
            biquads: Vec::new(),
            filter_response: Vec::new(),
        };
        assert!(matches!(
            callback(&update),
            crate::de::CallbackAction::Continue
        ));
        assert!(
            checkpoint_errors
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_none()
        );

        let saved = load_optimizer_state(&checkpoint_path)
            .unwrap()
            .expect("progress update should write an improved checkpoint");
        assert_eq!(
            saved.algorithm_version.as_deref(),
            Some(autoeq_optim::optim::OPTIMIZER_IMPLEMENTATION_VERSION)
        );
        assert_eq!(saved.iteration, 3);
        let identity = WarmStartIdentity {
            measurement_identity: &measurement_identity,
            config_identity: &config_identity,
            normalization_hash: Some(&normalization_hash),
            sample_rate: params.sample_rate,
            lower_bounds: &lower_bounds,
            upper_bounds: &upper_bounds,
            algorithm: &algorithm,
            algorithm_version: autoeq_optim::optim::OPTIMIZER_IMPLEMENTATION_VERSION,
            budget: params.maxeval,
        };
        saved
            .check_warm_start_compatible(&identity)
            .expect("progress checkpoint should be reusable with its exact optimizer identity");
    }
}
