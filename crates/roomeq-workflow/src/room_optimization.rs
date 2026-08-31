//! Main optimization entry points for room EQ.
//!
//! This module provides the primary public API for room optimization.

use crate::fir;
use crate::pipeline::{RoomPipeline, RoomPipelineRequest};
use gd::*;
use log::{debug, info, warn};
use math_audio_dsp::analysis::compute_average_response;
use math_audio_iir_fir::Biquad;
use num_complex::Complex64;
use phase::*;
use reports::*;
use roomeq_engine::analysis::crossover_utils::check_group_consistency;
use roomeq_engine::{
    PipelineEvent, PipelineObserver, PipelineStepId, PipelineStepStatus, output, phase_alignment,
};
use roomeq_model::{
    AutoeqError, ChannelDspChain, Curve, MeasurementSource, OptimizationMetadata, OptimizerConfig,
    PerceptualMetrics, ProcessingMode, Result, RoomConfig, SpeakerConfig, StageOutcome,
    StageStatus, SystemConfig, SystemModel, TargetCurveConfig,
};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

mod gd;
mod phase;
mod reports;

mod misc;
mod process;
mod room_optimization_callback_observer;
mod room_optimization_progress;
mod room_optimization_result;
#[cfg(test)]
mod tests;
mod types;
mod validation_scorecard;

pub use room_optimization_progress::*;
pub use room_optimization_result::*;
pub use types::*;

use misc::ARRIVAL_TIME_WARNING_THRESHOLD_MS;
use misc::LEVEL_DIFFERENCE_WARNING_THRESHOLD;
use misc::channels_for_generic_optimization;
use misc::compute_shared_mean_spl_with_frequency_samples;
use misc::find_sub_main_pairings;
use misc::generic_channel_progress_iterations;
use misc::identify_acoustic_groups;
use misc::is_subwoofer_channel;
use misc::optimizer_progress_iterations;
use misc::prepare_room_config_with_frequency_samples;
use misc::validate_room_config_or_fail_with_frequency_samples;
use process::process_generic_channels;
use process::process_speaker_internal;
use room_optimization_callback_observer::callback_pipeline_observer;
use room_optimization_progress::send_progress;
use room_optimization_result::apply_ctc_if_enabled;
use room_optimization_result::sanity_check_result;
use types::GenericChannelCollection;
use types::SharedPipelineObserver;

fn should_run_standalone_phase_correction(config: &RoomConfig) -> bool {
    config.optimizer.processing_mode != ProcessingMode::MixedPhase
        && config.optimizer.phase_correction.is_some()
}
use types::collect_generic_channel_results;
use types::emit_pipeline_event;

/// Optimize a complete room configuration
///
/// Processes all speakers in parallel and returns DSP chains for each channel.
///
/// # Arguments
/// * `config` - Complete room configuration
/// * `sample_rate` - Sample rate for filter design (e.g., 48000.0)
/// * `callback` - Optional progress callback
/// * `output_dir` - Optional directory for writing intermediate artifacts
///
/// # Returns
/// * `RoomOptimizationResult` containing DSP chains and optimization results
pub fn optimize_room(
    config: &RoomConfig,
    sample_rate: f64,
    callback: Option<RoomOptimizationCallback>,
    output_dir: Option<&Path>,
) -> Result<RoomOptimizationResult> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(roomeq_model::AutoeqError::InvalidConfiguration {
            message: format!("sample rate must be finite and positive, got {sample_rate}"),
        });
    }
    RoomPipeline::new(RoomPipelineRequest {
        config,
        sample_rate,
        output_dir,
        probe_arrival_overrides: None,
    })
    .run(callback.map(callback_pipeline_observer))
}

/// Same as [`optimize_room`] but accepts per-channel probe-based arrival times.
///
/// When the delay-detection UI step has measured arrival times with a tone
/// burst probe, pass them here so the channel workflow uses the measured
/// values instead of falling back to WAV-onset detection. Channels absent from
/// `probe_arrival_ms` still use the WAV-onset fallback.
pub fn optimize_room_with_probe_arrivals(
    config: &RoomConfig,
    sample_rate: f64,
    callback: Option<RoomOptimizationCallback>,
    output_dir: Option<&Path>,
    probe_arrival_ms: &HashMap<String, f64>,
) -> Result<RoomOptimizationResult> {
    RoomPipeline::new(RoomPipelineRequest {
        config,
        sample_rate,
        output_dir,
        probe_arrival_overrides: Some(probe_arrival_ms),
    })
    .run(callback.map(callback_pipeline_observer))
}

pub(super) fn optimize_room_pipeline_impl_with_frequency_samples(
    request: roomeq_engine::EngineRequest<'_>,
    context: &crate::WorkflowContext<'_>,
    observer: Option<Box<dyn PipelineObserver>>,
    frequency_samples: usize,
) -> Result<RoomOptimizationResult> {
    let mut result = optimize_room_impl_with_frequency_samples(
        request.config,
        request.sample_rate,
        context.output_dir,
        request.probe_arrival_overrides,
        observer,
        context.artifact_store,
        frequency_samples,
    )?;
    if !context.validation_measurements.is_empty() {
        validation_scorecard::attach_validation_scorecard(
            &mut result,
            context.validation_measurements,
            request.sample_rate,
        )?;
    }
    Ok(result)
}

fn prepare_room_optimization_with_frequency_samples(
    config: &RoomConfig,
    observer: Option<Box<dyn PipelineObserver>>,
    frequency_samples: usize,
) -> Result<(SharedPipelineObserver, RoomConfig)> {
    let observer_shared: SharedPipelineObserver = Arc::new(Mutex::new(observer));

    emit_pipeline_event(
        &observer_shared,
        PipelineEvent::started(
            PipelineStepId::ConfigPreparation,
            "Preparing room optimization configuration",
        ),
    )?;

    let config = prepare_room_config_with_frequency_samples(config, frequency_samples);

    emit_pipeline_event(
        &observer_shared,
        PipelineEvent::completed(
            PipelineStepId::ConfigPreparation,
            "Room optimization configuration prepared",
        ),
    )?;

    Ok((observer_shared, config))
}

fn validate_room_optimization_with_frequency_samples(
    config: &RoomConfig,
    observer_shared: &SharedPipelineObserver,
    frequency_samples: usize,
) -> Result<()> {
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::started(PipelineStepId::Validation, "Validating room configuration"),
    )?;
    validate_room_config_or_fail_with_frequency_samples(config, frequency_samples)?;
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::completed(PipelineStepId::Validation, "Room configuration validated"),
    )?;
    Ok(())
}

fn select_topology_route(
    config: &RoomConfig,
    observer_shared: &SharedPipelineObserver,
) -> Result<TopologyRoute> {
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::started(
            PipelineStepId::TopologyRouteSelection,
            "Selecting Room EQ topology route",
        ),
    )?;

    let Some(sys) = &config.system else {
        return Ok(TopologyRoute::Generic);
    };

    // Multi-driver main channels use the generic topology-aware path.  The
    // home-cinema executor owns its designated bass output and explicitly
    // supports MultiSub/MSO, cardioid, and DBA preprocessing; treating that
    // output as an unsupported main group silently bypasses routed bass
    // management and leaves the published routing graph unrealized.
    let home_cinema_bass_output = (sys.model == SystemModel::HomeCinema)
        .then(|| roomeq_engine::home_cinema::bass_output_role(config, sys));
    let has_group = sys.speakers.iter().any(|(role, key)| {
        if home_cinema_bass_output
            .as_ref()
            .is_some_and(|bass_output| role == bass_output)
        {
            return false;
        }
        matches!(
            config.speakers.get(key),
            Some(
                SpeakerConfig::Group(_)
                    | SpeakerConfig::Topology(_)
                    | SpeakerConfig::MultiSub(_)
                    | SpeakerConfig::Cardioid(_)
                    | SpeakerConfig::Dba(_)
            )
        )
    });

    if has_group {
        return Ok(TopologyRoute::Generic);
    }

    match sys.model {
        SystemModel::Stereo => {
            let workflow_name = if sys.subwoofers.is_some() {
                "Stereo 2.1"
            } else {
                "Stereo 2.0"
            };
            emit_pipeline_event(
                observer_shared,
                PipelineEvent::completed(
                    PipelineStepId::TopologyRouteSelection,
                    format!("Selected {} workflow", workflow_name),
                ),
            )?;
            if sys.subwoofers.is_some() {
                Ok(TopologyRoute::Stereo2_1)
            } else {
                Ok(TopologyRoute::Stereo2_0)
            }
        }
        SystemModel::HomeCinema => {
            emit_pipeline_event(
                observer_shared,
                PipelineEvent::completed(
                    PipelineStepId::TopologyRouteSelection,
                    "Selected Home Cinema workflow",
                ),
            )?;
            Ok(TopologyRoute::HomeCinema)
        }
        SystemModel::Custom => Ok(TopologyRoute::Generic),
    }
}

#[cfg(test)]
fn execute_topology_workflow(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
    observer_shared: &SharedPipelineObserver,
    route: TopologyRoute,
) -> Result<RoomOptimizationResult> {
    execute_topology_workflow_with_probe_arrivals(
        config,
        sys,
        sample_rate,
        output_dir,
        None,
        observer_shared,
        route,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_topology_workflow_with_probe_arrivals(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
    probe_arrival_overrides: Option<&HashMap<String, f64>>,
    observer_shared: &SharedPipelineObserver,
    route: TopologyRoute,
    frequency_samples: usize,
) -> Result<RoomOptimizationResult> {
    let worker_count = config
        .optimizer
        .parallel_threads
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(1)
        })
        .max(1);
    let executor = crate::executor::RoomEqExecutor::new(worker_count)?;
    executor.install(|| {
        execute_topology_workflow_on_pool(
            config,
            sys,
            sample_rate,
            output_dir,
            probe_arrival_overrides,
            observer_shared,
            route,
            frequency_samples,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_topology_workflow_on_pool(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
    probe_arrival_overrides: Option<&HashMap<String, f64>>,
    observer_shared: &SharedPipelineObserver,
    route: TopologyRoute,
    frequency_samples: usize,
) -> Result<RoomOptimizationResult> {
    let workflow_name = match sys.model {
        SystemModel::Stereo => {
            if sys.subwoofers.is_some() {
                "Stereo 2.1"
            } else {
                "Stereo 2.0"
            }
        }
        SystemModel::HomeCinema => "Home Cinema",
        SystemModel::Custom => "Custom",
    };

    // Send pre-workflow progress message
    if sys.model != SystemModel::Custom {
        send_progress(
            observer_shared,
            PipelineStepId::TopologyWorkflowExecution,
            PipelineStepStatus::Started,
            &RoomOptimizationProgress {
                current_speaker: String::new(),
                speaker_index: 0,
                total_speakers: sys.speakers.len(),
                iteration: 0,
                max_iterations: 0,
                loss: 0.0,
                overall_progress: 0.0,
                message: Some(format!(
                    "Starting {} workflow ({} channels)",
                    workflow_name,
                    sys.speakers.len()
                )),
                epa_preference: None,
                step_id: None,
                step_status: None,
            },
        )?;
        send_progress(
            observer_shared,
            PipelineStepId::GenericChannelOptimization,
            PipelineStepStatus::Started,
            &RoomOptimizationProgress {
                current_speaker: String::new(),
                speaker_index: 0,
                total_speakers: sys.speakers.len(),
                iteration: 0,
                max_iterations: 0,
                loss: 0.0,
                overall_progress: 0.0,
                message: Some(format!(
                    "Starting channel optimization for {} workflow",
                    workflow_name
                )),
                epa_preference: None,
                step_id: None,
                step_status: None,
            },
        )?;
    }

    let workflow_max_iterations = optimizer_progress_iterations(config);
    let workflow_cancelled = Arc::new(AtomicBool::new(false));
    let workflow_progress_factory = {
        let observer = Arc::clone(observer_shared);
        let workflow_cancelled = Arc::clone(&workflow_cancelled);
        move |channel_name: &str,
              speaker_idx: usize,
              total_speakers: usize,
              _max_iterations: usize|
              -> Option<crate::topology::WorkflowProgressCallback> {
            let observer = Arc::clone(&observer);
            let name = channel_name.to_string();
            let total = total_speakers.max(1);
            let max_iterations = workflow_max_iterations;
            let stopped = Arc::new(AtomicBool::new(false));
            let stopped_for_callback = Arc::clone(&stopped);
            let workflow_cancelled = Arc::clone(&workflow_cancelled);
            let callback: roomeq_engine::OptimProgressCallback =
                Box::new(move |iter: usize, loss: f64, epa: Option<f64>| {
                    if workflow_cancelled.load(Ordering::Relaxed) {
                        stopped_for_callback.store(true, Ordering::Relaxed);
                        return roomeq_engine::CallbackAction::Stop;
                    }
                    let base_progress = speaker_idx as f64 / total as f64;
                    let speaker_progress = if max_iterations > 0 {
                        iter as f64 / max_iterations as f64
                    } else {
                        0.0
                    };
                    let overall =
                        ((base_progress + speaker_progress / total as f64) * 0.90).min(0.90);

                    match send_progress(
                        &observer,
                        PipelineStepId::GenericChannelOptimization,
                        PipelineStepStatus::InProgress,
                        &RoomOptimizationProgress {
                            current_speaker: name.clone(),
                            speaker_index: speaker_idx,
                            total_speakers,
                            iteration: iter,
                            max_iterations,
                            loss,
                            overall_progress: overall,
                            message: None,
                            epa_preference: epa,
                            step_id: None,
                            step_status: None,
                        },
                    ) {
                        Ok(()) => roomeq_engine::CallbackAction::Continue,
                        Err(_) => {
                            stopped_for_callback.store(true, Ordering::Relaxed);
                            workflow_cancelled.store(true, Ordering::Relaxed);
                            roomeq_engine::CallbackAction::Stop
                        }
                    }
                });
            Some(crate::topology::WorkflowProgressCallback { callback, stopped })
        }
    };
    let mut workflow_stage_callback = {
        let observer = Arc::clone(observer_shared);
        move |step_id: PipelineStepId,
              status: PipelineStepStatus,
              message: &str,
              overall_progress: f64|
              -> Result<()> {
            emit_pipeline_event(
                &observer,
                PipelineEvent::new(step_id, status)
                    .with_message(message)
                    .with_overall_progress(overall_progress),
            )
        }
    };

    match route {
        TopologyRoute::Stereo2_0 => {
            crate::topology::optimize_stereo_2_0_with_progress_and_probe_arrivals(
                config,
                sys,
                sample_rate,
                output_dir.unwrap_or(Path::new(".")),
                probe_arrival_overrides,
                Some(&workflow_progress_factory),
                Some(&mut workflow_stage_callback),
                frequency_samples,
            )
        }
        TopologyRoute::Stereo2_1 => {
            crate::topology::optimize_stereo_2_1_with_progress_and_probe_arrivals(
                config,
                sys,
                sample_rate,
                output_dir.unwrap_or(Path::new(".")),
                probe_arrival_overrides,
                Some(&workflow_progress_factory),
                Some(&mut workflow_stage_callback),
                frequency_samples,
            )
        }
        TopologyRoute::HomeCinema => {
            crate::topology::optimize_home_cinema_with_progress_and_probe_arrivals(
                config,
                sys,
                sample_rate,
                output_dir.unwrap_or(Path::new(".")),
                probe_arrival_overrides,
                Some(&workflow_progress_factory),
                Some(&mut workflow_stage_callback),
                frequency_samples,
            )
        }
        TopologyRoute::Generic => Err(AutoeqError::InvalidConfiguration {
            message: "execute_topology_workflow called with generic route".to_string(),
        }),
    }
}

fn execute_generic_channels_with_frequency_samples(
    config: &RoomConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
    probe_arrival_overrides: Option<&HashMap<String, f64>>,
    observer_shared: &SharedPipelineObserver,
    frequency_samples: usize,
) -> Result<(GenericChannelCollection, usize)> {
    let channels_to_process = channels_for_generic_optimization(config);
    let total_speakers = channels_to_process.len();
    info!("Processing {} channels", total_speakers);

    let shared_mean_spl = compute_shared_mean_spl_with_frequency_samples(
        config,
        &channels_to_process,
        frequency_samples,
    );
    let results = process_generic_channels(
        channels_to_process,
        config,
        sample_rate,
        output_dir,
        shared_mean_spl,
        probe_arrival_overrides,
        observer_shared,
        frequency_samples,
    )?;

    let collection = collect_generic_channel_results(
        results,
        config,
        sample_rate,
        output_dir,
        total_speakers,
        observer_shared,
    )?;

    Ok((collection, total_speakers))
}

fn optimize_room_impl_with_frequency_samples(
    config: &RoomConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
    probe_arrival_overrides: Option<&HashMap<String, f64>>,
    observer: Option<Box<dyn PipelineObserver>>,
    store: &dyn autoeq_artifacts::ArtifactStore,
    frequency_samples: usize,
) -> Result<RoomOptimizationResult> {
    let (observer_shared, config) =
        prepare_room_optimization_with_frequency_samples(config, observer, frequency_samples)?;
    validate_room_optimization_with_frequency_samples(
        &config,
        &observer_shared,
        frequency_samples,
    )?;
    let route = select_topology_route(&config, &observer_shared)?;
    let config = &config;

    let mut result = match route {
        TopologyRoute::Stereo2_0 | TopologyRoute::Stereo2_1 | TopologyRoute::HomeCinema => {
            let sys = config
                .system
                .as_ref()
                .ok_or_else(|| AutoeqError::InvalidConfiguration {
                    message: format!("topology route {route:?} requires system configuration"),
                })?;
            let workflow_result = execute_topology_workflow_with_probe_arrivals(
                config,
                sys,
                sample_rate,
                output_dir,
                probe_arrival_overrides,
                &observer_shared,
                route,
                frequency_samples,
            )?;
            assemble_workflow_result_with_frequency_samples(
                workflow_result,
                config,
                sys,
                sample_rate,
                output_dir,
                probe_arrival_overrides,
                &observer_shared,
                store,
                frequency_samples,
            )?
        }
        TopologyRoute::Generic => {
            emit_pipeline_event(
                &observer_shared,
                PipelineEvent::completed(
                    PipelineStepId::TopologyRouteSelection,
                    "Selected generic channel optimization",
                ),
            )?;
            let (generic, total_speakers) = execute_generic_channels_with_frequency_samples(
                config,
                sample_rate,
                output_dir,
                probe_arrival_overrides,
                &observer_shared,
                frequency_samples,
            )?;
            assemble_generic_result_with_frequency_samples(
                generic,
                total_speakers,
                config,
                sample_rate,
                output_dir,
                &observer_shared,
                store,
                frequency_samples,
            )?
        }
    };

    if let Some(reports) = roomeq_engine::output::take_mixed_phase_reports(&mut result.channels) {
        result.metadata.mixed_phase_per_channel = Some(reports);
    }

    emit_pipeline_event(
        &observer_shared,
        PipelineEvent::started(
            PipelineStepId::SanityCheck,
            "Checking final optimization result",
        )
        .with_overall_progress(1.0),
    )?;
    sanity_check_result(&result)?;
    emit_pipeline_event(
        &observer_shared,
        PipelineEvent::completed(
            PipelineStepId::SanityCheck,
            "Final optimization result checked",
        )
        .with_overall_progress(1.0),
    )?;
    Ok(result)
}

fn shared_alignment_fit_band(
    config: &RoomConfig,
    corrected_curves: &HashMap<String, Curve>,
) -> (f64, f64) {
    // Height/timbre fitting compares full-range channels. Including the routed
    // subwoofer band makes the intersection collapse at the crossover, which
    // triggers the full-band fallback and reintroduces intentional rolloff.
    let (min_freq, max_freq) = corrected_curves
        .keys()
        .filter(|channel_name| !is_subwoofer_channel(config, channel_name))
        .fold(
            (config.optimizer.min_freq, config.optimizer.max_freq),
            |(shared_min, shared_max), channel_name| {
                let (channel_min, channel_max) =
                    final_score_band_for_channel(config, channel_name, None);
                (shared_min.max(channel_min), shared_max.min(channel_max))
            },
        );

    if max_freq > min_freq {
        (min_freq, max_freq)
    } else {
        (config.optimizer.min_freq, config.optimizer.max_freq)
    }
}

fn reported_curve_with_user_preferences(
    base_curve: &Curve,
    chain: &ChannelDspChain,
    sample_rate: f64,
) -> Curve {
    let preference_plugins = chain
        .plugins
        .iter()
        .filter(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("user_preference")
        })
        .cloned()
        .collect::<Vec<_>>();
    if preference_plugins.is_empty() {
        return base_curve.clone();
    }

    let mut preference_chain = chain.clone();
    preference_chain.plugins = preference_plugins;
    let mut no_convolution = roomeq_engine::dsp_realization::NoConvolutionIr;
    match roomeq_engine::dsp_realization::RealizedDsp::new(
        &preference_chain,
        sample_rate,
        &mut no_convolution,
    )
    .and_then(|mut realized| realized.apply_to_curve(base_curve))
    {
        Ok(curve) => curve,
        Err(error) => {
            warn!(
                "Could not include user-preference filters in reported curve for '{}': {}",
                chain.channel, error
            );
            base_curve.clone()
        }
    }
}

fn routed_target_underfill_db(curve: &Curve, chain: &ChannelDspChain) -> Option<f64> {
    let crossover_hz = chain
        .plugins
        .iter()
        .find(|plugin| plugin.plugin_type == "crossover")?
        .parameters
        .get("frequency")?
        .as_f64()?;
    let target = chain.target_curve.clone().map(Curve::from)?;
    roomeq_engine::topology::bass_management_max_underfill_db_with_target(
        Some(curve),
        Some(&target),
        crossover_hz,
    )
}

fn routed_correction_candidate_underfill_db(
    base_curve: &Curve,
    chain: &ChannelDspChain,
    plugins: &[roomeq_model::PluginConfigWrapper],
    sample_rate: f64,
) -> Option<f64> {
    let mut candidate_chain = chain.clone();
    candidate_chain.plugins = plugins.to_vec();
    let mut no_convolution = roomeq_engine::dsp_realization::NoConvolutionIr;
    let candidate_curve = roomeq_engine::dsp_realization::RealizedDsp::new(
        &candidate_chain,
        sample_rate,
        &mut no_convolution,
    )
    .and_then(|mut realized| realized.apply_to_curve(base_curve))
    .ok()?;

    routed_target_underfill_db(&candidate_curve, chain)
}

fn apply_inter_channel_timbre_matching_stage(
    config: &RoomConfig,
    sample_rate: f64,
    deployed_source_curves: &HashMap<String, Curve>,
    channel_results: &mut HashMap<String, ChannelOptimizationResult>,
    channel_chains: &mut HashMap<String, ChannelDspChain>,
) -> Option<StageOutcome> {
    let timbre_config = config
        .optimizer
        .inter_channel_timbre_matching
        .as_ref()
        .filter(|config| config.enabled)?;
    let corrected_curves: HashMap<String, Curve> = channel_results
        .iter()
        .filter(|(name, _)| !is_subwoofer_channel(config, name))
        .map(|(name, result)| {
            (
                name.clone(),
                deployed_source_curves
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| result.final_curve.clone()),
            )
        })
        .collect();
    let (fit_min_freq, fit_max_freq) = shared_alignment_fit_band(config, &corrected_curves);

    Some(
        match roomeq_engine::inter_channel_timbre_matching::compute_inter_channel_timbre_matching_with_threshold(
            &corrected_curves,
            &timbre_config.reference_channel,
            sample_rate,
            fit_min_freq,
            fit_max_freq,
            timbre_config.min_improvement_db,
        ) {
        Ok(timbre_results) => {
                let applied_count = timbre_results
                    .values()
                    .filter(|result| {
                        result.status
                            == roomeq_engine::inter_channel_timbre_matching::TimbreMatchingChannelStatus::Applied
                    })
                    .count();
                let failed_count = timbre_results
                    .values()
                    .filter(|result| {
                        result.status
                            == roomeq_engine::inter_channel_timbre_matching::TimbreMatchingChannelStatus::Failed
                    })
                    .count();

            let mut routed_rejections = Vec::new();
            for (channel_name, timbre_result) in &timbre_results {
                let plugins =
                    roomeq_engine::inter_channel_timbre_matching::create_timbre_matching_plugins(
                        timbre_result,
                        sample_rate,
                    );
                let plugins = stage_logical_input_plugins(config, plugins);
                if let (Some(base_curve), Some(chain)) = (
                    deployed_source_curves.get(channel_name),
                    channel_chains.get(channel_name),
                ) && let Some(baseline_underfill_db) =
                    routed_target_underfill_db(base_curve, chain)
                    && let Some(underfill_db) = routed_correction_candidate_underfill_db(
                    base_curve,
                    chain,
                    &plugins,
                    sample_rate,
                ) && underfill_db > baseline_underfill_db + 1.0e-6
                {
                    routed_rejections.push(format!(
                        "{channel_name}: rejected timbre correction because routed target underfill would regress from {baseline_underfill_db:.3} to {underfill_db:.3} dB"
                    ));
                    continue;
                }
                if !plugins.is_empty()
                    && let Some(chain) = channel_chains.get_mut(channel_name)
                {
                        chain.plugins.extend(plugins);
                    }
                    if let Some(alignment) = &timbre_result.alignment {
                        let shelf_filters =
                            roomeq_engine::spectral_align::create_alignment_filters(alignment, sample_rate);
                        sync_reported_biquad_adjustment(
                            channel_name,
                            channel_results,
                            channel_chains,
                            &shelf_filters,
                            sample_rate,
                        );
                        if alignment.flat_gain_db.abs()
                            >= roomeq_engine::spectral_align::MIN_CORRECTION_DB
                        {
                            sync_reported_gain_adjustment(
                                channel_name,
                                channel_results,
                                channel_chains,
                alignment.flat_gain_db,
                false,
                sample_rate,
            );
                        }
                    }
                }

            let status = if !routed_rejections.is_empty() {
                StageStatus::Degraded
            } else if failed_count > 0 && applied_count > 0 {
                    StageStatus::Degraded
                } else if failed_count > 0 {
                    StageStatus::Failed
                } else if applied_count > 0 {
                    StageStatus::Applied
                } else {
                    StageStatus::Skipped
                };
            let mut advisories = timbre_results
                    .values()
                    .flat_map(|result| result.advisories.iter().cloned())
                    .filter(|advisory| advisory != "reference_channel")
                .collect::<Vec<_>>();
            advisories.extend(routed_rejections);
                advisories.sort();
                advisories.dedup();
                StageOutcome {
                    stage: "inter_channel_timbre_matching".to_string(),
                    status,
                    advisories,
                }
            }
            Err(error) => {
                warn!("Inter-channel timbre matching failed: {error}");
                StageOutcome {
                    stage: "inter_channel_timbre_matching".to_string(),
                    status: StageStatus::Failed,
                    advisories: vec![format!("invalid_reference: {error}")],
                }
            }
        },
    )
}

fn uses_routed_home_cinema_inputs(config: &RoomConfig) -> bool {
    config.system.as_ref().is_some_and(|system| {
        system.model == SystemModel::HomeCinema && system.subwoofers.is_some()
    })
}

fn stage_logical_input_plugins(
    config: &RoomConfig,
    plugins: Vec<roomeq_model::PluginConfigWrapper>,
) -> Vec<roomeq_model::PluginConfigWrapper> {
    if uses_routed_home_cinema_inputs(config) {
        plugins
            .into_iter()
            .map(|plugin| roomeq_engine::topology::mark_plugin_stage(plugin, "pre_route"))
            .collect()
    } else {
        plugins
    }
}

fn create_time_alignment_plugin(
    config: &RoomConfig,
    delay_ms: f64,
) -> roomeq_model::PluginConfigWrapper {
    let plugin = output::create_delay_plugin(delay_ms);
    if uses_routed_home_cinema_inputs(config) {
        // Arrival alignment is inserted at the beginning of the logical input
        // chain. Both its high-passed self route and redirected-bass route must
        // see the same delay or their already-optimized crossover phase is
        // destroyed during final workflow assembly.
        roomeq_engine::topology::mark_plugin_stage(plugin, "pre_route")
    } else {
        plugin
    }
}

#[allow(clippy::too_many_arguments)]
fn assemble_workflow_result_with_frequency_samples(
    mut result: RoomOptimizationResult,
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
    probe_arrival_overrides: Option<&HashMap<String, f64>>,
    observer_shared: &SharedPipelineObserver,
    store: &dyn autoeq_artifacts::ArtifactStore,
    frequency_samples: usize,
) -> Result<RoomOptimizationResult> {
    let workflow_name = match sys.model {
        SystemModel::Stereo => {
            if sys.subwoofers.is_some() {
                "Stereo 2.1"
            } else {
                "Stereo 2.0"
            }
        }
        SystemModel::HomeCinema => "Home Cinema",
        SystemModel::Custom => "Custom",
    };
    let mut workflow_refresh_needed = false;
    let channel_arrivals = phase_arrivals_for_channels_with_frequency_samples(
        config,
        &result.channel_results,
        probe_arrival_overrides,
        frequency_samples,
    );

    // Probe arrivals are explicit measurement metadata and must respect the
    // normal delay policy. Phase-derived arrivals are the topology equivalent
    // of generic-route auto IR sync and may insert alignment delays.
    let phase_ir_sync = probe_arrival_overrides.is_none();
    if (config.optimizer.allow_delay() || phase_ir_sync) && channel_arrivals.len() > 1 {
        let alignment_delays =
            roomeq_engine::analysis::time_align::calculate_alignment_delays(&channel_arrivals);
        for (channel_name, delay_ms) in &alignment_delays {
            let applied = if *delay_ms > 0.01
                && let Some(chain) = result.channels.get_mut(channel_name)
            {
                chain
                    .plugins
                    .insert(0, create_time_alignment_plugin(config, *delay_ms));
                true
            } else {
                false
            };
            if applied {
                sync_reported_phase_adjustment(
                    channel_name,
                    &mut result.channel_results,
                    &mut result.channels,
                    *delay_ms,
                    false,
                    sample_rate,
                );
                workflow_refresh_needed = true;
            }
        }
    }

    // Topology executors own crossover/routing, but sub-main phase alignment
    // remains a final-channel operation. Run the same optimizer used by the
    // generic route after their chains have been assembled.
    if !should_run_standalone_phase_correction(config)
        && apply_topology_phase_alignment(
            config,
            &mut result.channel_results,
            &mut result.channels,
            observer_shared,
            sample_rate,
        )?
    {
        workflow_refresh_needed = true;
    }

    // Send post-workflow summary
    let summary: Vec<String> = result
        .channel_results
        .iter()
        .map(|(name, ch)| format!("  {}: {:.4} -> {:.4}", name, ch.pre_score, ch.post_score))
        .collect();
    send_progress(
        observer_shared,
        PipelineStepId::TopologyWorkflowExecution,
        PipelineStepStatus::Completed,
        &RoomOptimizationProgress {
            current_speaker: String::new(),
            speaker_index: result.channel_results.len(),
            total_speakers: result.channel_results.len(),
            iteration: 0,
            max_iterations: 0,
            loss: result.combined_post_score,
            overall_progress: 0.90,
            message: Some(format!(
                "{} workflow complete:\n{}",
                workflow_name,
                summary.join("\n")
            )),
            epa_preference: None,
            step_id: None,
            step_status: None,
        },
    )?;
    // Workflows only do IIR. If FIR/Hybrid mode is requested, post-generate
    // full FIR coefficients for each channel.
    if matches!(
        config.optimizer.processing_mode,
        ProcessingMode::PhaseLinear | ProcessingMode::Hybrid
    ) {
        send_progress(
            observer_shared,
            PipelineStepId::FirGeneration,
            PipelineStepStatus::Started,
            &RoomOptimizationProgress {
                current_speaker: "FIR generation".to_string(),
                speaker_index: 0,
                total_speakers: result.channel_results.len(),
                iteration: 0,
                max_iterations: 0,
                loss: 0.0,
                overall_progress: 0.95,
                message: Some("Generating FIR coefficients...".to_string()),
                epa_preference: None,
                step_id: None,
                step_status: None,
            },
        )?;
        let out_dir = output_dir.unwrap_or(Path::new("."));
        let names: Vec<String> = result.channel_results.keys().cloned().collect();
        for name in names {
            let chain_has_correction = result.channels.get(&name).is_some_and(|chain| {
                chain
                    .plugins
                    .iter()
                    .any(|plugin| matches!(plugin.plugin_type.as_str(), "eq" | "convolution"))
            });
            let chain_has_fir = result.channels.get(&name).is_some_and(|chain| {
                chain
                    .plugins
                    .iter()
                    .any(|plugin| plugin.plugin_type == "convolution")
            });
            let generated = if let Some(ch) = result.channel_results.get_mut(&name) {
                if !should_post_generate_fir(
                    config.optimizer.processing_mode.clone(),
                    ch.fir_coeffs.is_some(),
                    chain_has_correction,
                    chain_has_fir,
                ) {
                    None
                } else {
                    let generated = post_generate_fir(
                        &name,
                        &ch.initial_curve,
                        &ch.final_curve,
                        &config.optimizer,
                        config.target_curve.as_ref(),
                        sample_rate,
                        Some(out_dir),
                        result.channels.get(&name),
                    );
                    if let Some(generated) = &generated {
                        ch.fir_coeffs = Some(generated.coeffs.clone());
                    }
                    generated
                }
            } else {
                None
            };

            let Some(generated) = generated else {
                continue;
            };

            if let Some(chain) = result.channels.get_mut(&name) {
                chain
                    .plugins
                    .push(roomeq_engine::output::create_convolution_plugin(
                        &generated.filename,
                    ));
            }
            sync_reported_fir_adjustment(
                &name,
                &mut result.channel_results,
                &mut result.channels,
                &generated.coeffs,
                sample_rate,
            );
            workflow_refresh_needed = true;
        }
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::completed(PipelineStepId::FirGeneration, "FIR coefficients generated")
                .with_overall_progress(0.95),
        )?;
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(PipelineStepId::FirGeneration, "FIR generation not needed")
                .with_overall_progress(0.95),
        )?;
    }
    // MixedPhase: post-generate short excess-phase FIR for each channel
    // and add convolution plugin to the DSP chain.
    if config.optimizer.processing_mode == ProcessingMode::MixedPhase {
        send_progress(
            observer_shared,
            PipelineStepId::MixedPhaseFirGeneration,
            PipelineStepStatus::Started,
            &RoomOptimizationProgress {
                current_speaker: "Mixed-phase FIR".to_string(),
                speaker_index: 0,
                total_speakers: result.channel_results.len(),
                iteration: 0,
                max_iterations: 0,
                loss: 0.0,
                overall_progress: 0.95,
                message: Some("Generating mixed-phase FIR...".to_string()),
                epa_preference: None,
                step_id: None,
                step_status: None,
            },
        )?;
        let out_dir = output_dir.unwrap_or(Path::new("."));
        let names: Vec<String> = result.channel_results.keys().cloned().collect();
        for name in names {
            let chain_has_fir = result.channels.get(&name).is_some_and(|chain| {
                chain
                    .plugins
                    .iter()
                    .any(|plugin| plugin.plugin_type == "convolution")
            });
            let generated = if let Some(ch) = result.channel_results.get_mut(&name) {
                if ch.fir_coeffs.is_some() || chain_has_fir {
                    None
                } else {
                    let generated = match post_generate_mixed_phase_fir(
                        &name,
                        &ch.initial_curve,
                        &config.optimizer,
                        sample_rate,
                        Some(out_dir),
                    ) {
                        Ok(generated) => generated,
                        Err(error) => {
                            warn!(
                                "Mixed-phase FIR candidate rejected for '{}': {}",
                                name, error
                            );
                            None
                        }
                    };
                    if let Some(generated) = &generated {
                        ch.fir_coeffs = Some(generated.coeffs.clone());
                    }
                    generated
                }
            } else {
                None
            };

            let Some(generated) = generated else {
                continue;
            };

            if let Some(chain) = result.channels.get_mut(&name) {
                chain.plugins.push(
                    if let Some(report) = generated.mixed_phase_report.as_ref() {
                        roomeq_engine::output::create_mixed_phase_convolution_plugin(
                            &generated.filename,
                            report,
                        )
                    } else {
                        roomeq_engine::output::create_convolution_plugin(&generated.filename)
                    },
                );
            }
            sync_reported_fir_adjustment(
                &name,
                &mut result.channel_results,
                &mut result.channels,
                &generated.coeffs,
                sample_rate,
            );
            workflow_refresh_needed = true;
        }
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::completed(
                PipelineStepId::MixedPhaseFirGeneration,
                "Mixed-phase FIR generated",
            )
            .with_overall_progress(0.955),
        )?;
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::MixedPhaseFirGeneration,
                "Mixed-phase FIR not needed",
            )
            .with_overall_progress(0.955),
        )?;
    }
    // Standalone phase correction (rePhase-style)
    if should_run_standalone_phase_correction(config) {
        send_progress(
            observer_shared,
            PipelineStepId::PhaseCorrection,
            PipelineStepStatus::Started,
            &RoomOptimizationProgress {
                current_speaker: "Phase correction".to_string(),
                speaker_index: 0,
                total_speakers: result.channel_results.len(),
                iteration: 0,
                max_iterations: 0,
                loss: 0.0,
                overall_progress: 0.96,
                message: Some("Phase correction...".to_string()),
                epa_preference: None,
                step_id: None,
                step_status: None,
            },
        )?;
    }
    if should_run_standalone_phase_correction(config)
        && let Some(ref pc_config) = config.optimizer.phase_correction
    {
        let out_dir = output_dir.unwrap_or(Path::new("."));
        let names: Vec<String> = result.channel_results.keys().cloned().collect();
        for name in &names {
            if let Some(ch) = result.channel_results.get_mut(name)
                && let Some(chain) = result.channels.get_mut(name)
            {
                let before_plugins = chain.plugins.len();
                apply_phase_correction(name, ch, chain, pc_config, sample_rate, Some(out_dir));
                workflow_refresh_needed |= chain.plugins.len() != before_plugins;
            }
        }
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::completed(PipelineStepId::PhaseCorrection, "Phase correction complete")
                .with_overall_progress(0.96),
        )?;
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::PhaseCorrection,
                "Phase correction not enabled",
            )
            .with_overall_progress(0.96),
        )?;
    }

    if should_run_standalone_phase_correction(config)
        && apply_topology_phase_alignment(
            config,
            &mut result.channel_results,
            &mut result.channels,
            observer_shared,
            sample_rate,
        )?
    {
        workflow_refresh_needed = true;
    }

    if config
        .optimizer
        .inter_channel_timbre_matching
        .as_ref()
        .is_some_and(|config| config.enabled)
    {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::started(
                PipelineStepId::InterChannelTimbreMatching,
                "Running inter-channel timbre matching",
            )
            .with_overall_progress(0.962),
        )?;
        if let Some(stage_outcome) = apply_inter_channel_timbre_matching_stage(
            config,
            sample_rate,
            &result.deployed_source_curves,
            &mut result.channel_results,
            &mut result.channels,
        ) {
            workflow_refresh_needed |= matches!(
                stage_outcome.status,
                StageStatus::Applied | StageStatus::Degraded
            );
            let event = match stage_outcome.status {
                StageStatus::Applied | StageStatus::Degraded => PipelineEvent::completed(
                    PipelineStepId::InterChannelTimbreMatching,
                    format!("Inter-channel timbre matching: {:?}", stage_outcome.status),
                ),
                StageStatus::Skipped | StageStatus::Failed => PipelineEvent::skipped(
                    PipelineStepId::InterChannelTimbreMatching,
                    format!(
                        "Inter-channel timbre matching: {:?} ({})",
                        stage_outcome.status,
                        stage_outcome.advisories.join(", ")
                    ),
                ),
            };
            emit_pipeline_event(observer_shared, event.with_overall_progress(0.962))?;
            result.metadata.stage_outcomes.push(stage_outcome);
        }
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::InterChannelTimbreMatching,
                "Inter-channel timbre matching not enabled",
            )
            .with_overall_progress(0.962),
        )?;
    }

    // Topology workflows return a pre-assembled result and therefore do not
    // pass through `assemble_generic_result`, where height alignment
    // historically lived. Apply the same feature before GD optimization so
    // home-cinema configurations do not silently ignore it.
    if let Some(height_config) = config
        .optimizer
        .height_channel_alignment
        .as_ref()
        .filter(|height| height.enabled)
    {
        let stage_outcome = apply_topology_height_alignment_with_frequency_samples(
            &mut result,
            height_config,
            config,
            probe_arrival_overrides,
            sample_rate,
            frequency_samples,
        );
        let event = match stage_outcome.status {
            StageStatus::Applied | StageStatus::Degraded => PipelineEvent::completed(
                PipelineStepId::HeightChannelAlignment,
                format!("Height-channel alignment: {:?}", stage_outcome.status),
            ),
            StageStatus::Skipped | StageStatus::Failed => PipelineEvent::skipped(
                PipelineStepId::HeightChannelAlignment,
                format!(
                    "Height-channel alignment: {:?} ({})",
                    stage_outcome.status,
                    stage_outcome.advisories.join(", ")
                ),
            ),
        };
        emit_pipeline_event(observer_shared, event.with_overall_progress(0.964))?;
        result.metadata.stage_outcomes.push(stage_outcome);
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::HeightChannelAlignment,
                "Height-channel alignment not enabled",
            )
            .with_overall_progress(0.964),
        )?;
    }

    result.metadata.timing_diagnostics =
        build_timing_diagnostics(config, &channel_arrivals, &result.channels);
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::started(
            PipelineStepId::GroupDelayOptimization,
            "Running GD optimization",
        )
        .with_overall_progress(0.965),
    )?;
    // Heartbeat: GD optimization can take several seconds on
    // wide configurations. Without this InProgress event the
    // UI stays on "Running GD optimization" Started state for
    // the whole duration which reads as a hang.
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::new(
            PipelineStepId::GroupDelayOptimization,
            PipelineStepStatus::InProgress,
        )
        .with_message(
            if config.optimizer.processing_mode == ProcessingMode::PhaseLinear {
                "Phase-linear FIR group-delay optimization..."
            } else {
                "Group-delay optimization..."
            },
        )
        .with_overall_progress(0.965),
    )?;
    let workflow_group_delay_summary =
        if config.optimizer.processing_mode == ProcessingMode::PhaseLinear {
            try_run_phase_linear_fir_gd(
                config,
                &mut result.channel_results,
                &mut result.channels,
                sample_rate,
                output_dir,
            )
        } else {
            try_run_gd_opt_with_frequency_samples(
                config,
                &mut result.channel_results,
                &mut result.channels,
                sample_rate,
                frequency_samples,
            )
        };
    if let Some(summary) = workflow_group_delay_summary {
        workflow_refresh_needed |= summary.applied;
        result.metadata.group_delay = Some(summary);
    }
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::completed(
            PipelineStepId::GroupDelayOptimization,
            "GD optimization complete",
        )
        .with_overall_progress(0.965),
    )?;

    // Compute IR waveforms for the workflow result
    send_progress(
        observer_shared,
        PipelineStepId::ImpulseResponseComputation,
        PipelineStepStatus::Started,
        &RoomOptimizationProgress {
            current_speaker: "IR computation".to_string(),
            speaker_index: 0,
            total_speakers: result.channel_results.len(),
            iteration: 0,
            max_iterations: 0,
            loss: 0.0,
            overall_progress: 0.97,
            message: Some("Computing impulse responses...".to_string()),
            epa_preference: None,
            step_id: None,
            step_status: None,
        },
    )?;
    let ir_total = result.channel_results.len();
    let ir_names: Vec<String> = result.channel_results.keys().cloned().collect();
    for (ir_index, channel_name) in ir_names.iter().enumerate() {
        // Per-channel heartbeat: long IR convolutions can run
        // for hundreds of milliseconds each, so users see one
        // chip moving instead of a single "Started" stuck for
        // the whole loop.
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::new(
                PipelineStepId::ImpulseResponseComputation,
                PipelineStepStatus::InProgress,
            )
            .with_channel(channel_name.clone())
            .with_channels(ir_index, ir_total)
            .with_message(format!("Computing impulse response for {channel_name}"))
            .with_overall_progress(0.97),
        )?;
        let ch_result = match result.channel_results.get(channel_name) {
            Some(ch) => ch,
            None => continue,
        };
        let delay_ms = result
            .channels
            .get(channel_name)
            .map(total_chain_delay_ms)
            .unwrap_or(0.0);
        let initial_curve = ch_result.initial_curve.clone();
        let biquads = ch_result.biquads.clone();
        let fir_coeffs = ch_result.fir_coeffs.clone();
        if let Some((pre_ir, post_ir)) =
            roomeq_engine::analysis::ir_waveform::compute_channel_ir_waveforms(
                &initial_curve,
                &biquads,
                fir_coeffs.as_deref(),
                delay_ms,
                sample_rate,
            )
            && let Some(chain) = result.channels.get_mut(channel_name)
        {
            chain.pre_ir = Some(pre_ir);
            chain.post_ir = Some(post_ir);
        }
    }
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::completed(
            PipelineStepId::ImpulseResponseComputation,
            "Impulse responses computed",
        )
        .with_overall_progress(0.97),
    )?;

    // Compute inter-channel deviation and optionally correct it
    if result.channel_results.len() > 1 {
        send_progress(
            observer_shared,
            PipelineStepId::ChannelMatching,
            PipelineStepStatus::Started,
            &RoomOptimizationProgress {
                current_speaker: "Channel matching".to_string(),
                speaker_index: 0,
                total_speakers: result.channel_results.len(),
                iteration: 0,
                max_iterations: 0,
                loss: 0.0,
                overall_progress: 0.98,
                message: Some("Channel matching analysis...".to_string()),
                epa_preference: None,
                step_id: None,
                step_status: None,
            },
        )?;
        let plugin_count_before_icd: usize = result
            .channels
            .values()
            .map(|chain| chain.plugins.len())
            .sum();
        compute_and_correct_icd(&mut result, config, sample_rate);
        let plugin_count_after_icd: usize = result
            .channels
            .values()
            .map(|chain| chain.plugins.len())
            .sum();
        workflow_refresh_needed |= plugin_count_after_icd != plugin_count_before_icd;
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::completed(PipelineStepId::ChannelMatching, "Channel matching complete")
                .with_overall_progress(0.98),
        )?;
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::ChannelMatching,
                "Channel matching not needed",
            )
            .with_overall_progress(0.98),
        )?;
    }
    // The runtime acceptance gate reads FIR temporal-masking evidence; stages
    // that add FIR taps late (e.g. redirected bass) must have that evidence
    // computed before the gate runs, not only in the post-gate refresh.
    let sidecar_dir = output_dir.unwrap_or(Path::new("."));
    refresh_temporal_ir_evidence(&mut result, config, sample_rate, sidecar_dir);
    apply_final_correction_safety_gate_preserving_routed_crossover(
        &mut result,
        sample_rate,
        config.optimizer.smooth_n,
        (config.optimizer.min_freq, config.optimizer.max_freq),
        sidecar_dir,
        config.optimizer.processing_mode.clone(),
        group_delay_budget_ms(config),
    )?;
    record_missing_mixed_phase_fir_reversions(
        &mut result,
        config.optimizer.processing_mode.clone(),
    );

    emit_pipeline_event(
        observer_shared,
        PipelineEvent::started(PipelineStepId::MetadataRefresh, "Refreshing reports")
            .with_overall_progress(0.99),
    )?;
    if workflow_refresh_needed {
        log::debug!("Refreshing reports after workflow DSP mutations");
        refresh_final_reports(&mut result, config, sample_rate, sidecar_dir);
    } else {
        refresh_direct_early_late_reports(&mut result, config);
        refresh_perceptual_policy_reports(&mut result, config);
    }
    update_perceptual_metrics(&mut result.metadata, Some(&result.channels), Some(config));
    apply_ctc_if_enabled(&mut result, config, sample_rate, output_dir)?;
    generate_validation_bundle_report(&mut result, config, output_dir, store)?;
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::completed(PipelineStepId::MetadataRefresh, "Reports refreshed")
            .with_overall_progress(0.99),
    )?;

    Ok(result)
}

/// The induced-group-delay budget an enabled GD-Opt stage is explicitly
/// allowed to spend: its configured `max_delay_ms`. `None` when group-delay
/// optimization is disabled, leaving the runtime acceptance policy's default
/// side-effect limit in force.
fn group_delay_budget_ms(config: &RoomConfig) -> Option<f64> {
    config
        .optimizer
        .group_delay
        .as_ref()
        .filter(|gd| gd.enabled)
        .map(|gd| gd.max_delay_ms)
}

fn retained_fir_coeffs_by_channel(result: &RoomOptimizationResult) -> HashMap<String, Vec<f64>> {
    result
        .channel_results
        .iter()
        .filter_map(|(name, channel)| {
            channel
                .fir_coeffs
                .as_ref()
                .map(|coeffs| (name.clone(), coeffs.clone()))
        })
        .collect()
}

fn commit_or_restore_routed_safety_replay(
    result: &mut RoomOptimizationResult,
    pre_safety_result: RoomOptimizationResult,
    pre_safety_deployed: HashMap<String, Curve>,
    post_safety_deployed: Result<HashMap<String, Curve>>,
) {
    match post_safety_deployed {
        Ok(deployed) => result.deployed_source_curves = deployed,
        Err(error) => {
            *result = pre_safety_result;
            result.deployed_source_curves = pre_safety_deployed;
            result.metadata.stage_outcomes.push(StageOutcome {
                stage: "final_correction_safety_routed_replay".to_string(),
                status: StageStatus::Degraded,
                advisories: vec![format!(
                    "safety reversion rejected because it broke the routed crossover: {error}"
                )],
            });
            warn!(
                "Final correction safety reversion rejected; preserving the pre-gate routed DSP: {error}"
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_final_correction_safety_gate_preserving_routed_crossover(
    result: &mut RoomOptimizationResult,
    sample_rate: f64,
    smoothing_n: usize,
    evaluation_band: (f64, f64),
    sidecar_dir: &Path,
    processing_mode: ProcessingMode,
    group_delay_budget_ms: Option<f64>,
) -> Result<()> {
    let routed_snapshot = if let Some(graph) = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|report| report.routing_graph.clone())
    {
        let fir_coeffs = retained_fir_coeffs_by_channel(result);
        let deployed = crate::topology::reconstruct_deployed_source_curves(
            &result.channels,
            &fir_coeffs,
            &graph,
            sample_rate,
            sidecar_dir,
        )?;
        Some((result.clone(), deployed, graph))
    } else {
        None
    };

    room_optimization_result::apply_final_correction_safety_gate(
        result,
        sample_rate,
        smoothing_n,
        evaluation_band,
        sidecar_dir,
        processing_mode,
        group_delay_budget_ms,
    );

    if let Some((pre_safety_result, pre_safety_deployed, graph)) = routed_snapshot {
        let fir_coeffs = retained_fir_coeffs_by_channel(result);
        let post_safety_deployed = crate::topology::reconstruct_deployed_source_curves(
            &result.channels,
            &fir_coeffs,
            &graph,
            sample_rate,
            sidecar_dir,
        );
        commit_or_restore_routed_safety_replay(
            result,
            pre_safety_result,
            pre_safety_deployed,
            post_safety_deployed,
        );
    }

    Ok(())
}

fn record_missing_mixed_phase_fir_reversions(
    result: &mut RoomOptimizationResult,
    processing_mode: ProcessingMode,
) {
    if processing_mode != ProcessingMode::MixedPhase {
        return;
    }

    let mut reverted_stages = result
        .channel_results
        .iter()
        .filter(|(_, channel)| {
            channel
                .initial_curve
                .phase
                .as_ref()
                .is_some_and(|phase| !phase.is_empty())
                && channel.fir_coeffs.is_none()
        })
        .map(|(name, _)| format!("{name}:fir"))
        .collect::<Vec<_>>();
    reverted_stages.sort();
    reverted_stages.dedup();
    if reverted_stages.is_empty() {
        return;
    }

    if let Some(report) = result.metadata.correction_acceptance.as_mut() {
        report.accepted = false;
        report.decision = roomeq_model::CorrectionDecision::RevertedStage;
        report
            .violations
            .push("mixed_phase_fir_safety_reverted".to_string());
        report
            .reverted_stages
            .extend(reverted_stages.iter().cloned());
        report.reverted_stages.sort();
        report.reverted_stages.dedup();
        report.violations.sort();
        report.violations.dedup();
    }

    result
        .metadata
        .stage_outcomes
        .push(roomeq_model::StageOutcome {
            stage: "mixed_phase_fir_generation".to_string(),
            status: roomeq_model::StageStatus::Degraded,
            advisories: reverted_stages
                .into_iter()
                .map(|stage| format!("{stage}: candidate rejected or unavailable"))
                .collect(),
        });
}

fn phase_arrivals_for_channels_with_frequency_samples(
    config: &RoomConfig,
    channel_results: &HashMap<String, roomeq_engine::room_result::ChannelOptimizationResult>,
    probe_arrival_overrides: Option<&HashMap<String, f64>>,
    frequency_samples: usize,
) -> HashMap<String, f64> {
    let primary_seat = config
        .optimizer
        .multi_seat
        .as_ref()
        .map(|policy| policy.primary_seat)
        .unwrap_or(0);
    channel_results
        .keys()
        .filter_map(|channel_name| {
            if let Some(arrival_ms) =
                probe_arrival_overrides.and_then(|overrides| overrides.get(channel_name).copied())
            {
                return Some((channel_name.clone(), arrival_ms));
            }
            let source = gd::source_for_output_channel(config, channel_name)?;
            let curves = crate::measurement::load_source_individual_with_frequency_samples(
                source,
                frequency_samples,
            )
            .ok()?;
            let curve = curves.get(primary_seat)?;
            let (phase_min, phase_max) =
                roomeq_engine::analysis::time_align::phase_arrival_regression_band(
                    curve, 200.0, 2_000.0,
                )?;
            let arrival =
                roomeq_engine::analysis::time_align::estimate_arrival_from_phase_detailed(
                    curve, phase_min, phase_max,
                )
                .ok()?;
            Some((channel_name.clone(), arrival))
        })
        .collect()
}

fn routed_topology_owns_phase_alignment(channel_chains: &HashMap<String, ChannelDspChain>) -> bool {
    channel_chains.values().any(|chain| {
        chain.plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("room_eq_stage")
                .and_then(|value| value.as_str())
                == Some("route_owned")
        })
    })
}

fn apply_topology_phase_alignment(
    config: &RoomConfig,
    channel_results: &mut HashMap<String, roomeq_engine::room_result::ChannelOptimizationResult>,
    channel_chains: &mut HashMap<String, ChannelDspChain>,
    observer_shared: &SharedPipelineObserver,
    sample_rate: f64,
) -> Result<bool> {
    let Some(phase_config) = config
        .optimizer
        .phase_alignment
        .as_ref()
        .filter(|phase| phase.enabled && config.optimizer.allow_delay())
    else {
        return Ok(false);
    };

    // Routed topologies already optimize main/sub relative delay and polarity
    // per logical source. Adding a physical-output delay here would shift every
    // sub route after that optimization and recreate a crossover cancellation.
    if routed_topology_owns_phase_alignment(channel_chains) {
        debug!("Skipping standalone phase alignment for route-owned topology");
        return Ok(false);
    }

    let curves = collect_current_final_curves(channel_results);
    let pairings = find_sub_main_pairings(config, &curves);
    if pairings.is_empty() {
        return Ok(false);
    }
    send_progress(
        observer_shared,
        PipelineStepId::PhaseAlignment,
        PipelineStepStatus::Started,
        &RoomOptimizationProgress {
            current_speaker: String::new(),
            speaker_index: 0,
            total_speakers: pairings.len(),
            iteration: 0,
            max_iterations: 0,
            loss: 0.0,
            overall_progress: 0.0,
            message: Some("Running topology phase alignment...".to_string()),
            epa_preference: None,
            step_id: None,
            step_status: None,
        },
    )?;
    let mut results = HashMap::new();
    for (sub_name, main_name) in pairings {
        let Some(sub_curve) = curves.get(&sub_name) else {
            continue;
        };
        let Some(main_curve) = curves.get(&main_name) else {
            continue;
        };
        if sub_curve.phase.is_none() || main_curve.phase.is_none() {
            continue;
        }
        match phase_alignment::optimize_phase_alignment(sub_curve, main_curve, phase_config) {
            Ok(result) => {
                results.insert(
                    main_name,
                    (result.delay_ms, result.invert_polarity, sub_name),
                );
            }
            Err(error) => warn!("Topology phase alignment failed: {error}"),
        }
    }
    for (main_name, (_, invert, _)) in &results {
        if *invert && let Some(chain) = channel_chains.get_mut(main_name) {
            chain
                .plugins
                .insert(0, output::create_gain_plugin_with_invert(0.0, true));
            sync_reported_phase_adjustment(
                main_name,
                channel_results,
                channel_chains,
                0.0,
                true,
                sample_rate,
            );
        }
    }
    apply_phase_alignment_delay_schedule(&results, channel_results, channel_chains, sample_rate);
    if !results.is_empty() {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::completed(
                PipelineStepId::PhaseAlignment,
                "Topology phase alignment complete",
            ),
        )?;
    }
    Ok(!results.is_empty())
}

fn insert_topology_height_residual_delay(
    chain: &mut ChannelDspChain,
    delay_ms: f64,
    allow_delay: bool,
    routed_inputs: bool,
) -> bool {
    if !allow_delay || delay_ms <= 0.01 {
        return false;
    }
    let plugin = output::create_delay_plugin(delay_ms);
    let plugin = if routed_inputs {
        roomeq_engine::topology::mark_plugin_stage(plugin, "pre_route")
    } else {
        plugin
    };
    chain.plugins.insert(0, plugin);
    true
}

fn apply_topology_height_alignment_with_frequency_samples(
    result: &mut RoomOptimizationResult,
    height_config: &roomeq_model::HeightChannelAlignmentConfig,
    config: &RoomConfig,
    probe_arrival_overrides: Option<&HashMap<String, f64>>,
    sample_rate: f64,
    frequency_samples: usize,
) -> StageOutcome {
    let mut channel_arrivals = phase_arrivals_for_channels_with_frequency_samples(
        config,
        &result.channel_results,
        probe_arrival_overrides,
        frequency_samples,
    );
    for (channel_name, arrival_ms) in &mut channel_arrivals {
        if let Some(chain) = result.channels.get(channel_name) {
            *arrival_ms += total_chain_delay_ms(chain);
        }
    }
    let corrected_curves = collect_current_final_curves(&result.channel_results);
    let (fit_min_freq, fit_max_freq) = shared_alignment_fit_band(config, &corrected_curves);
    let mut height_results =
        match roomeq_engine::height_channel_alignment::compute_height_channel_alignment_with_coherence_threshold(
            &corrected_curves,
            &channel_arrivals,
            height_config,
            sample_rate,
            fit_min_freq,
            fit_max_freq,
            config
                .recording_config
                .as_ref()
                .and_then(|recording| recording.coherence_threshold)
                .map(f64::from)
                .unwrap_or(roomeq_engine::bass_phase_confidence::DEFAULT_COHERENCE_THRESHOLD),
        ) {
            Ok(results) => results,
            Err(error) => {
                return StageOutcome {
                    stage: "height_channel_alignment".to_string(),
                    status: StageStatus::Failed,
                    advisories: vec![format!("height_alignment_failed: {error}")],
                };
            }
        };

    let failed_count = height_results
        .values()
        .filter(|height| {
            height.status == roomeq_engine::height_channel_alignment::HeightAlignmentStatus::Failed
        })
        .count();
    let mut applied_count = 0;
    for (channel_name, height_result) in &mut height_results {
        let mut applied = false;
        if let Some(alignment) = &height_result.alignment {
            let (eq_plugin, gain_plugin) =
                roomeq_engine::spectral_align::create_alignment_plugins(alignment, sample_rate);
            let plugins = stage_logical_input_plugins(
                config,
                eq_plugin.into_iter().chain(gain_plugin).collect(),
            );
            if let (Some(base_curve), Some(chain)) = (
                result.deployed_source_curves.get(channel_name),
                result.channels.get(channel_name),
            ) && let Some(baseline_underfill_db) = routed_target_underfill_db(base_curve, chain)
                && let Some(candidate_underfill_db) = routed_correction_candidate_underfill_db(
                    base_curve,
                    chain,
                    &plugins,
                    sample_rate,
                )
                && candidate_underfill_db > baseline_underfill_db + 1.0e-6
            {
                height_result.advisories.push(format!(
                    "{channel_name}: rejected height spectral alignment because routed target underfill would regress from {baseline_underfill_db:.3} to {candidate_underfill_db:.3} dB"
                ));
                continue;
            }
            if let Some(chain) = result.channels.get_mut(channel_name) {
                chain.plugins.extend(plugins);
            }
            let filters =
                roomeq_engine::spectral_align::create_alignment_filters(alignment, sample_rate);
            sync_reported_biquad_adjustment(
                channel_name,
                &mut result.channel_results,
                &mut result.channels,
                &filters,
                sample_rate,
            );
            if alignment.flat_gain_db.abs() >= roomeq_engine::spectral_align::MIN_CORRECTION_DB {
                sync_reported_gain_adjustment(
                    channel_name,
                    &mut result.channel_results,
                    &mut result.channels,
                    alignment.flat_gain_db,
                    false,
                    sample_rate,
                );
            }
            applied = true;
        }
        // `channel_arrivals` already includes every delay currently present
        // in the chain. A positive residual is therefore additional delay,
        // even when an earlier alignment stage inserted another delay plugin.
        let delay_applied = result.channels.get_mut(channel_name).is_some_and(|chain| {
            insert_topology_height_residual_delay(
                chain,
                height_result.delay_ms,
                config.optimizer.allow_delay(),
                uses_routed_home_cinema_inputs(config),
            )
        });
        if delay_applied {
            sync_reported_phase_adjustment(
                channel_name,
                &mut result.channel_results,
                &mut result.channels,
                height_result.delay_ms,
                false,
                sample_rate,
            );
            applied = true;
        }
        applied_count += usize::from(applied);
    }

    let mut advisories = height_results
        .values()
        .flat_map(|height| height.advisories.iter().cloned())
        .collect::<Vec<_>>();
    if height_results.is_empty() {
        advisories.push("no_height_channels".to_string());
    }
    advisories.sort();
    advisories.dedup();
    let degraded = advisories.iter().any(|advisory| {
        advisory.ends_with("_missing")
            || advisory.ends_with("_untrustworthy")
            || advisory == "height_objective_acceptance_failed"
            || advisory == "height_arrives_after_reference"
            || advisory == "height_delay_limit_exceeded"
            || advisory.contains("rejected height spectral alignment")
    });
    let status = if applied_count > 0 && (failed_count > 0 || degraded) {
        StageStatus::Degraded
    } else if failed_count > 0 {
        StageStatus::Failed
    } else if applied_count > 0 {
        StageStatus::Applied
    } else {
        StageStatus::Skipped
    };
    StageOutcome {
        stage: "height_channel_alignment".to_string(),
        status,
        advisories,
    }
}

#[allow(clippy::too_many_arguments)]
fn assemble_generic_result_with_frequency_samples(
    generic: GenericChannelCollection,
    total_speakers: usize,
    config: &RoomConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
    observer_shared: &SharedPipelineObserver,
    store: &dyn autoeq_artifacts::ArtifactStore,
    frequency_samples: usize,
) -> Result<RoomOptimizationResult> {
    let GenericChannelCollection {
        mut channel_chains,
        mut channel_results,
        pre_scores,
        post_scores,
        mut curves,
        channel_means,
        mut channel_arrivals,
    } = generic;
    let mut stage_outcomes = Vec::new();

    // Auto IR sync: if no WAV-based arrivals were collected, estimate from phase data.
    // Runs unconditionally (does not require allow_delay = true).
    let phase_ir_sync = channel_arrivals.is_empty() && channel_results.len() > 1;
    if phase_ir_sync {
        for (channel_name, result) in &channel_results {
            // Multi-measurement aggregation intentionally drops phase because
            // averaging wrapped complex arrival across seats is not physical.
            // Use the configured primary seat for channel timing instead.
            let primary_seat = config
                .optimizer
                .multi_seat
                .as_ref()
                .map(|policy| policy.primary_seat)
                .unwrap_or(0);
            let primary_curve = gd::source_for_output_channel(config, channel_name)
                .and_then(|source| {
                    crate::measurement::load_source_individual_with_frequency_samples(
                        source,
                        frequency_samples,
                    )
                    .ok()
                })
                .and_then(|curves| curves.get(primary_seat).cloned());
            let phase_curve = primary_curve.as_ref().unwrap_or(&result.initial_curve);
            let Some((phase_min, phase_max)) =
                roomeq_engine::analysis::time_align::phase_arrival_regression_band(
                    phase_curve,
                    200.0,
                    2000.0,
                )
            else {
                debug!(
                    "Auto IR sync: channel '{}' has no usable phase-arrival regression band",
                    channel_name
                );
                continue;
            };

            match roomeq_engine::analysis::time_align::estimate_arrival_from_phase_detailed(
                phase_curve,
                phase_min,
                phase_max,
            ) {
                Ok(arrival_ms) => {
                    channel_arrivals.insert(channel_name.clone(), arrival_ms);
                }
                Err(roomeq_engine::analysis::time_align::PhaseArrivalError::MissingPhase)
                | Err(roomeq_engine::analysis::time_align::PhaseArrivalError::InsufficientBandPoints { .. }) => {
                    debug!(
                        "Auto IR sync: channel '{}' lacks usable phase data in {:.1}-{:.1} Hz",
                        channel_name, phase_min, phase_max
                    );
                }
                Err(err) => {
                    warn!(
                        "Auto IR sync: rejected phase-derived arrival for channel '{}' in {:.1}-{:.1} Hz: {:?}",
                        channel_name, phase_min, phase_max, err
                    );
                }
            }
        }
        if channel_arrivals.len() > 1 {
            info!(
                "Auto IR sync: phase-estimated arrival times for {} channels",
                channel_arrivals.len()
            );
            for (name, arrival) in &channel_arrivals {
                info!(
                    "  Channel '{}': phase-estimated arrival = {:.2} ms",
                    name, arrival
                );
            }
        } else {
            // Clear partial arrivals — not enough channels have phase data
            channel_arrivals.clear();
        }
    }

    // Time alignment: add delay plugins to align all channels to the slowest one
    // This is done PRE-EQ by inserting at the beginning of the plugin chain
    if (config.optimizer.allow_delay() || phase_ir_sync) && channel_arrivals.len() > 1 {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::started(PipelineStepId::TimeAlignment, "Aligning channel timing"),
        )?;
        let arrivals: Vec<f64> = channel_arrivals.values().copied().collect();
        let min_arrival = arrivals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_arrival = arrivals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let arrival_spread = max_arrival - min_arrival;

        // Warn if arrival time differences are significant (might indicate measurement issues)
        if arrival_spread > ARRIVAL_TIME_WARNING_THRESHOLD_MS {
            warn!(
                "Channel arrival times differ by {:.1} ms (threshold: {:.1} ms). \
                This may indicate measurement issues or very different speaker distances.",
                arrival_spread, ARRIVAL_TIME_WARNING_THRESHOLD_MS
            );
            for (name, arrival) in &channel_arrivals {
                info!("  Channel '{}': arrival time = {:.2} ms", name, arrival);
            }
        }

        // Calculate alignment delays (slowest channel = reference, others get delays)
        let alignment_delays =
            roomeq_engine::analysis::time_align::calculate_alignment_delays(&channel_arrivals);

        // Add delay plugins at the BEGINNING of the chain (pre-EQ)
        for (channel_name, delay_ms) in &alignment_delays {
            // Only add delay plugin if the adjustment is significant (> 0.01 ms = ~0.5 samples at 48kHz)
            let applied = if *delay_ms > 0.01
                && let Some(chain) = channel_chains.get_mut(channel_name)
            {
                chain
                    .plugins
                    .insert(0, create_time_alignment_plugin(config, *delay_ms));
                true
            } else {
                false
            };

            if applied {
                sync_reported_phase_adjustment(
                    channel_name,
                    &mut channel_results,
                    &mut channel_chains,
                    *delay_ms,
                    false,
                    sample_rate,
                );
                info!(
                    "  Channel '{}': added {:.3} ms delay for time alignment",
                    channel_name, delay_ms
                );
            }
        }
        curves = collect_current_final_curves(&channel_results);
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::completed(PipelineStepId::TimeAlignment, "Channel timing aligned"),
        )?;
    } else if channel_arrivals.is_empty() && config.speakers.len() > 1 {
        info!("No arrival time data (WAV or phase) available for time alignment. Skipping.");
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::TimeAlignment,
                "No arrival time data available for time alignment",
            ),
        )?;
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(PipelineStepId::TimeAlignment, "Time alignment not needed"),
        )?;
    }

    // Spectral channel alignment: fit low-shelf + high-shelf + flat gain to each
    // channel's deviation from the average post-EQ curve. This corrects both broadband
    // level differences and frequency-dependent tilt between channels.
    let spectral_curves: HashMap<String, Curve> = curves
        .iter()
        .filter(|(name, _)| !is_subwoofer_channel(config, name))
        .map(|(name, curve)| (name.clone(), curve.clone()))
        .collect();
    if spectral_curves.len() > 1 {
        send_progress(
            observer_shared,
            PipelineStepId::SpectralAlignment,
            PipelineStepStatus::Started,
            &RoomOptimizationProgress {
                current_speaker: "Spectral alignment".to_string(),
                speaker_index: 0,
                total_speakers,
                iteration: 0,
                max_iterations: 0,
                loss: 0.0,
                overall_progress: 0.92,
                message: Some("Spectral channel alignment...".to_string()),
                epa_preference: None,
                step_id: None,
                step_status: None,
            },
        )?;
        let min_freq = config.optimizer.min_freq;
        let max_freq = config.optimizer.max_freq;
        // Compute post-EQ mean SPL per channel for the level spread warning
        let mut post_eq_means: HashMap<String, f64> = HashMap::new();
        for (channel_name, final_curve) in &spectral_curves {
            let freqs_f32: Vec<f32> = final_curve.freq.iter().map(|&f| f as f32).collect();
            let spl_f32: Vec<f32> = final_curve.spl.iter().map(|&s| s as f32).collect();
            let post_mean = compute_average_response(
                &freqs_f32,
                &spl_f32,
                Some((min_freq as f32, max_freq as f32)),
            ) as f64;
            post_eq_means.insert(channel_name.clone(), post_mean);
        }

        let means: Vec<f64> = post_eq_means.values().copied().collect();
        let min_mean = means.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_mean = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let level_spread = max_mean - min_mean;

        info!(
            "Post-EQ spectral alignment: level spread = {:.2} dB across {} channels",
            level_spread,
            post_eq_means.len()
        );
        for (name, mean) in &post_eq_means {
            info!("  Channel '{}': post-EQ mean SPL = {:.1} dB", name, mean);
        }

        // Warn if level differences are significant (might indicate measurement issues)
        if level_spread > LEVEL_DIFFERENCE_WARNING_THRESHOLD {
            warn!(
                "Channel levels differ by {:.1} dB (threshold: {:.1} dB). \
                This may indicate measurement issues (mic placement, cable problems, etc.).",
                level_spread, LEVEL_DIFFERENCE_WARNING_THRESHOLD
            );
        }

        // Compute spectral alignment (shelf + gain) for each channel
        let alignment_results = roomeq_engine::spectral_align::compute_spectral_alignment(
            &spectral_curves,
            sample_rate,
            min_freq,
            max_freq,
        );
        roomeq_engine::spectral_align::log_spectral_alignment(&alignment_results);

        // Insert alignment plugins after the per-channel PEQ.
        //
        // Shelves and flat-gain are gated independently: a constant SPL shift
        // can never change flatness (the score normalizes by mean), so a
        // legitimate inter-channel level correction must not be discarded just
        // because the shelves would regress flatness.
        for (channel_name, result) in &alignment_results {
            let shelf_filters =
                roomeq_engine::spectral_align::create_alignment_filters(result, sample_rate);

            let (apply_shelves, apply_gain) = if channel_results.contains_key(channel_name) {
                // No topology result yet at this stage: the applied crossover
                // is unknown, so fall back to the configured static value.
                let (score_min, score_max) =
                    final_score_band_for_channel(config, channel_name, None);
                let shelves_ok = should_apply_spectral_shelves(
                    &curves,
                    channel_name,
                    &shelf_filters,
                    sample_rate,
                    score_min,
                    score_max,
                );

                let gain_ok =
                    result.flat_gain_db.abs() >= roomeq_engine::spectral_align::MIN_CORRECTION_DB;
                (shelves_ok, gain_ok)
            } else {
                (false, false)
            };

            if !apply_shelves && !apply_gain {
                continue;
            }

            if let Some(chain) = channel_chains.get_mut(channel_name) {
                let (eq_plugin, gain_plugin) =
                    roomeq_engine::spectral_align::create_alignment_plugins(result, sample_rate);
                if apply_shelves && let Some(eq) = eq_plugin {
                    chain.plugins.push(eq);
                }
                if apply_gain && let Some(gain) = gain_plugin {
                    chain.plugins.push(gain);
                }
            }

            if apply_shelves {
                sync_reported_biquad_adjustment(
                    channel_name,
                    &mut channel_results,
                    &mut channel_chains,
                    &shelf_filters,
                    sample_rate,
                );
            }
            if apply_gain {
                sync_reported_gain_adjustment(
                    channel_name,
                    &mut channel_results,
                    &mut channel_chains,
                    result.flat_gain_db,
                    false,
                    sample_rate,
                );
            }
        }
        curves = collect_current_final_curves(&channel_results);
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::completed(
                PipelineStepId::SpectralAlignment,
                "Spectral channel alignment complete",
            ),
        )?;
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::SpectralAlignment,
                "Spectral channel alignment not needed",
            ),
        )?;
    }

    // ========================================================================
    // Inter-channel timbre matching
    // ========================================================================
    let timbre_config = config.optimizer.inter_channel_timbre_matching.as_ref();
    if let Some(timbre_config) = timbre_config
        && timbre_config.enabled
    {
        send_progress(
            observer_shared,
            PipelineStepId::InterChannelTimbreMatching,
            PipelineStepStatus::Started,
            &RoomOptimizationProgress {
                current_speaker: "Inter-channel timbre matching".to_string(),
                speaker_index: 0,
                total_speakers,
                iteration: 0,
                max_iterations: 0,
                loss: 0.0,
                overall_progress: 0.93,
                message: Some(format!(
                    "Inter-channel timbre matching (ref: '{}')...",
                    timbre_config.reference_channel
                )),
                epa_preference: None,
                step_id: None,
                step_status: None,
            },
        )?;
        info!(
            "Running inter-channel timbre matching (reference: '{}')...",
            timbre_config.reference_channel
        );

        // Build corrected curves from the current channel results
        let corrected_curves: HashMap<String, Curve> = channel_results
            .iter()
            .filter(|(name, _)| !is_subwoofer_channel(config, name))
            .map(|(name, result)| (name.clone(), result.final_curve.clone()))
            .collect();
        let (fit_min_freq, fit_max_freq) = shared_alignment_fit_band(config, &corrected_curves);

        let stage_outcome = match roomeq_engine::inter_channel_timbre_matching::compute_inter_channel_timbre_matching_with_threshold(
            &corrected_curves,
            &timbre_config.reference_channel,
            sample_rate,
            fit_min_freq,
            fit_max_freq,
            timbre_config.min_improvement_db,
        ) {
            Ok(timbre_results) => {
                let applied_count = timbre_results
                    .values()
                    .filter(|result| {
                        result.status == roomeq_engine::inter_channel_timbre_matching::TimbreMatchingChannelStatus::Applied
                    })
                    .count();
                let failed_count = timbre_results
                    .values()
                    .filter(|result| result.status == roomeq_engine::inter_channel_timbre_matching::TimbreMatchingChannelStatus::Failed)
                    .count();
                for (channel_name, timbre_result) in &timbre_results {
                    let plugins = roomeq_engine::inter_channel_timbre_matching::create_timbre_matching_plugins(timbre_result, sample_rate);
                    if !plugins.is_empty()
                        && let Some(chain) = channel_chains.get_mut(channel_name)
                    {
                        for plugin in plugins {
                            chain.plugins.push(plugin);
                        }
                    }
                    if let Some(alignment) = &timbre_result.alignment {
                        let shelf_filters =
                            roomeq_engine::spectral_align::create_alignment_filters(alignment, sample_rate);
                        sync_reported_biquad_adjustment(
                            channel_name,
                            &mut channel_results,
                            &mut channel_chains,
                            &shelf_filters,
                            sample_rate,
                        );
                        if alignment.flat_gain_db.abs() >= roomeq_engine::spectral_align::MIN_CORRECTION_DB
                        {
                            sync_reported_gain_adjustment(
                                channel_name,
                                &mut channel_results,
                                &mut channel_chains,
                                alignment.flat_gain_db,
                                false,
                                sample_rate,
                            );
                        }
                    }
                }
                curves = collect_current_final_curves(&channel_results);
                let status = if failed_count > 0 && applied_count > 0 {
                    StageStatus::Degraded
                } else if failed_count > 0 {
                    StageStatus::Failed
                } else if applied_count > 0 {
                    StageStatus::Applied
                } else {
                    StageStatus::Skipped
                };
                let mut advisories = timbre_results
                    .values()
                    .flat_map(|result| result.advisories.iter().cloned())
                    .filter(|advisory| advisory != "reference_channel")
                    .collect::<Vec<_>>();
                advisories.sort();
                advisories.dedup();
                StageOutcome {
                    stage: "inter_channel_timbre_matching".to_string(),
                    status,
                    advisories,
                }
            }
            Err(e) => {
                warn!("Inter-channel timbre matching failed: {}", e);
                StageOutcome {
                    stage: "inter_channel_timbre_matching".to_string(),
                    status: StageStatus::Failed,
                    advisories: vec![format!("invalid_reference: {e}")],
                }
            }
        };
        let event = match stage_outcome.status {
            StageStatus::Applied | StageStatus::Degraded => PipelineEvent::completed(
                PipelineStepId::InterChannelTimbreMatching,
                format!("Inter-channel timbre matching: {:?}", stage_outcome.status),
            ),
            StageStatus::Skipped | StageStatus::Failed => PipelineEvent::skipped(
                PipelineStepId::InterChannelTimbreMatching,
                format!(
                    "Inter-channel timbre matching: {:?} ({})",
                    stage_outcome.status,
                    stage_outcome.advisories.join(", ")
                ),
            ),
        };
        emit_pipeline_event(observer_shared, event)?;
        stage_outcomes.push(stage_outcome);
    }
    if !timbre_config.is_some_and(|config| config.enabled) {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::InterChannelTimbreMatching,
                "Inter-channel timbre matching not enabled",
            ),
        )?;
    }

    // ========================================================================
    // Role-aware height-channel alignment
    // ========================================================================
    if let Some(height_config) = config
        .optimizer
        .height_channel_alignment
        .as_ref()
        .filter(|config| config.enabled)
    {
        send_progress(
            observer_shared,
            PipelineStepId::HeightChannelAlignment,
            PipelineStepStatus::Started,
            &RoomOptimizationProgress {
                current_speaker: "Height-channel alignment".to_string(),
                speaker_index: 0,
                total_speakers,
                iteration: 0,
                max_iterations: 0,
                loss: 0.0,
                overall_progress: 0.935,
                message: Some("Aligning overhead channels to role-aware references...".to_string()),
                epa_preference: None,
                step_id: None,
                step_status: None,
            },
        )?;
        let corrected_curves = collect_current_final_curves(&channel_results);
        let (fit_min_freq, fit_max_freq) = shared_alignment_fit_band(config, &corrected_curves);
        let stage_outcome =
            match roomeq_engine::height_channel_alignment::compute_height_channel_alignment_with_coherence_threshold(
                &corrected_curves,
                &channel_arrivals,
            height_config,
            sample_rate,
            fit_min_freq,
            fit_max_freq,
            config
                .recording_config
                .as_ref()
                .and_then(|recording| recording.coherence_threshold)
                .map(f64::from)
                .unwrap_or(roomeq_engine::bass_phase_confidence::DEFAULT_COHERENCE_THRESHOLD),
        ) {
                Ok(mut height_results) => {
                    let mut applied_count = 0;
                    let failed_count = height_results
                    .values()
                    .filter(|result| {
                        result.status
                            == roomeq_engine::height_channel_alignment::HeightAlignmentStatus::Failed
                    })
                    .count();
                    for (channel_name, height_result) in &mut height_results {
                        let mut applied = false;
                        if let Some(alignment) = &height_result.alignment {
                            let (eq_plugin, gain_plugin) =
                                roomeq_engine::spectral_align::create_alignment_plugins(
                                    alignment,
                                    sample_rate,
                                );
                            if let Some(chain) = channel_chains.get_mut(channel_name) {
                                if let Some(eq) = eq_plugin {
                                    chain.plugins.push(eq);
                                }
                                if let Some(gain) = gain_plugin {
                                    chain.plugins.push(gain);
                                }
                            }
                            let shelf_filters =
                                roomeq_engine::spectral_align::create_alignment_filters(
                                    alignment,
                                    sample_rate,
                                );
                            sync_reported_biquad_adjustment(
                                channel_name,
                                &mut channel_results,
                                &mut channel_chains,
                                &shelf_filters,
                                sample_rate,
                            );
                            if alignment.flat_gain_db.abs()
                                >= roomeq_engine::spectral_align::MIN_CORRECTION_DB
                            {
                                sync_reported_gain_adjustment(
                                    channel_name,
                                    &mut channel_results,
                                    &mut channel_chains,
                                alignment.flat_gain_db,
                                false,
                                sample_rate,
                            );
                            }
                            applied = true;
                        }
        let delay_applied = config.optimizer.allow_delay()
            && height_result.delay_ms > 0.01
                            && channel_chains.get_mut(channel_name).is_some_and(|chain| {
                                if chain
                                    .plugins
                                    .iter()
                                    .any(|plugin| plugin.plugin_type == "delay")
                                {
                                    height_result
                                        .advisories
                                        .push("height_arrival_already_aligned".to_string());
                                    false
                                } else {
                                    chain.plugins.insert(
                                        0,
                                        output::create_delay_plugin(height_result.delay_ms),
                                    );
                                    true
                                }
                            });
                        if delay_applied {
                            sync_reported_phase_adjustment(
                                channel_name,
                                &mut channel_results,
                                &mut channel_chains,
                                height_result.delay_ms,
                                false,
                                sample_rate,
                            );
                            applied = true;
                        }
                        if applied {
                            applied_count += 1;
                        }
                    }
                    curves = collect_current_final_curves(&channel_results);
                    let mut advisories = height_results
                        .values()
                        .flat_map(|result| result.advisories.iter().cloned())
                        .collect::<Vec<_>>();
                    if height_results.is_empty() {
                        advisories.push("no_height_channels".to_string());
                    }
                    advisories.sort();
                    advisories.dedup();
                    let degraded = advisories.iter().any(|advisory| {
                        advisory.ends_with("_missing")
                            || advisory.ends_with("_untrustworthy")
                            || advisory == "height_objective_acceptance_failed"
                            || advisory == "height_arrives_after_reference"
                            || advisory == "height_delay_limit_exceeded"
                    });
                    let status = if applied_count > 0 && (failed_count > 0 || degraded) {
                        StageStatus::Degraded
                    } else if failed_count > 0 {
                        StageStatus::Failed
                    } else if applied_count > 0 {
                        StageStatus::Applied
                    } else {
                        StageStatus::Skipped
                    };
                    StageOutcome {
                        stage: "height_channel_alignment".to_string(),
                        status,
                        advisories,
                    }
                }
                Err(error) => StageOutcome {
                    stage: "height_channel_alignment".to_string(),
                    status: StageStatus::Failed,
                    advisories: vec![format!("height_alignment_failed: {error}")],
                },
            };
        let event = match stage_outcome.status {
            StageStatus::Applied | StageStatus::Degraded => PipelineEvent::completed(
                PipelineStepId::HeightChannelAlignment,
                format!("Height-channel alignment: {:?}", stage_outcome.status),
            ),
            StageStatus::Skipped | StageStatus::Failed => PipelineEvent::skipped(
                PipelineStepId::HeightChannelAlignment,
                format!(
                    "Height-channel alignment: {:?} ({})",
                    stage_outcome.status,
                    stage_outcome.advisories.join(", ")
                ),
            ),
        };
        emit_pipeline_event(observer_shared, event)?;
        stage_outcomes.push(stage_outcome);
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::HeightChannelAlignment,
                "Height-channel alignment not enabled",
            ),
        )?;
    }

    // ========================================================================
    // Phase Alignment Optimization (Scenario A: WITH Subwoofers)
    // ========================================================================
    // Phase alignment maximizes energy sum in the crossover region by optimizing
    // delay and polarity. This runs BEFORE group delay optimization.
    // Uses the same sub-main pairing logic as GD-Opt v2 (system config or heuristic).
    // (delay_ms, invert_polarity, sub_name) keyed by main speaker name
    let mut phase_alignment_results: HashMap<String, (f64, bool, String)> = HashMap::new();

    if !should_run_standalone_phase_correction(config)
        && config.optimizer.allow_delay()
        && let Some(phase_config) = &config.optimizer.phase_alignment
        && phase_config.enabled
    {
        let pairings = find_sub_main_pairings(config, &curves);

        if pairings.is_empty() {
            warn!("Phase alignment enabled but no valid sub-main pairings found.");
        } else {
            info!("Running phase alignment optimization...");
            send_progress(
                observer_shared,
                PipelineStepId::PhaseAlignment,
                PipelineStepStatus::Started,
                &RoomOptimizationProgress {
                    current_speaker: String::new(),
                    speaker_index: 0,
                    total_speakers: pairings.len(),
                    iteration: 0,
                    max_iterations: 0,
                    loss: 0.0,
                    overall_progress: 0.0,
                    message: Some("Running phase alignment...".to_string()),
                    epa_preference: None,
                    step_id: None,
                    step_status: None,
                },
            )?;

            for (sub_name, main_name) in &pairings {
                let sub_curve = match curves.get(sub_name) {
                    Some(c) => c,
                    None => {
                        warn!(
                            "Subwoofer channel '{}' not found for phase alignment",
                            sub_name
                        );
                        continue;
                    }
                };

                if let Some(speaker_curve) = curves.get(main_name) {
                    // Phase alignment requires phase data
                    if sub_curve.phase.is_some() && speaker_curve.phase.is_some() {
                        match phase_alignment::optimize_phase_alignment(
                            sub_curve,
                            speaker_curve,
                            phase_config,
                        ) {
                            Ok(result) => {
                                info!(
                                    "  Phase alignment '{}' with '{}': delay={:.2}ms, invert={}, improvement={:.2}dB",
                                    main_name,
                                    sub_name,
                                    result.delay_ms,
                                    result.invert_polarity,
                                    result.improvement_db
                                );
                                if phase_alignment_results
                                    .insert(
                                        main_name.clone(),
                                        (result.delay_ms, result.invert_polarity, sub_name.clone()),
                                    )
                                    .is_some()
                                {
                                    warn!(
                                        "Multiple subwoofers mapped to main '{}'; retaining the latest phase-alignment result",
                                        main_name
                                    );
                                }
                            }
                            Err(e) => {
                                warn!("  Phase alignment failed for '{}': {}", main_name, e);
                            }
                        }
                    } else {
                        debug!(
                            "  Skipping phase alignment for '{}': no phase data available",
                            main_name
                        );
                    }
                }
            }
        }
    }

    // Apply phase alignment results. The optimizer returns pairwise relative
    // delays, so resolve all pairs into one absolute non-negative delay schedule
    // before inserting DSP plugins.
    for (speaker_name, (delay_ms, invert, sub_name)) in &phase_alignment_results {
        if *invert {
            let applied = if let Some(chain) = channel_chains.get_mut(speaker_name) {
                // Insert polarity inversion at the beginning of the chain
                let invert_plugin = output::create_gain_plugin_with_invert(0.0, true);
                chain.plugins.insert(0, invert_plugin);
                true
            } else {
                false
            };

            if applied {
                sync_reported_phase_adjustment(
                    speaker_name,
                    &mut channel_results,
                    &mut channel_chains,
                    0.0,
                    true,
                    sample_rate,
                );
                info!("  Applied polarity inversion to '{}'", speaker_name);
            }
        }

        debug!(
            "  Phase alignment constraint: delay('{}') - delay('{}') = {:.3} ms",
            speaker_name, sub_name, delay_ms
        );
    }

    apply_phase_alignment_delay_schedule(
        &phase_alignment_results,
        &mut channel_results,
        &mut channel_chains,
        sample_rate,
    );
    if !phase_alignment_results.is_empty() {
        curves = collect_current_final_curves(&channel_results);
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::completed(PipelineStepId::PhaseAlignment, "Phase alignment complete"),
        )?;
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::PhaseAlignment,
                "Phase alignment not applied",
            ),
        )?;
    }

    // Group Delay Optimization v1 was removed in the 2.0 simplification pass.
    // The redesigned v2 integration runs below, after phase correction and
    // before IR/EPA/metadata generation.

    // Standalone phase correction (rePhase-style)
    if should_run_standalone_phase_correction(config) {
        send_progress(
            observer_shared,
            PipelineStepId::PhaseCorrection,
            PipelineStepStatus::Started,
            &RoomOptimizationProgress {
                current_speaker: "Phase correction".to_string(),
                speaker_index: 0,
                total_speakers,
                iteration: 0,
                max_iterations: 0,
                loss: 0.0,
                overall_progress: 0.96,
                message: Some("Phase correction...".to_string()),
                epa_preference: None,
                step_id: None,
                step_status: None,
            },
        )?;
    }
    if should_run_standalone_phase_correction(config)
        && let Some(ref pc_config) = config.optimizer.phase_correction
    {
        let names: Vec<String> = channel_results.keys().cloned().collect();
        for name in &names {
            if let Some(ch) = channel_results.get_mut(name.as_str())
                && let Some(chain) = channel_chains.get_mut(name.as_str())
            {
                apply_phase_correction(name, ch, chain, pc_config, sample_rate, output_dir);
            }
        }
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::completed(PipelineStepId::PhaseCorrection, "Phase correction complete"),
        )?;
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::PhaseCorrection,
                "Phase correction not enabled",
            ),
        )?;
    }

    if should_run_standalone_phase_correction(config)
        && apply_topology_phase_alignment(
            config,
            &mut channel_results,
            &mut channel_chains,
            observer_shared,
            sample_rate,
        )?
    {
        curves = collect_current_final_curves(&channel_results);
    }

    // ─── GD-Opt v2 integration (Phase GD-5) ──────────────────────────────
    // Run after all earlier phase/EQ stages have updated final_curve, but
    // before IR/EPA/metadata so exported reports reflect the audible chain.
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::started(
            PipelineStepId::GroupDelayOptimization,
            "Running GD optimization",
        ),
    )?;
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::new(
            PipelineStepId::GroupDelayOptimization,
            PipelineStepStatus::InProgress,
        )
        .with_message(
            if config.optimizer.processing_mode == ProcessingMode::PhaseLinear {
                "Phase-linear FIR group-delay optimization..."
            } else {
                "Group-delay optimization..."
            },
        ),
    )?;
    let group_delay_summary = if config.optimizer.processing_mode == ProcessingMode::PhaseLinear {
        try_run_phase_linear_fir_gd(
            config,
            &mut channel_results,
            &mut channel_chains,
            sample_rate,
            output_dir,
        )
    } else {
        try_run_gd_opt_with_frequency_samples(
            config,
            &mut channel_results,
            &mut channel_chains,
            sample_rate,
            frequency_samples,
        )
    };
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::completed(
            PipelineStepId::GroupDelayOptimization,
            "GD optimization complete",
        ),
    )?;
    if group_delay_summary
        .as_ref()
        .is_some_and(|summary| summary.applied)
    {
        curves = collect_current_final_curves(&channel_results);
    }

    // Compute IR waveforms (pre- and post-correction) for each channel
    send_progress(
        observer_shared,
        PipelineStepId::ImpulseResponseComputation,
        PipelineStepStatus::Started,
        &RoomOptimizationProgress {
            current_speaker: "IR computation".to_string(),
            speaker_index: 0,
            total_speakers,
            iteration: 0,
            max_iterations: 0,
            loss: 0.0,
            overall_progress: 0.97,
            message: Some("Computing impulse responses...".to_string()),
            epa_preference: None,
            step_id: None,
            step_status: None,
        },
    )?;
    let ir_total = channel_results.len();
    for (ir_index, (channel_name, result)) in channel_results.iter().enumerate() {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::new(
                PipelineStepId::ImpulseResponseComputation,
                PipelineStepStatus::InProgress,
            )
            .with_channel(channel_name.clone())
            .with_channels(ir_index, ir_total)
            .with_message(format!("Computing impulse response for {channel_name}"))
            .with_overall_progress(0.97),
        )?;
        let delay_ms = channel_chains
            .get(channel_name)
            .map(total_chain_delay_ms)
            .unwrap_or(0.0);

        if let Some((pre_ir, post_ir)) =
            roomeq_engine::analysis::ir_waveform::compute_channel_ir_waveforms(
                &result.initial_curve,
                &result.biquads,
                result.fir_coeffs.as_deref(),
                delay_ms,
                sample_rate,
            )
            && let Some(chain) = channel_chains.get_mut(channel_name)
        {
            chain.pre_ir = Some(pre_ir);
            chain.post_ir = Some(post_ir);
        }
    }
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::completed(
            PipelineStepId::ImpulseResponseComputation,
            "Impulse responses computed",
        ),
    )?;

    // Aggregate scores
    let avg_pre_score = if !pre_scores.is_empty() {
        pre_scores.iter().sum::<f64>() / pre_scores.len() as f64
    } else {
        0.0
    };
    let avg_post_score = if !post_scores.is_empty() {
        post_scores.iter().sum::<f64>() / post_scores.len() as f64
    } else {
        0.0
    };

    info!(
        "Average pre-score: {:.4}, post-score: {:.4}",
        avg_pre_score, avg_post_score
    );

    // Identify acoustic groups for consistency checks
    let acoustic_groups = identify_acoustic_groups(config);
    for (group_name, group_channels) in &acoustic_groups {
        if group_channels.len() > 1 {
            debug!("Acoustic Group '{}': {:?}", group_name, group_channels);

            // Perform consistency checks between speakers in the same group
            check_group_consistency(group_name, group_channels, &channel_means, &curves);
        }
    }

    let epa_cfg = config.optimizer.epa_config.clone().unwrap_or_default();
    let epa_per_channel = roomeq_engine::output::compute_epa_per_channel(&channel_chains, &epa_cfg);
    let epa_multichannel =
        roomeq_engine::output::compute_epa_multichannel(&channel_chains, &epa_cfg);
    let mixed_phase_per_channel =
        roomeq_engine::output::take_mixed_phase_reports(&mut channel_chains);

    let metadata = OptimizationMetadata {
        pre_score: avg_pre_score,
        post_score: avg_post_score,
        algorithm: config.optimizer.algorithm.clone(),
        loss_type: Some(config.optimizer.loss_type.clone()),
        iterations: config.optimizer.max_iter,
        timestamp: chrono::Utc::now().to_rfc3339(),
        inter_channel_deviation: None,
        epa_per_channel,
        epa_multichannel,
        group_delay: group_delay_summary,
        mixed_phase_per_channel,
        perceptual_metrics: None,
        home_cinema_layout: None,
        multi_seat_coverage: None,
        multi_seat_correction: None,
        bass_management: None,
        timing_diagnostics: build_timing_diagnostics(config, &channel_arrivals, &channel_chains),
        ctc: None,
        perceptual_policy: None,
        bootstrap_uncertainty: None,
        validation_bundle: None,
        supporting_source: None,
        correction_acceptance: None,
        optimizer_evidence: None,
        stage_outcomes,
        effective_config: None,
    };

    let mut result = RoomOptimizationResult {
        channels: channel_chains,
        channel_results,
        deployed_source_curves: HashMap::new(),
        combined_pre_score: avg_pre_score,
        combined_post_score: avg_post_score,
        metadata,
    };

    // Compute inter-channel deviation and optionally correct it
    if curves.len() > 1 {
        send_progress(
            observer_shared,
            PipelineStepId::ChannelMatching,
            PipelineStepStatus::Started,
            &RoomOptimizationProgress {
                current_speaker: "Channel matching".to_string(),
                speaker_index: 0,
                total_speakers,
                iteration: 0,
                max_iterations: 0,
                loss: 0.0,
                overall_progress: 0.98,
                message: Some("Channel matching analysis...".to_string()),
                epa_preference: None,
                step_id: None,
                step_status: None,
            },
        )?;
        compute_and_correct_icd(&mut result, config, sample_rate);
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::completed(PipelineStepId::ChannelMatching, "Channel matching complete"),
        )?;
    } else {
        emit_pipeline_event(
            observer_shared,
            PipelineEvent::skipped(
                PipelineStepId::ChannelMatching,
                "Channel matching not needed",
            ),
        )?;
    }

    let sidecar_dir = output_dir.unwrap_or(Path::new("."));
    refresh_temporal_ir_evidence(&mut result, config, sample_rate, sidecar_dir);
    apply_final_correction_safety_gate_preserving_routed_crossover(
        &mut result,
        sample_rate,
        config.optimizer.smooth_n,
        (config.optimizer.min_freq, config.optimizer.max_freq),
        sidecar_dir,
        config.optimizer.processing_mode.clone(),
        group_delay_budget_ms(config),
    )?;
    record_missing_mixed_phase_fir_reversions(
        &mut result,
        config.optimizer.processing_mode.clone(),
    );

    emit_pipeline_event(
        observer_shared,
        PipelineEvent::started(PipelineStepId::MetadataRefresh, "Refreshing reports"),
    )?;
    refresh_final_reports(&mut result, config, sample_rate, sidecar_dir);
    apply_ctc_if_enabled(&mut result, config, sample_rate, output_dir)?;
    generate_validation_bundle_report(&mut result, config, output_dir, store)?;
    emit_pipeline_event(
        observer_shared,
        PipelineEvent::completed(PipelineStepId::MetadataRefresh, "Reports refreshed"),
    )?;

    Ok(result)
}

/// Optimize a single speaker (simple or group)
///
/// # Arguments
/// * `channel_name` - Name of the channel
/// * `speaker_config` - Speaker configuration
/// * `optimizer_config` - Optimizer parameters
/// * `target_curve` - Optional target curve configuration
/// * `sample_rate` - Sample rate for filter design
/// * `callback` - Optional per-iteration optimizer progress callback
///
/// # Returns
/// * `SpeakerOptimizationResult` containing DSP chain and optimization results
pub fn optimize_speaker(
    channel_name: &str,
    speaker_config: &SpeakerConfig,
    optimizer_config: &OptimizerConfig,
    target_curve: Option<&TargetCurveConfig>,
    sample_rate: f64,
    callback: Option<SpeakerOptimizationCallback>,
) -> Result<SpeakerOptimizationResult> {
    let optimizer_config = optimizer_config.clone();

    // Create a minimal RoomConfig for internal processing
    let room_config = RoomConfig {
        version: roomeq_model::default_config_version(),
        system: None,
        speakers: HashMap::new(),
        crossovers: None,
        target_curve: target_curve.cloned(),
        optimizer: optimizer_config,
        recording_config: None,
        ctc: None,
        cea2034_cache: None,
        provenance: Default::default(),
    };
    let max_iterations = generic_channel_progress_iterations(&room_config);
    let channel_for_progress = channel_name.to_string();
    let eq_callback: Option<roomeq_engine::OptimProgressCallback> = callback.map(|mut callback| {
        Box::new(move |iteration, loss, epa_preference| {
            let action = callback(&RoomOptimizationProgress {
                current_speaker: channel_for_progress.clone(),
                speaker_index: 0,
                total_speakers: 1,
                iteration,
                max_iterations,
                loss,
                overall_progress: if max_iterations > 0 {
                    (iteration as f64 / max_iterations as f64).min(1.0)
                } else {
                    0.0
                },
                message: None,
                epa_preference,
                step_id: None,
                step_status: None,
            });
            match action {
                CallbackAction::Continue => roomeq_engine::CallbackAction::Continue,
                CallbackAction::Stop => roomeq_engine::CallbackAction::Stop,
            }
        }) as roomeq_engine::OptimProgressCallback
    });

    let (
        chain,
        pre_score,
        post_score,
        initial_curve,
        final_curve,
        biquads,
        _mean_spl,
        _arrival_time_ms,
        fir_coeffs,
        optimizer_evidence,
    ) = process_speaker_internal(
        channel_name,
        speaker_config,
        &room_config,
        sample_rate,
        None,
        eq_callback,
        None, // no shared mean for standalone single-channel optimization
        None, // no probe_arrival_overrides on the standalone path
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )?;

    Ok(SpeakerOptimizationResult {
        chain,
        pre_score,
        post_score,
        initial_curve,
        final_curve,
        biquads,
        fir_coeffs,
        optimizer_evidence,
    })
}
