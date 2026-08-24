// Home-cinema workflow executor (X.0 / X.1, any channel count).

use super::bass_management::*;
use super::run::run_channel_via_generic_path_with_frequency_samples;
use super::run::run_post_eq;
use super::supporting_source::process_supporting_source_channels_with_frequency_samples;
use super::types::{WorkflowAssembly, WorkflowExecutor};
use super::workflow::workflow_progress_callback;
use super::workflow::workflow_stage_event;
use crate::measurement::{
    load_source_individual_with_frequency_samples, load_source_with_frequency_samples,
};
use log::info;
use rayon::prelude::*;
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_engine::room_result::{ChannelOptimizationResult, RoomOptimizationResult};
use roomeq_engine::topology::{
    align_channels_to_lowest, all_curves_have_usable_phase, all_curves_share_frequency_grid,
    apply_crossover_response_to_curve, apply_curve_delta_to_reference_curve,
    apply_delay_and_polarity_to_curve, average_mains_magnitude, bass_management_objective,
    complex_sum_mains, compute_flat_loss, mark_plugin_stage, mark_plugins_stage,
    mark_route_owned_plugin, normalize_crossover_delays, predict_bass_management_sum,
    select_bass_management_crossover_type,
};
use roomeq_engine::{
    Curve, OptimizerRunEvidence, PipelineStepId, PipelineStepStatus,
    bass_management as engine_bass_management, crossover, home_cinema as engine_home_cinema,
    output, response,
};
use roomeq_model::{
    BassManagementRoutingGraph, ChannelDspChain, CurveData, DriverDspChain, MeasurementSource,
    OptimizationMetadata, PluginConfigWrapper, RoomConfig, SpeakerConfig, SystemConfig,
};
use std::collections::HashMap;

pub(super) struct HomeCinemaExecutor;

/// Common acoustic calibration band for home-cinema main channels.
///
/// Level trims must use the same band as the main-channel correction target;
/// otherwise independently flattened channels retain the difference between
/// their broad-band and correction-band means.
fn main_level_alignment_band(config: &RoomConfig) -> (f64, f64) {
    let high = config.optimizer.max_freq.max(config.optimizer.min_freq);
    let low = config.optimizer.min_freq.max(100.0).min(high);
    (low, high)
}

fn average_spl(curve: &Curve, band: (f64, f64)) -> f64 {
    let (sum, count) = curve
        .freq
        .iter()
        .zip(curve.spl.iter())
        .filter(|(frequency, spl)| {
            **frequency >= band.0 && **frequency <= band.1 && spl.is_finite()
        })
        .fold((0.0, 0_usize), |(sum, count), (_, spl)| {
            (sum + *spl, count + 1)
        });
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn shift_curve_level(curve: &mut Curve, gain_db: f64) {
    curve.spl.mapv_inplace(|value| value + gain_db);
}

fn apply_gain_to_main_chain(chain: &mut ChannelDspChain, gain_db: f64) {
    if gain_db.abs() <= 0.01 {
        return;
    }
    let mut plugin = mark_plugin_stage(output::create_gain_plugin(gain_db), "pre_route");
    if let Some(parameters) = plugin.parameters.as_object_mut() {
        parameters.insert(
            "label".to_string(),
            serde_json::Value::String("post_dsp_input_level_alignment".to_string()),
        );
    }
    let route_owned_index = chain
        .plugins
        .iter()
        .position(|plugin| {
            plugin
                .parameters
                .get("room_eq_stage")
                .and_then(|value| value.as_str())
                == Some("route_owned")
        })
        .unwrap_or(chain.plugins.len());
    chain.plugins.insert(route_owned_index, plugin);

    if let Some(final_curve) = chain.final_curve.take() {
        let mut curve: Curve = final_curve.into();
        shift_curve_level(&mut curve, gain_db);
        let final_data: CurveData = (&curve).into();
        chain.eq_response = chain
            .initial_curve
            .as_ref()
            .map(|initial| output::compute_eq_response(initial, &final_data));
        chain.final_curve = Some(final_data);
    }
}

/// Align every logical input at the microphone after the complete routed DSP
/// graph. Trims are down-only so calibration cannot consume headroom.
#[allow(clippy::too_many_arguments)]
fn calibrate_post_dsp_input_levels(
    config: &RoomConfig,
    main_roles: &[String],
    lfe_role: &str,
    main_band: (f64, f64),
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
    channels: &mut HashMap<String, ChannelDspChain>,
    graph: &mut BassManagementRoutingGraph,
) -> Result<HashMap<String, f64>> {
    let mut common_sub_chain =
        channels
            .get(lfe_role)
            .cloned()
            .ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: format!("missing physical sub chain '{lfe_role}' for level calibration"),
            })?;
    common_sub_chain.plugins.retain(|plugin| {
        plugin
            .parameters
            .get("room_eq_stage")
            .and_then(|value| value.as_str())
            != Some("route_owned")
    });
    let sub_initial: Curve = common_sub_chain
        .initial_curve
        .clone()
        .ok_or_else(|| AutoeqError::InvalidMeasurement {
            message: format!("physical sub chain '{lfe_role}' has no initial curve"),
        })?
        .into();
    let common_sub_curve = crate::ctc::apply_channel_dsp_chain_to_curve_with_sidecar_dir(
        &common_sub_chain,
        &sub_initial,
        sample_rate,
        sidecar_dir,
    )?;

    let mut means = HashMap::new();
    for role in main_roles {
        let main_curve: Curve = channels
            .get(role)
            .and_then(|chain| chain.final_curve.clone())
            .ok_or_else(|| AutoeqError::InvalidMeasurement {
                message: format!("main channel '{role}' has no final curve"),
            })?
            .into();
        let observed = if let Some(sub_branch) =
            engine_bass_management::predict_bass_source_curve_from_routes(
                &common_sub_curve,
                graph,
                role,
                sample_rate,
            ) {
            complex_sum_mains(&[&main_curve, &sub_branch])
        } else {
            main_curve
        };
        means.insert(role.clone(), average_spl(&observed, main_band));
    }

    let lfe_upper_hz = graph
        .routes
        .iter()
        .filter(|route| {
            route.source_channel == lfe_role && route.route_kind == "lfe_lowpass_to_sub"
        })
        .filter_map(|route| route.low_pass_hz)
        .fold(f64::NEG_INFINITY, f64::max);
    if lfe_upper_hz.is_finite()
        && let Some(lfe_curve) = engine_bass_management::predict_bass_source_curve_from_routes(
            &common_sub_curve,
            graph,
            lfe_role,
            sample_rate,
        )
    {
        means.insert(
            lfe_role.to_string(),
            average_spl(&lfe_curve, (20.0, lfe_upper_hz)),
        );
    }

    let target = means
        .values()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::INFINITY, f64::min);
    if !target.is_finite() || means.len() != main_roles.len() + 1 {
        return Ok(HashMap::new());
    }
    let mut trims: HashMap<String, f64> = means
        .into_iter()
        .map(|(role, mean)| (role, (target - mean).min(0.0)))
        .collect();

    for role in main_roles {
        if let Some(chain) = channels.get_mut(role) {
            apply_gain_to_main_chain(chain, *trims.get(role).unwrap_or(&0.0));
        }
    }
    for route in &mut graph.routes {
        if matches!(
            route.route_kind.as_str(),
            "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
        ) {
            let trim = *trims.get(&route.source_channel).unwrap_or(&0.0);
            route.gain_db += trim;
            route.gain_linear = 10.0_f64.powf(route.gain_db / 20.0);
            route.matrix_gain = route.gain_linear;
        }
    }
    // Only the configured correlated-bus headroom model may aggregate
    // independent logical inputs. Preserve all source relationships with one
    // common down-only safety trim if that model predicts overload.
    if let Some(effective) = engine_home_cinema::effective_bass_management(config)
        && let Some(headroom) = engine_home_cinema::simulate_bass_bus_headroom(
            Some(graph),
            &effective.config.headroom_model,
            effective.config.headroom_margin_db,
            sample_rate,
        )
    {
        let safety_trim_db = (-headroom.margin_db + 0.1).max(0.0);
        if safety_trim_db > 0.0 {
            for role in main_roles {
                if let Some(chain) = channels.get_mut(role) {
                    apply_gain_to_main_chain(chain, -safety_trim_db);
                }
            }
            for trim_db in trims.values_mut() {
                *trim_db -= safety_trim_db;
            }
            for route in &mut graph.routes {
                if matches!(
                    route.route_kind.as_str(),
                    "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
                ) {
                    route.gain_db -= safety_trim_db;
                    route.gain_linear = 10.0_f64.powf(route.gain_db / 20.0);
                    route.matrix_gain = route.gain_linear;
                }
            }
            graph.advisories.push(format!(
                "common_input_headroom_safety_trim_db:{safety_trim_db:.3}"
            ));
        }
    }
    if let Some(matrix) = graph.matrix.as_mut() {
        matrix.matrix = graph
            .routes
            .iter()
            .filter(|route| {
                matches!(
                    route.route_kind.as_str(),
                    "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
                )
            })
            .map(|route| route.matrix_gain as f32)
            .collect();
        matrix.route_count = matrix.matrix.len();
    }
    graph.input_trim_db = trims.clone();
    graph
        .advisories
        .push("post_dsp_input_levels_aligned_down".to_string());

    if let Some(final_bass_bus) = engine_bass_management::predict_bass_output_curve_from_routes(
        &common_sub_curve,
        graph,
        &graph.physical_sub_output,
        sample_rate,
    ) && let Some(sub_chain) = channels.get_mut(lfe_role)
    {
        let final_data: CurveData = (&final_bass_bus).into();
        sub_chain.eq_response = sub_chain
            .initial_curve
            .as_ref()
            .map(|initial| output::compute_eq_response(initial, &final_data));
        sub_chain.final_curve = Some(final_data);
    }

    Ok(trims)
}

impl WorkflowExecutor for HomeCinemaExecutor {
    fn execute<'cfg, 'p, 's>(
        &self,
        assembly: &mut WorkflowAssembly<'cfg, 'p, 's>,
    ) -> Result<RoomOptimizationResult> {
        let config = assembly.config;
        let sys = assembly.sys;
        let sample_rate = assembly.sample_rate;
        let output_dir = assembly.output_dir;

        let sub_role = engine_home_cinema::bass_output_role(config, sys);
        let has_sub = sys.speakers.contains_key(&sub_role);

        // Classify channels into main and sub
        let main_roles = canonical_main_roles(sys, &sub_role);

        // Partition mains into single-source and supporting-source channels.
        let mut single_roles: Vec<String> = Vec::new();
        let mut supporting_roles: Vec<String> = Vec::new();
        let mut curves = HashMap::new();
        for role in &main_roles {
            let meas_key = sys
                .speakers
                .get(role)
                .ok_or(AutoeqError::InvalidConfiguration {
                    message: format!("Missing speaker mapping for '{}'", role),
                })?;
            let cfg = config
                .speakers
                .get(meas_key)
                .ok_or(AutoeqError::InvalidConfiguration {
                    message: format!("Missing speaker config for key '{}'", meas_key),
                })?;
            match cfg {
                SpeakerConfig::Single(s) => {
                    let curve = load_source_with_frequency_samples(s, assembly.frequency_samples)
                        .map_err(|e| AutoeqError::InvalidMeasurement {
                        message: e.to_string(),
                    })?;
                    curves.insert(role.clone(), curve);
                    single_roles.push(role.clone());
                }
                SpeakerConfig::SupportingSource(_) => {
                    supporting_roles.push(role.clone());
                }
                _ => {
                    return Err(AutoeqError::InvalidConfiguration {
                        message: format!(
                            "'{}' must be a Single or SupportingSource speaker config in home cinema workflow",
                            role
                        ),
                    });
                }
            };
        }

        info!(
            "Running Home Cinema Optimization Workflow ({} single mains, {} supporting sources{})",
            single_roles.len(),
            supporting_roles.len(),
            if has_sub { " + bass-managed sub" } else { "" }
        );

        // Bass-management and score aggregation require a primary main. Do
        // not let a schema-valid supporting-only layout reach `main_roles[0]`.
        if single_roles.is_empty() && has_sub {
            return Err(AutoeqError::InvalidConfiguration {
                message: "Home-cinema supporting-source layouts with bass management require at least one Single main channel to establish a crossover reference".to_string(),
            });
        }

        if single_roles.is_empty() {
            let mut result = supporting_only_home_cinema_result(config);
            process_supporting_source_channels_with_frequency_samples(
                config,
                sys,
                sample_rate,
                output_dir,
                &mut result.channels,
                &mut result.channel_results,
                &mut result.metadata,
                assembly.frequency_samples,
            )?;
            return Ok(result);
        }

        // Load bass output if present (handles Single, MultiSub/MSO, Cardioid, DBA)
        let sub_preprocess = if has_sub {
            let sub_sys = sys
                .subwoofers
                .as_ref()
                .ok_or(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "Missing subwoofers configuration for home cinema with '{}'",
                        sub_role
                    ),
                })?;
            let lfe_meas_key =
                sys.speakers
                    .get(&sub_role)
                    .ok_or(AutoeqError::InvalidConfiguration {
                        message: format!("Missing speaker mapping for '{}'", sub_role),
                    })?;
            let lfe_speaker_config =
                config
                    .speakers
                    .get(lfe_meas_key)
                    .ok_or(AutoeqError::InvalidConfiguration {
                        message: format!("Missing speaker config for key '{}'", lfe_meas_key),
                    })?;
            let sp = preprocess_sub_with_frequency_samples(
                lfe_speaker_config,
                &sub_sys.config,
                &config.optimizer,
                sample_rate,
                assembly.frequency_samples,
            )?;
            curves.insert(sub_role.clone(), sp.combined_curve.clone());
            Some(sp)
        } else {
            None
        };

        let mut result = if has_sub {
            let total_channels = single_roles.len() + 1;
            optimize_home_cinema_with_sub(
                config,
                sys,
                &single_roles,
                &curves,
                sub_preprocess.unwrap(),
                sample_rate,
                output_dir,
                assembly,
                total_channels,
            )
        } else {
            let total_channels = single_roles.len();
            optimize_home_cinema_no_sub(
                config,
                sys,
                &single_roles,
                &curves,
                sample_rate,
                output_dir,
                assembly,
                total_channels,
            )
        }?;

        if !supporting_roles.is_empty() {
            info!(
                "Processing {} supporting-source channel(s) after mains",
                supporting_roles.len()
            );
            process_supporting_source_channels_with_frequency_samples(
                config,
                sys,
                sample_rate,
                output_dir,
                &mut result.channels,
                &mut result.channel_results,
                &mut result.metadata,
                assembly.frequency_samples,
            )?;
        }

        Ok(result)
    }
}

fn canonical_main_roles(sys: &SystemConfig, sub_role: &str) -> Vec<String> {
    let mut roles: Vec<String> = sys
        .speakers
        .keys()
        .filter(|role| {
            *role != sub_role && !engine_home_cinema::role_for_channel(role).is_sub_or_lfe()
        })
        .cloned()
        .collect();
    roles.sort();
    roles
}

fn supporting_only_home_cinema_result(config: &RoomConfig) -> RoomOptimizationResult {
    RoomOptimizationResult {
        channels: HashMap::new(),
        channel_results: HashMap::new(),
        combined_pre_score: 0.0,
        combined_post_score: 0.0,
        metadata: OptimizationMetadata {
            pre_score: 0.0,
            post_score: 0.0,
            algorithm: config.optimizer.algorithm.clone(),
            loss_type: Some(config.optimizer.loss_type.clone()),
            iterations: config.optimizer.max_iter,
            timestamp: chrono::Utc::now().to_rfc3339(),
            inter_channel_deviation: None,
            epa_per_channel: None,
            epa_multichannel: None,
            group_delay: None,
            mixed_phase_per_channel: None,
            perceptual_metrics: None,
            home_cinema_layout: Some(engine_home_cinema::analyze_layout(config)),
            multi_seat_coverage: Some(crate::home_cinema::multi_seat_coverage(config)),
            multi_seat_correction: None,
            bass_management: None,
            timing_diagnostics: None,
            ctc: None,
            perceptual_policy: None,
            bootstrap_uncertainty: None,
            validation_bundle: None,
            supporting_source: None,
            correction_acceptance: None,
            optimizer_evidence: None,
            stage_outcomes: Vec::new(),
            effective_config: None,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn optimize_home_cinema_no_sub(
    config: &RoomConfig,
    sys: &SystemConfig,
    main_roles: &[String],
    curves: &HashMap<String, Curve>,
    sample_rate: f64,
    output_dir: &std::path::Path,
    assembly: &mut WorkflowAssembly<'_, '_, '_>,
    total_channels: usize,
) -> Result<RoomOptimizationResult> {
    // Level alignment: mains measured from 100 Hz to 2000 Hz
    let mut ranges = HashMap::new();
    for role in main_roles {
        ranges.insert(role.clone(), (100.0, 2000.0));
    }
    let gains = align_channels_to_lowest(curves, &ranges);

    let mut channel_chains = HashMap::new();
    let mut channel_results = HashMap::new();
    let mut pre_scores = Vec::new();
    let mut post_scores = Vec::new();
    let mut multi_seat_rejections: HashMap<String, Vec<String>> = HashMap::new();

    let max_iterations = config.optimizer.max_iter;
    let progress_factory = assembly.progress_factory;
    let probe_arrival_overrides = assembly.probe_arrival_overrides;
    let frequency_samples = assembly.frequency_samples;
    let channel_outputs: Result<Vec<_>> = main_roles
        .par_iter()
        .enumerate()
        .map(|(channel_index, role)| {
            let gain = *gains.get(role).unwrap_or(&0.0);
            let source = resolve_single_source(role, config, sys)?;

            info!("  Optimizing '{}' with alignment gain {:.2} dB", role, gain);

            let (chain, ch_result, pre_score, post_score, _fir, multiseat_rejection) =
                run_channel_via_generic_path_with_frequency_samples(
                    role,
                    source,
                    config,
                    gain,
                    sample_rate,
                    output_dir,
                    &progress_factory,
                    channel_index,
                    total_channels,
                    max_iterations,
                    probe_arrival_overrides,
                    frequency_samples,
                )?;

            info!(
                "  '{}' pre_score={:.4} post_score={:.4}",
                role, pre_score, post_score
            );

            Ok((
                role.clone(),
                chain,
                ch_result,
                pre_score,
                post_score,
                multiseat_rejection,
            ))
        })
        .collect();
    for (role, chain, ch_result, pre_score, post_score, multiseat_rejection) in channel_outputs? {
        if let Some(advisories) = multiseat_rejection {
            multi_seat_rejections.insert(role.clone(), advisories);
        }
        channel_chains.insert(role.clone(), chain);
        channel_results.insert(role, ch_result);
        pre_scores.push(pre_score);
        post_scores.push(post_score);
    }
    workflow_stage_event(
        &mut assembly.stage_callback,
        PipelineStepId::GenericChannelOptimization,
        PipelineStepStatus::Completed,
        "Optimized home-cinema channels",
        0.90,
    )?;

    let avg_pre = pre_scores.iter().sum::<f64>() / pre_scores.len() as f64;
    let avg_post = post_scores.iter().sum::<f64>() / post_scores.len() as f64;

    info!(
        "Average pre-score: {:.4}, post-score: {:.4}",
        avg_pre, avg_post
    );

    let epa_cfg = config.optimizer.epa_config.clone().unwrap_or_default();
    let epa_per_channel = output::compute_epa_per_channel(&channel_chains, &epa_cfg);
    let epa_multichannel = output::compute_epa_multichannel(&channel_chains, &epa_cfg);
    let multi_seat_correction = Some(
        crate::home_cinema::multi_seat_correction_report_with_frequency_samples(
            config,
            &channel_results,
            Some(&multi_seat_rejections),
            assembly.frequency_samples,
        ),
    );
    Ok(RoomOptimizationResult {
        channels: channel_chains,
        channel_results,
        combined_pre_score: avg_pre,
        combined_post_score: avg_post,
        metadata: OptimizationMetadata {
            pre_score: avg_pre,
            post_score: avg_post,
            algorithm: config.optimizer.algorithm.clone(),
            loss_type: Some(config.optimizer.loss_type.clone()),
            iterations: config.optimizer.max_iter,
            timestamp: chrono::Utc::now().to_rfc3339(),
            inter_channel_deviation: None,
            epa_per_channel,
            epa_multichannel,
            group_delay: None,
            mixed_phase_per_channel: None,
            perceptual_metrics: None,
            home_cinema_layout: Some(engine_home_cinema::analyze_layout(config)),
            multi_seat_coverage: Some(crate::home_cinema::multi_seat_coverage(config)),
            multi_seat_correction,
            bass_management: None,
            timing_diagnostics: None,
            ctc: None,
            perceptual_policy: None,
            bootstrap_uncertainty: None,
            validation_bundle: None,
            supporting_source: None,
            correction_acceptance: None,
            optimizer_evidence: None,
            stage_outcomes: Vec::new(),
            effective_config: None,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn optimize_home_cinema_with_sub(
    config: &RoomConfig,
    sys: &SystemConfig,
    main_roles: &[String],
    curves: &HashMap<String, Curve>,
    sub_preprocess: SubPreprocessResult,
    sample_rate: f64,
    output_dir: &std::path::Path,
    assembly: &mut WorkflowAssembly<'_, '_, '_>,
    total_channels: usize,
) -> Result<RoomOptimizationResult> {
    let sub_role = engine_home_cinema::bass_output_role(config, sys);

    // Resolve crossover config
    let sub_sys = sys.subwoofers.as_ref().unwrap();
    let xover_key = sub_sys
        .crossover
        .as_deref()
        .ok_or(AutoeqError::InvalidConfiguration {
            message: "Subwoofer config requires 'crossover' reference".to_string(),
        })?;
    let xover_config = config
        .crossovers
        .as_ref()
        .and_then(|m| m.get(xover_key))
        .ok_or(AutoeqError::InvalidConfiguration {
            message: format!("Crossover '{}' not found in crossovers section", xover_key),
        })?;
    let xover_type_str = &xover_config.crossover_type;
    let bass_management = engine_home_cinema::effective_bass_management(config);

    let (min_xo, max_xo, est_xo) = if let Some(f) = xover_config.frequency {
        (f, f, f)
    } else if let Some((min, max)) = xover_config.frequency_range {
        (min, max, (min * max).sqrt())
    } else {
        return Err(AutoeqError::InvalidConfiguration {
            message: "Subwoofer crossover requires 'frequency' or 'frequency_range'".to_string(),
        });
    };

    // 1. Level alignment
    let mut ranges = HashMap::new();
    let main_alignment_band = main_level_alignment_band(config);
    for role in main_roles {
        ranges.insert(role.clone(), main_alignment_band);
    }
    let sub_min_align = config.optimizer.min_freq.max(20.0);
    ranges.insert(sub_role.clone(), (sub_min_align, max_xo));

    let gains = align_channels_to_lowest(curves, &ranges);

    let mut aligned_curves = HashMap::new();
    for (role, curve) in curves {
        let mut c = curve.clone();
        let g = *gains.get(role).unwrap_or(&0.0);
        for s in c.spl.iter_mut() {
            *s += g;
        }
        aligned_curves.insert(role.clone(), c);
    }

    // 2. Pre-EQ
    let mut pre_eq_plugins: HashMap<String, Vec<PluginConfigWrapper>> = HashMap::new();
    let mut pre_eq_initial_curves: HashMap<String, Curve> = HashMap::new();
    let mut linearized_curves: HashMap<String, Curve> = HashMap::new();
    let mut optimizer_evidence_by_channel: HashMap<String, Vec<OptimizerRunEvidence>> =
        HashMap::new();
    let mut multi_seat_rejections: HashMap<String, Vec<String>> = HashMap::new();

    let max_iterations = config.optimizer.max_iter;
    let progress_factory = assembly.progress_factory;
    let probe_arrival_overrides = assembly.probe_arrival_overrides;
    let frequency_samples = assembly.frequency_samples;
    let (pre_eq_outputs, sub_pre_eq_output) = rayon::join(
        || {
            main_roles
                .par_iter()
                .enumerate()
                .map(|(channel_index, role)| {
                    let source = resolve_single_source(role, config, sys)?;
                    let mut per_config = config.clone();
                    if min_xo < per_config.optimizer.max_freq {
                        per_config.optimizer.min_freq = per_config.optimizer.min_freq.max(min_xo);
                    } else {
                        log::warn!(
                            "  Main Pre-EQ crossover lower bound {:.1} Hz does not overlap configured optimization band [{:.1}, {:.1}] Hz; retaining the configured band",
                            min_xo,
                            per_config.optimizer.min_freq,
                            per_config.optimizer.max_freq
                        );
                    }
                    info!(
                        "  Pre-EQ via generic path for '{}' (min_freq={:.1} Hz)",
                        role, min_xo
                    );
                    let (chain, ch_result, _pre, _post, _fir, multiseat_rejection) =
                        run_channel_via_generic_path_with_frequency_samples(
                            role,
                            source,
                            &per_config,
                            0.0,
                            sample_rate,
                            output_dir,
                            &progress_factory,
                            channel_index,
                            total_channels,
                            max_iterations,
                            probe_arrival_overrides,
                            frequency_samples,
                        )?;
                    Ok((role.clone(), chain, ch_result, multiseat_rejection))
                })
                .collect::<Result<Vec<_>>>()
        },
        || {
            let sub_source = MeasurementSource::InMemory(sub_preprocess.combined_curve.clone());
            let mut sub_config = config.clone();
            if max_xo > sub_config.optimizer.min_freq {
                sub_config.optimizer.max_freq = sub_config.optimizer.max_freq.min(max_xo);
            } else {
                log::warn!(
                    "  Sub Pre-EQ crossover upper bound {:.1} Hz does not overlap configured optimization band [{:.1}, {:.1}] Hz; retaining the configured band",
                    max_xo,
                    sub_config.optimizer.min_freq,
                    sub_config.optimizer.max_freq
                );
            }
            info!(
                "  Pre-EQ via generic path for '{}' (max_freq={:.1} Hz)",
                sub_role, max_xo
            );
            let (chain, ch_result, _pre, _post, _fir, multiseat_rejection) =
                run_channel_via_generic_path_with_frequency_samples(
                    &sub_role,
                    &sub_source,
                    &sub_config,
                    0.0,
                    sample_rate,
                    output_dir,
                    &progress_factory,
                    main_roles.len(),
                    total_channels,
                    max_iterations,
                    probe_arrival_overrides,
                    frequency_samples,
                )?;
            Ok::<_, AutoeqError>((chain, ch_result, multiseat_rejection))
        },
    );
    for (role, chain, ch_result, multiseat_rejection) in pre_eq_outputs? {
        if let Some(advisories) = multiseat_rejection {
            multi_seat_rejections.insert(role.clone(), advisories);
        }
        pre_eq_plugins.insert(role.clone(), mark_plugins_stage(chain.plugins, "pre_route"));
        pre_eq_initial_curves.insert(role.clone(), ch_result.initial_curve);
        optimizer_evidence_by_channel.insert(role.clone(), ch_result.optimizer_evidence);
        linearized_curves.insert(role, ch_result.final_curve);
    }

    // Main and sub Pre-EQ are independent after level alignment. Both branches
    // join before crossover selection and bass-management routing.
    {
        let (chain, ch_result, multiseat_rejection) = sub_pre_eq_output?;
        if let Some(advisories) = multiseat_rejection {
            multi_seat_rejections.insert(sub_role.clone(), advisories);
        }
        pre_eq_plugins.insert(
            sub_role.clone(),
            mark_plugins_stage(chain.plugins, "pre_route"),
        );
        pre_eq_initial_curves.insert(sub_role.clone(), ch_result.initial_curve);
        optimizer_evidence_by_channel.insert(sub_role.clone(), ch_result.optimizer_evidence);
        linearized_curves.insert(sub_role.clone(), ch_result.final_curve);
    }
    workflow_stage_event(
        &mut assembly.stage_callback,
        PipelineStepId::GenericChannelOptimization,
        PipelineStepStatus::Completed,
        "Optimized home-cinema channels",
        0.90,
    )?;
    workflow_stage_event(
        &mut assembly.stage_callback,
        PipelineStepId::TopologyWorkflowExecution,
        PipelineStepStatus::InProgress,
        "Optimizing bass-management crossover and routing",
        0.91,
    )?;

    let mut aligned_pre_eq_curves: HashMap<String, Curve> = HashMap::new();
    for role in main_roles {
        let mut c = linearized_curves[role].clone();
        let g = *gains.get(role).unwrap_or(&0.0);
        for s in c.spl.iter_mut() {
            *s += g;
        }
        aligned_pre_eq_curves.insert(role.clone(), c);
    }
    {
        let mut c = linearized_curves[&sub_role].clone();
        let g = *gains.get(&sub_role).unwrap_or(&0.0);
        for s in c.spl.iter_mut() {
            *s += g;
        }
        aligned_pre_eq_curves.insert(sub_role.clone(), c);
    }

    // 3. Bass-managed virtual main
    let crossover_grid = aligned_pre_eq_curves[&main_roles[0]].freq.clone();
    let aligned_main_phase_curves: Vec<Curve> = main_roles
        .iter()
        .map(|role| {
            autoeq_measurements::read::interpolate_log_space(
                &crossover_grid,
                &aligned_pre_eq_curves[role],
            )
        })
        .collect();
    let main_refs: Vec<&Curve> = aligned_main_phase_curves.iter().collect();
    let sub_curve_aligned = autoeq_measurements::read::interpolate_log_space(
        &crossover_grid,
        &aligned_pre_eq_curves[&sub_role],
    );
    let sub_curve = &sub_curve_aligned;
    // `load_source` intentionally power-averages multi-seat magnitudes and
    // drops phase. Crossover timing must instead use one synchronously
    // measured seat (the configured primary seat), just as MSO does.
    let primary_seat = config
        .optimizer
        .multi_seat
        .as_ref()
        .map(|multi_seat| multi_seat.primary_seat)
        .unwrap_or(0);
    let measured_main_curves: Vec<Curve> = main_roles
        .iter()
        .map(|role| {
            let source = resolve_single_source(role, config, sys)?;
            let individual = load_source_individual_with_frequency_samples(
                source,
                assembly.frequency_samples,
            )
                .map_err(|error| AutoeqError::InvalidMeasurement {
                    message: error.to_string(),
                })?;
            let index = if individual.len() == 1 { 0 } else { primary_seat };
            let mut curve = individual.get(index).cloned().ok_or_else(|| {
                AutoeqError::InvalidConfiguration {
                    message: format!(
                        "primary seat {primary_seat} unavailable for home-cinema channel '{role}' with {} measurement(s)",
                        individual.len()
                    ),
                }
            })?;
            let gain = *gains.get(role).unwrap_or(&0.0);
            curve.spl.mapv_inplace(|spl| spl + gain);
            Ok(autoeq_measurements::read::interpolate_log_space(
                &crossover_grid,
                &curve,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut measured_phase_check_refs: Vec<&Curve> = measured_main_curves.iter().collect();
    measured_phase_check_refs.push(sub_curve);
    let mut phase_check_refs = main_refs.clone();
    phase_check_refs.push(sub_curve);
    let measured_phase_available = all_curves_have_usable_phase(&measured_phase_check_refs);
    let processed_phase_available = all_curves_have_usable_phase(&phase_check_refs);
    let measured_grid_available = all_curves_share_frequency_grid(&measured_phase_check_refs);
    let processed_grid_available = all_curves_share_frequency_grid(&phase_check_refs);
    let shared_grid_available = measured_grid_available && processed_grid_available;
    let phase_available =
        measured_phase_available && processed_phase_available && shared_grid_available;
    let mut optimization_advisories = Vec::new();
    if !measured_phase_available || !processed_phase_available {
        optimization_advisories.push("missing_phase_crossover_alignment_skipped".to_string());
        let mut missing_roles: Vec<_> = main_roles
            .iter()
            .zip(&main_refs)
            .filter_map(|(role, curve)| {
                (!roomeq_engine::topology::curve_has_usable_phase(curve)).then_some(role.as_str())
            })
            .collect();
        if !roomeq_engine::topology::curve_has_usable_phase(sub_curve) {
            missing_roles.push(sub_role.as_str());
        }
        optimization_advisories.push(format!(
            "missing_phase_channels:{}",
            missing_roles.join(",")
        ));
    } else if !shared_grid_available {
        optimization_advisories
            .push("frequency_grid_mismatch_crossover_alignment_skipped".to_string());
        optimization_advisories.push(format!(
            "crossover_grid_status:measured={measured_grid_available},processed={processed_grid_available}"
        ));
    }
    // Programme channels are independent inputs. Their phases must not enter
    // a shared tonal target; source-paired crossover sums are optimized below.
    let virtual_main = average_mains_magnitude(&main_refs);

    // 4. Crossover optimization between virtual main and physical bass output
    let final_xover_type = select_bass_management_crossover_type(
        xover_type_str,
        &virtual_main,
        sub_curve,
        est_xo,
        sample_rate,
    );
    let xover_type_str = final_xover_type.as_str();
    let crossover_type_enum: roomeq_engine::loss::CrossoverType = xover_type_str
        .parse()
        .map_err(|e: String| AutoeqError::InvalidConfiguration { message: e })?;

    let (fixed_freqs, range_opt) = if xover_config.frequency.is_some() {
        (Some(vec![est_xo]), None)
    } else {
        (None, Some((min_xo, max_xo)))
    };

    let mut xo_optimizer_config = config.optimizer.clone();
    xo_optimizer_config.min_db = 0.0;
    xo_optimizer_config.max_db = 0.0;

    let objective_before_curve = predict_bass_management_sum(
        &virtual_main,
        sub_curve,
        xover_type_str,
        est_xo,
        sample_rate,
        0.0,
        0.0,
        0.0,
        0.0,
        false,
    );
    let objective_before = bass_management_objective(objective_before_curve.as_ref(), est_xo);

    let (main_gain_post, main_delay_raw, sub_gain_raw, sub_delay_raw, sub_inverted, final_xo_freq) =
        if phase_available && roomeq_engine::topology::curve_has_usable_phase(&virtual_main) {
            let (xo_gains, xo_delays, xo_freqs, _, inversions) = crossover::optimize_crossover(
                vec![virtual_main.clone(), sub_curve.clone()],
                crossover_type_enum,
                sample_rate,
                &xo_optimizer_config,
                fixed_freqs,
                range_opt,
            )
            .map_err(|e| AutoeqError::OptimizationFailed {
                message: e.to_string(),
            })?;

            (
                xo_gains[0],
                xo_delays[0],
                xo_gains[1],
                xo_delays[1],
                inversions[1],
                xo_freqs[0],
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, false, est_xo)
        };
    let (main_delay_post, sub_delay_post) =
        normalize_crossover_delays(main_delay_raw, sub_delay_raw);
    let sub_gain_post = sub_gain_raw;

    info!(
        "  Crossover Optimized: Freq={:.1} Hz, Main Gain={:.2}, Sub Gain={:.2}, Main Delay={:.2}, Sub Delay={:.2}",
        final_xo_freq, main_gain_post, sub_gain_post, main_delay_post, sub_delay_post
    );

    let mut group_results_by_id = if bass_management
        .as_ref()
        .map(|bm| bm.config.optimize_groups)
        .unwrap_or(true)
    {
        optimize_home_cinema_group_crossovers(
            config,
            main_roles,
            &aligned_curves,
            &aligned_pre_eq_curves,
            &sub_role,
            xover_config,
            sample_rate,
            bass_management.as_ref(),
        )?
    } else {
        engine_home_cinema::bass_management_groups(config, None)
            .into_iter()
            .map(|group| (group.group_id.clone(), group))
            .collect()
    };

    // 5. Apply crossover filters
    let apply_chain = |curve: &Curve,
                       xover_type: &str,
                       xover_freq: f64,
                       is_lowpass: bool,
                       gain: f64,
                       delay: f64,
                       invert: bool|
     -> Curve {
        let mut c = apply_crossover_response_to_curve(
            curve,
            xover_type,
            xover_freq,
            sample_rate,
            is_lowpass,
        );
        for s in c.spl.iter_mut() {
            *s += gain;
        }
        apply_delay_and_polarity_to_curve(&c, delay, invert)
    };

    let mut main_post_curves = HashMap::new();
    for role in main_roles {
        let group_id =
            engine_home_cinema::group_id_for_role(engine_home_cinema::role_for_channel(role));
        let group = group_results_by_id.get(group_id);
        let role_xover_type = group
            .map(|g| g.crossover_type.as_str())
            .unwrap_or(xover_type_str);
        let role_xover_freq = group
            .and_then(|g| g.selected_crossover_hz)
            .unwrap_or(final_xo_freq);
        let role_main_delay = group.map(|g| g.main_delay_ms).unwrap_or(main_delay_post);
        let post = apply_chain(
            &aligned_pre_eq_curves[role],
            role_xover_type,
            role_xover_freq,
            false,
            main_gain_post,
            role_main_delay,
            false,
        );
        main_post_curves.insert(role.clone(), post);
    }
    let preliminary_sub_output_results = bass_management_sub_output_results(
        &sub_role,
        sub_preprocess.drivers.as_deref(),
        sub_gain_post,
        &sub_sys.config,
    );
    let sub_output_base_curves: HashMap<String, Curve> = sub_preprocess
        .drivers
        .as_ref()
        .map(|drivers| {
            let combined_initial = aligned_curves.get(&sub_role);
            let combined_final = aligned_pre_eq_curves.get(&sub_role);
            let sub_alignment_gain = *gains.get(&sub_role).unwrap_or(&0.0);
            drivers
                .iter()
                .filter_map(|driver| {
                    driver.initial_curve.as_ref().map(|curve| {
                        let mut aligned_driver = curve.clone();
                        if sub_alignment_gain.abs() > f64::EPSILON {
                            for spl in aligned_driver.spl.iter_mut() {
                                *spl += sub_alignment_gain;
                            }
                        }
                        let corrected = match (combined_initial, combined_final) {
                            (Some(initial_curve), Some(final_curve)) => {
                                apply_curve_delta_to_reference_curve(
                                    &aligned_driver,
                                    initial_curve,
                                    final_curve,
                                )
                            }
                            _ => aligned_driver,
                        };
                        (driver.name.clone(), corrected)
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let preliminary_bass_management_optimization = joint_bass_management_report_from_parts(
        &group_results_by_id.values().cloned().collect::<Vec<_>>(),
        &[],
        &preliminary_sub_output_results,
    );
    let preliminary_bass_routing_graph = engine_home_cinema::bass_management_routing_graph(
        config,
        Some(&preliminary_bass_management_optimization),
    );
    let sub_post_initial = if let Some(graph) = preliminary_bass_routing_graph.as_ref()
        && let Some(route_predicted_sub) = predict_bass_bus_curve_from_routes(
            &aligned_pre_eq_curves[&sub_role],
            graph,
            &sub_output_base_curves,
            &aligned_pre_eq_curves[&sub_role],
            sample_rate,
        ) {
        route_predicted_sub
    } else {
        apply_chain(
            &aligned_pre_eq_curves[&sub_role],
            xover_type_str,
            final_xo_freq,
            true,
            sub_gain_post,
            sub_delay_post,
            sub_inverted,
        )
    };

    // Re-align sub level post-crossover (use first main as reference)
    let ref_main_post = &main_post_curves[&main_roles[0]];
    let main_freqs_f32: Vec<f32> = ref_main_post.freq.iter().map(|&f| f as f32).collect();
    let main_spl_f32: Vec<f32> = ref_main_post.spl.iter().map(|&s| s as f32).collect();
    let sub_freqs_f32: Vec<f32> = sub_post_initial.freq.iter().map(|&f| f as f32).collect();
    let sub_spl_f32: Vec<f32> = sub_post_initial.spl.iter().map(|&s| s as f32).collect();

    let main_mean = math_audio_dsp::analysis::compute_average_response(
        &main_freqs_f32,
        &main_spl_f32,
        Some((main_alignment_band.0 as f32, main_alignment_band.1 as f32)),
    ) as f64;
    let sub_mean = math_audio_dsp::analysis::compute_average_response(
        &sub_freqs_f32,
        &sub_spl_f32,
        Some((
            20.0,
            preliminary_bass_routing_graph
                .as_ref()
                .map(|graph| bass_route_upper_frequency_hz(Some(graph), final_xo_freq))
                .unwrap_or(final_xo_freq) as f32,
        )),
    ) as f64;

    let sub_correction = 0.0;
    info!(
        "  Physical sub level retained: Main={:.2} dB, Sub={:.2} dB, Common tonal correction={:+.2} dB",
        main_mean, sub_mean, sub_correction
    );

    let lfe_physical_gain = bass_management
        .as_ref()
        .filter(|bm| bm.config.apply_lfe_gain_to_chain)
        .map(|bm| bm.config.lfe_playback_gain_db)
        .unwrap_or(0.0);
    let requested_sub_gain = sub_gain_post + sub_correction + lfe_physical_gain;
    let (sub_gain_post, mut sub_gain_limited) =
        engine_home_cinema::limited_sub_gain(requested_sub_gain, bass_management.as_ref());
    if sub_gain_limited {
        log::warn!(
            "  Bass management limited sub gain from {:+.2} dB to {:+.2} dB for headroom",
            requested_sub_gain,
            sub_gain_post
        );
        optimization_advisories.push("sub_gain_limited_for_headroom".to_string());
    }
    let mut sub_post = sub_post_initial.clone();
    for s in sub_post.spl.iter_mut() {
        *s += sub_gain_post - sub_gain_raw;
    }
    let objective_after_curve = predict_bass_management_sum(
        &virtual_main,
        sub_curve,
        xover_type_str,
        final_xo_freq,
        sample_rate,
        main_gain_post,
        sub_gain_post,
        main_delay_post,
        sub_delay_post,
        sub_inverted,
    );
    let objective_after = bass_management_objective(objective_after_curve.as_ref(), final_xo_freq);
    if optimization_advisories.is_empty() {
        optimization_advisories.push("ok".to_string());
    }
    let mut sub_output_results = bass_management_sub_output_results(
        &sub_role,
        sub_preprocess.drivers.as_deref(),
        sub_gain_post,
        &sub_sys.config,
    );
    if limit_bass_management_sub_output_gains(&mut sub_output_results, bass_management.as_ref()) {
        sub_gain_limited = true;
        optimization_advisories.retain(|existing| existing != "ok");
        if !optimization_advisories.contains(&"sub_gain_limited_for_headroom".to_string()) {
            optimization_advisories.push("sub_gain_limited_for_headroom".to_string());
        }
    }
    let optimize_source_routes = phase_available
        && bass_management
            .as_ref()
            .map(|bm| bm.config.optimize_groups)
            .unwrap_or(true);
    let baseline_reason = if !phase_available {
        "source_route_optimizer_skipped_missing_phase"
    } else if !optimize_source_routes {
        "source_route_optimization_disabled"
    } else {
        "source_route_optimizer_baseline"
    };
    let mut source_results = engine_bass_management::baseline_bass_management_source_reports(
        main_roles,
        &aligned_pre_eq_curves,
        &group_results_by_id,
        &sub_output_results,
        sub_preprocess.drivers.as_deref(),
        &sub_role,
        sample_rate,
        baseline_reason,
    );
    let sub_output_advisories = if optimize_source_routes {
        optimize_bass_management_joint_solution(
            config,
            main_roles,
            &aligned_curves,
            &aligned_pre_eq_curves,
            &mut group_results_by_id,
            &mut source_results,
            &mut sub_output_results,
            sub_preprocess.drivers.as_deref(),
            &sub_role,
            sample_rate,
        )
    } else {
        vec![baseline_reason.to_string()]
    };
    for advisory in sub_output_advisories {
        optimization_advisories.retain(|existing| existing != "ok");
        if !optimization_advisories.contains(&advisory) {
            optimization_advisories.push(advisory);
        }
    }
    if limit_bass_management_sub_output_gains(&mut sub_output_results, bass_management.as_ref()) {
        sub_gain_limited = true;
        optimization_advisories.retain(|existing| existing != "ok");
        if !optimization_advisories.contains(&"sub_gain_limited_for_headroom".to_string()) {
            optimization_advisories.push("sub_gain_limited_for_headroom".to_string());
        }
    }

    // Joint bass-management optimization updates group crossover frequency,
    // type, and delay. Re-render the main routes from that accepted solution
    // before post-EQ and export so reported curves and the canonical DSP graph
    // describe the same serial processing.
    for role in main_roles {
        let group_id =
            engine_home_cinema::group_id_for_role(engine_home_cinema::role_for_channel(role));
        let group = group_results_by_id.get(group_id);
        let role_xover_type = group
            .map(|group| group.crossover_type.as_str())
            .unwrap_or(xover_type_str);
        let role_xover_freq = group
            .and_then(|group| group.selected_crossover_hz)
            .unwrap_or(final_xo_freq);
        let role_main_delay = engine_home_cinema::resolved_source_route_settings(
            role,
            group_id,
            Some(&joint_bass_management_report_from_parts(
                &group_results_by_id.values().cloned().collect::<Vec<_>>(),
                &source_results,
                &sub_output_results,
            )),
        )
        .main_delay_ms;
        main_post_curves.insert(
            role.clone(),
            apply_chain(
                &aligned_pre_eq_curves[role],
                role_xover_type,
                role_xover_freq,
                false,
                main_gain_post,
                role_main_delay,
                false,
            ),
        );
    }

    let route_applied_sub_gain_db = sub_output_results
        .iter()
        .map(|output| output.gain_db)
        .fold(f64::NEG_INFINITY, f64::max);
    let route_applied_sub_gain_db = if route_applied_sub_gain_db.is_finite() {
        route_applied_sub_gain_db
    } else {
        sub_gain_post
    };
    let primary_group = group_results_by_id
        .get("lcr")
        .or_else(|| group_results_by_id.values().next());
    let metadata_main_delay_ms = primary_group
        .map(|group| group.main_delay_ms)
        .unwrap_or(main_delay_post);
    let metadata_sub_delay_ms = primary_group
        .map(|group| group.bass_route_delay_ms)
        .unwrap_or(sub_delay_post);
    let metadata_sub_inverted = primary_group
        .map(|group| group.polarity_inverted)
        .unwrap_or(sub_inverted);
    let metadata_crossover_type = primary_group
        .map(|group| group.crossover_type.clone())
        .unwrap_or_else(|| xover_type_str.to_string());
    let metadata_crossover_hz = primary_group
        .and_then(|group| group.selected_crossover_hz)
        .unwrap_or(final_xo_freq);
    let aggregate_objective_before = group_results_by_id
        .values()
        .filter_map(|group| group.objective_before)
        .reduce(|a, b| a + b)
        .or(objective_before);
    let aggregate_objective_after = group_results_by_id
        .values()
        .filter_map(|group| group.objective_after)
        .reduce(|a, b| a + b)
        .or(objective_after);
    let mut bass_management_optimization = engine_home_cinema::BassManagementOptimizationReport {
        applied: phase_available,
        phase_required: true,
        phase_available,
        configured_crossover_hz: Some(est_xo),
        optimized_crossover_hz: Some(metadata_crossover_hz),
        crossover_range_hz: xover_config.frequency_range,
        crossover_type: metadata_crossover_type,
        main_delay_ms: metadata_main_delay_ms,
        sub_delay_ms: metadata_sub_delay_ms,
        relative_sub_delay_ms: metadata_sub_delay_ms - metadata_main_delay_ms,
        sub_polarity_inverted: metadata_sub_inverted,
        requested_sub_gain_db: requested_sub_gain,
        applied_sub_gain_db: route_applied_sub_gain_db,
        gain_limited: sub_gain_limited,
        estimated_bass_bus_peak_gain_db: None,
        objective_before: aggregate_objective_before,
        objective_after: aggregate_objective_after,
        group_results: group_results_by_id.values().cloned().collect(),
        source_results,
        sub_output_results,
        advisories: optimization_advisories,
    };
    let mut bass_routing_graph = engine_home_cinema::bass_management_routing_graph(
        config,
        Some(&bass_management_optimization),
    );
    let deprecated_peak_gain_extra = if bass_management_optimization.sub_output_results.is_empty() {
        sub_gain_post
    } else {
        0.0
    };
    bass_management_optimization.estimated_bass_bus_peak_gain_db =
        engine_home_cinema::estimated_bass_bus_peak_gain_db_for_config(
            config,
            bass_routing_graph.as_ref(),
            deprecated_peak_gain_extra,
            sample_rate,
        );
    let bass_route_upper_hz =
        bass_route_upper_frequency_hz(bass_routing_graph.as_ref(), final_xo_freq);
    let (representative_bass_route_type, representative_bass_route_hz) =
        representative_bass_route_signature(
            bass_routing_graph.as_ref(),
            xover_type_str,
            final_xo_freq,
        );
    if let Some(graph) = bass_routing_graph.as_ref()
        && let Some(route_predicted_sub) = predict_bass_bus_curve_from_routes(
            &aligned_pre_eq_curves[&sub_role],
            graph,
            &sub_output_base_curves,
            &aligned_pre_eq_curves[&sub_role],
            sample_rate,
        )
    {
        sub_post = route_predicted_sub;
    }

    // 6. Post-EQ
    let mut post_eq_filters = HashMap::new();
    let main_post_max_freq = config.optimizer.max_freq;
    let total_post_eq_passes = main_roles.len() + 1;

    for (role_index, role) in main_roles.iter().enumerate() {
        let role_progress_base = 0.91 + (role_index as f64 / total_post_eq_passes as f64) * 0.03;
        workflow_stage_event(
            &mut assembly.stage_callback,
            PipelineStepId::TopologyWorkflowExecution,
            PipelineStepStatus::InProgress,
            &format!("Post-EQ for {role}"),
            role_progress_base,
        )?;
        let mut opt_config = config.optimizer.clone();
        let group_id =
            engine_home_cinema::group_id_for_role(engine_home_cinema::role_for_channel(role));
        let role_xover_freq = group_results_by_id
            .get(group_id)
            .and_then(|g| g.selected_crossover_hz)
            .unwrap_or(final_xo_freq);
        let requested_min_freq = role_xover_freq + 20.0;
        opt_config.min_freq = opt_config.min_freq.max(requested_min_freq);
        if opt_config.min_freq >= opt_config.max_freq {
            log::warn!(
                "  Skipping {role} Post-EQ: crossover guard band starts at {:.1} Hz, outside configured optimization band [{:.1}, {:.1}] Hz",
                requested_min_freq,
                config.optimizer.min_freq,
                config.optimizer.max_freq
            );
            post_eq_filters.insert(role.clone(), Vec::new());
            continue;
        }

        let post_curve = &main_post_curves[role];
        let post_eq_callback = workflow_progress_callback(
            &assembly.progress_factory,
            &format!("Post-EQ {role}"),
            role_index,
            total_post_eq_passes,
            opt_config.max_iter,
        );
        let mut post_eq_result = run_post_eq(
            post_curve,
            &opt_config,
            config.target_curve.as_ref(),
            sample_rate,
            post_eq_callback,
        )?;
        let filters = post_eq_result.filters;

        // Evaluate acceptance over the same band reported in the final channel
        // score. The optimizer keeps its 20 Hz crossover guard band, but a
        // candidate must not damage that guard band enough to regress the
        // published response score.
        let pre = compute_flat_loss(post_curve, role_xover_freq, main_post_max_freq);
        let eq_resp =
            response::compute_peq_complex_response(&filters, &post_curve.freq, sample_rate);
        let post_curve_after = response::apply_complex_response(post_curve, &eq_resp);
        let post = compute_flat_loss(&post_curve_after, role_xover_freq, main_post_max_freq);
        if post < pre {
            optimizer_evidence_by_channel
                .entry(role.clone())
                .or_default()
                .append(&mut post_eq_result.optimizer_evidence);
            post_eq_filters.insert(role.clone(), filters);
        } else {
            for evidence in &mut post_eq_result.optimizer_evidence {
                evidence.selected_for_output = false;
            }
            optimizer_evidence_by_channel
                .entry(role.clone())
                .or_default()
                .append(&mut post_eq_result.optimizer_evidence);
            log::warn!(
                "  {} Post-EQ discarded: score regressed from {:.4} to {:.4}",
                role,
                pre,
                post
            );
            post_eq_filters.insert(role.clone(), Vec::new());
        }
    }

    // Sub Post-EQ
    {
        let sub_progress_base =
            0.91 + (main_roles.len() as f64 / total_post_eq_passes as f64) * 0.03;
        workflow_stage_event(
            &mut assembly.stage_callback,
            PipelineStepId::TopologyWorkflowExecution,
            PipelineStepStatus::InProgress,
            &format!("Post-EQ for {sub_role}"),
            sub_progress_base,
        )?;
        let mut opt_config = config.optimizer.clone();
        opt_config.max_freq = bass_route_upper_hz - 20.0;
        let sub_min_score = config.optimizer.min_freq.max(20.0);
        let sub_callback = workflow_progress_callback(
            &assembly.progress_factory,
            &format!("Post-EQ {sub_role}"),
            main_roles.len(),
            total_post_eq_passes,
            opt_config.max_iter,
        );
        let mut post_eq_result = run_post_eq(
            &sub_post,
            &opt_config,
            config.target_curve.as_ref(),
            sample_rate,
            sub_callback,
        )?;
        let filters = post_eq_result.filters;

        let pre = compute_flat_loss(&sub_post, sub_min_score, bass_route_upper_hz);
        let eq_resp = response::compute_peq_complex_response(&filters, &sub_post.freq, sample_rate);
        let sub_after_eq = response::apply_complex_response(&sub_post, &eq_resp);
        let post = compute_flat_loss(&sub_after_eq, sub_min_score, bass_route_upper_hz);
        if post < pre {
            optimizer_evidence_by_channel
                .entry(sub_role.clone())
                .or_default()
                .append(&mut post_eq_result.optimizer_evidence);
            post_eq_filters.insert(sub_role.clone(), filters);
        } else {
            for evidence in &mut post_eq_result.optimizer_evidence {
                evidence.selected_for_output = false;
            }
            optimizer_evidence_by_channel
                .entry(sub_role.clone())
                .or_default()
                .append(&mut post_eq_result.optimizer_evidence);
            log::warn!(
                "  Sub Post-EQ discarded: score regressed from {:.4} to {:.4}",
                pre,
                post
            );
        }
    }

    // 7. Build output chains
    let mut channel_chains = HashMap::new();

    for role in main_roles {
        let mut plugins = Vec::new();
        let align_gain = *gains.get(role).unwrap_or(&0.0);
        if align_gain.abs() > 0.01 {
            plugins.push(mark_plugin_stage(
                output::create_gain_plugin(align_gain),
                "pre_route",
            ));
        }

        if let Some(stack) = pre_eq_plugins.get(role) {
            plugins.extend(stack.clone());
        }

        let group_id =
            engine_home_cinema::group_id_for_role(engine_home_cinema::role_for_channel(role));
        let group = group_results_by_id.get(group_id);
        let role_xover_type = group
            .map(|g| g.crossover_type.as_str())
            .unwrap_or(xover_type_str);
        let role_xover_freq = group
            .and_then(|g| g.selected_crossover_hz)
            .unwrap_or(final_xo_freq);
        let role_main_delay = engine_home_cinema::resolved_source_route_settings(
            role,
            group_id,
            Some(&bass_management_optimization),
        )
        .main_delay_ms;

        plugins.push(mark_route_owned_plugin(output::create_crossover_plugin(
            role_xover_type,
            role_xover_freq,
            "high",
        )));

        if main_gain_post.abs() > 0.01 {
            plugins.push(mark_route_owned_plugin(output::create_gain_plugin(
                main_gain_post,
            )));
        }

        if role_main_delay.abs() > 0.01 {
            plugins.push(mark_route_owned_plugin(output::create_delay_plugin(
                role_main_delay,
            )));
        }

        let eqs = post_eq_filters.get(role);
        if let Some(e) = eqs
            && !e.is_empty()
        {
            plugins.push(mark_plugin_stage(
                output::create_labeled_eq_plugin(e, "post_eq"),
                "post_route",
            ));
        }

        let intermediate = &main_post_curves[role];
        let final_curve_obj = if let Some(e) = eqs {
            if !e.is_empty() {
                let resp =
                    response::compute_peq_complex_response(e, &intermediate.freq, sample_rate);
                response::apply_complex_response(intermediate, &resp)
            } else {
                intermediate.clone()
            }
        } else {
            intermediate.clone()
        };

        // The canonical DSP graph owns level alignment, while its PEQ was
        // designed against the generic channel workflow's prepared input.
        // Preserve that exact input so reconstructing the final response does
        // not apply alignment twice or evaluate PEQ against a different curve.
        let initial_data: CurveData = (&pre_eq_initial_curves[role]).into();
        let final_data: CurveData = (&final_curve_obj).into();
        let eq_resp = output::compute_eq_response(&initial_data, &final_data);
        let chain = ChannelDspChain {
            channel: role.clone(),
            plugins,
            drivers: None,
            initial_curve: Some(initial_data),
            final_curve: Some(final_data),
            eq_response: Some(eq_resp),
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
            target_curve: None,
        };
        channel_chains.insert(role.clone(), chain);
    }

    let mut sub_plugins = Vec::new();
    let sub_align_gain = *gains.get(&sub_role).unwrap_or(&0.0);
    if sub_align_gain.abs() > 0.01 {
        sub_plugins.push(mark_plugin_stage(
            output::create_gain_plugin(sub_align_gain),
            "pre_route",
        ));
    }

    if let Some(stack) = pre_eq_plugins.get(&sub_role) {
        sub_plugins.extend(stack.clone());
    }

    sub_plugins.push(mark_route_owned_plugin(output::create_crossover_plugin(
        &representative_bass_route_type,
        representative_bass_route_hz,
        "low",
    )));

    if metadata_sub_inverted || route_applied_sub_gain_db.abs() > 0.01 {
        sub_plugins.push(mark_route_owned_plugin(
            output::create_gain_plugin_with_invert(
                route_applied_sub_gain_db,
                metadata_sub_inverted,
            ),
        ));
    }

    if metadata_sub_delay_ms.abs() > 0.01 {
        sub_plugins.push(mark_route_owned_plugin(output::create_delay_plugin(
            metadata_sub_delay_ms,
        )));
    }

    let sub_eqs = post_eq_filters.get(&sub_role);
    if let Some(e) = sub_eqs
        && !e.is_empty()
    {
        sub_plugins.push(mark_plugin_stage(
            output::create_labeled_eq_plugin(e, "post_eq"),
            "post_route",
        ));
    }

    let final_sub_curve = if let Some(e) = sub_eqs {
        if !e.is_empty() {
            let resp = response::compute_peq_complex_response(e, &sub_post.freq, sample_rate);
            response::apply_complex_response(&sub_post, &resp)
        } else {
            sub_post.clone()
        }
    } else {
        sub_post.clone()
    };

    let sub_output_by_role: HashMap<String, engine_home_cinema::BassManagementSubOutputReport> =
        bass_management_optimization
            .sub_output_results
            .iter()
            .cloned()
            .map(|output| (output.output_role.clone(), output))
            .collect();
    let driver_chains = sub_preprocess.drivers.as_ref().map(|drivers| {
        drivers
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let mut driver_plugins = Vec::new();
                let output_settings = sub_output_by_role.get(&d.name);
                let gain_db = output_settings
                    .map(|output| output.gain_db - route_applied_sub_gain_db)
                    .unwrap_or(d.gain);
                let delay_ms = output_settings
                    .map(|output| output.delay_ms)
                    .unwrap_or(d.delay);
                let inverted = output_settings
                    .map(|output| output.polarity_inverted)
                    .unwrap_or(d.inverted);
                if inverted || gain_db.abs() > 0.01 {
                    if inverted {
                        driver_plugins.push(mark_plugin_stage(
                            output::create_gain_plugin_with_invert(gain_db, true),
                            "post_route",
                        ));
                    } else {
                        driver_plugins.push(mark_plugin_stage(
                            output::create_gain_plugin(gain_db),
                            "post_route",
                        ));
                    }
                }
                if delay_ms.abs() > 0.001 {
                    driver_plugins.push(mark_plugin_stage(
                        output::create_delay_plugin(delay_ms),
                        "post_route",
                    ));
                }
                let driver_curve = d
                    .initial_curve
                    .as_ref()
                    .map(output::extend_curve_to_full_range)
                    .map(|c| (&c).into());
                DriverDspChain {
                    name: d.name.clone(),
                    index: i,
                    plugins: driver_plugins,
                    initial_curve: driver_curve,
                }
            })
            .collect()
    });

    // The sub PEQ uses the preprocessed combined measurement as its input.
    let sub_initial_data: CurveData = (&pre_eq_initial_curves[&sub_role]).into();
    let sub_final_data: CurveData = (&final_sub_curve).into();
    let sub_eq_resp = output::compute_eq_response(&sub_initial_data, &sub_final_data);
    let sub_chain = ChannelDspChain {
        channel: sub_role.clone(),
        plugins: sub_plugins,
        drivers: driver_chains,
        initial_curve: Some(sub_initial_data),
        final_curve: Some(sub_final_data),
        eq_response: Some(sub_eq_resp),
        pre_ir: None,
        post_ir: None,
        fir_temporal_masking: None,
        direct_early_late_correction: None,
        target_curve: None,
    };
    channel_chains.insert(sub_role.clone(), sub_chain);

    let post_dsp_input_trims = if let Some(graph) = bass_routing_graph.as_mut() {
        calibrate_post_dsp_input_levels(
            config,
            main_roles,
            &sub_role,
            main_alignment_band,
            sample_rate,
            output_dir,
            &mut channel_chains,
            graph,
        )?
    } else {
        HashMap::new()
    };
    for (role, trim_db) in &post_dsp_input_trims {
        info!(" Post-DSP input level trim '{}': {:+.2} dB", role, trim_db);
    }

    // 8. Compute scores
    let max_freq = config.optimizer.max_freq;
    let sub_min_score = config.optimizer.min_freq.max(20.0);
    let mut channel_results = HashMap::new();
    let mut pre_scores = Vec::new();
    let mut post_scores = Vec::new();

    for role in main_roles {
        let intermediate = &main_post_curves[role];
        let group_id =
            engine_home_cinema::group_id_for_role(engine_home_cinema::role_for_channel(role));
        let role_xover_freq = group_results_by_id
            .get(group_id)
            .and_then(|g| g.selected_crossover_hz)
            .unwrap_or(final_xo_freq);
        let pre_score = compute_flat_loss(&pre_eq_initial_curves[role], role_xover_freq, max_freq);
        let final_curve_obj = if let Some(e) = post_eq_filters.get(role) {
            if !e.is_empty() {
                let resp =
                    response::compute_peq_complex_response(e, &intermediate.freq, sample_rate);
                response::apply_complex_response(intermediate, &resp)
            } else {
                intermediate.clone()
            }
        } else {
            intermediate.clone()
        };
        let post_score = compute_flat_loss(&final_curve_obj, role_xover_freq, max_freq);

        pre_scores.push(pre_score);
        post_scores.push(post_score);
        channel_results.insert(
            role.clone(),
            ChannelOptimizationResult {
                name: role.clone(),
                pre_score,
                post_score,
                initial_curve: pre_eq_initial_curves[role].clone(),
                final_curve: final_curve_obj,
                biquads: post_eq_filters.get(role).cloned().unwrap_or_default(),
                fir_coeffs: None,
                optimizer_evidence: optimizer_evidence_by_channel
                    .remove(role)
                    .unwrap_or_default(),
            },
        );
    }

    {
        let pre_score = compute_flat_loss(
            &pre_eq_initial_curves[&sub_role],
            sub_min_score,
            bass_route_upper_hz,
        );
        let post_score = compute_flat_loss(&final_sub_curve, sub_min_score, bass_route_upper_hz);
        pre_scores.push(pre_score);
        post_scores.push(post_score);
        channel_results.insert(
            sub_role.clone(),
            ChannelOptimizationResult {
                name: sub_role.clone(),
                pre_score,
                post_score,
                initial_curve: pre_eq_initial_curves[&sub_role].clone(),
                final_curve: final_sub_curve.clone(),
                biquads: post_eq_filters.get(&sub_role).cloned().unwrap_or_default(),
                fir_coeffs: None,
                optimizer_evidence: optimizer_evidence_by_channel
                    .remove(&sub_role)
                    .unwrap_or_default(),
            },
        );
    }

    // Scores and result curves must describe the calibrated graph, not the
    // pre-calibration optimizer intermediates.
    for role in main_roles {
        let group_id =
            engine_home_cinema::group_id_for_role(engine_home_cinema::role_for_channel(role));
        let role_xover_freq = group_results_by_id
            .get(group_id)
            .and_then(|group| group.selected_crossover_hz)
            .unwrap_or(final_xo_freq);
        if let Some(final_data) = channel_chains
            .get(role)
            .and_then(|chain| chain.final_curve.clone())
            && let Some(result) = channel_results.get_mut(role)
        {
            let final_curve: Curve = final_data.into();
            result.post_score = compute_flat_loss(&final_curve, role_xover_freq, max_freq);
            result.final_curve = final_curve;
        }
    }
    if let Some(final_data) = channel_chains
        .get(&sub_role)
        .and_then(|chain| chain.final_curve.clone())
        && let Some(result) = channel_results.get_mut(&sub_role)
    {
        let final_curve: Curve = final_data.into();
        result.post_score = compute_flat_loss(&final_curve, sub_min_score, bass_route_upper_hz);
        result.final_curve = final_curve;
    }
    post_scores = channel_results
        .values()
        .map(|result| result.post_score)
        .collect();

    let avg_pre = pre_scores.iter().sum::<f64>() / pre_scores.len() as f64;
    let avg_post = post_scores.iter().sum::<f64>() / post_scores.len() as f64;

    info!(
        "Average pre-score: {:.4}, post-score: {:.4}",
        avg_pre, avg_post
    );

    let epa_cfg = config.optimizer.epa_config.clone().unwrap_or_default();
    let epa_per_channel = output::compute_epa_per_channel(&channel_chains, &epa_cfg);
    let epa_multichannel = output::compute_epa_multichannel(&channel_chains, &epa_cfg);
    let multi_seat_correction = Some(
        crate::home_cinema::multi_seat_correction_report_with_frequency_samples(
            config,
            &channel_results,
            Some(&multi_seat_rejections),
            assembly.frequency_samples,
        ),
    );
    workflow_stage_event(
        &mut assembly.stage_callback,
        PipelineStepId::TopologyWorkflowExecution,
        PipelineStepStatus::Completed,
        "Home-cinema bass-management topology complete",
        0.94,
    )?;

    let mut bass_management_report =
        engine_home_cinema::bass_management_report_with_optimization_and_sample_rate(
            config,
            Some(route_applied_sub_gain_db),
            sub_gain_limited,
            Some(bass_management_optimization),
            sample_rate,
        );
    if let Some(report) = bass_management_report.as_mut()
        && let Some(graph) = bass_routing_graph.as_ref()
    {
        report.routing_graph = Some(graph.clone());
        if let Some(effective) = bass_management.as_ref() {
            report.headroom_simulation = engine_home_cinema::simulate_bass_bus_headroom(
                Some(graph),
                &effective.config.headroom_model,
                effective.config.headroom_margin_db,
                sample_rate,
            );
        }
        if !report
            .advisory
            .contains("post_dsp_input_levels_aligned_down")
        {
            if report.advisory == "ok" {
                report.advisory = "post_dsp_input_levels_aligned_down".to_string();
            } else {
                report
                    .advisory
                    .push_str(";post_dsp_input_levels_aligned_down");
            }
        }
    }

    Ok(RoomOptimizationResult {
        channels: channel_chains,
        channel_results,
        combined_pre_score: avg_pre,
        combined_post_score: avg_post,
        metadata: OptimizationMetadata {
            pre_score: avg_pre,
            post_score: avg_post,
            algorithm: config.optimizer.algorithm.clone(),
            loss_type: Some(config.optimizer.loss_type.clone()),
            iterations: config.optimizer.max_iter,
            timestamp: chrono::Utc::now().to_rfc3339(),
            inter_channel_deviation: None,
            epa_per_channel,
            epa_multichannel,
            group_delay: None,
            mixed_phase_per_channel: None,
            perceptual_metrics: None,
            home_cinema_layout: Some(engine_home_cinema::analyze_layout(config)),
            multi_seat_coverage: Some(crate::home_cinema::multi_seat_coverage(config)),
            multi_seat_correction,
            bass_management: bass_management_report,
            timing_diagnostics: None,
            ctc: None,
            perceptual_policy: None,
            bootstrap_uncertainty: None,
            validation_bundle: None,
            supporting_source: None,
            correction_acceptance: None,
            optimizer_evidence: None,
            stage_outcomes: Vec::new(),
            effective_config: None,
        },
    })
}

#[cfg(test)]
mod post_dsp_level_tests {
    use super::{average_spl, calibrate_post_dsp_input_levels};
    use roomeq_engine::Curve;
    use roomeq_engine::bass_management::predict_bass_source_curve_from_routes;
    use roomeq_engine::topology::complex_sum_mains;
    use roomeq_model::{
        BassManagementMatrix, BassManagementRoute, BassManagementRoutingGraph, ChannelDspChain,
        CurveData,
    };
    use std::collections::HashMap;

    fn curve(level: f64) -> Curve {
        let frequencies = ndarray::array![20.0, 40.0, 80.0, 100.0, 200.0, 400.0];
        Curve {
            spl: ndarray::Array1::from_elem(frequencies.len(), level),
            phase: Some(ndarray::Array1::zeros(frequencies.len())),
            freq: frequencies,
            ..Curve::default()
        }
    }

    fn chain(name: &str, initial: Curve, final_curve: Option<Curve>) -> ChannelDspChain {
        ChannelDspChain {
            channel: name.to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: Some(CurveData::from(&initial)),
            final_curve: final_curve.as_ref().map(CurveData::from),
            eq_response: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
            target_curve: None,
        }
    }

    fn low_route(source: &str, source_index: usize) -> BassManagementRoute {
        BassManagementRoute {
            group_id: None,
            source_channel: source.to_string(),
            source_index,
            destination: "LFE".to_string(),
            destination_index: 2,
            pre_chain_channel: Some("LFE".to_string()),
            post_chain_channel: Some("LFE".to_string()),
            route_kind: if source == "LFE" {
                "lfe_lowpass_to_sub".to_string()
            } else {
                "redirected_bass_lowpass_to_sub".to_string()
            },
            crossover_type: "LR24".to_string(),
            high_pass_hz: None,
            low_pass_hz: Some(80.0),
            gain_db: 0.0,
            gain_linear: 1.0,
            matrix_gain: 1.0,
            delay_ms: 0.0,
            polarity_inverted: false,
        }
    }

    #[test]
    fn final_input_calibration_aligns_mains_and_lfe_down_only() {
        let sub = curve(60.0);
        let mut channels = HashMap::from([
            ("L".to_string(), chain("L", curve(70.0), Some(curve(70.0)))),
            ("R".to_string(), chain("R", curve(66.0), Some(curve(66.0)))),
            ("LFE".to_string(), chain("LFE", sub.clone(), None)),
        ]);
        let mut graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".to_string(),
            input_channels: vec!["L".to_string(), "R".to_string(), "LFE".to_string()],
            output_channels: vec!["L".to_string(), "R".to_string(), "LFE".to_string()],
            routes: vec![low_route("L", 0), low_route("R", 1), low_route("LFE", 2)],
            matrix: Some(BassManagementMatrix {
                input_channel_map: vec![0, 1, 2],
                output_channel_map: vec![2, 2, 2],
                matrix: vec![1.0, 1.0, 1.0],
                route_count: 3,
            }),
            input_trim_db: HashMap::new(),
            advisories: Vec::new(),
        };

        let trims = calibrate_post_dsp_input_levels(
            &roomeq_model::RoomConfig::default(),
            &["L".to_string(), "R".to_string()],
            "LFE",
            (100.0, 400.0),
            48_000.0,
            std::path::Path::new("."),
            &mut channels,
            &mut graph,
        )
        .unwrap();

        assert!(trims.values().all(|trim| *trim <= 1.0e-12));
        let common_sub = sub;
        let mut observed_means = Vec::new();
        for role in ["L", "R"] {
            let main: Curve = channels[role].final_curve.clone().unwrap().into();
            let bass =
                predict_bass_source_curve_from_routes(&common_sub, &graph, role, 48_000.0).unwrap();
            observed_means.push(average_spl(
                &complex_sum_mains(&[&main, &bass]),
                (100.0, 400.0),
            ));
        }
        let lfe =
            predict_bass_source_curve_from_routes(&common_sub, &graph, "LFE", 48_000.0).unwrap();
        observed_means.push(average_spl(&lfe, (20.0, 80.0)));
        let spread = observed_means
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            - observed_means.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(spread < 1.0e-5, "post-DSP level spread was {spread} dB");
        assert_eq!(graph.matrix.as_ref().unwrap().route_count, 3);
        assert!(
            graph
                .advisories
                .contains(&"post_dsp_input_levels_aligned_down".to_string())
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::executor_tests::{flat_curve, flat_curve_with_phase, make_assembly};
    use super::super::types::WorkflowExecutor;
    use super::{HomeCinemaExecutor, canonical_main_roles};
    use roomeq_model::{
        BassManagementConfig, CrossoverConfig, MeasurementSource, MultiMeasurementStrategy,
        MultiSeatConfig, OptimizerConfig, ProcessingMode, RoomConfig, SpeakerConfig,
        SubwooferStrategy, SubwooferSystemConfig, SupportingSourceConfig,
        SupportingSourceDecorrelation, SupportingSourceGroup, SystemConfig, SystemModel,
        TargetCurveConfig, default_config_version,
    };
    use std::collections::HashMap;

    #[test]
    fn canonical_main_roles_is_independent_of_map_insertion_order() {
        let mut first = SystemConfig::default();
        first.speakers.insert("Right".into(), "right".into());
        first.speakers.insert("LFE".into(), "sub".into());
        first.speakers.insert("Left".into(), "left".into());
        let mut second = SystemConfig::default();
        second.speakers.insert("Left".into(), "left".into());
        second.speakers.insert("Right".into(), "right".into());
        second.speakers.insert("LFE".into(), "sub".into());

        assert_eq!(
            canonical_main_roles(&first, "LFE"),
            canonical_main_roles(&second, "LFE")
        );
        assert_eq!(canonical_main_roles(&first, "LFE"), vec!["Left", "Right"]);
    }

    fn tiny_optimizer() -> OptimizerConfig {
        OptimizerConfig {
            processing_mode: ProcessingMode::LowLatency,
            num_filters: 1,
            max_iter: 20,
            population: 6,
            seed: Some(1),
            ..Default::default()
        }
    }

    fn stereo_speakers() -> HashMap<String, SpeakerConfig> {
        HashMap::from([
            (
                "left".to_string(),
                SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
            ),
            (
                "right".to_string(),
                SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
            ),
        ])
    }

    fn stereo_speakers_with_phase() -> HashMap<String, SpeakerConfig> {
        HashMap::from([
            (
                "left".to_string(),
                SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
            ),
            (
                "right".to_string(),
                SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
            ),
        ])
    }

    fn home_cinema_sys_with_sub() -> SystemConfig {
        SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([
                ("Left".to_string(), "left".to_string()),
                ("Right".to_string(), "right".to_string()),
                ("LFE".to_string(), "sub".to_string()),
            ]),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                crossover: Some("bass_xo".to_string()),
                mapping: HashMap::from([("sub".to_string(), "Left".to_string())]),
            }),
            bass_management: None,
            ..Default::default()
        }
    }

    fn home_cinema_no_sub_sys() -> SystemConfig {
        SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([
                ("Left".to_string(), "left".to_string()),
                ("Right".to_string(), "right".to_string()),
            ]),
            subwoofers: None,
            bass_management: None,
            ..Default::default()
        }
    }

    fn crossovers_fixed() -> HashMap<String, CrossoverConfig> {
        HashMap::from([(
            "bass_xo".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: Some(80.0),
                frequencies: None,
                frequency_range: None,
            },
        )])
    }

    fn room_config(
        speakers: HashMap<String, SpeakerConfig>,
        sys: &SystemConfig,
        optimizer: OptimizerConfig,
        crossovers: Option<HashMap<String, CrossoverConfig>>,
        target_curve: Option<TargetCurveConfig>,
    ) -> RoomConfig {
        RoomConfig {
            version: default_config_version(),
            system: Some(sys.clone()),
            speakers,
            crossovers,
            target_curve,
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        }
    }

    #[test]
    fn home_cinema_no_sub_with_target_curve_runs() {
        let sys = home_cinema_no_sub_sys();
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2_000.0;
        let config = room_config(
            stereo_speakers(),
            &sys,
            optimizer,
            None,
            Some(TargetCurveConfig::Predefined("flat".to_string())),
        );
        let mut assembly = make_assembly(&config, &sys);
        let result = HomeCinemaExecutor.execute(&mut assembly);
        assert!(
            result.is_ok(),
            "home-cinema no-sub with target curve should run: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.channels.len(), 2);
    }

    #[test]
    fn home_cinema_no_sub_multiseat_rejection_reports() {
        let sys = home_cinema_no_sub_sys();
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2_000.0;
        optimizer.multi_seat = Some(MultiSeatConfig {
            all_channel_enabled: true,
            all_channel_strategy: MultiMeasurementStrategy::SpatialRobustness,
            max_deviation_db: 0.001,
            ..Default::default()
        });

        let mut speakers = HashMap::new();
        let seat0 = flat_curve();
        let mut seat1 = flat_curve();
        for (index, spl) in seat1.spl.iter_mut().enumerate() {
            *spl += if index % 2 == 0 { 5.0 } else { -5.0 };
        }
        speakers.insert(
            "left".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![seat0, seat1])),
        );
        speakers.insert(
            "right".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
        );

        let config = room_config(speakers, &sys, optimizer, None, None);
        let mut assembly = make_assembly(&config, &sys);
        let result = HomeCinemaExecutor.execute(&mut assembly);
        assert!(
            result.is_ok(),
            "home-cinema no-sub multiseat rejection should recover: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.channels.len(), 2);
        let correction = result
            .metadata
            .multi_seat_correction
            .expect("correction report");
        assert!(
            correction
                .advisories
                .iter()
                .any(|a| a.contains("rejected") || a.contains("Left")),
            "rejection advisory should mention rejected channel: {:?}",
            correction.advisories
        );
    }

    #[test]
    fn home_cinema_with_sub_optimize_groups_disabled_runs() {
        let sys = home_cinema_sys_with_sub();
        let mut speakers = stereo_speakers_with_phase();
        speakers.insert(
            "sub".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
        );
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2_000.0;
        let config = RoomConfig {
            version: default_config_version(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: sys.speakers.clone(),
                subwoofers: sys.subwoofers.clone(),
                bass_management: Some(BassManagementConfig {
                    enabled: true,
                    optimize_groups: false,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            speakers,
            crossovers: Some(crossovers_fixed()),
            target_curve: None,
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        };
        let mut assembly = make_assembly(&config, config.system.as_ref().unwrap());
        let result = HomeCinemaExecutor.execute(&mut assembly);
        assert!(
            result.is_ok(),
            "optimize_groups=false should run: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.channels.len(), 3);
        let optimization = result
            .metadata
            .bass_management
            .expect("bass-management report")
            .optimization
            .expect("bass-management optimization report");
        assert_eq!(optimization.source_results.len(), 2);
        for source in optimization.source_results {
            assert!(!source.accepted);
            assert_eq!(source.objective_before, source.objective_after);
            assert!(source.objective_before.is_some_and(f64::is_finite));
            assert!(
                source
                    .advisories
                    .contains(&"source_route_optimization_disabled".to_string())
            );
        }
    }

    #[test]
    fn home_cinema_no_sub_with_supporting_source_runs() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut speakers = stereo_speakers();
        speakers.insert(
            "left_ss".to_string(),
            SpeakerConfig::SupportingSource(SupportingSourceGroup {
                name: "Left wide".to_string(),
                speaker_name: None,
                primary: MeasurementSource::InMemory(flat_curve()),
                support: MeasurementSource::InMemory(flat_curve()),
                supporting_source: SupportingSourceConfig {
                    delay_ms: 2.0,
                    fir_taps: 128,
                    decorrelation: SupportingSourceDecorrelation::None,
                    ..Default::default()
                },
            }),
        );
        let sys = SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([
                ("Left".to_string(), "left".to_string()),
                ("Right".to_string(), "right".to_string()),
                ("WideLeft".to_string(), "left_ss".to_string()),
            ]),
            subwoofers: None,
            bass_management: None,
            ..Default::default()
        };
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2_000.0;
        let config = room_config(speakers, &sys, optimizer, None, None);
        let mut assembly = super::super::types::WorkflowAssembly {
            config: &config,
            sys: &sys,
            sample_rate: 48000.0,
            frequency_samples: crate::DEFAULT_FREQUENCY_SAMPLES,
            output_dir: temp_dir.path(),
            probe_arrival_overrides: None,
            progress_factory: None,
            stage_callback: None,
        };
        let result = HomeCinemaExecutor.execute(&mut assembly);
        assert!(
            result.is_ok(),
            "home-cinema no-sub with supporting source should run: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert!(result.channels.contains_key("Left"));
        assert!(result.channels.contains_key("Right"));
        assert!(result.channels.contains_key("WideLeft"));
        assert!(result.channels.contains_key("WideLeft_support"));
        assert!(
            result
                .metadata
                .supporting_source
                .as_ref()
                .unwrap()
                .contains_key("WideLeft")
        );
    }

    #[test]
    fn home_cinema_supporting_only_no_sub_emits_primary_and_support() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut speakers = HashMap::new();
        speakers.insert(
            "wide".to_string(),
            SpeakerConfig::SupportingSource(SupportingSourceGroup {
                name: "Wide".to_string(),
                speaker_name: None,
                primary: MeasurementSource::InMemory(flat_curve()),
                support: MeasurementSource::InMemory(flat_curve()),
                supporting_source: SupportingSourceConfig {
                    delay_ms: 2.0,
                    fir_taps: 128,
                    decorrelation: SupportingSourceDecorrelation::None,
                    ..Default::default()
                },
            }),
        );
        let sys = SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([("WideLeft".to_string(), "wide".to_string())]),
            subwoofers: None,
            bass_management: None,
            ..Default::default()
        };
        let config = room_config(speakers, &sys, tiny_optimizer(), None, None);
        let mut assembly = super::super::types::WorkflowAssembly {
            config: &config,
            sys: &sys,
            sample_rate: 48_000.0,
            frequency_samples: crate::DEFAULT_FREQUENCY_SAMPLES,
            output_dir: temp_dir.path(),
            probe_arrival_overrides: None,
            progress_factory: None,
            stage_callback: None,
        };
        let result = HomeCinemaExecutor
            .execute(&mut assembly)
            .expect("supporting-only layout");
        assert!(result.channels.contains_key("WideLeft"));
        assert!(result.channels.contains_key("WideLeft_support"));
    }

    #[test]
    fn home_cinema_supporting_only_with_sub_returns_configuration_error() {
        let temp_dir = tempfile::tempdir().unwrap();
        let speakers = HashMap::from([(
            "wide".to_string(),
            SpeakerConfig::SupportingSource(SupportingSourceGroup {
                name: "Wide".to_string(),
                speaker_name: None,
                primary: MeasurementSource::InMemory(flat_curve()),
                support: MeasurementSource::InMemory(flat_curve()),
                supporting_source: SupportingSourceConfig {
                    fir_taps: 128,
                    decorrelation: SupportingSourceDecorrelation::None,
                    ..Default::default()
                },
            }),
        )]);
        let sys = SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([
                ("WideLeft".to_string(), "wide".to_string()),
                ("LFE".to_string(), "missing_sub".to_string()),
            ]),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                crossover: None,
                mapping: HashMap::new(),
            }),
            bass_management: None,
            ..Default::default()
        };
        let config = room_config(speakers, &sys, tiny_optimizer(), None, None);
        let mut assembly = super::super::types::WorkflowAssembly {
            config: &config,
            sys: &sys,
            sample_rate: 48_000.0,
            frequency_samples: crate::DEFAULT_FREQUENCY_SAMPLES,
            output_dir: temp_dir.path(),
            probe_arrival_overrides: None,
            progress_factory: None,
            stage_callback: None,
        };
        let error = HomeCinemaExecutor.execute(&mut assembly).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("require at least one Single main")
        );
    }

    #[test]
    fn home_cinema_with_sub_lfe_gain_applied_runs() {
        let sys = home_cinema_sys_with_sub();
        let mut speakers = stereo_speakers_with_phase();
        speakers.insert(
            "sub".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
        );
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2_000.0;
        let config = RoomConfig {
            version: default_config_version(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: sys.speakers.clone(),
                subwoofers: sys.subwoofers.clone(),
                bass_management: Some(BassManagementConfig {
                    enabled: true,
                    apply_lfe_gain_to_chain: true,
                    lfe_playback_gain_db: 10.0,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            speakers,
            crossovers: Some(crossovers_fixed()),
            target_curve: None,
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        };
        let mut assembly = make_assembly(&config, config.system.as_ref().unwrap());
        let result = HomeCinemaExecutor.execute(&mut assembly);
        assert!(
            result.is_ok(),
            "apply_lfe_gain_to_chain should run: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.channels.len(), 3);
        let bass_report = result
            .metadata
            .bass_management
            .expect("bass management report");
        assert!(bass_report.lfe_gain_applied_to_chain);
    }

    #[test]
    fn home_cinema_with_sub_gain_limit_advisory_runs() {
        let sys = home_cinema_sys_with_sub();
        let mut speakers = stereo_speakers_with_phase();
        speakers.insert(
            "sub".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
        );
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2_000.0;
        let config = RoomConfig {
            version: default_config_version(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: sys.speakers.clone(),
                subwoofers: sys.subwoofers.clone(),
                bass_management: Some(BassManagementConfig {
                    enabled: true,
                    apply_lfe_gain_to_chain: true,
                    lfe_playback_gain_db: 10.0,
                    max_sub_boost_db: -3.0,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            speakers,
            crossovers: Some(crossovers_fixed()),
            target_curve: None,
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        };
        let mut assembly = make_assembly(&config, config.system.as_ref().unwrap());
        let result = HomeCinemaExecutor.execute(&mut assembly);
        assert!(
            result.is_ok(),
            "sub gain limit should run: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.channels.len(), 3);
        let bass_report = result
            .metadata
            .bass_management
            .expect("bass management report");
        assert!(bass_report.gain_limited);
    }

    #[test]
    fn home_cinema_with_sub_optimize_groups_and_phase_runs() {
        let sys = home_cinema_sys_with_sub();
        let mut speakers = stereo_speakers_with_phase();
        speakers.insert(
            "sub".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
        );
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2_000.0;
        let config = RoomConfig {
            version: default_config_version(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: sys.speakers.clone(),
                subwoofers: sys.subwoofers.clone(),
                bass_management: Some(BassManagementConfig {
                    enabled: true,
                    optimize_groups: true,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            speakers,
            crossovers: Some(crossovers_fixed()),
            target_curve: None,
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        };
        let mut assembly = make_assembly(&config, config.system.as_ref().unwrap());
        let result = HomeCinemaExecutor.execute(&mut assembly);
        assert!(
            result.is_ok(),
            "optimize_groups=true with phase should run: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.channels.len(), 3);
    }

    #[test]
    fn home_cinema_with_sub_frequency_range_crossover_runs() {
        let sys = home_cinema_sys_with_sub();
        let mut speakers = stereo_speakers_with_phase();
        speakers.insert(
            "sub".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve_with_phase())),
        );
        let mut crossovers = HashMap::new();
        crossovers.insert(
            "bass_xo".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: None,
                frequencies: None,
                frequency_range: Some((60.0, 100.0)),
            },
        );
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2_000.0;
        let config = RoomConfig {
            version: default_config_version(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: sys.speakers.clone(),
                subwoofers: sys.subwoofers.clone(),
                bass_management: Some(BassManagementConfig {
                    enabled: true,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            speakers,
            crossovers: Some(crossovers),
            target_curve: None,
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        };
        let mut assembly = make_assembly(&config, config.system.as_ref().unwrap());
        let result = HomeCinemaExecutor.execute(&mut assembly);
        assert!(
            result.is_ok(),
            "frequency_range crossover should run: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.channels.len(), 3);
    }

    #[test]
    fn home_cinema_with_sub_no_phase_runs() {
        let sys = home_cinema_sys_with_sub();
        let mut speakers = stereo_speakers();
        speakers.insert(
            "sub".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
        );
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2_000.0;
        let config = RoomConfig {
            version: default_config_version(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: sys.speakers.clone(),
                subwoofers: sys.subwoofers.clone(),
                bass_management: Some(BassManagementConfig {
                    enabled: true,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            speakers,
            crossovers: Some(crossovers_fixed()),
            target_curve: None,
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        };
        let mut assembly = make_assembly(&config, config.system.as_ref().unwrap());
        let result = HomeCinemaExecutor.execute(&mut assembly);
        assert!(
            result.is_ok(),
            "no-phase home cinema should run: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.channels.len(), 3);
        let bass_report = result
            .metadata
            .bass_management
            .expect("bass management report");
        let optimization = bass_report
            .optimization
            .expect("bass management optimization report");
        assert!(!optimization.phase_available);
    }

    #[test]
    fn home_cinema_with_target_curve_runs() {
        let sys = home_cinema_no_sub_sys();
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2_000.0;
        let config = room_config(
            stereo_speakers(),
            &sys,
            optimizer,
            None,
            Some(TargetCurveConfig::Predefined("flat".to_string())),
        );
        let mut assembly = make_assembly(&config, &sys);
        let result = HomeCinemaExecutor.execute(&mut assembly);
        assert!(
            result.is_ok(),
            "home-cinema with target curve should run: {:?}",
            result.err()
        );
    }

    #[test]
    fn home_cinema_with_sub_multiseat_rejection_reports() {
        let sys = home_cinema_sys_with_sub();
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2_000.0;
        optimizer.multi_seat = Some(MultiSeatConfig {
            all_channel_enabled: true,
            all_channel_strategy: MultiMeasurementStrategy::SpatialRobustness,
            max_deviation_db: 0.001,
            ..Default::default()
        });

        let mut speakers = HashMap::new();
        let seat0 = flat_curve();
        let mut seat1 = flat_curve();
        for (index, spl) in seat1.spl.iter_mut().enumerate() {
            *spl += if index % 2 == 0 { 5.0 } else { -5.0 };
        }
        speakers.insert(
            "left".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![seat0, seat1])),
        );
        speakers.insert(
            "right".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
        );
        speakers.insert(
            "sub".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
        );

        let config = RoomConfig {
            version: default_config_version(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: sys.speakers.clone(),
                subwoofers: sys.subwoofers.clone(),
                bass_management: Some(BassManagementConfig {
                    enabled: true,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            speakers,
            crossovers: Some(crossovers_fixed()),
            target_curve: None,
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        };
        let mut assembly = make_assembly(&config, config.system.as_ref().unwrap());
        let result = HomeCinemaExecutor.execute(&mut assembly);
        assert!(
            result.is_ok(),
            "home-cinema sub multiseat rejection should recover: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.channels.len(), 3);
    }
}
