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
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use rayon::prelude::*;
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_engine::room_result::{ChannelOptimizationResult, RoomOptimizationResult};
use roomeq_engine::topology::{
    align_channels_to_lowest, all_curves_have_usable_phase, all_curves_share_frequency_grid,
    apply_crossover_response_to_curve, apply_delay_and_polarity_to_curve, average_mains_magnitude,
    bass_management_objective, complex_sum_mains, compute_flat_loss, mark_plugin_stage,
    mark_plugins_stage, mark_route_owned_plugin, normalize_crossover_delays,
    predict_bass_management_sum, select_bass_management_crossover_type,
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

const DESIRED_CROSSOVER_TARGET_UNDERFILL_DB: f64 = 1.0;

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
    // Per-source calibration belongs before the route split so the main and
    // its redirected bass retain the same level relationship.
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

fn apply_output_safety_gain(chain: &mut ChannelDspChain, gain_db: f64) {
    if gain_db.abs() <= 0.01 {
        return;
    }
    let mut plugin = mark_plugin_stage(output::create_gain_plugin(gain_db), "post_route");
    if let Some(parameters) = plugin.parameters.as_object_mut() {
        parameters.insert(
            "label".to_string(),
            serde_json::Value::String("post_dsp_output_headroom_safety".to_string()),
        );
    }
    chain.plugins.push(plugin);
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

fn stage_main_correction_plugins(plugins: Vec<PluginConfigWrapper>) -> Vec<PluginConfigWrapper> {
    // Redirected bass is tapped by the route matrix before this stage; only
    // the main self-route receives the main-channel correction.
    mark_plugins_stage(plugins, "post_route")
}

fn stage_sub_correction_plugins(plugins: Vec<PluginConfigWrapper>) -> Vec<PluginConfigWrapper> {
    // The sub chain describes processing at the physical bass output, after
    // redirected main and native LFE routes have been summed.
    mark_plugins_stage(plugins, "post_route")
}

/// Build the tonal objective for EQ shared by every routed bass source.
///
/// The physical-sub transfer and common output gain are the only terms shared
/// by every route. Source crossover, delay, polarity, input trim, and source
/// count are deliberately excluded: coherently summing those independent
/// programme inputs would turn their accidental phase relationship into a
/// permanent correction on the common physical output.
fn physical_sub_tonal_objective_curve(curve: &Curve, common_gain_db: f64) -> Curve {
    let mut objective = curve.clone();
    for spl in objective.spl.iter_mut() {
        *spl += common_gain_db;
    }
    objective
}

fn is_source_pre_route_plugin(plugin: &PluginConfigWrapper) -> bool {
    let parameters = &plugin.parameters;
    let is_pre_route = parameters
        .get("room_eq_stage")
        .and_then(serde_json::Value::as_str)
        == Some("pre_route");
    let is_calibration_gain = plugin.plugin_type == "gain"
        && parameters.get("label").and_then(serde_json::Value::as_str)
            == Some("post_dsp_input_level_alignment");
    is_pre_route && !is_calibration_gain
}

fn realize_source_pre_route_transfer(
    source_channel: &str,
    plugins: impl IntoIterator<Item = PluginConfigWrapper>,
    reference: &Curve,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
    embedded_irs: &HashMap<String, Vec<f64>>,
) -> Result<Curve> {
    let plugins = plugins
        .into_iter()
        .filter(is_source_pre_route_plugin)
        .collect::<Vec<_>>();
    let transfer_input = Curve {
        freq: reference.freq.clone(),
        spl: ndarray::Array1::zeros(reference.freq.len()),
        phase: Some(ndarray::Array1::zeros(reference.freq.len())),
        ..Curve::default()
    };
    let transfer_chain = ChannelDspChain {
        channel: source_channel.to_string(),
        plugins,
        drivers: None,
        initial_curve: None,
        final_curve: None,
        eq_response: None,
        target_curve: None,
        pre_ir: None,
        post_ir: None,
        fir_temporal_masking: None,
        direct_early_late_correction: None,
    };
    crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
        &transfer_chain,
        &transfer_input,
        sample_rate,
        sidecar_dir,
        embedded_irs,
    )
}

fn embedded_convolution_irs(
    plugins: &[PluginConfigWrapper],
    fir_coeffs: Option<&[f64]>,
) -> Result<HashMap<String, Vec<f64>>> {
    let Some(fir_coeffs) = fir_coeffs else {
        return Ok(HashMap::new());
    };
    let ir_files = plugins
        .iter()
        .filter(|plugin| plugin.plugin_type == "convolution")
        .filter_map(|plugin| {
            plugin
                .parameters
                .get("ir_file")
                .and_then(serde_json::Value::as_str)
        })
        .collect::<Vec<_>>();
    match ir_files.as_slice() {
        [] => Ok(HashMap::new()),
        [ir_file] => Ok(HashMap::from([(
            (*ir_file).to_string(),
            fir_coeffs.to_vec(),
        )])),
        _ => Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "cannot associate one retained FIR with {} convolution plugins",
                ir_files.len()
            ),
        }),
    }
}

fn source_pre_route_transfers(
    channels: &HashMap<String, ChannelDspChain>,
    fir_coeffs_by_channel: &HashMap<String, Vec<f64>>,
    source_roles: impl IntoIterator<Item = String>,
    reference: &Curve,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
) -> Result<HashMap<String, Curve>> {
    source_roles
        .into_iter()
        .map(|role| {
            let chain = channels
                .get(&role)
                .ok_or_else(|| AutoeqError::InvalidConfiguration {
                    message: format!("missing source chain '{role}' for pre-route realization"),
                })?;
            let embedded_irs = embedded_convolution_irs(
                &chain.plugins,
                fir_coeffs_by_channel.get(&role).map(Vec::as_slice),
            )?;
            let transfer = realize_source_pre_route_transfer(
                &role,
                chain.plugins.clone(),
                reference,
                sample_rate,
                sidecar_dir,
                &embedded_irs,
            )?;
            Ok((role, transfer))
        })
        .collect()
}

/// Rebuild per-logical-input deployed curves from the final serialized DSP
/// chains and routing graph.
///
/// Workflow-wide safety stages may remove or append correction plugins after
/// topology optimization. The authoritative curves must describe the graph
/// that is actually exported, not that earlier intermediate state.
pub(crate) fn reconstruct_deployed_source_curves(
    channels: &HashMap<String, ChannelDspChain>,
    fir_coeffs_by_channel: &HashMap<String, Vec<f64>>,
    graph: &BassManagementRoutingGraph,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
) -> Result<HashMap<String, Curve>> {
    let lfe_role = &graph.physical_sub_output;
    let mut common_sub_chain =
        channels
            .get(lfe_role)
            .cloned()
            .ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: format!("missing physical sub chain '{lfe_role}' for reconstruction"),
            })?;
    common_sub_chain.plugins.retain(|plugin| {
        plugin
            .parameters
            .get("room_eq_stage")
            .and_then(serde_json::Value::as_str)
            == Some("post_route")
    });
    let sub_initial: Curve = common_sub_chain
        .initial_curve
        .clone()
        .ok_or_else(|| AutoeqError::InvalidMeasurement {
            message: format!("physical sub chain '{lfe_role}' has no initial curve"),
        })?
        .into();
    let common_sub_embedded_irs = embedded_convolution_irs(
        &common_sub_chain.plugins,
        fir_coeffs_by_channel.get(lfe_role).map(Vec::as_slice),
    )?;
    let common_sub_curve = crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
        &common_sub_chain,
        &sub_initial,
        sample_rate,
        sidecar_dir,
        &common_sub_embedded_irs,
    )?;
    let source_roles = graph
        .routes
        .iter()
        .filter(|route| {
            matches!(
                route.route_kind.as_str(),
                "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
            )
        })
        .map(|route| route.source_channel.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let source_transfers = source_pre_route_transfers(
        channels,
        fir_coeffs_by_channel,
        source_roles.iter().cloned(),
        &common_sub_curve,
        sample_rate,
        sidecar_dir,
    )?;

    source_roles
        .into_iter()
        .map(|role| {
            let main_curve = if role == *lfe_role {
                None
            } else {
                let chain =
                    channels
                        .get(&role)
                        .ok_or_else(|| AutoeqError::InvalidConfiguration {
                            message: format!("missing main channel chain '{role}'"),
                        })?;
                let initial: Curve = chain
                    .initial_curve
                    .clone()
                    .ok_or_else(|| AutoeqError::InvalidMeasurement {
                        message: format!("main channel '{role}' has no initial curve"),
                    })?
                    .into();
                let embedded_irs = embedded_convolution_irs(
                    &chain.plugins,
                    fir_coeffs_by_channel.get(&role).map(Vec::as_slice),
                )?;
                Some(
                    crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
                        chain,
                        &initial,
                        sample_rate,
                        sidecar_dir,
                        &embedded_irs,
                    )?,
                )
            };
            let deployed = engine_bass_management::predict_deployed_source_curve_from_routes(
                main_curve.as_ref(),
                &common_sub_curve,
                source_transfers.get(&role),
                graph,
                &role,
                sample_rate,
            )
            .ok_or_else(|| AutoeqError::InvalidMeasurement {
                message: format!("could not reconstruct deployed source curve '{role}'"),
            })?;
            if let Some(main) = main_curve.as_ref() {
                let bass = engine_bass_management::predict_bass_source_curve_from_routes(
                    &common_sub_curve,
                    source_transfers.get(&role),
                    graph,
                    &role,
                    sample_rate,
                )
                .ok_or_else(|| AutoeqError::InvalidMeasurement {
                    message: format!("could not reconstruct routed bass branch '{role}'"),
                })?;
                let crossover_hz = graph
                    .routes
                    .iter()
                    .find(|route| {
                        route.source_channel == role
                            && route.route_kind == "redirected_bass_lowpass_to_sub"
                    })
                    .and_then(|route| route.low_pass_hz)
                    .ok_or_else(|| AutoeqError::InvalidConfiguration {
                        message: format!("missing routed crossover frequency for '{role}'"),
                    })?;
                let underfill_db =
                    roomeq_engine::topology::bass_management_crossover_cancellation_underfill_db(
                        main,
                        &bass,
                        &deployed,
                        crossover_hz,
                    )
                    .ok_or_else(|| AutoeqError::InvalidMeasurement {
                        message: format!("mismatched crossover reconstruction grid for '{role}'"),
                    })?;
            if !roomeq_engine::topology::bass_management_underfill_is_acceptable(underfill_db) {
                    return Err(AutoeqError::OptimizationFailed {
                        message: format!(
                            "final routed crossover underfill for '{role}' is \
                             {underfill_db:.3} dB at {crossover_hz:.1} Hz (limit {:.1} dB)",
                            roomeq_engine::topology::MAX_ACCEPTED_CROSSOVER_UNDERFILL_DB
                    ),
                });
            }
            if let Some(target) = channels
                .get(&role)
                .and_then(|chain| chain.target_curve.clone())
                .map(Curve::from)
                && let Some(target_underfill_db) =
                    roomeq_engine::topology::bass_management_max_underfill_db_with_target(
                        Some(&deployed),
                        Some(&target),
                        crossover_hz,
                    )
                && !roomeq_engine::topology::bass_management_underfill_is_acceptable(
                    target_underfill_db,
                )
            {
                return Err(AutoeqError::OptimizationFailed {
                    message: format!(
                        "final routed target underfill for '{role}' is {target_underfill_db:.3} dB at {crossover_hz:.1} Hz (limit {:.1} dB)",
                        roomeq_engine::topology::MAX_ACCEPTED_CROSSOVER_UNDERFILL_DB
                    ),
                });
            }
        }
            Ok((role, deployed))
        })
        .collect()
}

/// Align main inputs at the microphone after the complete routed DSP graph.
///
/// LFE keeps its separately configured cinema playback gain; treating its
/// band-limited mean as another main-channel reference would cancel that gain.
/// Trims are down-only so calibration cannot consume headroom.
#[allow(clippy::too_many_arguments)]
fn calibrate_post_dsp_input_levels(
    config: &RoomConfig,
    main_roles: &[String],
    lfe_role: &str,
    main_band: (f64, f64),
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
    fir_coeffs_by_channel: &HashMap<String, Vec<f64>>,
    channels: &mut HashMap<String, ChannelDspChain>,
    graph: &mut BassManagementRoutingGraph,
) -> Result<(HashMap<String, f64>, HashMap<String, Curve>)> {
    let mut common_sub_chain =
        channels
            .get(lfe_role)
            .cloned()
            .ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: format!("missing physical sub chain '{lfe_role}' for level calibration"),
            })?;
    common_sub_chain.plugins.retain(|plugin| {
        // Redirected mains enter the physical sub after the LFE logical
        // input's pre-route chain.  Only destination post-route processing is
        // common to every signal emitted by the subwoofer.
        plugin
            .parameters
            .get("room_eq_stage")
            .and_then(|value| value.as_str())
            == Some("post_route")
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
    let source_transfers = source_pre_route_transfers(
        channels,
        fir_coeffs_by_channel,
        main_roles
            .iter()
            .cloned()
            .chain(std::iter::once(lfe_role.to_string())),
        &common_sub_curve,
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
        let observed = engine_bass_management::predict_deployed_source_curve_from_routes(
            Some(&main_curve),
            &common_sub_curve,
            source_transfers.get(role),
            graph,
            role,
            sample_rate,
        )
        .unwrap_or(main_curve);
        means.insert(role.clone(), average_spl(&observed, main_band));
    }

    let target = means
        .values()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::INFINITY, f64::min);
    if !target.is_finite() || means.len() != main_roles.len() {
        return Ok((HashMap::new(), HashMap::new()));
    }
    let mut trims: HashMap<String, f64> = means
        .into_iter()
        .map(|(role, mean)| (role, (target - mean).min(0.0)))
        .collect();
    trims.insert(lfe_role.to_string(), 0.0);

    for role in main_roles {
        if let Some(chain) = channels.get_mut(role) {
            apply_gain_to_main_chain(chain, *trims.get(role).unwrap_or(&0.0));
        }
    }
    let logical_input_trims = trims.clone();

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
                    apply_output_safety_gain(chain, -safety_trim_db);
                }
            }
            if let Some(chain) = channels.get_mut(lfe_role) {
                apply_output_safety_gain(chain, -safety_trim_db);
            }
            for trim_db in trims.values_mut() {
                *trim_db -= safety_trim_db;
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
    graph.input_trim_db = logical_input_trims;
    graph
        .advisories
        .push("post_dsp_input_levels_aligned_down".to_string());

    let mut calibrated_common_sub_curve = common_sub_curve.clone();
    shift_curve_level(
        &mut calibrated_common_sub_curve,
        *trims.get(lfe_role).unwrap_or(&0.0),
    );
    if let Some(final_bass_bus) = engine_bass_management::predict_bass_output_curve_from_routes(
        &calibrated_common_sub_curve,
        Some(&source_transfers),
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

    let mut deployed_source_curves = HashMap::new();
    for role in main_roles {
        let main_curve: Curve = channels
            .get(role)
            .and_then(|chain| chain.final_curve.clone())
            .ok_or_else(|| AutoeqError::InvalidMeasurement {
                message: format!("main channel '{role}' has no final curve"),
            })?
            .into();
        if let Some(deployed) = engine_bass_management::predict_deployed_source_curve_from_routes(
            Some(&main_curve),
            &calibrated_common_sub_curve,
            source_transfers.get(role),
            graph,
            role,
            sample_rate,
        ) {
            deployed_source_curves.insert(role.clone(), deployed);
        }
    }
    if let Some(lfe) = engine_bass_management::predict_deployed_source_curve_from_routes(
        None,
        &calibrated_common_sub_curve,
        source_transfers.get(lfe_role),
        graph,
        lfe_role,
        sample_rate,
    ) {
        deployed_source_curves.insert(lfe_role.to_string(), lfe);
    }

    Ok((trims, deployed_source_curves))
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
        deployed_source_curves: HashMap::new(),
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
        deployed_source_curves: HashMap::new(),
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
    let post_eq_resources =
        crate::prepare_eq_resources(&config.optimizer, config.target_curve.as_ref()).map_err(
            |error| AutoeqError::InvalidConfiguration {
                message: format!("failed to prepare home-cinema target curve: {error}"),
            },
        )?;

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
    let mut pre_eq_fir_coeffs: HashMap<String, Vec<f64>> = HashMap::new();
    let mut pre_eq_target_curves: HashMap<String, CurveData> = HashMap::new();
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
        if let Some(target) = chain.target_curve.clone() {
            pre_eq_target_curves.insert(role.clone(), target);
        }
        if let Some(fir_coeffs) = ch_result.fir_coeffs.clone() {
            pre_eq_fir_coeffs.insert(role.clone(), fir_coeffs);
        }
        pre_eq_plugins.insert(role.clone(), stage_main_correction_plugins(chain.plugins));
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
            stage_sub_correction_plugins(chain.plugins),
        );
        if let Some(target) = chain.target_curve {
            pre_eq_target_curves.insert(sub_role.clone(), target);
        }
        if let Some(fir_coeffs) = ch_result.fir_coeffs.clone() {
            pre_eq_fir_coeffs.insert(sub_role.clone(), fir_coeffs);
        }
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
    // Bass-route optimization models the physical sub output shared by every
    // redirected main. The LFE logical-input alignment gain is applied only on
    // the LFE pre-route path and must not be folded into this common transfer.
    aligned_pre_eq_curves.insert(sub_role.clone(), linearized_curves[&sub_role].clone());

    let optimizer_source_pre_route_transfers = main_roles
        .iter()
        .map(|role| {
            let mut plugins = pre_eq_plugins.get(role).cloned().unwrap_or_default();
            let align_gain = *gains.get(role).unwrap_or(&0.0);
            if align_gain.abs() > 0.01 {
                plugins.insert(
                    0,
                    mark_plugin_stage(output::create_gain_plugin(align_gain), "pre_route"),
                );
            }
            let embedded_irs =
                embedded_convolution_irs(&plugins, pre_eq_fir_coeffs.get(role).map(Vec::as_slice))?;
            let transfer = realize_source_pre_route_transfer(
                role,
                plugins,
                &aligned_pre_eq_curves[&sub_role],
                sample_rate,
                output_dir,
                &embedded_irs,
            )?;
            Ok::<_, AutoeqError>((role.clone(), transfer))
        })
        .collect::<Result<HashMap<_, _>>>()?;

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
            let optimized = crossover::optimize_main_sub_crossover(
                crossover::MainSubCrossoverInput {
                    main_highpass: virtual_main.clone(),
                    sub_lowpass: sub_curve.clone(),
                },
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
                optimized.main_gain_db,
                optimized.main_delay_ms,
                optimized.sub_gain_db,
                optimized.sub_delay_ms,
                optimized.sub_inverted,
                optimized.crossover_frequency_hz,
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
    let preliminary_bass_management_optimization = joint_bass_management_report_from_parts(
        &group_results_by_id.values().cloned().collect::<Vec<_>>(),
        &[],
        &preliminary_sub_output_results,
    );
    let preliminary_bass_routing_graph = engine_home_cinema::bass_management_routing_graph(
        config,
        Some(&preliminary_bass_management_optimization),
    );
    let sub_post_initial =
        physical_sub_tonal_objective_curve(&aligned_pre_eq_curves[&sub_role], sub_gain_post);

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
    let bass_management_target_curves = post_eq_resources.target.as_ref().map(|_| {
        main_roles
            .iter()
            .filter_map(|role| {
                aligned_pre_eq_curves.get(role).map(|curve| {
                    (
                        role.clone(),
                        roomeq_engine::fir::prepared_fir_target_curve(
                            curve,
                            &config.optimizer,
                            &post_eq_resources,
                        ),
                    )
                })
            })
            .collect::<HashMap<_, _>>()
    });
    let sub_output_advisories = if optimize_source_routes {
        optimize_bass_management_joint_solution(
            config,
            main_roles,
            &aligned_curves,
            &aligned_pre_eq_curves,
            Some(&optimizer_source_pre_route_transfers),
            bass_management_target_curves.as_ref(),
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
    for source in &source_results {
        info!(
            "  Source route '{}': main_delay={:.3} ms, bass_delay={:.3} ms, invert={}, trim={:+.2} dB, accepted={}, advisories={:?}",
            source.source_channel,
            source.main_delay_ms,
            source.bass_route_delay_ms,
            source.polarity_inverted,
            source.trim_db,
            source.accepted,
            source.advisories,
        );
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
    // 6. Post-EQ
    let mut post_eq_filters = HashMap::new();
    let mut routed_target_curves: HashMap<String, CurveData> = HashMap::new();
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
        // The broadband main correction has already run. Reserve this pass
        // for the routed crossover residual so its filters are not spent on
        // unrelated high-frequency details.
        opt_config.min_freq = opt_config.min_freq.max(role_xover_freq * 0.5);
        opt_config.max_freq = opt_config.max_freq.min(role_xover_freq * 2.0);
        let post_curve = bass_routing_graph
            .as_ref()
            .and_then(|graph| {
                engine_bass_management::predict_deployed_source_curve_from_routes(
                    Some(&main_post_curves[role]),
                    &sub_post,
                    optimizer_source_pre_route_transfers.get(role),
                    graph,
                    role,
                    sample_rate,
                )
            })
            .unwrap_or_else(|| main_post_curves[role].clone());
        let prepared_target = post_eq_resources.target.as_ref().map(|_| {
            roomeq_engine::fir::prepared_fir_target_curve(
                &post_curve,
                &opt_config,
                &post_eq_resources,
            )
        });
        if let Some(target) = prepared_target.as_ref() {
            routed_target_curves.insert(role.clone(), target.into());
        }
        if opt_config.min_freq >= opt_config.max_freq {
            log::warn!(
                "  Skipping {role} routed Post-EQ: invalid optimization band [{:.1}, {:.1}] Hz",
                config.optimizer.min_freq,
                config.optimizer.max_freq
            );
            post_eq_filters.insert(role.clone(), Vec::new());
            continue;
        }

        let post_eq_callback = workflow_progress_callback(
            &assembly.progress_factory,
            &format!("Post-EQ {role}"),
            role_index,
            total_post_eq_passes,
            opt_config.max_iter,
        );
        let mut post_eq_result = run_post_eq(
            &post_curve,
            &opt_config,
            config.target_curve.as_ref(),
            sample_rate,
            post_eq_callback,
        )?;
        let mut filters = post_eq_result.filters;
        // The broad optimizer minimizes aggregate target error. Close any
        // remaining narrow crossover dip explicitly because this EQ is
        // serialized pre-route and therefore corrects both branches equally.
        for _ in 0..2 {
            let eq_response =
                response::compute_peq_complex_response(&filters, &post_curve.freq, sample_rate);
            let corrected_sum = response::apply_complex_response(&post_curve, &eq_response);
            let Some((frequency, underfill_db)) =
                roomeq_engine::topology::bass_management_worst_underfill_with_target(
                    Some(&corrected_sum),
                    prepared_target.as_ref(),
                    role_xover_freq,
                )
            else {
                break;
            };
            let excess_db = underfill_db - DESIRED_CROSSOVER_TARGET_UNDERFILL_DB;
            if excess_db <= 1.0e-9 {
                break;
            }
            let gain_db = (excess_db + 0.25).min(opt_config.max_db.max(0.0));
            if gain_db <= 0.01 {
                break;
            }
            filters.push(Biquad::new(
                BiquadFilterType::Peak,
                frequency,
                sample_rate,
                1.0,
                gain_db,
            ));
        }

        // Evaluate acceptance over the same band reported in the final channel
        // score. The optimizer keeps its 20 Hz crossover guard band, but a
        // candidate must not damage that guard band enough to regress the
        // published response score.
        let pre = roomeq_engine::topology::bass_management_objective_with_target(
            Some(&post_curve),
            prepared_target.as_ref(),
            role_xover_freq,
        )
        .unwrap_or_else(|| compute_flat_loss(&post_curve, role_xover_freq, main_post_max_freq));
        let eq_resp =
            response::compute_peq_complex_response(&filters, &post_curve.freq, sample_rate);
        let main_curve_after = response::apply_complex_response(&main_post_curves[role], &eq_resp);
        let bass_branch = bass_routing_graph
            .as_ref()
            .and_then(|graph| {
                engine_bass_management::predict_bass_source_curve_from_routes(
                    &sub_post,
                    optimizer_source_pre_route_transfers.get(role),
                    graph,
                    role,
                    sample_rate,
                )
            })
            .map(|bass| response::apply_complex_response(&bass, &eq_resp));
        let post_curve_after = bass_branch
            .as_ref()
            .map(|bass| complex_sum_mains(&[&main_curve_after, bass]))
            .unwrap_or_else(|| main_curve_after.clone());
        let post = roomeq_engine::topology::bass_management_objective_with_target(
            Some(&post_curve_after),
            prepared_target.as_ref(),
            role_xover_freq,
        )
        .unwrap_or_else(|| {
            compute_flat_loss(&post_curve_after, role_xover_freq, main_post_max_freq)
        });
        let cancellation_underfill_db = bass_branch.as_ref().and_then(|bass| {
            roomeq_engine::topology::bass_management_crossover_cancellation_underfill_db(
                &main_curve_after,
                bass,
                &post_curve_after,
                role_xover_freq,
            )
        });
        let target_underfill_db =
            roomeq_engine::topology::bass_management_max_underfill_db_with_target(
                Some(&post_curve_after),
                prepared_target.as_ref(),
                role_xover_freq,
            );
        let post_underfill_db = cancellation_underfill_db
            .into_iter()
            .chain(target_underfill_db)
            .reduce(f64::max);
        log::debug!(
            "  {role} Post-EQ underfill: cancellation={cancellation_underfill_db:?} dB, target={target_underfill_db:?} dB"
        );
        let underfill_accepted = post_underfill_db
            .is_none_or(roomeq_engine::topology::bass_management_underfill_is_acceptable);
        if post < pre && underfill_accepted {
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
            if let Some(underfill_db) = post_underfill_db.filter(|_| !underfill_accepted) {
                log::warn!(
                    "  {} Post-EQ discarded: crossover underfill {:.3} dB exceeds {:.3} dB",
                    role,
                    underfill_db,
                    roomeq_engine::topology::MAX_ACCEPTED_CROSSOVER_UNDERFILL_DB,
                );
            } else {
                log::warn!(
                    "  {} Post-EQ discarded: score regressed from {:.4} to {:.4}",
                    role,
                    pre,
                    post
                );
            }
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
        let sub_post_eq_band_empty = opt_config.max_freq <= opt_config.min_freq;
        if sub_post_eq_band_empty {
            log::warn!(
                "  Sub Post-EQ skipped: bass-route upper bound {:.1} Hz leaves no optimization band above min_freq {:.1} Hz after the 20 Hz guard band",
                bass_route_upper_hz,
                opt_config.min_freq,
            );
        }
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
        if sub_post_eq_band_empty {
            post_eq_filters.insert(sub_role.clone(), Vec::new());
        } else if post < pre {
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
                "pre_route",
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
        let mut chain = ChannelDspChain {
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
            target_curve: routed_target_curves
                .get(role)
                .cloned()
                .or_else(|| pre_eq_target_curves.get(role).cloned()),
        };
        let embedded_irs = embedded_convolution_irs(
            &chain.plugins,
            pre_eq_fir_coeffs.get(role).map(Vec::as_slice),
        )?;
        match crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
            &chain,
            &pre_eq_initial_curves[role],
            sample_rate,
            output_dir,
            &embedded_irs,
        ) {
            Ok(realized) => {
                let realized_data: CurveData = (&realized).into();
                chain.eq_response = chain
                    .initial_curve
                    .as_ref()
                    .map(|initial| output::compute_eq_response(initial, &realized_data));
                chain.final_curve = Some(realized_data);
            }
            Err(error) => log::warn!(
                "Could not reconstruct canonical home-cinema response for '{}': {}",
                role,
                error
            ),
        }
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
        target_curve: pre_eq_target_curves.get(&sub_role).cloned(),
    };
    channel_chains.insert(sub_role.clone(), sub_chain);

    let (post_dsp_input_trims, mut deployed_source_curves) =
        if let Some(graph) = bass_routing_graph.as_mut() {
            calibrate_post_dsp_input_levels(
                config,
                main_roles,
                &sub_role,
                main_alignment_band,
                sample_rate,
                output_dir,
                &pre_eq_fir_coeffs,
                &mut channel_chains,
                graph,
            )?
        } else {
            (HashMap::new(), HashMap::new())
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
                fir_coeffs: pre_eq_fir_coeffs.get(role).cloned(),
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
                fir_coeffs: pre_eq_fir_coeffs.get(&sub_role).cloned(),
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
    if let Some(graph) = bass_routing_graph.as_ref() {
        deployed_source_curves = reconstruct_deployed_source_curves(
            &channel_chains,
            &pre_eq_fir_coeffs,
            graph,
            sample_rate,
            output_dir,
        )?;
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
        deployed_source_curves,
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
    use super::{
        apply_gain_to_main_chain, apply_output_safety_gain, average_spl,
        calibrate_post_dsp_input_levels, physical_sub_tonal_objective_curve,
        realize_source_pre_route_transfer, reconstruct_deployed_source_curves,
        stage_main_correction_plugins, stage_sub_correction_plugins,
    };
    use roomeq_engine::Curve;
    use roomeq_engine::topology::mark_plugin_stage;
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
    fn main_correction_is_staged_after_route_matrix() {
        let plugins =
            stage_main_correction_plugins(vec![roomeq_engine::output::create_gain_plugin(1.0)]);
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].parameters["room_eq_stage"], "post_route");
    }

    #[test]
    fn sub_correction_is_staged_after_route_sum() {
        let plugins =
            stage_sub_correction_plugins(vec![roomeq_engine::output::create_gain_plugin(1.0)]);
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].parameters["room_eq_stage"], "post_route");
    }

    #[test]
    fn logical_input_calibration_precedes_route_split() {
        let initial = curve(70.0);
        let mut channel = chain("L", initial.clone(), Some(initial));

        apply_gain_to_main_chain(&mut channel, -6.0);

        let calibration = channel.plugins.last().unwrap();
        assert_eq!(calibration.parameters["room_eq_stage"], "pre_route");
        assert_eq!(
            calibration.parameters["label"],
            "post_dsp_input_level_alignment"
        );
        assert!(
            channel
                .final_curve
                .as_ref()
                .unwrap()
                .spl
                .iter()
                .all(|level| (*level - 64.0).abs() < 1.0e-12)
        );
    }

    #[test]
    fn source_pre_route_realization_excludes_route_owned_calibration_and_post_route_dsp() {
        let reference = curve(0.0);
        let delay_ms = 2.5;
        let mut calibration =
            mark_plugin_stage(roomeq_engine::output::create_gain_plugin(-7.0), "pre_route");
        calibration.parameters["label"] = serde_json::json!("post_dsp_input_level_alignment");
        let plugins = vec![
            mark_plugin_stage(
                roomeq_engine::output::create_delay_plugin(delay_ms),
                "pre_route",
            ),
            mark_plugin_stage(roomeq_engine::output::create_gain_plugin(-2.0), "pre_route"),
            calibration,
            mark_plugin_stage(roomeq_engine::output::create_gain_plugin(9.0), "post_route"),
        ];
        let transfer = realize_source_pre_route_transfer(
            "L",
            plugins,
            &reference,
            48_000.0,
            std::path::Path::new("."),
            &HashMap::new(),
        )
        .expect("pre-route transfer");

        assert!(
            transfer
                .spl
                .iter()
                .all(|level| (*level + 2.0).abs() < 1.0e-12)
        );
        let phase = transfer.phase.as_ref().expect("delay phase");
        for (frequency, phase) in transfer.freq.iter().zip(phase.iter()) {
            let expected = -360.0 * frequency * delay_ms / 1_000.0;
            let wrapped_error = (phase - expected + 180.0).rem_euclid(360.0) - 180.0;
            assert!(wrapped_error.abs() < 1.0e-10);
        }
    }

    #[test]
    fn deployed_source_refresh_uses_final_exported_chain_after_post_eq_reversion() {
        let initial = curve(60.0);
        let mut lfe = chain("LFE", initial.clone(), Some(initial.clone()));
        lfe.plugins = vec![
            mark_plugin_stage(roomeq_engine::output::create_delay_plugin(2.5), "pre_route"),
            mark_plugin_stage(roomeq_engine::output::create_gain_plugin(-6.0), "pre_route"),
            mark_plugin_stage(roomeq_engine::output::create_gain_plugin(3.0), "post_route"),
        ];
        let graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".to_string(),
            input_channels: vec!["LFE".to_string()],
            output_channels: vec!["LFE".to_string()],
            routes: vec![low_route("LFE", 0)],
            matrix: None,
            input_trim_db: HashMap::new(),
            advisories: Vec::new(),
        };
        let mut channels = HashMap::from([("LFE".to_string(), lfe)]);
        let before = reconstruct_deployed_source_curves(
            &channels,
            &HashMap::new(),
            &graph,
            48_000.0,
            std::path::Path::new("."),
        )
        .expect("deployed curve with post-EQ");

        channels.get_mut("LFE").unwrap().plugins.retain(|plugin| {
            plugin
                .parameters
                .get("room_eq_stage")
                .and_then(serde_json::Value::as_str)
                != Some("post_route")
        });
        let after = reconstruct_deployed_source_curves(
            &channels,
            &HashMap::new(),
            &graph,
            48_000.0,
            std::path::Path::new("."),
        )
        .expect("deployed curve after post-EQ reversion");

        for (with_post_eq, reverted) in before["LFE"].spl.iter().zip(after["LFE"].spl.iter()) {
            assert!((with_post_eq - reverted - 3.0).abs() < 1.0e-9);
        }
        for (with_post_eq, reverted) in before["LFE"]
            .phase
            .as_ref()
            .unwrap()
            .iter()
            .zip(after["LFE"].phase.as_ref().unwrap().iter())
        {
            let error = (with_post_eq - reverted + 180.0).rem_euclid(360.0) - 180.0;
            assert!(error.abs() < 1.0e-9);
        }
    }

    #[test]
    fn deployed_source_refresh_replays_main_chain_instead_of_stale_final_curve() {
        let initial = curve(60.0);
        let mut main = chain("L", initial.clone(), Some(curve(90.0)));
        main.plugins.push(mark_plugin_stage(
            roomeq_engine::output::create_gain_plugin(-3.0),
            "pre_route",
        ));
        let graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".to_string(),
            input_channels: vec!["L".to_string(), "LFE".to_string()],
            output_channels: vec!["L".to_string(), "LFE".to_string()],
            routes: vec![low_route("L", 0)],
            matrix: None,
            input_trim_db: HashMap::new(),
            advisories: Vec::new(),
        };
        let mut channels = HashMap::from([
            ("L".to_string(), main),
            (
                "LFE".to_string(),
                chain("LFE", curve(20.0), Some(curve(20.0))),
            ),
        ]);

        let before = reconstruct_deployed_source_curves(
            &channels,
            &HashMap::new(),
            &graph,
            48_000.0,
            std::path::Path::new("."),
        )
        .unwrap();
        channels.get_mut("L").unwrap().final_curve = Some(CurveData::from(&curve(20.0)));
        let after = reconstruct_deployed_source_curves(
            &channels,
            &HashMap::new(),
            &graph,
            48_000.0,
            std::path::Path::new("."),
        )
        .unwrap();

        assert_eq!(before["L"].spl, after["L"].spl);
        assert_eq!(before["L"].phase, after["L"].phase);
    }

    #[test]
    fn deployed_source_refresh_rejects_more_than_three_db_of_cancellation() {
        let initial = curve(60.0);
        let graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".to_string(),
            input_channels: vec!["L".to_string(), "LFE".to_string()],
            output_channels: vec!["L".to_string(), "LFE".to_string()],
            routes: vec![low_route("L", 0)],
            matrix: None,
            input_trim_db: HashMap::new(),
            advisories: Vec::new(),
        };
        let bass = super::engine_bass_management::predict_bass_source_curve_from_routes(
            &initial, None, &graph, "L", 48_000.0,
        )
        .expect("routed bass branch");
        let mut cancelling_main = bass.clone();
        cancelling_main
            .phase
            .as_mut()
            .unwrap()
            .mapv_inplace(|phase| phase + 180.0);
        let channels = HashMap::from([
            (
                "L".to_string(),
                chain("L", initial.clone(), Some(cancelling_main)),
            ),
            (
                "LFE".to_string(),
                chain("LFE", initial.clone(), Some(initial)),
            ),
        ]);

        let error = reconstruct_deployed_source_curves(
            &channels,
            &HashMap::new(),
            &graph,
            48_000.0,
            std::path::Path::new("."),
        )
        .expect_err("anti-phase crossover must be rejected");
        assert!(
            error
                .to_string()
                .contains("final routed crossover underfill")
        );
    }

    #[test]
    fn common_headroom_safety_is_applied_after_route_sum() {
        let initial = curve(70.0);
        let mut channel = chain("L", initial.clone(), Some(initial));

        apply_output_safety_gain(&mut channel, -6.0);

        let safety = channel.plugins.last().unwrap();
        assert_eq!(safety.parameters["room_eq_stage"], "post_route");
        assert_eq!(
            safety.parameters["label"],
            "post_dsp_output_headroom_safety"
        );
        assert!(
            channel
                .final_curve
                .as_ref()
                .unwrap()
                .spl
                .iter()
                .all(|level| (*level - 64.0).abs() < 1.0e-12)
        );
    }

    #[test]
    fn physical_sub_tonal_objective_uses_only_common_transfer_and_gain() {
        let physical_sub = curve(60.0);
        let objective = physical_sub_tonal_objective_curve(&physical_sub, 3.0);

        assert!(
            objective
                .spl
                .iter()
                .all(|level| (*level - 63.0).abs() < 1.0e-12)
        );
        assert_eq!(objective.freq, physical_sub.freq);
        assert_eq!(objective.phase, physical_sub.phase);
    }

    #[test]
    fn final_input_calibration_aligns_mains_without_cancelling_lfe_gain() {
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

        let (trims, deployed_source_curves) = calibrate_post_dsp_input_levels(
            &roomeq_model::RoomConfig::default(),
            &["L".to_string(), "R".to_string()],
            "LFE",
            (100.0, 400.0),
            48_000.0,
            std::path::Path::new("."),
            &HashMap::new(),
            &mut channels,
            &mut graph,
        )
        .unwrap();

        assert!(trims.values().all(|trim| *trim <= 1.0e-12));
        let mut observed_means = Vec::new();
        for role in ["L", "R"] {
            observed_means.push(average_spl(&deployed_source_curves[role], (100.0, 400.0)));
        }
        let spread = observed_means
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            - observed_means.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            spread < 1.0e-5,
            "post-DSP main level spread was {spread} dB"
        );
        assert!((trims["LFE"] - trims["R"]).abs() < 1.0e-12);
        assert_eq!(graph.input_trim_db, trims);
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
    fn home_cinema_with_sub_keeps_prepared_target_on_output_chains() {
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
            system: Some(sys.clone()),
            speakers,
            crossovers: Some(crossovers_fixed()),
            target_curve: Some(TargetCurveConfig::Predefined("harman".to_string())),
            optimizer,
            provenance: Default::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        };

        let mut assembly = make_assembly(&config, &sys);
        let result = HomeCinemaExecutor
            .execute(&mut assembly)
            .expect("home cinema with a sub and prepared target should run");

        for channel_name in ["Left", "Right", "LFE"] {
            assert!(
                result.channels[channel_name].target_curve.is_some(),
                "{channel_name} output chain must retain its prepared target"
            );
        }
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
