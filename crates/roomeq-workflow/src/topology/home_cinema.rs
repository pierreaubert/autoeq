// Shared stereo and home-cinema bass-management workflow executor.

use super::bass_management::*;
use super::run::run_channel_via_generic_path_with_frequency_samples;
use super::run::{run_post_eq, run_routed_training_post_eq};
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
    bass_management_objective, complex_sum_mains, compute_flat_loss, curve_has_usable_phase,
    mark_plugin_stage, mark_plugins_stage, mark_route_owned_plugin, normalize_crossover_delays,
    predict_bass_management_sum, select_bass_management_crossover_type,
};
use roomeq_engine::{
    Curve, OptimizerRunEvidence, PipelineStepId, PipelineStepStatus,
    bass_management as engine_bass_management, crossover, home_cinema as engine_home_cinema,
    output, response,
};
use roomeq_model::{
    BassManagementMatrix, BassManagementRoute, BassManagementRoutingGraph,
    BassManagementSourceReport, BassManagementSubOutputReport, ChannelDspChain, CurveData,
    DriverDspChain, MeasurementSource, MultiSubGroup, OptimizationMetadata, PluginConfigWrapper,
    RoomConfig, SpeakerConfig, StageOutcome, StageStatus, StereoBassRoutingCandidateReport,
    StereoBassRoutingReport, StereoBassTopology, SystemConfig, SystemModel,
};
use std::collections::HashMap;

pub(super) struct HomeCinemaExecutor;

const DESIRED_CROSSOVER_TARGET_UNDERFILL_DB: f64 = 1.0;
const BASS_MANAGEMENT_LOG_TARGET: &str = "roomeq_workflow::bass_management";

/// Common acoustic calibration band for home-cinema main channels.
///
/// Calibration is independent of the requested EQ range. Prefer the shared
/// midrange, avoiding bass crossover transitions and measured stopbands.
pub(crate) fn main_level_alignment_band(
    curves: &HashMap<String, Curve>,
    main_roles: &[String],
    crossover_hz: f64,
) -> Result<(f64, f64)> {
    // A two-octave midrange reference avoids room-bass peaks and treble rolloff.
    // Limited-band speakers may narrow it, but must share at least one octave.
    // The passband reference is detected inside the search range: a room mode
    // below the crossover region must not set the reference and veto the
    // midrange.
    let search_low = (2.0 * crossover_hz).max(100.0);
    let search_high = 2_000.0_f64;
    let mut low = search_low;
    let mut high = search_high;
    for role in main_roles {
        let curve = curves
            .get(role)
            .ok_or_else(|| AutoeqError::InvalidMeasurement {
                message: format!("missing main-channel calibration measurement '{role}'"),
            })?;
        let (passband, _) =
            roomeq_engine::analysis::response_metrics::detect_passband_and_mean_in_range(
                curve,
                search_low,
                search_high,
            );
        let (pass_low, pass_high) = passband.ok_or_else(|| AutoeqError::InvalidMeasurement {
            message: format!("no usable calibration passband for '{role}'"),
        })?;
        low = low.max(pass_low);
        high = high.min(pass_high);
    }
    low = low.max(500.0_f64.min(high / 2.0));
    if main_roles.is_empty() || !low.is_finite() || !high.is_finite() || high < 2.0 * low {
        return Err(AutoeqError::InvalidMeasurement {
            message: "main speakers have no shared calibration octave above their crossovers"
                .into(),
        });
    }
    Ok((low, high))
}

fn average_spl(curve: &Curve, band: (f64, f64)) -> f64 {
    roomeq_engine::analysis::response_metrics::mean_response_in_range(curve, band.0, band.1)
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

/// Assess Post-EQ cancellation on the receiving main's measurement grid.
fn post_eq_crossover_cancellation(
    config: &RoomConfig,
    role: &str,
    main: &Curve,
    bass: &Curve,
    crossover_hz: f64,
) -> Option<roomeq_model::CrossoverCancellationEvidence> {
    if !curve_has_usable_phase(main) || !curve_has_usable_phase(bass) {
        return None;
    }
    let bass = roomeq_engine::topology::interpolate_bass_response(&main.freq, bass);
    let combined = complex_sum_mains(&[main, &bass]);
    roomeq_engine::topology::assess_configured_crossover_cancellation(
        config,
        role,
        main,
        &bass,
        &combined,
        crossover_hz,
    )
}

fn is_source_pre_route_plugin(plugin: &PluginConfigWrapper) -> bool {
    plugin
        .parameters
        .get("room_eq_stage")
        .and_then(serde_json::Value::as_str)
        == Some("pre_route")
}

fn is_source_post_route_plugin(plugin: &PluginConfigWrapper) -> bool {
    plugin
        .parameters
        .get("room_eq_stage")
        .and_then(serde_json::Value::as_str)
        == Some("post_route")
}

fn validate_routed_plugin_stage_ownership(
    channels: &HashMap<String, ChannelDspChain>,
    graph: &BassManagementRoutingGraph,
) -> Result<()> {
    for (channel, chain) in channels {
        for plugin in &chain.plugins {
            let owner = plugin
                .parameters
                .get("room_eq_stage")
                .and_then(serde_json::Value::as_str);
            let valid = match owner {
                Some("pre_route" | "post_route") => true,
                Some("route_owned") => {
                    matches!(plugin.plugin_type.as_str(), "gain" | "delay" | "crossover")
                }
                None if plugin.plugin_type == "crossover" => {
                    let parameters = &plugin.parameters;
                    let is_high = parameters.get("output").and_then(serde_json::Value::as_str)
                        == Some("high");
                    let is_low =
                        parameters.get("output").and_then(serde_json::Value::as_str) == Some("low");
                    let frequency = parameters
                        .get("frequency")
                        .and_then(serde_json::Value::as_f64);
                    graph.routes.iter().any(|route| {
                        let expected = if is_high {
                            if route.source_channel == *channel
                                && route.destination == *channel
                                && route.route_kind == "main_highpass_to_self"
                                && route.high_pass_hz.is_some()
                            {
                                route.high_pass_hz
                            } else {
                                None
                            }
                        } else if is_low {
                            if channel == &graph.physical_sub_output
                                && matches!(
                                    route.route_kind.as_str(),
                                    "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
                                )
                                && route.low_pass_hz.is_some()
                            {
                                route.low_pass_hz
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        expected.is_some_and(|expected| {
                            frequency.is_some_and(|frequency| (expected - frequency).abs() < 1e-6)
                        })
                    })
                }
                _ => false,
            };
            if !valid {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!("channel {channel} has unresolved plugin ownership"),
                });
            }
        }
    }
    Ok(())
}

/// Extract the failing role from a final routed crossover underfill error.
fn underfill_error_role(message: &str) -> Option<String> {
    const PREFIX: &str = "final routed crossover underfill for '";
    let start = message.find(PREFIX)? + PREFIX.len();
    let rest = &message[start..];
    let end = rest.find("' is ")?;
    let role = &rest[..end];
    (!role.is_empty() && !role.contains('\'')).then(|| role.to_string())
}

/// Necessary representative-stage guard; final native-seat replay remains
/// authoritative for cumulative loss and spatial outcomes.
fn post_eq_useful_output_loss(
    before: &Curve,
    after: &Curve,
    target: Option<&Curve>,
    min_freq: f64,
    max_freq: f64,
) -> Result<f64> {
    let score = roomeq_engine::quality::evaluate_acoustic_quality_with_permitted_gain(
        std::slice::from_ref(before),
        std::slice::from_ref(after),
        &[],
        &[],
        target,
        roomeq_engine::quality::QualityEvaluationConfig {
            min_freq_hz: min_freq,
            max_freq_hz: max_freq,
            schroeder_hz: None,
            normalize_level: true,
        },
        Default::default(),
        0.0,
    )
    .map_err(|message| AutoeqError::InvalidMeasurement { message })?;
    Ok(score
        .useful_output
        .iter()
        .map(|output| {
            output
                .unexplained_loss_rms_db
                .max(output.bass_unexplained_loss_rms_db.unwrap_or(0.0))
        })
        .fold(0.0_f64, f64::max))
}

/// A post-route correction EQ can disturb the calibrated splice; structural
/// routing, alignment and safety plugins are never splice-breaking candidates.
fn is_splice_breaking_correction_eq(plugin: &PluginConfigWrapper) -> bool {
    if plugin.plugin_type != "eq" {
        return false;
    }
    let stage = plugin
        .parameters
        .get("room_eq_stage")
        .and_then(serde_json::Value::as_str);
    if stage != Some("post_route") {
        return false;
    }
    matches!(
        plugin
            .parameters
            .get("label")
            .and_then(serde_json::Value::as_str),
        Some("room_eq_correction") | Some("post_eq")
    )
}

/// Remove the next splice-breaking post-route correction stage class from a
/// mains chain: the mains-only FIR first (excess-phase rotation breaks the
/// calibrated splice), then the post-route correction EQ. Returns the removed
/// stage name, or `None` when nothing revertible remains.
fn strip_next_splice_breaking_stage(chain: &mut ChannelDspChain) -> Option<&'static str> {
    if chain
        .plugins
        .iter()
        .any(|plugin| plugin.plugin_type == "convolution")
    {
        chain
            .plugins
            .retain(|plugin| plugin.plugin_type != "convolution");
        return Some("fir");
    }
    if chain.plugins.iter().any(is_splice_breaking_correction_eq) {
        chain
            .plugins
            .retain(|plugin| !is_splice_breaking_correction_eq(plugin));
        return Some("peq");
    }
    None
}

pub(super) fn realize_plugins_on_curve(
    source_channel: &str,
    plugins: Vec<PluginConfigWrapper>,
    input: &Curve,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
    embedded_irs: &HashMap<String, Vec<f64>>,
) -> Result<Curve> {
    let chain = ChannelDspChain {
        physical_correction_target: None,
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
        joint_sub: None,
        early_reflections: None,
        t60_octaves: None,
        speech_transmission: None,
        waterfall: None,
        resonance_decays: None,
        wavelet: None,
        early_late_curves: None,
    };
    crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
        &chain,
        input,
        sample_rate,
        sidecar_dir,
        embedded_irs,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "explicit curve and reference-plane evidence for routed main preservation"
)]
fn protected_main_target_preservation(
    raw_initial: &Curve,
    routed_baseline: &Curve,
    routed_before: &Curve,
    routed_after: &Curve,
    aligned_target: &Curve,
    alignment_gain_db: f64,
    smoothing_n: usize,
    band: (f64, f64),
) -> Option<roomeq_model::CorrectionAcceptanceReport> {
    if !alignment_gain_db.is_finite() {
        return None;
    }
    let de_route = |curve: &Curve| {
        let mut pressure =
            crate::room_optimization::remove_routing_transfer(raw_initial, routed_baseline, curve)?;
        pressure.spl.mapv_inplace(|level| level + alignment_gain_db);
        Some(pressure)
    };
    crate::room_optimization::evaluate_preserved_target_passband(
        &de_route(routed_before)?,
        &de_route(routed_after)?,
        aligned_target,
        smoothing_n,
        band,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "explicit routed branch controls mirror the deployed signal path"
)]
fn realize_routed_main_training_branch(
    source_channel: &str,
    plugins: &[PluginConfigWrapper],
    input: &Curve,
    align_gain_db: f64,
    crossover_type: &str,
    crossover_hz: f64,
    main_gain_db: f64,
    main_delay_ms: f64,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
    fir_coeffs: Option<&[f64]>,
) -> Result<Curve> {
    let mut pre_route_plugins = plugins
        .iter()
        .filter(|plugin| is_source_pre_route_plugin(plugin))
        .cloned()
        .collect::<Vec<_>>();
    if align_gain_db.abs() > 0.01 {
        pre_route_plugins.insert(
            0,
            mark_plugin_stage(output::create_gain_plugin(align_gain_db), "pre_route"),
        );
    }
    let pre_route_embedded_irs = embedded_convolution_irs(&pre_route_plugins, None)?;
    let main_input = realize_plugins_on_curve(
        source_channel,
        pre_route_plugins,
        input,
        sample_rate,
        sidecar_dir,
        &pre_route_embedded_irs,
    )?;
    let mut main_route = apply_crossover_response_to_curve(
        &main_input,
        crossover_type,
        crossover_hz,
        sample_rate,
        false,
    );
    main_route.spl.mapv_inplace(|level| level + main_gain_db);
    main_route = apply_delay_and_polarity_to_curve(&main_route, main_delay_ms, false);

    let post_route_plugins = plugins
        .iter()
        .filter(|plugin| is_source_post_route_plugin(plugin))
        .cloned()
        .collect::<Vec<_>>();
    let post_route_embedded_irs = embedded_convolution_irs(&post_route_plugins, fir_coeffs)?;
    realize_plugins_on_curve(
        source_channel,
        post_route_plugins,
        &main_route,
        sample_rate,
        sidecar_dir,
        &post_route_embedded_irs,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "explicit routed branch controls mirror the deployed signal path"
)]
fn realize_routed_sub_training_branch(
    sub_role: &str,
    plugins: &[PluginConfigWrapper],
    input: &Curve,
    gain_adjust_db: f64,
    post_eq_filters: &[Biquad],
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
    fir_coeffs: Option<&[f64]>,
) -> Result<Curve> {
    let embedded_irs = embedded_convolution_irs(plugins, fir_coeffs)?;
    let mut sub_output = realize_plugins_on_curve(
        sub_role,
        plugins.to_vec(),
        input,
        sample_rate,
        sidecar_dir,
        &embedded_irs,
    )?;
    sub_output.spl.mapv_inplace(|level| level + gain_adjust_db);
    if !post_eq_filters.is_empty() {
        let response =
            response::compute_peq_complex_response(post_eq_filters, &sub_output.freq, sample_rate);
        sub_output = response::apply_complex_response(&sub_output, &response);
    }
    Ok(sub_output)
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
    realize_plugins_on_curve(
        source_channel,
        plugins,
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
            let Some(chain) = channels.get(&role) else {
                if engine_home_cinema::role_for_channel(&role) == roomeq_model::HomeCinemaRole::Lfe
                {
                    let transfer = realize_source_pre_route_transfer(
                        &role,
                        std::iter::empty(),
                        reference,
                        sample_rate,
                        sidecar_dir,
                        &HashMap::new(),
                    )?;
                    return Ok((role, transfer));
                }
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!("missing source chain '{role}' for pre-route realization"),
                });
            };
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

/// Advisory prefix recording a refused crossover timing reference.
///
/// Coherent main/sub assessment (delay/polarity alignment and cancellation
/// verdicts) is only meaningful on a shared stationary time base.
pub(crate) const CROSSOVER_TIMING_REFUSED_ADVISORY_PREFIX: &str =
    "crossover_timing_reference_refused:";

/// Advisory recorded when coherent splice assessment is skipped.
///
/// A refused timing reference means main/sub phases share no calibrated time
/// zero, so any coherent splice verdict computed from them is arbitrary.
/// Enforcement sites skip the verdict (keeping magnitude correction) instead
/// of rejecting or accepting corrections on luck.
pub(crate) const CROSSOVER_CANCELLATION_UNASSESSED_ADVISORY: &str =
    "crossover_cancellation_unassessed_unverified_timing";

/// Whether the bass report records a refused crossover timing reference.
///
/// Missing reports fail closed (enforcement stays on); only an explicit
/// refusal skips coherent verdicts. Missing phase or grid mismatches keep
/// their own fail-closed handling elsewhere.
pub(crate) fn crossover_timing_refused(
    optimization: Option<&engine_home_cinema::BassManagementOptimizationReport>,
) -> bool {
    optimization.is_some_and(|report| {
        report
            .advisories
            .iter()
            .any(|advisory| advisory.starts_with(CROSSOVER_TIMING_REFUSED_ADVISORY_PREFIX))
    })
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
    optimization: Option<&engine_home_cinema::BassManagementOptimizationReport>,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
) -> Result<HashMap<String, Curve>> {
    reconstruct_deployed_source_curves_impl(
        channels,
        fir_coeffs_by_channel,
        graph,
        optimization,
        sample_rate,
        sidecar_dir,
        SpliceSafety::Strict,
    )
}

/// Ungated deployed reconstruction for calibration and diagnostic callers.
/// The returned curves describe the graph, but do not establish splice safety.
/// Acceptance callers use numeric source-specific baseline comparisons.
pub(crate) fn reconstruct_deployed_source_curves_unenforced(
    channels: &HashMap<String, ChannelDspChain>,
    fir_coeffs_by_channel: &HashMap<String, Vec<f64>>,
    graph: &BassManagementRoutingGraph,
    optimization: Option<&engine_home_cinema::BassManagementOptimizationReport>,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
) -> Result<HashMap<String, Curve>> {
    reconstruct_deployed_source_curves_impl(
        channels,
        fir_coeffs_by_channel,
        graph,
        optimization,
        sample_rate,
        sidecar_dir,
        SpliceSafety::Unenforced,
    )
}

pub(crate) fn reconstruct_deployed_source_curves_with_evidence(
    channels: &HashMap<String, ChannelDspChain>,
    fir_coeffs_by_channel: &HashMap<String, Vec<f64>>,
    graph: &BassManagementRoutingGraph,
    optimization: Option<&engine_home_cinema::BassManagementOptimizationReport>,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
) -> Result<(
    HashMap<String, Curve>,
    Vec<roomeq_model::CrossoverCancellationEvidence>,
)> {
    let mut evidence = Vec::new();
    let curves = reconstruct_deployed_source_curves_impl(
        channels,
        fir_coeffs_by_channel,
        graph,
        optimization,
        sample_rate,
        sidecar_dir,
        SpliceSafety::Record(&mut evidence),
    )?;
    evidence.sort_by(|a, b| a.source_channel.cmp(&b.source_channel));
    Ok((curves, evidence))
}

/// Deployed reconstruction with per-role splice outcomes for every role.
///
/// Unlike the evidence variant, one role's rejection or hard assessment
/// failure never aborts its siblings: every outcome lands in the returned
/// vector in sorted role order. WP4 records these margins at the splice
/// verdict (transfer evidence) and on the published baseline after
/// infeasible selection (crossover residual plus responsible branches).
pub(crate) fn collect_routed_splice_outcomes(
    channels: &HashMap<String, ChannelDspChain>,
    fir_coeffs_by_channel: &HashMap<String, Vec<f64>>,
    graph: &BassManagementRoutingGraph,
    optimization: Option<&engine_home_cinema::BassManagementOptimizationReport>,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
) -> Result<(HashMap<String, Curve>, Vec<RoleSpliceOutcome>)> {
    let mut outcomes = Vec::new();
    let curves = reconstruct_deployed_source_curves_impl(
        channels,
        fir_coeffs_by_channel,
        graph,
        optimization,
        sample_rate,
        sidecar_dir,
        SpliceSafety::Collect(&mut outcomes),
    )?;
    outcomes.sort_by(|a, b| a.role.cmp(&b.role));
    Ok((curves, outcomes))
}

/// One-line splice margin for a collected per-role outcome.
///
/// Accepted roles report underfill against the limit (the transfer margin
/// post-verdict stages preserve); computed-but-rejected roles report the
/// residual; hard failures stay explicitly unassessed, never zero-filled.
fn format_splice_margin(outcome: &RoleSpliceOutcome) -> String {
    match &outcome.assessed {
        Ok(evidence) => {
            let baseline = evidence
                .baseline_db
                .map(|db| format!(":baseline_db={db:.3}"))
                .unwrap_or_default();
            let improvement = evidence
                .improvement_db
                .map(|db| format!(":improvement_db={db:.3}"))
                .unwrap_or_default();
            format!(
                "splice_margin:{}:underfill_db={:.3}:limit_db={:.3}:margin_db={:.3}:worst_hz={:.1}:reason={}{baseline}{improvement}",
                outcome.role,
                evidence.final_db,
                evidence.limit_db,
                evidence.limit_db - evidence.final_db,
                evidence.final_worst_frequency_hz,
                evidence.reason,
            )
        }
        Err(error) => format!("splice_margin:{}:unassessed:{error}", outcome.role),
    }
}

enum SpliceSafety<'a> {
    Strict,
    Record(&'a mut Vec<roomeq_model::CrossoverCancellationEvidence>),
    Collect(&'a mut Vec<RoleSpliceOutcome>),
    Unenforced,
}

impl SpliceSafety<'_> {
    fn enforces(&self, _role: &str) -> bool {
        match self {
            Self::Strict | Self::Record(_) | Self::Collect(_) => true,
            Self::Unenforced => false,
        }
    }
}

/// Per-role splice assessment that never aborts its siblings.
///
/// `Ok` carries the computed cancellation evidence whether or not it met
/// the acceptance limit; `Err` carries a hard assessment failure (missing
/// crossover, mismatched grid) that left the role unassessed. WP4 records
/// these margins at the splice verdict (transfer evidence) and on the
/// published baseline after infeasible selection (crossover residual).
#[derive(Debug, Clone)]
pub(crate) struct RoleSpliceOutcome {
    pub role: String,
    pub assessed: std::result::Result<roomeq_model::CrossoverCancellationEvidence, String>,
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_deployed_source_curves_impl(
    channels: &HashMap<String, ChannelDspChain>,
    fir_coeffs_by_channel: &HashMap<String, Vec<f64>>,
    graph: &BassManagementRoutingGraph,
    optimization: Option<&engine_home_cinema::BassManagementOptimizationReport>,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
    mut splice_safety: SpliceSafety<'_>,
) -> Result<HashMap<String, Curve>> {
    // Stage ownership is part of the emitted routing contract. Fail closed
    // before reconstructing a graph whose untagged correction could belong
    // either before the source split or after a physical output sum.
    validate_routed_plugin_stage_ownership(channels, graph)?;
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
    // Realize drivers once against their own measurements. The common transfer
    // must not multiply the acoustic array by another electrical driver sum.
    let sub_driver_chains = common_sub_chain.drivers.take();
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
    let redirected_destinations = graph
        .routes
        .iter()
        .filter(|route| route.route_kind == "redirected_bass_lowpass_to_sub")
        .map(|route| route.destination.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    // Sum and filter on the receiving main's grid. Interpolating a previously
    // summed dB/phase curve across a sub cancellation changes the acoustic model.
    let realize_sub_on_grid = |frequencies: &ndarray::Array1<f64>| -> Result<Curve> {
        let input = roomeq_engine::topology::interpolate_bass_response(frequencies, &sub_initial);
        let common_sub_curve = crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
            &common_sub_chain,
            &input,
            sample_rate,
            sidecar_dir,
            &common_sub_embedded_irs,
        )?;
        Ok(if redirected_destinations.len() > 1 {
            let optimization = optimization.ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: "missing multi-sub optimization metadata for deployed reconstruction"
                    .to_string(),
            })?;
            let drivers =
                sub_driver_chains
                    .as_ref()
                    .ok_or_else(|| AutoeqError::InvalidConfiguration {
                        message: "missing multi-sub driver chains for deployed reconstruction"
                            .to_string(),
                    })?;
            let mut realized_outputs = Vec::with_capacity(drivers.len());
            for driver in drivers {
                let initial = driver.initial_curve.clone().ok_or_else(|| {
                    AutoeqError::InvalidMeasurement {
                        message: format!(
                            "multi-sub driver '{}' has no initial curve for reconstruction",
                            driver.name
                        ),
                    }
                })?;
                let mut curve = roomeq_engine::topology::interpolate_bass_response(
                    &common_sub_curve.freq,
                    &Curve::from(initial),
                );
                let output = optimization
                    .sub_output_results
                    .iter()
                    .find(|output| output.output_role == driver.name)
                    .ok_or_else(|| AutoeqError::InvalidConfiguration {
                        message: format!(
                            "missing multi-sub output settings for driver '{}'",
                            driver.name
                        ),
                    })?;
                let mut filter_chain = common_sub_chain.clone();
                filter_chain.drivers = None;
                filter_chain.plugins = driver
                    .plugins
                    .iter()
                    .filter(|plugin| {
                        (plugin.plugin_type != "gain"
                            || plugin
                                .parameters
                                .get("room_eq_correction_gain")
                                .and_then(|v| v.as_bool())
                                == Some(true))
                            && plugin.plugin_type != "delay"
                            && plugin.plugin_type != "crossover"
                    })
                    .cloned()
                    .collect();
                curve = crate::ctc::apply_channel_dsp_chain_to_curve_with_sidecar_dir(
                    &filter_chain,
                    &curve,
                    sample_rate,
                    sidecar_dir,
                )?;
                curve.spl.mapv_inplace(|level| level + output.gain_db);
                let mut curve = roomeq_engine::topology::apply_delay_and_polarity_to_curve(
                    &curve,
                    output.delay_ms,
                    output.polarity_inverted,
                );
                // Realize each physical driver's low-pass before the sum.
                // Redirected routes omit a second group LP; LFE retains its
                // independent cutoff, exactly as in the exported graph.
                if let Some((frequency, crossover_type)) = output::driver_low_pass(driver) {
                    curve = apply_crossover_response_to_curve(
                        &curve,
                        &crossover_type,
                        frequency,
                        sample_rate,
                        true,
                    );
                }
                realized_outputs.push(curve);
            }
            let output_refs = realized_outputs.iter().collect::<Vec<_>>();
            // The multi-output replay is a coherent sum and needs measured
            // phase on every driver. Genuinely phaseless measurements must
            // fail here with a descriptive error, never a rayon panic.
            if let Some(driver) = drivers
                .iter()
                .zip(output_refs.iter())
                .find_map(|(driver, curve)| (!curve_has_usable_phase(curve)).then_some(driver))
            {
                return Err(AutoeqError::InvalidMeasurement {
                    message: format!(
                        "multi-sub driver '{}' has no usable phase for deployed reconstruction; \
                         coherent replay of {} sub outputs requires measured phase on every driver",
                        driver.name,
                        output_refs.len(),
                    ),
                });
            }
            let mut combined = complex_sum_mains(&output_refs);
            let initial_on_grid = &input;
            combined.spl = &combined.spl + &common_sub_curve.spl - &initial_on_grid.spl;
            if let (Some(combined_phase), Some(common_phase), Some(initial_phase)) = (
                combined.phase.as_mut(),
                common_sub_curve.phase.as_ref(),
                initial_on_grid.phase.as_ref(),
            ) {
                *combined_phase = &*combined_phase + common_phase - initial_phase;
            }
            combined
        } else {
            common_sub_curve.clone()
        })
    };
    let realize_sub_outputs_on_grid =
        |frequencies: &ndarray::Array1<f64>| -> Result<(HashMap<String, Curve>, Curve)> {
            let single_output_fallback = (graph.physical_sub_outputs.len() <= 1)
                .then(|| realize_sub_on_grid(frequencies))
                .transpose()?;
            let input =
                roomeq_engine::topology::interpolate_bass_response(frequencies, &sub_initial);
            let common = crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
                &common_sub_chain,
                &input,
                sample_rate,
                sidecar_dir,
                &common_sub_embedded_irs,
            )?;
            let Some(drivers) = sub_driver_chains.as_ref() else {
                return Ok((HashMap::new(), single_output_fallback.unwrap_or(common)));
            };
            let mut outputs = HashMap::with_capacity(drivers.len());
            for driver in drivers {
                let initial = driver.initial_curve.clone().ok_or_else(|| {
                    AutoeqError::InvalidMeasurement {
                        message: format!(
                            "multi-sub driver '{}' has no initial curve for deployed reconstruction",
                            driver.name
                        ),
                    }
                })?;
                let mut curve = roomeq_engine::topology::interpolate_bass_response(
                    &common.freq,
                    &Curve::from(initial),
                );
                let mut filter_chain = common_sub_chain.clone();
                filter_chain.drivers = None;
                filter_chain.plugins = driver
                    .plugins
                    .iter()
                    .filter(|plugin| {
                        (plugin.plugin_type != "gain"
                            || plugin
                                .parameters
                                .get("room_eq_correction_gain")
                                .and_then(|value| value.as_bool())
                                == Some(true))
                            && plugin.plugin_type != "delay"
                            && plugin.plugin_type != "crossover"
                    })
                    .cloned()
                    .collect();
                curve = crate::ctc::apply_channel_dsp_chain_to_curve_with_sidecar_dir(
                    &filter_chain,
                    &curve,
                    sample_rate,
                    sidecar_dir,
                )?;
                if let Some((frequency, crossover_type)) = output::driver_low_pass(driver) {
                    curve = apply_crossover_response_to_curve(
                        &curve,
                        &crossover_type,
                        frequency,
                        sample_rate,
                        true,
                    );
                }
                if !curve_has_usable_phase(&curve) {
                    return Err(AutoeqError::InvalidMeasurement {
                        message: format!(
                            "multi-sub driver '{}' has no usable phase for deployed reconstruction",
                            driver.name
                        ),
                    });
                }
                // The common post-route transfer belongs to every physical
                // output, but route gain/delay/polarity remain owned by the
                // matrix edge and are applied by the replay function below.
                curve.spl = &curve.spl + &common.spl - &input.spl;
                if let (Some(phase), Some(common_phase), Some(input_phase)) = (
                    curve.phase.as_mut(),
                    common.phase.as_ref(),
                    input.phase.as_ref(),
                ) {
                    *phase = &*phase + common_phase - input_phase;
                }
                outputs.insert(driver.name.clone(), curve);
            }
            Ok((outputs, single_output_fallback.unwrap_or(common)))
        };
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
    source_roles
        .into_iter()
        .map(|role| {
            let main_curve = if role == *lfe_role
            || engine_home_cinema::role_for_channel(&role)
                == roomeq_model::HomeCinemaRole::Lfe
        {
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
                // The physical routing contract places source `pre_route`
                // processing before the route matrix and output `post_route`
                // processing after it. A main channel's correction EQ is
                // output-owned: realize the direct high-pass branch first,
                // then apply only its output stage before summing redirected
                // bass below.
                let mut main_route_chain = chain.clone();
                main_route_chain.plugins.retain(|plugin| {
                    is_source_pre_route_plugin(plugin)
                        || plugin
                            .parameters
                            .get("room_eq_stage")
                            .and_then(serde_json::Value::as_str)
                            == Some("route_owned")
                });
                let route_embedded_irs = embedded_convolution_irs(
                    &main_route_chain.plugins,
                    fir_coeffs_by_channel.get(&role).map(Vec::as_slice),
                )?;
                let direct_main =
                    crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
                        &main_route_chain,
                        &initial,
                        sample_rate,
                        sidecar_dir,
                        &route_embedded_irs,
                    )?;
                let mut main_output_chain = chain.clone();
                main_output_chain
                    .plugins
                    .retain(is_source_post_route_plugin);
                let output_embedded_irs = embedded_convolution_irs(
                    &main_output_chain.plugins,
                    fir_coeffs_by_channel.get(&role).map(Vec::as_slice),
                )?;
                Some(
                    crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
                        &main_output_chain,
                        &direct_main,
                        sample_rate,
                        sidecar_dir,
                        &output_embedded_irs,
                    )?,
                )
            };
            let frequencies = main_curve.as_ref().map(|main| &main.freq).unwrap_or(&common_sub_curve.freq);
            let (sub_output_curves, routed_common_sub_curve) =
                realize_sub_outputs_on_grid(frequencies)?;
            let source_transfers = source_pre_route_transfers(
                channels, fir_coeffs_by_channel, [role.clone()], &routed_common_sub_curve,
                sample_rate, sidecar_dir,
            )?;
            let bass = engine_bass_management::predict_bass_source_curve_from_output_routes(
                &routed_common_sub_curve,
                &sub_output_curves,
                &routed_common_sub_curve,
                source_transfers.get(&role),
                graph,
                &role,
                sample_rate,
            )
            .ok_or_else(|| AutoeqError::InvalidMeasurement {
                message: format!("could not reconstruct routed bass branch '{role}'"),
            })?;
            let deployed = main_curve
                .as_ref()
                .map(|main| complex_sum_mains(&[main, &bass]))
                .unwrap_or_else(|| bass.clone());
            if let Some(main) = main_curve.as_ref() && splice_safety.enforces(&role) {
                let crossover_lookup = graph
                    .routes
                    .iter()
                    .find(|route| {
                        route.source_channel == role
                            && route.route_kind == "redirected_bass_lowpass_to_sub"
                    })
                    .and_then(|route| route.low_pass_hz)
                    .or_else(|| graph.routes.iter().find(|route| route.source_channel == role && route.route_kind == "main_highpass_to_self").and_then(|route| route.high_pass_hz));
                let assessed: std::result::Result<
                    (f64, roomeq_model::CrossoverCancellationEvidence),
                    AutoeqError,
                > = (|| {
                    let crossover_hz = crossover_lookup.ok_or_else(|| {
                        AutoeqError::InvalidConfiguration {
                            message: format!("missing routed crossover frequency for '{role}'"),
                        }
                    })?;
                    let main_on_deployed_grid =
                        autoeq_core::curve_transforms::interpolate_log_space(&deployed.freq, main);
                    let bass_on_deployed_grid =
                        autoeq_core::curve_transforms::interpolate_log_space(&deployed.freq, &bass);
                    let reconstructed =
                        complex_sum_mains(&[&main_on_deployed_grid, &bass_on_deployed_grid]);
                    let cancellation =
                        roomeq_engine::topology::assess_crossover_cancellation(
                            &role,
                            &main_on_deployed_grid,
                            &bass_on_deployed_grid,
                            &reconstructed,
                            crossover_hz,
                            optimization.and_then(|o| o.crossover_cancellation.as_ref()),
                        )
                            .ok_or_else(|| AutoeqError::InvalidMeasurement {
                                message: format!("mismatched crossover reconstruction grid for '{role}'"),
                            })?;
                    Ok((crossover_hz, cancellation))
                })();
                // Collect mode records every role (computed or hard-failed)
                // without aborting its siblings; all other modes keep their
                // exact first-failure behavior.
                if let SpliceSafety::Collect(outcomes) = &mut splice_safety {
                    outcomes.push(RoleSpliceOutcome {
                        role: role.clone(),
                        assessed: assessed
                            .map(|(_, cancellation)| cancellation)
                            .map_err(|error| error.to_string()),
                    });
                } else {
                    let (crossover_hz, cancellation) = assessed?;
                    let underfill_db = cancellation.final_db;
                    let worst_frequency_hz = cancellation.final_worst_frequency_hz;
                    if !cancellation.accepted

                    {
                        return Err(AutoeqError::OptimizationFailed {
                            message: format!(
                                "final routed crossover underfill for '{role}' is \
                                 {underfill_db:.3} dB at {worst_frequency_hz:.1} Hz (crossover {crossover_hz:.1} Hz; limit {:.1} dB; baseline {:?} dB; improvement {:?} dB required >0.05 dB; {})",
                                cancellation.limit_db, cancellation.baseline_db, cancellation.improvement_db, cancellation.reason
                            ),
                        });
                    }
                    if cancellation.reason == "improved_residual_cancellation" {
                        log::warn!(target: BASS_MANAGEMENT_LOG_TARGET, "Crossover cancellation for '{role}' accepted as improved residual: baseline={:?} dB, final={underfill_db:.3} dB, limit={:.3} dB", cancellation.baseline_db, cancellation.limit_db);
                    }
                    if let SpliceSafety::Record(evidence) = &mut splice_safety {
                        evidence.push(cancellation);
                    }
                }
            if let Some(crossover_hz) = crossover_lookup
                && let Some(target) = channels
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
                // Target mismatch is a correction-quality limitation, not a
                // routing-safety failure. Deep measured room nulls can remain
                // after bounded EQ even when the main/sub crossover sums
                // safely. Keep the truthful deployed curve and surface the
                // residual instead of suppressing the complete DSP artifact.
                log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
                    "  Final routed target underfill for '{}' is {:.3} dB at {:.1} Hz (quality target {:.1} dB)",
                    role,
                    target_underfill_db,
                    crossover_hz,
                    roomeq_engine::topology::MAX_ACCEPTED_CROSSOVER_UNDERFILL_DB,
                );
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
    optimization: Option<&engine_home_cinema::BassManagementOptimizationReport>,
) -> Result<(HashMap<String, f64>, HashMap<String, Curve>)> {
    // Calibration uses the same per-input acoustic graph as final replay.
    // The splice gate runs after correction rollback; this step only selects
    // common logical-input level trims and must never sum unrelated inputs.
    let before = reconstruct_deployed_source_curves_unenforced(
        channels,
        fir_coeffs_by_channel,
        graph,
        optimization,
        sample_rate,
        sidecar_dir,
    )?;
    let means: HashMap<String, f64> = main_roles
        .iter()
        .filter_map(|role| {
            before
                .get(role)
                .map(|curve| (role.clone(), average_spl(curve, main_band)))
        })
        .collect();
    let target = means.values().copied().fold(f64::INFINITY, f64::min);
    if !target.is_finite() || means.len() != main_roles.len() {
        return Ok((HashMap::new(), before));
    }
    let mut trims: HashMap<String, f64> = means
        .into_iter()
        .map(|(role, mean)| (role, (target - mean).min(0.0)))
        .collect();
    trims.insert(lfe_role.to_string(), 0.0);
    let logical_input_trims = trims.clone();
    // The correlated-bus simulator reads these source trims from the graph.
    // Publish the freshly computed pre-route values before invoking it so a
    // repeated finalizer trial cannot reuse the previous trial's route basis.
    graph.input_trim_db.clone_from(&logical_input_trims);
    graph.post_dsp_main_alignment_band_hz = Some([main_band.0, main_band.1]);

    for role in main_roles {
        if let Some(chain) = channels.get_mut(role) {
            apply_gain_to_main_chain(chain, *trims.get(role).unwrap_or(&0.0));
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
    graph
        .advisories
        .push("post_dsp_input_levels_aligned_down".to_string());

    let deployed_source_curves = reconstruct_deployed_source_curves_unenforced(
        channels,
        fir_coeffs_by_channel,
        graph,
        optimization,
        sample_rate,
        sidecar_dir,
    )?;
    if let (Some(curve), Some(chain)) = (
        deployed_source_curves.get(lfe_role),
        channels.get_mut(lfe_role),
    ) {
        chain.final_curve = Some(curve.into());
    }
    Ok((trims, deployed_source_curves))
}

/// Recompute Home Cinema post-DSP level gains for a finalizer trial.
///
/// This removes only gains and routing-graph state generated by the original
/// post-DSP calibration pass. The measured calibration band is retained on the
/// routing graph so candidate rebuilding does not infer it from an expanded
/// response grid or held-out measurements.
pub(crate) fn recalibrate_post_dsp_levels(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
) -> Result<bool> {
    let Some(bass_report) = result.metadata.bass_management.as_ref() else {
        return Ok(false);
    };
    let Some(mut graph) = bass_report.routing_graph.clone() else {
        if bass_report.enabled {
            return Err(AutoeqError::InvalidConfiguration {
                message: "cannot recalculate Home Cinema level gains without a routing graph"
                    .into(),
            });
        }
        return Ok(false);
    };

    let has_alignment_marker = graph
        .advisories
        .iter()
        .any(|advisory| advisory == "post_dsp_input_levels_aligned_down");
    let has_derived_gain = result.channels.values().any(|chain| {
        chain.plugins.iter().any(|plugin| {
            matches!(
                plugin
                    .parameters
                    .get("label")
                    .and_then(serde_json::Value::as_str),
                Some("post_dsp_input_level_alignment" | "post_dsp_output_headroom_safety")
            )
        })
    });
    let has_derived_graph_state = !graph.input_trim_db.is_empty()
        || graph
            .advisories
            .iter()
            .any(|advisory| advisory.starts_with("common_input_headroom_safety_trim_db:"));
    if !has_alignment_marker && !has_derived_gain && !has_derived_graph_state {
        return Ok(false);
    }
    let Some([band_low_hz, band_high_hz]) = graph.post_dsp_main_alignment_band_hz else {
        return Err(AutoeqError::InvalidConfiguration {
            message: "cannot recalculate Home Cinema level gains: original measured alignment band is unavailable"
                .into(),
        });
    };
    if !band_low_hz.is_finite()
        || !band_high_hz.is_finite()
        || band_low_hz <= 0.0
        || band_low_hz >= band_high_hz
    {
        return Err(AutoeqError::InvalidConfiguration {
            message:
                "cannot recalculate Home Cinema level gains: recorded alignment band is invalid"
                    .into(),
        });
    }

    let system = config
        .system
        .as_ref()
        .ok_or_else(|| AutoeqError::InvalidConfiguration {
            message: "cannot resolve Home Cinema level-gain roles without a system config".into(),
        })?;
    let sub_role = engine_home_cinema::bass_output_role(config, system);
    if graph.physical_sub_output != sub_role
        || !graph.physical_sub_outputs.contains(&sub_role)
        || !result.channels.contains_key(&sub_role)
    {
        return Err(AutoeqError::InvalidConfiguration {
            message: "Home Cinema routing graph does not match the configured physical sub output"
                .into(),
        });
    }
    let main_roles = canonical_main_roles(system, &sub_role);
    let graph_main_roles: std::collections::BTreeSet<_> = graph
        .routes
        .iter()
        .filter(|route| route.route_kind == "main_highpass_to_self")
        .map(|route| route.source_channel.as_str())
        .collect();
    if main_roles.is_empty()
        || main_roles
            .iter()
            .any(|role| !graph.input_channels.contains(role))
        || main_roles
            .iter()
            .any(|role| !graph_main_roles.contains(role.as_str()))
        || main_roles
            .iter()
            .any(|role| !result.channels.contains_key(role))
    {
        return Err(AutoeqError::InvalidConfiguration {
            message: "Home Cinema level-gain roles do not match the retained routing graph".into(),
        });
    }

    let mut invalid_derived_gain = None;
    for (channel, chain) in &mut result.channels {
        chain.plugins.retain(|plugin| {
            let label = plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str);
            let expected_stage = match label {
                Some("post_dsp_input_level_alignment") => Some("pre_route"),
                Some("post_dsp_output_headroom_safety") => Some("post_route"),
                _ => None,
            };
            let Some(expected_stage) = expected_stage else {
                return true;
            };
            let actual_stage = plugin
                .parameters
                .get("room_eq_stage")
                .and_then(serde_json::Value::as_str);
            if actual_stage != Some(expected_stage) {
                invalid_derived_gain = Some(channel.clone());
                return true;
            }
            false
        });
        if let Some(drivers) = &chain.drivers
            && drivers.iter().any(|driver| {
                driver.plugins.iter().any(|plugin| {
                    matches!(
                        plugin
                            .parameters
                            .get("label")
                            .and_then(serde_json::Value::as_str),
                        Some("post_dsp_input_level_alignment" | "post_dsp_output_headroom_safety")
                    )
                })
            })
        {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!(
                    "derived Home Cinema level gain is on unsupported driver chain '{channel}'"
                ),
            });
        }
    }
    if let Some(channel) = invalid_derived_gain {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "derived Home Cinema level gain on '{channel}' has an unexpected route stage"
            ),
        });
    }

    graph.input_trim_db.clear();
    graph.advisories.retain(|advisory| {
        advisory != "post_dsp_input_levels_aligned_down"
            && !advisory.starts_with("common_input_headroom_safety_trim_db:")
    });
    let optimization = bass_report.optimization.clone();
    let fir_coeffs_by_channel: HashMap<_, _> = result
        .channel_results
        .iter()
        .filter_map(|(role, channel)| {
            channel
                .fir_coeffs
                .clone()
                .map(|coefficients| (role.clone(), coefficients))
        })
        .collect();
    let (_, deployed_source_curves) = calibrate_post_dsp_input_levels(
        config,
        &main_roles,
        &sub_role,
        (band_low_hz, band_high_hz),
        sample_rate,
        sidecar_dir,
        &fir_coeffs_by_channel,
        &mut result.channels,
        &mut graph,
        optimization.as_ref(),
    )?;
    result.deployed_source_curves = deployed_source_curves;

    let effective = engine_home_cinema::effective_bass_management(config).ok_or_else(|| {
        AutoeqError::InvalidConfiguration {
            message: "cannot recalculate Home Cinema level gains without effective bass management"
                .into(),
        }
    })?;
    let headroom_simulation = engine_home_cinema::simulate_bass_bus_headroom(
        Some(&graph),
        &effective.config.headroom_model,
        effective.config.headroom_margin_db,
        sample_rate,
    );
    // This is the correlated-bus margin before the separate post-route safety
    // gains. Those gains remain explicit in the channel chains and are checked
    // by the final electrical/replay gates.
    let report = result
        .metadata
        .bass_management
        .as_mut()
        .expect("report was present while rebuilding Home Cinema levels");
    report.routing_graph = Some(graph);
    report.headroom_simulation = headroom_simulation;
    let mut advisory_parts: Vec<_> = report
        .advisory
        .split(';')
        .filter(|part| {
            *part != "post_dsp_input_levels_aligned_down"
                && !part.starts_with("common_input_headroom_safety_trim_db:")
        })
        .filter(|part| !part.is_empty())
        .collect();
    advisory_parts.push("post_dsp_input_levels_aligned_down");
    report.advisory = advisory_parts.join(";");
    Ok(true)
}

fn stereo_route_objective(sources: &[BassManagementSourceReport]) -> Option<f64> {
    let values = sources
        .iter()
        .map(|source| source.objective_after.or(source.objective_before))
        .collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then(|| values.into_iter().sum())
}

fn stereo_route_headroom_margin_db(
    config: &RoomConfig,
    matrix: &[Vec<f64>],
    outputs: &[BassManagementSubOutputReport],
) -> f64 {
    let configured_margin = config
        .system
        .as_ref()
        .and_then(|system| system.bass_management.as_ref())
        .map(|policy| policy.headroom_margin_db)
        .unwrap_or(6.0);
    let coherent_peak = matrix
        .iter()
        .zip(outputs)
        .map(|(row, output)| {
            row.iter().map(|coefficient| coefficient.abs()).sum::<f64>()
                * 10.0_f64.powf(output.gain_db.max(0.0) / 20.0)
        })
        .fold(0.0_f64, f64::max);
    let peak_gain_db = if coherent_peak > 0.0 {
        20.0 * coherent_peak.log10()
    } else {
        0.0
    };
    configured_margin - peak_gain_db
}

#[allow(clippy::too_many_arguments)]
fn stereo_candidate_serialized_replay_rejection(
    config: &RoomConfig,
    main_roles: &[String],
    aligned_pre_eq_curves: &HashMap<String, Curve>,
    source_pre_route_transfers: &HashMap<String, Curve>,
    groups: &std::collections::BTreeMap<String, roomeq_model::BassManagementGroupReport>,
    sources: &[BassManagementSourceReport],
    outputs: &[BassManagementSubOutputReport],
    drivers: &[engine_bass_management::SubDriverInfo],
    common_sub_correction: Option<(&Curve, &Curve)>,
    matrix: &[Vec<f64>],
    sample_rate: f64,
) -> std::result::Result<(), String> {
    if main_roles.len() != 2
        || outputs.len() != 2
        || drivers.len() != outputs.len()
        || matrix.len() != outputs.len()
        || matrix.iter().any(|row| row.len() != main_roles.len())
    {
        return Err("serialized_replay_shape_mismatch".to_string());
    }

    let representative_group = sources
        .first()
        .and_then(|source| groups.get(&source.group_id))
        .ok_or_else(|| "serialized_replay_missing_group".to_string())?;
    let mut output_curves = HashMap::with_capacity(outputs.len());
    for (driver, output) in drivers.iter().zip(outputs) {
        if driver.name != output.output_role {
            return Err(format!(
                "serialized_replay_output_identity_mismatch:{}:{}",
                driver.name, output.output_role
            ));
        }
        let mut curve = driver
            .processing
            .as_ref()
            .map(|processing| processing.curve.clone())
            .or_else(|| driver.initial_curve.clone())
            .ok_or_else(|| {
                format!(
                    "serialized_replay_missing_output_measurement:{}",
                    driver.name
                )
            })?;
        if let Some((measured, corrected)) = common_sub_correction {
            curve =
                engine_bass_management::apply_common_sub_correction(&curve, measured, corrected)
                    .ok_or_else(|| "serialized_replay_invalid_common_sub_correction".to_string())?;
        }
        if let Some(low_pass_hz) = output.selected_low_pass_hz {
            curve = apply_crossover_response_to_curve(
                &curve,
                &representative_group.crossover_type,
                low_pass_hz,
                sample_rate,
                true,
            );
        }
        output_curves.insert(output.output_role.clone(), curve);
    }

    let mut output_channels = main_roles.to_vec();
    output_channels.extend(outputs.iter().map(|output| output.output_role.clone()));
    let mut routes = Vec::new();
    let mut input_channel_map = Vec::new();
    let mut output_channel_map = Vec::new();
    let mut serialized_matrix = Vec::new();
    for (source_index, role) in main_roles.iter().enumerate() {
        let source = sources
            .iter()
            .find(|source| source.source_channel == *role)
            .ok_or_else(|| format!("serialized_replay_missing_source:{role}"))?;
        let group = groups
            .get(&source.group_id)
            .ok_or_else(|| format!("serialized_replay_missing_group:{}", source.group_id))?;
        let crossover_hz = group
            .selected_crossover_hz
            .ok_or_else(|| format!("serialized_replay_missing_crossover:{}", group.group_id))?;
        routes.push(BassManagementRoute {
            group_id: Some(group.group_id.clone()),
            source_channel: role.clone(),
            source_index,
            destination: role.clone(),
            destination_index: source_index,
            pre_chain_channel: Some(role.clone()),
            post_chain_channel: Some(role.clone()),
            route_kind: "main_highpass_to_self".to_string(),
            crossover_type: group.crossover_type.clone(),
            high_pass_hz: Some(crossover_hz),
            low_pass_hz: None,
            gain_db: 0.0,
            gain_linear: 1.0,
            matrix_gain: 1.0,
            delay_ms: source.main_delay_ms,
            polarity_inverted: false,
        });
        for (output_index, output) in outputs.iter().enumerate() {
            let coefficient = matrix[output_index][source_index];
            if coefficient <= f64::EPSILON {
                continue;
            }
            let gain_db = source.trim_db + output.gain_db + 20.0 * coefficient.log10();
            let gain_linear = 10.0_f64.powf(gain_db / 20.0);
            let destination_index = main_roles.len() + output_index;
            routes.push(BassManagementRoute {
                group_id: Some(group.group_id.clone()),
                source_channel: role.clone(),
                source_index,
                destination: output.output_role.clone(),
                destination_index,
                pre_chain_channel: Some(role.clone()),
                post_chain_channel: Some(output.output_role.clone()),
                route_kind: "redirected_bass_lowpass_to_sub".to_string(),
                crossover_type: group.crossover_type.clone(),
                high_pass_hz: None,
                low_pass_hz: output
                    .selected_low_pass_hz
                    .is_none()
                    .then_some(crossover_hz),
                gain_db,
                gain_linear,
                matrix_gain: gain_linear,
                delay_ms: source.bass_route_delay_ms + output.delay_ms,
                polarity_inverted: source.polarity_inverted ^ output.polarity_inverted,
            });
            input_channel_map.push(source_index);
            output_channel_map.push(destination_index);
            serialized_matrix.push(gain_linear as f32);
        }
    }
    let graph = BassManagementRoutingGraph {
        physical_sub_output: outputs[0].output_role.clone(),
        physical_sub_outputs: outputs
            .iter()
            .map(|output| output.output_role.clone())
            .collect(),
        input_channels: main_roles.to_vec(),
        output_channels,
        routes,
        matrix: Some(BassManagementMatrix {
            input_channel_map,
            output_channel_map,
            matrix: serialized_matrix,
            route_count: outputs
                .iter()
                .enumerate()
                .map(|(output_index, _)| {
                    (0..main_roles.len())
                        .filter(|source_index| matrix[output_index][*source_index] > f64::EPSILON)
                        .count()
                })
                .sum(),
        }),
        input_trim_db: HashMap::new(),
        post_dsp_main_alignment_band_hz: None,
        stereo_routing: None,
        advisories: Vec::new(),
    };
    let serialized = serde_json::to_vec(&graph)
        .map_err(|error| format!("serialized_replay_encode_failed:{error}"))?;
    let graph: BassManagementRoutingGraph = serde_json::from_slice(&serialized)
        .map_err(|error| format!("serialized_replay_decode_failed:{error}"))?;
    let fallback = output_curves
        .get(&outputs[0].output_role)
        .ok_or_else(|| "serialized_replay_missing_fallback_output".to_string())?;

    for role in main_roles {
        let source = sources
            .iter()
            .find(|source| source.source_channel == *role)
            .ok_or_else(|| format!("serialized_replay_missing_source:{role}"))?;
        let group = groups
            .get(&source.group_id)
            .ok_or_else(|| format!("serialized_replay_missing_group:{}", source.group_id))?;
        let crossover_hz = group
            .selected_crossover_hz
            .ok_or_else(|| format!("serialized_replay_missing_crossover:{}", group.group_id))?;
        let main = aligned_pre_eq_curves
            .get(role)
            .ok_or_else(|| format!("serialized_replay_missing_main:{role}"))?;
        let main_branch = apply_delay_and_polarity_to_curve(
            &apply_crossover_response_to_curve(
                main,
                &group.crossover_type,
                crossover_hz,
                sample_rate,
                false,
            ),
            source.main_delay_ms,
            false,
        );
        let bass_branch = engine_bass_management::predict_bass_source_curve_from_output_routes(
            &main_branch,
            &output_curves,
            fallback,
            source_pre_route_transfers.get(role),
            &graph,
            role,
            sample_rate,
        )
        .ok_or_else(|| format!("serialized_replay_failed_bass_branch:{role}"))?;
        let combined = complex_sum_mains(&[&main_branch, &bass_branch]);
        let evidence = roomeq_engine::topology::assess_configured_crossover_cancellation(
            config,
            role,
            &main_branch,
            &bass_branch,
            &combined,
            crossover_hz,
        )
        .ok_or_else(|| format!("serialized_replay_failed_splice_metric:{role}"))?;
        let underfill = evidence.final_db;
        if !evidence.accepted {
            return Err(format!(
                "serialized_graph_acoustic_splice_underfill:{role}:{underfill:.3}db"
            ));
        }
    }
    Ok(())
}

fn physical_sub_speaker_config(
    config: &RoomConfig,
    sys: &SystemConfig,
) -> Result<Option<SpeakerConfig>> {
    let Some(subwoofers) = sys.subwoofers.as_ref() else {
        return Ok(None);
    };
    if subwoofers.outputs.is_empty() {
        return Ok(None);
    }

    let mut configs = subwoofers
        .outputs
        .iter()
        .map(|output| {
            config
                .speakers
                .get(&output.speaker)
                .cloned()
                .ok_or_else(|| AutoeqError::InvalidConfiguration {
                    message: format!(
                        "Physical sub output '{}' references missing speaker config '{}'",
                        output.id, output.speaker
                    ),
                })
        })
        .collect::<Result<Vec<_>>>()?;

    if configs.len() == 1 {
        return Ok(configs.pop());
    }

    // A grouped measurement owns several physical outputs. Keep its topology
    // (including DBA/cardioid controls and per-sub all-pass optimization) when
    // every declared output references that same group.
    if subwoofers
        .outputs
        .iter()
        .all(|output| output.speaker == subwoofers.outputs[0].speaker)
    {
        let branches = match &configs[0] {
            SpeakerConfig::MultiSub(group) => Some(group.subwoofers.len()),
            SpeakerConfig::Dba(group) => Some(group.front.len() + group.rear.len()),
            SpeakerConfig::Cardioid(_) => Some(2),
            _ => None,
        };
        if let Some(branches) = branches {
            if branches != configs.len() {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "Physical sub group has {branches} branches but {} outputs were declared",
                        configs.len()
                    ),
                });
            }
            return Ok(configs.pop());
        }
    }

    let mut measurements = Vec::with_capacity(configs.len());
    for (output, speaker) in subwoofers.outputs.iter().zip(configs) {
        let SpeakerConfig::Single(source) = speaker else {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!(
                    "Physical sub output '{}' must reference a single measurement when multiple outputs are configured",
                    output.id
                ),
            });
        };
        measurements.push(source);
    }

    Ok(Some(SpeakerConfig::MultiSub(MultiSubGroup {
        name: "physical_sub_outputs".to_string(),
        speaker_name: None,
        subwoofers: measurements,
        allpass_optimization: false,
        // Physical-output grouping, not a user joint request: the
        // detailed mode stays authoritative here.
        joint_optimization: false,
    })))
}

// Coherent predictions need an actual capture, never a spatial power average.
// A single capture retains the existing single-position workflow semantics.
fn load_primary_crossover_curve(
    source: &MeasurementSource,
    primary_seat: usize,
    frequency_samples: usize,
    channel: &str,
) -> Result<Curve> {
    let individual = load_source_individual_with_frequency_samples(source, frequency_samples)
        .map_err(|error| AutoeqError::InvalidMeasurement {
            message: error.to_string(),
        })?;
    let index = if individual.len() == 1 {
        0
    } else {
        primary_seat
    };
    individual.get(index).cloned().ok_or_else(|| {
        AutoeqError::InvalidConfiguration {
            message: format!(
                "primary seat {primary_seat} unavailable for home-cinema channel '{channel}' with {} measurement(s)",
                individual.len()
            ),
        }
    })
}

fn capture_crossover_cancellation_baseline(
    config: &RoomConfig,
    mains: &HashMap<String, Curve>,
    fs: f64,
    frequency_samples: usize,
) -> Result<roomeq_model::CrossoverCancellationContext> {
    let mut context = roomeq_model::CrossoverCancellationContext {
        limit_db: config.optimizer.max_crossover_cancellation_db,
        sources: Default::default(),
    };
    let Some(system) = &config.system else {
        return Ok(context);
    };
    let Some(subs) = &system.subwoofers else {
        return Ok(context);
    };
    let primary_seat = config
        .optimizer
        .multi_seat
        .as_ref()
        .map_or(0, |seat| seat.primary_seat);
    // Automatic searches start at the configured range's geometric centre.
    let mut baseline_config = config.clone();
    if let Some(crossovers) = baseline_config.crossovers.as_mut() {
        for crossover in crossovers.values_mut() {
            if crossover.frequency.is_none() {
                crossover.frequency = crossover.frequency_range.map(|(lo, hi)| (lo * hi).sqrt());
            }
            crossover.crossover_type =
                roomeq_engine::topology::bass_management_crossover_type_candidates(
                    &crossover.crossover_type,
                )[0]
                .clone();
        }
    }
    let Some(mut graph) = engine_home_cinema::bass_management_routing_graph(&baseline_config, None)
    else {
        return Ok(context);
    };
    let mut outputs = HashMap::new();
    for output in &subs.outputs {
        if let Some(SpeakerConfig::Cardioid(c)) = config.speakers.get(&output.speaker) {
            outputs.insert(
                output.id.clone(),
                preprocess_cardioid_with_frequency_samples(c, frequency_samples, primary_seat)?
                    .combined_curve,
            );
            continue;
        }
        if let Some(SpeakerConfig::Dba(d)) = config.speakers.get(&output.speaker) {
            let mut branches = Vec::new();
            for (sources, rear) in [(&d.front, false), (&d.rear, true)] {
                for source in sources {
                    let raw = load_primary_crossover_curve(
                        source,
                        primary_seat,
                        frequency_samples,
                        &output.id,
                    )?;
                    let mut curve = apply_delay_and_polarity_to_curve(
                        &raw,
                        if rear { 10.0 } else { 0.0 },
                        rear,
                    );
                    if rear {
                        curve
                            .spl
                            .mapv_inplace(|v| v + (-3.0_f64).clamp(config.optimizer.min_db, 0.0));
                    }
                    branches.push(curve);
                }
            }
            if !branches.is_empty() && branches.iter().all(curve_has_usable_phase) {
                let refs = branches.iter().collect::<Vec<_>>();
                if let Some(grid) = roomeq_engine::topology::shared_measurement_grid(&refs) {
                    let aligned: Vec<_> = branches
                        .iter()
                        .map(|c| autoeq_core::curve_transforms::interpolate_log_space(&grid, c))
                        .collect();
                    outputs.insert(
                        output.id.clone(),
                        complex_sum_mains(&aligned.iter().collect::<Vec<_>>()),
                    );
                }
            }
            continue;
        }
        let sources: Vec<&MeasurementSource> = match config.speakers.get(&output.speaker) {
            Some(SpeakerConfig::Single(s)) => vec![s],
            Some(SpeakerConfig::MultiSub(group)) => group.subwoofers.iter().collect(),
            // These layouts require their own structural array controls. Do not
            // substitute an optimized response or assume identity controls.
            _ => continue,
        };
        let raw: Vec<Curve> = sources
            .into_iter()
            .map(|source| {
                load_primary_crossover_curve(source, primary_seat, frequency_samples, &output.id)
            })
            .collect::<Result<_>>()?;
        if raw.is_empty() || !raw.iter().all(curve_has_usable_phase) {
            continue;
        }
        let refs = raw.iter().collect::<Vec<_>>();
        let Some(grid) = roomeq_engine::topology::shared_measurement_grid(&refs) else {
            continue;
        };
        let aligned: Vec<_> = raw
            .iter()
            .map(|c| autoeq_core::curve_transforms::interpolate_log_space(&grid, c))
            .collect();
        let sum = complex_sum_mains(&aligned.iter().collect::<Vec<_>>());
        outputs.insert(output.id.clone(), sum);
    }
    if outputs.len() != subs.outputs.len() {
        return Ok(context);
    }
    // Explicit sub trim and optional physical LFE gain belong to the baseline;
    // automatic level alignment does not.
    let effective = engine_home_cinema::effective_bass_management(config);
    let physical_lfe_gain = effective
        .as_ref()
        .filter(|bm| bm.config.apply_lfe_gain_to_chain)
        .map_or(0.0, |bm| bm.config.lfe_playback_gain_db);
    let (configured_gain, _) =
        engine_home_cinema::limited_sub_gain(physical_lfe_gain, effective.as_ref());
    for output in outputs.values_mut() {
        output.spl.mapv_inplace(|db| db + configured_gain);
    }
    for route in &mut graph.routes {
        route.crossover_type = roomeq_engine::topology::bass_management_crossover_type_candidates(
            &route.crossover_type,
        )[0]
        .clone();
        if route.route_kind == "redirected_bass_lowpass_to_sub"
            && let Some(roomeq_model::SubwooferCrossoverRef::PerSub(keys)) = &subs.crossover
            && keys.len() >= 2
            && let Some(index) = subs.outputs.iter().position(|o| o.id == route.destination)
            && let Some(crossover) = keys
                .get(index)
                .and_then(|key| baseline_config.crossovers.as_ref()?.get(key))
        {
            route.low_pass_hz = crossover.frequency;
            route.crossover_type = crossover.crossover_type.clone();
        }
    }
    let Some(fallback) = outputs
        .get(&graph.physical_sub_output)
        .or_else(|| outputs.values().next())
    else {
        return Ok(context);
    };
    for role in mains.keys() {
        let source = resolve_single_source(role, config, system)?;
        let raw = load_primary_crossover_curve(source, primary_seat, frequency_samples, role)?;
        let Some(route) = graph
            .routes
            .iter()
            .find(|r| r.source_channel == *role && r.route_kind == "main_highpass_to_self")
        else {
            continue;
        };
        let Some(xo) = route.high_pass_hz else {
            continue;
        };
        let mut support = vec![&raw];
        support.extend(outputs.values());
        let Some(grid) = roomeq_engine::topology::shared_measurement_grid(&support) else {
            continue;
        };
        let raw = autoeq_core::curve_transforms::interpolate_log_space(&grid, &raw);
        let mut main =
            apply_crossover_response_to_curve(&raw, &route.crossover_type, xo, fs, false);
        main = apply_delay_and_polarity_to_curve(&main, route.delay_ms, route.polarity_inverted);
        main.spl.mapv_inplace(|v| v + route.gain_db);
        let Some(bass) = engine_bass_management::predict_bass_source_curve_from_output_routes(
            &main, &outputs, fallback, None, &graph, role, fs,
        ) else {
            continue;
        };
        if let Some(baseline) = roomeq_engine::topology::cancellation_baseline(&main, &bass, xo) {
            context.sources.insert(role.clone(), baseline);
        }
    }
    Ok(context)
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
        let physical_sub_config = physical_sub_speaker_config(config, sys)?;
        let has_sub = physical_sub_config.is_some();

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

        let layout = if matches!(sys.model, SystemModel::Stereo) {
            "Stereo"
        } else {
            "Home cinema"
        };
        info!(target: BASS_MANAGEMENT_LOG_TARGET,
            "{layout} bass management ({} single mains, {} supporting sources, {} physical sub outputs)",
            single_roles.len(),
            supporting_roles.len(),
            sys.subwoofers.as_ref().map_or(0, |subs| subs.outputs.len()),
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

        // Check raw evidence before any phase-sensitive sub-array optimization
        // or baseline prediction, not only before the final crossover search.
        let phase_quality_advisories = if has_sub {
            configured_crossover_phase_advisories(config, sys)?
        } else {
            Vec::new()
        };

        // Freeze raw configured routing before sub-array optimization or EQ.
        let mut routed_config = config.clone();
        routed_config.optimizer.crossover_cancellation_baseline =
            Some(capture_crossover_cancellation_baseline(
                config,
                &curves,
                sample_rate,
                assembly.frequency_samples,
            )?);

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
            let mut sp = preprocess_sub_with_frequency_samples(
                physical_sub_config
                    .as_ref()
                    .expect("has_sub proves physical sub configuration exists"),
                &sub_sys.config,
                &config.optimizer,
                sample_rate,
                assembly.frequency_samples,
            )?;
            if let Some(drivers) = sp.drivers.as_mut() {
                for (driver, output) in drivers.iter_mut().zip(&sub_sys.outputs) {
                    driver.name = output.id.clone();
                }
            }
            curves.insert(sub_role.clone(), sp.combined_curve.clone());
            Some(sp)
        } else {
            None
        };

        let mut result = if has_sub {
            let total_channels = single_roles.len() + 1;
            optimize_home_cinema_with_sub(
                &routed_config,
                sys,
                &single_roles,
                &curves,
                sub_preprocess.unwrap(),
                phase_quality_advisories,
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
            info!(target: BASS_MANAGEMENT_LOG_TARGET,
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
        finalized_decisions: None,
        channels: HashMap::new(),
        channel_results: HashMap::new(),
        deployed_source_curves: HashMap::new(),
        combined_pre_score: 0.0,
        combined_post_score: 0.0,
        metadata: OptimizationMetadata {
            final_convolution_sha256: None,
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
            audibility_veto: None,
            veto_adjudication: None,
            optimizer_evidence: None,
            stage_outcomes: Vec::new(),
            qa_seed_distribution: None,
            effective_config: None,
            t60_flatness_tolerance_s: config.report_t60_tolerance_s(),
            operation_gates: None,
            provisional_decisions: Vec::new(),
            epa_provenance: None,
            playback_summary: None,
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

            info!(target: BASS_MANAGEMENT_LOG_TARGET, "  Optimizing '{}' with alignment gain {:.2} dB", role, gain);

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

            info!(target: BASS_MANAGEMENT_LOG_TARGET,
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

    info!(target: BASS_MANAGEMENT_LOG_TARGET,
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
        finalized_decisions: None,
        channels: channel_chains,
        channel_results,
        deployed_source_curves: HashMap::new(),
        combined_pre_score: avg_pre,
        combined_post_score: avg_post,
        metadata: OptimizationMetadata {
            final_convolution_sha256: None,
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
            audibility_veto: None,
            veto_adjudication: None,
            optimizer_evidence: None,
            stage_outcomes: Vec::new(),
            qa_seed_distribution: None,
            effective_config: None,
            t60_flatness_tolerance_s: config.report_t60_tolerance_s(),
            operation_gates: None,
            provisional_decisions: Vec::new(),
            epa_provenance: None,
            playback_summary: None,
        },
    })
}

/// Read-only snapshot using the same baseline policy as final replay.
/// Publish residual advisories only after every source has passed.
pub(crate) fn reconstruct_deployed_snapshot_best_effort(
    channels: &HashMap<String, ChannelDspChain>,
    fir_coeffs_by_channel: &HashMap<String, Vec<f64>>,
    graph: &BassManagementRoutingGraph,
    optimization: Option<&engine_home_cinema::BassManagementOptimizationReport>,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
    degraded: &mut Vec<String>,
) -> Result<HashMap<String, Curve>> {
    let (curves, evidence) = reconstruct_deployed_source_curves_with_evidence(
        channels,
        fir_coeffs_by_channel,
        graph,
        optimization,
        sample_rate,
        sidecar_dir,
    )?;
    degraded.extend(
        evidence
            .iter()
            .filter(|e| e.reason == "improved_residual_cancellation")
            .map(|e| format!("{}:improved_residual_cancellation", e.source_channel)),
    );
    Ok(curves)
}

/// Replay every mains splice, reverting offending post-route FIR/EQ stages.
///
/// Improved residuals require numeric baseline evidence at every stage.
/// Failure without removable correction is an error, regardless of old advisories.
#[allow(clippy::too_many_arguments)]
pub(crate) fn replay_until_splice_safe(
    channel_chains: &mut HashMap<String, ChannelDspChain>,
    channel_results: &mut HashMap<String, ChannelOptimizationResult>,
    splice_reverted: &mut Vec<String>,
    fir_coeffs_by_channel: &HashMap<String, Vec<f64>>,
    graph: &BassManagementRoutingGraph,
    optimization: Option<&engine_home_cinema::BassManagementOptimizationReport>,
    sample_rate: f64,
    sidecar_dir: &std::path::Path,
    role_crossover_hz: &dyn Fn(&str) -> f64,
    sub_role: &str,
    sub_min_score: f64,
    bass_route_upper_hz: f64,
    max_freq: f64,
) -> Result<HashMap<String, Curve>> {
    if crossover_timing_refused(optimization) {
        // Uncalibrated main/sub phase makes every coherent splice verdict
        // arbitrary. Never strip correction stages on such verdicts; the
        // final selection records the skipped assessment explicitly.
        return reconstruct_deployed_source_curves_unenforced(
            channel_chains,
            fir_coeffs_by_channel,
            graph,
            optimization,
            sample_rate,
            sidecar_dir,
        );
    }
    loop {
        let mut cancellation_evidence = Vec::new();
        let replay = reconstruct_deployed_source_curves_impl(
            channel_chains,
            fir_coeffs_by_channel,
            graph,
            optimization,
            sample_rate,
            sidecar_dir,
            SpliceSafety::Record(&mut cancellation_evidence),
        );
        let error = match replay {
            Ok(curves) => {
                splice_reverted.extend(
                    cancellation_evidence
                        .iter()
                        .filter(|e| e.reason == "improved_residual_cancellation")
                        .map(|e| format!("{}:improved_residual_cancellation", e.source_channel)),
                );

                return Ok(curves);
            }
            Err(error) => error,
        };
        let Some(role) = underfill_error_role(&error.to_string()) else {
            return Err(error);
        };
        let Some(chain) = channel_chains.get_mut(&role) else {
            return Err(error);
        };
        let Some(stage) = strip_next_splice_breaking_stage(chain) else {
            return Err(error);
        };
        log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
            "  Final routed replay for '{role}' cancels at the crossover; reverting {stage} and replaying: {error}"
        );
        let initial: Curve = chain
            .initial_curve
            .clone()
            .ok_or_else(|| AutoeqError::InvalidMeasurement {
                message: format!("channel '{role}' has no initial curve for splice revert"),
            })?
            .into();
        let embedded_irs = embedded_convolution_irs(
            &chain.plugins,
            fir_coeffs_by_channel.get(&role).map(Vec::as_slice),
        )?;
        let realized = crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
            chain,
            &initial,
            sample_rate,
            sidecar_dir,
            &embedded_irs,
        )?;
        let realized_data: CurveData = (&realized).into();
        chain.eq_response = chain
            .initial_curve
            .as_ref()
            .map(|initial| output::compute_eq_response(initial, &realized_data));
        chain.final_curve = Some(realized_data);
        if let Some(channel_result) = channel_results.get_mut(&role) {
            channel_result.final_curve = realized.clone();
            if stage == "fir" {
                channel_result.fir_coeffs = None;
            } else {
                channel_result.biquads.clear();
            }
            channel_result.post_score = if role == sub_role {
                compute_flat_loss(&realized, sub_min_score, bass_route_upper_hz)
            } else {
                compute_flat_loss(&realized, role_crossover_hz(&role), max_freq)
            };
        }
        splice_reverted.push(format!("{role}:{stage}"));
    }
}

/// Resolve the deployable per-driver low-pass for one sub driver.
///
/// The frequency is the optimizer-selected per-sub splice value (`LP_i`)
/// recorded on the matching sub-output report; `None` (skipped pair or legacy
/// single-crossover run) deploys nothing. The crossover family resolves
/// positionally from the per-sub crossover list (entry `i` for sub `i`),
/// falling back to the shared bus low-pass family. Returns `None` unless the
/// entry is deployable, so legacy behavior stays bit-identical.
fn per_driver_low_pass_plan(
    config: &RoomConfig,
    driver_index: usize,
    driver_name: &str,
    sub_outputs: &HashMap<String, engine_home_cinema::BassManagementSubOutputReport>,
    fallback_crossover_type: &str,
) -> Option<output::PerDriverLowPass> {
    let frequency_hz = sub_outputs
        .get(driver_name)
        .and_then(|output| output.selected_low_pass_hz)?;
    // Legacy single-crossover configs (shared key or one-element list) deploy
    // nothing by construction: the per-sub splice stage never records values
    // for them, and this guard keeps them bit-identical even if a value were
    // ever present.
    let keys = config
        .system
        .as_ref()
        .and_then(|system| system.subwoofers.as_ref())
        .and_then(|subwoofers| subwoofers.crossover.as_ref())
        .map(|crossover| crossover.as_list())
        .filter(|keys| keys.len() >= 2)?;
    let crossover_type = keys
        .get(driver_index)
        .copied()
        .and_then(|key| {
            config
                .crossovers
                .as_ref()
                .and_then(|crossovers| crossovers.get(key))
                .map(|crossover| crossover.crossover_type.clone())
        })
        .unwrap_or_else(|| fallback_crossover_type.to_string());
    let plan = output::PerDriverLowPass {
        frequency_hz,
        crossover_type,
    };
    plan.is_deployable().then_some(plan)
}

fn configured_crossover_phase_advisories(
    config: &RoomConfig,
    sys: &SystemConfig,
) -> Result<Vec<String>> {
    let mut keys = match &sys.subwoofers {
        Some(sub) => sub
            .crossover
            .as_ref()
            .ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: "Subwoofer config requires 'crossover' reference".to_string(),
            })?
            .as_list(),
        None => Vec::new(),
    };
    if let Some(bass) = &sys.bass_management {
        keys.extend(bass.group_crossovers.values().map(String::as_str));
    }
    let mut low = f64::INFINITY;
    let mut high = 0.0_f64;
    for key in keys {
        let crossover = config
            .crossovers
            .as_ref()
            .and_then(|values| values.get(key))
            .ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: format!("missing crossover '{key}' for phase assessment"),
            })?;
        let (lo, hi) = crossover
            .frequency
            .map(|hz| (hz, hz))
            .or(crossover.frequency_range)
            .ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: format!("crossover '{key}' needs 'frequency' or 'frequency_range'"),
            })?;
        low = low.min(lo);
        high = high.max(hi);
    }
    // One octave either side covers the configured candidate overlap,
    // including per-group/per-sub overrides, without requiring treble data.
    crate::room_optimization::seat_replay::crossover_phase_advisories(
        config,
        (low / 2.0, high * 2.0),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "workflow stage carries the prepared routing context"
)]
fn optimize_home_cinema_with_sub(
    config: &RoomConfig,
    sys: &SystemConfig,
    main_roles: &[String],
    curves: &HashMap<String, Curve>,
    sub_preprocess: SubPreprocessResult,
    phase_quality_advisories: Vec<String>,
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
    let main_alignment_band = main_level_alignment_band(curves, main_roles, max_xo)?;
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
                        log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
                            "  Main Pre-EQ crossover lower bound {:.1} Hz does not overlap configured optimization band [{:.1}, {:.1}] Hz; retaining the configured band",
                            min_xo,
                            per_config.optimizer.min_freq,
                            per_config.optimizer.max_freq
                        );
                    }
                    info!(target: BASS_MANAGEMENT_LOG_TARGET,
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
            let sub_source = match &sub_preprocess.shared_eq_seats {
                Some(seats) => MeasurementSource::InMemoryMultiple(seats.clone()),
                None => MeasurementSource::InMemory(sub_preprocess.combined_curve.clone()),
            };
            let mut sub_config = config.clone();
            if sub_preprocess.common_eq_complete {
                // The dedicated engine already applied the configured spatial
                // PEQ/global-EQ policy. Do not replace it with a primary-only fit.
                sub_config.optimizer.num_filters = 0;
                sub_config.optimizer.processing_mode = roomeq_model::ProcessingMode::LowLatency;
                sub_config.optimizer.phase_correction = None;
            }
            if max_xo > sub_config.optimizer.min_freq {
                sub_config.optimizer.max_freq = sub_config.optimizer.max_freq.min(max_xo);
            } else {
                log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
                    "  Sub Pre-EQ crossover upper bound {:.1} Hz does not overlap configured optimization band [{:.1}, {:.1}] Hz; retaining the configured band",
                    max_xo,
                    sub_config.optimizer.min_freq,
                    sub_config.optimizer.max_freq
                );
            }
            info!(target: BASS_MANAGEMENT_LOG_TARGET,
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
        if let Some(mut target) = chain.target_curve.clone() {
            // The generic solve uses raw measurements. Its target must follow
            // the same physical calibration gain applied when routing the main.
            let align_gain = gains.get(&role).copied().unwrap_or(0.0);
            for level in &mut target.spl {
                *level += align_gain;
            }
            pre_eq_target_curves.insert(role.clone(), target);
        }
        if let Some(fir_coeffs) = ch_result.fir_coeffs.clone() {
            pre_eq_fir_coeffs.insert(role.clone(), fir_coeffs);
        }
        pre_eq_plugins.insert(role.clone(), stage_main_correction_plugins(chain.plugins));
        // The EQ solve uses every configured seat, but physical route timing
        // must use one synchronous complex capture. Its magnitude-only
        // multi-seat summary is not a measured transfer function.
        let source = resolve_single_source(&role, config, sys)?;
        let primary_seat = config
            .optimizer
            .multi_seat
            .as_ref()
            .map(|seat| seat.primary_seat)
            .unwrap_or(0);
        let mut primary = crate::multisub::load_primary_measurements_with_frequency_samples(
            std::slice::from_ref(source),
            primary_seat,
            assembly.frequency_samples,
        )
        .map_err(|error| AutoeqError::InvalidMeasurement {
            message: error.to_string(),
        })?;
        pre_eq_initial_curves.insert(role.clone(), primary.remove(0));
        optimizer_evidence_by_channel.insert(role.clone(), ch_result.optimizer_evidence);
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
        // Spatial EQ inputs must not replace routing's complex representative.
        pre_eq_initial_curves.insert(
            sub_role.clone(),
            if sub_preprocess.shared_eq_seats.is_some() {
                sub_preprocess.combined_curve.clone()
            } else {
                ch_result.initial_curve
            },
        );
        let mut sub_evidence = sub_preprocess.optimizer_evidence.clone();
        sub_evidence.extend(ch_result.optimizer_evidence);
        optimizer_evidence_by_channel.insert(sub_role.clone(), sub_evidence);
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

    // Include every native sub frequency before realizing main filters. The
    // receiving main retains its full measured span, even for a short sub file.
    let bass_samples = sub_preprocess
        .drivers
        .as_ref()
        .map(|drivers| {
            drivers
                .iter()
                .filter_map(|driver| driver.initial_curve.as_ref())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![&sub_preprocess.combined_curve]);
    for role in main_roles {
        let initial = &pre_eq_initial_curves[role];
        let grid =
            roomeq_engine::topology::bass_management_measurement_grid(initial, &bass_samples);
        let expanded = autoeq_core::interpolate_log_space(&grid, initial);
        pre_eq_initial_curves.insert(role.clone(), expanded);
        if let Some(aligned) = aligned_curves.get_mut(role) {
            *aligned = autoeq_core::interpolate_log_space(&grid, aligned);
        }
    }

    let mut aligned_pre_eq_curves: HashMap<String, Curve> = HashMap::new();
    for role in main_roles {
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
        let realized = realize_plugins_on_curve(
            role,
            plugins,
            &pre_eq_initial_curves[role],
            sample_rate,
            output_dir,
            &embedded_irs,
        )?;
        aligned_pre_eq_curves.insert(role.clone(), realized);
    }
    // Bass-route optimization models the physical sub output shared by every
    // redirected main. The LFE logical-input alignment gain is applied only on
    // the LFE pre-route path and must not be folded into this common transfer.
    let sub_plugins = pre_eq_plugins.get(&sub_role).cloned().unwrap_or_default();
    let sub_embedded_irs = embedded_convolution_irs(
        &sub_plugins,
        pre_eq_fir_coeffs.get(&sub_role).map(Vec::as_slice),
    )?;
    let realized_sub = realize_plugins_on_curve(
        &sub_role,
        sub_plugins,
        &pre_eq_initial_curves[&sub_role],
        sample_rate,
        output_dir,
        &sub_embedded_irs,
    )?;
    aligned_pre_eq_curves.insert(sub_role.clone(), realized_sub);

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
    let crossover_grid = roomeq_engine::topology::bass_management_measurement_grid(
        &aligned_pre_eq_curves[&main_roles[0]],
        &aligned_pre_eq_curves.values().collect::<Vec<_>>(),
    );
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
            let mut curve = load_primary_crossover_curve(
                source,
                primary_seat,
                assembly.frequency_samples,
                role,
            )?;
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
    let overlap_factor = 2.0_f64.powf(crate::crossover_summation::OVERLAP_BAND_HALF_OCTAVES);
    let timing_reference = crate::evidence_intake::crossover_timing_reference(
        config,
        main_roles,
        [
            (min_xo / overlap_factor).max(crossover_grid.first().copied().unwrap_or(f64::NAN)),
            (max_xo * overlap_factor).min(crossover_grid.last().copied().unwrap_or(f64::NAN)),
        ],
    );
    let phase_available = measured_phase_available
        && processed_phase_available
        && shared_grid_available
        && timing_reference.is_ok();
    let mut optimization_advisories = sub_preprocess.advisories.clone();
    optimization_advisories.extend(phase_quality_advisories);
    if let Err(reason) = &timing_reference {
        optimization_advisories.push(format!(
            "{CROSSOVER_TIMING_REFUSED_ADVISORY_PREFIX}{reason}"
        ));
    } else if let Ok(reference) = &timing_reference {
        optimization_advisories.push(format!(
            "crossover_declared_common_timing_reference:{reference}"
        ));
    }
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
    // C06: the overlap-band summation search selects polarity/delay/gain
    // against measured phase and reconciles with the optimizer candidate.
    // The search wins only on strict full-band improvement; refusals and
    // ties keep the optimizer values with explicit reason codes. The
    // reconciled gain feeds the single downstream application (with the
    // separate LFE gain and the headroom limit), so no gain applies twice.
    // Adopted selections also emit provisional ledger rows (C08); retained
    // or refused searches leave only advisory reason codes, never Applied
    // rows for numbers that were not emitted.
    let mut crossover_provisional = Vec::new();
    let (main_delay_raw, sub_delay_raw, sub_gain_raw, sub_inverted) = if phase_available {
        let reconciled = crate::crossover_summation::reconcile_main_sub_summation(
            &main_refs,
            sub_curve,
            final_xo_freq,
            sample_rate,
            &crate::crossover_summation::MainSubOptimizerValues {
                main_delay_ms: main_delay_raw,
                sub_delay_ms: sub_delay_raw,
                sub_gain_db: sub_gain_raw,
                sub_inverted,
            },
            timing_reference.as_deref().ok(),
        );
        optimization_advisories.extend(reconciled.advisories.iter().cloned());
        if let Some(report) = &reconciled.report
            && reconciled
                .advisories
                .iter()
                .any(|advisory| advisory.contains("search_selected"))
        {
            let mut measurement_refs = main_roles.to_vec();
            measurement_refs.push(sub_role.clone());
            crossover_provisional.extend(
                crate::crossover_summation::reconcile_summation_decisions(
                    report,
                    &main_roles.join("+"),
                    sub_role.as_str(),
                    measurement_refs,
                    Vec::new(),
                    None,
                ),
            );
        }
        (
            reconciled.main_delay_ms,
            reconciled.sub_delay_ms,
            reconciled.sub_gain_db,
            reconciled.sub_inverted,
        )
    } else {
        (main_delay_raw, sub_delay_raw, sub_gain_raw, sub_inverted)
    };
    let (main_delay_post, sub_delay_post) =
        normalize_crossover_delays(main_delay_raw, sub_delay_raw);
    let sub_gain_post = sub_gain_raw;

    info!(target: BASS_MANAGEMENT_LOG_TARGET,
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
    info!(target: BASS_MANAGEMENT_LOG_TARGET,
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
        log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
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
    let baseline_reason = if timing_reference.is_err() {
        "source_route_optimizer_skipped_unverified_timing"
    } else if !phase_available {
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
    let mut stereo_routing = None;
    let is_stereo_pair = matches!(sys.model, SystemModel::Stereo)
        && main_roles == ["L", "R"]
        && sub_output_results.len() == 2
        && sub_preprocess
            .drivers
            .as_ref()
            .is_some_and(|drivers| drivers.len() == 2);
    let sub_output_advisories = if optimize_source_routes && is_stereo_pair {
        let candidate_specs = [
            (
                StereoBassTopology::DirectPair,
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            ),
            (
                StereoBassTopology::CrossedPair,
                vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            ),
            (
                StereoBassTopology::DualMono,
                vec![vec![0.5, 0.5], vec![0.5, 0.5]],
            ),
        ];
        let mut candidate_runs = Vec::with_capacity(candidate_specs.len());
        for (topology, matrix) in candidate_specs {
            let mut candidate_groups = group_results_by_id.clone();
            let mut candidate_sources = source_results.clone();
            let mut candidate_outputs = sub_output_results.clone();
            let advisories = optimize_bass_management_joint_solution_with_matrix(
                config,
                main_roles,
                &aligned_curves,
                &aligned_pre_eq_curves,
                Some(&optimizer_source_pre_route_transfers),
                bass_management_target_curves.as_ref(),
                &mut candidate_groups,
                &mut candidate_sources,
                &mut candidate_outputs,
                sub_preprocess.drivers.as_deref(),
                &sub_role,
                sample_rate,
                Some(&matrix),
            );
            let objective = stereo_route_objective(&candidate_sources);
            let headroom_margin_db =
                stereo_route_headroom_margin_db(config, &matrix, &candidate_outputs);
            let replay_rejection = sub_preprocess.drivers.as_deref().map_or_else(
                || Some("serialized_replay_missing_physical_outputs".to_string()),
                |drivers| {
                    stereo_candidate_serialized_replay_rejection(
                        config,
                        main_roles,
                        &aligned_pre_eq_curves,
                        &optimizer_source_pre_route_transfers,
                        &candidate_groups,
                        &candidate_sources,
                        &candidate_outputs,
                        drivers,
                        aligned_curves
                            .get(&sub_role)
                            .zip(aligned_pre_eq_curves.get(&sub_role)),
                        &matrix,
                        sample_rate,
                    )
                    .err()
                },
            );
            let rejection_reason = if objective.is_none_or(|value| !value.is_finite()) {
                Some("candidate_objective_unavailable_after_serialized_route_replay".to_string())
            } else if headroom_margin_db < 0.0 {
                Some(format!(
                    "electrical_headroom_margin_exceeded:{headroom_margin_db:.3}db"
                ))
            } else if replay_rejection.is_some() {
                replay_rejection
            } else {
                None
            };
            let report = StereoBassRoutingCandidateReport {
                topology,
                objective,
                rejection_reason,
                headroom_margin_db,
                matrix,
            };
            candidate_runs.push((
                report,
                candidate_groups,
                candidate_sources,
                candidate_outputs,
                advisories,
            ));
        }

        let reports = candidate_runs
            .iter()
            .map(|(report, ..)| report.clone())
            .collect::<Vec<_>>();
        let selected = engine_home_cinema::select_stereo_bass_candidate(&reports)
            .cloned()
            .ok_or_else(|| AutoeqError::OptimizationFailed {
                message: format!(
                    "all stereo bass routing candidates failed acoustic or electrical safety: {}",
                    reports
                        .iter()
                        .map(|report| format!(
                            "{:?}: {}",
                            report.topology,
                            report
                                .rejection_reason
                                .as_deref()
                                .unwrap_or("objective unavailable")
                        ))
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
            })?;
        let selected_index = candidate_runs
            .iter()
            .position(|(report, ..)| report.topology == selected.topology)
            .expect("selected stereo routing candidate belongs to evaluated set");
        let (_, selected_groups, selected_sources, selected_outputs, advisories) =
            candidate_runs.swap_remove(selected_index);
        group_results_by_id = selected_groups;
        source_results = selected_sources;
        sub_output_results = selected_outputs;
        stereo_routing = Some(StereoBassRoutingReport {
            selected_topology: selected.topology,
            matrix: selected.matrix.clone(),
            candidates: reports,
            selection_basis:
                "serialized_graph_splice_and_headroom_pass_then_lowest_robust_objective_then_headroom_margin_then_direct_crossed_dual_mono"
                    .to_string(),
        });
        advisories
    } else if optimize_source_routes {
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
    log::debug!(target: BASS_MANAGEMENT_LOG_TARGET,
        "  Bass-management source-route optimizer advisories: {:?}",
        sub_output_advisories
    );
    for advisory in sub_output_advisories {
        optimization_advisories.retain(|existing| existing != "ok");
        if !optimization_advisories.contains(&advisory) {
            optimization_advisories.push(advisory);
        }
    }
    for source in &source_results {
        info!(target: BASS_MANAGEMENT_LOG_TARGET,
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
    // `sub_post` includes the physical-output gain so the serialized sub chain
    // and its standalone response remain complete. Routed prediction applies
    // that same gain from `BassManagementRoute`, therefore its common physical
    // sub input must exclude the route-owned gain or post-EQ sees it twice.
    let mut routed_common_sub_post = sub_post.clone();
    routed_common_sub_post
        .spl
        .mapv_inplace(|level| level - route_applied_sub_gain_db);
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
        crossover_cancellation: config.optimizer.crossover_cancellation_baseline.clone(),
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
        stereo_routing,
        advisories: optimization_advisories,
    };
    let mut bass_routing_graph = engine_home_cinema::bass_management_routing_graph(
        config,
        Some(&bass_management_optimization),
    );
    log::debug!(target: BASS_MANAGEMENT_LOG_TARGET, "  Bass-management routing graph: {bass_routing_graph:?}");
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
    let mut post_eq_filters: HashMap<String, Vec<Biquad>> = HashMap::new();
    let mut post_eq_output_rejections: Vec<(String, f64)> = Vec::new();
    let mut post_eq_main_preservation = Vec::new();
    let mut routed_target_curves: HashMap<String, CurveData> = HashMap::new();
    let main_post_max_freq = config.optimizer.max_freq;
    let main_post_eq_offset = usize::from(!sub_preprocess.common_eq_complete);
    let total_post_eq_passes = main_roles.len() + main_post_eq_offset;

    // Dedicated multi-seat/all-pass sub EQ is already complete.
    if !sub_preprocess.common_eq_complete {
        let sub_progress_base = 0.91;
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
            log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
                "  Sub Post-EQ skipped: bass-route upper bound {:.1} Hz leaves no optimization band above min_freq {:.1} Hz after the 20 Hz guard band",
                bass_route_upper_hz,
                opt_config.min_freq,
            );
        }
        let sub_min_score = config.optimizer.min_freq.max(20.0);
        let sub_callback = workflow_progress_callback(
            &assembly.progress_factory,
            &format!("Post-EQ {sub_role}"),
            0,
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
        let routed_common_sub_after_eq =
            response::apply_complex_response(&routed_common_sub_post, &eq_resp);
        let routed_underfill = bass_routing_graph.as_ref().and_then(|graph| {
            main_roles
                .iter()
                .map(|role| {
                    let group_id = engine_home_cinema::group_id_for_role(
                        engine_home_cinema::role_for_channel(role),
                    );
                    let role_xover_freq = group_results_by_id
                        .get(group_id)
                        .and_then(|group| group.selected_crossover_hz)
                        .unwrap_or(final_xo_freq);
                    let mut main = main_post_curves[role].clone();
                    let mut bass = engine_bass_management::predict_bass_source_curve_from_routes(
                        &routed_common_sub_after_eq,
                        optimizer_source_pre_route_transfers.get(role),
                        graph,
                        role,
                        sample_rate,
                    )?;
                    if let Some(filters) = post_eq_filters
                        .get(role)
                        .filter(|filters| !filters.is_empty())
                    {
                        let main_response = response::compute_peq_complex_response(
                            filters,
                            &main.freq,
                            sample_rate,
                        );
                        main = response::apply_complex_response(&main, &main_response);
                        let bass_response = response::compute_peq_complex_response(
                            filters,
                            &bass.freq,
                            sample_rate,
                        );
                        bass = response::apply_complex_response(&bass, &bass_response);
                    }
                    post_eq_crossover_cancellation(config, role, &main, &bass, role_xover_freq)
                        .map(|evidence| (role.as_str(), evidence))
                })
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .max_by(|left, right| {
                    (!left.1.accepted)
                        .cmp(&!right.1.accepted)
                        .then_with(|| left.1.final_db.total_cmp(&right.1.final_db))
                })
        });
        // Without calibrated main/sub timing the coherent cancellation
        // verdict is arbitrary; keep Sub Post-EQ on its score instead of
        // rejecting corrections on luck.
        let routed_underfill_accepted =
            crossover_timing_refused(Some(&bass_management_optimization))
                || routed_underfill
                    .as_ref()
                    .is_some_and(|(_, evidence)| evidence.accepted);
        // The SPL-loss allowance belongs to mains/surrounds/heights, not
        // subwoofer peak reduction. Sub EQ must still improve its response
        // and preserve every receiving main's crossover integration.
        if sub_post_eq_band_empty {
            post_eq_filters.insert(sub_role.clone(), Vec::new());
        } else if post < pre && routed_underfill_accepted {
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
            if let Some((role, underfill_db)) = routed_underfill
                .filter(|_| !routed_underfill_accepted)
                .map(|(role, evidence)| (role, evidence.final_db))
            {
                log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
                    "  Sub Post-EQ discarded: routed crossover underfill for '{}' is {:.3} dB",
                    role,
                    underfill_db,
                );
            } else {
                log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
                    "  Sub Post-EQ discarded: score {:.4} -> {:.4} or full-band useful output loss exceeded its budget",
                    pre,
                    post
                );
            }
            post_eq_filters.insert(sub_role.clone(), Vec::new());
        }
    }

    // The physical-sub Post-EQ is output-owned and is shared by every
    // redirected source. Select it before source-specific Post-EQ so each
    // routed training curve contains the same common output transfer as the
    // emitted graph. The existing cancellation screen is invariant to a
    // common linear PEQ applied to both splice branches.
    if let Some(filters) = post_eq_filters
        .get(&sub_role)
        .filter(|filters| !filters.is_empty())
    {
        let response = response::compute_peq_complex_response(
            filters,
            &routed_common_sub_post.freq,
            sample_rate,
        );
        routed_common_sub_post =
            response::apply_complex_response(&routed_common_sub_post, &response);
    }

    for (role_index, role) in main_roles.iter().enumerate() {
        let post_eq_ordinal = role_index + main_post_eq_offset;
        let role_progress_base =
            0.91 + (post_eq_ordinal as f64 / total_post_eq_passes as f64) * 0.03;
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
        let role_group = group_results_by_id.get(group_id);
        let role_xover_type = role_group
            .map(|group| group.crossover_type.as_str())
            .unwrap_or(xover_type_str);
        let role_xover_freq = group_results_by_id
            .get(group_id)
            .and_then(|g| g.selected_crossover_hz)
            .unwrap_or(final_xo_freq);
        let role_main_delay = engine_home_cinema::resolved_source_route_settings(
            role,
            group_id,
            Some(&bass_management_optimization),
        )
        .main_delay_ms;
        // The broadband main correction has already run. Reserve this pass
        // for the routed crossover residual so its filters are not spent on
        // unrelated high-frequency details.
        opt_config.min_freq = opt_config.min_freq.max(role_xover_freq * 0.5);
        opt_config.max_freq = opt_config.max_freq.min(role_xover_freq * 2.0);
        if let Err(reason) = &timing_reference {
            post_eq_main_preservation.push(StageOutcome {
                stage: format!("routed_common_eq_timing_{role}"),
                status: StageStatus::Degraded,
                checks: vec![roomeq_model::StageCheck {
                    id: "coherent_common_correction_timing_reference".into(),
                    kind: roomeq_model::StageCheckKind::Structural,
                    passed: false,
                    observed: None,
                    limit: None,
                    diagnostic: Some(serde_json::json!({
                        "verdict": "InsufficientEvidence",
                        "refusal_reason": format!("coherent timing reference refused: {reason}"),
                        "common_coherent_correction_applied": false,
                        "individual_correction_scope": "retained; requires final native graph validation",
                    }).to_string()),
                }],
                advisories: vec!["common_coherent_eq_refused_before_candidate_evaluation".into()],
            });
            post_eq_filters.insert(role.clone(), Vec::new());
            continue;
        }
        let post_curve = bass_routing_graph
            .as_ref()
            .and_then(|graph| {
                engine_bass_management::predict_deployed_source_curve_from_routes(
                    Some(&main_post_curves[role]),
                    &routed_common_sub_post,
                    optimizer_source_pre_route_transfers.get(role),
                    graph,
                    role,
                    sample_rate,
                )
            })
            .unwrap_or_else(|| main_post_curves[role].clone());
        let routed_training_curves = if let Some(graph) = bass_routing_graph.as_ref() {
            let source = resolve_single_source(role, config, sys)?;
            let main_seats =
                load_source_individual_with_frequency_samples(source, assembly.frequency_samples)
                    .map_err(|error| AutoeqError::InvalidMeasurement {
                    message: format!("could not load routed training seats for '{role}': {error}"),
                })?;
            let sub_seats = sub_preprocess
                .shared_eq_seats
                .as_ref()
                .cloned()
                .unwrap_or_else(|| vec![sub_preprocess.combined_curve.clone()]);
            let sub_speaker = physical_sub_speaker_config(config, sys)?;
            let identity_order = sub_speaker
                .as_ref()
                .ok_or_else(|| "physical_sub_source_identity_unavailable".to_string())
                .and_then(|sub| {
                    crate::group_measurements::routed_seat_identity_order(
                        config,
                        source,
                        sub,
                        main_seats.len(),
                        sub_seats.len(),
                    )
                });
            let seat_count = match identity_order {
                Ok(ids) => ids.len(),
                Err(reason) => {
                    post_eq_main_preservation.push(StageOutcome {
                        stage: format!("routed_common_eq_identity_{role}"),
                        status: StageStatus::Degraded,
                        checks: vec![roomeq_model::StageCheck {
                            id: "physical_seat_identity_and_configured_weights".into(),
                            kind: roomeq_model::StageCheckKind::Structural,
                            passed: false, observed: None, limit: None,
                            diagnostic: Some(serde_json::json!({
                                "verdict": "InsufficientEvidence",
                                "refusal_reason": reason,
                                "common_coherent_correction_applied": false,
                                "individual_correction_scope": "retained; requires final native graph validation",
                            }).to_string()),
                        }],
                        advisories: vec!["common_coherent_eq_refused_without_physical_seat_join; no_primary_fallback_or_singleton_broadcast".into()],
                    });
                    post_eq_filters.insert(role.clone(), Vec::new());
                    continue;
                }
            };
            if let Some(derived) =
                crate::home_cinema::derive_all_channel_multiseat_config_with_frequency_samples(
                    config,
                    role,
                    source,
                    assembly.frequency_samples,
                )
            {
                opt_config.multi_measurement = Some(derived);
            }
            if seat_count == 0 {
                log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
                    "  {role} routed Post-EQ refused: routed training seats are unavailable"
                );
                None
            } else {
                let common_freq = &aligned_pre_eq_curves[role].freq;
                let common_low = common_freq[0];
                let common_high = common_freq[common_freq.len() - 1];
                let plugins = pre_eq_plugins.get(role).cloned().unwrap_or_default();
                let align_gain = *gains.get(role).unwrap_or(&0.0);
                let sub_plugins = pre_eq_plugins.get(&sub_role).cloned().unwrap_or_default();
                let sub_post_eq = post_eq_filters
                    .get(&sub_role)
                    .map(Vec::as_slice)
                    .unwrap_or_default();
                let mut routed = Vec::with_capacity(seat_count);
                let mut unavailable_reason = None;
                for seat_index in 0..seat_count {
                    let main_seat = &main_seats[seat_index];
                    if main_seat.freq[0] > common_low
                        || main_seat.freq[main_seat.freq.len() - 1] < common_high
                    {
                        unavailable_reason = Some(format!(
                            "main training seat {seat_index} does not cover the common routed grid"
                        ));
                        break;
                    }
                    let main_input = autoeq_core::interpolate_log_space(common_freq, main_seat);
                    let main_post = realize_routed_main_training_branch(
                        role,
                        &plugins,
                        &main_input,
                        align_gain,
                        role_xover_type,
                        role_xover_freq,
                        main_gain_post,
                        role_main_delay,
                        sample_rate,
                        output_dir,
                        pre_eq_fir_coeffs.get(role).map(Vec::as_slice),
                    )?;
                    let sub_seat = &sub_seats[seat_index];
                    if sub_seat.freq[0] > common_low
                        || sub_seat.freq[sub_seat.freq.len() - 1] < common_high
                    {
                        unavailable_reason = Some(format!(
                            "sub training seat {seat_index} does not cover the common routed grid"
                        ));
                        break;
                    }
                    let sub_common = realize_routed_sub_training_branch(
                        &sub_role,
                        &sub_plugins,
                        sub_seat,
                        sub_gain_post - sub_gain_raw - route_applied_sub_gain_db,
                        sub_post_eq,
                        sample_rate,
                        output_dir,
                        pre_eq_fir_coeffs.get(&sub_role).map(Vec::as_slice),
                    )?;
                    let Some(routed_seat) =
                        engine_bass_management::predict_deployed_source_curve_from_routes(
                            Some(&main_post),
                            &sub_common,
                            optimizer_source_pre_route_transfers.get(role),
                            graph,
                            role,
                            sample_rate,
                        )
                    else {
                        unavailable_reason = Some(format!(
                            "routed prediction unavailable for training seat {seat_index}; phase or route transfer is missing"
                        ));
                        break;
                    };
                    if routed_seat.freq[0] > post_curve.freq[0]
                        || routed_seat.freq[routed_seat.freq.len() - 1]
                            < post_curve.freq[post_curve.freq.len() - 1]
                    {
                        unavailable_reason = Some(format!(
                            "routed training seat {seat_index} does not cover primary-seat scorecard support"
                        ));
                        break;
                    }
                    routed.push(autoeq_core::interpolate_log_space(
                        &post_curve.freq,
                        &routed_seat,
                    ));
                }
                if let Some(reason) = unavailable_reason {
                    log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
                        "  {role} routed Post-EQ refused: {reason}"
                    );
                    None
                } else {
                    Some(routed)
                }
            }
        } else {
            None
        };
        if bass_routing_graph.is_some() && routed_training_curves.is_none() {
            post_eq_main_preservation.push(StageOutcome {
                stage: format!("routed_common_eq_training_evidence_{role}"),
                status: StageStatus::Degraded,
                checks: vec![roomeq_model::StageCheck {
                    id: "complete_coherent_training_response".into(),
                    kind: roomeq_model::StageCheckKind::Structural,
                    passed: false, observed: None, limit: None,
                    diagnostic: Some("InsufficientEvidence: common correction refused; routed training transfer or grid unavailable".into()),
                }],
                advisories: vec!["no_primary_fallback_for_missing_coherent_training_evidence".into()],
            });
            post_eq_filters.insert(role.clone(), Vec::new());
            continue;
        }
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
            log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
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
            post_eq_ordinal,
            total_post_eq_passes,
            opt_config.max_iter,
        );
        let mut post_eq_result = if let Some(training_curves) = routed_training_curves {
            run_routed_training_post_eq(
                role,
                &training_curves,
                &opt_config,
                config.target_curve.as_ref(),
                sample_rate,
                post_eq_callback,
            )?
        } else {
            run_post_eq(
                &post_curve,
                &opt_config,
                config.target_curve.as_ref(),
                sample_rate,
                post_eq_callback,
            )?
        };
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
        let main_curve = &main_post_curves[role];
        let main_eq_resp =
            response::compute_peq_complex_response(&filters, &main_curve.freq, sample_rate);
        let main_curve_after = response::apply_complex_response(main_curve, &main_eq_resp);
        let bass_before_post_eq = bass_routing_graph.as_ref().and_then(|graph| {
            engine_bass_management::predict_bass_source_curve_from_routes(
                &routed_common_sub_post,
                optimizer_source_pre_route_transfers.get(role),
                graph,
                role,
                sample_rate,
            )
        });
        let cancellation_before_post_eq = bass_before_post_eq.as_ref().and_then(|bass| {
            post_eq_crossover_cancellation(config, role, main_curve, bass, role_xover_freq)
        });
        let bass_branch = bass_before_post_eq.as_ref().map(|bass| {
            let bass_eq_resp =
                response::compute_peq_complex_response(&filters, &bass.freq, sample_rate);
            response::apply_complex_response(bass, &bass_eq_resp)
        });
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
        let cancellation_evidence = bass_branch.as_ref().and_then(|bass| {
            post_eq_crossover_cancellation(config, role, &main_curve_after, bass, role_xover_freq)
        });
        let target_underfill_db =
            roomeq_engine::topology::bass_management_max_underfill_db_with_target(
                Some(&post_curve_after),
                prepared_target.as_ref(),
                role_xover_freq,
            );
        let pre_target_underfill_db =
            roomeq_engine::topology::bass_management_max_underfill_db_with_target(
                Some(&post_curve),
                prepared_target.as_ref(),
                role_xover_freq,
            );
        // Without calibrated main/sub timing the coherent cancellation
        // verdict is arbitrary; screen Post-EQ on its mains and output
        // preservation instead of rejecting corrections on luck.
        let cancellation_screened = !crossover_timing_refused(Some(&bass_management_optimization));
        // Common pre-route EQ inherits the incoming cancellation. Screen its
        // incremental effect here; final routed replay retains the original
        // configured-baseline policy for the complete correction.
        let cancellation_accepted = !cancellation_screened
            || cancellation_before_post_eq
                .as_ref()
                .zip(cancellation_evidence.as_ref())
                .is_some_and(|(before, after)| {
                    before.final_db.is_finite()
                        && after.final_db.is_finite()
                        && after.final_db
                            <= before.final_db + roomeq_model::CROSSOVER_CANCELLATION_TOLERANCE_DB
                });
        let target_accepted = match (pre_target_underfill_db, target_underfill_db) {
            (Some(before), Some(after)) => {
                roomeq_engine::topology::post_eq_underfill_is_acceptable(before, after)
            }
            (None, None) => prepared_target.is_none(),
            _ => false,
        };
        // Judge the crossover on its delivered main-plus-bass sum. The
        // isolated main must preserve its response above the splice window,
        // where the common EQ should not trade away unrelated correction.
        let main_pre_score = compute_flat_loss(main_curve, role_xover_freq, main_post_max_freq);
        let main_post_score =
            compute_flat_loss(&main_curve_after, role_xover_freq, main_post_max_freq);
        let main_protected_min_hz = role_xover_freq * 2.0;
        // The stored owning-main target includes initial level alignment.
        // Remove route-only transfer while retaining that alignment on both
        // pressure curves; a raw-plane curve must not meet an aligned target.
        let align_gain_db = gains.get(role).copied().unwrap_or(0.0);
        let mut routing_input = pre_eq_initial_curves[role].clone();
        routing_input
            .spl
            .mapv_inplace(|level| level + align_gain_db);
        let routing_baseline = apply_chain(
            &routing_input,
            role_xover_type,
            role_xover_freq,
            false,
            main_gain_post,
            role_main_delay,
            false,
        );
        let protected_target = pre_eq_target_curves.get(role).cloned().map(Curve::from);
        let protected_report = protected_target.as_ref().and_then(|target| {
            protected_main_target_preservation(
                &pre_eq_initial_curves[role],
                &routing_baseline,
                main_curve,
                &main_curve_after,
                target,
                align_gain_db,
                config.optimizer.smooth_n,
                (main_protected_min_hz, main_post_max_freq),
            )
        });
        let (main_protected_pre, main_protected_post) = protected_report
            .as_ref()
            .map(|report| {
                (
                    report.metrics.pre_target_weighted_rms_db,
                    report.metrics.post_target_weighted_rms_db,
                )
            })
            .unwrap_or((f64::NAN, f64::NAN));
        let mains_preserved = main_protected_pre.is_finite()
            && main_protected_post.is_finite()
            && main_protected_post <= main_protected_pre + 1e-6;
        post_eq_main_preservation.push(StageOutcome {
            stage: format!("post_eq_main_target_preservation_{role}"),
            status: if mains_preserved { StageStatus::Applied } else { StageStatus::Degraded },
            checks: vec![roomeq_model::StageCheck {
                id: "preserved_main_target_weighted_rms_db".to_string(),
                kind: roomeq_model::StageCheckKind::Quality,
                passed: mains_preserved,
                observed: main_protected_post.is_finite().then_some(main_protected_post),
                limit: main_protected_pre.is_finite().then_some(main_protected_pre + 1e-6),
                diagnostic: Some(serde_json::json!({
                    "before_target_weighted_rms_db": main_protected_pre.is_finite().then_some(main_protected_pre),
                    "protected_band_hz": [main_protected_min_hz, main_post_max_freq],
                    "smoothing_n": config.optimizer.smooth_n,
                    "target_source": "owning_main_pre_eq_target",
                    "reference_plane": "initially_aligned_main_output_before_structural_routing",
                    "initial_alignment_gain_db": align_gain_db,
                    "structural_main_route_gain_db": main_gain_post,
                    "evidence_available": protected_report.is_some(),
                }).to_string()),
            }],
            advisories: vec![if protected_report.is_some() {
                "historical_common_peq_candidate_screen; final_native_graph_validation_required".to_string()
            } else {
                "common_peq_refused_insufficient_owning_target_evidence".to_string()
            }],
        });
        let output_loss = post_eq_useful_output_loss(
            &post_curve,
            &post_curve_after,
            prepared_target.as_ref(),
            config.optimizer.min_freq,
            main_post_max_freq,
        )?;
        let output_preserved =
            output_loss <= config.optimizer.finalization.max_useful_output_loss_db;
        if !output_preserved {
            post_eq_output_rejections.push((role.clone(), output_loss));
        }
        // Compare this pass against its immediate input, and label the older
        // configured baseline separately. Common pre-route EQ cannot repair
        // the relative cancellation already present in that input.
        let failed_gates: Vec<_> = [
            (post < pre, "combined_objective_not_improved"),
            (
                cancellation_accepted,
                "cancellation_regressed_vs_without_post_eq",
            ),
            (
                target_accepted,
                "target_shortfall_not_good_or_materially_improved",
            ),
            (mains_preserved, "main_score_outside_crossover_regressed"),
            (output_preserved, "useful_output_loss"),
        ]
        .into_iter()
        .filter_map(|(passed, reason)| (!passed).then_some(reason))
        .collect();
        let accepted = failed_gates.is_empty();
        let metric = |value: Option<f64>| {
            value.map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.6}"))
        };
        let target_without = metric(pre_target_underfill_db);
        let target_with = metric(target_underfill_db);
        let target_reduction_pct = metric(
            pre_target_underfill_db
                .zip(target_underfill_db)
                .filter(|(before, _)| *before > 1e-9)
                .map(|(before, after)| 100.0 * (before - after) / before),
        );
        let cancellation_without = metric(cancellation_before_post_eq.as_ref().map(|e| e.final_db));
        let cancellation_with = metric(cancellation_evidence.as_ref().map(|e| e.final_db));
        let configured_baseline =
            metric(cancellation_evidence.as_ref().and_then(|e| e.baseline_db));
        let peak_eq_gain_db = main_eq_resp
            .iter()
            .map(|value| 20.0 * value.norm().log10())
            .fold(f64::NEG_INFINITY, f64::max);
        let cancellation_limit_db = config.optimizer.max_crossover_cancellation_db;
        let target_limit_db = roomeq_engine::topology::MAX_ACCEPTED_CROSSOVER_UNDERFILL_DB;
        let output_loss_limit_db = config.optimizer.finalization.max_useful_output_loss_db;
        let decision = if accepted { "accepted" } else { "discarded" };
        log::log!(target: BASS_MANAGEMENT_LOG_TARGET,
            if accepted { log::Level::Info } else { log::Level::Warn },
            "{role} Post-EQ {decision}: without/with this pass: target_shortfall_db={target_without}/{target_with}, target_shortfall_db_reduction_pct={target_reduction_pct}, cancellation_db={cancellation_without}/{cancellation_with}, combined_objective={pre:.6}/{post:.6}, main_only_score={main_pre_score:.6}/{main_post_score:.6}, main_score_above_{main_protected_min_hz:.1}_hz={main_protected_pre:.6}/{main_protected_post:.6}; configured_baseline_cancellation_db={configured_baseline}; cancellation_screened={cancellation_screened}, configured_cancellation_limit_db={cancellation_limit_db:.3}, target_limit_db={target_limit_db:.3}+0.05_or_20pct_and_1dB_improvement; useful_output_loss_db={output_loss:.6}, output_loss_limit_db={output_loss_limit_db:.6}, peak_eq_gain_db={peak_eq_gain_db:.6}; failed_gates={failed_gates:?}"
        );
        if accepted {
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
            post_eq_filters.insert(role.clone(), Vec::new());
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
            physical_correction_target: None,
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
            joint_sub: None,
            early_reflections: None,
            t60_octaves: None,
            speech_transmission: None,
            waterfall: None,
            resonance_decays: None,
            wavelet: None,
            early_late_curves: None,
            target_curve: routed_target_curves
                .get(role)
                .cloned()
                .or_else(|| pre_eq_target_curves.get(role).cloned()),
        };
        // This owning target was calibrated at the Main solve handoff.
        // Its pressure plane is independent of the common routed-sum target.
        chain.physical_correction_target = pre_eq_target_curves.get(role).cloned().map(|curve| {
            roomeq_model::PhysicalCorrectionTarget {
                curve,
                measurement_alignment_gain_db: gains.get(role).copied().unwrap_or(0.0),
            }
        });
        if let Some(target) = &chain.physical_correction_target {
            target
                .validate()
                .map_err(|message| AutoeqError::InvalidMeasurement { message })?;
        }
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
            Err(error) => log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
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
    // Deploy each selected low-pass before the physical driver sum. The
    // redirected routes omit a second group low-pass when these are active.
    // Legacy single-crossover configurations retain their shared route LP.
    let per_driver_low_passes: Vec<Option<output::PerDriverLowPass>> = sub_preprocess
        .drivers
        .as_ref()
        .map(|drivers| {
            drivers
                .iter()
                .enumerate()
                .map(|(index, driver)| {
                    per_driver_low_pass_plan(
                        config,
                        index,
                        &driver.name,
                        &sub_output_by_role,
                        &representative_bass_route_type,
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let mut driver_chains = sub_preprocess.drivers.as_ref().map(|drivers| {
        drivers
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let mut driver_plugins = d
                    .processing
                    .as_ref()
                    .map(|processing| {
                        processing
                            .plugins
                            .iter()
                            .cloned()
                            .map(|plugin| mark_plugin_stage(plugin, "post_route"))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
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
                let driver_curve = d.initial_curve.as_ref().map(|c| c.into());
                DriverDspChain {
                    measured_acoustics: None,
                    name: d.name.clone(),
                    index: i,
                    plugins: driver_plugins,
                    initial_curve: driver_curve,
                    measured_band_hz: None,
                }
            })
            .collect::<Vec<DriverDspChain>>()
    });
    if let Some(chains) = driver_chains.as_mut() {
        output::stamp_per_driver_low_passes(chains, &per_driver_low_passes, Some("post_route"));
        let deployed = per_driver_low_passes
            .iter()
            .flatten()
            .filter(|plan| plan.is_deployable())
            .count();
        if deployed > 0 {
            info!(target: BASS_MANAGEMENT_LOG_TARGET,
                "  Deployed per-sub low-pass to {deployed} sub driver(s) pre-sum (redirected-bass cutoff owned by drivers)"
            );
            bass_management_optimization
                .advisories
                .push(format!("per_sub_lp_deployed_to_drivers:{deployed}"));
        }
    }

    // The sub PEQ uses the preprocessed combined measurement as its input.
    let sub_initial_data: CurveData = (&pre_eq_initial_curves[&sub_role]).into();
    let sub_final_data: CurveData = (&final_sub_curve).into();
    let sub_eq_resp = output::compute_eq_response(&sub_initial_data, &sub_final_data);
    let sub_chain = ChannelDspChain {
        physical_correction_target: None,
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
        joint_sub: sub_preprocess.joint_sub.clone(),
        // Bass-managed sub chains combine drivers and routing; they carry
        // no single measured-IR acoustic report.
        early_late_curves: None,
        early_reflections: None,
        t60_octaves: None,
        speech_transmission: None,
        waterfall: None,
        resonance_decays: None,
        wavelet: None,
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
                Some(&bass_management_optimization),
            )?
        } else {
            (HashMap::new(), HashMap::new())
        };
    for (role, trim_db) in &post_dsp_input_trims {
        info!(target: BASS_MANAGEMENT_LOG_TARGET, " Post-DSP input level trim '{}': {:+.2} dB", role, trim_db);
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
                measurement_conditioning: None,
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
                audibility_veto: Vec::new(),
                veto_adjudication: None,
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
                measurement_conditioning: None,
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
                audibility_veto: Vec::new(),
                veto_adjudication: None,
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
    let mut final_post_eq_reverted = false;
    let mut splice_reverted_roles: Vec<String> = Vec::new();
    let mut splice_verdict_margins: Vec<String> = Vec::new();
    let role_crossover_hz = |role: &str| {
        let group_id =
            engine_home_cinema::group_id_for_role(engine_home_cinema::role_for_channel(role));
        group_results_by_id
            .get(group_id)
            .and_then(|group| group.selected_crossover_hz)
            .unwrap_or(final_xo_freq)
    };
    if let Some(graph) = bass_routing_graph.as_ref() {
        // Without calibrated main/sub timing there is no coherent verdict to
        // defer: reconstruct unenforced directly.
        let first_replay = if crossover_timing_refused(Some(&bass_management_optimization)) {
            reconstruct_deployed_source_curves_unenforced(
                &channel_chains,
                &pre_eq_fir_coeffs,
                graph,
                Some(&bass_management_optimization),
                sample_rate,
                output_dir,
            )
        } else {
            reconstruct_deployed_source_curves(
                &channel_chains,
                &pre_eq_fir_coeffs,
                graph,
                Some(&bass_management_optimization),
                sample_rate,
                output_dir,
            )
        }
        .or_else(|error| {
            // This executor assembles an internal correction candidate. Keep a
            // crossover-rejected candidate available to cumulative refinement;
            // the public pipeline must still pass strict final routed replay.
            // Missing measurements and other reconstruction errors are not
            // repairable by changing correction strength.
            if underfill_error_role(&error.to_string()).is_none() {
                return Err(error);
            }
            log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
                "Deferring candidate crossover rejection to final correction selection: {error}"
            );
            reconstruct_deployed_source_curves_unenforced(
                &channel_chains,
                &pre_eq_fir_coeffs,
                graph,
                Some(&bass_management_optimization),
                sample_rate,
                output_dir,
            )
        });
        deployed_source_curves = match first_replay {
            Ok(curves) => curves,
            Err(first_error)
                if channel_chains.values().any(|chain| {
                    chain.plugins.iter().any(|plugin| {
                        plugin
                            .parameters
                            .get("label")
                            .and_then(serde_json::Value::as_str)
                            == Some("post_eq")
                    })
                }) =>
            {
                log::warn!(target: BASS_MANAGEMENT_LOG_TARGET,
                    "Final serialized routed replay rejected Post-EQ; restoring the pre-Post-EQ snapshot: {first_error}"
                );
                let names = channel_chains.keys().cloned().collect::<Vec<_>>();
                for name in names {
                    let Some(mut chain) = channel_chains.get(&name).cloned() else {
                        continue;
                    };
                    let plugin_count = chain.plugins.len();
                    chain.plugins.retain(|plugin| {
                        plugin
                            .parameters
                            .get("label")
                            .and_then(serde_json::Value::as_str)
                            != Some("post_eq")
                    });
                    if chain.plugins.len() == plugin_count {
                        continue;
                    }

                    let initial: Curve = chain
                        .initial_curve
                        .clone()
                        .ok_or_else(|| AutoeqError::InvalidMeasurement {
                            message: format!(
                                "channel '{}' has no initial curve for Post-EQ rollback",
                                name
                            ),
                        })?
                        .into();
                    let embedded_irs = embedded_convolution_irs(
                        &chain.plugins,
                        pre_eq_fir_coeffs.get(&name).map(Vec::as_slice),
                    )?;
                    let realized = crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
                        &chain,
                        &initial,
                        sample_rate,
                        output_dir,
                        &embedded_irs,
                    )?;
                    let realized_data: CurveData = (&realized).into();
                    chain.eq_response = chain
                        .initial_curve
                        .as_ref()
                        .map(|initial| output::compute_eq_response(initial, &realized_data));
                    chain.final_curve = Some(realized_data);
                    channel_chains.insert(name.clone(), chain);

                    if let Some(channel_result) = channel_results.get_mut(&name) {
                        channel_result.final_curve = realized.clone();
                        channel_result.biquads.clear();
                        channel_result.post_score = if name == sub_role {
                            compute_flat_loss(&realized, sub_min_score, bass_route_upper_hz)
                        } else {
                            let group_id = engine_home_cinema::group_id_for_role(
                                engine_home_cinema::role_for_channel(&name),
                            );
                            let crossover_hz = group_results_by_id
                                .get(group_id)
                                .and_then(|group| group.selected_crossover_hz)
                                .unwrap_or(final_xo_freq);
                            compute_flat_loss(&realized, crossover_hz, max_freq)
                        };
                    }
                }
                final_post_eq_reverted = true;
                replay_until_splice_safe(
                    &mut channel_chains,
                    &mut channel_results,
                    &mut splice_reverted_roles,
                    &pre_eq_fir_coeffs,
                    graph,
                    Some(&bass_management_optimization),
                    sample_rate,
                    output_dir,
                    &role_crossover_hz,
                    &sub_role,
                    sub_min_score,
                    bass_route_upper_hz,
                    max_freq,
                )?
            }
            Err(_) => replay_until_splice_safe(
                &mut channel_chains,
                &mut channel_results,
                &mut splice_reverted_roles,
                &pre_eq_fir_coeffs,
                graph,
                Some(&bass_management_optimization),
                sample_rate,
                output_dir,
                &role_crossover_hz,
                &sub_role,
                sub_min_score,
                bass_route_upper_hz,
                max_freq,
            )?,
        };
        // WP4d transfer evidence: per-role splice margins at the verdict,
        // on the exact chains the gate just passed. Without calibrated
        // main/sub timing there is no coherent verdict to record.
        splice_verdict_margins = if crossover_timing_refused(Some(&bass_management_optimization)) {
            vec!["splice_verdict_margins_unassessed:coherent_timing_refused".to_string()]
        } else {
            match collect_routed_splice_outcomes(
                &channel_chains,
                &pre_eq_fir_coeffs,
                graph,
                Some(&bass_management_optimization),
                sample_rate,
                output_dir,
            ) {
                Ok((_, outcomes)) => outcomes.iter().map(format_splice_margin).collect(),
                Err(error) => {
                    vec![format!("splice_verdict_margins_unassessed:{error}")]
                }
            }
        };
    }
    post_scores = channel_results
        .values()
        .map(|result| result.post_score)
        .collect();

    let avg_pre = pre_scores.iter().sum::<f64>() / pre_scores.len() as f64;
    let avg_post = post_scores.iter().sum::<f64>() / post_scores.len() as f64;

    info!(target: BASS_MANAGEMENT_LOG_TARGET,
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
        finalized_decisions: None,
        channels: channel_chains,
        channel_results,
        deployed_source_curves,
        combined_pre_score: avg_pre,
        combined_post_score: avg_post,
        metadata: OptimizationMetadata {
            final_convolution_sha256: None,
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
            audibility_veto: None,
            veto_adjudication: None,
            optimizer_evidence: None,
            stage_outcomes: {
                let mut outcomes = post_eq_main_preservation;
                for (channel, loss) in post_eq_output_rejections {
                    outcomes.push(StageOutcome {
                        stage: format!("post_eq_useful_output_{channel}"),
                        status: StageStatus::Degraded,
                        checks: Vec::new(),
                        advisories: vec![format!(
                            "candidate_discarded; representative_stage_unexplained_loss_db={loss}; limit_db={}; final_native_seat_validation_still_required",
                            config.optimizer.finalization.max_useful_output_loss_db
                        )],
                    });
                }
                let mut generated_channels: Vec<_> = pre_eq_fir_coeffs.iter().collect();
                generated_channels.sort_by_key(|(name, _)| *name);
                for (name, coefficients) in generated_channels {
                    outcomes.push(StageOutcome {
                        checks: Vec::new(),
                        stage: format!("routed_pre_eq_fir_{name}"),
                        status: StageStatus::Applied,
                        advisories: vec![format!("generated_taps={}", coefficients.len())],
                    });
                }
                if final_post_eq_reverted {
                    outcomes.push(StageOutcome {
                        checks: Vec::new(),
                        stage: "routed_post_eq_final_replay".to_string(),
                        status: StageStatus::Degraded,
                        advisories: vec![
                            "post_eq_reverted_after_serialized_route_replay".to_string(),
                        ],
                    });
                }
                if !splice_reverted_roles.is_empty() {
                    outcomes.push(StageOutcome {
                        checks: Vec::new(),
                        stage: "routed_splice_final_replay".to_string(),
                        status: StageStatus::Degraded,
                        advisories: splice_reverted_roles
                            .iter()
                            .map(|stage| format!("splice_cancellation_reverted_{stage}"))
                            .collect(),
                    });
                }
                if !splice_verdict_margins.is_empty() {
                    outcomes.push(StageOutcome {
                        checks: Vec::new(),
                        stage: "splice_verdict_margins".to_string(),
                        status: StageStatus::Applied,
                        advisories: splice_verdict_margins,
                    });
                }
                outcomes
            },
            qa_seed_distribution: None,
            effective_config: None,
            t60_flatness_tolerance_s: config.report_t60_tolerance_s(),
            operation_gates: None,
            provisional_decisions: crossover_provisional,
            epa_provenance: None,
            playback_summary: None,
        },
    })
}

#[cfg(test)]
mod post_dsp_level_tests {
    use super::{
        apply_gain_to_main_chain, apply_output_safety_gain, average_spl,
        calibrate_post_dsp_input_levels, physical_sub_tonal_objective_curve,
        realize_routed_main_training_branch, realize_routed_sub_training_branch,
        realize_source_pre_route_transfer, reconstruct_deployed_source_curves,
        stage_main_correction_plugins, stage_sub_correction_plugins,
    };
    use roomeq_engine::Curve;
    use roomeq_engine::topology::{mark_plugin_stage, mark_route_owned_plugin};
    use roomeq_engine::{bass_management as engine_bass_management, output, response};
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

    #[test]
    fn common_post_eq_preserves_inherited_crossover_cancellation() {
        let config = roomeq_model::RoomConfig::default();
        let main = curve(0.0);
        let mut bass = main.clone();
        bass.phase.as_mut().unwrap().fill(160.0);
        let before =
            super::post_eq_crossover_cancellation(&config, "L", &main, &bass, 80.0).unwrap();
        let filters = [math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            80.0,
            48_000.0,
            1.0,
            6.0,
        )];
        let transfer =
            roomeq_engine::response::compute_peq_complex_response(&filters, &main.freq, 48_000.0);
        let main_after = roomeq_engine::response::apply_complex_response(&main, &transfer);
        let bass_after = roomeq_engine::response::apply_complex_response(&bass, &transfer);
        let after =
            super::post_eq_crossover_cancellation(&config, "L", &main_after, &bass_after, 80.0)
                .unwrap();
        let sum_before = roomeq_engine::topology::complex_sum_mains(&[&main, &bass]);
        let sum_after = roomeq_engine::topology::complex_sum_mains(&[&main_after, &bass_after]);
        assert!((after.final_db - before.final_db).abs() < 1e-9);
        assert!((sum_after.spl[2] - sum_before.spl[2] - 6.0).abs() < 1e-9);
        assert!(!before.accepted && !after.accepted);
    }

    #[test]
    fn post_eq_cancellation_checks_mismatched_grids_and_missing_phase() {
        let config = roomeq_model::RoomConfig::default();
        let main = curve(0.0);
        let mut bass = Curve {
            freq: ndarray::array![20.0, 80.0, 200.0, 400.0],
            spl: ndarray::Array1::from_elem(4, -12.0),
            phase: Some(ndarray::Array1::from_elem(4, 180.0)),
            ..Curve::default()
        };
        let before = super::post_eq_crossover_cancellation(&config, "L", &main, &bass, 80.0)
            .expect("different grids still provide cancellation evidence");
        assert!(before.accepted);
        // A sub boost brings the opposite-phase branches closer in level.
        // The old Post-EQ screening silently lost this evidence on unequal grids.
        bass.spl.fill(-6.0);
        let after = super::post_eq_crossover_cancellation(&config, "L", &main, &bass, 80.0)
            .expect("boosted sub must also be assessed");
        assert!(!after.accepted);
        assert!(after.final_db > 6.0);
        bass.phase = None;
        assert!(super::post_eq_crossover_cancellation(&config, "L", &main, &bass, 80.0).is_none());
    }

    #[test]
    fn calibration_uses_shared_passband_not_bass_correction_bounds() {
        let mut left = Curve {
            freq: ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 192),
            spl: ndarray::Array1::from_elem(192, 80.0),
            ..Curve::default()
        };
        let mut right = left.clone();
        for ((frequency, left_level), right_level) in left
            .freq
            .iter()
            .zip(left.spl.iter_mut())
            .zip(right.spl.iter_mut())
        {
            // Different room bass must not turn into broadband level trims.
            if *frequency < 200.0 {
                *left_level += 10.0;
                *right_level -= 10.0;
            }
            // A limited-band surround need not extend to 16 or 20 kHz.
            if *frequency > 4_000.0 {
                *right_level -= 50.0;
            }
        }
        let roles = vec!["L".to_string(), "R".to_string()];
        let curves = HashMap::from([("L".into(), left), ("R".into(), right)]);
        let band = super::main_level_alignment_band(&curves, &roles, 120.0).unwrap();
        assert!(band.0 >= 240.0 && band.1 <= 2_000.0);
        assert!(band.1 >= 2.0 * band.0);
        assert!((average_spl(&curves["L"], band) - average_spl(&curves["R"], band)).abs() < 0.01);
    }

    #[test]
    fn calibration_refuses_disjoint_passbands_instead_of_using_eq_bounds() {
        let mut low = curve(80.0);
        low.freq = ndarray::array![20.0, 40.0, 60.0, 80.0, 100.0, 120.0];
        let mut high = curve(80.0);
        high.freq = ndarray::array![500.0, 800.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0];
        let curves = HashMap::from([("L".into(), low), ("R".into(), high)]);
        assert!(
            super::main_level_alignment_band(&curves, &["L".into(), "R".into()], 80.0).is_err()
        );
    }

    #[test]
    fn calibration_ignores_bass_mode_below_search_range() {
        // Measured 2.2_genelec mains carry a +20 dB room mode near 41 Hz.
        // The mode sits below twice the crossover, so it must not set the
        // passband reference and veto the shared midrange octave.
        let mut left = Curve {
            freq: ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 192),
            spl: ndarray::Array1::from_elem(192, 80.0),
            ..Curve::default()
        };
        for (frequency, level) in left.freq.iter().zip(left.spl.iter_mut()) {
            if *frequency < 60.0 {
                *level += 20.0;
            }
        }
        let right = left.clone();
        let curves = HashMap::from([("L".into(), left), ("R".into(), right)]);
        let band =
            super::main_level_alignment_band(&curves, &["L".into(), "R".into()], 72.0).unwrap();
        assert!(band.0 >= 144.0 && band.1 <= 2_000.0);
        assert!(band.1 >= 2.0 * band.0);
    }

    #[test]
    fn post_eq_output_guard_retains_bass_below_main_scoring_band() {
        use super::post_eq_useful_output_loss;
        let before = curve(80.0);
        let mut after = before.clone();
        for (frequency, level) in after.freq.iter().zip(after.spl.iter_mut()) {
            if *frequency <= 80.0 {
                *level -= 20.0;
            }
        }
        assert!(
            post_eq_useful_output_loss(&before, &after, Some(&before), 20.0, 400.0).unwrap() > 10.0
        );
        assert_eq!(
            post_eq_useful_output_loss(&before, &after, Some(&before), 100.0, 400.0).unwrap(),
            0.0
        );
        assert_eq!(
            post_eq_useful_output_loss(&before, &before, Some(&before), 20.0, 400.0).unwrap(),
            0.0
        );
        let peak = curve(95.0);
        assert_eq!(
            post_eq_useful_output_loss(&peak, &before, Some(&before), 20.0, 400.0).unwrap(),
            0.0
        );
    }

    fn chain(name: &str, initial: Curve, final_curve: Option<Curve>) -> ChannelDspChain {
        ChannelDspChain {
            physical_correction_target: None,
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
            joint_sub: None,
            early_reflections: None,
            t60_octaves: None,
            speech_transmission: None,
            waterfall: None,
            resonance_decays: None,
            wavelet: None,
            early_late_curves: None,
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

    fn recalibration_fixture(
        route_gain_db: f64,
    ) -> (
        roomeq_engine::room_result::RoomOptimizationResult,
        roomeq_model::RoomConfig,
    ) {
        use roomeq_model::{
            BassManagementConfig, CrossoverConfig, MeasurementSource, RoomConfig, SpeakerConfig,
            SubwooferCrossoverRef, SubwooferOutput, SubwooferStrategy, SubwooferSystemConfig,
            SystemConfig, SystemModel,
        };

        let left = curve(80.0);
        let right = curve(80.0);
        let sub = curve(70.0);
        let mut config = RoomConfig::default();
        config.system = Some(SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([
                ("L".to_string(), "left".to_string()),
                ("R".to_string(), "right".to_string()),
            ]),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                routing: Default::default(),
                outputs: vec![SubwooferOutput {
                    id: "Sub1".to_string(),
                    speaker: "sub".to_string(),
                }],
                crossover: Some(SubwooferCrossoverRef::Shared("bass".to_string())),
            }),
            bass_management: Some(BassManagementConfig {
                headroom_margin_db: 10.0,
                ..Default::default()
            }),
            ..Default::default()
        });
        config.crossovers = Some(HashMap::from([(
            "bass".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: Some(80.0),
                frequencies: None,
                frequency_range: None,
            },
        )]));
        for (name, measured) in [("left", &left), ("right", &right), ("sub", &sub)] {
            config.speakers.insert(
                name.to_string(),
                SpeakerConfig::Single(MeasurementSource::InMemory(measured.clone())),
            );
        }

        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let channel_template = result.channel_results["L"].clone();
        for (role, measured) in [("L", &left), ("R", &right), ("Sub1", &sub)] {
            if !result.channels.contains_key(role) {
                result.channels.insert(
                    role.to_string(),
                    chain(role, measured.clone(), Some(measured.clone())),
                );
            }
            let channel = result
                .channel_results
                .entry(role.to_string())
                .or_insert_with(|| channel_template.clone());
            channel.name = role.to_string();
            channel.initial_curve = measured.clone();
            channel.final_curve = measured.clone();
            channel.biquads.clear();
            channel.fir_coeffs = None;
            let dsp_chain = result.channels.get_mut(role).unwrap();
            dsp_chain.initial_curve = Some(CurveData::from(measured));
            dsp_chain.final_curve = Some(CurveData::from(measured));
            dsp_chain.target_curve = Some(CurveData::from(measured));
        }
        let mut speaker_calibration =
            mark_plugin_stage(roomeq_engine::output::create_gain_plugin(2.0), "pre_route");
        speaker_calibration.parameters["label"] = serde_json::json!("speaker_calibration");
        result
            .channels
            .get_mut("L")
            .unwrap()
            .plugins
            .push(speaker_calibration);
        let mut configured_sub_gain =
            mark_plugin_stage(roomeq_engine::output::create_gain_plugin(3.0), "post_route");
        configured_sub_gain.parameters["label"] = serde_json::json!("configured_sub_gain");
        result
            .channels
            .get_mut("Sub1")
            .unwrap()
            .plugins
            .push(configured_sub_gain);

        let mut report = roomeq_engine::home_cinema::bass_management_report(&config, None, false)
            .expect("fixture config should produce a Home Cinema report");
        let graph = report
            .routing_graph
            .as_mut()
            .expect("fixture report should include its routing graph");
        for route in graph
            .routes
            .iter_mut()
            .filter(|route| route.route_kind == "redirected_bass_lowpass_to_sub")
        {
            route.gain_db = route_gain_db;
            route.gain_linear = 10.0_f64.powf(route_gain_db / 20.0);
            route.matrix_gain = route.gain_linear;
        }
        result.metadata.bass_management = Some(report);
        (result, config)
    }

    fn seed_post_dsp_calibration(
        result: &mut roomeq_engine::room_result::RoomOptimizationResult,
        config: &roomeq_model::RoomConfig,
    ) {
        let mut graph = result
            .metadata
            .bass_management
            .as_ref()
            .and_then(|report| report.routing_graph.clone())
            .expect("fixture has a routing graph");
        let roles = ["L".to_string(), "R".to_string()];
        let (_, deployed) = calibrate_post_dsp_input_levels(
            config,
            &roles,
            "Sub1",
            (100.0, 400.0),
            48_000.0,
            std::path::Path::new("."),
            &HashMap::new(),
            &mut result.channels,
            &mut graph,
            None,
        )
        .expect("fixture calibration should complete");
        result.deployed_source_curves = deployed;
        let report = result.metadata.bass_management.as_mut().unwrap();
        report.routing_graph = Some(graph.clone());
        let effective = roomeq_engine::home_cinema::effective_bass_management(config).unwrap();
        report.headroom_simulation = roomeq_engine::home_cinema::simulate_bass_bus_headroom(
            Some(&graph),
            &effective.config.headroom_model,
            effective.config.headroom_margin_db,
            48_000.0,
        );
    }

    fn set_redirected_route_gain(graph: &mut BassManagementRoutingGraph, gain_db: f64) {
        for route in graph
            .routes
            .iter_mut()
            .filter(|route| route.route_kind == "redirected_bass_lowpass_to_sub")
        {
            route.gain_db = gain_db;
            route.gain_linear = 10.0_f64.powf(gain_db / 20.0);
            route.matrix_gain = route.gain_linear;
        }
    }

    fn final_electrical_headroom_cuts(
        result: &roomeq_engine::room_result::RoomOptimizationResult,
    ) -> HashMap<String, f64> {
        result
            .channels
            .iter()
            .filter_map(|(output, chain)| {
                let cut_db = chain
                    .plugins
                    .iter()
                    .filter(|plugin| {
                        plugin
                            .parameters
                            .get("label")
                            .and_then(serde_json::Value::as_str)
                            == Some("final_electrical_headroom")
                    })
                    .map(|plugin| -plugin.parameters["gain_db"].as_f64().unwrap())
                    .sum::<f64>();
                (cut_db > 0.0).then(|| (output.clone(), cut_db))
            })
            .collect()
    }

    #[test]
    fn deployed_array_applies_driver_controls_exactly_once() {
        use roomeq_model::{
            BassManagementSourceReport, BassManagementSubOutputReport, DriverDspChain,
        };
        let bass_measurement = Curve {
            freq: ndarray::array![20.0, 40.0, 80.0, 100.0, 200.0],
            spl: ndarray::array![80.0, 80.0, 80.0, 70.0, 20.0],
            phase: Some(ndarray::Array1::zeros(5)),
            ..Default::default()
        };
        let mut initial_sum = bass_measurement.clone();
        initial_sum.spl.mapv_inplace(|level| level + 6.020599913);
        let mut sub_chain = chain("LFE", initial_sum, None);
        sub_chain.plugins = vec![mark_plugin_stage(
            roomeq_engine::output::create_gain_plugin(-12.020599913),
            "post_route",
        )];
        let mut drivers = Vec::new();
        let mut outputs = Vec::new();
        let mut routes = Vec::new();
        for (index, delay) in [4.0, 10.0].into_iter().enumerate() {
            let name = format!("sub{index}");
            let mut raw = bass_measurement.clone();
            raw.phase = Some(
                raw.freq
                    .mapv(|frequency| 360.0 * frequency * delay / 1000.0),
            );
            drivers.push(DriverDspChain {
                measured_acoustics: None,
                name: name.clone(),
                index,
                // The serialized driver gain and output report describe the
                // same physical control, not two cascaded gains.
                plugins: vec![
                    roomeq_engine::output::create_gain_plugin(6.0),
                    roomeq_engine::output::create_delay_plugin(delay),
                ],
                initial_curve: Some((&raw).into()),
                measured_band_hz: match (raw.freq.first(), raw.freq.last()) {
                    (Some(&low), Some(&high)) => Some([low, high]),
                    _ => None,
                },
            });
            outputs.push(BassManagementSubOutputReport {
                output_role: name.clone(),
                gain_db: 6.0,
                delay_ms: delay,
                polarity_inverted: false,
                strategy_source: "mso".into(),
                headroom_contribution_db: 0.0,
                selected_low_pass_hz: None,
            });
            let mut route = low_route("L", 0);
            route.destination = name;
            route.destination_index = index + 2;
            route.gain_db = 6.0;
            route.gain_linear = 10.0_f64.powf(6.0 / 20.0);
            route.matrix_gain = route.gain_linear;
            route.delay_ms = delay;
            routes.push(route);
        }
        sub_chain.drivers = Some(drivers);
        let source = BassManagementSourceReport {
            source_channel: "L".into(),
            group_id: "lcr".into(),
            main_delay_ms: 0.0,
            bass_route_delay_ms: 0.0,
            polarity_inverted: false,
            trim_db: 0.0,
            objective_before: None,
            objective_after: None,
            accepted: false,
            safety_restored: false,
            advisories: Vec::new(),
        };
        let optimization = super::joint_bass_management_report_from_parts(&[], &[source], &outputs);
        let graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".into(),
            physical_sub_outputs: vec!["sub0".into(), "sub1".into()],
            stereo_routing: None,
            input_channels: vec!["L".into()],
            output_channels: vec!["L".into(), "LFE".into(), "sub0".into(), "sub1".into()],
            routes,
            matrix: None,
            input_trim_db: HashMap::new(),
            post_dsp_main_alignment_band_hz: None,
            advisories: Vec::new(),
        };
        let main = Curve {
            freq: ndarray::array![20.0, 40.0, 80.0, 100.0, 200.0, 400.0, 16000.0],
            spl: ndarray::Array1::from_elem(7, 20.0),
            phase: Some(ndarray::Array1::zeros(7)),
            ..Default::default()
        };
        let channels = HashMap::from([
            ("L".into(), chain("L", main.clone(), None)),
            ("LFE".into(), sub_chain),
        ]);
        let actual = reconstruct_deployed_source_curves(
            &channels,
            &HashMap::new(),
            &graph,
            Some(&optimization),
            48000.0,
            std::path::Path::new("."),
        )
        .unwrap();
        let bass = roomeq_engine::topology::apply_crossover_response_to_curve(
            &roomeq_engine::topology::interpolate_bass_response(&main.freq, &bass_measurement),
            "LR24",
            80.0,
            48000.0,
            true,
        );
        let expected = roomeq_engine::topology::complex_sum_mains(&[&main, &bass]);
        for (index, &frequency) in main.freq.iter().enumerate() {
            if frequency <= 200.0 {
                assert!(
                    (actual["L"].spl[index] - expected.spl[index]).abs() < 1e-6,
                    "driver controls mismatch at {frequency}: {} vs {}",
                    actual["L"].spl[index],
                    expected.spl[index]
                );
            } else {
                assert!(
                    (actual["L"].spl[index] - main.spl[index]).abs() < 0.01,
                    "unmeasured rolled-off bass affected full-range main at {frequency}: {}",
                    actual["L"].spl[index]
                );
            }
        }
    }

    #[test]
    fn collect_splice_outcomes_isolates_sibling_assessment_failure() {
        use roomeq_model::{
            BassManagementSourceReport, BassManagementSubOutputReport, DriverDspChain,
        };
        // L carries a crossover; R routes omit every crossover frequency so
        // only R is unassessable. Strict keeps first-failure behavior while
        // Collect reports both roles.
        let bass_measurement = Curve {
            freq: ndarray::array![20.0, 40.0, 80.0, 100.0, 200.0],
            spl: ndarray::array![80.0, 80.0, 80.0, 70.0, 20.0],
            phase: Some(ndarray::Array1::zeros(5)),
            ..Default::default()
        };
        let mut initial_sum = bass_measurement.clone();
        initial_sum.spl.mapv_inplace(|level| level + 6.020599913);
        let mut sub_chain = chain("LFE", initial_sum, None);
        let mut drivers = Vec::new();
        let mut outputs = Vec::new();
        let mut routes = Vec::new();
        for (index, delay) in [4.0, 10.0].into_iter().enumerate() {
            let name = format!("sub{index}");
            let mut raw = bass_measurement.clone();
            raw.phase = Some(
                raw.freq
                    .mapv(|frequency| 360.0 * frequency * delay / 1000.0),
            );
            drivers.push(DriverDspChain {
                measured_acoustics: None,
                name: name.clone(),
                index,
                plugins: vec![
                    roomeq_engine::output::create_gain_plugin(6.0),
                    roomeq_engine::output::create_delay_plugin(delay),
                ],
                initial_curve: Some((&raw).into()),
                measured_band_hz: match (raw.freq.first(), raw.freq.last()) {
                    (Some(&low), Some(&high)) => Some([low, high]),
                    _ => None,
                },
            });
            outputs.push(BassManagementSubOutputReport {
                output_role: name.clone(),
                gain_db: 6.0,
                delay_ms: delay,
                polarity_inverted: false,
                strategy_source: "mso".into(),
                headroom_contribution_db: 0.0,
                selected_low_pass_hz: None,
            });
            for (source, source_index, low_pass_hz) in [("L", 0, Some(80.0)), ("R", 1, None)] {
                let mut route = low_route(source, source_index);
                route.destination = name.clone();
                route.destination_index = index + 2;
                route.gain_db = 6.0;
                route.gain_linear = 10.0_f64.powf(6.0 / 20.0);
                route.matrix_gain = route.gain_linear;
                route.delay_ms = delay;
                route.low_pass_hz = low_pass_hz;
                routes.push(route);
            }
        }
        sub_chain.drivers = Some(drivers);
        let source = |role: &str| BassManagementSourceReport {
            source_channel: role.into(),
            group_id: "lcr".into(),
            main_delay_ms: 0.0,
            bass_route_delay_ms: 0.0,
            polarity_inverted: false,
            trim_db: 0.0,
            objective_before: None,
            objective_after: None,
            accepted: false,
            safety_restored: false,
            advisories: Vec::new(),
        };
        let optimization = super::joint_bass_management_report_from_parts(
            &[],
            &[source("L"), source("R")],
            &outputs,
        );
        let graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".into(),
            physical_sub_outputs: vec!["sub0".into(), "sub1".into()],
            stereo_routing: None,
            input_channels: vec!["L".into(), "R".into()],
            output_channels: vec!["L".into(), "R".into(), "LFE".into()],
            routes,
            matrix: None,
            input_trim_db: HashMap::new(),
            post_dsp_main_alignment_band_hz: None,
            advisories: Vec::new(),
        };
        let main = Curve {
            freq: ndarray::array![20.0, 40.0, 80.0, 100.0, 200.0, 400.0, 16000.0],
            spl: ndarray::Array1::from_elem(7, 20.0),
            phase: Some(ndarray::Array1::zeros(7)),
            ..Default::default()
        };
        let channels = HashMap::from([
            ("L".into(), chain("L", main.clone(), None)),
            ("R".into(), chain("R", main.clone(), None)),
            ("LFE".into(), sub_chain),
        ]);
        let strict = reconstruct_deployed_source_curves(
            &channels,
            &HashMap::new(),
            &graph,
            Some(&optimization),
            48000.0,
            std::path::Path::new("."),
        )
        .expect_err("strict reconstruction keeps first-failure behavior");
        assert!(
            strict
                .to_string()
                .contains("missing routed crossover frequency for 'R'"),
            "unexpected strict failure: {strict}"
        );
        let (curves, outcomes) = super::collect_routed_splice_outcomes(
            &channels,
            &HashMap::new(),
            &graph,
            Some(&optimization),
            48000.0,
            std::path::Path::new("."),
        )
        .expect("collect must isolate the R failure");
        assert!(curves.contains_key("L") && curves.contains_key("R"));
        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0].role, "L");
        assert!(
            outcomes[0].assessed.is_ok(),
            "L must stay assessed: {:?}",
            outcomes[0].assessed
        );
        assert_eq!(outcomes[1].role, "R");
        assert!(
            outcomes[1]
                .assessed
                .as_ref()
                .is_err_and(|error| error.contains("missing routed crossover frequency")),
            "R must record its hard failure: {:?}",
            outcomes[1].assessed
        );
    }

    #[test]
    fn stereo_candidate_serialized_replay_rejects_acoustic_splice_cancellation() {
        let main_roles = vec!["L".to_string(), "R".to_string()];
        let measured = Curve {
            freq: ndarray::array![20.0, 40.0, 80.0, 160.0, 320.0],
            spl: ndarray::Array1::from_elem(5, 80.0),
            phase: Some(ndarray::Array1::zeros(5)),
            ..Default::default()
        };
        let transfer = Curve {
            freq: measured.freq.clone(),
            spl: ndarray::Array1::zeros(5),
            phase: Some(ndarray::Array1::zeros(5)),
            ..Default::default()
        };
        let mains = HashMap::from([
            ("L".to_string(), measured.clone()),
            ("R".to_string(), measured.clone()),
        ]);
        let transfers = HashMap::from([
            ("L".to_string(), transfer.clone()),
            ("R".to_string(), transfer),
        ]);
        let groups = std::collections::BTreeMap::from([(
            "lcr".to_string(),
            roomeq_model::BassManagementGroupReport {
                group_id: "lcr".to_string(),
                roles: main_roles.clone(),
                crossover_type: "LR24".to_string(),
                selected_crossover_hz: Some(80.0),
                configured_crossover_hz: Some(80.0),
                main_delay_ms: 0.0,
                bass_route_delay_ms: 0.0,
                polarity_inverted: false,
                trim_db: 0.0,
                objective_before: None,
                objective_after: None,
                selected_sub_low_pass_hz: Vec::new(),
                advisories: Vec::new(),
            },
        )]);
        let source = |role: &str| roomeq_model::BassManagementSourceReport {
            source_channel: role.to_string(),
            group_id: "lcr".to_string(),
            main_delay_ms: 0.0,
            bass_route_delay_ms: 0.0,
            polarity_inverted: false,
            trim_db: 0.0,
            objective_before: Some(1.0),
            objective_after: Some(0.5),
            accepted: true,
            safety_restored: false,
            advisories: Vec::new(),
        };
        let mut sources = vec![source("L"), source("R")];
        let outputs = ["Sub1", "Sub2"]
            .into_iter()
            .map(|role| roomeq_model::BassManagementSubOutputReport {
                output_role: role.to_string(),
                gain_db: 0.0,
                delay_ms: 0.0,
                polarity_inverted: false,
                strategy_source: "mso".to_string(),
                headroom_contribution_db: 0.0,
                selected_low_pass_hz: None,
            })
            .collect::<Vec<_>>();
        let drivers = ["Sub1", "Sub2"]
            .into_iter()
            .map(|name| roomeq_engine::bass_management::SubDriverInfo {
                name: name.to_string(),
                gain: 0.0,
                delay: 0.0,
                inverted: false,
                processing: None,
                initial_curve: Some(measured.clone()),
            })
            .collect::<Vec<_>>();
        let direct = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        assert!(
            super::stereo_candidate_serialized_replay_rejection(
                &roomeq_model::RoomConfig::default(),
                &main_roles,
                &mains,
                &transfers,
                &groups,
                &sources,
                &outputs,
                &drivers,
                None,
                &direct,
                48_000.0,
            )
            .is_ok()
        );

        sources[0].polarity_inverted = true;
        let rejection = super::stereo_candidate_serialized_replay_rejection(
            &roomeq_model::RoomConfig::default(),
            &main_roles,
            &mains,
            &transfers,
            &groups,
            &sources,
            &outputs,
            &drivers,
            None,
            &direct,
            48_000.0,
        )
        .expect_err("inverting one LR24 bass branch must fail serialized splice replay");
        assert!(rejection.contains("serialized_graph_acoustic_splice_underfill:L"));
        sources[1].polarity_inverted = true;
        let mut corrected_sub = measured.clone();
        corrected_sub
            .phase
            .as_mut()
            .unwrap()
            .mapv_inplace(|phase| phase + 180.0);
        super::stereo_candidate_serialized_replay_rejection(
            &roomeq_model::RoomConfig::default(),
            &main_roles,
            &mains,
            &transfers,
            &groups,
            &sources,
            &outputs,
            &drivers,
            Some((&measured, &corrected_sub)),
            &direct,
            48_000.0,
        )
        .expect("shared sub correction must participate in serialized phase replay");
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
    fn source_pre_route_realization_includes_calibration_and_excludes_post_route_dsp() {
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
                .all(|level| (*level + 9.0).abs() < 1.0e-12)
        );
        let phase = transfer.phase.as_ref().expect("delay phase");
        for (frequency, phase) in transfer.freq.iter().zip(phase.iter()) {
            let expected = -360.0 * frequency * delay_ms / 1_000.0;
            let wrapped_error = (phase - expected + 180.0).rem_euclid(360.0) - 180.0;
            assert!(wrapped_error.abs() < 1.0e-10);
        }
    }

    #[test]
    fn deployed_main_post_route_processing_only_changes_direct_branch() {
        let main_input = curve(80.0);
        let sub_input = curve(80.0);
        let mut main = chain("L", main_input.clone(), Some(main_input));
        main.plugins.push(mark_plugin_stage(
            roomeq_engine::output::create_gain_plugin(2.0),
            "pre_route",
        ));
        main.plugins.push(mark_plugin_stage(
            roomeq_engine::output::create_gain_plugin(3.0),
            "post_route",
        ));
        let mut sub = chain("LFE", sub_input.clone(), Some(sub_input));
        sub.plugins.push(mark_plugin_stage(
            roomeq_engine::output::create_gain_plugin(4.0),
            "post_route",
        ));

        let mut direct = low_route("L", 0);
        direct.destination = "L".to_string();
        direct.destination_index = 0;
        direct.pre_chain_channel = Some("L".to_string());
        direct.post_chain_channel = Some("L".to_string());
        direct.route_kind = "main_highpass_to_self".to_string();
        direct.low_pass_hz = None;
        let mut redirected = low_route("L", 0);
        redirected.destination_index = 1;
        redirected.gain_db = -6.0;
        redirected.gain_linear = 10.0_f64.powf(-6.0 / 20.0);
        redirected.matrix_gain = redirected.gain_linear;
        redirected.low_pass_hz = None;
        let mut lfe = low_route("LFE", 1);
        lfe.low_pass_hz = None;
        let graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".to_string(),
            physical_sub_outputs: Vec::new(),
            stereo_routing: None,
            input_channels: vec!["L".to_string(), "LFE".to_string()],
            output_channels: vec!["L".to_string(), "LFE".to_string()],
            routes: vec![direct, redirected, lfe],
            matrix: None,
            input_trim_db: HashMap::new(),
            post_dsp_main_alignment_band_hz: None,
            advisories: Vec::new(),
        };

        let deployed = super::reconstruct_deployed_source_curves_unenforced(
            &HashMap::from([("L".to_string(), main), ("LFE".to_string(), sub)]),
            &HashMap::new(),
            &graph,
            None,
            48_000.0,
            std::path::Path::new("."),
        )
        .expect("resolve input, direct output, and physical-sub stages");

        let direct_amplitude = 10.0_f64.powf(85.0 / 20.0);
        let redirected_amplitude = 10.0_f64.powf(80.0 / 20.0);
        let expected_level = 20.0 * (direct_amplitude + redirected_amplitude).log10();
        let actual = &deployed["L"].spl;
        assert!(
            actual
                .iter()
                .all(|level| (level - expected_level).abs() < 1.0e-9),
            "expected direct branch +2 dB input +3 dB output and redirected branch +2 dB input -6 dB route +4 dB sub output (no main output EQ leak): {actual:?}"
        );
    }

    #[test]
    fn routed_training_with_sub_post_eq_and_trim_matches_emitted_graph() {
        let frequencies = ndarray::array![
            20.0, 31.0, 40.0, 60.0, 80.0, 100.0, 160.0, 250.0, 400.0, 800.0, 2_000.0
        ];
        let main_input = Curve {
            freq: frequencies.clone(),
            spl: ndarray::array![
                61.0, 63.0, 66.0, 69.0, 72.0, 74.0, 76.0, 77.0, 78.0, 79.0, 80.0
            ],
            phase: Some(ndarray::Array1::zeros(frequencies.len())),
            ..Curve::default()
        };
        let sub_input = Curve {
            freq: frequencies.clone(),
            spl: ndarray::array![
                78.0, 80.0, 82.0, 84.0, 82.0, 78.0, 70.0, 55.0, 30.0, 10.0, 0.0
            ],
            phase: Some(ndarray::Array1::zeros(frequencies.len())),
            ..Curve::default()
        };
        let sample_rate = 48_000.0;
        let main_output_eq = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            110.0,
            sample_rate,
            1.1,
            2.5,
        );
        let sub_output_eq = [
            math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                40.0,
                sample_rate,
                1.4,
                -4.0,
            ),
            math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                63.0,
                sample_rate,
                1.0,
                2.0,
            ),
        ];
        let source_post_eq = [math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            48.0,
            sample_rate,
            0.9,
            1.25,
        )];
        let mut main_plugins = vec![mark_plugin_stage(
            roomeq_engine::output::create_gain_plugin(-0.3),
            "pre_route",
        )];
        main_plugins.push(mark_plugin_stage(
            roomeq_engine::output::create_eq_plugin(std::slice::from_ref(&main_output_eq)),
            "post_route",
        ));
        let mut emitted_main_plugins = main_plugins.clone();
        emitted_main_plugins.push(mark_plugin_stage(
            roomeq_engine::output::create_eq_plugin(&source_post_eq),
            "pre_route",
        ));
        emitted_main_plugins.push(mark_route_owned_plugin(output::create_crossover_plugin(
            "LR24", 80.0, "high",
        )));
        let mut emitted_sub_plugins = vec![mark_plugin_stage(
            roomeq_engine::output::create_eq_plugin(&sub_output_eq),
            "post_route",
        )];
        emitted_sub_plugins.push(mark_route_owned_plugin(output::create_crossover_plugin(
            "LR24", 80.0, "low",
        )));

        let mut direct = low_route("L", 0);
        direct.destination = "L".to_string();
        direct.destination_index = 0;
        direct.pre_chain_channel = Some("L".to_string());
        direct.post_chain_channel = Some("L".to_string());
        direct.route_kind = "main_highpass_to_self".to_string();
        direct.high_pass_hz = Some(80.0);
        direct.low_pass_hz = None;
        let mut redirected = low_route("L", 0);
        redirected.destination_index = 1;
        redirected.gain_db = -6.020599913279624;
        redirected.gain_linear = 0.5;
        redirected.matrix_gain = 0.5;
        let lfe = low_route("LFE", 1);
        let graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".to_string(),
            physical_sub_outputs: vec!["LFE".to_string()],
            stereo_routing: None,
            input_channels: vec!["L".to_string(), "LFE".to_string()],
            output_channels: vec!["L".to_string(), "LFE".to_string()],
            routes: vec![direct, redirected, lfe],
            matrix: None,
            input_trim_db: HashMap::from([("L".to_string(), -0.3)]),
            post_dsp_main_alignment_band_hz: Some([100.0, 400.0]),
            advisories: Vec::new(),
        };

        let sidecar_dir = std::path::Path::new(".");
        let main_base = realize_routed_main_training_branch(
            "L",
            &main_plugins,
            &main_input,
            0.0,
            "LR24",
            80.0,
            0.0,
            0.0,
            sample_rate,
            sidecar_dir,
            None,
        )
        .expect("main pre-route and direct output stages");
        let sub_base = realize_routed_sub_training_branch(
            "LFE",
            &[],
            &sub_input,
            0.0,
            &sub_output_eq,
            sample_rate,
            sidecar_dir,
            None,
        )
        .expect("common physical-sub output EQ");
        let source_pre_route_transfer = realize_source_pre_route_transfer(
            "L",
            main_plugins.clone(),
            &main_input,
            sample_rate,
            sidecar_dir,
            &HashMap::new(),
        )
        .expect("source transfer through finalizer trim");
        let training_before_source_post_eq =
            engine_bass_management::predict_deployed_source_curve_from_routes(
                Some(&main_base),
                &sub_base,
                Some(&source_pre_route_transfer),
                &graph,
                "L",
                sample_rate,
            )
            .expect("training curve through main and redirected output routes");
        let source_post_eq_response = response::compute_peq_complex_response(
            &source_post_eq,
            &training_before_source_post_eq.freq,
            sample_rate,
        );
        let training_curve = response::apply_complex_response(
            &training_before_source_post_eq,
            &source_post_eq_response,
        );

        let main_chain = chain("L", main_input.clone(), None);
        let mut main_chain = main_chain;
        main_chain.plugins = emitted_main_plugins;
        let sub_chain = chain("LFE", sub_input.clone(), None);
        let mut sub_chain = sub_chain;
        sub_chain.plugins = emitted_sub_plugins;
        let emitted_curve = super::reconstruct_deployed_source_curves_unenforced(
            &HashMap::from([
                ("L".to_string(), main_chain),
                ("LFE".to_string(), sub_chain),
            ]),
            &HashMap::new(),
            &graph,
            None,
            sample_rate,
            sidecar_dir,
        )
        .expect("reconstruct the emitted staged graph")["L"]
            .clone();

        assert!(
            training_curve
                .spl
                .iter()
                .zip(emitted_curve.spl.iter())
                .all(|(training, emitted)| (training - emitted).abs() < 1.0e-8),
            "training-vs-emitted SPL mismatch: training={:?}, emitted={:?}",
            training_curve.spl,
            emitted_curve.spl
        );
        let training_phase = training_curve.phase.as_ref().unwrap();
        let emitted_phase = emitted_curve.phase.as_ref().unwrap();
        assert!(training_phase.iter().zip(emitted_phase).all(|(a, b)| {
            let wrapped = (a - b + 180.0).rem_euclid(360.0) - 180.0;
            wrapped.abs() < 1.0e-8
        }));
    }

    #[test]
    fn routed_training_nonzero_phase_matches_independent_emitted_graph() {
        for common_gain in [0.0, 1.25] {
            let frequencies = ndarray::array![
                20.0, 31.0, 40.0, 60.0, 80.0, 100.0, 160.0, 250.0, 400.0, 800.0, 2_000.0
            ];
            let main_input = Curve {
                freq: frequencies.clone(),
                spl: ndarray::array![
                    61.0, 63.0, 66.0, 69.0, 72.0, 74.0, 76.0, 77.0, 78.0, 79.0, 80.0
                ],
                phase: Some(frequencies.mapv(|f| 17.0 - 360.0 * f * 0.0007)),
                ..Curve::default()
            };
            let sub_input = Curve {
                freq: frequencies.clone(),
                spl: ndarray::array![
                    78.0, 80.0, 82.0, 84.0, 82.0, 78.0, 70.0, 55.0, 30.0, 10.0, 0.0
                ],
                phase: Some(frequencies.mapv(|f| -31.0 - 360.0 * f * 0.0011)),
                ..Curve::default()
            };
            let sample_rate = 48_000.0;
            let main_output_eq = math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                110.0,
                sample_rate,
                1.1,
                2.5,
            );
            let sub_output_eq = [
                math_audio_iir_fir::Biquad::new(
                    math_audio_iir_fir::BiquadFilterType::Peak,
                    40.0,
                    sample_rate,
                    1.4,
                    -4.0,
                ),
                math_audio_iir_fir::Biquad::new(
                    math_audio_iir_fir::BiquadFilterType::Peak,
                    63.0,
                    sample_rate,
                    1.0,
                    2.0,
                ),
            ];
            let source_post_eq = [math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                48.0,
                sample_rate,
                0.9,
                common_gain,
            )];
            let mut main_plugins = vec![mark_plugin_stage(
                roomeq_engine::output::create_gain_plugin(-0.3),
                "pre_route",
            )];
            main_plugins.push(mark_plugin_stage(
                roomeq_engine::output::create_delay_plugin(0.9),
                "pre_route",
            ));
            main_plugins.push(mark_plugin_stage(
                roomeq_engine::output::create_eq_plugin(std::slice::from_ref(&main_output_eq)),
                "post_route",
            ));
            let mut emitted_main_plugins = main_plugins.clone();
            emitted_main_plugins.push(mark_plugin_stage(
                roomeq_engine::output::create_eq_plugin(&source_post_eq),
                "pre_route",
            ));
            emitted_main_plugins.push(mark_route_owned_plugin(output::create_delay_plugin(0.65)));
            emitted_main_plugins.push(mark_route_owned_plugin(output::create_crossover_plugin(
                "LR24", 80.0, "high",
            )));
            let mut emitted_sub_plugins = vec![mark_plugin_stage(
                roomeq_engine::output::create_eq_plugin(&sub_output_eq),
                "post_route",
            )];
            emitted_sub_plugins.push(mark_plugin_stage(
                output::create_delay_plugin(0.4),
                "post_route",
            ));
            emitted_sub_plugins.push(mark_route_owned_plugin(output::create_crossover_plugin(
                "LR24", 80.0, "low",
            )));

            let mut direct = low_route("L", 0);
            direct.destination = "L".to_string();
            direct.destination_index = 0;
            direct.pre_chain_channel = Some("L".to_string());
            direct.post_chain_channel = Some("L".to_string());
            direct.route_kind = "main_highpass_to_self".to_string();
            direct.high_pass_hz = Some(80.0);
            direct.low_pass_hz = None;
            direct.delay_ms = 0.65;
            let mut redirected = low_route("L", 0);
            redirected.destination_index = 1;
            redirected.gain_db = -6.020599913279624;
            redirected.gain_linear = 0.5;
            redirected.matrix_gain = 0.5;
            redirected.delay_ms = 1.7;
            redirected.polarity_inverted = true;
            let lfe = low_route("LFE", 1);
            let graph = BassManagementRoutingGraph {
                physical_sub_output: "LFE".to_string(),
                physical_sub_outputs: vec!["LFE".to_string()],
                stereo_routing: None,
                input_channels: vec!["L".to_string(), "LFE".to_string()],
                output_channels: vec!["L".to_string(), "LFE".to_string()],
                routes: vec![direct, redirected, lfe],
                matrix: None,
                input_trim_db: HashMap::from([("L".to_string(), -0.3)]),
                post_dsp_main_alignment_band_hz: Some([100.0, 400.0]),
                advisories: Vec::new(),
            };

            let sidecar_dir = std::path::Path::new(".");
            let main_base = realize_routed_main_training_branch(
                "L",
                &main_plugins,
                &main_input,
                0.0,
                "LR24",
                80.0,
                0.0,
                0.65,
                sample_rate,
                sidecar_dir,
                None,
            )
            .expect("main pre-route and direct output stages");
            let sub_base = realize_routed_sub_training_branch(
                "LFE",
                &[mark_plugin_stage(
                    output::create_delay_plugin(0.4),
                    "post_route",
                )],
                &sub_input,
                0.0,
                &sub_output_eq,
                sample_rate,
                sidecar_dir,
                None,
            )
            .expect("common physical-sub output EQ");
            let source_pre_route_transfer = realize_source_pre_route_transfer(
                "L",
                main_plugins.clone(),
                &main_input,
                sample_rate,
                sidecar_dir,
                &HashMap::new(),
            )
            .expect("source transfer through finalizer trim");
            let training_before_source_post_eq =
                engine_bass_management::predict_deployed_source_curve_from_routes(
                    Some(&main_base),
                    &sub_base,
                    Some(&source_pre_route_transfer),
                    &graph,
                    "L",
                    sample_rate,
                )
                .expect("training curve through main and redirected output routes");
            let source_post_eq_response = response::compute_peq_complex_response(
                &source_post_eq,
                &training_before_source_post_eq.freq,
                sample_rate,
            );
            let training_curve = response::apply_complex_response(
                &training_before_source_post_eq,
                &source_post_eq_response,
            );

            let main_chain = chain("L", main_input.clone(), None);
            let mut main_chain = main_chain;
            main_chain.plugins = emitted_main_plugins;
            let sub_chain = chain("LFE", sub_input.clone(), None);
            let mut sub_chain = sub_chain;
            sub_chain.plugins = emitted_sub_plugins;
            let emitted_curve = super::reconstruct_deployed_source_curves_unenforced(
                &HashMap::from([
                    ("L".to_string(), main_chain),
                    ("LFE".to_string(), sub_chain),
                ]),
                &HashMap::new(),
                &graph,
                None,
                sample_rate,
                sidecar_dir,
            )
            .expect("reconstruct the emitted staged graph")["L"]
                .clone();

            assert!(
                training_curve
                    .spl
                    .iter()
                    .zip(emitted_curve.spl.iter())
                    .all(|(training, emitted)| (training - emitted).abs() < 1.0e-8),
                "training-vs-emitted SPL mismatch: training={:?}, emitted={:?}",
                training_curve.spl,
                emitted_curve.spl
            );
            let training_phase = training_curve.phase.as_ref().unwrap();
            let emitted_phase = emitted_curve.phase.as_ref().unwrap();
            assert!(training_phase.iter().zip(emitted_phase).all(|(a, b)| {
                let wrapped = (a - b + 180.0).rem_euclid(360.0) - 180.0;
                wrapped.abs() < 1.0e-8
            }));
            // Independent cookbook coefficients and complex pressure sum. No
            // production crossover, PEQ, route, or curve replay helper is used.
            use num_complex::Complex64;
            let delay = |f: f64, milliseconds: f64| {
                Complex64::from_polar(
                    1.0,
                    -2.0 * std::f64::consts::PI * f * milliseconds / 1_000.0,
                )
            };
            let response = |f: f64, center: f64, q: f64, gain: Option<f64>, high: bool| {
                let omega = 2.0 * std::f64::consts::PI * center / sample_rate;
                let cosine = omega.cos();
                let alpha = omega.sin() / (2.0 * q);
                let (b, a) = if let Some(gain) = gain {
                    let amplitude = 10.0_f64.powf(gain / 40.0);
                    (
                        [
                            1.0 + alpha * amplitude,
                            -2.0 * cosine,
                            1.0 - alpha * amplitude,
                        ],
                        [
                            1.0 + alpha / amplitude,
                            -2.0 * cosine,
                            1.0 - alpha / amplitude,
                        ],
                    )
                } else if high {
                    (
                        [(1.0 + cosine) / 2.0, -(1.0 + cosine), (1.0 + cosine) / 2.0],
                        [1.0 + alpha, -2.0 * cosine, 1.0 - alpha],
                    )
                } else {
                    (
                        [(1.0 - cosine) / 2.0, 1.0 - cosine, (1.0 - cosine) / 2.0],
                        [1.0 + alpha, -2.0 * cosine, 1.0 - alpha],
                    )
                };
                let z = Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * f / sample_rate);
                (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
            };
            let pressure = |curve: &Curve, index: usize| {
                Complex64::from_polar(
                    10.0_f64.powf(curve.spl[index] / 20.0),
                    curve.phase.as_ref().unwrap()[index].to_radians(),
                )
            };
            for (index, &frequency) in frequencies.iter().enumerate() {
                let high =
                    response(frequency, 80.0, std::f64::consts::FRAC_1_SQRT_2, None, true).powu(2);
                let low = response(
                    frequency,
                    80.0,
                    std::f64::consts::FRAC_1_SQRT_2,
                    None,
                    false,
                )
                .powu(2);
                let fixed_source = 10.0_f64.powf(-0.3 / 20.0) * delay(frequency, 0.9);
                let main = pressure(&main_input, index)
                    * high
                    * delay(frequency, 0.65)
                    * response(frequency, 110.0, 1.1, Some(2.5), false);
                let sub = pressure(&sub_input, index)
                    * low
                    * -0.5
                    * delay(frequency, 1.7 + 0.4)
                    * response(frequency, 40.0, 1.4, Some(-4.0), false)
                    * response(frequency, 63.0, 1.0, Some(2.0), false);
                let fixed_baseline = fixed_source * (main + sub);
                assert!(
                    fixed_baseline.norm() > 1.0e-6,
                    "fixture must not divide by near cancellation"
                );
                let common = response(frequency, 48.0, 0.9, Some(common_gain), false);
                let expected = fixed_baseline * common;
                let emitted = pressure(&emitted_curve, index);
                let trained = pressure(&training_curve, index);
                assert!(
                    (emitted - expected).norm() / expected.norm() < 1.0e-8,
                    "emitted graph differs from independent sum at {frequency} Hz"
                );
                assert!(
                    (trained - expected).norm() / expected.norm() < 1.0e-8,
                    "training graph differs from independent sum at {frequency} Hz"
                );
                let correction = emitted / fixed_baseline;
                assert!((correction - common).norm() < 1.0e-8);
                if common_gain == 0.0 {
                    assert!(
                        (correction - Complex64::new(1.0, 0.0)).norm() < 1.0e-8,
                        "identity correction must preserve the same fixed routed graph"
                    );
                }
            }
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
            physical_sub_outputs: Vec::new(),
            stereo_routing: None,
            input_channels: vec!["LFE".to_string()],
            output_channels: vec!["LFE".to_string()],
            routes: vec![low_route("LFE", 0)],
            matrix: None,
            input_trim_db: HashMap::new(),
            post_dsp_main_alignment_band_hz: None,
            advisories: Vec::new(),
        };
        let mut channels = HashMap::from([("LFE".to_string(), lfe)]);
        let before = reconstruct_deployed_source_curves(
            &channels,
            &HashMap::new(),
            &graph,
            None,
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
            None,
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
            physical_sub_outputs: Vec::new(),
            stereo_routing: None,
            input_channels: vec!["L".to_string(), "LFE".to_string()],
            output_channels: vec!["L".to_string(), "LFE".to_string()],
            routes: vec![low_route("L", 0)],
            matrix: None,
            input_trim_db: HashMap::new(),
            post_dsp_main_alignment_band_hz: None,
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
            None,
            48_000.0,
            std::path::Path::new("."),
        )
        .unwrap();
        channels.get_mut("L").unwrap().final_curve = Some(CurveData::from(&curve(20.0)));
        let after = reconstruct_deployed_source_curves(
            &channels,
            &HashMap::new(),
            &graph,
            None,
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
            physical_sub_outputs: Vec::new(),
            stereo_routing: None,
            input_channels: vec!["L".to_string(), "LFE".to_string()],
            output_channels: vec!["L".to_string(), "LFE".to_string()],
            routes: vec![low_route("L", 0)],
            matrix: None,
            input_trim_db: HashMap::new(),
            post_dsp_main_alignment_band_hz: None,
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
            None,
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

    fn cancelling_setup() -> (HashMap<String, ChannelDspChain>, BassManagementRoutingGraph) {
        let initial = curve(60.0);
        let graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".to_string(),
            physical_sub_outputs: Vec::new(),
            stereo_routing: None,
            input_channels: vec!["L".to_string(), "LFE".to_string()],
            output_channels: vec!["L".to_string(), "LFE".to_string()],
            routes: vec![low_route("L", 0)],
            matrix: None,
            input_trim_db: HashMap::new(),
            post_dsp_main_alignment_band_hz: None,
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
        (channels, graph)
    }

    fn accepted_source(accepted: bool) -> roomeq_model::BassManagementSourceReport {
        accepted_source_with_advisory(accepted, "source_route_de_optimized")
    }

    #[test]
    fn post_eq_screening_matches_serialized_graph_with_different_sub_grid() {
        let config = roomeq_model::RoomConfig::default();
        let (_, mut graph) = cancelling_setup();
        graph.physical_sub_outputs = vec!["LFE".to_string()];
        let full_sub = curve(60.0);
        let mut main = super::engine_bass_management::predict_bass_source_curve_from_routes(
            &full_sub, None, &graph, "L", 48_000.0,
        )
        .unwrap();
        main.spl.mapv_inplace(|level| level + 12.0);
        main.phase
            .as_mut()
            .unwrap()
            .mapv_inplace(|phase| phase + 180.0);
        let sub = Curve {
            freq: ndarray::array![20.0, 80.0, 400.0],
            spl: ndarray::Array1::from_elem(3, 60.0),
            phase: Some(ndarray::Array1::zeros(3)),
            ..Curve::default()
        };
        for (boost, accepted) in [(0.0, true), (6.0, false)] {
            let mut corrected_sub = sub.clone();
            corrected_sub.spl.mapv_inplace(|level| level + boost);
            let bass = super::engine_bass_management::predict_bass_source_curve_from_routes(
                &corrected_sub,
                None,
                &graph,
                "L",
                48_000.0,
            )
            .unwrap();
            let screening = super::post_eq_crossover_cancellation(&config, "L", &main, &bass, 80.0)
                .expect("mismatched grids must not erase screening evidence");
            assert_eq!(screening.accepted, accepted);
            let mut sub_chain = chain("LFE", sub.clone(), None);
            sub_chain.plugins.push(mark_plugin_stage(
                roomeq_engine::output::create_gain_plugin(boost),
                "post_route",
            ));
            let channels = HashMap::from([
                ("L".to_string(), chain("L", main.clone(), None)),
                ("LFE".to_string(), sub_chain),
            ]);
            let serialized = serde_json::to_string(&(channels, &graph)).unwrap();
            let (channels, graph): (HashMap<String, ChannelDspChain>, BassManagementRoutingGraph) =
                serde_json::from_str(&serialized).unwrap();
            let replay = reconstruct_deployed_source_curves(
                &channels,
                &HashMap::new(),
                &graph,
                None,
                48_000.0,
                std::path::Path::new("."),
            );
            assert_eq!(replay.is_ok(), accepted, "serialized playback: {replay:?}");
        }
    }

    #[test]
    fn final_cancellation_replay_accepts_ten_to_four_and_keeps_original_baseline() {
        let (_, graph) = cancelling_setup();
        let raw_sub = curve(80.0);
        let bass = super::engine_bass_management::predict_bass_source_curve_from_routes(
            &raw_sub, None, &graph, "L", 48000.0,
        )
        .unwrap();
        let main_with_deficit = |deficit: f64| {
            let mut main = bass.clone();
            let delta = (10_f64.powf(-deficit / 20.0) / 2.0).acos().to_degrees() * 2.0;
            main.phase.as_mut().unwrap().mapv_inplace(|p| p + delta);
            main
        };
        let baseline =
            roomeq_engine::topology::cancellation_baseline(&main_with_deficit(10.0), &bass, 80.0)
                .unwrap();
        let mut optimization = super::joint_bass_management_report_from_parts(&[], &[], &[]);
        optimization.crossover_cancellation = Some(roomeq_model::CrossoverCancellationContext {
            limit_db: 3.0,
            sources: [("L".into(), baseline)].into(),
        });
        for deficit in [4.0, 4.0, 10.0, 11.0] {
            let channels = HashMap::from([
                ("L".into(), chain("L", main_with_deficit(deficit), None)),
                ("LFE".into(), chain("LFE", raw_sub.clone(), None)),
            ]);
            let result = super::reconstruct_deployed_source_curves_with_evidence(
                &channels,
                &HashMap::new(),
                &graph,
                Some(&optimization),
                48000.0,
                std::path::Path::new("."),
            );
            if deficit == 4.0 {
                let (_, evidence) = result.unwrap();
                assert_eq!(evidence.len(), 1);
                assert!((evidence[0].baseline_db.unwrap() - 10.0).abs() < 1e-6);
                assert!((evidence[0].final_db - 4.0).abs() < 1e-6);
                assert_eq!(evidence[0].reason, "improved_residual_cancellation");
            } else {
                assert!(result.is_err());
            }
        }
    }

    #[test]
    fn configured_cancellation_baseline_is_captured_before_eq_and_array_optimization() {
        use roomeq_model::*;
        let mut config = RoomConfig::default();
        config.optimizer.max_crossover_cancellation_db = 5.0;
        config.speakers.insert(
            "sub".into(),
            SpeakerConfig::Single(MeasurementSource::InMemory(curve(80.0))),
        );
        config.system = Some(SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([("L".into(), "left".into())]),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                routing: Default::default(),
                outputs: vec![SubwooferOutput {
                    id: "Sub1".into(),
                    speaker: "sub".into(),
                }],
                crossover: Some("bass".into()),
            }),
            ..Default::default()
        });
        config.crossovers = Some(HashMap::from([(
            "bass".into(),
            CrossoverConfig {
                crossover_type: "LR24".into(),
                frequency: Some(80.0),
                frequencies: None,
                frequency_range: None,
            },
        )]));
        let mut main = curve(80.0);
        main.phase.as_mut().unwrap().fill(180.0);
        config.speakers.insert(
            "left".into(),
            SpeakerConfig::Single(MeasurementSource::InMemory(main.clone())),
        );
        let mains = HashMap::from([("L".into(), main)]);
        let baseline =
            super::capture_crossover_cancellation_baseline(&config, &mains, 48000.0, 256).unwrap();
        assert_eq!(baseline.limit_db, 5.0);
        assert!(baseline.sources.contains_key("L"));
        let frozen = baseline.sources["L"].cancellation_db.clone();
        // A spatial power average has no phase. The baseline must instead
        // select the same primary seat for every physical source.
        let mut multiseat = config.clone();
        multiseat.optimizer.multi_seat = Some(MultiSeatConfig {
            primary_seat: 1,
            ..Default::default()
        });
        multiseat.speakers.insert(
            "left".into(),
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![
                curve(40.0),
                mains["L"].clone(),
            ])),
        );
        multiseat.speakers.insert(
            "sub".into(),
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![
                curve(60.0),
                curve(80.0),
            ])),
        );
        let mut averaged_main = curve(70.0);
        averaged_main.phase = None;
        let averaged = HashMap::from([("L".into(), averaged_main)]);
        let selected =
            super::capture_crossover_cancellation_baseline(&multiseat, &averaged, 48000.0, 256)
                .unwrap();
        assert_eq!(
            selected
                .sources
                .get("L")
                .map(|source| &source.cancellation_db),
            Some(&frozen),
            "primary-seat baseline must equal the same-seat single-capture reference"
        );
        multiseat.speakers.insert(
            "sub".into(),
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![
                curve(110.0),
                curve(80.0),
            ])),
        );
        multiseat.speakers.insert(
            "left".into(),
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![
                curve(120.0),
                mains["L"].clone(),
            ])),
        );
        let changed_other_seat =
            super::capture_crossover_cancellation_baseline(&multiseat, &averaged, 48000.0, 256)
                .unwrap();
        assert_eq!(changed_other_seat.sources["L"].cancellation_db, frozen);
        multiseat
            .optimizer
            .multi_seat
            .as_mut()
            .unwrap()
            .primary_seat = 2;
        assert!(
            super::capture_crossover_cancellation_baseline(&multiseat, &averaged, 48000.0, 256,)
                .is_err(),
            "an unavailable primary seat must not fall back to another seat"
        );
        multiseat
            .optimizer
            .multi_seat
            .as_mut()
            .unwrap()
            .primary_seat = 1;
        let mut magnitude_only = curve(80.0);
        magnitude_only.phase = None;
        multiseat.speakers.insert(
            "sub".into(),
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![
                curve(80.0),
                magnitude_only,
            ])),
        );
        assert!(
            super::capture_crossover_cancellation_baseline(&multiseat, &averaged, 48000.0, 256,)
                .unwrap()
                .sources
                .is_empty(),
            "phase at an unselected seat cannot justify a coherent baseline"
        );
        config.system.as_mut().unwrap().bass_management = Some(BassManagementConfig {
            sub_trim_db: -6.0,
            ..Default::default()
        });
        let trimmed =
            super::capture_crossover_cancellation_baseline(&config, &mains, 48000.0, 256).unwrap();
        assert_ne!(
            trimmed.sources["L"].cancellation_db, frozen,
            "configured trim must be preserved in the baseline"
        );
        config.optimizer.crossover_cancellation_baseline = Some(baseline);
        config.speakers.insert(
            "sub".into(),
            SpeakerConfig::Single(MeasurementSource::InMemory(curve(20.0))),
        );
        assert_eq!(
            config
                .optimizer
                .crossover_cancellation_baseline
                .as_ref()
                .unwrap()
                .sources["L"]
                .cancellation_db,
            frozen
        );
    }

    fn accepted_source_with_advisory(
        accepted: bool,
        advisory: &str,
    ) -> roomeq_model::BassManagementSourceReport {
        roomeq_model::BassManagementSourceReport {
            source_channel: "L".to_string(),
            group_id: "lcr".to_string(),
            main_delay_ms: 0.0,
            bass_route_delay_ms: 0.0,
            polarity_inverted: false,
            trim_db: 0.0,
            objective_before: None,
            objective_after: None,
            accepted,
            safety_restored: false,
            advisories: if accepted {
                vec![advisory.to_string()]
            } else {
                Vec::new()
            },
        }
    }

    fn baseline_report(
        sources: &[roomeq_model::BassManagementSourceReport],
    ) -> roomeq_model::BassManagementOptimizationReport {
        let mut report = super::joint_bass_management_report_from_parts(&[], sources, &[]);
        report.crossover_cancellation = Some(roomeq_model::CrossoverCancellationContext {
            limit_db: 3.0,
            sources: sources
                .iter()
                .filter(|s| s.accepted)
                .map(|s| {
                    (
                        s.source_channel.clone(),
                        roomeq_model::CrossoverCancellationBaseline {
                            crossover_hz: 80.0,
                            frequencies_hz: vec![20.0, 20000.0],
                            cancellation_db: vec![30.0, 30.0],
                        },
                    )
                })
                .collect(),
        });
        report
    }

    #[test]
    fn splice_replay_ships_best_effort_for_accepted_residual_after_stages_exhausted() {
        // Both the clean acceptance and the still-excessive improvement record
        // a reviewed tradeoff; either must ship best-effort, not fail the run.
        for advisory in [
            "source_route_de_optimized",
            "source_route_de_optimized_pending_underfill_correction:4.791db",
        ] {
            let (mut channels, graph) = cancelling_setup();
            // No FIR and no post-route correction EQ: nothing revertible remains.
            let optimization = baseline_report(&[accepted_source_with_advisory(true, advisory)]);
            let mut reverted = Vec::new();
            let deployed = super::replay_until_splice_safe(
                &mut channels,
                &mut HashMap::new(),
                &mut reverted,
                &HashMap::new(),
                &graph,
                Some(&optimization),
                48_000.0,
                std::path::Path::new("."),
                &|_| 80.0,
                "LFE",
                20.0,
                130.0,
                16_000.0,
            )
            .expect("accepted residual must ship best-effort, not fail the run");
            assert!(deployed.contains_key("L"));
            assert_eq!(
                reverted,
                vec!["L:improved_residual_cancellation".to_string()]
            );
        }
    }

    #[test]
    fn crossover_timing_refused_detects_explicit_refusal_only() {
        assert!(!super::crossover_timing_refused(None));
        let mut report = baseline_report(&[accepted_source(true)]);
        assert!(!super::crossover_timing_refused(Some(&report)));
        report.advisories.push(format!(
            "{}source 'L' lacks stationary timing evidence",
            super::CROSSOVER_TIMING_REFUSED_ADVISORY_PREFIX
        ));
        assert!(super::crossover_timing_refused(Some(&report)));
    }

    #[test]
    fn splice_replay_skips_stripping_when_timing_refused() {
        // A refused timing reference leaves main/sub phase uncalibrated, so a
        // coherent splice verdict is arbitrary: ship the curves instead of
        // stripping correction stages on luck.
        let (mut channels, graph) = cancelling_setup();
        let mut optimization = baseline_report(&[accepted_source(true)]);
        optimization.advisories.push(format!(
            "{}source 'L' lacks stationary timing evidence",
            super::CROSSOVER_TIMING_REFUSED_ADVISORY_PREFIX
        ));
        let mut reverted = Vec::new();
        let deployed = super::replay_until_splice_safe(
            &mut channels,
            &mut HashMap::new(),
            &mut reverted,
            &HashMap::new(),
            &graph,
            Some(&optimization),
            48_000.0,
            std::path::Path::new("."),
            &|_| 80.0,
            "LFE",
            20.0,
            130.0,
            16_000.0,
        )
        .expect("refused timing must not strip corrections");
        assert!(deployed.contains_key("L"));
        assert!(reverted.is_empty());
    }

    #[test]
    fn splice_replay_keeps_hard_error_without_accepted_route_evidence() {
        let (channels, graph) = cancelling_setup();
        for optimization in [
            None,
            Some(baseline_report(&[accepted_source(false)])),
            Some(super::joint_bass_management_report_from_parts(
                &[],
                &[accepted_source(true)],
                &[],
            )),
        ] {
            let mut reverted = Vec::new();
            let error = super::replay_until_splice_safe(
                &mut channels.clone(),
                &mut HashMap::new(),
                &mut reverted,
                &HashMap::new(),
                &graph,
                optimization.as_ref(),
                48_000.0,
                std::path::Path::new("."),
                &|_| 80.0,
                "LFE",
                20.0,
                130.0,
                16_000.0,
            )
            .expect_err("unaccepted splice must stay a hard error");
            assert!(
                error
                    .to_string()
                    .contains("final routed crossover underfill")
            );
            assert!(reverted.is_empty());
        }
    }

    fn two_cancelling_sources() -> (HashMap<String, ChannelDspChain>, BassManagementRoutingGraph) {
        let (mut channels, mut graph) = cancelling_setup();
        let mut right = channels["L"].clone();
        right.channel = "R".into();
        channels.insert("R".into(), right);
        graph.input_channels.insert(1, "R".into());
        graph.output_channels.insert(1, "R".into());
        graph.routes.push(low_route("R", 1));
        (channels, graph)
    }

    #[test]
    fn best_effort_snapshot_checks_every_unaccepted_source() {
        let (channels, graph) = two_cancelling_sources();
        let optimization = baseline_report(&[accepted_source(true)]);
        let mut degraded = Vec::new();
        let error = super::reconstruct_deployed_snapshot_best_effort(
            &channels,
            &HashMap::new(),
            &graph,
            Some(&optimization),
            48_000.0,
            std::path::Path::new("."),
            &mut degraded,
        )
        .expect_err("the accepted L residual must not exempt unaccepted R");
        assert_eq!(
            super::underfill_error_role(&error.to_string()).as_deref(),
            Some("R")
        );
        assert!(
            degraded.is_empty(),
            "failed snapshot must not publish partial acceptance"
        );
    }

    #[test]
    fn splice_replay_checks_every_unaccepted_source() {
        let (mut channels, graph) = two_cancelling_sources();
        let optimization = baseline_report(&[accepted_source(true)]);
        let mut reverted = Vec::new();
        let error = super::replay_until_splice_safe(
            &mut channels,
            &mut HashMap::new(),
            &mut reverted,
            &HashMap::new(),
            &graph,
            Some(&optimization),
            48_000.0,
            std::path::Path::new("."),
            &|_| 80.0,
            "LFE",
            20.0,
            130.0,
            16_000.0,
        )
        .expect_err("the accepted L residual must not skip R's replay");
        assert_eq!(
            super::underfill_error_role(&error.to_string()).as_deref(),
            Some("R")
        );
        assert!(
            reverted.is_empty(),
            "no correction was reverted and replay failed"
        );
    }

    #[test]
    fn best_effort_records_each_accepted_residual_source() {
        let (mut channels, graph) = two_cancelling_sources();
        let left = accepted_source(true);
        let mut right = left.clone();
        right.source_channel = "R".into();
        let optimization = baseline_report(&[left, right]);
        let expected = vec![
            "L:improved_residual_cancellation".to_string(),
            "R:improved_residual_cancellation".to_string(),
        ];
        let mut degraded = Vec::new();
        let snapshot = super::reconstruct_deployed_snapshot_best_effort(
            &channels,
            &HashMap::new(),
            &graph,
            Some(&optimization),
            48_000.0,
            std::path::Path::new("."),
            &mut degraded,
        )
        .unwrap();
        assert_eq!(degraded, expected);
        let mut reverted = Vec::new();
        let deployed = super::replay_until_splice_safe(
            &mut channels,
            &mut HashMap::new(),
            &mut reverted,
            &HashMap::new(),
            &graph,
            Some(&optimization),
            48_000.0,
            std::path::Path::new("."),
            &|_| 80.0,
            "LFE",
            20.0,
            130.0,
            16_000.0,
        )
        .unwrap();
        assert_eq!(reverted, expected);
        assert_eq!(deployed["R"].spl, snapshot["R"].spl);
        assert!(
            reconstruct_deployed_source_curves(
                &channels,
                &HashMap::new(),
                &graph,
                Some(&optimization),
                48_000.0,
                std::path::Path::new("."),
            )
            .is_ok(),
            "strict replay must use the same numeric baseline policy"
        );
    }

    #[test]
    fn splice_replay_reverts_later_source_after_accepted_residual() {
        let (mut channels, graph) = two_cancelling_sources();
        let bass = super::engine_bass_management::predict_bass_source_curve_from_routes(
            &curve(60.0),
            None,
            &graph,
            "R",
            48_000.0,
        )
        .unwrap();
        let mut right = chain("R", bass, None);
        right.plugins.push(mark_plugin_stage(
            roomeq_engine::output::create_labeled_eq_plugin(
                &[super::Biquad::new(
                    super::BiquadFilterType::AllPass,
                    80.0,
                    48_000.0,
                    1.0,
                    0.0,
                )],
                "post_eq",
            ),
            "post_route",
        ));
        channels.insert("R".into(), right);
        let optimization = baseline_report(&[accepted_source(true)]);
        let mut reverted = Vec::new();
        let deployed = super::replay_until_splice_safe(
            &mut channels,
            &mut HashMap::new(),
            &mut reverted,
            &HashMap::new(),
            &graph,
            Some(&optimization),
            48_000.0,
            std::path::Path::new("."),
            &|_| 80.0,
            "LFE",
            20.0,
            130.0,
            16_000.0,
        )
        .unwrap();
        assert!(
            channels["R"].plugins.is_empty(),
            "R's phase-breaking EQ was not examined"
        );
        assert!(reverted.contains(&"R:peq".to_string()));
        assert!(reverted.contains(&"L:improved_residual_cancellation".to_string()));
        assert!(!reverted.contains(&"R:improved_residual_cancellation".to_string()));
        assert!(deployed["R"].spl.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn deployed_source_refresh_keeps_safe_route_with_large_target_residual() {
        let initial = curve(60.0);
        let mut main = chain("L", initial.clone(), Some(initial.clone()));
        main.target_curve = Some(CurveData::from(&curve(90.0)));
        let graph = BassManagementRoutingGraph {
            physical_sub_output: "LFE".to_string(),
            physical_sub_outputs: Vec::new(),
            stereo_routing: None,
            input_channels: vec!["L".to_string(), "LFE".to_string()],
            output_channels: vec!["L".to_string(), "LFE".to_string()],
            routes: vec![low_route("L", 0)],
            matrix: None,
            input_trim_db: HashMap::new(),
            post_dsp_main_alignment_band_hz: None,
            advisories: Vec::new(),
        };
        let channels = HashMap::from([
            ("L".to_string(), main),
            (
                "LFE".to_string(),
                chain("LFE", curve(20.0), Some(curve(20.0))),
            ),
        ]);

        let deployed = reconstruct_deployed_source_curves(
            &channels,
            &HashMap::new(),
            &graph,
            None,
            48_000.0,
            std::path::Path::new("."),
        )
        .expect("target residual must not suppress a cancellation-safe routed artifact");
        assert!(deployed.contains_key("L"));
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
            physical_sub_outputs: Vec::new(),
            stereo_routing: None,
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
            post_dsp_main_alignment_band_hz: None,
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
            None,
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

    #[test]
    fn untagged_legacy_eq_stage_is_refused_by_deployed_reconstruction() {
        use math_audio_iir_fir::{Biquad, BiquadFilterType};

        let (mut result, _config) = recalibration_fixture(10.0);
        let peak = Biquad::new(BiquadFilterType::Peak, 200.0, 48_000.0, 0.8, 6.0);
        result.channels.get_mut("L").unwrap().plugins.push(
            roomeq_engine::output::create_eq_plugin(std::slice::from_ref(&peak)),
        );
        let graph = result
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .routing_graph
            .as_ref()
            .unwrap();
        let error = super::reconstruct_deployed_source_curves_unenforced(
            &result.channels,
            &HashMap::new(),
            graph,
            None,
            48_000.0,
            std::path::Path::new("."),
        )
        .expect_err("the serialized physical routing contract requires EQ ownership");
        assert!(
            error.to_string().contains("unresolved plugin ownership"),
            "untagged EQ must fail closed instead of being silently assigned to a stage: {error}"
        );
    }

    #[test]
    fn zero_strength_recalibration_removes_obsolete_gains_and_preserves_configuration() {
        use math_audio_iir_fir::{Biquad, BiquadFilterType};

        let (mut result, config) = recalibration_fixture(10.0);
        let peak = Biquad::new(BiquadFilterType::Peak, 200.0, 48_000.0, 0.8, 6.0);
        result
            .channels
            .get_mut("L")
            .unwrap()
            .plugins
            .push(mark_plugin_stage(
                roomeq_engine::output::create_eq_plugin(std::slice::from_ref(&peak)),
                "post_route",
            ));
        result.channel_results.get_mut("L").unwrap().biquads = vec![peak];
        seed_post_dsp_calibration(&mut result, &config);
        let seeded_graph = result
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .routing_graph
            .as_ref()
            .unwrap();
        assert!(
            seeded_graph
                .advisories
                .iter()
                .any(|advisory| { advisory.starts_with("common_input_headroom_safety_trim_db:") })
        );
        assert!(result.channels["L"].plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("post_dsp_output_headroom_safety")
        }));

        let graph = result
            .metadata
            .bass_management
            .as_mut()
            .unwrap()
            .routing_graph
            .as_mut()
            .unwrap();
        set_redirected_route_gain(graph, 0.0);
        let routes_after_trial_rollback = graph.routes.clone();
        result
            .channels
            .get_mut("L")
            .unwrap()
            .plugins
            .retain(|plugin| plugin.plugin_type != "eq");
        result.channel_results.get_mut("L").unwrap().biquads.clear();
        let untouched_original = result.clone();

        assert!(
            super::recalibrate_post_dsp_levels(
                &mut result,
                &config,
                48_000.0,
                std::path::Path::new(".")
            )
            .unwrap()
        );

        let original_l_trim = untouched_original
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .routing_graph
            .as_ref()
            .unwrap()
            .input_trim_db["L"];
        let fresh_graph = result
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .routing_graph
            .as_ref()
            .unwrap();
        assert!(
            original_l_trim < -5.0,
            "expected the boosted trial to retain its old trim, got {original_l_trim}"
        );
        assert!((fresh_graph.input_trim_db["L"] + 2.0).abs() < 1.0e-5);
        assert!(fresh_graph.input_trim_db["R"].abs() < 1.0e-12);
        assert!(fresh_graph.input_trim_db["Sub1"].abs() < 1.0e-12);
        assert_eq!(
            serde_json::to_value(&fresh_graph.routes).unwrap(),
            serde_json::to_value(&routes_after_trial_rollback).unwrap()
        );
        assert_eq!(
            fresh_graph.post_dsp_main_alignment_band_hz,
            Some([100.0, 400.0])
        );
        assert!(
            fresh_graph
                .advisories
                .contains(&"post_dsp_input_levels_aligned_down".to_string())
        );
        assert!(
            !fresh_graph
                .advisories
                .iter()
                .any(|advisory| { advisory.starts_with("common_input_headroom_safety_trim_db:") })
        );
        assert!(result.channels["L"].plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("speaker_calibration")
        }));
        assert!(result.channels["Sub1"].plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("configured_sub_gain")
        }));
        assert!(!result.channels["Sub1"].plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("post_dsp_input_level_alignment")
        }));
        assert!(result.channels["L"].plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("post_dsp_input_level_alignment")
        }));
        assert!(
            untouched_original.channels["L"]
                .plugins
                .iter()
                .any(|plugin| {
                    plugin
                        .parameters
                        .get("label")
                        .and_then(serde_json::Value::as_str)
                        == Some("post_dsp_output_headroom_safety")
                }),
            "the independent original candidate must remain unchanged"
        );
    }

    #[test]
    fn recalibration_recomputes_required_output_safety_and_is_idempotent() {
        let (mut result, config) = recalibration_fixture(0.0);
        seed_post_dsp_calibration(&mut result, &config);
        let graph = result
            .metadata
            .bass_management
            .as_mut()
            .unwrap()
            .routing_graph
            .as_mut()
            .unwrap();
        set_redirected_route_gain(graph, 10.0);

        assert!(
            super::recalibrate_post_dsp_levels(
                &mut result,
                &config,
                48_000.0,
                std::path::Path::new(".")
            )
            .unwrap()
        );
        let first_safety: HashMap<_, _> = result
            .channels
            .iter()
            .filter_map(|(role, chain)| {
                chain.plugins.iter().find_map(|plugin| {
                    (plugin
                        .parameters
                        .get("label")
                        .and_then(serde_json::Value::as_str)
                        == Some("post_dsp_output_headroom_safety"))
                    .then(|| (role.clone(), plugin.parameters["gain_db"].as_f64().unwrap()))
                })
            })
            .collect();
        assert!(first_safety.contains_key("L"));
        assert!(first_safety.contains_key("R"));
        assert!(first_safety.contains_key("Sub1"));
        assert!(first_safety.values().all(|gain_db| *gain_db < 0.0));
        let first_graph = result
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .routing_graph
            .as_ref()
            .unwrap();
        assert!(first_graph.input_trim_db["Sub1"].abs() < 1.0e-12);
        let pre_protection_simulation = result
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .headroom_simulation
            .as_ref()
            .unwrap();
        assert!(
            pre_protection_simulation.margin_db < 0.0,
            "this report is the correlated-bus margin before the separate post-route safety gains"
        );

        assert!(
            super::recalibrate_post_dsp_levels(
                &mut result,
                &config,
                48_000.0,
                std::path::Path::new(".")
            )
            .unwrap()
        );
        let second_safety: HashMap<_, _> = result
            .channels
            .iter()
            .filter_map(|(role, chain)| {
                chain.plugins.iter().find_map(|plugin| {
                    (plugin
                        .parameters
                        .get("label")
                        .and_then(serde_json::Value::as_str)
                        == Some("post_dsp_output_headroom_safety"))
                    .then(|| (role.clone(), plugin.parameters["gain_db"].as_f64().unwrap()))
                })
            })
            .collect();
        assert_eq!(
            first_safety, second_safety,
            "recalibration must not stack gains"
        );
        assert!(result.channels["L"].plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("speaker_calibration")
        }));
        assert!(result.channels["Sub1"].plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("configured_sub_gain")
        }));
    }

    #[test]
    fn structural_gain_fallback_retains_native_refusal_without_acceptance() {
        let (mut result, config) = recalibration_fixture(10.0);
        let directory = tempfile::tempdir().unwrap();
        crate::room_optimization::rebuild_routed_pruning_test_candidate(
            &mut result,
            &config,
            &HashMap::new(),
            48_000.0,
            directory.path(),
        )
        .expect("structural fallback must follow the production publication policy");
        let report = result.metadata.correction_acceptance.as_ref().unwrap();
        assert!(!report.accepted);
        assert!(matches!(
            report.decision,
            roomeq_model::CorrectionDecision::IdentityFallback
                | roomeq_model::CorrectionDecision::Rejected
        ));
        assert!(
            report
                .violations
                .iter()
                .any(|reason| { reason.contains("max_boost_limit_exceeded") })
        );
        assert!(result.metadata.stage_outcomes.iter().any(|stage| {
            stage.stage == "final_correction_selection"
                && stage.advisories.iter().any(|reason| {
                    reason.contains("baseline_quality_limit")
                        && reason.contains("15.000")
                        && reason.contains("12.500")
                })
        }));
    }

    #[test]
    fn finalization_gate_rollback_recomputes_derived_levels_without_stacking() {
        use math_audio_iir_fir::{Biquad, BiquadFilterType};

        let (mut result, config) = recalibration_fixture(10.0);
        let peak = Biquad::new(BiquadFilterType::Peak, 200.0, 48_000.0, 0.8, 6.0);
        let plugin = roomeq_engine::topology::mark_plugin_stage(
            roomeq_engine::output::create_eq_plugin(std::slice::from_ref(&peak)),
            "post_route",
        );
        result.channels.get_mut("L").unwrap().plugins.push(plugin);
        result.channel_results.get_mut("L").unwrap().biquads = vec![peak];
        seed_post_dsp_calibration(&mut result, &config);

        let stale_left_trim = result
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .routing_graph
            .as_ref()
            .unwrap()
            .input_trim_db["L"];
        assert!(
            stale_left_trim < -5.0,
            "fixture must carry the boosted-trial trim"
        );
        assert!(result.channels["L"].plugins.iter().any(|plugin| {
            plugin.plugin_type == "eq"
                && plugin
                    .parameters
                    .get("room_eq_stage")
                    .and_then(serde_json::Value::as_str)
                    == Some("post_route")
        }));

        let mut published_fallback = result.clone();
        let fallback_dir = tempfile::tempdir().unwrap();
        crate::room_optimization::rebuild_routed_pruning_test_candidate(
            &mut published_fallback,
            &config,
            &HashMap::new(),
            48_000.0,
            fallback_dir.path(),
        )
        .expect("test candidate needs a refreshed acceptance report before fallback publishing");
        crate::room_optimization::publish_structural_baseline_for_test(
            &mut published_fallback,
            &config,
            48_000.0,
            fallback_dir.path(),
        )
        .expect("structural fallback should rebuild Home Cinema-derived level state");
        assert!(
            !published_fallback.channels["L"]
                .plugins
                .iter()
                .any(|plugin| plugin.plugin_type == "eq"),
            "the published structural fallback must not retain the candidate PEQ"
        );
        let fallback_report = published_fallback
            .metadata
            .bass_management
            .as_ref()
            .unwrap();
        let fallback_graph = fallback_report.routing_graph.as_ref().unwrap();
        assert!(
            (fallback_graph.input_trim_db["L"] + 2.0).abs() < 1.0e-5,
            "fallback routing graph retained stale L trim {} instead of the structural graph's recalculated trim",
            fallback_graph.input_trim_db["L"]
        );
        assert_eq!(
            fallback_graph.post_dsp_main_alignment_band_hz,
            Some([100.0, 400.0])
        );
        assert!(
            fallback_graph
                .advisories
                .contains(&"post_dsp_input_levels_aligned_down".to_string())
        );
        let effective = roomeq_engine::home_cinema::effective_bass_management(&config).unwrap();
        let expected_pre_protection_margin =
            roomeq_engine::home_cinema::simulate_bass_bus_headroom(
                Some(fallback_graph),
                &effective.config.headroom_model,
                effective.config.headroom_margin_db,
                48_000.0,
            )
            .unwrap();
        let recorded_pre_protection_margin = fallback_report.headroom_simulation.as_ref().unwrap();
        assert!(
            (recorded_pre_protection_margin.margin_db - expected_pre_protection_margin.margin_db)
                .abs()
                < 1.0e-9,
            "fallback headroom report must match recalculated routing trims before output safety gains"
        );

        let directory = tempfile::tempdir().unwrap();
        crate::room_optimization::rebuild_routed_pruning_test_candidate(
            &mut result,
            &config,
            &HashMap::new(),
            48_000.0,
            directory.path(),
        )
        .expect("the safety gate should roll back the regressing PEQ and rebuild its gains");

        assert!(
            !result.channels["L"]
                .plugins
                .iter()
                .any(|plugin| plugin.plugin_type == "eq"),
            "the real correction safety gate should remove the regressing PEQ"
        );
        assert!(result.channel_results["L"].biquads.is_empty());
        let graph = result
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .routing_graph
            .as_ref()
            .unwrap();
        let recalculated_left_trim = graph.input_trim_db["L"];
        assert!(
            (recalculated_left_trim + 2.0).abs() < 1.0e-5,
            "rollback must replace the boosted trial's {stale_left_trim} dB trim with the current graph's {recalculated_left_trim} dB trim"
        );
        assert!(graph.input_trim_db["R"].abs() < 1.0e-12);
        assert!(graph.input_trim_db["Sub1"].abs() < 1.0e-12);
        assert!(result.channels["L"].plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("speaker_calibration")
        }));
        assert!(result.channels["Sub1"].plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("configured_sub_gain")
        }));
        assert!(
            result.metadata.stage_outcomes.iter().any(|stage| {
                stage.stage == "final_correction_safety_L"
                    && stage
                        .advisories
                        .iter()
                        .any(|advisory| advisory == "audibility_regression_reverted_L:peq")
            }),
            "expected the safety gate to record PEQ rollback; outcomes: {:#?}",
            result.metadata.stage_outcomes
        );

        type GeneratedGainSignature = (Vec<(String, String, f64)>, HashMap<String, f64>);
        fn generated_gain_signature(
            result: &roomeq_engine::room_result::RoomOptimizationResult,
        ) -> GeneratedGainSignature {
            let mut gains: Vec<_> = result
                .channels
                .iter()
                .flat_map(|(role, chain)| {
                    chain.plugins.iter().filter_map(move |plugin| {
                        let label = plugin.parameters.get("label")?.as_str()?;
                        matches!(
                            label,
                            "post_dsp_input_level_alignment" | "post_dsp_output_headroom_safety"
                        )
                        .then(|| {
                            (
                                role.clone(),
                                label.to_string(),
                                plugin.parameters["gain_db"].as_f64().unwrap(),
                            )
                        })
                    })
                })
                .collect();
            gains.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
            let input_trims = result
                .metadata
                .bass_management
                .as_ref()
                .and_then(|report| report.routing_graph.as_ref())
                .unwrap()
                .input_trim_db
                .clone();
            (gains, input_trims)
        }
        let first_rebuilt_gains = generated_gain_signature(&result);
        assert!(
            super::recalibrate_post_dsp_levels(&mut result, &config, 48_000.0, directory.path())
                .unwrap()
        );
        assert_eq!(
            first_rebuilt_gains,
            generated_gain_signature(&result),
            "recalibration after rollback must replace, not stack, generated gains"
        );
    }

    #[test]
    fn later_output_attenuation_keeps_per_output_cuts_current() {
        use math_audio_iir_fir::{Biquad, BiquadFilterType};

        let (mut original, mut config) = recalibration_fixture(0.0);
        let frequencies =
            ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 256);
        let peak = Biquad::new(BiquadFilterType::Peak, 100.0, 48_000.0, 0.8, 9.0);
        let filters = [peak];
        let transfer =
            roomeq_engine::response::compute_peq_complex_response(&filters, &frequencies, 48_000.0);
        let correction_db =
            ndarray::Array1::from_iter(transfer.iter().map(|value| 20.0 * value.norm().log10()));
        for (role, measurement_name, level) in [
            ("L", "left", 80.0),
            ("R", "right", 80.0),
            ("Sub1", "sub", 70.0),
        ] {
            let spl = if role == "Sub1" {
                ndarray::Array1::from_elem(frequencies.len(), level)
            } else {
                correction_db.mapv(|response| level - response)
            };
            let curve = Curve {
                freq: frequencies.clone(),
                spl,
                phase: Some(ndarray::Array1::zeros(frequencies.len())),
                ..Curve::default()
            };
            let channel = original.channel_results.get_mut(role).unwrap();
            channel.initial_curve = curve.clone();
            channel.final_curve = curve.clone();
            let chain = original.channels.get_mut(role).unwrap();
            chain.initial_curve = Some((&curve).into());
            chain.final_curve = Some((&curve).into());
            chain.target_curve = Some((&curve).into());
            config.speakers.insert(
                measurement_name.to_string(),
                roomeq_model::SpeakerConfig::Single(roomeq_model::MeasurementSource::InMemory(
                    curve,
                )),
            );
        }

        // A matched +9 dB PEQ fills the synthetic measured notch, so the
        // first audibility gate accepts the correction against the flat
        // measured-level target. Electrically, however, the PEQ still
        // requires a substantial output cut; the later gate rechecks the
        // correction after per-output attenuation is installed.
        for role in ["L", "R"] {
            let mut target = original.channel_results[role].initial_curve.clone();
            target.spl.fill(80.0);
            original.channels.get_mut(role).unwrap().target_curve = Some((&target).into());
            let mut plugin = mark_plugin_stage(
                roomeq_engine::output::create_eq_plugin(&filters),
                "post_route",
            );
            plugin.parameters["label"] = serde_json::json!("fixture_room_eq_gain");
            original
                .channels
                .get_mut(role)
                .unwrap()
                .plugins
                .push(plugin);
            original.channel_results.get_mut(role).unwrap().biquads = filters.to_vec();
        }
        seed_post_dsp_calibration(&mut original, &config);

        let speaker_calibration_parameters = original.channels["L"]
            .plugins
            .iter()
            .find(|plugin| {
                plugin
                    .parameters
                    .get("label")
                    .and_then(serde_json::Value::as_str)
                    == Some("speaker_calibration")
            })
            .unwrap()
            .parameters
            .clone();
        let configured_sub_gain_parameters = original.channels["Sub1"]
            .plugins
            .iter()
            .find(|plugin| {
                plugin
                    .parameters
                    .get("label")
                    .and_then(serde_json::Value::as_str)
                    == Some("configured_sub_gain")
            })
            .unwrap()
            .parameters
            .clone();

        let directory = tempfile::tempdir().unwrap();
        let mut before_attenuation = original.clone();
        crate::room_optimization::rebuild_routed_pruning_test_candidate(
            &mut before_attenuation,
            &config,
            &HashMap::new(),
            48_000.0,
            directory.path(),
        )
        .expect("the unattenuated target-matching correction should pass the first gate");
        assert!(
            before_attenuation.channels["L"]
                .plugins
                .iter()
                .any(|plugin| {
                    plugin
                        .parameters
                        .get("label")
                        .and_then(serde_json::Value::as_str)
                        == Some("fixture_room_eq_gain")
                }),
            "the first correction safety pass unexpectedly changed the candidate: {:#?}",
            before_attenuation.metadata.stage_outcomes
        );
        let pre_attenuation_peaks = crate::electrical_headroom::assess_final_graph(
            &before_attenuation.to_dsp_chain_output(),
            48_000.0,
            directory.path(),
            &config.optimizer.finalization,
        )
        .unwrap();
        assert!(
            pre_attenuation_peaks
                .iter()
                .any(|output| { output.output == "Sub1" && output.inputs.len() >= 2 }),
            "fixture must retain a correlated multi-input sub output: {pre_attenuation_peaks:?}"
        );
        assert!(
            pre_attenuation_peaks.iter().any(|output| {
                output.output == "L"
                    && output.peak_dbfs.is_some_and(|peak| {
                        peak > config.optimizer.finalization.output_ceiling_dbfs + 5.0
                    })
            }),
            "the unattenuated correction must exceed the configured L output ceiling: {pre_attenuation_peaks:?}"
        );
        let (candidate, initial_required) =
            crate::room_optimization::run_full_strength_output_attenuation_trial_for_test(
                &original,
                &config,
                48_000.0,
                directory.path(),
            )
            .expect("the production output-attenuation trial path should complete");

        assert!(
            initial_required.get("L").is_some_and(|cut| *cut > 5.0),
            "fixture must enter the later output-attenuation route with a substantial L cut: {initial_required:?}"
        );
        assert!(
            initial_required
                .get("Sub1")
                .zip(initial_required.get("L"))
                .is_some_and(|(sub, main)| *sub > *main + 5.0),
            "the routed sub output must exercise a distinct per-output cut: {initial_required:?}"
        );
        assert!(
            candidate.channels["L"].plugins.iter().any(|plugin| {
                plugin
                    .parameters
                    .get("label")
                    .and_then(serde_json::Value::as_str)
                    == Some("fixture_room_eq_gain")
            }),
            "the post-attenuation safety gate must preserve this accepted correction; cuts={initial_required:?}, outcomes={:#?}",
            candidate.metadata.stage_outcomes
        );
        assert!(!candidate.metadata.stage_outcomes.iter().any(|stage| {
            stage
                .advisories
                .iter()
                .any(|advisory| advisory == "audibility_regression_reverted_L:peq")
        }));

        let graph = candidate
            .metadata
            .bass_management
            .as_ref()
            .unwrap()
            .routing_graph
            .as_ref()
            .unwrap();
        assert_eq!(graph.post_dsp_main_alignment_band_hz, Some([100.0, 400.0]));
        assert!(graph.input_trim_db.values().all(|trim| trim.is_finite()));
        assert!(candidate.channels["L"].plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("speaker_calibration")
        }));
        assert!(candidate.channels["Sub1"].plugins.iter().any(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str)
                == Some("configured_sub_gain")
        }));

        assert_eq!(
            candidate.channels["L"]
                .plugins
                .iter()
                .find(|plugin| {
                    plugin
                        .parameters
                        .get("label")
                        .and_then(serde_json::Value::as_str)
                        == Some("speaker_calibration")
                })
                .unwrap()
                .parameters,
            speaker_calibration_parameters,
            "the output trial must preserve the user's speaker calibration gain"
        );
        assert_eq!(
            candidate.channels["Sub1"]
                .plugins
                .iter()
                .find(|plugin| {
                    plugin
                        .parameters
                        .get("label")
                        .and_then(serde_json::Value::as_str)
                        == Some("configured_sub_gain")
                })
                .unwrap()
                .parameters,
            configured_sub_gain_parameters,
            "the output trial must preserve the user's configured sub gain"
        );

        let delivered_peaks = crate::electrical_headroom::assess_final_graph(
            &candidate.to_dsp_chain_output(),
            48_000.0,
            directory.path(),
            &config.optimizer.finalization,
        )
        .unwrap();
        let output_ceiling = config.optimizer.finalization.output_ceiling_dbfs;
        for output_name in ["L", "R", "Sub1"] {
            let output = delivered_peaks
                .iter()
                .find(|output| output.output == output_name)
                .unwrap_or_else(|| panic!("missing electrical assessment for {output_name}"));
            let peak = output
                .peak_dbfs
                .unwrap_or_else(|| panic!("missing delivered peak for {output_name}: {output:?}"));
            assert!(
                peak.is_finite() && peak <= output_ceiling + 1e-5,
                "delivered output {output_name} exceeds the configured electrical ceiling: peak={peak:.9} dBFS, ceiling={output_ceiling:.9} dBFS"
            );
            assert!(
                output.required_attenuation_db <= 1e-5,
                "delivered output {output_name} still requires attenuation: {output:?}"
            );
        }

        let final_cuts = final_electrical_headroom_cuts(&candidate);
        assert!(
            final_cuts.get("L").is_some_and(|cut| *cut > 5.0),
            "the test must observe the full-strength cut after the later gate: {final_cuts:?}"
        );
        let mut without_trial_cuts = candidate.clone();
        for chain in without_trial_cuts.channels.values_mut() {
            chain.plugins.retain(|plugin| {
                plugin
                    .parameters
                    .get("label")
                    .and_then(serde_json::Value::as_str)
                    != Some("final_electrical_headroom")
            });
        }
        let required_without_trial_cuts = crate::electrical_headroom::assess_final_graph(
            &without_trial_cuts.to_dsp_chain_output(),
            48_000.0,
            directory.path(),
            &config.optimizer.finalization,
        )
        .unwrap()
        .into_iter()
        .map(|output| (output.output, output.required_attenuation_db))
        .collect::<HashMap<_, _>>();
        for output in final_cuts.keys().chain(required_without_trial_cuts.keys()) {
            let retained = final_cuts.get(output).copied().unwrap_or(0.0);
            let required = required_without_trial_cuts
                .get(output)
                .copied()
                .unwrap_or(0.0);
            assert!(
                (retained - required).abs() <= 2e-5,
                "final output cut for {output} differs from independently recomputed current-chain requirement: retained={retained:.6} dB, required={required:.6} dB; all cuts={final_cuts:?}, recalculated={required_without_trial_cuts:?}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn protected_main_target_preserves_alignment_route_gains_and_slope() {
        use ndarray::Array1;
        let frequencies =
            Array1::from_iter((0..301).map(|i| 80.0 * (1000.0_f64 / 80.0).powf(i as f64 / 300.0)));
        let shape = frequencies.mapv(|frequency| -1.5 * (frequency / 220.0).log2());
        let peak =
            frequencies.mapv(|frequency| 6.0 * (-((frequency / 220.0).log2() / 0.4).powi(2)).exp());
        let curve = |spl| roomeq_model::Curve {
            freq: frequencies.clone(),
            spl,
            phase: Some(frequencies.mapv(|frequency| 23.0 - 360.0 * frequency * 0.002)),
            ..roomeq_model::Curve::default()
        };
        let raw_initial = curve(&shape + &peak);
        let raw_good = curve(&shape + &peak * 0.5);
        let raw_bad = curve(&raw_initial.spl - 8.0);
        let raw_target = curve(shape.clone());
        let band = (160.0, 500.0);
        let reference = crate::room_optimization::evaluate_preserved_target_passband(
            &raw_initial,
            &raw_good,
            &raw_target,
            2,
            band,
        )
        .unwrap();
        for alignment_gain in [-4.0, 0.0, 5.0] {
            for route_gain in [-3.0, 2.0] {
                // Independent fixed route transfer: acoustic phase, delay,
                // polarity, crossover rolloff, and two distinct level planes.
                let route_db = frequencies.mapv(|frequency| {
                    let ratio = (frequency / 80.0).powi(4);
                    20.0 * (ratio / (1.0 + ratio)).log10() + route_gain + alignment_gain
                });
                let phase_shift = frequencies.mapv(|frequency| 180.0 - 360.0 * frequency * 0.0013);
                let routed = |input: &roomeq_model::Curve| {
                    let mut output = input.clone();
                    output.spl += &route_db;
                    *output.phase.as_mut().unwrap() += &phase_shift;
                    output
                };
                let baseline = routed(&raw_initial);
                let before = routed(&raw_initial);
                let mut target = raw_target.clone();
                target.spl += alignment_gain;
                let evaluate = |after: &roomeq_model::Curve| {
                    super::protected_main_target_preservation(
                        &raw_initial,
                        &baseline,
                        &before,
                        after,
                        &target,
                        alignment_gain,
                        2,
                        band,
                    )
                    .unwrap()
                };
                let identity = evaluate(&before);
                assert!(
                    (identity.metrics.pre_target_weighted_rms_db
                        - identity.metrics.post_target_weighted_rms_db)
                        .abs()
                        < 1e-12
                );
                let routed_good = routed(&raw_good);
                let good = evaluate(&routed_good);
                assert!(
                    good.metrics.post_target_weighted_rms_db
                        < good.metrics.pre_target_weighted_rms_db
                );
                assert!(
                    (good.metrics.pre_target_weighted_rms_db
                        - reference.metrics.pre_target_weighted_rms_db)
                        .abs()
                        < 1e-10
                );
                assert!(
                    (good.metrics.post_target_weighted_rms_db
                        - reference.metrics.post_target_weighted_rms_db)
                        .abs()
                        < 1e-10
                );
                let routed_bad = routed(&raw_bad);
                let bad = evaluate(&routed_bad);
                assert!(
                    bad.metrics.post_target_weighted_rms_db
                        > bad.metrics.pre_target_weighted_rms_db
                );
            }
        }
        let mut unsupported = raw_target.clone();
        unsupported.freq = ndarray::array![200.0, 400.0];
        unsupported.spl = ndarray::array![0.0, 0.0];
        unsupported.phase = None;
        assert!(
            crate::room_optimization::evaluate_preserved_target_passband(
                &raw_initial,
                &raw_good,
                &unsupported,
                2,
                band
            )
            .is_none()
        );
    }

    #[test]
    fn supporting_only_result_carries_declared_t60_tolerance() {
        // The operator-declared report tolerance reaches output metadata so
        // the viewer flatness cell can light up; absent stays pending and
        // changes no acceptance math.
        let mut config = roomeq_model::RoomConfig::default();
        assert_eq!(config.report_t60_tolerance_s(), None);
        assert!(
            super::supporting_only_home_cinema_result(&config)
                .metadata
                .t60_flatness_tolerance_s
                .is_none()
        );
        config.reporting = Some(roomeq_model::ReportingConfig {
            t60_flatness_tolerance_s: Some(0.05),
        });
        // Structural validation of the knob itself lives in the model suite;
        // here only the config-to-metadata plumbing is under test.
        let result = super::supporting_only_home_cinema_result(&config);
        assert_eq!(result.metadata.t60_flatness_tolerance_s, Some(0.05));
    }

    #[test]
    fn physical_sub_group_preserves_topology_and_requires_complete_outputs() {
        let mut config = roomeq_model::RoomConfig::default();
        config.speakers.insert(
            String::from("subs"),
            roomeq_model::SpeakerConfig::MultiSub(roomeq_model::MultiSubGroup {
                name: String::from("subs"),
                speaker_name: None,
                subwoofers: vec![
                    roomeq_model::MeasurementSource::InMemory(
                        crate::test_fixtures::flat_curve()
                    );
                    2
                ],
                allpass_optimization: true,
                joint_optimization: false,
            }),
        );
        let mut system = roomeq_model::SystemConfig {
            subwoofers: Some(roomeq_model::SubwooferSystemConfig {
                config: roomeq_model::SubwooferStrategy::Mso,
                crossover: None,
                routing: Default::default(),
                outputs: (0..2)
                    .map(|index| roomeq_model::SubwooferOutput {
                        id: format!("Sub{index}"),
                        speaker: String::from("subs"),
                    })
                    .collect(),
            }),
            ..Default::default()
        };
        let Some(roomeq_model::SpeakerConfig::MultiSub(group)) =
            super::physical_sub_speaker_config(&config, &system).unwrap()
        else {
            panic!("physical group must retain MSO topology");
        };
        assert!(group.allpass_optimization);
        assert_eq!(group.subwoofers.len(), 2);
        system
            .subwoofers
            .as_mut()
            .unwrap()
            .outputs
            .push(roomeq_model::SubwooferOutput {
                id: String::from("Sub2"),
                speaker: String::from("subs"),
            });
        assert!(super::physical_sub_speaker_config(&config, &system).is_err());
    }

    use super::super::executor_tests::{flat_curve, flat_curve_with_phase, make_assembly};
    use super::super::types::WorkflowExecutor;
    use super::{HomeCinemaExecutor, canonical_main_roles, per_driver_low_pass_plan};
    use roomeq_model::{
        BassManagementConfig, BassManagementSubOutputReport, CrossoverConfig, MeasurementSource,
        MultiMeasurementStrategy, MultiSeatConfig, OptimizerConfig, ProcessingMode, RoomConfig,
        SpeakerConfig, SubwooferCrossoverRef, SubwooferStrategy, SubwooferSystemConfig,
        SupportingSourceConfig, SupportingSourceDecorrelation, SupportingSourceGroup, SystemConfig,
        SystemModel, TargetCurveConfig, default_config_version,
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

    fn per_sub_test_config() -> RoomConfig {
        let mut sys = home_cinema_sys_with_sub();
        sys.subwoofers.as_mut().unwrap().crossover = Some(SubwooferCrossoverRef::PerSub(vec![
            "bass_xover1".to_string(),
            "bass_xover2".to_string(),
        ]));
        room_config(
            stereo_speakers(),
            &sys,
            tiny_optimizer(),
            Some(HashMap::from([
                (
                    "bass_xover1".to_string(),
                    CrossoverConfig {
                        crossover_type: "LR24".to_string(),
                        frequency: Some(80.0),
                        frequencies: None,
                        frequency_range: None,
                    },
                ),
                (
                    "bass_xover2".to_string(),
                    CrossoverConfig {
                        crossover_type: "LR48".to_string(),
                        frequency: Some(95.0),
                        frequencies: None,
                        frequency_range: None,
                    },
                ),
            ])),
            None,
        )
    }

    fn sub_output(name: &str, low_pass_hz: Option<f64>) -> BassManagementSubOutputReport {
        BassManagementSubOutputReport {
            output_role: name.to_string(),
            gain_db: 0.0,
            delay_ms: 0.0,
            polarity_inverted: false,
            strategy_source: "mso".to_string(),
            headroom_contribution_db: 0.0,
            selected_low_pass_hz: low_pass_hz,
        }
    }

    #[test]
    fn per_driver_low_pass_plan_pairs_positionally_with_own_family() {
        let config = per_sub_test_config();
        let outputs = HashMap::from([
            ("subs_1".to_string(), sub_output("subs_1", Some(80.0))),
            ("subs_2".to_string(), sub_output("subs_2", Some(95.0))),
        ]);
        let first = per_driver_low_pass_plan(&config, 0, "subs_1", &outputs, "LR24").unwrap();
        assert_eq!(first.frequency_hz, 80.0);
        assert_eq!(first.crossover_type, "LR24");
        let second = per_driver_low_pass_plan(&config, 1, "subs_2", &outputs, "LR24").unwrap();
        assert_eq!(second.frequency_hz, 95.0);
        assert_eq!(second.crossover_type, "LR48");
    }

    #[test]
    fn per_driver_low_pass_plan_deploys_nothing_without_selection() {
        let config = per_sub_test_config();
        // Skipped pair: no low-pass recorded, no plugin deployed.
        let outputs = HashMap::from([("subs_1".to_string(), sub_output("subs_1", None))]);
        assert!(per_driver_low_pass_plan(&config, 0, "subs_1", &outputs, "LR24").is_none());
        // Legacy single-crossover config: deploys nothing by construction,
        // even if a value were ever recorded.
        let mut legacy = home_cinema_sys_with_sub();
        legacy.subwoofers.as_mut().unwrap().crossover =
            Some(SubwooferCrossoverRef::Shared("bass_xo".to_string()));
        let legacy_config = room_config(
            stereo_speakers(),
            &legacy,
            tiny_optimizer(),
            Some(crossovers_fixed()),
            None,
        );
        let outputs = HashMap::from([("sub".to_string(), sub_output("sub", Some(80.0)))]);
        assert!(per_driver_low_pass_plan(&legacy_config, 0, "sub", &outputs, "LR24").is_none());
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
                SpeakerConfig::Single(stationary_fixture_source(&[flat_curve_with_phase()])),
            ),
            (
                "right".to_string(),
                SpeakerConfig::Single(stationary_fixture_source(&[flat_curve_with_phase()])),
            ),
        ])
    }

    fn stationary_fixture_source(curves: &[roomeq_model::Curve]) -> MeasurementSource {
        let measurements: Vec<_> = curves
            .iter()
            .enumerate()
            .map(|(seat, curve)| {
                serde_json::json!({"inline": {
                    "name": format!("seat-{seat}"),
                    "frequencies": curve.freq.to_vec(),
                    "magnitude_db": curve.spl.to_vec(),
                    "phase_deg": curve.phase.as_ref().map(|phase| phase.to_vec())
                }})
            })
            .collect();
        let mut source = if measurements.len() == 1 {
            measurements[0].clone()
        } else {
            serde_json::json!({"measurements": measurements.iter().map(|entry| entry["inline"].clone()).collect::<Vec<_>>()})
        };
        source["provenance"] = serde_json::json!({
            "capture_kind": "stationary_ir", "timing_reference_id": "fixture-common-clock"
        });
        serde_json::from_value(source).expect("declared stationary fixture")
    }

    fn home_cinema_sys_with_sub() -> SystemConfig {
        SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([
                ("Left".to_string(), "left".to_string()),
                ("Right".to_string(), "right".to_string()),
            ]),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                crossover: Some(roomeq_model::SubwooferCrossoverRef::PerSub(vec![
                    "bass_xo".to_string(),
                ])),
                routing: Default::default(),
                outputs: vec![roomeq_model::SubwooferOutput {
                    id: "sub".to_string(),
                    speaker: "sub".to_string(),
                }],
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
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
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
            SpeakerConfig::Single(stationary_fixture_source(&[flat_curve_with_phase()])),
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
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
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
                    // Synthetic fixture has no shared-time acoustic evidence.
                    allow_unverified_acoustics: true,
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
        optimizer.allow_delay = Some(true);
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
                    // Synthetic fixture has no shared-time acoustic evidence.
                    allow_unverified_acoustics: true,
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
        let mut optimizer = tiny_optimizer();
        optimizer.allow_delay = Some(true);
        let config = room_config(speakers, &sys, optimizer, None, None);
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
        let mut speakers = HashMap::from([(
            "wide".to_string(),
            SpeakerConfig::SupportingSource(SupportingSourceGroup {
                name: "Wide".to_string(),
                speaker_name: None,
                primary: MeasurementSource::InMemory(flat_curve()),
                support: MeasurementSource::InMemory(flat_curve()),
                supporting_source: SupportingSourceConfig {
                    // Synthetic fixture has no shared-time acoustic evidence.
                    allow_unverified_acoustics: true,
                    fir_taps: 128,
                    decorrelation: SupportingSourceDecorrelation::None,
                    ..Default::default()
                },
            }),
        )]);
        speakers.insert(
            "sub".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve())),
        );
        let sys = SystemConfig {
            model: SystemModel::HomeCinema,
            speakers: HashMap::from([("WideLeft".to_string(), "wide".to_string())]),
            subwoofers: Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                crossover: None,
                routing: Default::default(),
                outputs: vec![roomeq_model::SubwooferOutput {
                    id: "Sub1".to_string(),
                    speaker: "sub".to_string(),
                }],
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
            SpeakerConfig::Single(stationary_fixture_source(&[flat_curve_with_phase()])),
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
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
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
        assert!(bass_report.lfe.unwrap().gain_applied_to_chain);
    }

    #[test]
    fn home_cinema_with_sub_gain_limit_advisory_runs() {
        let sys = home_cinema_sys_with_sub();
        let mut speakers = stereo_speakers_with_phase();
        speakers.insert(
            "sub".to_string(),
            SpeakerConfig::Single(stationary_fixture_source(&[flat_curve_with_phase()])),
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
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
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
            SpeakerConfig::Single(stationary_fixture_source(&[flat_curve_with_phase()])),
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
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
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
            SpeakerConfig::Single(stationary_fixture_source(&[flat_curve_with_phase()])),
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
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
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
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
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
            SpeakerConfig::Single(stationary_fixture_source(&[flat_curve_with_phase()])),
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
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
            cea2034_cache: None,
        };

        let mut assembly = make_assembly(&config, &sys);
        let result = HomeCinemaExecutor
            .execute(&mut assembly)
            .expect("home cinema with a sub and prepared target should run");

        for channel_name in ["Left", "Right", "sub"] {
            assert!(
                result.channels[channel_name].target_curve.is_some(),
                "{channel_name} output chain must retain its prepared target"
            );
        }
    }

    #[test]
    fn home_cinema_routing_retains_same_primary_main_and_sub_capture() {
        let sys = home_cinema_sys_with_sub();
        let mut speakers = stereo_speakers_with_phase();
        let mut first = flat_curve_with_phase();
        first.spl.fill(80.0);
        first.phase.as_mut().unwrap().fill(37.0);
        let mut second = first.clone();
        second.spl.fill(90.0);
        second.phase.as_mut().unwrap().fill(-89.0);
        for name in ["left", "right", "sub"] {
            speakers.insert(
                name.into(),
                SpeakerConfig::Single(stationary_fixture_source(&[first.clone(), second.clone()])),
            );
        }
        let mut optimizer = tiny_optimizer();
        optimizer.max_freq = 2000.0;
        optimizer.multi_seat = Some(MultiSeatConfig {
            primary_seat: 1,
            ..Default::default()
        });
        let config = room_config(speakers, &sys, optimizer, Some(crossovers_fixed()), None);
        let mut assembly = make_assembly(&config, &sys);
        let result = HomeCinemaExecutor.execute(&mut assembly).unwrap();
        for name in ["Left", "Right", "sub"] {
            let initial: roomeq_engine::Curve =
                result.channels[name].initial_curve.clone().unwrap().into();
            assert!(initial.spl.iter().all(|spl| (*spl - 90.0).abs() < 1e-8));
            assert!(
                initial
                    .phase
                    .as_ref()
                    .expect("routing lost measured primary phase")
                    .iter()
                    .all(|phase| (*phase + 89.0).abs() < 1e-8)
            );
        }
        let optimization = result
            .metadata
            .bass_management
            .unwrap()
            .optimization
            .unwrap();
        assert!(optimization.phase_available);
        assert!(
            optimization
                .advisories
                .iter()
                .any(|value| value == "crossover_phase_coherence_unverified")
        );
        assert!(
            !optimization
                .advisories
                .iter()
                .any(|value| value.contains("missing_phase"))
        );
    }

    #[test]
    fn home_cinema_rejects_bad_phase_quality_even_on_nonprimary_seat() {
        let sys = home_cinema_sys_with_sub();
        let mut speakers = stereo_speakers_with_phase();
        let mut good = flat_curve_with_phase();
        good.coherence = Some(ndarray::Array1::ones(good.freq.len()));
        let mut bad = good.clone();
        bad.coherence.as_mut().unwrap().fill(0.1);
        speakers.insert(
            "sub".into(),
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![good, bad])),
        );
        let config = room_config(
            speakers,
            &sys,
            tiny_optimizer(),
            Some(crossovers_fixed()),
            None,
        );
        let mut assembly = make_assembly(&config, &sys);
        let error = HomeCinemaExecutor
            .execute(&mut assembly)
            .unwrap_err()
            .to_string();
        assert!(error.contains("channel 'sub' seat 1"), "{error}");
        assert!(error.contains("unreliable coherence"), "{error}");
    }

    #[test]
    fn crossover_quality_gate_includes_group_frequency_overrides() {
        let mut sys = home_cinema_sys_with_sub();
        let mut speakers = stereo_speakers_with_phase();
        let mut sub = flat_curve_with_phase();
        sub.coherence = Some(sub.freq.mapv(|hz| {
            if (200.0..600.0).contains(&hz) {
                0.1
            } else {
                1.0
            }
        }));
        speakers.insert(
            "sub".into(),
            SpeakerConfig::Single(MeasurementSource::InMemory(sub)),
        );
        let mut config = room_config(
            speakers,
            &sys,
            tiny_optimizer(),
            Some(crossovers_fixed()),
            None,
        );
        assert!(super::configured_crossover_phase_advisories(&config, &sys).is_ok());
        let mut higher = config.crossovers.as_ref().unwrap()["bass_xo"].clone();
        higher.frequency = Some(300.0);
        config
            .crossovers
            .as_mut()
            .unwrap()
            .insert("higher".into(), higher);
        sys.bass_management
            .get_or_insert_with(Default::default)
            .group_crossovers
            .insert("lcr".into(), "higher".into());
        let error = super::configured_crossover_phase_advisories(&config, &sys).unwrap_err();
        assert!(
            error.to_string().contains("unreliable coherence"),
            "{error}"
        );
    }

    #[test]
    fn explicit_seats_do_not_authorize_mismatched_coherent_post_eq_clocks() {
        use autoeq_core::{
            InlineMeasurement, MeasurementMultiple, MeasurementProvenance, MeasurementRef,
            ProvenanceCaptureKind,
        };
        let source = |clock: &str, declared: &str, phase_deg: f64, moving: bool| {
            let curve = flat_curve_with_phase();
            let takes = ["seat-a", "seat-b"].iter().enumerate().map(|(index, seat)| serde_json::json!({
                "microphone_id": if moving { "moving-mic".into() } else { format!("mic-{index}") }, "seat_id": seat,
                "device_id": "fixture-device", "offset_samples": 0.0,
                "skew_ppm": 0.0, "residual_uncertainty_us": 1.0,
                "correction_applied": "resampled", "timing_reference_id": clock,
                "calibration_id": "fixture-calibration", "gain_db": 0.0,
                "calibration_orientation": "on_axis", "position_m": [index as f64, 0.0, 0.0],
                "position_uncertainty_mm": 1.0, "preserves_acoustic_delay": true,
                "quality_passed": true,
            })).collect::<Vec<_>>();
            let capture: autoeq_core::capture_provenance::CaptureProvenance =
                serde_json::from_value(serde_json::json!({
                    "geometry": "spread", "takes": takes,
                }))
                .unwrap();
            if moving {
                assert!(capture.coherent_reference_at_frequency(2, 160.0).is_err());
            } else {
                assert_eq!(
                    capture.coherent_reference_at_frequency(2, 160.0).unwrap(),
                    clock
                );
            }
            MeasurementSource::Multiple(MeasurementMultiple {
                measurements: ["seat-a", "seat-b"]
                    .iter()
                    .map(|seat| {
                        MeasurementRef::Inline(InlineMeasurement {
                            frequencies: curve.freq.to_vec(),
                            magnitude_db: curve.spl.to_vec(),
                            phase_deg: Some(vec![phase_deg; curve.freq.len()]),
                            name: Some((*seat).into()),
                            wav_path: None,
                            csv_path: None,
                        })
                    })
                    .collect(),
                speaker_name: None,
                provenance: MeasurementProvenance {
                    capture_kind: ProvenanceCaptureKind::StationaryIr,
                    timing_reference_id: Some(declared.into()),
                    capture: Some(capture),
                    ..Default::default()
                },
            })
        };
        let sys = home_cinema_sys_with_sub();
        for (sub_clock, sub_declared, moving, eligible) in [
            ("main-clock", "main-clock", false, true),
            ("sub-clock", "sub-clock", false, false),
            ("sub-clock", "main-clock", false, false),
            ("main-clock", "main-clock", true, false),
        ] {
            let mut optimizer = tiny_optimizer();
            optimizer.max_freq = 2_000.0;
            optimizer.multi_seat = Some(MultiSeatConfig {
                all_channel_enabled: true,
                seat_identity: Some(roomeq_model::SeatIdentityMap {
                    ids: vec!["seat-a".into(), "seat-b".into()],
                }),
                ..Default::default()
            });
            let config = RoomConfig {
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
                speakers: HashMap::from([
                    (
                        "left".into(),
                        SpeakerConfig::Single(source("main-clock", "main-clock", 17.0, moving)),
                    ),
                    (
                        "right".into(),
                        SpeakerConfig::Single(source("main-clock", "main-clock", 17.0, moving)),
                    ),
                    (
                        "sub".into(),
                        SpeakerConfig::Single(source(sub_clock, sub_declared, -31.0, moving)),
                    ),
                ]),
                crossovers: Some(crossovers_fixed()),
                optimizer,
                ..Default::default()
            };
            let inventory = serde_json::to_value(&config.speakers).unwrap();
            let mut assembly = make_assembly(&config, config.system.as_ref().unwrap());
            let result = HomeCinemaExecutor.execute(&mut assembly).unwrap();
            assert_eq!(serde_json::to_value(&config.speakers).unwrap(), inventory);
            let main = match &config.speakers["left"] {
                SpeakerConfig::Single(source) => source,
                _ => unreachable!(),
            };
            assert_eq!(
                crate::group_measurements::routed_seat_identity_order(
                    &config,
                    main,
                    &config.speakers["sub"],
                    2,
                    2,
                )
                .unwrap(),
                vec!["seat-a", "seat-b"]
            );
            for role in ["Left", "Right"] {
                let refused = result.metadata.stage_outcomes.iter().any(|stage| {
                    stage.stage == format!("routed_common_eq_timing_{role}")
                        && stage.checks.iter().any(|check| {
                            !check.passed
                                && check.diagnostic.as_ref().is_some_and(|diagnostic| {
                                    diagnostic.contains("InsufficientEvidence")
                                        && diagnostic.contains("timing")
                                })
                        })
                });
                assert_eq!(
                    refused, !eligible,
                    "clock={sub_clock}; declared={sub_declared}; moving={moving}; {:#?}",
                    result.metadata.stage_outcomes
                );
                if eligible {
                    assert!(
                        result.metadata.stage_outcomes.iter().any(|stage| {
                            stage.stage == format!("post_eq_main_target_preservation_{role}")
                        }),
                        "valid physical-output-only mapping never reached the actual common optimizer"
                    );
                }
            }
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
            measured_impulse_responses: Default::default(),
            ctc: None,
            reporting: None,
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

#[cfg(test)]
mod splice_revert_tests {
    use super::{
        is_splice_breaking_correction_eq, strip_next_splice_breaking_stage, underfill_error_role,
    };
    use roomeq_model::{ChannelDspChain, PluginConfigWrapper};

    fn staged_plugin(
        plugin_type: &str,
        stage: Option<&str>,
        label: Option<&str>,
    ) -> PluginConfigWrapper {
        let mut parameters = serde_json::Map::new();
        if let Some(stage) = stage {
            parameters.insert(
                "room_eq_stage".to_string(),
                serde_json::Value::String(stage.to_string()),
            );
        }
        if let Some(label) = label {
            parameters.insert(
                "label".to_string(),
                serde_json::Value::String(label.to_string()),
            );
        }
        PluginConfigWrapper {
            plugin_type: plugin_type.to_string(),
            parameters: serde_json::Value::Object(parameters),
        }
    }

    fn correction_chain() -> ChannelDspChain {
        ChannelDspChain {
            physical_correction_target: None,
            channel: "R".to_string(),
            plugins: vec![
                staged_plugin("delay", Some("pre_route"), None),
                staged_plugin("eq", Some("post_route"), Some("room_eq_correction")),
                staged_plugin("convolution", Some("post_route"), None),
                staged_plugin(
                    "crossover",
                    Some("route_owned"),
                    Some("room_eq_route_owned"),
                ),
            ],
            drivers: None,
            initial_curve: None,
            final_curve: None,
            eq_response: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
            joint_sub: None,
            early_reflections: None,
            t60_octaves: None,
            speech_transmission: None,
            waterfall: None,
            resonance_decays: None,
            wavelet: None,
            early_late_curves: None,
            target_curve: None,
        }
    }

    #[test]
    fn underfill_role_parses_routed_error() {
        let message = "optimization failed: final routed crossover underfill for 'R' is 9.062 dB at 80.0 Hz (limit 3.0 dB)";
        assert_eq!(underfill_error_role(message).as_deref(), Some("R"));
        assert_eq!(underfill_error_role("unrelated failure"), None);
        assert_eq!(
            underfill_error_role("final routed crossover underfill for '' is "),
            None
        );
    }

    #[test]
    fn splice_strip_removes_fir_then_peq_and_preserves_routing() {
        let mut chain = correction_chain();
        assert_eq!(strip_next_splice_breaking_stage(&mut chain), Some("fir"));
        assert!(
            chain
                .plugins
                .iter()
                .all(|plugin| plugin.plugin_type != "convolution"),
            "FIR stage must be gone"
        );
        assert_eq!(strip_next_splice_breaking_stage(&mut chain), Some("peq"));
        let remaining: Vec<_> = chain
            .plugins
            .iter()
            .map(|plugin| plugin.plugin_type.as_str())
            .collect();
        assert_eq!(remaining, vec!["delay", "crossover"]);
        assert_eq!(strip_next_splice_breaking_stage(&mut chain), None);
    }

    #[test]
    fn splice_predicates_ignore_structural_plugins() {
        assert!(!is_splice_breaking_correction_eq(&staged_plugin(
            "eq",
            Some("pre_route"),
            Some("room_eq_correction")
        )));
        assert!(!is_splice_breaking_correction_eq(&staged_plugin(
            "eq",
            Some("post_route"),
            Some("channel_matching")
        )));
        assert!(is_splice_breaking_correction_eq(&staged_plugin(
            "eq",
            Some("post_route"),
            Some("room_eq_correction")
        )));
    }
}
