//! Raw, position-identified evidence survives optimization until final replay.
use super::RoomOptimizationResult;
use roomeq_model::{
    AutoeqError, ChannelDspChain, Curve, FinalSeatEvaluation, MeasurementSource,
    PluginConfigWrapper, Result, RoomConfig, SpeakerConfig,
};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
mod bounded_sum;
#[cfg(test)]
mod asymmetric;
use bounded_sum::{Summed, process_branch, sum_branches};

type DriverIdentity = (Option<String>, usize);
type SourceCapture<'a> = (Option<DriverIdentity>, &'a MeasurementSource);

/// Immutable, native-resolution capture loaded before processing begins.
pub struct Capture {
    channel: String,
    driver: Option<DriverIdentity>,
    curves: Vec<Curve>,
    seat_labels: Option<Vec<String>>,
}

/// Explicit label from the immutable pre-optimization capture, without inferring
/// identity for legacy unnamed measurements. Physical replay validates alignment.
pub fn training_seat_label(
    captures: &[Capture],
    logical_input: &str,
    seat_index: usize,
) -> Option<String> {
    captures
        .iter()
        .find(|capture| capture.channel == logical_input)
        .and_then(|capture| capture.seat_labels.as_ref())
        .and_then(|labels| labels.get(seat_index))
        .cloned()
}

fn invalid(message: impl Into<String>) -> AutoeqError {
    AutoeqError::InvalidMeasurement {
        message: message.into(),
    }
}

pub fn capture_training(config: &RoomConfig) -> Result<Vec<Capture>> {
    let roles: BTreeMap<String, String> = match &config.system {
        Some(system) => system
            .speakers
            .iter()
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect(),
        None => config
            .speakers
            .keys()
            .map(|name| (name.clone(), name.clone()))
            .collect(),
    };
    let mut captures = Vec::new();
    for (role, key) in roles {
        let speaker = config
            .speakers
            .get(&key)
            .ok_or_else(|| invalid(format!("unknown measurement '{key}'")))?;
        let sources: Vec<SourceCapture<'_>> = match speaker {
            SpeakerConfig::Single(source) => vec![(None, source)],
            SpeakerConfig::Topology(topology) => topology
                .drivers
                .iter()
                .enumerate()
                .map(|(i, d)| (Some((Some(d.id.clone()), i)), &d.measurement))
                .collect(),
            SpeakerConfig::MultiSub(group) => group
                .subwoofers
                .iter()
                .enumerate()
                .map(|(i, source)| (Some((None, i)), source))
                .collect(),
            SpeakerConfig::Dba(group) => group
                .front
                .iter()
                .chain(&group.rear)
                .enumerate()
                .map(|(i, source)| (Some((None, i)), source))
                .collect(),
            SpeakerConfig::Cardioid(group) => vec![
                (Some((None, 0)), &group.front),
                (Some((None, 1)), &group.rear),
            ],
            SpeakerConfig::Group(group) => group
                .measurements
                .iter()
                .enumerate()
                .map(|(i, source)| (Some((None, i)), source))
                .collect(),
            // This mode's timing and coherent evidence have a separate explicit
            // contract. Do not certify its spatial power average as measured seats.
            SpeakerConfig::SupportingSource(_) => continue,
        };
        for (driver, source) in sources {
            // No optimization/display cap: native narrow features are evidence.
            let curves = crate::measurement::load_source_individual_with_frequency_samples(
                source,
                usize::MAX,
            )
            .map_err(|error| invalid(error.to_string()))?;
            for curve in &curves {
                curve.validate("final-seat raw capture")?;
            }
            if matches!(speaker, SpeakerConfig::Group(_)) && curves.len() > 1 {
                return Err(invalid(
                    "multi-seat legacy driver groups require explicit topology IDs for final replay",
                ));
            }
            captures.push(Capture {
                channel: role.clone(),
                driver,
                curves,
                seat_labels: crate::group_measurements::seat_labels(source),
            });
        }
    }
    Ok(captures)
}

fn physical_captures(
    captures: &[Capture],
    result: &RoomOptimizationResult,
) -> Result<BTreeMap<String, Vec<Curve>>> {
    let mut physical = BTreeMap::new();
    let routed = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|b| b.routing_graph.as_ref())
        .is_some();
    for (i, capture) in captures.iter().enumerate() {
        for other in &captures[..i] {
            if (routed || capture.channel == other.channel)
                && let (Some(a), Some(b)) = (&capture.seat_labels, &other.seat_labels)
                && a != b
            {
                return Err(invalid(
                    "final-seat branch labels differ; captures must use identical seat order",
                ));
            }
        }
    }
    for capture in captures {
        let name = if let Some((id, index)) = &capture.driver {
            result
                .channels
                .get(&capture.channel)
                .and_then(|chain| chain.drivers.as_ref())
                .and_then(|drivers| {
                    drivers.iter().find(|driver| match id {
                        Some(id) => driver.name == *id,
                        None => driver.index == *index,
                    })
                })
                .map(|driver| driver.name.clone())
                .ok_or_else(|| {
                    invalid(format!(
                        "missing driver identity for '{}' index {index}",
                        capture.channel
                    ))
                })?
        } else {
            capture.channel.clone()
        };
        if physical
            .insert(name.clone(), capture.curves.clone())
            .is_some()
        {
            return Err(invalid(format!(
                "ambiguous physical output identity '{name}'"
            )));
        }
    }
    Ok(physical)
}

fn correction(plugin: &PluginConfigWrapper) -> bool {
    matches!(plugin.plugin_type.as_str(), "eq" | "convolution")
}

fn apply(
    result: &RoomOptimizationResult,
    owner: &str,
    mut plugins: Vec<PluginConfigWrapper>,
    curve: &Curve,
    baseline: bool,
    fs: f64,
    dir: &Path,
) -> Result<Curve> {
    if baseline {
        plugins.retain(|p| !correction(p));
    }
    let mut chain = result
        .channels
        .get(owner)
        .cloned()
        .ok_or_else(|| invalid(format!("missing DSP owner '{owner}'")))?;
    // channel_results[owner].fir_coeffs owns the channel-level FIR, not an
    // arbitrary single convolution in a driver/stage replay sharing this owner.
    let channel_convolutions: Vec<_> = chain.plugins.iter()
        .filter(|plugin| plugin.plugin_type == "convolution")
        .collect();
    let retained_reference = if channel_convolutions.len() == 1 {
        channel_convolutions[0].parameters.get("ir_file")
            .and_then(|value| value.as_str()).map(str::to_owned)
    } else {
        None
    };
    chain.plugins = plugins;
    chain.drivers = None;
    let mut embedded = HashMap::new();
    // A retained single FIR and its sidecar must describe the same transfer.
    // Do not guess ownership for multiple FIR plugins.
    let paths: Vec<_> = chain
        .plugins
        .iter()
        .filter(|p| p.plugin_type == "convolution")
        .filter_map(|p| p.parameters.get("ir_file").and_then(|v| v.as_str()))
        .collect();
    if paths.len() == 1
        && retained_reference.as_deref() == Some(paths[0])
        && let Some(taps) = result.channel_results.get(owner).and_then(|c| c.fir_coeffs.as_ref())
    {
        let path = dir.join(paths[0]);
        let evaluated_taps = if path.exists() {
            let mut channels = crate::ctc::read_wav_channels_f64(
                &path, crate::ctc::checked_sample_rate(fs)?, "final-seat convolution",
            )?;
            if channels.len() != 1 || channels[0].len() != taps.len()
                || channels[0].iter().zip(taps).any(|(stored, retained)| {
                    !stored.is_finite() || !retained.is_finite()
                        || (*stored as f32) != (*retained as f32)
                })
            {
                return Err(invalid(format!(
                    "final-seat convolution '{}' conflicts with retained FIR coefficients for '{owner}'",
                    path.display(),
                )));
            }
            // Replay this validated snapshot, avoiding a second filesystem read.
            // Preserve the actual WAV's float32 serialization quantization.
            channels.remove(0)
        } else {
            taps.clone()
        };
        embedded.insert(paths[0].to_string(), evaluated_taps);
    }
    let mut corrected = crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
        &chain, curve, fs, dir, &embedded,
    )?;
    // The generic response helper also supports electrical-only curves and
    // supplies their filter phase. Here the curve is acoustic evidence:
    // multiplying by a known filter cannot recover unknown acoustic phase.
    if curve.phase.is_none() {
        corrected.phase = None;
    }
    Ok(corrected)
}

#[cfg(test)]
fn sum(curves: &[Curve]) -> Result<Curve> {
    let branches: Vec<_> = curves
        .iter()
        .enumerate()
        .map(|(i, curve)| bounded_sum::Branch {
            output: i.to_string(),
            measured: curve.clone(),
            upper: None,
        })
        .collect();
    Ok(bounded_sum::sum_branches(&branches, 0.0, f64::INFINITY)?.curve)
}

fn stage(chain: &ChannelDspChain, name: &str) -> Vec<PluginConfigWrapper> {
    chain
        .plugins
        .iter()
        .filter(|p| p.parameters.get("room_eq_stage").and_then(|v| v.as_str()) == Some(name))
        .cloned()
        .collect()
}

fn measured<'a>(
    physical: &'a BTreeMap<String, Vec<Curve>>,
    output: &str,
    seat: usize,
) -> Result<&'a Curve> {
    physical.get(output).and_then(|curves| curves.get(seat))
        .ok_or_else(|| invalid(format!("missing physical output '{output}' at seat {seat}; singleton captures are not broadcast to other seats")))
}

struct ReplayContext<'a> {
    config: &'a RoomConfig,
    partition: &'a str,
    fs: f64,
    dir: &'a Path,
}

/// Baseline and delivered playback for one logical source at one physical seat.
/// The baseline retains structural routing/crossovers and disables correction.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FinalPhysicalSeatPlayback {
    /// Explicit capture label when supplied by the caller; never inferred.
    pub seat_label: Option<String>,
    pub partition: String,
    pub logical_input: String,
    pub seat_index: usize,
    pub physical_outputs: Vec<String>,
    pub baseline: Curve,
    pub delivered: Curve,
    pub baseline_support: Vec<roomeq_model::SummationSupportEvidence>,
    pub delivered_support: Vec<roomeq_model::SummationSupportEvidence>,
}

/// Resolve a pre-optimization capture snapshot using runtime output identities.
pub fn training_physical_captures(
    captures: &[Capture],
    result: &RoomOptimizationResult,
) -> Result<BTreeMap<String, Vec<Curve>>> {
    physical_captures(captures, result)
}

/// Replay immutable physical captures through the same final playback contract
/// used by runtime seat acceptance. Missing branches/phase/sidecars are errors.
/// `partition` must match the partition of declared acoustic support bounds.
#[allow(clippy::too_many_arguments)]
pub fn replay_final_physical_seat(
    result: &RoomOptimizationResult,
    physical: &BTreeMap<String, Vec<Curve>>,
    input: &str,
    seat: usize,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
    partition: &str,
) -> Result<FinalPhysicalSeatPlayback> {
    crate::export::validate_final_routed_stage_ownership(result)?;
    if result.metadata.ctc.is_some() {
        return Err(invalid(
            "final-seat replay requires ear-identified CTC transfer measurements",
        ));
    }
    if !matches!(partition, "training" | "held_out") {
        return Err(invalid("unknown physical capture partition"));
    }
    for curves in physical.values() {
        for curve in curves {
            curve.validate("final physical-seat playback")?;
        }
    }
    let context = ReplayContext {
        config,
        partition,
        fs,
        dir,
    };
    let (pre, outputs) = replay(result, physical, input, seat, true, &context)?;
    let (post, _) = replay(result, physical, input, seat, false, &context)?;
    Ok(FinalPhysicalSeatPlayback {
        seat_label: None,
        partition: partition.into(),
        logical_input: input.into(),
        seat_index: seat,
        physical_outputs: outputs,
        baseline: pre.curve,
        delivered: post.curve,
        baseline_support: pre.support,
        delivered_support: post.support,
    })
}

fn replay(
    result: &RoomOptimizationResult,
    physical: &BTreeMap<String, Vec<Curve>>,
    input: &str,
    seat: usize,
    baseline: bool,
    context: &ReplayContext<'_>,
) -> Result<(Summed, Vec<String>)> {
    let ReplayContext {
        config,
        partition,
        fs,
        dir,
    } = *context;
    let mut grid: Vec<_> = physical
        .values()
        .filter_map(|curves| curves.get(seat))
        .flat_map(|curve| curve.freq.iter().copied())
        .collect();
    grid.push(config.optimizer.max_freq);
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    let grid = ndarray::Array1::from(grid);
    if let Some(graph) = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|b| b.routing_graph.as_ref())
    {
        let mut branches = Vec::new();
        let mut outputs = Vec::new();
        for route in graph.routes.iter().filter(|r| r.source_channel == input) {
            let raw = measured(physical, &route.destination, seat)?;
            // Match serialized routed export: input pre-route -> route matrix
            // gain/polarity, crossover/delay -> destination post-route.
            let pre = result
                .channels
                .get(input)
                .ok_or_else(|| invalid("missing route input"))?;
            let (post_owner, post) = result
                .channels
                .get_key_value(&route.destination)
                .or_else(|| {
                    result
                        .channels
                        .get_key_value(&graph.physical_sub_output)
                        .filter(|(_, chain)| {
                            chain.drivers.as_ref().is_some_and(|drivers| {
                                drivers
                                    .iter()
                                    .any(|driver| driver.name == route.destination)
                            })
                        })
                })
                .ok_or_else(|| invalid("missing route output"))?;
            let mut route_plugins = vec![roomeq_engine::output::create_gain_plugin_with_invert(
                route.gain_db,
                route.polarity_inverted,
            )];
            if let Some(f) = route.high_pass_hz.or(route.low_pass_hz) {
                route_plugins.push(roomeq_engine::output::create_crossover_plugin(
                    &route.crossover_type,
                    f,
                    if route.high_pass_hz.is_some() {
                        "high"
                    } else {
                        "low"
                    },
                ));
            }
            if route.delay_ms.abs() > 0.001 {
                route_plugins.push(roomeq_engine::output::create_delay_plugin(route.delay_ms));
            }
            let branch = process_branch(
                &route.destination,
                raw,
                &config.optimizer.upper_band_acoustic_bounds,
                partition,
                seat,
                &grid,
                |curve| {
                    let curve = apply(
                        result,
                        input,
                        stage(pre, "pre_route"),
                        curve,
                        baseline,
                        fs,
                        dir,
                    )?;
                    let curve =
                        apply(result, input, route_plugins.clone(), &curve, false, fs, dir)?;
                    let mut curve = apply(
                        result,
                        post_owner,
                        stage(post, "post_route"),
                        &curve,
                        baseline,
                        fs,
                        dir,
                    )?;
                    if let Some(driver) = post.drivers.as_ref().and_then(|drivers| {
                        drivers
                            .iter()
                            .find(|driver| driver.name == route.destination)
                    }) {
                        curve = apply(
                            result,
                            post_owner,
                            driver.plugins.clone(),
                            &curve,
                            baseline,
                            fs,
                            dir,
                        )?;
                    }
                    Ok(curve)
                },
            )?;
            branches.push(branch);
            outputs.push(route.destination.clone());
        }
        outputs.sort();
        outputs.dedup();
        return Ok((sum_branches(&branches, config.optimizer.min_freq, config.optimizer.max_freq)?, outputs));
    }
    let chain = result
        .channels
        .get(input)
        .ok_or_else(|| invalid("missing channel for final seat replay"))?;
    if let Some(drivers) = &chain.drivers {
        let mut branches = Vec::new();
        let mut outputs = Vec::new();
        for driver in drivers {
            branches.push(process_branch(
                &driver.name,
                measured(physical, &driver.name, seat)?,
                &config.optimizer.upper_band_acoustic_bounds,
                partition,
                seat,
                &grid,
                |curve| {
                    let curve = apply(
                        result,
                        input,
                        driver.plugins.clone(),
                        curve,
                        baseline,
                        fs,
                        dir,
                    )?;
                    apply(
                        result,
                        input,
                        chain.plugins.clone(),
                        &curve,
                        baseline,
                        fs,
                        dir,
                    )
                },
            )?);
            outputs.push(driver.name.clone());
        }
        return Ok((sum_branches(&branches, config.optimizer.min_freq, config.optimizer.max_freq)?, outputs));
    }
    Ok((
        Summed {
            curve: apply(
                result,
                input,
                chain.plugins.clone(),
                measured(physical, input, seat)?,
                baseline,
                fs,
                dir,
            )?,
            support: vec![],
        },
        vec![input.into()],
    ))
}

pub(super) fn validate_final_seats(
    result: &mut RoomOptimizationResult,
    captures: &[Capture],
    held_out: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
) -> Result<()> {
    crate::export::validate_final_routed_stage_ownership(result)?;
    if !captures.iter().any(|c| c.curves.len() > 1) && held_out.is_empty() {
        return Ok(());
    }
    if captures.iter().any(|capture| {
        let key = config
            .system
            .as_ref()
            .and_then(|system| system.speakers.get(&capture.channel))
            .unwrap_or(&capture.channel);
        matches!(config.speakers.get(key), Some(SpeakerConfig::Group(_)))
    }) {
        return Err(invalid(
            "final-seat replay of legacy driver groups requires explicit topology IDs",
        ));
    }
    if result.metadata.ctc.is_some() {
        return Err(invalid(
            "final-seat replay requires ear-identified CTC transfer measurements",
        ));
    }
    let training = physical_captures(captures, result)?;
    let held: BTreeMap<_, _> = held_out
        .iter()
        .map(|(name, curves)| (name.clone(), curves.clone()))
        .collect();
    let mut evidence = Vec::new();
    let mut training_scores = Vec::new();
    let mut held_scores = Vec::new();
    let mut inputs: Vec<_> = if let Some(graph) = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|b| b.routing_graph.as_ref())
    {
        graph.input_channels.clone()
    } else {
        captures.iter().map(|c| c.channel.clone()).collect()
    };
    inputs.sort();
    inputs.dedup();
    for (partition, physical) in [("training", &training), ("held_out", &held)] {
        if physical.is_empty() {
            continue;
        }
        for curves in physical.values() {
            for curve in curves {
                curve.validate("final-seat replay")?;
            }
        }
        if partition == "held_out" {
            for output in physical.keys() {
                if !training.contains_key(output) {
                    return Err(invalid(format!(
                        "unknown held-out physical output '{output}'"
                    )));
                }
            }
        }
        for input in &inputs {
            // Independent channels may have distinct held-out sets. Routed
            // inputs require all contributing physical outputs for each seat.
            let routed = result
                .metadata
                .bass_management
                .as_ref()
                .and_then(|b| b.routing_graph.as_ref())
                .is_some();
            if partition == "held_out"
                && !routed
                && !physical.contains_key(input)
                && result
                    .channels
                    .get(input)
                    .and_then(|chain| chain.drivers.as_ref())
                    .is_none_or(|drivers| {
                        drivers
                            .iter()
                            .all(|driver| !physical.contains_key(&driver.name))
                    })
            {
                continue;
            }
            let seats = if routed {
                physical.values().map(Vec::len).max().unwrap_or(0)
            } else if let Some(drivers) =
                result.channels.get(input).and_then(|c| c.drivers.as_ref())
            {
                drivers
                    .iter()
                    .filter_map(|d| physical.get(&d.name))
                    .map(Vec::len)
                    .max()
                    .unwrap_or(0)
            } else {
                physical.get(input).map_or(0, Vec::len)
            };
            if seats == 0 {
                return Err(invalid(format!("no final-seat evidence for '{input}'")));
            }
            for seat in 0..seats {
                let context = ReplayContext {
                    config,
                    partition,
                    fs,
                    dir,
                };
                let (pre, outputs) = replay(result, physical, input, seat, true, &context)?;
                let (post, _) = replay(result, physical, input, seat, false, &context)?;
                let uncertainty_db = pre.uncertainty_db() + post.uncertainty_db();
                let pre_support = pre.support;
                let post_support = post.support;
                let pre = pre.curve;
                let post = post.curve;
                let target = result
                    .channels
                    .get(input)
                    .and_then(|c| c.target_curve.clone())
                    .map(Curve::from);
                let lo = config.optimizer.min_freq.max(pre.freq[0]).max(post.freq[0]);
                let hi = config
                    .optimizer
                    .max_freq
                    .min(*pre.freq.last().unwrap())
                    .min(*post.freq.last().unwrap());
                let mut score =
                    roomeq_engine::quality::evaluate_acoustic_quality_with_permitted_gain(
                        &[pre],
                        &[post],
                        &[],
                        &[],
                        target.as_ref(),
                        roomeq_engine::quality::QualityEvaluationConfig {
                            min_freq_hz: lo,
                            max_freq_hz: hi,
                            schroeder_hz: None,
                            normalize_level: true,
                        },
                        Default::default(),
                        config
                            .optimizer
                            .permitted_output_gain_db
                            .get(input)
                            .copied()
                            .unwrap_or(0.0),
                    )
                    .map_err(invalid)?;
                // Each evaluator invocation contains one seat, so its local index
                // is zero. Restore physical capture identity before aggregation.
                for output in &mut score.useful_output {
                    output.logical_input = Some(input.clone());
                    output.partition = partition.into();
                    output.seat_index = seat;
                }
                evidence.push(FinalSeatEvaluation {
                    partition: partition.into(),
                    logical_input: input.clone(),
                    seat_index: seat,
                    seat_label: if partition == "training" {
                        training_seat_label(captures, input, seat)
                    } else {
                        None
                    },
                    physical_outputs: outputs,
                    pre_summation_support: pre_support,
                    post_summation_support: post_support,
                    unassessed_bands_hz: {
                        let mut bands = Vec::new();
                        if lo > config.optimizer.min_freq * (1.0 + 1e-9) {
                            bands.push([config.optimizer.min_freq, lo]);
                        }
                        if hi < config.optimizer.max_freq * (1.0 - 1e-9) {
                            bands.push([hi, config.optimizer.max_freq]);
                        }
                        bands
                    },
                    evaluated_band_hz: score.evaluated_band_hz,
                    pre_weighted_rms_db: score.training.pre_weighted_rms_median_db,
                    post_weighted_rms_db: score.training.post_weighted_rms_median_db,
                    improvement_db: score.training.worst_position_improvement_db,
                    improvement_lower_bound_db: score.training.worst_position_improvement_db
                        - uncertainty_db,
                });
                if partition == "training" {
                    training_scores.push(score);
                } else {
                    held_scores.push(score);
                }
            }
        }
    }
    let mut score = super::room_optimization_result::aggregate_runtime_quality(
        &training_scores,
        Default::default(),
        config.optimizer.min_freq,
        config.optimizer.max_freq,
    )
    .ok_or_else(|| invalid("final-seat training evidence unavailable"))?;
    score.held_out = super::room_optimization_result::aggregate_runtime_quality(
        &held_scores,
        Default::default(),
        config.optimizer.min_freq,
        config.optimizer.max_freq,
    )
    .map(|s| s.training);
    score.useful_output.extend(
        held_scores
            .iter()
            .flat_map(|s| s.useful_output.iter().cloned()),
    );
    score.final_seats = evidence;
    score.evaluated_band_hz = [
        score
            .final_seats
            .iter()
            .map(|s| s.evaluated_band_hz[0])
            .fold(f64::INFINITY, f64::min),
        score
            .final_seats
            .iter()
            .map(|s| s.evaluated_band_hz[1])
            .fold(0.0_f64, f64::max),
    ];
    score.measurement_overlap_hz = [
        score
            .final_seats
            .iter()
            .map(|s| s.evaluated_band_hz[0])
            .fold(0.0_f64, f64::max),
        score
            .final_seats
            .iter()
            .map(|s| s.evaluated_band_hz[1])
            .fold(f64::INFINITY, f64::min),
    ];
    if score.measurement_overlap_hz[0] >= score.measurement_overlap_hz[1] {
        return Err(invalid(
            "final-seat scorecard has no common supported frequency band",
        ));
    }
    let report = result
        .metadata
        .correction_acceptance
        .as_mut()
        .ok_or_else(|| invalid("final-seat acceptance report unavailable"))?;
    if let Some(previous) = &report.acoustic_quality {
        score.temporal = previous.temporal;
    }
    let budget = report
        .runtime_policy
        .as_ref()
        .ok_or_else(|| invalid("final-seat runtime budget unavailable"))?
        .max_worst_position_regression_db;
    let failed = score
        .final_seats
        .iter()
        .find(|s| {
            !s.improvement_lower_bound_db.is_finite() || s.improvement_lower_bound_db < -budget
        })
        .map(|s| {
            format!(
                "{} '{}' seat {} regressed {:.3} dB beyond {:.3} dB budget",
                s.partition, s.logical_input, s.seat_index, -s.improvement_lower_bound_db, budget
            )
        });
    let output_budget =
        roomeq_engine::quality::QualityGatePolicy::default().max_unexplained_output_loss_db;
    let output_failed = score.useful_output.iter().find_map(|output| {
        let loss = output.unexplained_loss_rms_db
            .max(output.bass_unexplained_loss_rms_db.unwrap_or(0.0));
        (!loss.is_finite() || loss > output_budget).then(|| {
            format!(
                "{} '{}' seat {} lost {:.3} dB useful output beyond {:.3} dB budget (permitted gain {:.3} dB)",
                output.partition, output.logical_input.as_deref().unwrap_or("unknown"),
                output.seat_index, loss, output_budget, output.permitted_gain_db
            )
        })
    });
    report.acoustic_quality = Some(score);
    if output_failed.is_some() {
        report
            .violations
            .push("unexplained_useful_output_loss".into());
    }
    if failed.is_some() {
        report.violations.push("worst_position_regressed".into());
    }
    let failures: Vec<_> = failed.into_iter().chain(output_failed).collect();
    if !failures.is_empty() {
        report.accepted = false;
        report.decision = roomeq_model::CorrectionDecision::Rejected;
        report.violations.sort();
        report.violations.dedup();
        return Err(AutoeqError::OptimizationFailed { message: failures.join("; ") });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn training_labels_follow_source_and_original_index_without_inference() {
        let captures = vec![
            Capture {
                channel: "left".into(),
                driver: None,
                curves: vec![],
                seat_labels: Some(vec!["front".into(), "rear".into()]),
            },
            Capture {
                channel: "right".into(),
                driver: None,
                curves: vec![],
                seat_labels: None,
            },
        ];
        assert_eq!(
            training_seat_label(&captures, "left", 1).as_deref(),
            Some("rear")
        );
        assert_eq!(
            training_seat_label(&captures, "left", 0).as_deref(),
            Some("front")
        );
        assert_eq!(training_seat_label(&captures, "left", 2), None);
        assert_eq!(training_seat_label(&captures, "right", 0), None);
        assert_eq!(training_seat_label(&captures, "missing", 0), None);
    }
    use roomeq_model::{CorrectionAcceptancePolicy, RuntimeAcceptancePolicy, RuntimeOutputClass};

    fn fixture() -> (RoomOptimizationResult, Curve, Curve) {
        let mut result = crate::test_fixtures::single_channel_room_result("left");
        let mut flat = result.channel_results["left"].initial_curve.clone();
        flat.spl.fill(80.0);
        let mut peak = flat.clone();
        for (f, level) in peak.freq.iter().zip(peak.spl.iter_mut()) {
            *level += 6.0 * (-((*f - 120.0) / 25.0).powi(2)).exp();
        }
        let channel = result.channel_results.get_mut("left").unwrap();
        channel.initial_curve = peak.clone();
        channel.final_curve = flat.clone();
        result.channels.get_mut("left").unwrap().plugins = vec![
            roomeq_engine::output::create_eq_plugin(&[math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                120.0,
                48_000.0,
                1.0,
                -6.0,
            )]),
        ];
        let mut acceptance = roomeq_engine::quality::evaluate_correction_acceptance(
            &peak,
            &flat,
            &flat,
            None,
            CorrectionAcceptancePolicy::RuntimeSafety,
        )
        .unwrap();
        acceptance.runtime_policy = Some(RuntimeAcceptancePolicy::for_output_class(
            RuntimeOutputClass::LowLatencyIir,
        ));
        result.metadata.correction_acceptance = Some(acceptance);
        (result, peak, flat)
    }

    #[test]
    fn final_training_seat_replay_rejects_hidden_regression_after_post_pass() {
        let (mut result, peak, flat) = fixture();
        let captures = vec![Capture {
            channel: "left".into(),
            driver: None,
            seat_labels: None,
            curves: vec![peak, flat],
        }];
        let error = validate_final_seats(
            &mut result,
            &captures,
            &HashMap::new(),
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
        )
        .unwrap_err();
        assert!(error.to_string().contains("seat 1"), "{error}");
        let report = result.metadata.correction_acceptance.as_ref().unwrap();
        assert!(!report.accepted);
        assert_eq!(report.decision, roomeq_model::CorrectionDecision::Rejected);
        assert_eq!(serde_json::to_value(report).unwrap()["decision"], "rejected");
        let seats = &report.acoustic_quality.as_ref().unwrap().final_seats;
        assert_eq!(seats.len(), 2);
        assert_eq!(seats[1].physical_outputs, vec!["left"]);
        assert!(seats[1].improvement_db < 0.0);
    }

    #[test]
    fn final_seat_rejection_retains_both_shape_and_output_failures() {
        let (mut result, _, flat) = fixture();
        result.channels.get_mut("left").unwrap().plugins = vec![
            roomeq_engine::output::create_eq_plugin(&[math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak, 120.0, 48_000.0, 0.5, -30.0,
            )]),
        ];
        let captures = vec![Capture { channel: "left".into(), driver: None,
            seat_labels: None, curves: vec![flat.clone(), flat] }];
        let error = validate_final_seats(&mut result, &captures, &HashMap::new(),
            &RoomConfig::default(), 48_000.0, Path::new(".")).unwrap_err();
        let report = result.metadata.correction_acceptance.as_ref().unwrap();
        for reason in ["worst_position_regressed", "unexplained_useful_output_loss"] {
            assert!(report.violations.iter().any(|v| v == reason), "missing {reason}: {report:?}");
        }
        assert!(error.to_string().contains("regressed"));
        assert!(error.to_string().contains("useful output"));
        assert_eq!(report.decision, roomeq_model::CorrectionDecision::Rejected);
    }

    #[test]
    fn final_seat_replay_rejects_sidecar_conflicting_with_retained_fir() {
        let directory = tempfile::tempdir().unwrap();
        let (mut result, _, flat) = fixture();
        let plugins = vec![roomeq_engine::output::create_convolution_plugin("retained.wav")];
        result.channels.get_mut("left").unwrap().plugins = plugins.clone();
        let tap = 0.123456789_f64;
        result.channel_results.get_mut("left").unwrap().fir_coeffs = Some(vec![tap]);
        let write = |value: f32| {
            let mut writer = hound::WavWriter::create(directory.path().join("retained.wav"), hound::WavSpec {
                channels: 1, sample_rate: 48000, bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            }).unwrap();
            writer.write_sample(value).unwrap();
            writer.finalize().unwrap();
        };
        write(tap as f32);
        assert!(super::apply(&result, "left", plugins.clone(), &flat, false, 48000.0, directory.path()).is_ok(),
            "normal float32 serialization rounding must remain valid");
        write(0.5);
        let outcome = super::apply(&result, "left", plugins, &flat, false, 48000.0, directory.path());
        assert!(outcome.is_err(), "stale sidecar silently replaced retained FIR evidence");
        assert!(outcome.unwrap_err().to_string().contains("retained FIR coefficients"));
    }

    #[test]
    fn final_seat_output_loss_requires_an_explicit_gain_allowance() {
        for gain in [-40.0, -12.0, -6.0] {
            for authorized in [false, true] {
                let (mut result, _, flat) = fixture();
                result.channels.get_mut("left").unwrap().plugins =
                    vec![roomeq_engine::output::create_convolution_plugin(
                        "useful-output-test.wav",
                    )];
                result.channel_results.get_mut("left").unwrap().fir_coeffs =
                    Some(vec![10.0_f64.powf(gain / 20.0)]);
                let captures = vec![Capture {
                    channel: "left".into(),
                    driver: None,
                    seat_labels: None,
                    curves: vec![flat.clone(), flat.clone()],
                }];
                let held = HashMap::from([("left".into(), vec![flat])]);
                let mut config = RoomConfig::default();
                if authorized {
                    config
                        .optimizer
                        .permitted_output_gain_db
                        .insert("left".into(), gain);
                }
                let outcome = validate_final_seats(
                    &mut result,
                    &captures,
                    &held,
                    &config,
                    48_000.0,
                    Path::new("."),
                );
                assert_eq!(
                    outcome.is_ok(),
                    authorized,
                    "{gain} {authorized}: {outcome:?}"
                );
                let report = result.metadata.correction_acceptance.as_ref().unwrap();
                let evidence = &report.acoustic_quality.as_ref().unwrap().useful_output;
                assert_eq!(evidence.len(), 3);
                assert_eq!(evidence[1].seat_index, 1);
                assert_eq!(evidence[2].partition, "held_out");
                for seat in evidence {
                    assert!((seat.mean_level_change_db - gain).abs() < 1e-8);
                    assert!(
                        (seat.unexplained_loss_rms_db - if authorized { 0.0 } else { -gain }).abs()
                            < 1e-8
                    );
                }
            if !authorized {
                assert!(!report.accepted);
                assert_eq!(report.decision, roomeq_model::CorrectionDecision::Rejected);
                assert_eq!(serde_json::to_value(report).unwrap()["decision"], "rejected");
                    assert!(
                        report
                            .violations
                            .contains(&"unexplained_useful_output_loss".into())
                    );
                }
            }
        }
    }

    #[test]
    fn held_out_replay_uses_final_chain_and_preserves_partition_identity() {
        let (mut result, peak, flat) = fixture();
        let captures = vec![Capture {
            channel: "left".into(),
            driver: None,
            seat_labels: None,
            curves: vec![peak],
        }];
        let held = HashMap::from([("left".into(), vec![flat])]);
        assert!(
            validate_final_seats(
                &mut result,
                &captures,
                &held,
                &RoomConfig::default(),
                48_000.0,
                Path::new(".")
            )
            .is_err()
        );
        let score = result
            .metadata
            .correction_acceptance
            .as_ref()
            .unwrap()
            .acoustic_quality
            .as_ref()
            .unwrap();
        assert_eq!(score.final_seats[1].partition, "held_out");
        assert_eq!(score.held_out.as_ref().unwrap().curve_count, 1);
        assert_eq!(score.useful_output.len(), 2);
        assert_eq!(score.useful_output[0].partition, "training");
        assert_eq!(score.useful_output[1].partition, "held_out");
        assert_eq!(
            score.useful_output[1].logical_input.as_deref(),
            Some("left")
        );
    }

    #[test]
    fn native_capture_is_not_reduced_and_missing_seats_are_not_broadcast() {
        let grid: Vec<_> = (0..=1000).map(|i| 20.0 + 0.18 * i as f64).collect();
        let flat = Curve {
            freq: grid.clone().into(),
            spl: grid
                .iter()
                .map(|f| 80.0 + 12.0 * (-0.5 * ((f - 82.0) / 0.5).powi(2)).exp())
                .collect(),
            ..Default::default()
        };
        let physical = BTreeMap::from([("sub".into(), vec![flat.clone()])]);
        assert!(measured(&physical, "sub", 1).is_err());
        let config = RoomConfig {
            system: None,
            speakers: HashMap::from([(
                "left".into(),
                SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![
                    flat.clone(),
                    flat.clone(),
                ])),
            )]),
            ..Default::default()
        };
        let captured = capture_training(&config).unwrap();
        assert_eq!(captured[0].curves[1].freq, flat.freq);
        assert_eq!(captured[0].curves[1].spl, flat.spl);
        assert_eq!(captured[0].curves[1].freq.len(), 1001);
    }

    #[test]
    fn named_seat_permutation_is_not_silently_coherently_summed() {
        let (result, _, flat) = fixture();
        let captures = vec![
            Capture {
                channel: "left".into(),
                driver: None,
                curves: vec![flat.clone(), flat.clone()],
                seat_labels: Some(vec!["front".into(), "rear".into()]),
            },
            Capture {
                channel: "left".into(),
                driver: None,
                curves: vec![flat.clone(), flat],
                seat_labels: Some(vec!["rear".into(), "front".into()]),
            },
        ];
        assert!(
            physical_captures(&captures, &result)
                .unwrap_err()
                .to_string()
                .contains("labels differ")
        );
    }

    #[test]
    fn branch_sum_preserves_each_native_grid_and_rejects_unknown_phase() {
        let (_, _, mut flat) = fixture();
        flat.phase = Some(ndarray::Array1::zeros(flat.freq.len()));
        let mut other = flat.clone();
        for i in 1..other.freq.len() - 1 {
            other.freq[i] *= 1.001;
        }
        let combined = sum(&[flat.clone(), other.clone()]).unwrap();
        assert!(combined.freq.len() > flat.freq.len());
        other.phase = None;
        assert!(sum(&[flat, other]).is_err());
    }

    fn routed_fixture() -> (RoomOptimizationResult, RoomConfig, Curve) {
        use roomeq_model::{
            BassManagementRoute, BassManagementRoutingGraph, SystemConfig, SystemModel,
        };
        let (mut result, _, mut flat) = fixture();
        flat.phase = Some(ndarray::Array1::zeros(flat.freq.len()));
        let sub_result = crate::test_fixtures::single_channel_room_result("sub");
        result
            .channels
            .insert("sub".into(), sub_result.channels["sub"].clone());
        result
            .channel_results
            .insert("sub".into(), sub_result.channel_results["sub"].clone());
        result.channels.get_mut("left").unwrap().plugins.clear();
        let mut post = roomeq_engine::output::create_gain_plugin(-3.0);
        post.parameters["room_eq_stage"] = serde_json::json!("post_route");
        result.channels.get_mut("sub").unwrap().plugins = vec![post];
        let config = RoomConfig {
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: HashMap::from([
                    ("L".into(), "left".into()),
                    ("LFE".into(), "sub".into()),
                ]),
                subwoofers: Some(roomeq_model::SubwooferSystemConfig {
                    config: Default::default(),
                    crossover: None,
                    mapping: HashMap::new(),
                }),
                bass_management: Some(roomeq_model::BassManagementConfig {
                    enabled: true,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut report =
            roomeq_engine::home_cinema::bass_management_report(&config, None, false).unwrap();
        let route = |dest: &str, gain: f64, delay: f64| BassManagementRoute {
            group_id: None,
            source_channel: "left".into(),
            source_index: 0,
            destination: dest.into(),
            destination_index: if dest == "left" { 0 } else { 1 },
            pre_chain_channel: Some("left".into()),
            post_chain_channel: Some(dest.into()),
            route_kind: if dest == "left" {
                "main_highpass"
            } else {
                "redirected_bass_lowpass_to_sub"
            }
            .into(),
            crossover_type: "LR24".into(),
            high_pass_hz: None,
            low_pass_hz: None,
            gain_db: gain,
            gain_linear: 10.0_f64.powf(gain / 20.0),
            matrix_gain: 10.0_f64.powf(gain / 20.0),
            delay_ms: delay,
            polarity_inverted: false,
        };
        report.routing_graph = Some(BassManagementRoutingGraph {
            physical_sub_output: "sub".into(),
            input_channels: vec!["left".into()],
            output_channels: vec!["left".into(), "sub".into()],
            routes: vec![route("left", 0.0, 0.0), route("sub", -6.0, 2.0)],
            matrix: None,
            input_trim_db: HashMap::new(),
            advisories: Vec::new(),
        });
        result.metadata.bass_management = Some(report);
        (result, config, flat)
    }

    #[test]
    fn routed_finalization_and_replay_reject_unowned_channel_stages() {
        for tag in [None, Some(serde_json::json!(null)), Some(serde_json::json!(42)),
            Some(serde_json::json!("post_rout"))] {
            let (mut result, config, flat) = routed_fixture();
            let mut plugin = roomeq_engine::output::create_gain_plugin(-20.0);
            if let Some(tag) = tag { plugin.parameters["room_eq_stage"] = tag; }
            result.channels.get_mut("left").unwrap().plugins.push(plugin);
            let physical = BTreeMap::from([
                ("left".into(), vec![flat.clone()]), ("sub".into(), vec![flat]),
            ]);
            assert!(replay_final_physical_seat(
                &result, &physical, "left", 0, &config, 48_000.0, Path::new("."), "training",
            ).is_err(), "unowned gain disappeared from routed playback evidence");
            let store = autoeq_artifacts::MemoryArtifactStore::new();
            assert!(crate::export::bind_final_convolution_artifacts(
                &mut result, Path::new("."), &store, 48_000.0,
            ).is_err(), "malformed routed graph was finalized");
        }
    }

    #[test]
    fn channel_matching_is_applied_to_the_complete_routed_source() {
        let (mut result, config, flat) = routed_fixture();
        let physical = BTreeMap::from([
            ("left".into(), vec![flat.clone()]), ("sub".into(), vec![flat]),
        ]);
        let before = replay_final_physical_seat(
            &result, &physical, "left", 0, &config, 48_000.0, Path::new("."), "training",
        ).unwrap().delivered;
        let filters = vec![math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak, 120.0, 48_000.0, 1.0, 3.0,
        )];
        let correction = roomeq_engine::spectral_align::ChannelMatchingResult {
            channel_name: "left".into(), filters: filters.clone(),
        };
        super::super::reports::apply_channel_matching_correction(&mut result, &correction, 48_000.0);
        let after = replay_final_physical_seat(
            &result, &physical, "left", 0, &config, 48_000.0, Path::new("."), "training",
        ).unwrap().delivered;
        let response = roomeq_engine::response::compute_peq_complex_response(&filters, &before.freq, 48_000.0);
        let expected = roomeq_engine::response::apply_complex_response(&before, &response);
        let maximum_error = after.spl.iter().zip(expected.spl.iter())
            .map(|(actual, expected)| (actual - expected).abs()).fold(0.0_f64, f64::max);
        assert!(maximum_error < 1e-8, "routed matching transfer differs by {maximum_error} dB");
    }

    #[test]
    fn qualified_sub_bound_keeps_full_main_band_and_rejects_upper_midrange_regression() {
        let (mut result, _, _) = routed_fixture();
        let main = Curve {
            freq: ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 1001),
            spl: ndarray::Array1::from_elem(1001, 80.0),
            phase: Some(ndarray::Array1::zeros(1001)),
            ..Default::default()
        };
        let sub_grid = ndarray::Array1::linspace(20.0, 200.0, 1001);
        let sub = Curve {
            spl: sub_grid.mapv(|f: f64| 70.0 - 0.5 * (f - 100.0).max(0.0)),
            freq: sub_grid,
            phase: Some(ndarray::Array1::zeros(1001)),
            ..Default::default()
        };
        let captures = vec![
            Capture {
                channel: "left".into(),
                driver: None,
                curves: vec![main.clone(), main],
                seat_labels: Some(vec!["front".into(), "rear".into()]),
            },
            Capture {
                channel: "sub".into(),
                driver: None,
                curves: vec![sub.clone(), sub],
                seat_labels: Some(vec!["front".into(), "rear".into()]),
            },
        ];
        let mut config = RoomConfig::default();
        config.optimizer.upper_band_acoustic_bounds.insert(
            "sub".into(),
            (0..2)
                .map(|seat_index| roomeq_model::UpperBandAcousticBound {
                    partition: "training".into(),
                    seat_index,
                    band_hz: [200.0, 20_000.0],
                    max_spl_db: 20.0,
                    evidence_id: format!("analytic-qualified-stopband-seat-{seat_index}"),
                })
                .collect(),
        );
        config.optimizer.min_freq = 20.0;
        config.optimizer.max_freq = 20_000.0;
        validate_final_seats(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48_000.0,
            Path::new("."),
        )
        .unwrap();
        let score = result
            .metadata
            .correction_acceptance
            .as_ref()
            .unwrap()
            .acoustic_quality
            .as_ref()
            .unwrap();
        assert_eq!(score.final_seats.len(), 2);
        assert_eq!(score.useful_output.len(), 2);
        assert_eq!(score.useful_output[0].seat_index, 0);
        assert_eq!(score.useful_output[1].seat_index, 1);
        for seat in &score.final_seats {
            assert!((seat.evaluated_band_hz[0] - 20.0).abs() < 1e-9);
            assert_eq!(seat.evaluated_band_hz[1], 20_000.0);
            assert!(seat.unassessed_bands_hz.iter().all(|b| b[1] - b[0] < 1e-9));
            assert_eq!(seat.pre_summation_support[0].physical_output, "sub");
            assert!(seat.pre_summation_support[0].max_magnitude_uncertainty_db < 0.01);
            let evidence = &seat.pre_summation_support[0];
            assert!(evidence.max_phase_uncertainty_deg > 0.0);
            assert!(
                (evidence.max_phase_uncertainty_deg
                    - evidence.max_sum_omitted_amplitude_ratio.asin().to_degrees())
                .abs()
                    < 1e-12
            );
            let serialized = serde_json::to_value(seat).unwrap();
            assert!(
                serialized["pre_summation_support"][0]["max_phase_uncertainty_deg"].is_number()
            );
            assert!(seat.improvement_lower_bound_db <= seat.improvement_db);
        }
        let mut harmful =
            roomeq_engine::output::create_eq_plugin(&[math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                2000.0,
                48_000.0,
                1.0,
                -12.0,
            )]);
        harmful.parameters["room_eq_stage"] = serde_json::json!("post_route");
        result
            .channels
            .get_mut("left")
            .unwrap()
            .plugins
            .push(harmful);
        let error = validate_final_seats(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48_000.0,
            Path::new("."),
        )
        .unwrap_err();
        assert!(error.to_string().contains("regressed"), "{error}");
        let report = result.metadata.correction_acceptance.as_ref().unwrap();
        assert!(!report.accepted);
        assert_eq!(
            report.acoustic_quality.as_ref().unwrap().final_seats[0].evaluated_band_hz[1],
            20_000.0
        );
        result.channels.get_mut("left").unwrap().plugins.clear();
        let mut upper_boost =
            roomeq_engine::output::create_eq_plugin(&[math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Highshelf,
                2000.0,
                48_000.0,
                0.7,
                60.0,
            )]);
        upper_boost.parameters["room_eq_stage"] = serde_json::json!("post_route");
        result
            .channels
            .get_mut("sub")
            .unwrap()
            .plugins
            .push(upper_boost);
        let error = validate_final_seats(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48_000.0,
            Path::new("."),
        )
        .unwrap_err();
        assert!(error.to_string().contains("uncertainty budget"), "{error}");
    }

    #[test]
    fn significant_or_unqualified_unmeasured_sub_is_insufficient_evidence() {
        let (result, _, _) = routed_fixture();
        let main = Curve {
            freq: vec![20.0, 100.0, 200.0, 1000.0, 20_000.0].into(),
            spl: vec![80.0; 5].into(),
            phase: Some(vec![0.0; 5].into()),
            ..Default::default()
        };
        let sub = Curve {
            freq: vec![20.0, 100.0, 200.0].into(),
            spl: vec![80.0; 3].into(),
            phase: Some(vec![0.0; 3].into()),
            ..Default::default()
        };
        let physical = BTreeMap::from([("left".into(), vec![main]), ("sub".into(), vec![sub])]);
        let mut config = RoomConfig::default();
        let check = |config: &RoomConfig| {
            replay(
                &result,
                &physical,
                "left",
                0,
                false,
                &ReplayContext {
                    config,
                    partition: "training",
                    fs: 48_000.0,
                    dir: Path::new("."),
                },
            )
            .err()
            .unwrap()
            .to_string()
        };
        assert!(check(&config).contains("insufficient summation evidence"));
        config.optimizer.upper_band_acoustic_bounds.insert(
            "sub".into(),
            vec![roomeq_model::UpperBandAcousticBound {
                partition: "training".into(),
                seat_index: 0,
                band_hz: [200.0, 20_000.0],
                max_spl_db: 80.0,
                evidence_id: "analytic-energetic-tail".into(),
            }],
        );
        assert!(check(&config).contains("uncertainty budget"));
        config
            .optimizer
            .upper_band_acoustic_bounds
            .get_mut("sub")
            .unwrap()[0]
            .max_spl_db = 20.0;
        assert!(check(&config).contains("contradicts measured"));
    }

    #[test]
    fn branch_sum_keeps_a_narrow_native_bass_cancellation() {
        let main = Curve {
            freq: vec![20.0, 100.0, 200.0].into(),
            spl: vec![80.0; 3].into(),
            phase: Some(vec![0.0; 3].into()),
            ..Default::default()
        };
        let freq = ndarray::Array1::linspace(20.0, 200.0, 1801);
        let mut phase = ndarray::Array1::zeros(freq.len());
        phase[633] = 180.0;
        let sub = Curve {
            freq,
            spl: ndarray::Array1::from_elem(1801, 80.0),
            phase: Some(phase),
            ..Default::default()
        };
        let combined = sum(&[main, sub]).unwrap();
        let index = combined
            .freq
            .iter()
            .position(|f| (*f - 83.3).abs() < 1e-9)
            .unwrap();
        assert!(combined.spl[index] < -100.0);
        assert_eq!(combined.freq.len(), 1801);
    }

    #[test]
    fn routed_replay_uses_each_seat_and_applies_route_delay_gain_and_post_eq_once() {
        let (result, _, flat) = routed_fixture();
        let mut opposite = flat.clone();
        opposite.phase.as_mut().unwrap().fill(180.0);
        let physical = BTreeMap::from([
            ("left".into(), vec![flat.clone(), flat.clone()]),
            ("sub".into(), vec![flat.clone(), opposite]),
        ]);
        let config = RoomConfig::default();
        let context = ReplayContext {
            config: &config,
            partition: "training",
            fs: 48_000.0,
            dir: Path::new("."),
        };
        for seat in 0..2 {
            let (response, outputs) =
                replay(&result, &physical, "left", seat, false, &context).unwrap();
            let response = response.curve;
            assert_eq!(outputs, vec!["left", "sub"]);
            for (&f, &db) in response.freq.iter().zip(response.spl.iter()) {
                let sub = num_complex::Complex64::from_polar(
                    10.0_f64.powf(-9.0 / 20.0),
                    -2.0 * std::f64::consts::PI * f * 0.002 + seat as f64 * std::f64::consts::PI,
                );
                let expected =
                    80.0 + 20.0 * (num_complex::Complex64::new(1.0, 0.0) + sub).norm().log10();
                assert!(
                    (db - expected).abs() < 1e-6,
                    "seat {seat}, {f} Hz: {db} vs {expected}"
                );
            }
        }
        let mut phase_unknown = physical.clone();
        phase_unknown.get_mut("sub").unwrap()[0].phase = None;
        assert!(replay(&result, &phase_unknown, "left", 0, false, &context,).is_err());
        let mut missing = physical;
        missing.get_mut("sub").unwrap().pop();
        assert!(replay(&result, &missing, "left", 1, false, &context,).is_err());
    }
}
