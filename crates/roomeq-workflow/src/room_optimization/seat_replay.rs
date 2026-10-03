//! Raw, position-identified evidence survives optimization until final replay.
use super::RoomOptimizationResult;
use roomeq_model::{
    AutoeqError, ChannelDspChain, CoherentTimingEvidence, Curve, FinalSeatEvaluation,
    MeasurementSource, PluginConfigWrapper, Result, RoomConfig, SpeakerConfig,
};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
#[cfg(test)]
mod asymmetric;
mod bounded_sum;
mod intake;
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

/// Whether final replay would need coherent branch summation but one or more
/// measured branches carry magnitude only. Keep this check in the capture
/// module so callers cannot bypass the phase-evidence contract by inspecting
/// private capture representation.
pub(super) fn has_unmeasured_multi_branch_phase(captures: &[Capture]) -> bool {
    captures.iter().any(|capture| {
        capture.driver.is_some() && capture.curves.iter().any(|curve| curve.phase.is_none())
    })
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

/// Validate raw quality evidence before crossover optimization or seat replay.
/// Unknown quality is reported separately from explicitly unreliable evidence.
pub(crate) fn crossover_phase_advisories(
    config: &RoomConfig,
    band: (f64, f64),
) -> Result<Vec<String>> {
    let captures = capture_training(config)?;
    let mut advisories = std::collections::BTreeSet::new();
    let mut seat_order = None;
    for capture in &captures {
        if let Some(labels) = capture
            .seat_labels
            .as_ref()
            .filter(|labels| labels.len() > 1)
        {
            if seat_order.is_some_and(|previous| previous != labels) {
                return Err(invalid(
                    "crossover phase captures must use identical seat order",
                ));
            }
            seat_order = Some(labels);
        }
        for (seat, curve) in capture.curves.iter().enumerate() {
            let evidence = roomeq_engine::bass_phase_confidence::crossover_phase_advisories(
                curve,
                band,
                config.recording_config.as_ref(),
            )
            .map_err(|error| {
                invalid(format!(
                    "channel '{}' seat {seat}: {error}",
                    capture.channel
                ))
            })?;
            advisories.extend(evidence.into_iter().map(str::to_owned));
        }
    }
    Ok(advisories.into_iter().collect())
}

fn invalid(message: impl Into<String>) -> AutoeqError {
    AutoeqError::InvalidMeasurement {
        message: message.into(),
    }
}

pub fn capture_training(config: &RoomConfig) -> Result<Vec<Capture>> {
    capture_training_impl(config, None)
}

pub(super) fn capture_with_receipt(
    config: &RoomConfig,
    held_out: &HashMap<String, Vec<Curve>>,
) -> Result<(Vec<Capture>, roomeq_model::StageOutcome)> {
    let mut receipt = intake::Receipt::default();
    let captures = capture_training_impl(config, Some(&mut receipt))?;
    // Stable ordering makes retained identity independent of HashMap iteration.
    let held_out: BTreeMap<_, _> = held_out.iter().collect();
    for (channel, curves) in held_out {
        receipt.record(
            "held_out",
            None,
            &Capture {
                channel: channel.clone(),
                driver: None,
                curves: curves.clone(),
                seat_labels: None,
            },
            Default::default(),
        )?;
    }
    Ok((captures, receipt.into_stage()))
}

fn capture_training_impl(
    config: &RoomConfig,
    mut receipt: Option<&mut intake::Receipt>,
) -> Result<Vec<Capture>> {
    let roles: BTreeMap<String, String> = match &config.system {
        Some(system) => {
            let mut roles: BTreeMap<String, String> = system
                .speakers
                .iter()
                .map(|(a, b)| (a.clone(), b.clone()))
                .collect();
            if let Some(subwoofers) = system.subwoofers.as_ref() {
                roles.extend(
                    subwoofers
                        .outputs
                        .iter()
                        .map(|output| (output.id.clone(), output.speaker.clone())),
                );
            }
            roles
        }
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
        let mut sources: Vec<SourceCapture<'_>> = match speaker {
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
            // The primary loudspeaker remains a physical room-seat branch and
            // must participate in final replay. The supporting signal is a
            // generated, delayed/decorrelated source; it has no independent
            // room-seat measurement in this configuration, so it is deliberately
            // not treated as additional seat evidence.
            SpeakerConfig::SupportingSource(group) => vec![(None, &group.primary)],
        };
        // A v3 grouped sub measurement may be referenced by one declaration
        // per physical output. Assign each declaration its own branch instead
        // of loading the complete group again under every output name.
        if matches!(
            speaker,
            SpeakerConfig::MultiSub(_) | SpeakerConfig::Dba(_) | SpeakerConfig::Cardioid(_)
        ) && let Some(subwoofers) = config
            .system
            .as_ref()
            .and_then(|system| system.subwoofers.as_ref())
        {
            let outputs: Vec<_> = subwoofers
                .outputs
                .iter()
                .filter(|output| output.speaker == key)
                .collect();
            if outputs.len() > 1
                && let Some(index) = outputs.iter().position(|output| output.id == role)
            {
                if outputs.len() != sources.len() {
                    return Err(invalid(
                        "physical sub output declarations must cover every grouped capture branch",
                    ));
                }
                sources = vec![(None, sources[index].1)];
            }
        }
        for (driver, source) in sources {
            // Retain native support, grids, levels, and auxiliary evidence.
            // Alignment belongs to the later physical-branch summation, not
            // capture retention. Parsed responses are not raw recordings.
            let curves = autoeq_measurements::read::load_source_unaligned(source)
                .map_err(|error| invalid(format!("{error:#}")))?;
            for curve in &curves {
                curve.validate("final-seat raw capture")?;
            }
            if matches!(speaker, SpeakerConfig::Group(_)) && curves.len() > 1 {
                return Err(invalid(
                    "multi-seat legacy driver groups require explicit topology IDs for final replay",
                ));
            }
            let capture = Capture {
                channel: role.clone(),
                driver,
                curves,
                seat_labels: crate::group_measurements::seat_labels(source),
            };
            if let Some(receipt) = receipt.as_deref_mut() {
                receipt.record("training", Some(&key), &capture, source.provenance())?;
            }
            captures.push(capture);
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
                // Single-curve captures carry the measurement name, not seat
                // order; only multi-seat label sequences can disagree.
                && a.len() > 1
                && b.len() > 1
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
    super::room_optimization_result::is_baseline_correction(plugin)
}

/// Materialize the same structural baseline used by physical-seat replay.
/// Visit every output and driver, including outputs with no optimizer result.
pub(super) fn restore_structural_baseline(result: &mut RoomOptimizationResult) {
    let supporting = super::room_optimization_result::supporting_source_output_names(result);
    for (name, chain) in &mut result.channels {
        if supporting.contains(name) {
            continue;
        }
        chain.plugins.retain(|plugin| !correction(plugin));
        if let Some(drivers) = &mut chain.drivers {
            for driver in drivers {
                driver.plugins.retain(|plugin| !correction(plugin));
            }
        }
        chain.eq_response = None;
    }
    for (name, channel) in &mut result.channel_results {
        if supporting.contains(name) {
            continue;
        }
        channel.biquads.clear();
        channel.fir_coeffs = None;
    }
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
    if baseline
        && !super::room_optimization_result::supporting_source_output_names(result).contains(owner)
    {
        plugins.retain(|p| !correction(p));
    }
    let mut chain = result
        .channels
        .get(owner)
        .cloned()
        .ok_or_else(|| invalid(format!("missing DSP owner '{owner}'")))?;
    // channel_results[owner].fir_coeffs owns the channel-level FIR, not an
    // arbitrary single convolution in a driver/stage replay sharing this owner.
    let channel_convolutions: Vec<_> = chain
        .plugins
        .iter()
        .filter(|plugin| plugin.plugin_type == "convolution")
        .collect();
    let retained_reference = if channel_convolutions.len() == 1 {
        channel_convolutions[0]
            .parameters
            .get("ir_file")
            .and_then(|value| value.as_str())
            .map(str::to_owned)
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
        && let Some(taps) = result
            .channel_results
            .get(owner)
            .and_then(|c| c.fir_coeffs.as_ref())
    {
        let path = dir.join(paths[0]);
        let evaluated_taps = if path.exists() {
            let mut channels = crate::ctc::read_wav_channels_f64(
                &path,
                crate::ctc::checked_sample_rate(fs)?,
                "final-seat convolution",
            )?;
            if channels.len() != 1
                || channels[0].len() != taps.len()
                || channels[0].iter().zip(taps).any(|(stored, retained)| {
                    !stored.is_finite()
                        || !retained.is_finite()
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

/// Exact curves and settings used for one final useful-output score.
/// Routed physical-main scores use the main-only view; ordinary outputs use
/// the same logical-output replay that produced their retained seat score.
#[derive(Debug, Clone, serde::Serialize)]
pub(super) struct PhysicalMeasurementContributor {
    /// Physical output identity whose raw capture contributes to a logical input.
    pub physical_output: String,
    /// Raw captured response before routed playback processing.
    pub measured_curve: Curve,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(super) struct UsefulOutputReplayCapture {
    pub logical_input: String,
    pub partition: String,
    pub seat_index: usize,
    pub sample_rate_hz: f64,
    pub baseline_kind: String,
    /// Hash of the serialized graph projection emitted with the replay event.
    pub replayed_serialized_dsp_graph_projection_sha256: Option<String>,
    /// Hash of executable plugin/routing payloads; excludes evidence metadata.
    pub replayed_playback_graph_sha256: Option<String>,
    /// Raw physical captures used by baseline and delivered route sums.
    pub baseline_measurement_contributors: Vec<PhysicalMeasurementContributor>,
    pub delivered_measurement_contributors: Vec<PhysicalMeasurementContributor>,
    pub baseline_curve: Curve,
    pub delivered_curve: Curve,
    pub target_curve: Option<Curve>,
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    pub schroeder_hz: Option<f64>,
    pub normalize_level: bool,
    pub permitted_gain_db: f64,
    pub scorecard: roomeq_model::AcousticQualityScorecard,
}

fn capture_measurement_contributors(
    physical: &BTreeMap<String, Vec<Curve>>,
    outputs: &[String],
    seat: usize,
) -> Result<Vec<PhysicalMeasurementContributor>> {
    outputs
        .iter()
        .map(|output| {
            Ok(PhysicalMeasurementContributor {
                physical_output: output.clone(),
                measured_curve: measured(physical, output, seat)?.clone(),
            })
        })
        .collect()
}

struct FinalSeatReplayContext<'a> {
    baseline: Option<&'a RoomOptimizationResult>,
    trace: Option<&'a mut Vec<UsefulOutputReplayCapture>>,
}

struct PhysicalMainReplayContext<'a> {
    target: Option<&'a Curve>,
    trace: Option<&'a mut Vec<UsefulOutputReplayCapture>>,
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

/// Upper edge of the fixed mono-bass assessment band in Hz.
///
/// Covers every redirected-bass and LFE low-pass in the validated range
/// (the LFE low-pass validates within 20..=250 Hz) with margin, without
/// claiming a per-system crossover derivation.
const MONO_BASS_MAX_HZ: f64 = 300.0;

/// Correlated (identical-drive) bass sum over the mono band.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct MonoBassAssessment {
    /// Mains inputs summed.
    pub inputs: usize,
    /// Assessed band actually covered by common bins.
    pub band_hz: [f64; 2],
    /// Peak coherent level in dB.
    pub peak_coherent_db: f64,
    /// Frequency of the coherent peak in Hz.
    pub peak_freq_hz: f64,
    /// Maximum coherent buildup over the incoherent (RSS) sum in dB.
    pub max_buildup_db: f64,
    /// Frequency of maximum buildup in Hz.
    pub buildup_freq_hz: f64,
}

/// Coherent identical-drive sum of delivered per-input seat responses.
///
/// F6 answers correlated program (L=R and wider) through the exact
/// serialized matrix as the acoustic sibling of the electrical
/// correlated-bus headroom bound. Every curve must share one frequency
/// grid and carry measured phase; grid mismatch or missing phase refuses
/// rather than interpolates or invents (WP2a). Reports the peak coherent
/// level and the buildup over the incoherent power sum across the mono
/// band. Advisory: no acceptance limit is attached until matrix evidence
/// justifies one.
fn assess_mono_bass_sums(
    delivered: &[Curve],
    max_mono_hz: f64,
) -> std::result::Result<MonoBassAssessment, String> {
    if delivered.len() < 2 {
        return Err("mono-bass assessment needs at least two inputs".to_string());
    }
    let grid = &delivered[0].freq;
    for (index, curve) in delivered.iter().enumerate() {
        if !roomeq_analysis::frequency_grid::same_frequency_grid(grid, &curve.freq) {
            return Err(format!("mono-bass input {index} leaves the common grid"));
        }
        if curve.spl.len() != grid.len() {
            return Err(format!("mono-bass input {index} has ragged levels"));
        }
        match curve.phase.as_ref() {
            Some(phase) if phase.len() == grid.len() => {}
            _ => return Err(format!("mono-bass input {index} has no measured phase")),
        }
        if curve.spl.iter().any(|value| !value.is_finite())
            || curve
                .phase
                .as_ref()
                .is_some_and(|phase| phase.iter().any(|value| !value.is_finite()))
        {
            return Err(format!("mono-bass input {index} is nonfinite"));
        }
    }
    let bins: Vec<usize> = grid
        .iter()
        .enumerate()
        .filter(|(_, frequency)| **frequency <= max_mono_hz)
        .map(|(index, _)| index)
        .collect();
    if bins.len() < 2 {
        return Err("mono-bass band has no common support".to_string());
    }
    let mut peak = (f64::NEG_INFINITY, grid[bins[0]]);
    let mut buildup = (f64::NEG_INFINITY, grid[bins[0]]);
    for bin in &bins {
        let mut coherent = num_complex::Complex64::new(0.0, 0.0);
        let mut power = 0.0;
        for curve in delivered {
            let magnitude = 10.0_f64.powf(curve.spl[*bin] / 20.0);
            let angle = curve.phase.as_ref().expect("checked above")[*bin].to_radians();
            coherent += num_complex::Complex64::from_polar(magnitude, angle);
            power += magnitude * magnitude;
        }
        let coherent_db = 20.0 * coherent.norm().max(1e-24).log10();
        let rss_db = 10.0 * power.max(1e-24).log10();
        if coherent_db > peak.0 {
            peak = (coherent_db, grid[*bin]);
        }
        let ratio = coherent_db - rss_db;
        if ratio > buildup.0 {
            buildup = (ratio, grid[*bin]);
        }
    }
    if !peak.0.is_finite() || !buildup.0.is_finite() {
        return Err("mono-bass sums are nonfinite".to_string());
    }
    Ok(MonoBassAssessment {
        inputs: delivered.len(),
        band_hz: [grid[bins[0]], grid[*bins.last().expect("checked above")]],
        peak_coherent_db: peak.0,
        peak_freq_hz: peak.1,
        max_buildup_db: buildup.0,
        buildup_freq_hz: buildup.1,
    })
}

/// Final-boundary correlated-bass stage for the delivered graph.
///
/// Replays every mains input at the prime training seat through the final
/// serialized matrix and sums them as identical-drive program. Never fails:
/// single-input systems, CTC, replay errors, and missing phase each record
/// an explicit skip/unassessed advisory instead of a verdict.
pub(super) fn correlated_bass_stage(
    result: &RoomOptimizationResult,
    captures: &[Capture],
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
) -> roomeq_model::StageOutcome {
    let skipped = |reason: String| roomeq_model::StageOutcome {
        stage: "final_correlated_bass_sum".into(),
        status: roomeq_model::StageStatus::Skipped,
        advisories: vec![reason],
        checks: Vec::new(),
    };
    if result.metadata.ctc.is_some() {
        return skipped("mono_bass_skipped:ctc".to_string());
    }
    let mut inputs: Vec<String> = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
        .map(|graph| {
            graph
                .input_channels
                .iter()
                .filter(|input| {
                    graph
                        .routes
                        .iter()
                        .any(|route| &route.source_channel == *input)
                })
                .cloned()
                .collect()
        })
        .unwrap_or_else(|| independent_input_channels(captures, result).unwrap_or_default());
    inputs.retain(|input| {
        roomeq_model::home_cinema::role_for_channel(input) != roomeq_model::HomeCinemaRole::Lfe
    });
    inputs.sort();
    inputs.dedup();
    if inputs.len() < 2 {
        return skipped("mono_bass_skipped:single_mains_input".to_string());
    }
    let physical = match training_physical_captures(captures, result) {
        Ok(physical) => physical,
        Err(error) => return skipped(format!("mono_bass_unassessed:captures:{error}")),
    };
    let mut delivered = Vec::with_capacity(inputs.len());
    for input in &inputs {
        match replay_final_physical_seat(result, &physical, input, 0, config, fs, dir, "training") {
            Ok(playback) => delivered.push(playback.delivered),
            Err(error) => {
                return skipped(format!("mono_bass_unassessed:{input}:{error}"));
            }
        }
    }
    match assess_mono_bass_sums(&delivered, MONO_BASS_MAX_HZ) {
        Ok(assessment) => roomeq_model::StageOutcome {
            stage: "final_correlated_bass_sum".into(),
            status: roomeq_model::StageStatus::Applied,
            advisories: vec![format!(
                "mono_bass:inputs={}:band_hz={:.1}-{:.1}:peak_db={:.2}:peak_hz={:.1}:max_buildup_db={:.2}:buildup_hz={:.1}",
                assessment.inputs,
                assessment.band_hz[0],
                assessment.band_hz[1],
                assessment.peak_coherent_db,
                assessment.peak_freq_hz,
                assessment.max_buildup_db,
                assessment.buildup_freq_hz,
            )],
            checks: Vec::new(),
        },
        Err(reason) => skipped(format!("mono_bass_unassessed:{reason}")),
    }
}

/// Standalone subwoofer groups are assessed over their common measured band,
/// not the full-range optimizer ceiling. Do not apply this to routed main/sub
/// sums: their acoustic tails still need evidence across the main's band.
fn standalone_summation_max_hz(
    config: &RoomConfig,
    input: &str,
    branches: &[bounded_sum::Branch],
) -> f64 {
    let key = config
        .system
        .as_ref()
        .and_then(|system| system.speakers.get(input))
        .map(String::as_str)
        .unwrap_or(input);
    let subwoofer = matches!(
        config.speakers.get(key),
        Some(SpeakerConfig::MultiSub(_) | SpeakerConfig::Dba(_) | SpeakerConfig::Cardioid(_))
    ) || super::misc::is_subwoofer_channel(config, input);
    if subwoofer {
        branches
            .iter()
            .fold(config.optimizer.max_freq, |high, branch| {
                high.min(branch.measured.freq.last().copied().unwrap_or(high))
            })
    } else {
        config.optimizer.max_freq
    }
}

/// Observe routed playback over measured support, independently of EQ bounds.
fn passband_observation_config(
    config: &RoomConfig,
    result: &RoomOptimizationResult,
    physical: &BTreeMap<String, Vec<Curve>>,
    input: &str,
    seat: usize,
) -> Result<RoomConfig> {
    let mut observation = config.clone();
    let Some(graph) = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
    else {
        // Independent stereo speakers have the same distinction between EQ
        // bounds and playback observation as routed home-cinema speakers.
        // Driver groups retain their separate common-support calculation.
        if let Some(raw) = physical.get(input).and_then(|curves| curves.get(seat)) {
            let (passband, _) =
                roomeq_engine::analysis::response_metrics::detect_passband_and_mean(raw);
            let (low, high) = passband.ok_or_else(|| {
                invalid(format!("no measured passband for '{input}' seat {seat}"))
            })?;
            observation.optimizer.min_freq = low;
            observation.optimizer.max_freq = high;
        }
        return Ok(observation);
    };
    let routes: Vec<_> = graph
        .routes
        .iter()
        .filter(|route| route.source_channel == input)
        .collect();
    let main = routes
        .iter()
        .find(|route| route.high_pass_hz.is_some() || route.destination == input);
    let mut lower = f64::INFINITY;
    let mut upper = 0.0_f64;
    for route in &routes {
        let raw = measured(physical, &route.destination, seat)?;
        let (passband, _) =
            roomeq_engine::analysis::response_metrics::detect_passband_and_mean(raw);
        let (lo, hi) = passband.ok_or_else(|| {
            invalid(format!(
                "no measured passband for '{}' seat {seat}",
                route.destination
            ))
        })?;
        lower = lower.min(lo);
        if main.is_some_and(|main| main.destination == route.destination) {
            upper = hi;
        } else if main.is_none() {
            // A native LFE input is assessed in its actual low-pass band, not
            // through the treble merely because the optimizer allows it.
            upper = upper.max(hi.min(route.low_pass_hz.unwrap_or(hi)));
        }
    }
    if lower.is_finite() && upper > lower {
        observation.optimizer.min_freq = lower;
        observation.optimizer.max_freq = upper;
    } else {
        return Err(invalid(format!(
            "no supported playback band for '{input}' seat {seat}"
        )));
    }
    Ok(observation)
}

fn replay(
    result: &RoomOptimizationResult,
    physical: &BTreeMap<String, Vec<Curve>>,
    input: &str,
    seat: usize,
    baseline: bool,
    context: &ReplayContext<'_>,
) -> Result<(Summed, Vec<String>)> {
    replay_output(result, physical, input, seat, baseline, context, None)
}

/// Replay the same serialized graph, optionally selecting one physical output.
fn replay_output(
    result: &RoomOptimizationResult,
    physical: &BTreeMap<String, Vec<Curve>>,
    input: &str,
    seat: usize,
    baseline: bool,
    context: &ReplayContext<'_>,
    destination: Option<&str>,
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
        for route in graph.routes.iter().filter(|r| {
            r.source_channel == input && destination.is_none_or(|output| r.destination == output)
        }) {
            let raw = measured(physical, &route.destination, seat)?;
            // Match serialized routed export: input pre-route -> route matrix
            // gain/polarity, crossover/delay -> destination post-route.
            let pre_route_plugins = match result.channels.get(input) {
                Some(chain) => stage(chain, "pre_route"),
                None if roomeq_model::home_cinema::role_for_channel(input)
                    == roomeq_model::HomeCinemaRole::Lfe =>
                {
                    Vec::new()
                }
                None => return Err(invalid("missing route input")),
            };
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
            let subwoofer_low_pass_hz = matches!(
                route.route_kind.as_str(),
                "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
            )
            .then(|| {
                route.low_pass_hz.or_else(|| {
                    post.drivers
                        .as_ref()?
                        .iter()
                        .find(|driver| driver.name == route.destination)?
                        .plugins
                        .iter()
                        .filter(|plugin| {
                            plugin.plugin_type == "crossover"
                                && plugin.parameters["output"] == "low"
                                && plugin.parameters["room_eq_stage"] == "post_route"
                        })
                        .filter_map(|plugin| plugin.parameters["frequency"].as_f64())
                        .filter(|frequency| frequency.is_finite() && *frequency > 0.0)
                        .min_by(f64::total_cmp)
                })
            })
            .flatten();
            let branch = process_branch(
                &route.destination,
                raw,
                &config.optimizer.upper_band_acoustic_bounds,
                partition,
                seat,
                &grid,
                subwoofer_low_pass_hz,
                |curve| {
                    let curve = if pre_route_plugins.is_empty() {
                        curve.clone()
                    } else {
                        apply(
                            result,
                            input,
                            pre_route_plugins.clone(),
                            curve,
                            baseline,
                            fs,
                            dir,
                        )?
                    };
                    let curve = apply(
                        result,
                        post_owner,
                        route_plugins.clone(),
                        &curve,
                        false,
                        fs,
                        dir,
                    )?;
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
        return Ok((
            // Keep the configured observation band visible. `sum_branches`
            // requires an explicit acoustic bound for any branch that does
            // not cover it; capping at the common measured endpoint would
            // silently turn missing evidence into a better-looking score.
            sum_branches(
                &branches,
                config.optimizer.min_freq,
                config.optimizer.max_freq,
            )?,
            outputs,
        ));
    }
    let chain = result.channels.get(input).ok_or_else(|| {
        invalid(format!(
            "missing channel '{input}' for final seat replay (available: {:?})",
            result
                .channels
                .keys()
                .collect::<std::collections::BTreeSet<_>>()
        ))
    })?;
    if let Some(drivers) = &chain.drivers {
        let mut branches = Vec::new();
        let mut outputs = Vec::new();
        for driver in drivers {
            let raw = measured(physical, &driver.name, seat)?;
            branches.push(process_branch(
                &driver.name,
                raw,
                &config.optimizer.upper_band_acoustic_bounds,
                partition,
                seat,
                &grid,
                None,
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
        let max_hz = standalone_summation_max_hz(config, input, &branches);
        return Ok((
            sum_branches(&branches, config.optimizer.min_freq, max_hz)?,
            outputs,
        ));
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

#[cfg(test)]
pub(super) fn validate_final_seats(
    result: &mut RoomOptimizationResult,
    captures: &[Capture],
    held_out: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
) -> Result<()> {
    validate_final_seats_impl(
        result,
        captures,
        held_out,
        config,
        fs,
        dir,
        FinalSeatReplayContext {
            baseline: None,
            trace: None,
        },
    )
}

/// Final selection also checks single-seat systems: a single capture is not an
/// exemption from delivered gain or useful-output limits.
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_candidate_final_seats(
    result: &mut RoomOptimizationResult,
    baseline: &RoomOptimizationResult,
    captures: &[Capture],
    held_out: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
) -> Result<()> {
    validate_final_seats_impl(
        result,
        captures,
        held_out,
        config,
        fs,
        dir,
        FinalSeatReplayContext {
            baseline: Some(baseline),
            trace: None,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_candidate_final_seats_with_diagnostic_trace(
    result: &mut RoomOptimizationResult,
    baseline: &RoomOptimizationResult,
    captures: &[Capture],
    held_out: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
    replay_trace: &mut Vec<UsefulOutputReplayCapture>,
) -> Result<()> {
    validate_final_seats_impl(
        result,
        captures,
        held_out,
        config,
        fs,
        dir,
        FinalSeatReplayContext {
            baseline: Some(baseline),
            trace: Some(replay_trace),
        },
    )
}

/// Return the lowest frequency at which a deliberately configured excursion
/// high-pass is expected to preserve useful output.  Frequencies below this
/// point remain visible as `unassessed_bands_hz` rather than being counted as
/// an unexplained level loss.  Auto-detected F3 is intentionally not guessed
/// here; it must be carried as explicit evidence by a later measurement
/// contract instead of silently shrinking the score band.
fn excursion_supported_min_frequency(config: &RoomConfig) -> Option<f64> {
    let protection = config.optimizer.excursion_protection.as_ref()?;
    if !protection.enabled {
        return None;
    }
    let f3 = protection.manual_f3_hz?;
    if !f3.is_finite() || f3 <= 0.0 || !protection.margin_octaves.is_finite() {
        return None;
    }
    Some(f3 * 2.0_f64.powf(-protection.margin_octaves))
}

/// Separate physical-main SPL evidence from the combined source's bass quality.
fn physical_main_quality(
    result: &RoomOptimizationResult,
    baseline: &RoomOptimizationResult,
    physical: &BTreeMap<String, Vec<Curve>>,
    input: &str,
    seat: usize,
    context: &ReplayContext<'_>,
    replay_context: PhysicalMainReplayContext<'_>,
) -> Result<Option<roomeq_model::AcousticQualityScorecard>> {
    let PhysicalMainReplayContext {
        target,
        trace: replay_trace,
    } = replay_context;
    let main = baseline
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
        .and_then(|graph| {
            graph.routes.iter().find(|route| {
                route.source_channel == input
                    && (route.high_pass_hz.is_some() || route.destination == input)
            })
        });
    let Some(main) = main else {
        return Ok(None);
    };
    let raw = measured(physical, &main.destination, seat)?;
    let (passband, _) = roomeq_engine::analysis::response_metrics::detect_passband_and_mean(raw);
    let (low, high) = passband.ok_or_else(|| invalid("physical main has no measured passband"))?;
    // Freeze the supported band from the measurement and structural baseline;
    // a candidate may not hide damage by changing its own cutoff or response.
    let low = low
        .max(main.high_pass_hz.unwrap_or(low))
        .max(excursion_supported_min_frequency(context.config).unwrap_or(0.0));
    if low >= high {
        return Err(invalid("physical main has no supported playback band"));
    }
    let mut observation = context.config.clone();
    observation.optimizer.min_freq = low;
    observation.optimizer.max_freq = high;
    let main_context = ReplayContext {
        config: &observation,
        ..*context
    };
    let (pre, baseline_outputs) = replay_output(
        baseline,
        physical,
        input,
        seat,
        true,
        &main_context,
        Some(&main.destination),
    )?;
    let (post, delivered_outputs) = replay_output(
        result,
        physical,
        input,
        seat,
        false,
        &main_context,
        Some(&main.destination),
    )?;
    let permitted_gain_db = observation
        .optimizer
        .permitted_output_gain_db
        .get(input)
        .copied()
        .unwrap_or(0.0);
    let schroeder_hz = roomeq_model::auto_tune::resolved_schroeder_hz(&observation.optimizer);
    let scorecard = roomeq_engine::quality::evaluate_acoustic_quality_with_permitted_gain(
        std::slice::from_ref(&pre.curve),
        std::slice::from_ref(&post.curve),
        &[],
        &[],
        target,
        roomeq_engine::quality::QualityEvaluationConfig {
            min_freq_hz: low,
            max_freq_hz: high,
            schroeder_hz,
            normalize_level: true,
        },
        Default::default(),
        permitted_gain_db,
    )
    .map_err(invalid)?;
    if let Some(replay_trace) = replay_trace {
        replay_trace.push(UsefulOutputReplayCapture {
            logical_input: input.to_string(),
            partition: context.partition.to_string(),
            seat_index: seat,
            sample_rate_hz: context.fs,
            baseline_kind: "pre_finalization_optimized_graph_without_tagged_correction".into(),
            replayed_serialized_dsp_graph_projection_sha256: None,
            replayed_playback_graph_sha256: None,
            baseline_measurement_contributors: capture_measurement_contributors(
                physical,
                &baseline_outputs,
                seat,
            )?,
            delivered_measurement_contributors: capture_measurement_contributors(
                physical,
                &delivered_outputs,
                seat,
            )?,
            baseline_curve: pre.curve.clone(),
            delivered_curve: post.curve.clone(),
            target_curve: target.cloned(),
            min_freq_hz: low,
            max_freq_hz: high,
            schroeder_hz,
            normalize_level: true,
            permitted_gain_db,
            scorecard: scorecard.clone(),
        });
    }
    Ok(Some(scorecard))
}

/// Validate shared support per input without inventing overlap between independent inputs.
fn final_seat_overlap(seats: &[FinalSeatEvaluation]) -> Result<Option<[f64; 2]>> {
    let mut input_bands = BTreeMap::<&str, [f64; 2]>::new();
    for seat in seats {
        let band = seat.evaluated_band_hz;
        if !band.iter().all(|edge| edge.is_finite()) || band[0] <= 0.0 || band[0] >= band[1] {
            return Err(invalid(
                "final-seat scorecard has invalid frequency support",
            ));
        }
        let overlap = input_bands.entry(&seat.logical_input).or_insert(band);
        overlap[0] = overlap[0].max(band[0]);
        overlap[1] = overlap[1].min(band[1]);
        if overlap[0] >= overlap[1] {
            return Err(invalid(format!(
                "final-seat input '{}' has no common supported frequency band",
                seat.logical_input
            )));
        }
    }
    if input_bands.is_empty() {
        return Err(invalid("final-seat scorecard has no frequency support"));
    }
    Ok(input_bands
        .values()
        .try_fold([0.0_f64, f64::INFINITY], |[low, high], band| {
            let overlap = [low.max(band[0]), high.min(band[1])];
            (overlap[0] < overlap[1]).then_some(overlap)
        }))
}

fn independent_input_channels(
    captures: &[Capture],
    result: &RoomOptimizationResult,
) -> Result<Vec<String>> {
    captures
        .iter()
        .map(|capture| {
            let owners: Vec<_> = result
                .channels
                .iter()
                .filter_map(|(name, chain)| {
                    (name == &capture.channel
                        || chain.drivers.as_ref().is_some_and(|drivers| {
                            drivers.iter().any(|driver| driver.name == capture.channel)
                        }))
                    .then_some(name)
                })
                .collect();
            match owners.as_slice() {
                [owner] => Ok((*owner).clone()),
                [] => Err(invalid(format!(
                    "no DSP owner for final-seat capture '{}'",
                    capture.channel
                ))),
                _ => Err(invalid(format!(
                    "ambiguous DSP owners for final-seat capture '{}'",
                    capture.channel
                ))),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
/// Routed crossover frequency for one logical input, when the graph states it.
///
/// Mirrors the splice assessment lookup: the redirected-bass low-pass first,
/// then the main high-pass fallback. Absent on unrouted graphs and inputs.
fn routed_crossover_hz(result: &RoomOptimizationResult, input: &str) -> Option<f64> {
    let graph = result
        .metadata
        .bass_management
        .as_ref()?
        .routing_graph
        .as_ref()?;
    graph
        .routes
        .iter()
        .find(|route| {
            route.source_channel == input && route.route_kind == "redirected_bass_lowpass_to_sub"
        })
        .and_then(|route| route.low_pass_hz)
        .or_else(|| {
            graph
                .routes
                .iter()
                .find(|route| {
                    route.source_channel == input && route.route_kind == "main_highpass_to_self"
                })
                .and_then(|route| route.high_pass_hz)
        })
        .filter(|hertz| hertz.is_finite() && *hertz > 0.0)
}

/// Per-band raw improvements for one seat over modal bass, crossover
/// overlap, and the remaining upper band.
///
/// Re-evaluates the same pre/post/target evidence over each sub-band with
/// the shared quality evaluator. Bands with fewer than two supported bins
/// stay absent, as does the crossover band without a routed crossover
/// frequency. Without a configured Schroeder split the conventional 200 Hz
/// bass edge applies; the used value is recorded, never silent.
#[allow(clippy::too_many_arguments)]
fn band_improvements(
    pre: &Curve,
    post: &Curve,
    target: Option<&Curve>,
    lo: f64,
    hi: f64,
    schroeder_hz: Option<f64>,
    crossover_hz: Option<f64>,
    permitted_gain_db: f64,
) -> roomeq_model::BandImprovement {
    let split = schroeder_hz
        .filter(|hertz| hertz.is_finite() && *hertz > 0.0)
        .unwrap_or(200.0);
    let eval = |band_lo: f64, band_hi: f64| -> Option<f64> {
        // Degenerate, unordered, or NaN bands carry no evidence; refuse them.
        if band_lo.partial_cmp(&band_hi) != Some(std::cmp::Ordering::Less) {
            return None;
        }
        roomeq_engine::quality::evaluate_acoustic_quality_with_permitted_gain(
            std::slice::from_ref(pre),
            std::slice::from_ref(post),
            &[],
            &[],
            target,
            roomeq_engine::quality::QualityEvaluationConfig {
                min_freq_hz: band_lo,
                max_freq_hz: band_hi,
                schroeder_hz: Some(split),
                normalize_level: true,
            },
            Default::default(),
            permitted_gain_db,
        )
        .ok()
        .map(|score| score.training.worst_position_improvement_db)
        .filter(|value| value.is_finite())
    };
    let bass_band = [lo, split.min(hi)];
    let upper_band = [split.max(lo), hi];
    let crossover_band = crossover_hz
        .filter(|hertz| hertz.is_finite() && *hertz > 0.0)
        .map(|hertz| [(hertz / 2.0).max(lo), (hertz * 2.0).min(hi)])
        .filter(|band| band[0] < band[1]);
    roomeq_model::BandImprovement {
        schroeder_hz: split,
        bass_band_hz: bass_band,
        bass_improvement_db: eval(bass_band[0], bass_band[1]),
        crossover_band_hz: crossover_band,
        crossover_improvement_db: crossover_band.and_then(|band| eval(band[0], band[1])),
        upper_band_hz: upper_band,
        upper_improvement_db: eval(upper_band[0], upper_band[1]),
    }
}

fn validate_final_seats_impl(
    result: &mut RoomOptimizationResult,
    captures: &[Capture],
    held_out: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
    replay_context: FinalSeatReplayContext<'_>,
) -> Result<()> {
    let FinalSeatReplayContext {
        baseline,
        trace: mut replay_trace,
    } = replay_context;
    crate::export::validate_final_routed_stage_ownership(result)?;
    if baseline.is_none() && !captures.iter().any(|c| c.curves.len() > 1) && held_out.is_empty() {
        return Ok(());
    }
    if captures.iter().any(|capture| {
        let key = config
            .system
            .as_ref()
            .and_then(|system| system.speakers.get(&capture.channel))
            .unwrap_or(&capture.channel);
        capture.curves.len() > 1
            && matches!(config.speakers.get(key), Some(SpeakerConfig::Group(_)))
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
    // A power-averaged display curve intentionally has no phase. Establish
    // phase availability from every original capture and its delivered replay,
    // rather than rejecting measured multi-seat FIRs on that display omission.
    let mut phase_evidence_available = true;
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
        // A non-routed group can own several declared physical outputs under
        // its first output's logical channel. Replay the complete owning chain,
        // while retaining each physical capture and its original seat identity.
        independent_input_channels(captures, result)?
    };
    inputs.sort();
    inputs.dedup();
    // Some routing layouts retain a silent logical slot (e.g. an unused LFE
    // input). It has no acoustic branches in either graph. Do not confuse it
    // with a lost branch in a previously active input.
    if let Some(graph) = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
    {
        let baseline_graph = baseline
            .and_then(|before| before.metadata.bass_management.as_ref())
            .and_then(|bass| bass.routing_graph.as_ref());
        let silent: Vec<_> = inputs
            .iter()
            .filter(|input| {
                !graph
                    .routes
                    .iter()
                    .any(|route| &route.source_channel == *input)
                    && baseline_graph.is_none_or(|before| {
                        !before
                            .routes
                            .iter()
                            .any(|route| &route.source_channel == *input)
                    })
            })
            .cloned()
            .collect();
        inputs.retain(|input| !silent.contains(input));
        for input in silent {
            result
                .metadata
                .stage_outcomes
                .push(roomeq_model::StageOutcome {
                    stage: "final_acoustic_silent_input".into(),
                    status: roomeq_model::StageStatus::Skipped,
                    advisories: vec![format!(
                        "logical_input={input}; no_routes_in_baseline_or_delivered_graph"
                    )],
                    checks: Vec::new(),
                });
        }
    }
    for (partition, physical) in [("training", &training), ("held_out", &held)] {
        if physical.is_empty() {
            continue;
        }
        phase_evidence_available &= physical.values().flatten().all(|curve| {
            curve.phase.as_ref().is_some_and(|phase| {
                phase.len() == curve.freq.len() && phase.iter().all(|value| value.is_finite())
            })
        });
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
                // Filter placement and playback validation have different
                // domains. Bass-only optimization must still catch an upper-
                // passband gain or filter regression on the delivered graph.
                let correction_band = config
                    .optimizer
                    .correction_band
                    .map(|policy| [policy.min_hz, policy.max_hz])
                    .unwrap_or([config.optimizer.min_freq, config.optimizer.max_freq]);
                let observation =
                    passband_observation_config(config, result, physical, input, seat)?;
                let config = &observation;
                let context = ReplayContext {
                    config,
                    partition,
                    fs,
                    dir,
                };
                let (pre, outputs) = replay(
                    baseline.unwrap_or(result),
                    physical,
                    input,
                    seat,
                    true,
                    &context,
                )?;
                let (post, delivered_outputs) =
                    replay(result, physical, input, seat, false, &context)?;
                let uncertainty_db = pre.uncertainty_db() + post.uncertainty_db();
                let pre_support = pre.support;
                let post_support = post.support;
                let pre = pre.curve;
                let post = post.curve;
                phase_evidence_available &= [&pre, &post].iter().all(|curve| {
                    curve.phase.as_ref().is_some_and(|phase| {
                        phase.len() == curve.freq.len()
                            && phase.iter().all(|value| value.is_finite())
                    })
                });
                let target = result
                    .channels
                    .get(input)
                    .and_then(|c| c.target_curve.clone())
                    .map(Curve::from);
                let schroeder_hz =
                    roomeq_model::auto_tune::resolved_schroeder_hz(&config.optimizer);
                let lo = config
                    .optimizer
                    .min_freq
                    .max(pre.freq[0])
                    .max(post.freq[0])
                    .max(excursion_supported_min_frequency(config).unwrap_or(0.0));
                let hi = config
                    .optimizer
                    .max_freq
                    .min(*pre.freq.last().unwrap())
                    .min(*post.freq.last().unwrap());
                let mut score =
                    roomeq_engine::quality::evaluate_acoustic_quality_with_permitted_gain(
                        std::slice::from_ref(&pre),
                        std::slice::from_ref(&post),
                        &[],
                        &[],
                        target.as_ref(),
                        roomeq_engine::quality::QualityEvaluationConfig {
                            min_freq_hz: lo,
                            max_freq_hz: hi,
                            schroeder_hz,
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
                let scorecard_for_replay = score.clone();
                score.correction_band_hz = Some(correction_band);
                let replay_count_before = replay_trace.as_ref().map_or(0, |trace| trace.len());
                if let Some(main_score) = physical_main_quality(
                    result,
                    baseline.unwrap_or(result),
                    physical,
                    input,
                    seat,
                    &context,
                    PhysicalMainReplayContext {
                        target: target.as_ref(),
                        trace: replay_trace.as_deref_mut(),
                    },
                )? {
                    // Only replace the SPL-loss evidence. Combined-source
                    // shape, uncertainty, and crossover quality stay intact.
                    score.useful_output = main_score.useful_output;
                }
                // A logical output without a routed physical-main branch is
                // scored directly from these same pre/post curves. Retain
                // those inputs too; otherwise an opt-in diagnostic could
                // claim the replay was visited while losing its actual score
                // basis merely because this graph has no bass route.
                if let Some(replay_trace) = replay_trace.as_deref_mut()
                    && replay_trace.len() == replay_count_before
                {
                    replay_trace.push(UsefulOutputReplayCapture {
                        logical_input: input.clone(),
                        partition: partition.into(),
                        seat_index: seat,
                        sample_rate_hz: fs,
                        baseline_kind: "pre_finalization_optimized_graph_without_tagged_correction"
                            .into(),
                        replayed_serialized_dsp_graph_projection_sha256: None,
                        replayed_playback_graph_sha256: None,
                        baseline_measurement_contributors: capture_measurement_contributors(
                            physical, &outputs, seat,
                        )?,
                        delivered_measurement_contributors: capture_measurement_contributors(
                            physical,
                            &delivered_outputs,
                            seat,
                        )?,
                        baseline_curve: pre.clone(),
                        delivered_curve: post.clone(),
                        target_curve: target.clone(),
                        min_freq_hz: lo,
                        max_freq_hz: hi,
                        schroeder_hz,
                        normalize_level: true,
                        permitted_gain_db: config
                            .optimizer
                            .permitted_output_gain_db
                            .get(input)
                            .copied()
                            .unwrap_or(0.0),
                        scorecard: scorecard_for_replay,
                    });
                }
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
                    band_improvement_db: Some(band_improvements(
                        &pre,
                        &post,
                        target.as_ref(),
                        lo,
                        hi,
                        schroeder_hz,
                        routed_crossover_hz(result, input),
                        config
                            .optimizer
                            .permitted_output_gain_db
                            .get(input)
                            .copied()
                            .unwrap_or(0.0),
                    )),
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
    score.measurement_overlap_hz = final_seat_overlap(&score.final_seats)?;
    let report = result
        .metadata
        .correction_acceptance
        .as_mut()
        .ok_or_else(|| invalid("final-seat acceptance report unavailable"))?;
    if let Some(previous) = &report.acoustic_quality {
        score.temporal = previous.temporal.clone();
    }
    score.temporal.phase_evidence_available =
        phase_evidence_available && !score.final_seats.is_empty();
    // Presence is not verification: upgrade the scorecard verdict from the
    // evaluated inputs' declared provenance. Advisory-only: acceptance still
    // keys on phase presence; coherent claims must consult this verdict.
    score.temporal.coherent_timing =
        match crate::evidence_intake::shared_timing_reference(config, &inputs) {
            Ok(reference_id) => CoherentTimingEvidence::Verified { reference_id },
            Err(reason) => CoherentTimingEvidence::Refused { reason },
        };
    // Electrical upper bound: the gain backstop's checked quantity.
    // Enforcement compares this bound (see acceptance); the acoustic
    // ratio stays reported alongside for the near-null explanation.
    score.max_electrical_boost_db = crate::delay_compile::electrical_boost_bound_db(
        &result.channels,
        result
            .metadata
            .bass_management
            .as_ref()
            .and_then(|bass| bass.routing_graph.as_ref())
            .map(|graph| graph.routes.as_slice()),
    );
    if score.temporal.phase_evidence_available {
        let only_phase_missing = report.violations.as_slice() == ["phase_evidence_missing"];
        report
            .violations
            .retain(|violation| violation != "phase_evidence_missing");
        if only_phase_missing {
            report.accepted = true;
            report.decision = roomeq_model::CorrectionDecision::Accepted;
        }
        report.refresh_outcome();
    } else if report.runtime_policy.as_ref().is_some_and(|policy| {
        matches!(
            policy.output_class,
            roomeq_model::RuntimeOutputClass::Fir | roomeq_model::RuntimeOutputClass::Hybrid
        )
    }) {
        if !report
            .violations
            .iter()
            .any(|violation| violation == "phase_evidence_missing")
        {
            report
                .violations
                .push(String::from("phase_evidence_missing"));
        }
        report.accepted = false;
        if report.decision == roomeq_model::CorrectionDecision::Accepted {
            report.decision = roomeq_model::CorrectionDecision::IdentityFallback;
        }
        report.refresh_outcome();
    }
    let budget = report
        .runtime_policy
        .as_ref()
        .ok_or_else(|| invalid("final-seat runtime budget unavailable"))?
        .max_worst_position_regression_db;
    let failed: Vec<_> = score
        .final_seats
        .iter()
        .filter(|s| {
            !s.improvement_lower_bound_db.is_finite() || s.improvement_lower_bound_db < -budget
        })
        .map(|s| {
            format!(
                "{} '{}' seat {} regressed {:.3} dB beyond {:.3} dB budget",
                s.partition, s.logical_input, s.seat_index, -s.improvement_lower_bound_db, budget
            )
        })
        .collect();
    let output_budget = config.optimizer.finalization.max_useful_output_loss_db;
    let output_failed: Vec<_> = score.useful_output.iter().filter_map(|output| {
        // f64::max masks a NaN if its other operand is finite.
        let finite = output.unexplained_loss_rms_db.is_finite()
            && output.bass_unexplained_loss_rms_db.is_none_or(f64::is_finite)
            && output.extension_loss_db.is_none_or(f64::is_finite)
            && output.peak_demand_change_db.is_none_or(f64::is_finite);
        let loss = output.unexplained_loss_rms_db
            .max(output.bass_unexplained_loss_rms_db.unwrap_or(0.0));
        // Subwoofer peak cuts are not constrained by the main-speaker SPL
        // allowance. Keep their evidence and finite checks; electrical safety
        // and response-shape/crossover acceptance remain independent gates.
        let subwoofer = output.logical_input.as_deref().is_some_and(|input| {
            let key = config.system.as_ref()
                .and_then(|system| system.speakers.get(input))
                .map(String::as_str).unwrap_or(input);
            super::misc::is_subwoofer_channel(config, input)
                || matches!(config.speakers.get(key),
                    Some(SpeakerConfig::MultiSub(_) | SpeakerConfig::Dba(_) | SpeakerConfig::Cardioid(_)))
        });
        (!finite || !loss.is_finite() || (!subwoofer && loss > output_budget)).then(|| {
            format!(
                "{} '{}' seat {} lost {:.3} dB useful output beyond {:.3} dB budget (permitted gain {:.3} dB)",
                output.partition, output.logical_input.as_deref().unwrap_or("unknown"),
                output.seat_index, loss, output_budget, output.permitted_gain_db
            )
        })
    }).collect();
    report.acoustic_quality = Some(score);
    if !output_failed.is_empty() {
        report
            .violations
            .push("unexplained_useful_output_loss".into());
    }
    if !failed.is_empty() {
        report.violations.push("worst_position_regressed".into());
    }
    // WP3 final boundary: the replayed score replaces the evidence the
    // safety gate enforced, so reevaluate the runtime policy against what
    // is now attached. Without this, the report can show a limit, an
    // exceeding replayed value, and an accepted outcome that was computed
    // from different evidence. New violations fail closed like the seat
    // and output regressions above.
    let mut boundary_failures = Vec::new();
    if report.policy == roomeq_model::CorrectionAcceptancePolicy::RuntimeSafety
        && let Some(attached) = report.acoustic_quality.clone()
        && let Some(policy) = report.runtime_policy.clone()
    {
        if let Some(realization) = report.realization_quality.clone() {
            let before = report.violations.clone();
            roomeq_quality::enforce_runtime_acceptance_evidence(
                report,
                attached.clone(),
                realization,
                policy.clone(),
            )
            .map_err(|reason| invalid(format!("replayed evidence refused: {reason}")))?;
            let fresh: Vec<String> = report
                .violations
                .iter()
                .filter(|violation| !before.contains(violation))
                .cloned()
                .collect();
            for violation in &fresh {
                // The boost backstop carries two quantities; name the
                // values so a rejection names its evidence, not just the
                // limit. The applier re-pushes the same violation string,
                // deduplicated below.
                if violation == "max_boost_limit_exceeded"
                    && let Some(detail) = roomeq_quality::apply_boost_limit_agreement(
                        report,
                        attached.max_boost_db,
                        attached.max_electrical_boost_db,
                        &policy,
                    )
                {
                    boundary_failures.push(format!(
                        "replayed evidence violates max_boost_limit_exceeded ({detail})"
                    ));
                    continue;
                }
                boundary_failures.push(format!("replayed evidence violates {violation}"));
            }
            report.violations.sort();
            report.violations.dedup();
        } else if let Some(message) = roomeq_quality::apply_boost_limit_agreement(
            report,
            attached.max_boost_db,
            attached.max_electrical_boost_db,
            &policy,
        ) {
            boundary_failures.push(message);
        }
    }
    let failures: Vec<_> = failed
        .into_iter()
        .chain(output_failed)
        .chain(boundary_failures)
        .collect();
    if !failures.is_empty() {
        report.accepted = false;
        report.decision = roomeq_model::CorrectionDecision::Rejected;
        report.violations.sort();
        report.violations.dedup();
        report.refresh_outcome();
        return Err(AutoeqError::OptimizationFailed {
            message: failures.join("; "),
        });
    }
    let concealment = band_concealment_findings(
        &report
            .acoustic_quality
            .as_ref()
            .map(|score| score.final_seats.clone())
            .unwrap_or_default(),
        config.optimizer.finalization.min_improvement_lower_bound_db,
        budget,
    );
    result
        .metadata
        .stage_outcomes
        .push(concealment_stage(concealment));
    Ok(())
}

/// Training seats whose broadband gain conceals a band regression.
///
/// A seat conceals when its uncertainty-adjusted broadband improvement
/// clears the benefit floor while any assessed band's raw improvement
/// falls beyond the regression budget. Advisory findings; enforcement
/// waits on matrix evidence that no accepted correction trips it.
fn band_concealment_findings(
    final_seats: &[FinalSeatEvaluation],
    benefit_floor_db: f64,
    regression_budget_db: f64,
) -> Vec<String> {
    let mut findings = Vec::new();
    for seat in final_seats
        .iter()
        .filter(|seat| seat.partition == "training")
    {
        // NaN lower bounds fail closed: only a strict improvement counts.
        if seat
            .improvement_lower_bound_db
            .partial_cmp(&benefit_floor_db)
            != Some(std::cmp::Ordering::Greater)
        {
            continue;
        }
        let Some(bands) = seat.band_improvement_db.as_ref() else {
            continue;
        };
        for (name, value) in [
            ("bass", bands.bass_improvement_db),
            ("crossover", bands.crossover_improvement_db),
            ("upper", bands.upper_improvement_db),
        ] {
            if let Some(db) = value
                && db < -regression_budget_db
            {
                findings.push(format!(
                    "band_concealment:{}:{}:{}:band_improvement_db={:.3}:broadband_lb_db={:.3}:budget_db={:.3}",
                    seat.logical_input,
                    seat.seat_index,
                    name,
                    db,
                    seat.improvement_lower_bound_db,
                    regression_budget_db,
                ));
            }
        }
    }
    findings.sort();
    findings
}

/// Advisory stage carrying band-concealment findings, if any.
fn concealment_stage(findings: Vec<String>) -> roomeq_model::StageOutcome {
    roomeq_model::StageOutcome {
        stage: "final_band_concealment".into(),
        status: roomeq_model::StageStatus::Applied,
        advisories: if findings.is_empty() {
            vec!["no_band_concealment_detected".to_string()]
        } else {
            findings
        },
        checks: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phased_curve(level_db: f64, phase_deg: f64) -> Curve {
        let mut curve = crate::test_fixtures::flat_curve();
        curve.spl.fill(level_db);
        curve.phase = Some(ndarray::Array1::from_elem(curve.freq.len(), phase_deg));
        curve
    }

    fn concealment_seat(
        partition: &str,
        lower_bound_db: f64,
        bands: Option<roomeq_model::BandImprovement>,
    ) -> FinalSeatEvaluation {
        FinalSeatEvaluation {
            partition: partition.into(),
            logical_input: "L".into(),
            seat_index: 0,
            seat_label: None,
            physical_outputs: vec!["L".into()],
            pre_summation_support: Vec::new(),
            post_summation_support: Vec::new(),
            unassessed_bands_hz: Vec::new(),
            evaluated_band_hz: [20.0, 20_000.0],
            pre_weighted_rms_db: 5.0,
            post_weighted_rms_db: 4.0,
            improvement_db: 1.0,
            improvement_lower_bound_db: lower_bound_db,
            band_improvement_db: bands,
        }
    }

    fn bands(
        bass: Option<f64>,
        crossover: Option<f64>,
        upper: Option<f64>,
    ) -> roomeq_model::BandImprovement {
        roomeq_model::BandImprovement {
            schroeder_hz: 200.0,
            bass_band_hz: [20.0, 200.0],
            bass_improvement_db: bass,
            crossover_band_hz: Some([40.0, 160.0]),
            crossover_improvement_db: crossover,
            upper_band_hz: [200.0, 20_000.0],
            upper_improvement_db: upper,
        }
    }

    #[test]
    fn band_concealment_flags_only_passing_broadband_with_regressed_band() {
        // Broadband gain with a regressed upper band: concealed.
        let seats = vec![concealment_seat(
            "training",
            1.0,
            Some(bands(Some(2.0), Some(0.5), Some(-0.5))),
        )];
        let findings = super::band_concealment_findings(&seats, 0.0, 0.25);
        assert_eq!(findings.len(), 1);
        assert!(
            findings[0].contains("band_concealment:L:0:upper")
                && findings[0].contains("band_improvement_db=-0.500"),
            "unexpected finding: {}",
            findings[0]
        );
        // Broadband itself failing benefit: no concealment question.
        let seats = vec![concealment_seat(
            "training",
            -0.1,
            Some(bands(Some(2.0), Some(0.5), Some(-0.5))),
        )];
        assert!(super::band_concealment_findings(&seats, 0.0, 0.25).is_empty());
        // Clean bands, held-out partition, and absent bands: silent.
        let seats = vec![
            concealment_seat(
                "training",
                1.0,
                Some(bands(Some(2.0), Some(0.5), Some(0.1))),
            ),
            concealment_seat(
                "held_out",
                1.0,
                Some(bands(Some(2.0), Some(0.5), Some(-5.0))),
            ),
            concealment_seat("training", 1.0, None),
        ];
        assert!(super::band_concealment_findings(&seats, 0.0, 0.25).is_empty());
    }

    #[test]
    fn mono_bass_sums_report_coherent_buildup_and_refuse_without_phase() {
        // Identical in-phase inputs: coherent +6.02 dB, buildup +3.01 dB
        // over the incoherent power sum at every mono bin.
        let pair = vec![phased_curve(80.0, 0.0), phased_curve(80.0, 0.0)];
        let assessed = super::assess_mono_bass_sums(&pair, 300.0).unwrap();
        assert_eq!(assessed.inputs, 2);
        assert!((assessed.peak_coherent_db - 86.0206).abs() < 1e-3);
        assert!((assessed.max_buildup_db - 3.0103).abs() < 1e-3);
        assert!(assessed.band_hz[0] <= 20.0 + 1e-6 && assessed.band_hz[1] <= 300.0);
        // Opposite phase cancels instead of building up; the ratio stays
        // finite and deeply negative rather than NaN.
        let opposed = vec![phased_curve(80.0, 0.0), phased_curve(80.0, 180.0)];
        let assessed = super::assess_mono_bass_sums(&opposed, 300.0).unwrap();
        assert!(assessed.max_buildup_db < -100.0);
        // Missing phase refuses rather than invents a coherent sum.
        let mut phaseless = pair.clone();
        phaseless[1].phase = None;
        assert!(
            super::assess_mono_bass_sums(&phaseless, 300.0)
                .unwrap_err()
                .contains("no measured phase")
        );
        // Grid mismatch and single inputs refuse likewise.
        let mut shifted = pair.clone();
        shifted[1].freq[0] += 1.0;
        assert!(
            super::assess_mono_bass_sums(&shifted, 300.0)
                .unwrap_err()
                .contains("common grid")
        );
        assert!(super::assess_mono_bass_sums(&pair[..1], 300.0).is_err());
    }

    #[test]
    fn grouped_physical_output_captures_keep_declared_branch_order() {
        let source = |level| {
            let mut curve = crate::test_fixtures::flat_curve();
            curve.spl.fill(level);
            MeasurementSource::InMemory(curve)
        };
        let groups = [
            SpeakerConfig::MultiSub(roomeq_model::MultiSubGroup {
                name: String::from("subs"),
                speaker_name: None,
                subwoofers: vec![source(70.0), source(80.0)],
                allpass_optimization: false,
                joint_optimization: false,
            }),
            SpeakerConfig::Cardioid(Box::new(roomeq_model::CardioidConfig {
                name: String::from("subs"),
                speaker_name: None,
                front: source(70.0),
                rear: source(80.0),
                separation_meters: 1.0,
            })),
            SpeakerConfig::Dba(roomeq_model::DBAConfig {
                name: String::from("subs"),
                speaker_name: None,
                front: vec![source(70.0)],
                rear: vec![source(80.0)],
            }),
        ];
        for group in groups {
            let mut config = RoomConfig::default();
            config.speakers.insert(String::from("subs"), group);
            config.system = Some(roomeq_model::SystemConfig {
                subwoofers: Some(roomeq_model::SubwooferSystemConfig {
                    config: Default::default(),
                    crossover: None,
                    routing: Default::default(),
                    // Reverse lexical order: mapping follows declaration order.
                    outputs: ["Sub2", "Sub1"]
                        .into_iter()
                        .map(|id| roomeq_model::SubwooferOutput {
                            id: String::from(id),
                            speaker: String::from("subs"),
                        })
                        .collect(),
                }),
                ..Default::default()
            });
            let captures = capture_training(&config).unwrap();
            assert_eq!(captures.len(), 2);
            // Only the common first-output channel exists. The second output
            // is a physical branch, not another logical driver-group owner.
            let result = crate::test_fixtures::single_channel_room_result("Sub2");
            let physical = physical_captures(&captures, &result).unwrap();
            assert_eq!(physical.len(), 2);
            assert!(physical["Sub2"][0].spl.iter().all(|value| *value == 70.0));
            assert!(physical["Sub1"][0].spl.iter().all(|value| *value == 80.0));
            config
                .system
                .as_mut()
                .unwrap()
                .subwoofers
                .as_mut()
                .unwrap()
                .outputs
                .push(roomeq_model::SubwooferOutput {
                    id: String::from("Sub3"),
                    speaker: String::from("subs"),
                });
            assert!(capture_training(&config).is_err());
        }
    }

    #[test]
    fn standalone_subwoofer_replay_uses_measured_band_not_full_range_limit() {
        let input = "Two subs";
        let mut result = crate::test_fixtures::single_channel_room_result(input);
        let chain = result.channels.get_mut(input).unwrap();
        chain.plugins.clear();
        chain.drivers = Some(
            (0..2)
                .map(|index| roomeq_model::DriverDspChain {
                    measured_acoustics: None,
                    name: format!("Two subs_{}", index + 1),
                    index,
                    plugins: vec![],
                    initial_curve: None,
                    measured_band_hz: None,
                })
                .collect(),
        );
        let physical = (0..2)
            .map(|index| {
                let high = if index == 0 { 199.951172 } else { 200.0 };
                let curve = Curve {
                    freq: ndarray::Array1::from_vec(vec![20.0, 40.0, 80.0, 120.0, high]),
                    spl: ndarray::Array1::from_elem(5, 70.0),
                    phase: Some(ndarray::Array1::zeros(5)),
                    ..Default::default()
                };
                (format!("Two subs_{}", index + 1), vec![curve])
            })
            .collect();
        let mut config = RoomConfig::default();
        config.optimizer.min_freq = 20.0;
        config.optimizer.max_freq = 16_000.0;
        config.speakers.insert(
            input.into(),
            SpeakerConfig::MultiSub(roomeq_model::MultiSubGroup {
                name: input.into(),
                speaker_name: None,
                subwoofers: vec![],
                allpass_optimization: false,
                joint_optimization: false,
            }),
        );
        let playback = replay_final_physical_seat(
            &result,
            &physical,
            input,
            0,
            &config,
            48_000.0,
            Path::new("."),
            "training",
        )
        .expect("a standalone subwoofer needs evidence only in its measured band");
        assert_eq!(*playback.baseline.freq.last().unwrap(), 199.951172);
        assert_eq!(playback.baseline.freq, playback.delivered.freq);
        assert!(
            playback
                .baseline
                .spl
                .iter()
                .all(|spl| (*spl - 76.0206).abs() < 0.001)
        );

        // The same short captures must not silently narrow a full-range sum.
        config.speakers.clear();
        let error = replay_final_physical_seat(
            &result,
            &physical,
            input,
            0,
            &config,
            48_000.0,
            Path::new("."),
            "training",
        )
        .expect_err("full-range driver sums still require full-band evidence");
        assert!(
            error
                .to_string()
                .contains("insufficient summation evidence")
        );
    }
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
        assert_eq!(
            serde_json::to_value(report).unwrap()["decision"],
            "rejected"
        );
        let seats = &report.acoustic_quality.as_ref().unwrap().final_seats;
        assert_eq!(seats.len(), 2);
        assert_eq!(seats[1].physical_outputs, vec!["left"]);
        assert!(seats[1].improvement_db < 0.0);
    }

    #[test]
    fn independent_group_replays_declared_physical_outputs_under_their_owner() {
        let (mut result, _, mut flat) = fixture();
        flat.phase = Some(ndarray::Array1::zeros(flat.freq.len()));
        let chain = result.channels.get_mut("left").unwrap();
        chain.plugins.clear();
        chain.drivers = Some(
            ["Sub1", "Sub2"]
                .into_iter()
                .enumerate()
                .map(|(index, name)| roomeq_model::DriverDspChain {
                    measured_acoustics: None,
                    name: name.into(),
                    index,
                    plugins: vec![],
                    initial_curve: None,
                    measured_band_hz: None,
                })
                .collect(),
        );
        let captures: Vec<_> = ["Sub1", "Sub2"]
            .into_iter()
            .map(|name| Capture {
                channel: name.into(),
                driver: None,
                seat_labels: None,
                curves: vec![flat.clone(), flat.clone()],
            })
            .collect();
        let baseline = result.clone();
        validate_final_seats_impl(
            &mut result,
            &captures,
            &HashMap::new(),
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
            FinalSeatReplayContext {
                baseline: Some(&baseline),
                trace: None,
            },
        )
        .unwrap();
        let seats = &result
            .metadata
            .correction_acceptance
            .as_ref()
            .unwrap()
            .acoustic_quality
            .as_ref()
            .unwrap()
            .final_seats;
        assert_eq!(seats.len(), 2);
        for seat in seats {
            assert_eq!(seat.physical_outputs, vec!["Sub1", "Sub2"]);
            assert!(seat.improvement_db.abs() < 1e-8);
        }

        let mut incomplete = captures;
        incomplete[1].curves.pop();
        let error = validate_final_seats_impl(
            &mut result,
            &incomplete,
            &HashMap::new(),
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
            FinalSeatReplayContext {
                baseline: Some(&baseline),
                trace: None,
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("singleton captures are not broadcast")
        );

        let duplicate = result.channels["left"].clone();
        result.channels.insert("duplicate".into(), duplicate);
        assert!(
            independent_input_channels(&incomplete, &result)
                .unwrap_err()
                .to_string()
                .contains("ambiguous DSP owners")
        );
        result.channels.clear();
        assert!(
            independent_input_channels(&incomplete, &result)
                .unwrap_err()
                .to_string()
                .contains("no DSP owner")
        );
    }

    #[test]
    fn final_seat_overlap_preserves_per_input_support_and_reports_disjoint_inputs() {
        let seat = |input: &str, index, band| FinalSeatEvaluation {
            partition: "training".into(),
            logical_input: input.into(),
            seat_index: index,
            seat_label: None,
            physical_outputs: vec![input.into()],
            pre_summation_support: vec![],
            post_summation_support: vec![],
            unassessed_bands_hz: vec![],
            evaluated_band_hz: band,
            pre_weighted_rms_db: 0.0,
            post_weighted_rms_db: 0.0,
            improvement_db: 0.0,
            improvement_lower_bound_db: 0.0,
            band_improvement_db: None,
        };
        let mut seats = vec![
            seat("sub", 0, [20.0, 150.0]),
            seat("sub", 1, [25.0, 140.0]),
            seat("main", 0, [200.0, 20000.0]),
            seat("main", 1, [250.0, 18000.0]),
        ];
        assert_eq!(final_seat_overlap(&seats).unwrap(), None);
        assert_eq!(
            final_seat_overlap(&seats[..2]).unwrap(),
            Some([25.0, 140.0])
        );

        // A held-out seat of the SAME input still needs common support.
        seats[1].partition = "held_out".into();
        seats[1].evaluated_band_hz = [160.0, 200.0];
        assert!(
            final_seat_overlap(&seats)
                .unwrap_err()
                .to_string()
                .contains("input 'sub' has no common supported frequency band")
        );
        seats[1].evaluated_band_hz = [f64::NAN, 200.0];
        assert!(final_seat_overlap(&seats).is_err());
        assert!(final_seat_overlap(&[]).is_err());

        let (mut result, _, flat) = fixture();
        result.channels.get_mut("left").unwrap().plugins.clear();
        let baseline = result.clone();
        validate_final_seats_impl(
            &mut result,
            &[Capture {
                channel: "left".into(),
                driver: None,
                seat_labels: None,
                curves: vec![flat.clone(), flat],
            }],
            &HashMap::new(),
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
            FinalSeatReplayContext {
                baseline: Some(&baseline),
                trace: None,
            },
        )
        .unwrap();
        let mut sub = result
            .metadata
            .correction_acceptance
            .unwrap()
            .acoustic_quality
            .unwrap();
        sub.measurement_overlap_hz = Some([20.0, 150.0]);
        let mut main = sub.clone();
        main.measurement_overlap_hz = Some([200.0, 20000.0]);
        let aggregate = super::super::room_optimization_result::aggregate_runtime_quality(
            &[sub.clone(), main],
            Default::default(),
            20.0,
            20000.0,
        )
        .unwrap();
        assert_eq!(aggregate.measurement_overlap_hz, None);
        let json = serde_json::to_value(&aggregate).unwrap();
        assert!(json.get("measurement_overlap_hz").is_none());
        let decoded: roomeq_model::AcousticQualityScorecard = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.measurement_overlap_hz, None);
        let legacy = serde_json::to_value(&sub).unwrap();
        assert_eq!(
            legacy["measurement_overlap_hz"],
            serde_json::json!([20.0, 150.0])
        );
        let decoded: roomeq_model::AcousticQualityScorecard =
            serde_json::from_value(legacy).unwrap();
        assert_eq!(decoded.measurement_overlap_hz, Some([20.0, 150.0]));
    }

    #[test]
    fn final_seat_rejection_retains_both_shape_and_output_failures() {
        let (mut result, _, flat) = fixture();
        result.channels.get_mut("left").unwrap().plugins = vec![
            roomeq_engine::output::create_eq_plugin(&[math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                120.0,
                48_000.0,
                0.5,
                -30.0,
            )]),
        ];
        let captures = vec![Capture {
            channel: "left".into(),
            driver: None,
            seat_labels: None,
            curves: vec![flat.clone(), flat],
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
        let report = result.metadata.correction_acceptance.as_ref().unwrap();
        for reason in ["worst_position_regressed", "unexplained_useful_output_loss"] {
            assert!(
                report.violations.iter().any(|v| v == reason),
                "missing {reason}: {report:?}"
            );
        }
        assert!(error.to_string().contains("regressed"));
        assert!(error.to_string().contains("useful output"));
        assert!(error.to_string().contains("seat 0"));
        assert!(error.to_string().contains("seat 1"));
        assert_eq!(report.decision, roomeq_model::CorrectionDecision::Rejected);
    }

    #[test]
    fn final_seat_replay_rejects_sidecar_conflicting_with_retained_fir() {
        let directory = tempfile::tempdir().unwrap();
        let (mut result, _, flat) = fixture();
        let plugins = vec![roomeq_engine::output::create_convolution_plugin(
            "retained.wav",
        )];
        result.channels.get_mut("left").unwrap().plugins = plugins.clone();
        let tap = 0.123456789_f64;
        result.channel_results.get_mut("left").unwrap().fir_coeffs = Some(vec![tap]);
        let write = |value: f32| {
            let mut writer = hound::WavWriter::create(
                directory.path().join("retained.wav"),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 48000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            writer.write_sample(value).unwrap();
            writer.finalize().unwrap();
        };
        write(tap as f32);
        assert!(
            super::apply(
                &result,
                "left",
                plugins.clone(),
                &flat,
                false,
                48000.0,
                directory.path()
            )
            .is_ok(),
            "normal float32 serialization rounding must remain valid"
        );
        write(0.5);
        let outcome = super::apply(
            &result,
            "left",
            plugins,
            &flat,
            false,
            48000.0,
            directory.path(),
        );
        assert!(
            outcome.is_err(),
            "stale sidecar silently replaced retained FIR evidence"
        );
        assert!(
            outcome
                .unwrap_err()
                .to_string()
                .contains("retained FIR coefficients")
        );
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
                    assert_eq!(
                        serde_json::to_value(report).unwrap()["decision"],
                        "rejected"
                    );
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
    fn configured_useful_output_budget_accepts_only_the_authorized_loss() {
        for (budget, accepted) in [(3.0, false), (5.0, true)] {
            let (mut result, _, flat) = fixture();
            result.channels.get_mut("left").unwrap().plugins =
                vec![roomeq_engine::output::create_convolution_plugin(
                    "output-budget.wav",
                )];
            result.channel_results.get_mut("left").unwrap().fir_coeffs =
                Some(vec![10.0_f64.powf(-4.0 / 20.0)]);
            let captures = vec![Capture {
                channel: "left".into(),
                driver: None,
                seat_labels: None,
                curves: vec![flat.clone(), flat],
            }];
            let mut config = RoomConfig::default();
            config.optimizer.finalization.max_useful_output_loss_db = budget;
            let outcome = validate_final_seats(
                &mut result,
                &captures,
                &HashMap::new(),
                &config,
                48_000.0,
                Path::new("."),
            );
            assert_eq!(outcome.is_ok(), accepted, "budget={budget}: {outcome:?}");
            assert_eq!(config.optimizer.finalization.default_input_peak, 1.0);
        }
    }

    #[test]
    fn subwoofer_loss_is_exempt_but_main_surround_and_height_loss_is_not() {
        for name in ["LFE", "Sub1", "L", "R", "SL", "TFL"] {
            let (mut result, _, flat) = fixture();
            let mut channel = result.channels.remove("left").unwrap();
            channel.channel = name.into();
            channel.plugins = vec![roomeq_engine::output::create_convolution_plugin("loss.wav")];
            result.channels.insert(name.into(), channel);
            let mut channel_result = result.channel_results.remove("left").unwrap();
            channel_result.name = name.into();
            channel_result.fir_coeffs = Some(vec![10.0_f64.powf(-10.0 / 20.0)]);
            result.channel_results.insert(name.into(), channel_result);
            let captures = vec![Capture {
                channel: name.into(),
                driver: None,
                seat_labels: None,
                curves: vec![flat.clone(), flat],
            }];
            let outcome = validate_final_seats(
                &mut result,
                &captures,
                &HashMap::new(),
                &RoomConfig::default(),
                48_000.0,
                Path::new("."),
            );
            assert_eq!(
                outcome.is_ok(),
                matches!(name, "LFE" | "Sub1"),
                "{name}: {outcome:?}"
            );
            let evidence = result
                .metadata
                .correction_acceptance
                .as_ref()
                .unwrap()
                .acoustic_quality
                .as_ref()
                .unwrap();
            assert_eq!(evidence.useful_output.len(), 2);
            assert!(
                evidence
                    .useful_output
                    .iter()
                    .all(|seat| (seat.unexplained_loss_rms_db - 10.0).abs() < 1e-5)
            );
        }
    }

    #[test]
    fn independent_bass_only_correction_still_checks_the_speaker_passband() {
        let (mut result, _, flat) = fixture();
        let mut config = RoomConfig::default();
        config.optimizer.min_freq = 40.0;
        config.optimizer.max_freq = 200.0;
        let cut = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Highshelf,
            1_000.0,
            48_000.0,
            0.7,
            -12.0,
        );
        result.channels.get_mut("left").unwrap().plugins =
            vec![roomeq_engine::output::create_eq_plugin(&[cut])];
        let captures = vec![Capture {
            channel: "left".into(),
            driver: None,
            seat_labels: None,
            curves: vec![flat.clone(), flat],
        }];
        validate_final_seats(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48_000.0,
            Path::new("."),
        )
        .expect_err("a bass-only request must not hide treble loss");
        let report = result.metadata.correction_acceptance.as_ref().unwrap();
        let quality = report.acoustic_quality.as_ref().unwrap();
        assert!(
            quality
                .final_seats
                .iter()
                .all(|seat| seat.evaluated_band_hz[1] > 1_000.0)
        );
        assert_eq!(quality.correction_band_hz, Some([40.0, 200.0]));
        assert!(
            report
                .violations
                .iter()
                .any(|v| v == "unexplained_useful_output_loss")
        );
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
    fn native_capture_preserves_distinct_grids_and_auxiliary_evidence() {
        let first = Curve {
            freq: vec![40.0, 80.0, 120.0].into(),
            spl: vec![80.0; 3].into(),
            ..Default::default()
        };
        let second = Curve {
            freq: vec![20.0, 40.0, 79.0, 80.0, 81.0, 120.0, 200.0].into(),
            spl: vec![80.0, 80.0, 80.0, 95.0, 80.0, 80.0, 80.0].into(),
            phase: Some(vec![0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0].into()),
            coherence: Some(vec![0.9; 7].into()),
            noise_floor_db: Some(vec![20.0; 7].into()),
            ..Default::default()
        };
        let originals = vec![first, second];
        let config = RoomConfig {
            speakers: HashMap::from([(
                "left".into(),
                SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(originals.clone())),
            )]),
            ..Default::default()
        };
        let (captured, receipt) = capture_with_receipt(&config, &HashMap::new()).unwrap();
        let payload: serde_json::Value =
            serde_json::from_str(receipt.checks[0].diagnostic.as_ref().unwrap()).unwrap();
        for (index, original) in originals.iter().enumerate() {
            assert_eq!(
                payload["takes"][index]["curve"],
                serde_json::to_value(original).unwrap()
            );
        }
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].curves.len(), originals.len());
        for (actual, original) in captured[0].curves.iter().zip(&originals) {
            assert_eq!(actual.freq, original.freq);
            assert_eq!(actual.spl, original.spl);
            assert_eq!(actual.phase, original.phase);
            assert_eq!(actual.coherence, original.coherence);
            assert_eq!(actual.noise_floor_db, original.noise_floor_db);
        }
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
    fn routed_single_position_captures_do_not_compare_measurement_names() {
        // Regression (measured 2.1_sigberg2): single-seat file captures are
        // labeled with their measurement names ("Left" vs "Sub"), which are
        // not seat orders. Routed replay must accept them.
        let (result, _, flat) = routed_fixture();
        let captures = vec![
            Capture {
                channel: "left".into(),
                driver: None,
                curves: vec![flat.clone()],
                seat_labels: Some(vec!["Left".into()]),
            },
            Capture {
                channel: "sub".into(),
                driver: None,
                curves: vec![flat],
                seat_labels: Some(vec!["Sub".into()]),
            },
        ];
        let physical = physical_captures(&captures, &result)
            .expect("single-position routed captures must replay");
        assert!(physical.contains_key("left"));
        assert!(physical.contains_key("sub"));
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
                    routing: Default::default(),
                    outputs: Vec::new(),
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
            physical_sub_outputs: Vec::new(),
            stereo_routing: None,
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

    fn lfe_to_sub_fixture(
        add_second_sub: bool,
    ) -> (RoomOptimizationResult, RoomConfig, Curve, Vec<Capture>) {
        let (mut result, config, flat) = routed_fixture();
        let sub_chain = result.channels.remove("sub").unwrap();
        let sub_response = result.channel_results.remove("sub").unwrap();
        result.channels.insert("Sub1".into(), sub_chain);
        result.channel_results.insert("Sub1".into(), sub_response);
        let mut config = config;
        config
            .system
            .as_mut()
            .unwrap()
            .speakers
            .insert("LFE".into(), "Sub1".into());
        let graph = result
            .metadata
            .bass_management
            .as_mut()
            .unwrap()
            .routing_graph
            .as_mut()
            .unwrap();
        let mut lfe_route = graph.routes[1].clone();
        lfe_route.source_channel = "LFE".into();
        lfe_route.source_index = 0;
        lfe_route.pre_chain_channel = None;
        lfe_route.destination = "Sub1".into();
        lfe_route.post_chain_channel = Some("Sub1".into());
        lfe_route.route_kind = "lfe_lowpass_to_sub".into();
        lfe_route.low_pass_hz = Some(120.0);
        graph.input_channels = vec!["LFE".into()];
        graph.routes = vec![lfe_route.clone()];
        if add_second_sub {
            let sub_result = crate::test_fixtures::single_channel_room_result("Sub2");
            result
                .channels
                .insert("Sub2".into(), sub_result.channels["Sub2"].clone());
            result
                .channel_results
                .insert("Sub2".into(), sub_result.channel_results["Sub2"].clone());
            let mut second_route = lfe_route;
            second_route.destination = "Sub2".into();
            second_route.destination_index = 2;
            second_route.post_chain_channel = Some("Sub2".into());
            second_route.gain_db = 0.0;
            second_route.gain_linear = 1.0;
            second_route.matrix_gain = 1.0;
            graph.routes.push(second_route);
            graph.output_channels = vec!["Sub1".into(), "Sub2".into()];
            graph.physical_sub_output = "Sub1".into();
            graph.physical_sub_outputs = vec!["Sub1".into(), "Sub2".into()];
        }
        let outputs = if add_second_sub {
            vec!["Sub1", "Sub2"]
        } else {
            vec!["Sub1"]
        };
        let captures = outputs
            .into_iter()
            .map(|channel| Capture {
                channel: channel.into(),
                driver: None,
                curves: vec![if channel == "Sub2" {
                    let mut curve = flat.clone();
                    curve.spl.mapv_inplace(|level| level - 3.0);
                    curve
                        .phase
                        .as_mut()
                        .unwrap()
                        .mapv_inplace(|phase| phase + 17.0);
                    curve
                } else {
                    flat.clone()
                }],
                seat_labels: None,
            })
            .collect();
        (result, config, flat, captures)
    }

    fn assert_trace_is_observational(
        result: RoomOptimizationResult,
        config: &RoomConfig,
        captures: &[Capture],
    ) -> (RoomOptimizationResult, Vec<UsefulOutputReplayCapture>) {
        let baseline = result.clone();
        let mut normal = result.clone();
        let mut traced = result;
        let normal_result = validate_candidate_final_seats(
            &mut normal,
            &baseline,
            captures,
            &HashMap::new(),
            config,
            48_000.0,
            Path::new("."),
        );
        let mut trace = Vec::new();
        let traced_result = validate_candidate_final_seats_with_diagnostic_trace(
            &mut traced,
            &baseline,
            captures,
            &HashMap::new(),
            config,
            48_000.0,
            Path::new("."),
            &mut trace,
        );
        assert_eq!(
            normal_result.as_ref().map_err(ToString::to_string),
            traced_result.as_ref().map_err(ToString::to_string),
            "diagnostic tracing changed validation outcome"
        );
        let normal_report = normal.metadata.correction_acceptance.as_ref().unwrap();
        let traced_report = traced.metadata.correction_acceptance.as_ref().unwrap();
        assert_eq!(
            serde_json::to_value(normal_report).unwrap(),
            serde_json::to_value(traced_report).unwrap(),
            "diagnostic tracing changed score or acceptance evidence"
        );
        (traced, trace)
    }

    #[test]
    fn diagnostic_trace_uses_lfe_sub1_physical_capture_without_changing_validation() {
        let (result, config, raw_sub, captures) = lfe_to_sub_fixture(false);
        let (traced, trace) = assert_trace_is_observational(result, &config, &captures);
        assert_eq!(trace.len(), 1);
        let replay = &trace[0];
        assert_eq!(replay.logical_input, "LFE");
        assert_eq!(
            replay
                .baseline_measurement_contributors
                .iter()
                .map(|contributor| contributor.physical_output.as_str())
                .collect::<Vec<_>>(),
            ["Sub1"]
        );
        assert_eq!(
            replay
                .delivered_measurement_contributors
                .iter()
                .map(|contributor| contributor.physical_output.as_str())
                .collect::<Vec<_>>(),
            ["Sub1"]
        );
        for contributor in replay
            .baseline_measurement_contributors
            .iter()
            .chain(&replay.delivered_measurement_contributors)
        {
            assert_eq!(contributor.measured_curve.freq, raw_sub.freq);
            assert_eq!(contributor.measured_curve.spl, raw_sub.spl);
        }
        assert!(
            traced
                .metadata
                .correction_acceptance
                .as_ref()
                .unwrap()
                .acoustic_quality
                .is_some()
        );
    }

    #[test]
    fn diagnostic_trace_records_every_physical_contributor_to_a_logical_sum() {
        let (result, config, _, captures) = lfe_to_sub_fixture(true);
        let (_, trace) = assert_trace_is_observational(result, &config, &captures);
        assert_eq!(trace.len(), 1);
        let replay = &trace[0];
        for contributors in [
            &replay.baseline_measurement_contributors,
            &replay.delivered_measurement_contributors,
        ] {
            assert_eq!(
                contributors
                    .iter()
                    .map(|contributor| contributor.physical_output.as_str())
                    .collect::<Vec<_>>(),
                ["Sub1", "Sub2"]
            );
            for contributor in contributors {
                let expected = captures
                    .iter()
                    .find(|capture| capture.channel == contributor.physical_output)
                    .unwrap()
                    .curves
                    .first()
                    .unwrap();
                assert_eq!(contributor.measured_curve.freq, expected.freq);
                assert_eq!(contributor.measured_curve.spl, expected.spl);
                assert_eq!(contributor.measured_curve.phase, expected.phase);
            }
        }
    }

    #[test]
    fn diagnostic_trace_does_not_turn_missing_routed_capture_into_acceptance() {
        let (result, config, flat, _) = lfe_to_sub_fixture(false);
        let missing_sub_capture = [Capture {
            channel: "left".into(),
            driver: None,
            curves: vec![flat],
            seat_labels: None,
        }];
        let baseline = result.clone();
        let mut normal = result.clone();
        let mut traced = result;
        let normal_error = validate_candidate_final_seats(
            &mut normal,
            &baseline,
            &missing_sub_capture,
            &HashMap::new(),
            &config,
            48_000.0,
            Path::new("."),
        )
        .unwrap_err()
        .to_string();
        let mut trace = Vec::new();
        let traced_error = validate_candidate_final_seats_with_diagnostic_trace(
            &mut traced,
            &baseline,
            &missing_sub_capture,
            &HashMap::new(),
            &config,
            48_000.0,
            Path::new("."),
            &mut trace,
        )
        .unwrap_err()
        .to_string();
        assert_eq!(normal_error, traced_error);
        assert!(
            normal_error.contains("physical output 'Sub1'"),
            "{normal_error}"
        );
        assert!(trace.is_empty());
        assert_eq!(
            serde_json::to_value(normal.metadata.correction_acceptance.as_ref().unwrap()).unwrap(),
            serde_json::to_value(traced.metadata.correction_acceptance.as_ref().unwrap()).unwrap()
        );
    }

    #[test]
    fn routed_spl_budget_measures_main_output_not_redirected_sub_loss() {
        for main_loss in [0.0, 10.0] {
            let (mut result, mut config, flat) = routed_fixture();
            config.optimizer.min_freq = 40.0;
            config.optimizer.max_freq = 200.0;
            let graph = result
                .metadata
                .bass_management
                .as_mut()
                .unwrap()
                .routing_graph
                .as_mut()
                .unwrap();
            graph.routes[0].high_pass_hz = Some(80.0);
            graph.routes[1].low_pass_hz = Some(80.0);
            graph.routes[1].gain_db = 0.0;
            graph.routes[1].gain_linear = 1.0;
            graph.routes[1].matrix_gain = 1.0;
            graph.routes[1].delay_ms = 0.0;
            result.channels.get_mut("sub").unwrap().plugins.clear();
            for (output, loss) in [("sub", 18.0), ("left", main_loss)] {
                let mut gain = roomeq_engine::output::create_gain_plugin(-loss);
                gain.parameters["room_eq_stage"] = serde_json::json!("post_route");
                gain.parameters["room_eq_correction_gain"] = serde_json::json!(true);
                result.channels.get_mut(output).unwrap().plugins.push(gain);
            }
            let captures: Vec<_> = ["left", "sub"]
                .into_iter()
                .map(|channel| Capture {
                    channel: channel.into(),
                    driver: None,
                    seat_labels: None,
                    curves: vec![flat.clone(), flat.clone()],
                })
                .collect();
            let error = validate_final_seats(
                &mut result,
                &captures,
                &HashMap::new(),
                &config,
                48_000.0,
                Path::new("."),
            )
            .unwrap_err();
            let report = result.metadata.correction_acceptance.as_ref().unwrap();
            let quality = report.acoustic_quality.as_ref().unwrap();
            assert_eq!(quality.useful_output.len(), 2);
            for output in &quality.useful_output {
                assert_eq!(output.logical_input.as_deref(), Some("left"));
                assert!(output.evaluated_band_hz[0] >= 80.0, "{output:?}");
                assert!(output.evaluated_band_hz[1] > 1000.0);
                assert!((output.unexplained_loss_rms_db - main_loss).abs() < 1e-6);
            }
            assert_eq!(
                report
                    .violations
                    .iter()
                    .any(|v| v == "unexplained_useful_output_loss"),
                main_loss > 3.0,
                "{error}"
            );
            // Exempting sub SPL loss must not authorize a damaged bass response.
            assert!(
                report
                    .violations
                    .iter()
                    .any(|v| v == "worst_position_regressed"),
                "{error}"
            );
            assert!(
                quality
                    .final_seats
                    .iter()
                    .all(|seat| seat.evaluated_band_hz[0] < 80.0)
            );
        }
    }

    #[test]
    fn routed_finalization_and_replay_reject_unowned_channel_stages() {
        for tag in [
            None,
            Some(serde_json::json!(null)),
            Some(serde_json::json!(42)),
            Some(serde_json::json!("post_rout")),
        ] {
            let (mut result, config, flat) = routed_fixture();
            let mut plugin = roomeq_engine::output::create_gain_plugin(-20.0);
            if let Some(tag) = tag {
                plugin.parameters["room_eq_stage"] = tag;
            }
            result
                .channels
                .get_mut("left")
                .unwrap()
                .plugins
                .push(plugin);
            let physical = BTreeMap::from([
                ("left".into(), vec![flat.clone()]),
                ("sub".into(), vec![flat]),
            ]);
            assert!(
                replay_final_physical_seat(
                    &result,
                    &physical,
                    "left",
                    0,
                    &config,
                    48_000.0,
                    Path::new("."),
                    "training",
                )
                .is_err(),
                "unowned gain disappeared from routed playback evidence"
            );
            let store = autoeq_artifacts::MemoryArtifactStore::new();
            assert!(
                crate::export::bind_final_convolution_artifacts(
                    &mut result,
                    Path::new("."),
                    &store,
                    48_000.0,
                )
                .is_err(),
                "malformed routed graph was finalized"
            );
        }
    }

    #[test]
    fn channel_matching_is_applied_to_the_complete_routed_source() {
        let (mut result, config, flat) = routed_fixture();
        let physical = BTreeMap::from([
            ("left".into(), vec![flat.clone()]),
            ("sub".into(), vec![flat]),
        ]);
        let before = replay_final_physical_seat(
            &result,
            &physical,
            "left",
            0,
            &config,
            48_000.0,
            Path::new("."),
            "training",
        )
        .unwrap()
        .delivered;
        let filters = vec![math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            120.0,
            48_000.0,
            1.0,
            3.0,
        )];
        let correction = roomeq_engine::spectral_align::ChannelMatchingResult {
            channel_name: "left".into(),
            filters: filters.clone(),
        };
        super::super::reports::apply_channel_matching_correction(
            &mut result,
            &correction,
            48_000.0,
        );
        let after = replay_final_physical_seat(
            &result,
            &physical,
            "left",
            0,
            &config,
            48_000.0,
            Path::new("."),
            "training",
        )
        .unwrap()
        .delivered;
        let response =
            roomeq_engine::response::compute_peq_complex_response(&filters, &before.freq, 48_000.0);
        let expected = roomeq_engine::response::apply_complex_response(&before, &response);
        let maximum_error = after
            .spl
            .iter()
            .zip(expected.spl.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            maximum_error < 1e-8,
            "routed matching transfer differs by {maximum_error} dB"
        );
    }

    #[test]
    fn routed_observation_respects_limited_main_and_lfe_passbands() {
        let (mut result, mut config, _) = routed_fixture();
        config.optimizer.min_freq = 40.0;
        config.optimizer.max_freq = 200.0;
        let mut main = Curve {
            freq: ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 192),
            spl: ndarray::Array1::from_elem(192, 80.0),
            ..Curve::default()
        };
        for (frequency, spl) in main.freq.iter().zip(main.spl.iter_mut()) {
            if *frequency > 4_000.0 {
                *spl -= 50.0;
            }
        }
        let sub = Curve {
            freq: ndarray::array![20.0, 40.0, 80.0, 160.0, 250.0],
            spl: ndarray::Array1::from_elem(5, 80.0),
            ..Curve::default()
        };
        let graph = result
            .metadata
            .bass_management
            .as_mut()
            .unwrap()
            .routing_graph
            .as_mut()
            .unwrap();
        graph.routes[1].low_pass_hz = Some(80.0);
        let mut lfe = graph.routes[1].clone();
        lfe.source_channel = "LFE".into();
        graph.routes.push(lfe);
        let physical = BTreeMap::from([("left".into(), vec![main]), ("sub".into(), vec![sub])]);
        let observation =
            passband_observation_config(&config, &result, &physical, "left", 0).unwrap();
        assert!(observation.optimizer.max_freq > 2_000.0);
        assert!(
            observation.optimizer.max_freq < 10_000.0,
            "a measured main stopband is not useful playback support"
        );
        let lfe = passband_observation_config(&config, &result, &physical, "LFE", 0).unwrap();
        assert_eq!(lfe.optimizer.max_freq, 80.0);
        assert_eq!(
            config.optimizer.max_freq, 200.0,
            "observation must not change correction bounds"
        );
    }

    #[test]
    fn bass_only_optimization_detects_out_of_band_damage_in_routed_playback() {
        let (mut result, mut config, _) = routed_fixture();
        config.optimizer.min_freq = 40.0;
        config.optimizer.max_freq = 200.0;
        let main = Curve {
            freq: ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 192),
            spl: ndarray::Array1::from_elem(192, 80.0),
            phase: Some(ndarray::Array1::zeros(192)),
            ..Curve::default()
        };
        let sub = Curve {
            freq: ndarray::array![20.0, 40.0, 80.0, 160.0, 250.0],
            spl: ndarray::Array1::from_elem(5, 50.0),
            phase: Some(ndarray::Array1::zeros(5)),
            ..Curve::default()
        };
        result
            .metadata
            .bass_management
            .as_mut()
            .unwrap()
            .routing_graph
            .as_mut()
            .unwrap()
            .routes[1]
            .low_pass_hz = Some(80.0);
        let captures = vec![
            Capture {
                channel: "left".into(),
                driver: None,
                curves: vec![main.clone(), main],
                seat_labels: None,
            },
            Capture {
                channel: "sub".into(),
                driver: None,
                curves: vec![sub.clone(), sub],
                seat_labels: None,
            },
        ];
        let harmful = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Highshelf,
            1_000.0,
            48_000.0,
            0.7,
            -12.0,
        );
        let mut plugin = roomeq_engine::output::create_eq_plugin(&[harmful]);
        plugin.parameters["room_eq_stage"] = serde_json::json!("post_route");
        result
            .channels
            .get_mut("left")
            .unwrap()
            .plugins
            .push(plugin);

        let error = validate_final_seats(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48_000.0,
            Path::new("."),
        )
        .expect_err("a bass-only EQ request must not hide treble damage");
        assert!(
            error.to_string().contains("regressed") || error.to_string().contains("useful output"),
            "{error}"
        );
        let score = result
            .metadata
            .correction_acceptance
            .as_ref()
            .unwrap()
            .acoustic_quality
            .as_ref()
            .unwrap();
        assert!(score.final_seats[0].evaluated_band_hz[1] > 10_000.0);
        assert_eq!(score.correction_band_hz, Some([40.0, 200.0]));
    }

    #[test]
    fn routed_subwoofer_stopband_replays_full_main_band_without_declaration() {
        for per_driver in [false, true] {
            let (mut result, mut config, _) = routed_fixture();
            config.optimizer.min_freq = 20.0;
            config.optimizer.max_freq = 16_000.0;
            if per_driver {
                let mut low_pass =
                    roomeq_engine::output::create_crossover_plugin("LR24", 80.0, "low");
                low_pass.parameters["room_eq_stage"] = serde_json::json!("post_route");
                result.channels.get_mut("sub").unwrap().drivers =
                    Some(vec![roomeq_model::DriverDspChain {
                        measured_acoustics: None,
                        name: "sub".into(),
                        index: 0,
                        initial_curve: None,
                        measured_band_hz: None,
                        plugins: vec![low_pass],
                    }]);
            } else {
                result
                    .metadata
                    .bass_management
                    .as_mut()
                    .unwrap()
                    .routing_graph
                    .as_mut()
                    .unwrap()
                    .routes[1]
                    .low_pass_hz = Some(80.0);
            }
            let curve = |frequencies: Vec<f64>, level| {
                let len = frequencies.len();
                Curve {
                    freq: frequencies.into(),
                    spl: ndarray::Array1::from_elem(len, level),
                    phase: Some(ndarray::Array1::zeros(len)),
                    ..Default::default()
                }
            };
            let physical = BTreeMap::from([
                (
                    "left".into(),
                    vec![curve(vec![20.0, 100.0, 250.0, 1000.0, 16_000.0], 80.0)],
                ),
                ("sub".into(), vec![curve(vec![20.0, 100.0, 250.0], 50.0)]),
            ]);
            for partition in ["training", "held_out"] {
                for baseline in [false, true] {
                    let (sum, _) = replay(
                        &result,
                        &physical,
                        "left",
                        0,
                        baseline,
                        &ReplayContext {
                            config: &config,
                            partition,
                            fs: 48_000.0,
                            dir: Path::new("."),
                        },
                    )
                    .unwrap();
                    assert_eq!(*sum.curve.freq.last().unwrap(), 16_000.0);
                    assert_eq!(sum.support.len(), 1);
                    assert!(
                        serde_json::to_string(&sum.support)
                            .unwrap()
                            .contains("assumed_subwoofer_stopband_below_measured_tail")
                    );
                }
            }
        }
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
                    rolloff_db_per_oct: None,
                    evidence_id: format!("analytic-qualified-stopband-seat-{seat_index}"),
                })
                .collect(),
        );
        config.optimizer.min_freq = 20.0;
        config.optimizer.max_freq = 20_000.0;
        config.optimizer.schroeder_split = Some(roomeq_model::SchroederSplitConfig {
            enabled: true,
            schroeder_freq: 200.0,
            ..Default::default()
        });
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
        assert!(
            score.training.upper_pre_weighted_rms_db.is_some()
                && score.training.upper_post_weighted_rms_db.is_some(),
            "final-seat replay must preserve the configured Schroeder split"
        );
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
                rolloff_db_per_oct: None,
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
