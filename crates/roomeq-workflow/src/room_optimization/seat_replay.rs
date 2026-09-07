//! Raw, position-identified evidence survives optimization until final replay.
use super::RoomOptimizationResult;
use roomeq_model::{
    AutoeqError, ChannelDspChain, Curve, FinalSeatEvaluation, MeasurementSource,
    PluginConfigWrapper, Result, RoomConfig, SpeakerConfig,
};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

type DriverIdentity = (Option<String>, usize);
type SourceCapture<'a> = (Option<DriverIdentity>, &'a MeasurementSource);

pub(super) struct Capture {
    channel: String,
    driver: Option<DriverIdentity>,
    curves: Vec<Curve>,
    seat_labels: Option<Vec<String>>,
}

fn invalid(message: impl Into<String>) -> AutoeqError {
    AutoeqError::InvalidMeasurement {
        message: message.into(),
    }
}

pub(super) fn capture_training(config: &RoomConfig) -> Result<Vec<Capture>> {
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
    chain.plugins = plugins;
    chain.drivers = None;
    let mut embedded = HashMap::new();
    // Prefer the actual sidecars. In-memory single-FIR workflows retain the
    // one authoritative coefficient array; never guess among multiple FIRs.
    let paths: Vec<_> = chain
        .plugins
        .iter()
        .filter(|p| p.plugin_type == "convolution")
        .filter_map(|p| p.parameters.get("ir_file").and_then(|v| v.as_str()))
        .collect();
    if paths.len() == 1
        && !dir.join(paths[0]).exists()
        && let Some(taps) = result
            .channel_results
            .get(owner)
            .and_then(|c| c.fir_coeffs.as_ref())
    {
        embedded.insert(paths[0].to_string(), taps.clone());
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

fn sum(curves: &[Curve]) -> Result<Curve> {
    if curves.len() == 1 {
        return Ok(curves[0].clone());
    }
    if curves.is_empty()
        || curves
            .iter()
            .any(|c| !roomeq_engine::topology::curve_has_usable_phase(c))
    {
        return Err(invalid(
            "final-seat coherent replay needs phase for every physical branch",
        ));
    }
    let lo = curves.iter().map(|c| c.freq[0]).fold(0.0_f64, f64::max);
    let hi = curves
        .iter()
        .map(|c| *c.freq.last().unwrap())
        .fold(f64::INFINITY, f64::min);
    let mut grid: Vec<_> = curves
        .iter()
        .flat_map(|c| c.freq.iter().copied())
        .filter(|f| *f >= lo && *f <= hi)
        .collect();
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    if grid.len() < 3 {
        return Err(invalid(
            "insufficient shared support for final-seat branch sum",
        ));
    }
    let grid = ndarray::Array1::from(grid);
    let aligned: Vec<_> = curves
        .iter()
        .map(|c| autoeq_measurements::read::interpolate_log_space(&grid, c))
        .collect();
    Ok(roomeq_engine::topology::complex_sum_mains(
        &aligned.iter().collect::<Vec<_>>(),
    ))
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

fn replay(
    result: &RoomOptimizationResult,
    physical: &BTreeMap<String, Vec<Curve>>,
    input: &str,
    seat: usize,
    baseline: bool,
    fs: f64,
    dir: &Path,
) -> Result<(Curve, Vec<String>)> {
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
            let mut curve = apply(
                result,
                input,
                stage(pre, "pre_route"),
                raw,
                baseline,
                fs,
                dir,
            )?;
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
            curve = apply(result, input, route_plugins, &curve, false, fs, dir)?;
            curve = apply(
                result,
                post_owner,
                stage(post, "post_route"),
                &curve,
                baseline,
                fs,
                dir,
            )?;
            branches.push(curve);
            outputs.push(route.destination.clone());
        }
        outputs.sort();
        outputs.dedup();
        return Ok((sum(&branches)?, outputs));
    }
    let chain = result
        .channels
        .get(input)
        .ok_or_else(|| invalid("missing channel for final seat replay"))?;
    if let Some(drivers) = &chain.drivers {
        let mut branches = Vec::new();
        let mut outputs = Vec::new();
        for driver in drivers {
            branches.push(apply(
                result,
                input,
                driver.plugins.clone(),
                measured(physical, &driver.name, seat)?,
                baseline,
                fs,
                dir,
            )?);
            outputs.push(driver.name.clone());
        }
        let combined = sum(&branches)?;
        return Ok((
            apply(
                result,
                input,
                chain.plugins.clone(),
                &combined,
                baseline,
                fs,
                dir,
            )?,
            outputs,
        ));
    }
    Ok((
        apply(
            result,
            input,
            chain.plugins.clone(),
            measured(physical, input, seat)?,
            baseline,
            fs,
            dir,
        )?,
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
                let (pre, outputs) = replay(result, physical, input, seat, true, fs, dir)?;
                let (post, _) = replay(result, physical, input, seat, false, fs, dir)?;
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
                let score = roomeq_engine::quality::evaluate_acoustic_quality(
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
                )
                .map_err(invalid)?;
                evidence.push(FinalSeatEvaluation {
                    partition: partition.into(),
                    logical_input: input.clone(),
                    seat_index: seat,
                    seat_label: if partition == "training" {
                        captures
                            .iter()
                            .find(|c| c.channel == *input)
                            .and_then(|c| c.seat_labels.as_ref())
                            .and_then(|labels| labels.get(seat))
                            .cloned()
                    } else {
                        None
                    },
                    physical_outputs: outputs,
                    evaluated_band_hz: score.evaluated_band_hz,
                    pre_weighted_rms_db: score.training.pre_weighted_rms_median_db,
                    post_weighted_rms_db: score.training.post_weighted_rms_median_db,
                    improvement_db: score.training.worst_position_improvement_db,
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
        .find(|s| !s.improvement_db.is_finite() || s.improvement_db < -budget)
        .map(|s| {
            format!(
                "{} '{}' seat {} regressed {:.3} dB beyond {:.3} dB budget",
                s.partition, s.logical_input, s.seat_index, -s.improvement_db, budget
            )
        });
    report.acoustic_quality = Some(score);
    if let Some(message) = failed {
        report.accepted = false;
        report.violations.push("worst_position_regressed".into());
        report.violations.sort();
        report.violations.dedup();
        return Err(AutoeqError::OptimizationFailed { message });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let seats = &report.acoustic_quality.as_ref().unwrap().final_seats;
        assert_eq!(seats.len(), 2);
        assert_eq!(seats[1].physical_outputs, vec!["left"]);
        assert!(seats[1].improvement_db < 0.0);
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
        other.freq = &other.freq * 1.001;
        let combined = sum(&[flat.clone(), other.clone()]).unwrap();
        assert!(combined.freq.len() > flat.freq.len());
        other.phase = None;
        assert!(sum(&[flat, other]).is_err());
    }

    #[test]
    fn routed_replay_uses_each_seat_and_applies_route_delay_gain_and_post_eq_once() {
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
        let mut opposite = flat.clone();
        opposite.phase.as_mut().unwrap().fill(180.0);
        let physical = BTreeMap::from([
            ("left".into(), vec![flat.clone(), flat.clone()]),
            ("sub".into(), vec![flat.clone(), opposite]),
        ]);
        for seat in 0..2 {
            let (response, outputs) = replay(
                &result,
                &physical,
                "left",
                seat,
                false,
                48_000.0,
                Path::new("."),
            )
            .unwrap();
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
        assert!(
            replay(
                &result,
                &phase_unknown,
                "left",
                0,
                false,
                48_000.0,
                Path::new(".")
            )
            .is_err()
        );
        let mut missing = physical;
        missing.get_mut("sub").unwrap().pop();
        assert!(
            replay(
                &result,
                &missing,
                "left",
                1,
                false,
                48_000.0,
                Path::new(".")
            )
            .is_err()
        );
    }
}
