//! Cumulative pruning against frozen, delivered multi-output playback.

use super::{RoomOptimizationResult, finalization, room_optimization_result, seat_replay};
use ndarray::Array1;
use roomeq_engine::eq::audibility_veto::conditions::{
    ConditionEvaluator, VetoCondition, VetoConditionSet,
};
use roomeq_model::{
    AppliedThreshold, AssessmentConfidence, AssessmentProvenance, AssessmentRecord, Curve,
    EnforcementState, FilterVetoVerdict, PluginConfigWrapper, ReportOutcome, RoomConfig,
    SpeakerConfig, StageCheck, StageCheckKind, StageOutcome, StageStatus, VetoAdjudicationReport,
    VetoDecision, VetoReason,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

const REPORT_KEY: &str = "final_routed_graph";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qa_roomeq_pruning_conditions_correlated_cancellation_is_not_averaged_away() {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let right = crate::test_fixtures::single_channel_room_result("R");
        result.channels.extend(right.channels);
        result.channel_results.extend(right.channel_results);
        let filters = [math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            80.0,
            48000.0,
            1.0,
            0.1,
        )];
        let mut physical = BTreeMap::new();
        for (name, phase) in [("L", 0.0), ("R", 179.99)] {
            result.channels.get_mut(name).unwrap().plugins =
                vec![roomeq_engine::output::create_eq_plugin(&filters)];
            let mut curve = crate::test_fixtures::flat_curve();
            curve.phase = Some(Array1::from_elem(curve.freq.len(), phase));
            physical.insert(String::from(name), vec![curve]);
        }
        let mut config = RoomConfig::default();
        config.optimizer.pruning_budget = Some(roomeq_model::PruningBudget {
            evaluation: Some(serde_json::from_value(serde_json::json!({
                "version": "spectral-v1", "measurement_ids": ["seat-0"],
                "programmes": [{"id": "flat", "frequencies_hz": [20, 20000], "spectrum_db": [0, 0]}],
                "listening_levels_phon": [55]
            })).unwrap()),
            ..Default::default()
        });
        let directory = tempfile::tempdir().unwrap();
        let baseline = evidence(
            &result,
            &physical,
            &BTreeMap::new(),
            &config,
            48000.0,
            directory.path(),
            None,
        )
        .unwrap();
        let trial = without(&result, &inventory(&result), &BTreeSet::from([0]));
        let candidate = evidence(
            &trial,
            &physical,
            &BTreeMap::new(),
            &config,
            48000.0,
            directory.path(),
            Some(&baseline.frequencies),
        )
        .unwrap();
        let mut correlated_delta = 0.0_f64;
        let mut individual_delta = 0.0_f64;
        for (before, after) in baseline.conditions.iter().zip(&candidate.conditions) {
            assert_eq!(before.id, after.id);
            let delta = (&after.background_db - &before.background_db)
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            if before.id.contains("correlated-inputs") {
                correlated_delta = correlated_delta.max(delta);
            } else {
                individual_delta = individual_delta.max(delta);
            }
        }
        assert!(individual_delta < 0.11, "{individual_delta}");
        assert!(correlated_delta > 3.0, "{correlated_delta}");
    }

    #[test]
    fn qa_roomeq_pruning_conditions_routed_missing_evidence_retains_f0() {
        for missing in ["declaration", "seat counts", "measured phase"] {
            let mut result = crate::test_fixtures::single_channel_room_result("L");
            let right = crate::test_fixtures::single_channel_room_result("R");
            result.channels.extend(right.channels);
            result.channel_results.extend(right.channel_results);
            let filters = [math_audio_iir_fir::Biquad::new(
                math_audio_iir_fir::BiquadFilterType::Peak,
                80.0,
                48000.0,
                1.0,
                0.01,
            )];
            result.channels.get_mut("L").unwrap().plugins =
                vec![roomeq_engine::output::create_eq_plugin(&filters)];
            let frozen = processing(&result);
            let mut config = RoomConfig::default();
            config.optimizer.filter_audibility = Some(roomeq_model::FilterAudibilityConfig {
                report_only: false,
                allow_enforcement_with_experimental_proxy: true,
                ..Default::default()
            });
            config.optimizer.pruning_budget = Some(roomeq_model::PruningBudget {
                evaluation: (missing != "declaration").then(|| serde_json::from_value(
                    serde_json::json!({
                        "version": "spectral-v1", "measurement_ids": ["seat-0", "seat-1"],
                        "programmes": [{"id": "flat", "frequencies_hz": [20, 20000], "spectrum_db": [0, 0]}],
                        "listening_levels_phon": [55]
                    })
                ).unwrap()),
                ..Default::default()
            });
            for name in ["L", "R"] {
                config.speakers.insert(
                    String::from(name),
                    SpeakerConfig::Single(roomeq_model::MeasurementSource::InMemoryMultiple(vec![
                            crate::test_fixtures::flat_curve();
                            if missing == "seat counts" { 1 } else { 2 }
                        ])),
                );
            }
            let captures = seat_replay::capture_training(&config).unwrap();
            let directory = tempfile::tempdir().unwrap();
            apply(
                &mut result,
                &captures,
                &HashMap::new(),
                &config,
                48000.0,
                directory.path(),
            );
            assert_eq!(processing(&result), frozen);
            let report = &result.metadata.veto_adjudication.as_ref().unwrap()[REPORT_KEY];
            assert!(!report.enforced);
            assert!(report.removed_filter_indices.is_empty());
            let verdicts = &result.metadata.audibility_veto.as_ref().unwrap()[REPORT_KEY];
            assert_eq!(verdicts.len(), 1);
            assert_eq!(
                verdicts[0].acceptance.outcome,
                ReportOutcome::InsufficientEvidence
            );
            assert!(
                verdicts[0].acceptance.reason.contains(missing),
                "{verdicts:#?}"
            );
        }
    }
}

pub(super) fn requested(config: &RoomConfig, has_held_out: bool) -> bool {
    config
        .optimizer
        .filter_audibility
        .is_some_and(|veto| veto.enabled)
        && (has_held_out
            || config
                .system
                .as_ref()
                .is_some_and(|system| system.subwoofers.is_some())
            || config.speakers.values().any(|speaker| {
                matches!(
                    speaker,
                    SpeakerConfig::Group(_)
                        | SpeakerConfig::MultiSub(_)
                        | SpeakerConfig::Dba(_)
                        | SpeakerConfig::Cardioid(_)
                )
            }))
}

#[derive(Clone, Debug, serde::Serialize)]
struct Location {
    channel: String,
    driver: Option<usize>,
    plugin: usize,
    filter: usize,
    original: serde_json::Value,
}

fn inventory(result: &RoomOptimizationResult) -> Vec<Location> {
    let mut names: Vec<_> = result.channels.keys().collect();
    names.sort();
    let mut locations = Vec::new();
    for name in names {
        let chain = &result.channels[name];
        let lists = std::iter::once((None, &chain.plugins)).chain(
            chain
                .drivers
                .iter()
                .flatten()
                .enumerate()
                .map(|(index, driver)| (Some(index), &driver.plugins)),
        );
        for (driver, plugins) in lists {
            for (plugin, value) in plugins.iter().enumerate() {
                if value.plugin_type != "eq"
                    || !room_optimization_result::is_baseline_correction(value)
                {
                    continue;
                }
                for (filter, original) in value
                    .parameters
                    .get("filters")
                    .and_then(|v| v.as_array())
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    // Phase-only sections and structural crossovers are not
                    // priced by this experimental magnitude-difference model.
                    let kind = original
                        .get("filter_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    if !matches!(
                        kind,
                        "peak" | "lowshelf" | "highshelf" | "low_shelf" | "high_shelf"
                    ) || original.get("topology").is_some()
                    {
                        continue;
                    }
                    locations.push(Location {
                        channel: name.clone(),
                        driver,
                        plugin,
                        filter,
                        original: original.clone(),
                    });
                }
            }
        }
    }
    locations
}

fn plugin_mut<'a>(
    result: &'a mut RoomOptimizationResult,
    location: &Location,
) -> &'a mut PluginConfigWrapper {
    let chain = result
        .channels
        .get_mut(&location.channel)
        .expect("inventory belongs to frozen graph");
    match location.driver {
        Some(driver) => {
            &mut chain.drivers.as_mut().expect("frozen driver list")[driver].plugins
                [location.plugin]
        }
        None => &mut chain.plugins[location.plugin],
    }
}

fn without(
    f0: &RoomOptimizationResult,
    locations: &[Location],
    removed: &BTreeSet<usize>,
) -> RoomOptimizationResult {
    let mut result = f0.clone();
    // Reverse inventory order preserves every original array index.
    for &index in removed.iter().rev() {
        let location = &locations[index];
        plugin_mut(&mut result, location).parameters["filters"]
            .as_array_mut()
            .expect("frozen filter list")
            .remove(location.filter);
        // This list describes the common serial correction only. Parallel
        // driver filters must never be folded into its report/IR metadata.
        if location.driver.is_none()
            && let Some(channel) = result.channel_results.get_mut(&location.channel)
            && let Some(index) = channel.biquads.iter().rposition(|filter| {
                roomeq_engine::output::biquad_to_json(filter) == location.original
            })
        {
            channel.biquads.remove(index);
        }
    }
    result
}

fn processing(result: &RoomOptimizationResult) -> serde_json::Value {
    let channels: BTreeMap<_, _> = result
        .channels
        .iter()
        .map(|(name, chain)| {
            let drivers = chain.drivers.as_ref().map(|drivers| {
                drivers
                    .iter()
                    .map(|driver| {
                        serde_json::json!({
                            "name": driver.name,
                            "index": driver.index,
                            "plugins": driver.plugins,
                            "measured_band_hz": driver.measured_band_hz,
                        })
                    })
                    .collect::<Vec<_>>()
            });
            (
                name,
                serde_json::json!({"plugins": chain.plugins, "drivers": drivers}),
            )
        })
        .collect();
    serde_json::json!({"channels": channels, "routing": result.metadata.bass_management.as_ref().and_then(|b| b.routing_graph.as_ref())})
}

struct Evidence {
    ids: Vec<String>,
    conditions: Vec<VetoCondition>,
    frequencies: Array1<f64>,
}

struct Candidate {
    step: f64,
    cumulative: f64,
    local: f64,
    index: usize,
    trial: RoomOptimizationResult,
    responses: Vec<Array1<f64>>,
}

fn evidence(
    result: &RoomOptimizationResult,
    physical: &BTreeMap<String, Vec<Curve>>,
    held: &BTreeMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
    grid: Option<&Array1<f64>>,
) -> std::result::Result<Evidence, String> {
    let budget = config
        .optimizer
        .pruning_budget
        .as_ref()
        .ok_or("missing pruning budget")?;
    let declaration = budget
        .evaluation
        .as_ref()
        .ok_or("missing condition declaration")?;
    declaration.validate()?;
    let mut inputs: Vec<_> = if let Some(graph) = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|b| b.routing_graph.as_ref())
    {
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
    } else {
        result.channels.keys().cloned().collect()
    };
    inputs.sort();
    inputs.dedup();
    if inputs.is_empty() {
        return Err(String::from("no delivered logical inputs"));
    }
    let mut all = Vec::new();
    let mut frequencies = grid.cloned();
    for (partition, captures) in [("training", physical), ("held_out", held)] {
        if captures.is_empty() {
            if partition == "training" {
                return Err(String::from("missing training captures"));
            }
            continue;
        }
        if captures
            .values()
            .any(|curves| curves.len() != declaration.measurement_ids.len())
        {
            return Err(format!(
                "{partition} physical seat counts do not match the complete declaration"
            ));
        }
        if inputs.len() > 1
            && captures
                .values()
                .flatten()
                .any(|curve| !roomeq_engine::topology::curve_has_usable_phase(curve))
        {
            return Err(format!(
                "{partition} correlated-input replay requires measured phase at every seat"
            ));
        }
        let mut correlated: Vec<Vec<Curve>> = vec![Vec::new(); declaration.measurement_ids.len()];
        for input in &inputs {
            let mut delivered = Vec::new();
            for (seat, group) in correlated.iter_mut().enumerate() {
                let playback = seat_replay::replay_final_physical_seat(
                    result, captures, input, seat, config, fs, dir, partition,
                )
                .map_err(|error| error.to_string())?;
                if frequencies.is_none() {
                    // Replay unions native grids with exact configured bounds.
                    // The same physical bin can differ by a few floating-point
                    // ULPs, which cannot provide a positive ERB integration cell.
                    // Only the evaluation grid is collapsed, not capture data.
                    let mut grid: Vec<f64> = Vec::new();
                    for &frequency in &playback.delivered.freq {
                        if grid.last().is_none_or(|previous| {
                            (frequency - previous).abs()
                                > 8.0 * f64::EPSILON * frequency.abs().max(previous.abs())
                        }) {
                            grid.push(frequency);
                        }
                    }
                    frequencies = Some(Array1::from_vec(grid));
                }
                group.push(playback.delivered.clone());
                delivered.push(playback.delivered);
            }
            append_conditions(
                &mut all,
                &delivered,
                frequencies.as_ref().unwrap(),
                budget,
                &format!("{partition}/{input}"),
            )?;
        }
        if inputs.len() > 1 {
            let mut sums = Vec::new();
            for group in correlated {
                if group
                    .iter()
                    .any(|curve| !roomeq_engine::topology::curve_has_usable_phase(curve))
                    || !roomeq_engine::topology::all_curves_share_frequency_grid(
                        &group.iter().collect::<Vec<_>>(),
                    )
                {
                    return Err(String::from(
                        "correlated-input playback lacks aligned phase evidence",
                    ));
                }
                sums.push(roomeq_engine::topology::complex_sum_mains(
                    &group.iter().collect::<Vec<_>>(),
                ));
            }
            append_conditions(
                &mut all,
                &sums,
                frequencies.as_ref().unwrap(),
                budget,
                &format!("{partition}/correlated-inputs"),
            )?;
        }
    }
    Ok(Evidence {
        ids: all.iter().map(|condition| condition.id.clone()).collect(),
        conditions: all,
        frequencies: frequencies.ok_or("no delivered frequency grid")?,
    })
}

fn append_conditions(
    all: &mut Vec<VetoCondition>,
    curves: &[Curve],
    frequencies: &Array1<f64>,
    budget: &roomeq_model::PruningBudget,
    prefix: &str,
) -> std::result::Result<(), String> {
    let (_, mut conditions) = roomeq_engine::eq::audibility_veto::workflow::gather_conditions(
        budget,
        curves,
        frequencies,
    )?;
    for condition in &mut conditions {
        condition.id = format!("{prefix}/{}", condition.id);
    }
    all.extend(conditions);
    Ok(())
}

fn reference(
    f0: &RoomOptimizationResult,
    config: &RoomConfig,
    evidence: Option<&Evidence>,
) -> String {
    // Stable FNV identity, matching the engine's non-cryptographic F0 convention.
    let mut hash = 0xcbf29ce484222325_u64;
    let mut append = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    append(&serde_json::to_vec(&processing(f0)).expect("serializable graph"));
    append(
        &serde_json::to_vec(&config.optimizer.pruning_budget).expect("serializable declaration"),
    );
    if let Some(evidence) = evidence {
        for frequency in &evidence.frequencies {
            append(&frequency.to_bits().to_le_bytes());
        }
        for condition in &evidence.conditions {
            append(condition.id.as_bytes());
            append(&condition.listening_phon.to_bits().to_le_bytes());
            for value in &condition.background_db {
                append(&value.to_bits().to_le_bytes());
            }
        }
    }
    format!("F0:routed-conditions-v1:{hash:016x}")
}

fn validate_trial(
    trial: &mut RoomOptimizationResult,
    f0: &RoomOptimizationResult,
    captures: &[seat_replay::Capture],
    held: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
) -> std::result::Result<(), String> {
    let expected = processing(trial);
    finalization::rebuild(trial, config, held, fs, dir).map_err(|error| error.to_string())?;
    if processing(trial) != expected {
        return Err(String::from("safety replay changed the proposed graph"));
    }
    seat_replay::validate_candidate_final_seats(trial, f0, captures, held, config, fs, dir)
        .map_err(|error| error.to_string())?;
    let Some(report) = trial.metadata.correction_acceptance.as_ref() else {
        return Err(String::from("final acoustic acceptance report unavailable"));
    };
    if !report.accepted {
        let final_seats = report
            .acoustic_quality
            .as_ref()
            .map(|quality| {
                quality
                    .final_seats
                    .iter()
                    .map(|seat| {
                        (
                            seat.partition.as_str(),
                            seat.logical_input.as_str(),
                            seat.seat_index,
                            seat.improvement_lower_bound_db,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        return Err(format!(
            "final acoustic acceptance rejected the removal: decision={:?}; violations={:?}; improvement_db={:.6}; correction_rms_db={:.6}; final_seat_lower_bounds={final_seats:?}",
            report.decision,
            report.violations,
            report.metrics.improvement_db,
            report.metrics.correction_rms_db,
        ));
    }
    let outputs = crate::electrical_headroom::assess_final_graph(
        &trial.to_dsp_chain_output(),
        fs,
        dir,
        &config.optimizer.finalization,
    )
    .map_err(|error| error.to_string())?;
    if outputs.is_empty()
        || outputs.iter().any(|output| {
            !output.peak_dbfs.is_some_and(|peak| {
                peak.is_finite() && peak <= config.optimizer.finalization.output_ceiling_dbfs + 1e-6
            })
        })
    {
        return Err(String::from("final electrical headroom did not pass"));
    }
    Ok(())
}

pub(super) fn apply(
    result: &mut RoomOptimizationResult,
    captures: &[seat_replay::Capture],
    held: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
) {
    let f0 = result.clone();
    let locations = inventory(&f0);
    let veto = config.optimizer.filter_audibility.expect("requested veto");
    let default_budget = roomeq_model::PruningBudget::default();
    let budget = config
        .optimizer
        .pruning_budget
        .as_ref()
        .unwrap_or(&default_budget);
    let mut records = vec![
        (
            ReportOutcome::InsufficientEvidence,
            0.0,
            0.0,
            String::from("not evaluated")
        );
        locations.len()
    ];
    let mut removed = BTreeSet::new();
    let mut cumulative = 0.0;
    let mut local = 0.0;
    let mut frozen_id = reference(&f0, config, None);
    let evaluation = (|| -> std::result::Result<(), String> {
        let physical = seat_replay::training_physical_captures(captures, &f0)
            .map_err(|error| error.to_string())?;
        let held_physical = held
            .iter()
            .map(|(name, curves)| (name.clone(), curves.clone()))
            .collect();
        let baseline = evidence(&f0, &physical, &held_physical, config, fs, dir, None)?;
        frozen_id = reference(&f0, config, Some(&baseline));
        let set = VetoConditionSet {
            declared_ids: &baseline.ids,
            conditions: &baseline.conditions,
            aggregation: budget.aggregation,
        };
        let evaluator = ConditionEvaluator::new(
            &set,
            &baseline.frequencies,
            &Array1::zeros(baseline.frequencies.len()),
        )?;
        let cap = budget
            .max_cumulative_delta
            .unwrap_or(veto.elimination_loudness_delta_sones);
        if [cap, veto.elimination_loudness_delta_sones, veto.jnd_db]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(String::from("invalid pruning limits"));
        }
        let condition_context = format!(
            "complete delivered conditions={:?}; aggregation={:?}",
            baseline.ids, budget.aggregation
        );
        let mut current =
            vec![Array1::zeros(baseline.frequencies.len()); baseline.conditions.len()];
        let mut retained_trial = f0.clone();
        while removed.len() < locations.len() {
            let mut best: Option<Candidate> = None;
            for index in 0..locations.len() {
                if removed.contains(&index) {
                    continue;
                }
                let mut proposed = removed.clone();
                proposed.insert(index);
                let trial = without(&f0, &locations, &proposed);
                let delivered = evidence(
                    &trial,
                    &physical,
                    &held_physical,
                    config,
                    fs,
                    dir,
                    Some(&baseline.frequencies),
                )?;
                if delivered.ids != baseline.ids {
                    return Err(String::from("candidate changed condition identities"));
                }
                let responses: Vec<_> = delivered
                    .conditions
                    .iter()
                    .zip(&baseline.conditions)
                    .map(|(after, before)| &after.background_db - &before.background_db)
                    .collect();
                let (step, total) = evaluator
                    .differences_for_responses(&current, &responses)
                    .ok_or("nonfinite delivered response difference")?;
                let maximum = responses
                    .iter()
                    .flatten()
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max);
                if best.as_ref().is_none_or(|candidate| step < candidate.step) {
                    best = Some(Candidate {
                        step,
                        cumulative: total,
                        local: maximum,
                        index,
                        trial,
                        responses,
                    });
                }
            }
            let Some(Candidate {
                step,
                cumulative: total,
                local: maximum,
                index,
                mut trial,
                responses,
            }) = best
            else {
                break;
            };
            if step > veto.elimination_loudness_delta_sones || total > cap || maximum > veto.jnd_db
            {
                records[index] = (
                    ReportOutcome::Keep,
                    step,
                    maximum,
                    format!(
                        "{condition_context}; frozen graph limit exceeded: step={step}, cumulative={total}, local={maximum}"
                    ),
                );
                break;
            }
            if let Err(reason) = validate_trial(&mut trial, &f0, captures, held, config, fs, dir) {
                records[index] = (
                    ReportOutcome::Keep,
                    step,
                    maximum,
                    format!("{condition_context}; candidate validation rejected removal: {reason}"),
                );
                break;
            }
            records[index] = (
                if veto.enforcement_authorized() {
                    ReportOutcome::AcceptedRemoval
                } else {
                    ReportOutcome::CandidateRemoval
                },
                step,
                maximum,
                condition_context.clone(),
            );
            removed.insert(index);
            current = responses;
            cumulative = total;
            local = maximum;
            retained_trial = trial;
        }
        if veto.enforcement_authorized() {
            *result = retained_trial;
        }
        Ok(())
    })();
    if let Err(reason) = &evaluation {
        // Uncertain evidence invalidates the whole proposed walk. Publish F0.
        *result = f0.clone();
        removed.clear();
        cumulative = 0.0;
        local = 0.0;
        for record in &mut records {
            *record = (
                ReportOutcome::InsufficientEvidence,
                0.0,
                0.0,
                reason.clone(),
            );
        }
    }
    let enforced = veto.enforcement_authorized() && evaluation.is_ok();
    let reports: Vec<_> = locations
        .iter()
        .zip(records)
        .enumerate()
        .map(|(index, (location, (outcome, step, maximum, reason)))| {
            let applied = enforced && removed.contains(&index);
            FilterVetoVerdict {
                index,
                center_hz: location.original["freq"].as_f64().unwrap_or(0.0),
                q: location.original["q"].as_f64().unwrap_or(0.0),
                gain_db: location.original["db_gain"].as_f64().unwrap_or(0.0),
                peak_delta_db: maximum,
                affected_erb_width: 0.0,
                loudness_delta_sones: step,
                decision: if removed.contains(&index) {
                    VetoDecision::Remove
                } else {
                    VetoDecision::Keep
                },
                reason: VetoReason::SubJnd,
                enforced: applied,
                acceptance: AssessmentRecord {
                    outcome,
                    confidence: AssessmentConfidence::Low,
                    enforcement: if applied {
                        EnforcementState::Enforced
                    } else {
                        EnforcementState::Advisory
                    },
                    provenance: AssessmentProvenance {
                        model: String::from("heuristic-erb-proxy"),
                        model_version: env!("CARGO_PKG_VERSION").into(),
                        calibration: String::from("declared-nominal-levels;unknown-spl"),
                        reference: frozen_id.clone(),
                    },
                    thresholds: vec![
                        AppliedThreshold {
                            name: String::from("incremental_loudness"),
                            value: veto.elimination_loudness_delta_sones,
                            unit: String::from("sones-experimental-proxy"),
                        },
                        AppliedThreshold {
                            name: String::from("cumulative_loudness"),
                            value: budget
                                .max_cumulative_delta
                                .unwrap_or(veto.elimination_loudness_delta_sones),
                            unit: String::from("sones-experimental-proxy"),
                        },
                        AppliedThreshold {
                            name: String::from("local_deviation"),
                            value: veto.jnd_db,
                            unit: String::from("db"),
                        },
                    ],
                    reason: format!(
                        "{reason}; original_location={}",
                        serde_json::to_string(location).expect("serializable location")
                    ),
                },
            }
        })
        .collect();
    result
        .metadata
        .audibility_veto
        .get_or_insert_with(Default::default)
        .insert(REPORT_KEY.into(), reports);
    result
        .metadata
        .veto_adjudication
        .get_or_insert_with(Default::default)
        .insert(
            REPORT_KEY.into(),
            VetoAdjudicationReport {
                f0_reference_id: frozen_id,
                removed_filter_indices: if enforced {
                    removed.iter().copied().collect()
                } else {
                    Vec::new()
                },
                cumulative_loudness_delta_sones: cumulative,
                max_local_deviation_db: local,
                enforced,
            },
        );
    result.metadata.stage_outcomes.push(StageOutcome {
        stage: String::from("final_routed_pruning"),
        status: if evaluation.is_err() {
            StageStatus::Skipped
        } else if enforced && !removed.is_empty() {
            StageStatus::Applied
        } else {
            StageStatus::Skipped
        },
        advisories: vec![
            String::from("experimental_proxy_low_confidence"),
            format!(
                "eligible_filters={}; hypothetical_removals={}",
                locations.len(),
                removed.len()
            ),
        ],
        checks: vec![StageCheck {
            id: String::from("complete_delivered_condition_evidence"),
            kind: StageCheckKind::Safety,
            passed: evaluation.is_ok(),
            observed: None,
            limit: None,
            diagnostic: evaluation.err(),
        }],
    });
}
