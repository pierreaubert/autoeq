//! Correlated program must retain coherent bass efficiency at every seat.

// Rust guideline compliant 2026-02-21
use super::{Capture, ReplayContext, RoomOptimizationResult, invalid, physical_captures, replay};
use roomeq_model::{
    Curve, Result, RoomConfig, StageCheck, StageCheckKind, StageOutcome, StageStatus,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

const STAGE: &str = "final_correlated_bass_acceptance";

/// Band-integrated decrease in coherent-to-power-sum efficiency.
///
/// Programme channels reach a seat through different mains, so legitimate
/// per-source splice alignment shifts single-bin summation nulls between
/// channels: a worst-bin verdict vetoes the delay alignment correction
/// exists to apply, while band-integrated coherent power still collapses
/// under systematic decorrelation (for example an opposed sub polarity).
/// The worst single-bin frequency is retained in the returned pair for
/// diagnostics. The physical sub array, where every output carries the
/// same signal, keeps the worst-bin measure in the optimizer.
fn efficiency_regression(before: &[Curve], after: &[Curve]) -> Result<(f64, f64)> {
    roomeq_quality::correlated_bass_band_efficiency_regression(
        before,
        after,
        super::MONO_BASS_MAX_HZ,
    )
    .map_err(invalid)
}

fn groups(graph: &roomeq_model::BassManagementRoutingGraph) -> Vec<Vec<String>> {
    let mut shared: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for route in &graph.routes {
        if route.route_kind != "redirected_bass_lowpass_to_sub" {
            continue;
        }
        shared
            .entry(format!("output:{}", route.destination))
            .or_default()
            .insert(route.source_channel.clone());
        if let Some(group) = &route.group_id {
            shared
                .entry(format!("group:{group}"))
                .or_default()
                .insert(route.source_channel.clone());
        }
    }
    let active: BTreeSet<_> = graph
        .routes
        .iter()
        .filter(|route| route.route_kind == "redirected_bass_lowpass_to_sub")
        .map(|route| route.source_channel.clone())
        .collect();
    // Protect common stereo and front-stage program even when the same physical
    // bus also carries surrounds/heights; all-channel sums can hide pair nulls.
    use roomeq_model::HomeCinemaRole::{Center, FrontLeft, FrontRight};
    for roles in [
        &[FrontLeft, FrontRight][..],
        &[FrontLeft, Center, FrontRight][..],
    ] {
        let selected: BTreeSet<_> = active
            .iter()
            .filter(|input| roles.contains(&roomeq_model::home_cinema::role_for_channel(input)))
            .cloned()
            .collect();
        if selected.len() == roles.len() {
            shared.insert(format!("front:{roles:?}"), selected);
        }
    }
    shared
        .into_values()
        .filter(|inputs| inputs.len() >= 2)
        .map(|inputs| inputs.into_iter().collect())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(super) fn validate(
    result: &mut RoomOptimizationResult,
    captures: &[Capture],
    held_out: &HashMap<String, Vec<Curve>>,
    config: &RoomConfig,
    fs: f64,
    dir: &Path,
) -> Result<()> {
    result
        .metadata
        .stage_outcomes
        .retain(|stage| stage.stage != STAGE);
    let Some(graph) = result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
    else {
        return Ok(());
    };
    let groups = groups(graph);
    if groups.is_empty() {
        return Ok(());
    }
    let roles: Vec<_> = groups
        .iter()
        .flatten()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let unassessed = |reason: String| StageOutcome {
        stage: STAGE.into(),
        status: StageStatus::Skipped,
        advisories: vec![format!("correlated_bass_unassessed:{reason}")],
        checks: Vec::new(),
    };
    if let Err(reason) = crate::evidence_intake::crossover_timing_reference(
        config,
        &roles,
        [config.optimizer.min_freq.min(20.0), super::MONO_BASS_MAX_HZ],
    ) {
        result
            .metadata
            .stage_outcomes
            .push(unassessed(reason.clone()));
        if let Some(report) = result.metadata.correction_acceptance.as_mut() {
            let note = format!("correlated_bass_unassessed:{reason}");
            if !report.observations.contains(&note) {
                report.observations.push(note);
            }
        }
        return Ok(());
    }
    let training = physical_captures(captures, result)?;
    let held: BTreeMap<_, _> = held_out
        .iter()
        .map(|(name, curves)| (name.clone(), curves.clone()))
        .collect();
    let mut reference = result.clone();
    super::restore_configured_bass_routes(&mut reference, config)?;
    let limit = config
        .optimizer
        .finalization
        .max_correlated_bass_regression_db;
    let mut outcome = StageOutcome { stage: STAGE.into(), status: StageStatus::Applied, advisories: vec![
        "fixed_configured_routes_with_same_physical_array_controls_and_output_safety; common_gain_invariant; measured_phase_only".into()
    ], checks: Vec::new() };
    for (partition, physical) in [("training", &training), ("held_out", &held)] {
        if physical.is_empty() {
            continue;
        }
        let seats = physical.values().map(Vec::len).max().unwrap_or(0);
        let mut observation = config.clone();
        observation.optimizer.min_freq = physical
            .values()
            .flatten()
            .filter_map(|curve| curve.freq.first().copied())
            .fold(0.0, f64::max);
        observation.optimizer.max_freq = super::MONO_BASS_MAX_HZ;
        let context = ReplayContext {
            config: &observation,
            partition,
            fs,
            dir,
        };
        // Each logical input is replayed once per seat, then reused by groups.
        for seat in 0..seats {
            let mut responses = BTreeMap::new();
            for input in &roles {
                let (before, _) = replay(&reference, physical, input, seat, true, &context)?;
                let (after, _) = replay(result, physical, input, seat, false, &context)?;
                responses.insert(input, (before.curve, after.curve));
            }
            for group in &groups {
                let before: Vec<_> = group
                    .iter()
                    .map(|input| responses[input].0.clone())
                    .collect();
                let after: Vec<_> = group
                    .iter()
                    .map(|input| responses[input].1.clone())
                    .collect();
                let (loss, frequency) = efficiency_regression(&before, &after)?;
                outcome.checks.push(StageCheck {
                    id: format!(
                        "correlated_bass:{partition}:seat_{seat}:{}",
                        group.join("+")
                    ),
                    kind: StageCheckKind::Quality,
                    passed: loss <= limit + 1e-6,
                    observed: Some(loss),
                    limit: Some(limit),
                    diagnostic: Some(format!("worst_hz={frequency:.6}; loss_db={loss:.6}")),
                });
            }
        }
    }
    let rejected = outcome.checks.iter().any(|check| !check.passed);
    if outcome.checks.is_empty() {
        return Err(invalid("correlated bass has no replayable seats"));
    }
    if rejected {
        outcome.status = StageStatus::Failed;
    }
    let failed_checks = outcome
        .checks
        .iter()
        .filter(|check| !check.passed)
        .take(8)
        .map(|check| {
            format!(
                "{}: loss_db={:?}, limit_db={:?}, {}",
                check.id,
                check.observed,
                check.limit,
                check.diagnostic.as_deref().unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    result.metadata.stage_outcomes.push(outcome);
    if rejected {
        if let Some(report) = result.metadata.correction_acceptance.as_mut() {
            report.accepted = false;
            report.decision = roomeq_model::CorrectionDecision::Rejected;
            report
                .violations
                .push("correlated_bass_efficiency_regressed".into());
            report.refresh_outcome();
        }
        return Err(roomeq_model::AutoeqError::OptimizationFailed {
            message: format!(
                "final correlated-bass efficiency regressed against configured routes: {failed_checks}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn response(phase: f64) -> Curve {
        Curve {
            freq: ndarray::array![20.0, 50.0, 80.0, 150.0, 300.0],
            spl: ndarray::Array1::from_elem(5, 80.0),
            phase: Some(ndarray::Array1::from_elem(5, phase)),
            ..Default::default()
        }
    }
    #[test]
    fn opposed_correlated_polarity_cannot_hide_behind_peak_buildup() {
        let before = vec![response(0.0); 2];
        // A single-bin null still shows peak buildup elsewhere; the
        // band-integrated programme verdict below targets the systematic
        // decorrelation a polarity opposition causes at every bin.
        let mut narrow = before.clone();
        narrow[1].phase.as_mut().unwrap()[2] = 180.0;
        assert!(
            super::super::assess_mono_bass_sums(&narrow, 300.0)
                .unwrap()
                .max_buildup_db
                > 3.0
        );
        let mut after = before.clone();
        after[1]
            .phase
            .as_mut()
            .unwrap()
            .mapv_inplace(|phase| phase + 180.0);
        let (loss, _frequency) = efficiency_regression(&before, &after).unwrap();
        assert!(loss > 200.0, "band loss {loss}");
    }
    #[test]
    fn common_gain_and_phase_changes_preserve_efficiency() {
        let before = vec![response(0.0), response(40.0), response(-10.0)];
        let mut after = before.clone();
        for curve in &mut after {
            curve.spl -= 40.0;
            *curve.phase.as_mut().unwrap() += 115.0;
        }
        assert!(efficiency_regression(&before, &after).unwrap().0 < 1e-10);
        assert_eq!(efficiency_regression(&before, &before).unwrap().0, 0.0);
    }
    #[test]
    fn phase_absence_is_not_a_coherent_acceptance() {
        let before = vec![response(0.0); 2];
        let mut after = before.clone();
        after[1].phase = None;
        assert!(efficiency_regression(&before, &after).is_err());
    }

    #[test]
    fn final_serialized_routes_reject_opposed_correlated_bass_at_every_seat() {
        use roomeq_model::{
            CrossoverConfig, MeasurementSource, SpeakerConfig, SubwooferCrossoverRef,
            SubwooferOutput, SubwooferSystemConfig, SystemConfig, SystemModel,
        };
        let source = |main: bool| -> MeasurementSource {
            serde_json::from_value(serde_json::json!({
                "measurements": (0..2).map(|seat| serde_json::json!({
                    "name": format!("seat-{seat}"),
                    "frequencies": [20.0, 30.0, 50.0, 80.0, 150.0, 300.0, 1000.0, 20000.0],
                    "magnitude_db": if main { vec![20.0,20.0,20.0,80.0,80.0,80.0,80.0,80.0] } else { vec![80.0;8] },
                    "phase_deg": vec![0.0;8]
                })).collect::<Vec<_>>(),
                "provenance": {"capture_kind":"stationary_ir", "timing_reference_id":"common-clock"}
            })).unwrap()
        };
        let config = RoomConfig {
            speakers: [
                ("L".into(), SpeakerConfig::Single(source(true))),
                ("R".into(), SpeakerConfig::Single(source(true))),
                ("Sub".into(), SpeakerConfig::Single(source(false))),
            ]
            .into(),
            system: Some(SystemConfig {
                model: SystemModel::HomeCinema,
                speakers: [("L".into(), "L".into()), ("R".into(), "R".into())].into(),
                subwoofers: Some(SubwooferSystemConfig {
                    outputs: vec![SubwooferOutput {
                        id: "Sub".into(),
                        speaker: "Sub".into(),
                    }],
                    crossover: Some(SubwooferCrossoverRef::PerSub(vec!["bass".into()])),
                    config: roomeq_model::SubwooferStrategy::Single,
                    routing: Default::default(),
                }),
                ..Default::default()
            }),
            crossovers: Some(
                [(
                    "bass".into(),
                    CrossoverConfig {
                        crossover_type: "LR24".into(),
                        frequency: Some(80.0),
                        frequencies: None,
                        frequency_range: None,
                    },
                )]
                .into(),
            ),
            ..Default::default()
        };
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        for name in ["R", "Sub"] {
            let channel = crate::test_fixtures::single_channel_room_result(name);
            result.channels.extend(channel.channels);
            result.channel_results.extend(channel.channel_results);
        }
        for chain in result.channels.values_mut() {
            chain.plugins.clear();
        }
        result.metadata.bass_management =
            roomeq_engine::home_cinema::bass_management_report(&config, None, false);
        let captures = super::super::capture_training(&config).unwrap();
        let held: HashMap<_, _> = physical_captures(&captures, &result)
            .unwrap()
            .into_iter()
            .collect();
        let dir = tempfile::tempdir().unwrap();
        validate(&mut result, &captures, &held, &config, 48_000.0, dir.path()).unwrap();
        let checks = &result
            .metadata
            .stage_outcomes
            .iter()
            .find(|stage| stage.stage == STAGE)
            .unwrap()
            .checks;
        assert_eq!(checks.len(), 4);
        assert!(
            checks
                .iter()
                .any(|check| check.id.contains("held_out:seat_1"))
        );
        assert!(checks.iter().all(|check| check.passed));
        let graph = result
            .metadata
            .bass_management
            .as_mut()
            .unwrap()
            .routing_graph
            .as_mut()
            .unwrap();
        let route = graph
            .routes
            .iter_mut()
            .find(|route| route.source_channel == "R" && route.destination == "Sub")
            .unwrap();
        route.polarity_inverted = true;
        let mut front_graph = graph.clone();
        let mut center = front_graph
            .routes
            .iter()
            .find(|route| route.source_channel == "L" && route.destination == "Sub")
            .unwrap()
            .clone();
        center.source_channel = "Center".into();
        center.group_id = Some("lcr".into());
        front_graph.routes.push(center);
        let assessed_groups = groups(&front_graph);
        assert!(assessed_groups.iter().any(|group| group == &["L", "R"]));
        assert!(
            assessed_groups
                .iter()
                .any(|group| group == &["Center", "L", "R"])
        );
        let error =
            validate(&mut result, &captures, &held, &config, 48_000.0, dir.path()).unwrap_err();
        assert!(error.to_string().contains("correlated-bass"));
        let stage = result
            .metadata
            .stage_outcomes
            .iter()
            .find(|stage| stage.stage == STAGE)
            .unwrap();
        assert_eq!(stage.status, StageStatus::Failed);
        assert_eq!(stage.checks.len(), 4);
        assert!(stage.checks.iter().all(|check| !check.passed));
    }
}
