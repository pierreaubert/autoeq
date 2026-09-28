//! Current waveform-view availability, separate from acoustic acceptance.

use std::collections::BTreeMap;

use roomeq_model::{StageCheck, StageCheckKind, StageOutcome, StageStatus};

use super::super::RoomOptimizationResult;

/// Replace waveform diagnostics with availability for the current channel set.
pub(super) fn record(
    result: &mut RoomOptimizationResult,
    replay_errors: &BTreeMap<String, String>,
) {
    // Candidate rebuilds inherit the preceding report. Log a replay failure
    // only when its reason changes; always retain it in the current report.
    for (name, reason) in replay_errors {
        if replay_error_changed(result, name, reason) {
            log::warn!("Waveform unavailable for {name}: {reason}");
        }
    }
    let mut checks = Vec::new();
    for (name, chain) in &result.channels {
        let initial_reason = match result.channel_results.get(name) {
            None => Some("initial measurement unavailable".to_owned()),
            Some(channel) => match channel
                .initial_curve
                .validate("waveform initial measurement")
            {
                Err(error) => Some(error.to_string()),
                Ok(()) if channel.initial_curve.phase.is_none() => {
                    Some("initial measurement phase unavailable".to_owned())
                }
                Ok(()) => None,
            },
        };
        for (view, available) in [
            ("pre_ir", chain.pre_ir.is_some()),
            ("post_ir", chain.post_ir.is_some()),
        ] {
            let id = format!("{view}:{name}");
            checks.push(if available {
                StageCheck::pass(id, StageCheckKind::Quality)
            } else {
                let reason = if view == "pre_ir" {
                    initial_reason.as_ref().or_else(|| replay_errors.get(name))
                } else {
                    replay_errors.get(name).or(initial_reason.as_ref())
                };
                StageCheck::fail(
                    id,
                    StageCheckKind::Quality,
                    reason.cloned().unwrap_or_else(|| "waveform reconstruction unavailable: invalid spectrum, level reference, or sample rate".into()),
                )
            });
        }
    }
    checks.sort_by(|a, b| a.id.cmp(&b.id));
    let status = if checks.is_empty() {
        StageStatus::Skipped
    } else if checks.iter().all(|check| check.passed) {
        StageStatus::Applied
    } else {
        StageStatus::Degraded
    };
    result
        .metadata
        .stage_outcomes
        .retain(|stage| stage.stage != "waveform_views");
    result.metadata.stage_outcomes.push(StageOutcome {
        stage: "waveform_views".into(),
        status,
        advisories: vec!["predicted_waveforms_not_recorded_playback_or_acoustic_acceptance".into()],
        checks,
    });
}

fn replay_error_changed(result: &RoomOptimizationResult, name: &str, reason: &str) -> bool {
    let id = format!("post_ir:{name}");
    !result
        .metadata
        .stage_outcomes
        .iter()
        .filter(|stage| stage.stage == "waveform_views")
        .flat_map(|stage| &stage.checks)
        .any(|check| check.id == id && !check.passed && check.diagnostic.as_deref() == Some(reason))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_replay_failures_are_reported_without_repeating_warnings() {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let reason = "driver woofer: capture unavailable";
        let errors = BTreeMap::from([("L".into(), reason.into())]);
        assert!(replay_error_changed(&result, "L", reason));
        record(&mut result, &errors);
        assert!(!replay_error_changed(&result, "L", reason));
        assert!(!replay_error_changed(&result.clone(), "L", reason));
        assert!(replay_error_changed(&result, "L", "different failure"));
        record(&mut result, &errors);
        assert_eq!(
            result
                .metadata
                .stage_outcomes
                .iter()
                .filter(|stage| stage.stage == "waveform_views")
                .count(),
            1
        );
        record(&mut result, &BTreeMap::new());
        assert!(replay_error_changed(&result, "L", reason));
    }

    #[test]
    fn waveform_reasons_distinguish_initial_phase_from_branch_replay() {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        result
            .channel_results
            .get_mut("L")
            .unwrap()
            .initial_curve
            .phase = None;
        let errors = BTreeMap::from([("L".into(), "driver woofer: capture unavailable".into())]);
        record(&mut result, &errors);
        let stage = result.metadata.stage_outcomes.last().unwrap();
        assert_eq!(stage.status, StageStatus::Degraded);
        let pre = stage
            .checks
            .iter()
            .find(|check| check.id == "pre_ir:L")
            .unwrap();
        let post = stage
            .checks
            .iter()
            .find(|check| check.id == "post_ir:L")
            .unwrap();
        assert_eq!(
            pre.diagnostic.as_deref(),
            Some("initial measurement phase unavailable")
        );
        assert_eq!(
            post.diagnostic.as_deref(),
            Some("driver woofer: capture unavailable")
        );
        let decoded: StageOutcome =
            serde_json::from_value(serde_json::to_value(stage).unwrap()).unwrap();
        assert_eq!(&decoded, stage);
    }

    #[test]
    fn waveform_refresh_removes_stale_views_without_channel_measurements() {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let initial = &mut result.channel_results.get_mut("L").unwrap().initial_curve;
        initial.phase = Some(ndarray::Array1::zeros(initial.freq.len()));
        super::super::refresh::refresh_temporal_ir_evidence(
            &mut result,
            &roomeq_model::RoomConfig::default(),
            48000.0,
            std::path::Path::new("."),
        );
        assert!(result.channels["L"].pre_ir.is_some());
        result.channel_results.clear();
        super::super::refresh::refresh_temporal_ir_evidence(
            &mut result,
            &roomeq_model::RoomConfig::default(),
            48000.0,
            std::path::Path::new("."),
        );
        assert!(result.channels["L"].pre_ir.is_none());
        assert!(result.channels["L"].post_ir.is_none());
        let stages: Vec<_> = result
            .metadata
            .stage_outcomes
            .iter()
            .filter(|stage| stage.stage == "waveform_views")
            .collect();
        assert_eq!(stages.len(), 1);
        assert!(stages[0].checks.iter().all(|check| !check.passed
            && check.diagnostic.as_deref() == Some("initial measurement unavailable")));
    }
}
