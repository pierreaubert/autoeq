//! Reusable RoomEQ QA scenario matrices, runners, and reports.

use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{
    CorrectionDecision, QaSeedDistribution, QaSeedOutcome, RoomConfig, StageOutcome, StageStatus,
};
use std::collections::HashMap;
use std::path::Path;

pub mod acoustic;
pub mod coverage;
pub mod features;
pub mod fuzzer;
pub mod parameter_matrix;
pub mod quality;
pub mod registry;
pub mod release_gates;
pub mod stage_contracts;
pub mod synthetic;

const QA_SEED_OFFSETS: [u64; 5] = [0, 17, 41, 73, 109];

fn qa_seed_values(config: &RoomConfig) -> [u64; 5] {
    let base = config.optimizer.seed.unwrap_or(42);
    QA_SEED_OFFSETS.map(|offset| base.wrapping_add(offset))
}

fn median_accepted_seed(scores: &[(u64, f64, bool)]) -> u64 {
    let accepted: Vec<_> = scores.iter().filter(|(_, _, accepted)| *accepted).collect();
    let pool: Vec<_> = if accepted.is_empty() {
        scores.iter().collect()
    } else {
        accepted
    };
    pool[pool.len() / 2].0
}

fn select_median_seed<F>(
    config: &RoomConfig,
    sample_rate: f64,
    run: F,
) -> anyhow::Result<(u64, Vec<QaSeedOutcome>)>
where
    F: FnMut(&RoomConfig) -> anyhow::Result<RoomOptimizationResult>,
{
    select_median_seed_recording(config, run, |record| {
        append_seed_record_at_rate(record, sample_rate)
    })
}

fn select_median_seed_recording<F, E>(
    config: &RoomConfig,
    mut run: F,
    mut record_failure: E,
) -> anyhow::Result<(u64, Vec<QaSeedOutcome>)>
where
    F: FnMut(&RoomConfig) -> anyhow::Result<RoomOptimizationResult>,
    E: FnMut(&serde_json::Value) -> anyhow::Result<()>,
{
    let mut scores = Vec::with_capacity(QA_SEED_OFFSETS.len());
    let mut errors = Vec::new();
    for seed in qa_seed_values(config) {
        let mut seeded = config.clone();
        seeded.optimizer.seed = Some(seed);
        let result = match run(&seeded) {
            Ok(result) => result,
            Err(error) => {
                errors.push(serde_json::json!({"seed": seed, "error": format!("{error:#}")}));
                continue;
            }
        };
        if !result.combined_pre_score.is_finite() || !result.combined_post_score.is_finite() {
            errors.push(serde_json::json!({
                "seed": seed, "error": "non-finite pre/post score",
                "acceptance": result.metadata.correction_acceptance,
                "optimizer_evidence": result.metadata.optimizer_evidence,
                "stage_outcomes": result.metadata.stage_outcomes,
            }));
            continue;
        }
        let accepted = result
            .metadata
            .correction_acceptance
            .as_ref()
            .is_some_and(|report| {
                report.accepted && matches!(report.decision, CorrectionDecision::Accepted)
            });
        log::debug!(
            "QA seed candidate {seed}: post {:.6}, decision {:?}, violations {:?}, reverted stages {:?}, max boost {:?} dB, available headroom {:?} dB, accepted candidate {accepted}",
            result.combined_post_score,
            result
                .metadata
                .correction_acceptance
                .as_ref()
                .map(|report| &report.decision),
            result
                .metadata
                .correction_acceptance
                .as_ref()
                .map(|report| &report.violations),
            result
                .metadata
                .correction_acceptance
                .as_ref()
                .map(|report| &report.reverted_stages),
            result
                .metadata
                .correction_acceptance
                .as_ref()
                .and_then(|report| report.acoustic_quality.as_ref())
                .map(|quality| quality.max_boost_db),
            result
                .metadata
                .correction_acceptance
                .as_ref()
                .and_then(|report| report.acoustic_quality.as_ref())
                .and_then(|quality| quality.temporal.available_headroom_db),
        );
        // Absence of a Failed stage is not affirmative safety evidence:
        // degraded routed replay can restore DSP that failed acceptance.
        // Conservatively count only explicitly accepted final output. A
        // reverted/unknown output may be safe, but is not certified here.
        let safe_output = result
            .metadata
            .correction_acceptance
            .as_ref()
            .is_some_and(|report| report.accepted)
            && !result
                .metadata
                .stage_outcomes
                .iter()
                .any(|s| s.status == StageStatus::Failed);
        scores.push(QaSeedOutcome {
            seed,
            pre_score: result.combined_pre_score,
            post_score: result.combined_post_score,
            accepted_useful: safe_output
                && accepted
                && result.combined_pre_score - result.combined_post_score
                    > 1e-6 * result.combined_pre_score.abs().max(1.0),
            safe_output,
            acceptance: result.metadata.correction_acceptance,
            optimizer_evidence: result.metadata.optimizer_evidence,
            stage_outcomes: result.metadata.stage_outcomes,
        });
    }
    if !errors.is_empty() {
        record_failure(&failed_seed_record(
            config,
            "seed_selection",
            None,
            &scores,
            &errors,
        ))?;
        anyhow::bail!(
            "{} of {} QA seeds failed; reliability evidence recorded",
            errors.len(),
            QA_SEED_OFFSETS.len()
        );
    }
    scores.sort_by(|left, right| left.post_score.total_cmp(&right.post_score));
    let selection: Vec<_> = scores
        .iter()
        .map(|outcome| {
            (
                outcome.seed,
                outcome.post_score,
                outcome.acceptance.as_ref().is_some_and(|report| {
                    report.accepted && report.decision == CorrectionDecision::Accepted
                }),
            )
        })
        .collect();
    let selected_seed = median_accepted_seed(&selection);
    Ok((selected_seed, scores))
}

fn failed_seed_record(
    config: &RoomConfig,
    phase: &str,
    selected_seed: Option<u64>,
    outcomes: &[QaSeedOutcome],
    errors: &[serde_json::Value],
) -> serde_json::Value {
    let count = QA_SEED_OFFSETS.len() as f64;
    serde_json::json!({
        "status": "failed", "phase": phase, "selected_seed": selected_seed,
        "requested_seeds": qa_seed_values(config),
        "optimizer": config.optimizer,
        "outcomes": outcomes, "errors": errors,
        "accepted_useful_rate": outcomes.iter().filter(|o| o.accepted_useful).count() as f64 / count,
        "safe_output_rate": outcomes.iter().filter(|o| o.safe_output).count() as f64 / count,
        "rate_scope": "five_seed_selection_population",
        "safe_output_scope": "correction_policy_acceptance_not_electrical_or_native_safety",
        "final_artifact_delivered": false,
    })
}

fn append_seed_record(value: &serde_json::Value) -> anyhow::Result<()> {
    use std::io::Write;
    static ARTIFACT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = ARTIFACT_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("QA seed artifact lock poisoned"))?;
    std::fs::create_dir_all("target/qa")?;
    let mut artifact = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("target/qa/roomeq-seed-distributions.jsonl")?;
    let mut record = serde_json::to_vec(value)?;
    record.push(b'\n');
    artifact.write_all(&record)?;
    Ok(())
}

fn append_seed_record_at_rate(value: &serde_json::Value, sample_rate: f64) -> anyhow::Result<()> {
    let mut record = value.clone();
    record["sample_rate"] = serde_json::json!(sample_rate);
    append_seed_record(&record)
}

fn record_seed_distribution<E>(
    config: &RoomConfig,
    result: &mut RoomOptimizationResult,
    selected_seed: u64,
    scores: &[QaSeedOutcome],
    mut record: E,
) -> anyhow::Result<()>
where
    E: FnMut(&serde_json::Value) -> anyhow::Result<()>,
{
    let details = scores
        .iter()
        .map(|outcome| {
            format!(
                "{}:{:.6}:useful={}",
                outcome.seed, outcome.post_score, outcome.accepted_useful
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let min_score = scores
        .iter()
        .map(|outcome| outcome.post_score)
        .fold(f64::INFINITY, f64::min);
    let max_score = scores
        .iter()
        .map(|outcome| outcome.post_score)
        .fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "QA seed distribution: selected={selected_seed}, min={min_score:.6}, max={max_score:.6}, spread={:.6}, scores=[{details}]",
        max_score - min_score
    );
    result.metadata.stage_outcomes.push(StageOutcome {
        stage: "qa_seed_distribution".to_string(),
        status: StageStatus::Applied,
        checks: Vec::new(),
        advisories: vec![
            format!("selected_median_seed={selected_seed}"),
            format!("seed_post_scores={details}"),
            "safe_output_scope=correction_policy_acceptance_not_electrical_or_native_safety".into(),
        ],
    });
    let distribution = seed_distribution(selected_seed, scores);
    eprintln!(
        "QA reliability: useful={:.0}% policy_accepted={:.0}% ({} seeds; not electrical/native safety)",
        100.0 * distribution.accepted_useful_rate,
        100.0 * distribution.safe_output_rate,
        scores.len()
    );
    // Many QA adapters retain only their selected scorecard and remove their
    // temporary DSP directory. Preserve every population in a durable JSONL
    // artifact as well as in the returned DSP metadata.
    let mut channels: Vec<_> = result.channels.keys().collect();
    channels.sort();
    record(&serde_json::json!({
        "status": "completed",
        "phase": "selected_artifact_run",
        "safe_output_scope": "correction_policy_acceptance_not_electrical_or_native_safety",
        "optimizer": config.optimizer,
        "requested_seeds": qa_seed_values(config),
        "rate_scope": "five_seed_selection_population",
        "final_artifact_delivered": true,
        "final_artifact": {
            "seed": selected_seed,
            "pre_score": result.combined_pre_score,
            "post_score": result.combined_post_score,
            "acceptance": result.metadata.correction_acceptance,
            "optimizer_evidence": result.metadata.optimizer_evidence,
            "stage_outcomes": result.metadata.stage_outcomes,
        },
        "algorithm": result.metadata.algorithm,
        "loss_type": result.metadata.loss_type,
        "timestamp": result.metadata.timestamp,
        "channels": channels,
        "seed_distribution": &distribution,
    }))?;
    result.metadata.qa_seed_distribution = Some(distribution);
    Ok(())
}

fn seed_distribution(selected_seed: u64, outcomes: &[QaSeedOutcome]) -> QaSeedDistribution {
    let count = outcomes.len().max(1) as f64;
    let low = outcomes
        .iter()
        .map(|o| o.post_score)
        .reduce(f64::min)
        .unwrap_or(0.0);
    let high = outcomes
        .iter()
        .map(|o| o.post_score)
        .reduce(f64::max)
        .unwrap_or(0.0);
    QaSeedDistribution {
        selected_seed,
        accepted_useful_rate: outcomes.iter().filter(|o| o.accepted_useful).count() as f64 / count,
        safe_output_rate: outcomes.iter().filter(|o| o.safe_output).count() as f64 / count,
        post_score_spread: high - low,
        outcomes: outcomes.to_vec(),
    }
}

fn finish_selected_seed<E>(
    config: &RoomConfig,
    selected_seed: u64,
    scores: &[QaSeedOutcome],
    result: anyhow::Result<RoomOptimizationResult>,
    mut record_outcome: E,
) -> anyhow::Result<RoomOptimizationResult>
where
    E: FnMut(&serde_json::Value) -> anyhow::Result<()>,
{
    let result = result.and_then(|result| {
        anyhow::ensure!(
            result.combined_pre_score.is_finite() && result.combined_post_score.is_finite(),
            "selected QA seed {selected_seed} produced a non-finite pre/post score"
        );
        Ok(result)
    });
    match result {
        Ok(mut result) => {
            record_seed_distribution(
                config,
                &mut result,
                selected_seed,
                scores,
                &mut record_outcome,
            )?;
            Ok(result)
        }
        Err(error) => {
            record_outcome(&failed_seed_record(
                config,
                "selected_artifact_run",
                Some(selected_seed),
                scores,
                &[serde_json::json!({"seed": selected_seed, "error": format!("{error:#}")})],
            ))?;
            Err(error)
        }
    }
}

pub(crate) fn optimize_room(
    config: &RoomConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
) -> anyhow::Result<RoomOptimizationResult> {
    optimize_room_with_selected_seed(config, sample_rate, output_dir).map(|(result, _)| result)
}

pub(crate) fn optimize_room_with_selected_seed(
    config: &RoomConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
) -> anyhow::Result<(RoomOptimizationResult, u64)> {
    let (selected_seed, scores) = select_median_seed(config, sample_rate, |seeded| {
        roomeq_workflow::optimize_room(seeded, sample_rate, None, None)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    })?;
    let mut selected = config.clone();
    selected.optimizer.seed = Some(selected_seed);
    let result = roomeq_workflow::optimize_room(&selected, sample_rate, None, output_dir)
        .map_err(|error| anyhow::anyhow!(error.to_string()));
    let result = finish_selected_seed(config, selected_seed, &scores, result, |record| {
        append_seed_record_at_rate(record, sample_rate)
    })?;
    Ok((result, selected_seed))
}

pub(crate) fn optimize_room_single_seed(
    config: &RoomConfig,
    sample_rate: f64,
) -> anyhow::Result<RoomOptimizationResult> {
    roomeq_workflow::optimize_room(config, sample_rate, None, None)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

pub(crate) fn optimize_room_with_validation(
    config: &RoomConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
    validation_measurements: HashMap<String, Vec<roomeq_model::Curve>>,
) -> anyhow::Result<RoomOptimizationResult> {
    let (selected_seed, scores) = select_median_seed(config, sample_rate, |seeded| {
        roomeq_workflow::RoomPipeline::new(roomeq_workflow::RoomPipelineRequest {
            config: seeded,
            sample_rate,
            output_dir: None,
            probe_arrival_overrides: None,
        })
        .with_validation_measurements(validation_measurements.clone())
        .run(None)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
    })?;
    let mut selected = config.clone();
    selected.optimizer.seed = Some(selected_seed);
    let result = roomeq_workflow::RoomPipeline::new(roomeq_workflow::RoomPipelineRequest {
        config: &selected,
        sample_rate,
        output_dir,
        probe_arrival_overrides: None,
    })
    .with_validation_measurements(validation_measurements)
    .run(None)
    .map_err(|error| anyhow::anyhow!(error.to_string()));
    finish_selected_seed(config, selected_seed, &scores, result, |record| {
        append_seed_record_at_rate(record, sample_rate)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_fixture(post_score: f64) -> RoomOptimizationResult {
        RoomOptimizationResult {
            channels: HashMap::new(),
            channel_results: HashMap::new(),
            deployed_source_curves: HashMap::new(),
            combined_pre_score: 2.0,
            combined_post_score: post_score,
            metadata: serde_json::from_value(serde_json::json!({
                "pre_score": 2.0, "post_score": 1.0, "algorithm": "injected",
                "iterations": 0, "timestamp": "test"
            }))
            .unwrap(),
        }
    }

    #[test]
    fn seed_errors_preserve_all_attempts_and_fail_the_population() {
        for error_count in [0, 1, 5] {
            let config = RoomConfig::default();
            let mut attempted = Vec::new();
            let mut records = Vec::new();
            let result = select_median_seed_recording(
                &config,
                |seeded| {
                    let seed = seeded.optimizer.seed.unwrap();
                    attempted.push(seed);
                    if attempted.len() <= error_count {
                        anyhow::bail!("injected seed {seed} failure");
                    }
                    Ok(seed_fixture(1.0))
                },
                |record| {
                    records.push(record.clone());
                    Ok(())
                },
            );
            assert_eq!(attempted, qa_seed_values(&config));
            assert_eq!(result.is_err(), error_count != 0);
            if error_count == 0 {
                assert!(records.is_empty());
                continue;
            }
            assert_eq!(records.len(), 1);
            let record = &records[0];
            assert_eq!(record["status"], "failed");
            assert_eq!(record["errors"].as_array().unwrap().len(), error_count);
            assert_eq!(
                record["outcomes"].as_array().unwrap().len(),
                5 - error_count
            );
            assert_eq!(record["safe_output_rate"], 0.0);
            assert_eq!(record["accepted_useful_rate"], 0.0);
            assert_eq!(record["final_artifact_delivered"], false);
            assert!(record["selected_seed"].is_null());
        }
    }

    #[test]
    fn nonfinite_seed_preserves_failure_evidence_and_runs_remaining_seeds() {
        let mut attempted = 0;
        let mut records = Vec::new();
        let result = select_median_seed_recording(
            &RoomConfig::default(),
            |_| {
                attempted += 1;
                Ok(seed_fixture(if attempted == 1 { f64::NAN } else { 1.0 }))
            },
            |record| {
                records.push(record.clone());
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(attempted, 5);
        assert_eq!(
            records[0]["errors"][0]["error"],
            "non-finite pre/post score"
        );
        assert_eq!(records[0]["outcomes"].as_array().unwrap().len(), 4);
        assert!(records[0]["errors"][0].get("optimizer_evidence").is_some());
    }

    #[test]
    fn seed_failure_artifact_errors_cannot_be_reported_as_success() {
        let error = select_median_seed_recording(
            &RoomConfig::default(),
            |_| anyhow::bail!("injected optimizer failure"),
            |_| anyhow::bail!("injected evidence write failure"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("evidence write failure"));
    }

    #[test]
    fn selected_seed_completion_retains_context_and_independent_rerun_verdict() {
        let config = RoomConfig::default();
        let (selected, mut outcomes) = select_median_seed_recording(
            &config,
            |_| Ok(seed_fixture(1.0)),
            |_| panic!("no seed failed"),
        )
        .unwrap();
        for outcome in &mut outcomes {
            outcome.safe_output = true;
            outcome.accepted_useful = true;
        }
        let mut records = Vec::new();
        let result = finish_selected_seed(
            &config,
            selected,
            &outcomes,
            Ok(seed_fixture(1.5)),
            |record| {
                records.push(record.clone());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record["status"], "completed");
        assert_eq!(
            record["optimizer"],
            serde_json::to_value(&config.optimizer).unwrap()
        );
        assert_eq!(
            record["requested_seeds"],
            serde_json::json!(qa_seed_values(&config))
        );
        assert_eq!(record["rate_scope"], "five_seed_selection_population");
        assert_eq!(
            record["safe_output_scope"],
            "correction_policy_acceptance_not_electrical_or_native_safety"
        );
        assert_eq!(record["final_artifact_delivered"], true);
        assert_eq!(record["seed_distribution"]["accepted_useful_rate"], 1.0);
        // A successful rerun is not inferred to be accepted from selection rates.
        assert!(record["final_artifact"]["acceptance"].is_null());
        assert_eq!(record["final_artifact"]["post_score"], 1.5);
        assert_eq!(record["final_artifact"]["seed"], selected);
        assert_eq!(
            result.metadata.qa_seed_distribution.unwrap().outcomes,
            outcomes
        );
        let error =
            finish_selected_seed(&config, selected, &outcomes, Ok(seed_fixture(1.5)), |_| {
                anyhow::bail!("injected completion evidence write failure")
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("completion evidence write failure")
        );
    }

    #[test]
    fn selected_seed_artifact_failure_retains_population_without_claiming_delivery() {
        let config = RoomConfig::default();
        let (selected, mut outcomes) = select_median_seed_recording(
            &config,
            |_| Ok(seed_fixture(1.0)),
            |_| panic!("no seed failed"),
        )
        .unwrap();
        // Even a fully useful selection population does not prove the final
        // rerun successfully emitted its artifacts.
        for outcome in &mut outcomes {
            outcome.safe_output = true;
            outcome.accepted_useful = true;
        }
        let mut records = Vec::new();
        let result = finish_selected_seed(
            &config,
            selected,
            &outcomes,
            Err(anyhow::anyhow!("sidecar write failed")),
            |record| {
                records.push(record.clone());
                Ok(())
            },
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("sidecar write failed")
        );
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record["phase"], "selected_artifact_run");
        assert_eq!(record["selected_seed"], selected);
        assert_eq!(record["outcomes"].as_array().unwrap().len(), 5);
        assert_eq!(record["safe_output_rate"], 1.0);
        assert_eq!(record["accepted_useful_rate"], 1.0);
        assert_eq!(record["rate_scope"], "five_seed_selection_population");
        assert_eq!(record["final_artifact_delivered"], false);
    }

    #[test]
    fn selected_seed_nonfinite_rerun_is_a_recorded_failure() {
        let mut records = Vec::new();
        let result = finish_selected_seed(
            &RoomConfig::default(),
            42,
            &[],
            Ok(seed_fixture(f64::INFINITY)),
            |record| {
                records.push(record.clone());
                Ok(())
            },
        );
        assert!(result.unwrap_err().to_string().contains("non-finite"));
        assert_eq!(records[0]["status"], "failed");
        assert_eq!(records[0]["phase"], "selected_artifact_run");
        assert_eq!(records[0]["final_artifact_delivered"], false);
    }

    #[test]
    fn identical_selected_scores_do_not_hide_seed_reliability() {
        for accepted_count in [0, 1, 5] {
            let curve = |spl: Vec<f64>| roomeq_model::Curve {
                freq: vec![20.0, 100.0, 1000.0, 20_000.0].into(),
                spl: spl.into(),
                ..Default::default()
            };
            let mut index = 0;
            let (selected, outcomes) = select_median_seed(&RoomConfig::default(), 48_000.0, |_| {
                let accepted = index < accepted_count;
                index += 1;
                let mut acceptance = roomeq_quality::evaluate_correction_acceptance(
                    &curve(vec![0.0, 6.0, 0.0, 0.0]),
                    &curve(vec![0.0; 4]),
                    &curve(vec![0.0; 4]),
                    None,
                    roomeq_model::CorrectionAcceptancePolicy::CorrectableFixture,
                )
                .unwrap();
                acceptance.accepted = accepted;
                if !accepted {
                    acceptance.decision = CorrectionDecision::RevertedStage;
                    acceptance.violations.push("injected_regression".into());
                    acceptance.reverted_stages.push("primary_correction".into());
                }
                let mut metadata: roomeq_model::OptimizationMetadata =
                    serde_json::from_value(serde_json::json!({"pre_score": 2.0, "post_score": 1.0,
                        "algorithm": "injected", "iterations": 1, "timestamp": "test"}))
                    .unwrap();
                metadata.correction_acceptance = Some(acceptance);
                metadata.stage_outcomes = vec![StageOutcome {
                    stage: "injected_final_outcome".into(),
                    status: if accepted {
                        StageStatus::Applied
                    } else {
                        StageStatus::Degraded
                    },
                    checks: vec![],
                    advisories: if accepted {
                        vec![]
                    } else {
                        vec!["regression_reverted".into()]
                    },
                }];
                Ok(RoomOptimizationResult {
                    channels: HashMap::new(),
                    channel_results: HashMap::new(),
                    deployed_source_curves: HashMap::new(),
                    combined_pre_score: 2.0,
                    combined_post_score: 1.0,
                    metadata,
                })
            })
            .unwrap();
            let report = seed_distribution(selected, &outcomes);
            assert_eq!(report.accepted_useful_rate, accepted_count as f64 / 5.0);
            assert_eq!(report.safe_output_rate, accepted_count as f64 / 5.0);
            assert_eq!(report.post_score_spread, 0.0);
            assert_eq!(report.outcomes.len(), 5);
            let serialized = serde_json::to_string(&report).unwrap();
            let restored: QaSeedDistribution = serde_json::from_str(&serialized).unwrap();
            assert_eq!(report, restored);
            assert_eq!(
                restored
                    .outcomes
                    .iter()
                    .filter(|o| o.stage_outcomes[0]
                        .advisories
                        .contains(&"regression_reverted".to_string()))
                    .count(),
                5 - accepted_count as usize
            );
        }
    }

    #[test]
    fn missing_acceptance_is_not_safe_output_evidence() {
        let (selected, outcomes) = select_median_seed(&RoomConfig::default(), 48_000.0, |_| {
            let metadata = serde_json::from_value(serde_json::json!({
                "pre_score": 1.0, "post_score": 1.0, "algorithm": "fixture",
                "iterations": 0, "timestamp": "test"
            }))
            .unwrap();
            Ok(RoomOptimizationResult {
                channels: HashMap::new(),
                channel_results: HashMap::new(),
                deployed_source_curves: HashMap::new(),
                combined_pre_score: 1.0,
                combined_post_score: 1.0,
                metadata,
            })
        })
        .unwrap();
        let distribution = seed_distribution(selected, &outcomes);
        assert_eq!(distribution.safe_output_rate, 0.0);
        assert_eq!(distribution.accepted_useful_rate, 0.0);
        assert!(outcomes.iter().all(|outcome| outcome.acceptance.is_none()));
    }

    #[test]
    fn median_seed_prefers_runtime_accepted_candidates() {
        let scores = [
            (1, 1.0, false),
            (2, 2.0, true),
            (3, 3.0, false),
            (4, 4.0, true),
            (5, 5.0, false),
        ];
        assert_eq!(median_accepted_seed(&scores), 4);
    }

    #[test]
    fn median_seed_falls_back_to_all_candidates_when_every_run_reverts() {
        let scores = [
            (1, 1.0, false),
            (2, 2.0, false),
            (3, 3.0, false),
            (4, 4.0, false),
            (5, 5.0, false),
        ];
        assert_eq!(median_accepted_seed(&scores), 3);
    }
}
