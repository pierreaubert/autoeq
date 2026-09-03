//! Reusable RoomEQ QA scenario matrices, runners, and reports.

use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{CorrectionDecision, RoomConfig, StageOutcome, StageStatus};
use std::collections::HashMap;
use std::path::Path;

pub mod acoustic;
pub mod coverage;
pub mod features;
pub mod fuzzer;
pub mod quality;
pub mod registry;
pub mod stage_contracts;
pub mod parameter_matrix;
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

fn select_median_seed<F>(config: &RoomConfig, mut run: F) -> anyhow::Result<(u64, Vec<(u64, f64)>)>
where
    F: FnMut(&RoomConfig) -> anyhow::Result<RoomOptimizationResult>,
{
    let mut scores = Vec::with_capacity(QA_SEED_OFFSETS.len());
    for seed in qa_seed_values(config) {
        let mut seeded = config.clone();
        seeded.optimizer.seed = Some(seed);
        let result = run(&seeded)?;
        if !result.combined_post_score.is_finite() {
            anyhow::bail!("QA seed {seed} produced a non-finite post score");
        }
        let accepted = result
            .metadata
            .correction_acceptance
            .as_ref()
            .is_none_or(|report| matches!(report.decision, CorrectionDecision::Accepted));
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
        scores.push((seed, result.combined_post_score, accepted));
    }
    scores.sort_by(|left, right| left.1.total_cmp(&right.1));
    let selected_seed = median_accepted_seed(&scores);
    let scores = scores
        .into_iter()
        .map(|(seed, score, _)| (seed, score))
        .collect();
    Ok((selected_seed, scores))
}

fn record_seed_distribution(
    result: &mut RoomOptimizationResult,
    selected_seed: u64,
    scores: &[(u64, f64)],
) {
    let details = scores
        .iter()
        .map(|(seed, score)| format!("{seed}:{score:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let min_score = scores
        .iter()
        .map(|(_, score)| *score)
        .fold(f64::INFINITY, f64::min);
    let max_score = scores
        .iter()
        .map(|(_, score)| *score)
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
        ],
    });
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
    let (selected_seed, scores) = select_median_seed(config, |seeded| {
        roomeq_workflow::optimize_room(seeded, sample_rate, None, None)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    })?;
    let mut selected = config.clone();
    selected.optimizer.seed = Some(selected_seed);
    let mut result = roomeq_workflow::optimize_room(&selected, sample_rate, None, output_dir)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    record_seed_distribution(&mut result, selected_seed, &scores);
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
    let (selected_seed, scores) = select_median_seed(config, |seeded| {
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
    let mut result = roomeq_workflow::RoomPipeline::new(roomeq_workflow::RoomPipelineRequest {
        config: &selected,
        sample_rate,
        output_dir,
        probe_arrival_overrides: None,
    })
    .with_validation_measurements(validation_measurements)
    .run(None)
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    record_seed_distribution(&mut result, selected_seed, &scores);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::median_accepted_seed;

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
