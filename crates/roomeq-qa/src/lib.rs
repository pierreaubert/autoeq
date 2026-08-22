//! Reusable RoomEQ QA scenario matrices, runners, and reports.

use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{RoomConfig, StageOutcome, StageStatus};
use std::collections::HashMap;
use std::path::Path;

pub mod acoustic;
pub mod coverage;
pub mod features;
pub mod fuzzer;
pub mod quality;
pub mod registry;
pub mod synthetic;

const QA_SEED_OFFSETS: [u64; 5] = [0, 17, 41, 73, 109];

fn qa_seed_values(config: &RoomConfig) -> [u64; 5] {
    let base = config.optimizer.seed.unwrap_or(42);
    QA_SEED_OFFSETS.map(|offset| base.wrapping_add(offset))
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
        scores.push((seed, result.combined_post_score));
    }
    scores.sort_by(|left, right| left.1.total_cmp(&right.1));
    Ok((scores[scores.len() / 2].0, scores))
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
    let (selected_seed, scores) = select_median_seed(config, |seeded| {
        roomeq_workflow::optimize_room(seeded, sample_rate, None, None)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    })?;
    let mut selected = config.clone();
    selected.optimizer.seed = Some(selected_seed);
    let mut result = roomeq_workflow::optimize_room(&selected, sample_rate, None, output_dir)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    record_seed_distribution(&mut result, selected_seed, &scores);
    Ok(result)
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
