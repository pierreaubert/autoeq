//! Reusable RoomEQ QA scenario matrices, runners, and reports.

use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::RoomConfig;
use std::collections::HashMap;
use std::path::Path;

pub mod acoustic;
pub mod coverage;
pub mod features;
pub mod fuzzer;
pub mod quality;
pub mod synthetic;

pub(crate) fn optimize_room(
    config: &RoomConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
) -> anyhow::Result<RoomOptimizationResult> {
    roomeq_workflow::optimize_room(config, sample_rate, None, output_dir)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

pub(crate) fn optimize_room_with_validation(
    config: &RoomConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
    validation_measurements: HashMap<String, Vec<roomeq_model::Curve>>,
) -> anyhow::Result<RoomOptimizationResult> {
    roomeq_workflow::RoomPipeline::new(roomeq_workflow::RoomPipelineRequest {
        config,
        sample_rate,
        output_dir,
        probe_arrival_overrides: None,
    })
    .with_validation_measurements(validation_measurements)
    .run(None)
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}
