//! Thin compatibility launcher for crate-owned acoustic-corpus QA.

use autoeq::roomeq::{RoomPipeline, RoomPipelineRequest};
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{Curve, RoomConfig};
use std::collections::HashMap;
use std::path::Path;

struct LegacyPipelineRunner;

impl roomeq_qa::RoomPipelineRunner for LegacyPipelineRunner {
    fn optimize_room_with_validation(
        &self,
        config: &RoomConfig,
        sample_rate: f64,
        output_dir: Option<&Path>,
        validation_measurements: HashMap<String, Vec<Curve>>,
    ) -> anyhow::Result<RoomOptimizationResult> {
        RoomPipeline::new(RoomPipelineRequest {
            config,
            sample_rate,
            output_dir,
            probe_arrival_overrides: None,
        })
        .with_validation_measurements(validation_measurements)
        .run(None)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
    }
}

fn main() -> anyhow::Result<()> {
    roomeq_qa::acoustic::run(&LegacyPipelineRunner)
}
