//! Thin compatibility launcher for the crate-owned RoomEQ command.

use autoeq::roomeq::{RoomPipeline, RoomPipelineRequest};
use roomeq_engine::{PipelineObserver, room_result::RoomOptimizationResult};
use roomeq_model::RoomConfig;
use std::path::Path;

struct LegacyRunner;

impl roomeq_cli::RoomCommandRunner for LegacyRunner {
    fn optimize_room(
        &self,
        config: &RoomConfig,
        sample_rate: f64,
        output_dir: Option<&Path>,
        observer: Option<Box<dyn PipelineObserver>>,
    ) -> anyhow::Result<RoomOptimizationResult> {
        RoomPipeline::new(RoomPipelineRequest {
            config,
            sample_rate,
            output_dir,
            probe_arrival_overrides: None,
        })
        .run(observer)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
    }
}

fn main() -> anyhow::Result<()> {
    roomeq_cli::roomeq::run_command(&LegacyRunner)
}
