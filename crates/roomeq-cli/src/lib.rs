//! RoomEQ command-line adapters.

use roomeq_engine::{PipelineObserver, room_result::RoomOptimizationResult};
use roomeq_model::RoomConfig;
use std::path::Path;

pub mod convert_recording;
pub mod roomeq;

/// Temporary execution port used while WP11 moves the final root kernel.
pub trait RoomCommandRunner: Send + Sync {
    fn optimize_room(
        &self,
        config: &RoomConfig,
        sample_rate: f64,
        output_dir: Option<&Path>,
        observer: Option<Box<dyn PipelineObserver>>,
    ) -> anyhow::Result<RoomOptimizationResult>;
}
