//! Thin compatibility launcher for crate-owned RoomEQ quality QA.

use autoeq::roomeq::{CallbackAction, RoomOptimizationProgress};
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::RoomConfig;
use std::path::Path;
use std::sync::Arc;

struct LegacyOptimizer;

impl roomeq_qa::RoomOptimizer for LegacyOptimizer {
    fn optimize_room(
        &self,
        config: &RoomConfig,
        sample_rate: f64,
        output_dir: Option<&Path>,
    ) -> anyhow::Result<RoomOptimizationResult> {
        let callback = Box::new(|_: &RoomOptimizationProgress| CallbackAction::Continue);
        autoeq::roomeq::optimize_room(config, sample_rate, Some(callback), output_dir)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    }
}

fn main() -> anyhow::Result<()> {
    if roomeq_qa::quality::run(Arc::new(LegacyOptimizer))? {
        std::process::exit(1);
    }
    Ok(())
}
