//! Reusable RoomEQ QA scenario matrices, runners, and reports.

use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::{Curve, RoomConfig};
use std::collections::HashMap;
use std::path::Path;

pub mod acoustic;
pub mod coverage;
pub mod features;
pub mod fuzzer;
pub mod quality;
pub mod synthetic;

/// Injected execution service used by QA runners while the production
/// optimization kernel is migrated out of the compatibility root crate.
pub trait RoomOptimizer: Send + Sync {
    fn optimize_room(
        &self,
        config: &RoomConfig,
        sample_rate: f64,
        output_dir: Option<&Path>,
    ) -> anyhow::Result<RoomOptimizationResult>;
}

impl<F> RoomOptimizer for F
where
    F: Fn(&RoomConfig, f64, Option<&Path>) -> anyhow::Result<RoomOptimizationResult> + Send + Sync,
{
    fn optimize_room(
        &self,
        config: &RoomConfig,
        sample_rate: f64,
        output_dir: Option<&Path>,
    ) -> anyhow::Result<RoomOptimizationResult> {
        self(config, sample_rate, output_dir)
    }
}

/// Injected application-pipeline service for QA that supplies held-out
/// validation measurements to the temporary production kernel.
pub trait RoomPipelineRunner: Send + Sync {
    fn optimize_room_with_validation(
        &self,
        config: &RoomConfig,
        sample_rate: f64,
        output_dir: Option<&Path>,
        validation_measurements: HashMap<String, Vec<Curve>>,
    ) -> anyhow::Result<RoomOptimizationResult>;
}
