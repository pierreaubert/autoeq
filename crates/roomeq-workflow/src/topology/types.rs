use roomeq_engine::error::Result;
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_engine::{PipelineStepId, PipelineStepStatus};
use roomeq_model::{RoomConfig, SystemConfig};
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};

pub struct WorkflowProgressCallback {
    pub callback: roomeq_engine::OptimProgressCallback,
    pub stopped: Arc<AtomicBool>,
}

pub type WorkflowProgressCallbackFactory<'a> =
    dyn FnMut(&str, usize, usize, usize) -> Option<WorkflowProgressCallback> + 'a;

pub type WorkflowStageCallback<'a> =
    dyn FnMut(PipelineStepId, PipelineStepStatus, &str, f64) -> Result<()> + 'a;

/// Assembled inputs passed to a topology-specific workflow executor.
pub(crate) struct WorkflowAssembly<'cfg, 'p, 's> {
    pub config: &'cfg RoomConfig,
    pub sys: &'cfg SystemConfig,
    pub sample_rate: f64,
    pub output_dir: &'cfg Path,
    pub progress_factory: Option<&'p mut WorkflowProgressCallbackFactory<'p>>,
    pub stage_callback: Option<&'s mut WorkflowStageCallback<'s>>,
}

/// Route-specific executor for a RoomEQ workflow topology.
pub(crate) trait WorkflowExecutor {
    fn execute<'cfg, 'p, 's>(
        &self,
        assembly: &mut WorkflowAssembly<'cfg, 'p, 's>,
    ) -> Result<RoomOptimizationResult>;
}
