// Multi-sub workflow executor.
//
// Placeholder executor for configurations with multiple physical subwoofers.
// The actual bass-management/routing logic is exercised by the stereo 2.1 and
// home-cinema executors via `preprocess_sub`; this module exists so the
// topology registry can represent multi-sub layouts independently.

use super::super::optimize::RoomOptimizationResult;
use super::types::{WorkflowAssembly, WorkflowExecutor};
use crate::error::{AutoeqError, Result};
use log::info;

#[allow(dead_code)]
pub(in super::super) struct MultisubExecutor;

impl WorkflowExecutor for MultisubExecutor {
    fn execute<'cfg, 'p, 's>(
        &self,
        _assembly: &mut WorkflowAssembly<'cfg, 'p, 's>,
    ) -> Result<RoomOptimizationResult> {
        info!("Running Multi-sub Optimization Workflow");

        Err(AutoeqError::InvalidConfiguration {
            message: "Standalone multi-sub workflow is not yet implemented; use stereo 2.1 or home-cinema topology"
                .to_string(),
        })
    }
}
