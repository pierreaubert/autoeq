// r2factor:facade — do not pass this file back into r2factor
// Bass-management workflow executor.
//
// Placeholder executor for standalone bass-management optimization. The
// stereo 2.1 and home-cinema executors already embed bass-management
// crossover/routing logic; this module provides a registry entry point for
// pure bass-management workflows.

use super::super::types::{WorkflowAssembly, WorkflowExecutor};
use crate::error::{AutoeqError, Result};
use crate::roomeq::optimize::RoomOptimizationResult;
use log::info;

#[allow(dead_code)]
pub(in super::super) struct BassManagementExecutor;

impl WorkflowExecutor for BassManagementExecutor {
    fn execute<'cfg, 'p, 's>(
        &self,
        _assembly: &mut WorkflowAssembly<'cfg, 'p, 's>,
    ) -> Result<RoomOptimizationResult> {
        info!("Running Bass-Management Optimization Workflow");

        Err(AutoeqError::InvalidConfiguration {
            message: "Standalone bass-management workflow is not yet implemented; use stereo 2.1 or home-cinema topology"
                .to_string(),
        })
    }
}
