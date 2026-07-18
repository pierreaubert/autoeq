//! Backward-compatible RoomEQ pipeline facade.
//!
//! Application composition is owned by roomeq-workflow and the prepared
//! execution boundary is owned by roomeq-engine. The remaining root kernel is
//! migrated slice-by-slice by WP4-WP8.

use std::collections::HashMap;

use crate::error::Result;

use super::optimize::RoomOptimizationResult;

pub use roomeq_engine::{
    PipelineControl, PipelineEvent, PipelineObserver, PipelineStepId, PipelineStepStatus,
};
pub use roomeq_workflow::RoomPipelineRequest;

/// Compatibility wrapper for the historical root pipeline entry point.
pub struct RoomPipeline<'a> {
    inner: roomeq_workflow::RoomPipeline<'a>,
}

impl<'a> RoomPipeline<'a> {
    /// Create a pipeline for the given request.
    pub fn new(request: RoomPipelineRequest<'a>) -> Self {
        Self {
            inner: roomeq_workflow::RoomPipeline::new(request),
        }
    }

    /// Attach measurements excluded from optimization for runtime quality
    /// validation. Keys use routed output channel names.
    pub fn with_validation_measurements(
        mut self,
        validation_measurements: HashMap<String, Vec<crate::Curve>>,
    ) -> Self {
        self.inner = self
            .inner
            .with_validation_measurements(validation_measurements);
        self
    }

    /// Delegate application composition to the workflow and execution
    /// dispatch to the engine-owned boundary.
    pub fn run(
        self,
        observer: Option<Box<dyn PipelineObserver>>,
    ) -> Result<RoomOptimizationResult> {
        self.inner
            .run_with(observer, super::optimize::optimize_room_pipeline_impl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomeq::types::RoomConfig;

    #[test]
    fn compatibility_pipeline_delegates_error_path() {
        let config = RoomConfig::default();
        let request = RoomPipelineRequest {
            config: &config,
            sample_rate: 48_000.0,
            output_dir: None,
            probe_arrival_overrides: None,
        };

        let result = RoomPipeline::new(request).run(None);

        assert!(result.is_err(), "empty config should produce an error");
    }
}
