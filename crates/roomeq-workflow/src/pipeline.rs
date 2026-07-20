//! RoomEQ application pipeline composition.

use std::collections::HashMap;
use std::path::Path;

use autoeq_artifacts::{ArtifactStore, FsArtifactStore};
use roomeq_engine::{
    EngineRequest, PipelineObserver, RoomEngine, room_result::RoomOptimizationResult,
};
use roomeq_model::{Curve, Result, RoomConfig};

/// Application-owned data accompanying an in-memory engine request.
pub struct WorkflowContext<'a> {
    /// Optional destination for generated artifacts.
    pub output_dir: Option<&'a Path>,
    /// Artifact persistence selected by the application.
    pub artifact_store: &'a dyn ArtifactStore,
    /// Measurements excluded from optimization and reserved for validation.
    pub validation_measurements: &'a HashMap<String, Vec<Curve>>,
}

/// Request data for a RoomEQ workflow run.
#[derive(Clone, Copy)]
pub struct RoomPipelineRequest<'a> {
    /// Complete room configuration.
    pub config: &'a RoomConfig,
    /// Sample rate for filter design.
    pub sample_rate: f64,
    /// Optional directory for generated artifacts.
    pub output_dir: Option<&'a Path>,
    /// Optional per-channel probe-based arrival times in milliseconds.
    pub probe_arrival_overrides: Option<&'a HashMap<String, f64>>,
}

/// Observable RoomEQ application pipeline.
pub struct RoomPipeline<'a> {
    request: RoomPipelineRequest<'a>,
    validation_measurements: HashMap<String, Vec<Curve>>,
}

impl<'a> RoomPipeline<'a> {
    /// Create a workflow for the given request.
    pub fn new(request: RoomPipelineRequest<'a>) -> Self {
        Self {
            request,
            validation_measurements: HashMap::new(),
        }
    }

    /// Attach measurements excluded from optimization for runtime quality
    /// validation. Keys use routed output channel names.
    pub fn with_validation_measurements(
        mut self,
        validation_measurements: HashMap<String, Vec<Curve>>,
    ) -> Self {
        self.validation_measurements = validation_measurements;
        self
    }

    /// Run the canonical RoomEQ optimization workflow with the production
    /// filesystem artifact store.
    pub fn run(
        self,
        observer: Option<Box<dyn PipelineObserver>>,
    ) -> Result<RoomOptimizationResult> {
        let artifact_store = FsArtifactStore::new();
        self.run_with_store(&artifact_store, observer)
    }

    /// Run with an injected artifact store.
    ///
    /// This is the root-free test seam for application composition. Production
    /// uses [Self::run] and therefore selects the filesystem adapter here, not
    /// in the engine.
    pub fn run_with_store(
        self,
        artifact_store: &dyn ArtifactStore,
        observer: Option<Box<dyn PipelineObserver>>,
    ) -> Result<RoomOptimizationResult> {
        let engine_request = EngineRequest {
            config: self.request.config,
            sample_rate: self.request.sample_rate,
            probe_arrival_overrides: self.request.probe_arrival_overrides,
        };
        let context = WorkflowContext {
            output_dir: self.request.output_dir,
            artifact_store,
            validation_measurements: &self.validation_measurements,
        };

        RoomEngine.run(engine_request, observer, move |request, observer| {
            crate::room_optimization::optimize_room_pipeline_impl(request, &context, observer)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use autoeq_artifacts::MemoryArtifactStore;
    use roomeq_engine::{PipelineControl, PipelineEvent};

    use super::*;

    #[test]
    fn root_free_pipeline_composes_store_observer_and_engine() {
        let config = RoomConfig::default();
        let store = MemoryArtifactStore::new();
        let event_count = Arc::new(AtomicUsize::new(0));
        let observer_count = Arc::clone(&event_count);
        let observer = move |_: &PipelineEvent| {
            observer_count.fetch_add(1, Ordering::Relaxed);
            PipelineControl::Continue
        };
        let request = RoomPipelineRequest {
            config: &config,
            sample_rate: 48_000.0,
            output_dir: Some(Path::new("artifacts")),
            probe_arrival_overrides: None,
        };

        let result = RoomPipeline::new(request).run_with_store(&store, Some(Box::new(observer)));

        assert!(result.is_err(), "empty config should fail validation");
        assert!(event_count.load(Ordering::Relaxed) > 0);
    }
}
