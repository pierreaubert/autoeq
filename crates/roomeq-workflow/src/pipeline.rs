//! RoomEQ application pipeline composition.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use autoeq_artifacts::{ArtifactStore, FsArtifactStore};
use roomeq_engine::{
    EngineRequest, PipelineObserver, RoomEngine, room_result::RoomOptimizationResult,
};
use roomeq_model::{Curve, Result, RoomConfig};

use crate::DEFAULT_FREQUENCY_SAMPLES;

/// Receives immutable JSON events from an explicitly enabled finalization trace.
pub trait FinalizationDiagnosticSink {
    /// Persist one named event without changing optimization behavior.
    ///
    /// Implementations should reject duplicate event names and use atomic,
    /// no-overwrite file publication when writing to disk.
    ///
    /// # Errors
    ///
    /// Returns an error when the event cannot be serialized or durably stored.
    fn write_event(&self, name: &str, json: &[u8]) -> io::Result<()>;
}

/// Exact finalization trial selected for opt-in diagnostic capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalizationDiagnosticTrial {
    /// Capture the unmodified-strength, output-safety candidate before fallback selection.
    ZeroStrengthOutput,
}

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
    frequency_samples: usize,
    finalization_diagnostic: Option<(
        FinalizationDiagnosticTrial,
        &'a dyn FinalizationDiagnosticSink,
    )>,
    recovery_session: Option<crate::room_recovery::RoomRecoverySession>,
}

impl<'a> RoomPipeline<'a> {
    /// Create a workflow for the given request.
    pub fn new(request: RoomPipelineRequest<'a>) -> Self {
        Self {
            request,
            validation_measurements: HashMap::new(),
            frequency_samples: DEFAULT_FREQUENCY_SAMPLES,
            finalization_diagnostic: None,
            recovery_session: None,
        }
    }

    /// Attach a sink for opt-in finalization candidate diagnostics.
    ///
    /// The trace is restricted to the zero-strength output candidate. Normal
    /// runs perform no diagnostic writes when no sink is attached.
    pub fn with_finalization_diagnostic_sink(
        mut self,
        trial: FinalizationDiagnosticTrial,
        sink: &'a dyn FinalizationDiagnosticSink,
    ) -> Self {
        self.finalization_diagnostic = Some((trial, sink));
        self
    }

    /// Set the number of log-frequency samples used when reducing dense
    /// measurements before optimization.
    pub fn with_frequency_samples(mut self, frequency_samples: usize) -> Self {
        self.frequency_samples = frequency_samples;
        self
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

    /// Attach a single-channel exact-DE crash-recovery session.
    ///
    /// The session rejects configurations outside its explicitly supported
    /// lane before the optimizer dispatches any search work.
    pub fn with_recovery_session(
        mut self,
        recovery_session: crate::room_recovery::RoomRecoverySession,
    ) -> Self {
        self.recovery_session = Some(recovery_session);
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
        // Programmatic callers need the same canonical resolution as file loads.
        let mut config = self.request.config.clone();
        config.resolve_room_dimensions();
        let engine_request = EngineRequest {
            config: &config,
            sample_rate: self.request.sample_rate,
            probe_arrival_overrides: self.request.probe_arrival_overrides,
        };
        let context = WorkflowContext {
            output_dir: self.request.output_dir,
            artifact_store,
            validation_measurements: &self.validation_measurements,
        };
        let finalization_diagnostic = self.finalization_diagnostic;
        let frequency_samples = self.frequency_samples;
        let recovery_session = self.recovery_session;

        RoomEngine.run(
            engine_request,
            observer,
            move |request, observer| match recovery_session.as_ref() {
                Some(recovery) => {
                    crate::room_optimization::optimize_room_pipeline_impl_with_recovery(
                        request,
                        &context,
                        observer,
                        frequency_samples,
                        recovery,
                        finalization_diagnostic,
                    )
                }
                None => {
                    crate::room_optimization::optimize_room_pipeline_impl_with_frequency_samples(
                        request,
                        &context,
                        observer,
                        frequency_samples,
                        finalization_diagnostic,
                    )
                }
            },
        )
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
