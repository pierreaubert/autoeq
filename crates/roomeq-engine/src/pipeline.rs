//! Prepared RoomEQ pipeline contracts and execution boundary.
//!
//! The engine owns the in-memory request and observable event vocabulary. The
//! workflow layer owns resource loading and artifact destinations, while the
//! supplied kernel owns the processing stages that have not yet migrated into
//! this crate.

use std::collections::HashMap;

use autoeq_core::Result;
use roomeq_model::RoomConfig;

/// Prepared in-memory data required to execute a RoomEQ pipeline.
#[derive(Clone, Copy)]
pub struct EngineRequest<'a> {
    /// Complete, path-resolved room configuration.
    pub config: &'a RoomConfig,
    /// Sample rate used for filter design.
    pub sample_rate: f64,
    /// Optional per-channel probe-based arrival times in milliseconds.
    pub probe_arrival_overrides: Option<&'a HashMap<String, f64>>,
}

/// Production RoomEQ execution boundary.
///
/// RoomEngine deliberately knows nothing about filesystems, caches, or
/// artifact stores. The workflow prepares those resources and supplies a
/// processing kernel. As vertical slices migrate, that kernel becomes smaller
/// until execution is wholly engine-owned.
#[derive(Debug, Default, Clone, Copy)]
pub struct RoomEngine;

impl RoomEngine {
    /// Execute a prepared request through the engine-owned boundary.
    pub fn run<'a, T, F>(
        &self,
        request: EngineRequest<'a>,
        observer: Option<Box<dyn PipelineObserver>>,
        kernel: F,
    ) -> Result<T>
    where
        F: FnOnce(EngineRequest<'a>, Option<Box<dyn PipelineObserver>>) -> Result<T>,
    {
        kernel(request, observer)
    }
}

/// Stable identifier for a RoomEQ pipeline step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStepId {
    /// Clone and normalize request configuration, including CEA2034 prefetch.
    ConfigPreparation,
    /// Validate the prepared configuration.
    Validation,
    /// Decide whether to use a topology-specific workflow or the generic path.
    TopologyRouteSelection,
    /// Execute a topology-specific workflow.
    TopologyWorkflowExecution,
    /// Optimize channels via the generic per-channel path.
    GenericChannelOptimization,
    /// Generate full FIR coefficients after IIR-only stages.
    FirGeneration,
    /// Generate short mixed-phase FIR coefficients.
    MixedPhaseFirGeneration,
    /// Apply standalone phase correction.
    PhaseCorrection,
    /// Align channels in time from measured or phase-estimated arrivals.
    TimeAlignment,
    /// Match broad spectral balance across channels.
    SpectralAlignment,
    /// Apply inter-channel timbre matching.
    InterChannelTimbreMatching,
    /// Align overhead channels to role-appropriate bed references.
    HeightChannelAlignment,
    /// Optimize sub/main phase alignment.
    PhaseAlignment,
    /// Run group-delay optimization.
    GroupDelayOptimization,
    /// Compute pre/post impulse responses.
    ImpulseResponseComputation,
    /// Analyze and optionally correct inter-channel deviation.
    ChannelMatching,
    /// Refresh metadata and derived reports.
    MetadataRefresh,
    /// Check final result invariants.
    SanityCheck,
}

impl PipelineStepId {
    /// Canonical execution order for pipeline steps.
    pub const ALL: &'static [PipelineStepId] = &[
        PipelineStepId::ConfigPreparation,
        PipelineStepId::Validation,
        PipelineStepId::TopologyRouteSelection,
        PipelineStepId::TopologyWorkflowExecution,
        PipelineStepId::GenericChannelOptimization,
        PipelineStepId::FirGeneration,
        PipelineStepId::MixedPhaseFirGeneration,
        PipelineStepId::PhaseCorrection,
        PipelineStepId::TimeAlignment,
        PipelineStepId::SpectralAlignment,
        PipelineStepId::InterChannelTimbreMatching,
        PipelineStepId::HeightChannelAlignment,
        PipelineStepId::PhaseAlignment,
        PipelineStepId::GroupDelayOptimization,
        PipelineStepId::ImpulseResponseComputation,
        PipelineStepId::ChannelMatching,
        PipelineStepId::MetadataRefresh,
        PipelineStepId::SanityCheck,
    ];

    /// Short human-readable label suitable for progress UIs.
    pub fn label(&self) -> &'static str {
        match self {
            PipelineStepId::ConfigPreparation => "Config",
            PipelineStepId::Validation => "Validate",
            PipelineStepId::TopologyRouteSelection => "Route",
            PipelineStepId::TopologyWorkflowExecution => "Topology",
            PipelineStepId::GenericChannelOptimization => "Channels",
            PipelineStepId::FirGeneration => "FIR",
            PipelineStepId::MixedPhaseFirGeneration => "Mixed-Phase FIR",
            PipelineStepId::PhaseCorrection => "Phase Corr.",
            PipelineStepId::TimeAlignment => "Time Align",
            PipelineStepId::SpectralAlignment => "Spectral Align",
            PipelineStepId::InterChannelTimbreMatching => "Timbre Match",
            PipelineStepId::HeightChannelAlignment => "Height Align",
            PipelineStepId::PhaseAlignment => "Phase Align",
            PipelineStepId::GroupDelayOptimization => "GD-Opt",
            PipelineStepId::ImpulseResponseComputation => "IR",
            PipelineStepId::ChannelMatching => "Match",
            PipelineStepId::MetadataRefresh => "Metadata",
            PipelineStepId::SanityCheck => "Sanity",
        }
    }
}

/// Lifecycle status for a pipeline step event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStepStatus {
    /// The step is starting.
    Started,
    /// The step produced an intermediate progress update.
    InProgress,
    /// The step completed and may have changed the result.
    Completed,
    /// The step was intentionally skipped.
    Skipped,
}

/// Observer decision after receiving a pipeline event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineControl {
    /// Continue the pipeline.
    Continue,
    /// Stop the pipeline as soon as possible.
    Stop,
}

/// Structured event emitted by the RoomEQ pipeline.
#[derive(Debug, Clone)]
pub struct PipelineEvent {
    /// Stable step identifier.
    pub step_id: PipelineStepId,
    /// Lifecycle status for this event.
    pub status: PipelineStepStatus,
    /// Current channel, if the event is channel-specific.
    pub channel: Option<String>,
    /// Current channel index, if available.
    pub channel_index: Option<usize>,
    /// Total channels or units in the current stage, if available.
    pub total_channels: Option<usize>,
    /// Current optimizer iteration, if available.
    pub iteration: Option<usize>,
    /// Maximum optimizer iterations, if available.
    pub max_iterations: Option<usize>,
    /// Current loss value, if available.
    pub loss: Option<f64>,
    /// Overall pipeline progress in the range 0.0..=1.0.
    pub overall_progress: f64,
    /// Optional display/log message.
    pub message: Option<String>,
    /// EPA preference score, computed periodically by the optimizer.
    pub epa_preference: Option<f64>,
}

impl PipelineEvent {
    /// Create a new event with default optional fields.
    pub fn new(step_id: PipelineStepId, status: PipelineStepStatus) -> Self {
        Self {
            step_id,
            status,
            channel: None,
            channel_index: None,
            total_channels: None,
            iteration: None,
            max_iterations: None,
            loss: None,
            overall_progress: 0.0,
            message: None,
            epa_preference: None,
        }
    }

    /// Convenience constructor for a started event.
    pub fn started(step_id: PipelineStepId, message: impl Into<String>) -> Self {
        Self::new(step_id, PipelineStepStatus::Started).with_message(message)
    }

    /// Convenience constructor for a completed event.
    pub fn completed(step_id: PipelineStepId, message: impl Into<String>) -> Self {
        Self::new(step_id, PipelineStepStatus::Completed).with_message(message)
    }

    /// Convenience constructor for a skipped event.
    pub fn skipped(step_id: PipelineStepId, message: impl Into<String>) -> Self {
        Self::new(step_id, PipelineStepStatus::Skipped).with_message(message)
    }

    /// Attach a message.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Attach a channel name.
    pub fn with_channel(mut self, channel: impl Into<String>) -> Self {
        self.channel = Some(channel.into());
        self
    }

    /// Attach channel indexing.
    pub fn with_channels(mut self, channel_index: usize, total_channels: usize) -> Self {
        self.channel_index = Some(channel_index);
        self.total_channels = Some(total_channels);
        self
    }

    /// Attach optimizer iteration progress.
    pub fn with_iteration(mut self, iteration: usize, max_iterations: usize) -> Self {
        self.iteration = Some(iteration);
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Attach a loss value.
    pub fn with_loss(mut self, loss: f64) -> Self {
        self.loss = Some(loss);
        self
    }

    /// Attach overall pipeline progress.
    pub fn with_overall_progress(mut self, overall_progress: f64) -> Self {
        self.overall_progress = overall_progress;
        self
    }

    /// Attach EPA preference progress.
    pub fn with_epa_preference(mut self, epa_preference: Option<f64>) -> Self {
        self.epa_preference = epa_preference;
        self
    }
}

/// Observer for structured pipeline events.
pub trait PipelineObserver: Send {
    /// Called for every pipeline event.
    fn on_event(&mut self, event: &PipelineEvent) -> PipelineControl;
}

impl<F> PipelineObserver for F
where
    F: FnMut(&PipelineEvent) -> PipelineControl + Send,
{
    fn on_event(&mut self, event: &PipelineEvent) -> PipelineControl {
        self(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_step_id_all_and_label() {
        assert!(!PipelineStepId::ALL.is_empty());
        for step in PipelineStepId::ALL {
            assert!(!step.label().is_empty());
        }
        assert_eq!(PipelineStepId::ConfigPreparation.label(), "Config");
        assert_eq!(PipelineStepId::Validation.label(), "Validate");
    }

    #[test]
    fn pipeline_event_builders_preserve_progress_data() {
        let event = PipelineEvent::started(PipelineStepId::GenericChannelOptimization, "begin")
            .with_channel("left")
            .with_channels(0, 2)
            .with_iteration(5, 20)
            .with_loss(0.42)
            .with_overall_progress(0.5)
            .with_epa_preference(Some(0.7));

        assert_eq!(event.status, PipelineStepStatus::Started);
        assert_eq!(event.channel.as_deref(), Some("left"));
        assert_eq!(event.channel_index, Some(0));
        assert_eq!(event.total_channels, Some(2));
        assert_eq!(event.iteration, Some(5));
        assert_eq!(event.max_iterations, Some(20));
        assert_eq!(event.loss, Some(0.42));
        assert!((event.overall_progress - 0.5).abs() < f64::EPSILON);
        assert_eq!(event.epa_preference, Some(0.7));
    }

    #[test]
    fn observer_closure_receives_events() {
        let mut received = 0;
        let mut observer = |_: &PipelineEvent| -> PipelineControl {
            received += 1;
            PipelineControl::Continue
        };
        let event = PipelineEvent::new(PipelineStepId::Validation, PipelineStepStatus::Started);
        assert_eq!(observer.on_event(&event), PipelineControl::Continue);
        assert_eq!(received, 1);
    }

    #[test]
    fn room_engine_dispatches_a_real_prepared_run() {
        let config = RoomConfig::default();
        let request = EngineRequest {
            config: &config,
            sample_rate: 48_000.0,
            probe_arrival_overrides: None,
        };

        let graph = RoomEngine
            .run(request, None, |request, _observer| {
                let mut graph = roomeq_model::DspGraph::new("1");
                graph.add_channel(format!("{}-hz", request.sample_rate as usize), Vec::new());
                Ok(graph)
            })
            .expect("prepared engine run");

        assert!(graph.channels.contains_key("48000-hz"));
    }
}
