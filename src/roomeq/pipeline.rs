//! Observable Room EQ optimization pipeline.
//!
//! This module exposes a structured event stream for callers that need more
//! detail than the legacy progress callback while keeping the optimization
//! implementation in `optimize`.

use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;

use super::optimize::RoomOptimizationResult;
use super::types::RoomConfig;

/// Request data for a Room EQ pipeline run.
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

/// Observable Room EQ optimization pipeline.
pub struct RoomPipeline<'a> {
    request: RoomPipelineRequest<'a>,
    validation_measurements: HashMap<String, Vec<crate::Curve>>,
}

impl<'a> RoomPipeline<'a> {
    /// Create a new pipeline for the given request.
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
        validation_measurements: HashMap<String, Vec<crate::Curve>>,
    ) -> Self {
        self.validation_measurements = validation_measurements;
        self
    }

    /// Run the pipeline, optionally notifying an observer for each event.
    pub fn run(
        self,
        observer: Option<Box<dyn PipelineObserver>>,
    ) -> Result<RoomOptimizationResult> {
        let mut result = super::optimize::optimize_room_pipeline_impl(self.request, observer)?;
        if !self.validation_measurements.is_empty() {
            attach_validation_scorecard(
                &mut result,
                &self.validation_measurements,
                self.request.sample_rate,
            )?;
        }
        Ok(result)
    }
}

fn attach_validation_scorecard(
    result: &mut RoomOptimizationResult,
    validation: &HashMap<String, Vec<crate::Curve>>,
    sample_rate: f64,
) -> Result<()> {
    use super::acoustic_qa::{
        CorrectionAcceptancePolicy, QualityEvaluationConfig, TemporalChannelEvidence,
        derive_temporal_quality_evidence, evaluate_acoustic_quality,
        evaluate_correction_acceptance,
    };

    let mut names: Vec<_> = result.channel_results.keys().cloned().collect();
    names.sort();
    let training_pre: Vec<_> = names
        .iter()
        .map(|name| result.channel_results[name].initial_curve.clone())
        .collect();
    let training_post: Vec<_> = names
        .iter()
        .map(|name| result.channel_results[name].final_curve.clone())
        .collect();
    let mut held_out_pre = Vec::new();
    let mut held_out_post = Vec::new();
    for name in &names {
        let Some(curves) = validation.get(name) else {
            continue;
        };
        let chain =
            result
                .channels
                .get(name)
                .ok_or_else(|| crate::AutoeqError::InvalidConfiguration {
                    message: format!("validation channel '{name}' is absent from pipeline output"),
                })?;
        for curve in curves {
            held_out_pre.push(curve.clone());
            held_out_post.push(super::ctc::apply_channel_dsp_chain_to_curve(
                chain,
                curve,
                sample_rate,
            )?);
        }
    }
    if held_out_pre.is_empty() {
        return Ok(());
    }
    let min_freq_hz = training_pre
        .iter()
        .chain(&training_post)
        .chain(&held_out_pre)
        .chain(&held_out_post)
        .map(|curve| curve.freq[0])
        .fold(0.0_f64, f64::max);
    let max_freq_hz = training_pre
        .iter()
        .chain(&training_post)
        .chain(&held_out_pre)
        .chain(&held_out_post)
        .filter_map(|curve| curve.freq.last().copied())
        .fold(f64::INFINITY, f64::min);
    let temporal_channels: Vec<_> = names
        .iter()
        .map(|name| {
            let masking = result.channels[name].fir_temporal_masking.as_ref();
            TemporalChannelEvidence {
                pre_ringing_audible_db: masking.map(|metrics| metrics.pre_ringing_audible_db),
                main_time_ms: masking.map(|metrics| metrics.main_time_ms),
                fir_taps: result.channel_results[name]
                    .fir_coeffs
                    .as_ref()
                    .map(Vec::len),
            }
        })
        .collect();
    let temporal = derive_temporal_quality_evidence(
        &temporal_channels,
        &training_pre,
        &training_post,
        sample_rate,
    );
    let scorecard = evaluate_acoustic_quality(
        &training_pre,
        &training_post,
        &held_out_pre,
        &held_out_post,
        None,
        QualityEvaluationConfig {
            min_freq_hz,
            max_freq_hz,
            schroeder_hz: None,
            normalize_level: true,
        },
        temporal,
    )
    .map_err(|message| crate::AutoeqError::InvalidConfiguration { message })?;

    if result.metadata.correction_acceptance.is_none() {
        let first = &names[0];
        let channel = &result.channel_results[first];
        let mean = channel.initial_curve.spl.mean().unwrap_or(0.0);
        let mut target = channel.initial_curve.clone();
        target.spl.fill(mean);
        target.phase = None;
        result.metadata.correction_acceptance = Some(
            evaluate_correction_acceptance(
                &channel.initial_curve,
                &channel.final_curve,
                &target,
                None,
                CorrectionAcceptancePolicy::RuntimeSafety,
            )
            .map_err(|message| crate::AutoeqError::InvalidConfiguration { message })?,
        );
    }
    if let Some(report) = &mut result.metadata.correction_acceptance {
        report.acoustic_quality = Some(scorecard);
    }
    Ok(())
}

/// Stable identifier for a Room EQ pipeline step.
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
    /// Canonical execution order of every pipeline step. Step indicators
    /// in UIs render this list left-to-right; later steps are not all
    /// always visited (TopologyWorkflowExecution vs
    /// GenericChannelOptimization, MixedPhaseFirGeneration, etc.) and
    /// will appear as `Skipped` in the live event stream.
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

    /// Short, human-readable label for status indicators. Matches the
    /// log lines emitted by the pipeline so users can correlate the UI
    /// step strip with the log scroll buffer.
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

/// Structured event emitted by the Room EQ pipeline.
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
    use crate::roomeq::types::{OptimizerConfig, RoomConfig};

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
    fn pipeline_event_constructors() {
        let started = PipelineEvent::started(PipelineStepId::ConfigPreparation, "begin");
        assert_eq!(started.step_id, PipelineStepId::ConfigPreparation);
        assert_eq!(started.status, PipelineStepStatus::Started);
        assert_eq!(started.message, Some("begin".to_string()));

        let completed = PipelineEvent::completed(PipelineStepId::FirGeneration, "done");
        assert_eq!(completed.status, PipelineStepStatus::Completed);

        let skipped = PipelineEvent::skipped(PipelineStepId::PhaseCorrection, "skip");
        assert_eq!(skipped.status, PipelineStepStatus::Skipped);
    }

    #[test]
    fn pipeline_event_with_methods() {
        let event = PipelineEvent::new(
            PipelineStepId::GenericChannelOptimization,
            PipelineStepStatus::InProgress,
        )
        .with_channel("left")
        .with_channels(0, 2)
        .with_iteration(5, 20)
        .with_loss(0.42)
        .with_overall_progress(0.5)
        .with_epa_preference(Some(0.7));

        assert_eq!(event.channel, Some("left".to_string()));
        assert_eq!(event.channel_index, Some(0));
        assert_eq!(event.total_channels, Some(2));
        assert_eq!(event.iteration, Some(5));
        assert_eq!(event.max_iterations, Some(20));
        assert_eq!(event.loss, Some(0.42));
        assert!((event.overall_progress - 0.5).abs() < 1e-9);
        assert_eq!(event.epa_preference, Some(0.7));
    }

    #[test]
    fn pipeline_observer_closure() {
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
    fn room_pipeline_new_and_run_error_branch() {
        let config = RoomConfig {
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: std::collections::HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: OptimizerConfig::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        };
        let request = RoomPipelineRequest {
            config: &config,
            sample_rate: 48000.0,
            output_dir: None,
            probe_arrival_overrides: None,
        };
        let pipeline = RoomPipeline::new(request);
        let result = pipeline.run(None);
        // Empty config should fail at validation/loading.
        assert!(result.is_err(), "empty config should produce an error");
    }

    #[test]
    fn runtime_validation_measurements_populate_held_out_scorecard() {
        let mut result = crate::roomeq::test_fixtures::single_channel_room_result("left");
        let validation_curve = result.channel_results["left"].initial_curve.clone();
        let validation = HashMap::from([("left".to_string(), vec![validation_curve])]);

        attach_validation_scorecard(&mut result, &validation, 48_000.0)
            .expect("runtime validation");

        let quality = result
            .metadata
            .correction_acceptance
            .as_ref()
            .and_then(|report| report.acoustic_quality.as_ref())
            .expect("quality scorecard");
        assert_eq!(quality.held_out.as_ref().unwrap().curve_count, 1);
    }
}
