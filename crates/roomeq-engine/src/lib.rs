//! In-memory RoomEQ execution and deterministic processing.

#![forbid(unsafe_code)]

pub use autoeq_core::{
    AutoeqError, Curve, PeqModel, curve_transforms::build_target_curve_by_name,
    smooth_one_over_n_octave,
};
pub use autoeq_optim::de::CallbackAction;
pub use autoeq_optim::optim::{OptimProgressCallback, OptimizerConfidence, OptimizerRunEvidence};
/// Acoustic analysis used by engine and workflow orchestration without adding
/// parallel ownership of the underlying implementations.
pub mod analysis {
    pub use roomeq_analysis::{
        crossover_utils, frequency_grid, ir_waveform, quasi_anechoic, response_metrics, slope,
        time_align,
    };
}
pub mod error {
    pub use autoeq_core::error::*;
}
pub mod loss {
    pub use autoeq_optim::loss::*;
}
/// Runtime correction quality and acceptance contracts used at the engine
/// execution boundary.
pub mod quality {
    pub use roomeq_quality::*;
}
pub mod response {
    pub use autoeq_core::response::*;
}

/// Deterministic bass-management planning, prediction, and joint optimization.
pub mod bass_management;
/// Measurement-confidence gate for bass-band phase correction.
pub mod bass_phase_confidence;
/// Pure CEA-2034 speaker and preference correction.
pub mod cea2034;
/// Complete path-free preparation and execution for one channel.
pub mod channel_execution;
/// Path-free phase-linear, hybrid, and mixed-phase channel processing.
pub mod channel_fir;
/// Path-free low-latency, warped-IIR, and Kautz-modal channel processing.
pub mod channel_iir;
/// Complete path-free input prepared for channel processing.
pub mod channel_input;
/// Prepared, path-free measurement inputs for channel processing.
pub mod channel_measurements;
mod channel_optimizer;
mod channel_preference;
/// Deterministic preprocessing for a prepared channel.
pub mod channel_preprocessing;
/// Shared results and logical sidecar references for channel processing.
pub mod channel_result;
/// Target preparation for one channel.
pub mod channel_target;
pub mod config_adapter;
/// Multi-driver crossover optimization and polarity search.
pub mod crossover;
/// Path-free cross-talk cancellation matrix solving and diagnostics.
pub mod ctc;
/// Double-bass-array optimization and phase-critical array summation.
pub mod dba;
/// DSP convention and numerical-discipline audit checks.
pub mod dsp_conventions;
/// Canonical complex-response evaluation of serialized DSP chains.
pub mod dsp_realization;
/// In-memory per-channel and multi-measurement EQ optimization.
pub mod eq;
/// Speaker-excursion protection analysis and high-pass realization.
pub mod excursion;
/// RoomEQ FIR correction design.
pub mod fir;
/// Group-delay optimization and IIR all-pass alignment.
pub mod gd_opt;
/// Pure group and topology execution helpers.
pub mod group;
/// Prepared group/topology execution and DSP graph construction.
pub mod group_processing;
/// Role-aware height-channel spectral, phase, and arrival-time alignment.
pub mod height_channel_alignment;
/// In-memory home-cinema policy, routing, reporting, and seat analysis.
pub mod home_cinema;
/// Inter-channel tonal matching using broadband spectral correction.
pub mod inter_channel_timbre_matching;
/// Path-free frequency-split FIR/IIR channel processing.
pub mod mixed_crossover;
/// Mixed IIR/FIR phase decomposition and excess-phase correction.
pub mod mixed_phase;
/// Multi-seat continuous-listening-area subwoofer optimization.
pub mod multiseat;
/// Multi-subwoofer optimization and all-pass alignment.
pub mod multisub;
/// Deterministic DSP-chain and response assembly.
pub mod output;
pub mod phase_alignment;
/// Resolve canonical physical routing independently of playback backends.
pub mod physical_routing;
/// Prepared pipeline requests, observable events, and the production execution port.
pub mod pipeline;
/// Progress reporting for long-running RoomEQ operations.
pub mod progress;
/// Filesystem-capable validation of configured measurement provenance.
pub mod provenance;
/// Provisional correction decision records emitted at the decision site (E2).
pub mod provisional_decisions;
/// Cumulative pruning audit and source-summation checks (E3).
pub mod pruning_audit;
pub mod report_adapter;
/// Shared in-memory results returned by RoomEQ execution workflows.
pub mod room_result;
pub mod runtime_limiter;
/// Broadband spectral inter-channel response alignment.
pub mod spectral_align;
pub mod summation_search;
pub mod target_enforcement;
pub use roomeq_analysis::spatial_robustness;
/// Evidence-aware operation gating and local constraint evaluation (E1).
pub mod evidence_gate;
/// Supporting-source room compensation filter design.
pub mod supporting_source;
/// Deterministic topology, crossover, and bass-routing primitives.
pub mod topology;
/// Path-free time-alignment analysis used by workflow preparation.
pub mod time_align {
    pub use roomeq_analysis::time_align::{
        ArrivalTimeResult, ProbeDelayResult, detect_delay_with_probe, find_arrival_time_samples,
    };
}

pub use channel_input::{PreparedCea2034, PreparedChannelInput};
pub use channel_measurements::PreparedChannelMeasurements;
pub use pipeline::{
    EngineRequest, PipelineControl, PipelineEvent, PipelineObserver, PipelineStepId,
    PipelineStepStatus, RoomEngine,
};

/// QA evidence directory for test artifact retention (test builds only).
///
/// Tests that retain JSON evidence for the QA harness write through this
/// directory: `ROOMEQ_QA_DIR` when set, otherwise the workspace `target/qa`
/// directory beside the crate. The override keeps runs hermetic where the
/// shared workspace target is read-only, without changing default
/// artifact locations.
#[cfg(test)]
pub(crate) fn qa_evidence_dir() -> std::path::PathBuf {
    std::env::var_os("ROOMEQ_QA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/qa"))
}
