//! In-memory RoomEQ execution and deterministic processing.

#![forbid(unsafe_code)]

pub use autoeq_core::{AutoeqError, Curve};
pub mod error {
    pub use autoeq_core::error::*;
}

/// Measurement-confidence gate for bass-band phase correction.
pub mod bass_phase_confidence;
pub mod config_adapter;
/// Multi-driver crossover optimization and polarity search.
pub mod crossover;
/// Double-bass-array optimization and phase-critical array summation.
pub mod dba;
/// Speaker-excursion protection analysis and high-pass realization.
pub mod excursion;
/// RoomEQ FIR correction design.
pub mod fir;
/// Group-delay optimization and IIR all-pass alignment.
pub mod gd_opt;
/// Role-aware height-channel spectral, phase, and arrival-time alignment.
pub mod height_channel_alignment;
/// Inter-channel tonal matching using broadband spectral correction.
pub mod inter_channel_timbre_matching;
/// Mixed IIR/FIR phase decomposition and excess-phase correction.
pub mod mixed_phase;
/// Multi-seat continuous-listening-area subwoofer optimization.
pub mod multiseat;
/// Multi-subwoofer optimization and all-pass alignment.
pub mod multisub;
pub mod phase_alignment;
/// Prepared pipeline requests, observable events, and the production execution port.
pub mod pipeline;
/// Progress reporting for long-running RoomEQ operations.
pub mod progress;
pub mod report_adapter;
/// Broadband spectral inter-channel response alignment.
pub mod spectral_align;
/// Supporting-source room compensation filter design.
pub mod supporting_source;

pub use pipeline::{
    EngineRequest, PipelineControl, PipelineEvent, PipelineObserver, PipelineStepId,
    PipelineStepStatus, RoomEngine,
};
