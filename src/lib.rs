#![doc = include_str!("../README.md")]

pub use autoeq_workflow::qa_println;

// Re-export external crate functionality
pub use math_audio_iir_fir as iir;
pub use math_audio_optimisation as de;

/// CEA2034 (Spinorama) speaker measurement metrics
pub mod cea2034;

// Re-export types from CEA2034 module to ensure type compatibility
pub use cea2034::{
    Curve, DirectivityCurve, DirectivityData, SpinoramaBundle, SpinoramaBundleBuilder,
};

/// Error types for autoeq operations.
pub mod error;
pub use error::{AutoeqError, Result};

/// Common CLI argument definitions shared across binaries
pub mod cli;
/// Constraint functions for optimization
pub mod constraints;
/// FIR filter design and optimization.
pub use autoeq_fir as fir;
/// Smart initial guess generation
pub mod initial_guess;
/// Loss functions for optimization
pub mod loss;
/// Optimization algorithms, objective functions, and shared setup
pub mod optim;
/// Parameter vector utilities for different PEQ models
pub mod param_utils;
/// Plotting and visualization functions (requires the `plotly` feature).
#[cfg(feature = "plotly")]
pub use autoeq_plot as plot;
/// Data reading and parsing functions
pub mod read;
/// Frequency response utilities
pub mod response;
/// Shared workflow steps used by binaries
pub mod workflow;
/// Mapping
pub mod x2peq;

/// Artifact storage abstraction for reports, exports, and sidecars.
pub use autoeq_artifacts::{ArtifactStore, FsArtifactStore, MemoryArtifactStore};

/// Room EQ multi-channel optimization
pub mod roomeq;

/// Extracted RoomEQ contracts and execution/export boundaries.
pub use roomeq_engine;
pub use roomeq_export;
pub use roomeq_model;

// Backward-compatible re-exports for moved modules
pub use optim::callback as optim_callback;
pub use optim::de as optim_de;
pub use optim::mh as optim_mh;
pub use optim::params as optim_params;

// Re-export commonly used items
pub use cli::*;
pub use loss::{CrossoverType, HeadphoneLossData, LossType, SpeakerLossData};
pub use optim::params::{OptimParams, PeqModel};
pub use optim::*;
#[cfg(feature = "plotly")]
pub use plot::*;
pub use read::*;
pub use workflow::*;
pub use workflow::{
    HeadphoneOptResult, OptimizationOutput, ProgressCallbackConfig, ProgressUpdate,
    SpeakerOptResult, VisualizationCurves, compute_visualization_curves, optimize_headphone,
    optimize_speaker, perform_optimization_with_progress,
};
pub use x2peq::x2peq;

// Re-export commonly used roomeq types
pub use roomeq::{
    DspChainOutput, OptimizerConfig, OptimizerConfigBuilder, RecordingConfiguration, RoomConfig,
    RoomConfigBuilder, RoomOptimizationProgress, RoomOptimizationResult, SpeakerConfig,
    optimize_room, optimize_speaker as optimize_room_speaker,
};
