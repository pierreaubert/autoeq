//! RoomEQ measurement-analysis primitives.
//!
//! This module provides functions to analyze WAV files and determine arrival times
//! for time-aligning multiple speakers in a room EQ setup.

pub use autoeq_core::{AutoeqError, Curve, Result};
pub mod error {
    pub use autoeq_core::error::*;
}

pub mod crossover_utils;
pub mod frequency_grid;
pub mod impulse_analysis;
pub mod ir_waveform;
pub mod listening_area;
pub mod reflection_cancel;
pub mod rir_prototype;
pub mod slope;
pub mod spatial_robustness;
pub mod temporal_targets;
pub use roomeq_model::IrWaveform;

pub mod time_align {
    //! Time alignment utilities for speaker measurements.

    mod detect;
    mod error;
    mod estimate;
    mod find;
    mod misc;
    #[cfg(test)]
    mod tests;
    mod types;

    pub use detect::*;
    pub use error::*;
    pub use estimate::*;
    pub use find::*;
    pub use misc::*;
    pub use types::*;
}
