//! RoomEQ measurement-analysis primitives.
//!
//! This module provides functions to analyze WAV files and determine arrival times
//! for time-aligning multiple speakers in a room EQ setup.

pub use autoeq_core::{AutoeqError, Curve, Result};
pub mod error {
    pub use autoeq_core::error::{AutoeqError, Result};
}

pub mod crossover_utils;
pub mod eligibility;
pub mod excess_phase;
pub mod quasi_anechoic;

pub mod evidence;
pub mod frequency_grid;
pub mod impulse_analysis;
pub mod ir_waveform;
pub mod listening_area;
pub mod observations;
#[cfg(test)]
mod psycho_fast;
pub mod reflection_cancel;
pub mod response_metrics;
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

    pub use detect::{detect_delay_with_probe, detect_delays_multi_channel};
    pub use error::PhaseArrivalError;
    pub use estimate::{estimate_arrival_from_phase, estimate_arrival_from_phase_detailed};
    pub use find::{
        find_arrival_time, find_arrival_time_samples, find_arrival_time_with_reference,
    };
    pub use misc::{calculate_alignment_delays, phase_arrival_regression_band};
    pub use types::{ArrivalTimeResult, ProbeDelayResult};
}
