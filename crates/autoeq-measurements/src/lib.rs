//! Measurement sources, loading, preprocessing, and speaker metrics.

pub use autoeq_core::{AutoeqError, Curve, Result, build_target_curve_by_name};

pub mod error {
    pub use autoeq_core::error::*;
}

pub mod cea2034;
pub mod direct_sound;
pub mod evidence;
pub mod matrix;
pub mod mic_phase_calibration;
pub mod provenance;
pub mod quality;
pub mod read;
pub mod timing;

pub use cea2034::*;
pub use direct_sound::*;
pub use evidence::*;
pub use matrix::*;
pub use mic_phase_calibration::{MicPhaseCalibration, load_mic_phase_calibration};
pub use provenance::*;
pub use quality::*;
pub use read::*;
pub use timing::*;

#[cfg(test)]
mod psycho_fast;
