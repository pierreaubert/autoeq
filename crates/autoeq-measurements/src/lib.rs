//! Measurement sources, loading, preprocessing, and speaker metrics.

#![allow(unsafe_code)]

pub use autoeq_core::{AutoeqError, Curve, Result};

pub mod error {
    pub use autoeq_core::error::*;
}

pub mod cea2034;
pub mod mic_phase_calibration;
pub mod provenance;
pub mod quality;
pub mod read;

pub use cea2034::*;
pub use mic_phase_calibration::{MicPhaseCalibration, load_mic_phase_calibration};
pub use provenance::*;
pub use quality::*;
pub use read::*;
