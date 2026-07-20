//! Stable RoomEQ configuration and output contracts.

pub use autoeq_core::{
    AutoeqError, Curve, InlineMeasurement, MeasurementMultiple, MeasurementRef, MeasurementSingle,
    MeasurementSource, PeqModel, Result, SpinoramaBundle,
};
pub mod error {
    pub use autoeq_core::error::*;
}

pub mod roomeq {
    pub mod types {
        pub use crate::config::*;
    }
    pub mod rir_prototype {
        pub use crate::rir_prototype_config::*;
    }
}

pub mod auto_tune;
pub mod config;
pub mod contracts;
pub mod home_cinema;
pub mod home_cinema_resolution;
pub mod ir_waveform;
pub mod optimizer_settings;
pub mod output;
pub mod preset;
pub mod report_contracts;
pub mod rir_prototype_config;
pub mod target_tilt;
pub mod validation_rules;
pub use config::*;
pub use contracts::{ChannelChain, DspGraph, Plugin};
pub use home_cinema::*;
pub use ir_waveform::IrWaveform;
pub use optimizer_settings::*;
pub use output::*;
pub use preset::*;
pub use report_contracts::*;
