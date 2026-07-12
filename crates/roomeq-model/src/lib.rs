//! Stable RoomEQ configuration and output contracts.

#![allow(unsafe_code)]

pub use autoeq_core::{AutoeqError, Curve, Result};
pub mod error { pub use autoeq_core::error::*; }
pub mod read { pub use autoeq_measurements::read::*; }
pub use autoeq_measurements::{MeasurementMultiple, MeasurementRef, MeasurementSingle, MeasurementSource};
pub mod loss { pub use autoeq_optim::loss::*; }
pub mod optim { pub use autoeq_optim::optim::*; }
pub use autoeq_optim::{OptimParams, PeqModel};

pub mod roomeq {
    pub mod types {
        pub use crate::config::*;
    }
    pub mod rir_prototype {
        pub use crate::rir_prototype_config::*;
    }
}

pub mod config;
pub mod preset;
pub mod rir_prototype_config;
pub mod contracts;
pub use contracts::{ChannelChain, DspGraph, Plugin};
mod optim_adapter;
pub use config::*;
pub use preset::*;
