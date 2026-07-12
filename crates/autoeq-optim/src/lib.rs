//! Objective functions and optimization backends for AutoEQ.

#![allow(unsafe_code)]

pub use autoeq_core as core;
pub use autoeq_core::iir;
pub use autoeq_core::{AutoeqError, Curve, PeqModel, Result};
pub use autoeq_measurements as measurements;

pub mod error {
    pub use autoeq_core::error::*;
}
pub mod read {
    pub use autoeq_measurements::read::*;
}
pub mod cea2034 {
    pub use autoeq_measurements::cea2034::*;
}
pub mod param_utils {
    pub use autoeq_core::param_utils::*;
}
pub mod x2peq {
    pub use autoeq_core::x2peq::*;
}
pub mod roomeq {
    pub use crate::roomeq_types::{AudibilityDeadbandConfig, MultiMeasurementStrategy};
    pub mod phase_utils {
        pub use autoeq_core::phase_utils::*;
    }
}

pub mod constraints;
pub mod initial_guess;
pub mod loss;
pub mod cli;
pub mod optim;
pub mod problem;
pub mod roomeq_types;
pub mod penalty_mode;
pub mod smoothness_penalty_config;

pub use loss::{CrossoverType, HeadphoneLossData, LossType, SpeakerLossData};
pub use problem::{ObjectiveEvaluator, OptimizationProblem, OptimizationResult};
pub use penalty_mode::PenaltyMode;
pub use smoothness_penalty_config::SmoothnessPenaltyConfig;
pub use math_audio_optimisation as de;
pub use optim::params::OptimParams;
pub use optim::{MultiObjectiveData, ObjectiveData, ObjectiveDataBuilder};

/// Adapter implemented by higher-level RoomEQ configuration crates.
pub trait RoomOptimizerConfig {
    fn to_optim_params(&self) -> OptimParams;
}

#[macro_export]
macro_rules! qa_println {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        log::debug!($fmt $(, $arg)*);
    };
}
