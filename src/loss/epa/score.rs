//! EPA (Evaluation, Potency, Activity) composite score for room EQ optimization.
//!
//! Maps psychoacoustic metrics (loudness, sharpness, roughness) onto the three
//! semantic differential dimensions commonly used in sound quality research.
//! The composite preference score provides a single optimization target.

mod compute;
mod db;
mod default;
mod epa;
mod epa_config;
mod misc;
mod temporal;
mod temporal_masking_config;
#[cfg(test)]
mod tests;
mod types;

pub use compute::*;
pub use epa::*;
pub use epa_config::*;
pub use temporal::*;
pub use temporal_masking_config::*;
pub use types::*;
