//! Experimental EPA (Evaluation, Potency, Activity) transfer-response diagnostics.
//!
//! Maps transfer-response shape metrics onto semantic-differential dimensions
//! for reporting. These descriptors are not programme-audio perceptual
//! predictions and do not steer the active EPA spectral-flatness optimizer.

mod compute;
mod db;
mod default;
mod epa;
mod epa_config;
mod level_matching;
mod misc;
mod temporal;
mod temporal_masking_config;
#[cfg(test)]
mod tests;
mod types;

pub use compute::*;
pub use epa::*;
pub use epa_config::*;
#[doc(inline)]
pub use level_matching::reference_band_energy_db;
pub use temporal::*;
pub use temporal_masking_config::*;
pub use types::*;
