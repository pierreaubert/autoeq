//! Multi-seat variance optimization for subwoofer systems
//!
//! Optimizes subwoofer delays and gains to minimize response variance
//! across multiple listening positions (MSO - Multi-Subwoofer Optimizer logic).

mod average;
mod compute;
mod consts;
mod extension;
mod interpolate;
mod mean;
mod misc;
mod modal;
mod modal_basis;
mod mso;
mod mso_objective_context;
mod mso_search_options;
mod multi_seat_measurements;
mod objective;
mod optimize;
mod primary;
mod simple_rng;
#[cfg(test)]
mod tests;
mod types;

pub use compute::*;
pub use multi_seat_measurements::*;
pub use optimize::*;
pub use types::*;
