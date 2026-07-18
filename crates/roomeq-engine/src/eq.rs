//! EQ optimization for individual channels
//!
//! Provides per-channel PEQ optimization using autoeq's workflow.

mod consts;
mod misc;
mod multi_eq_auto_optimizer_context;
mod optimize;
mod prepared_single_channel_eq;
mod representative;
pub mod resources;
#[cfg(test)]
mod tests;
mod types;

pub use multi_eq_auto_optimizer_context::*;
pub use optimize::*;
pub use resources::{EqResources, PreparedEqTarget, PreparedImpulseResponse};
