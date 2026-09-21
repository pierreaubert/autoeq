//! Synthetic speaker curve generation for QA testing.
//!
//! Provides deterministic test scenarios with known ground truth for validating
//! optimization algorithms without relying on real measurement data.

pub use autoeq_core::{AutoeqError, Curve, Result};
pub mod error {
    pub use autoeq_core::error::*;
}

pub mod catalog;
mod generate;
mod misc;
pub mod spatial;
pub mod stimulus;
#[cfg(test)]
mod tests;
pub mod timing;
mod types;

pub use catalog::*;
pub use generate::*;
pub use misc::*;
pub use spatial::*;
pub use stimulus::*;
pub use timing::*;
pub use types::*;
