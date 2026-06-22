//! Synthetic speaker curve generation for QA testing.
//!
//! Provides deterministic test scenarios with known ground truth for validating
//! optimization algorithms without relying on real measurement data.

mod generate;
mod misc;
#[cfg(test)]
mod tests;
mod types;

pub use generate::*;
pub use misc::*;
pub use types::*;
