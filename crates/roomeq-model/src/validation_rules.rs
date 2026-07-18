//! Configuration validation for room EQ.
//!
//! Performs comprehensive validation of RoomConfig before optimization.

mod misc;
mod optimizer_rules;
#[cfg(test)]
mod tests;
mod validate;
pub mod validation_result;

pub use misc::collect_sources;
pub use validate::*;
pub use validation_result::*;
