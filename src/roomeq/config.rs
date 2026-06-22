//! Configuration validation for room EQ.
//!
//! Performs comprehensive validation of RoomConfig before optimization.

mod misc;
mod optimizer_rules;
#[cfg(test)]
mod tests;
mod validate;
mod validation_result;

pub use validate::*;
pub use validation_result::*;
