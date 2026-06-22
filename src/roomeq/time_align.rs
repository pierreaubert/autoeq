//! Time alignment utilities for speaker measurements
//!
//! This module provides functions to analyze WAV files and determine arrival times
//! for time-aligning multiple speakers in a room EQ setup.

mod detect;
mod error;
mod estimate;
mod find;
mod misc;
#[cfg(test)]
mod tests;
mod types;

pub use detect::*;
pub use error::*;
pub use estimate::*;
pub use find::*;
pub use misc::*;
pub use types::*;
