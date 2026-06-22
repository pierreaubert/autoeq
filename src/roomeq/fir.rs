//! FIR filter optimization for room correction
//!
//! This module provides high-level FIR correction generation for room EQ,
//! using the core FIR design functions from `math_audio_iir_fir`.

mod apply;
mod generate;
#[cfg(test)]
mod tests;

pub(crate) use apply::*;
pub use generate::*;
