//! FIR filter optimization for room correction
//!
//! This module provides high-level FIR correction generation for room EQ,
//! using the core FIR design functions from `math_audio_iir_fir`.

mod generate;
mod per_driver;
#[cfg(test)]
mod tests;

pub use generate::*;
pub use per_driver::*;
pub use roomeq_dsp::fir as apply;
pub use roomeq_dsp::fir::{
    GD_DELAY_KERNEL_HALF, GD_DELAY_MAGNITUDE_TOLERANCE_DB, GD_DELAY_MAX_NORMALIZED_FREQUENCY,
    GdFirDelay, apply_gd_delay_to_fir_coefficients, gd_delay_padding_samples, realize_gd_fir_delay,
};
