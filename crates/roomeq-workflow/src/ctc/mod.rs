//! Cross-talk cancellation / binaural transfer-matrix support.

mod compute;
mod dsp_response_cache;
mod fft;
mod load;
mod misc;
mod plugin;
#[cfg(test)]
mod tests;
mod types;

pub use dsp_response_cache::apply_channel_dsp_chain_to_curve;
pub use types::*;
