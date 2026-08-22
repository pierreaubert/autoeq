//! Cross-talk cancellation / binaural transfer-matrix support.

mod dsp_response_cache;
mod fft;
mod load;
mod misc;
mod plugin;
#[cfg(test)]
mod tests;
mod types;

pub use dsp_response_cache::{
    apply_channel_dsp_chain_to_curve, apply_channel_dsp_chain_to_curve_with_sidecar_dir,
};
pub use types::*;
