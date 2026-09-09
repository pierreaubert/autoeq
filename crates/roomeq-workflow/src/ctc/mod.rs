//! Cross-talk cancellation / binaural transfer-matrix support.

mod dsp_response_cache;
mod fft;
mod load;
mod misc;
#[cfg(test)]
mod tests;
mod types;

pub use dsp_response_cache::{
    apply_channel_dsp_chain_to_curve, apply_channel_dsp_chain_to_curve_with_embedded_irs,
    apply_channel_dsp_chain_to_curve_with_sidecar_dir,
    channel_electrical_response_with_embedded_irs,
};
pub use types::*;
pub(crate) use misc::{checked_sample_rate, read_wav_bytes_channels_f64, read_wav_channels_f64};
