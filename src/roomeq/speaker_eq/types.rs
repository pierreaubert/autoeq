use super::super::types::{ChannelDspChain as PublicChannelDspChain, RoomConfig};
use crate::Curve;
use math_audio_iir_fir::Biquad;
use roomeq_engine::PreparedChannelInput;
use std::path::Path;

pub(in crate::roomeq) use roomeq_engine::channel_preprocessing::PreprocessedFeatures;
pub(in crate::roomeq) use roomeq_engine::channel_target::TargetContext;

pub(in super::super) type MixedModeResult = (
    PublicChannelDspChain,
    f64,
    f64,
    Curve,
    Curve,
    Vec<Biquad>,
    f64,
    Option<f64>,
    Option<Vec<f64>>,
    Vec<crate::optim::OptimizerRunEvidence>,
);

pub(in crate::roomeq) struct ChannelOptimizationInput<'a> {
    pub channel_name: &'a str,
    pub prepared: &'a PreparedChannelInput,
    pub room_config: &'a RoomConfig,
    pub sample_rate: f64,
    pub output_dir: &'a Path,
    pub callback: Option<crate::optim::OptimProgressCallback>,
    pub shared_mean_spl: Option<f64>,
}

pub(in crate::roomeq) struct PreparedMeasurement {
    pub curve: Curve,
    pub curve_raw: Curve,
    pub arrival_time_ms: Option<f64>,
}
