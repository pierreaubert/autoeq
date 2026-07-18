use super::super::types::{
    ChannelDspChain as PublicChannelDspChain, CurveData, PluginConfigWrapper, RoomConfig,
};
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

/// Decomposed DSP-chain parts for a single channel.
///
/// This is an intermediate representation used while assembling the final
/// [`PublicChannelDspChain`]. It separates plugins into the logical passes
/// of the 3-pass pipeline (pre-EQ, EQ, post-EQ) so that ordering and
/// labelling can be tested independently of the final serialization wrapper.
#[allow(dead_code)]
pub(in crate::roomeq) struct ChannelDspChain {
    /// Plugins applied before the main room-EQ pass (CEA2034 correction,
    /// broadband shelf alignment, etc.).
    pub pre_eq_plugins: Vec<PluginConfigWrapper>,
    /// Plugins that make up the main room-EQ pass.
    pub eq_plugins: Vec<PluginConfigWrapper>,
    /// Plugins applied after the main EQ pass (user preference shelves,
    /// excess-phase FIR convolution, etc.).
    pub post_eq_plugins: Vec<PluginConfigWrapper>,
    /// Final ordered plugin list: pre-EQ, then EQ, then post-EQ.
    pub plugin_order: Vec<PluginConfigWrapper>,
    /// Delay plugins for this channel (empty for simple single-speaker paths).
    pub delays: Vec<PluginConfigWrapper>,
    /// Gain plugins for this channel (empty for simple single-speaker paths).
    pub gains: Vec<PluginConfigWrapper>,
    /// IIR filters used alongside the FIR/phase optimizer output.
    pub filters: Vec<Biquad>,
}

/// Assembled report data for a single channel.
///
/// This bundles the curves, scores and metadata that [`process_single_speaker`]
/// ultimately returns in its [`MixedModeResult`] tuple.
pub(in crate::roomeq) struct ChannelReport {
    pub channel_name: String,
    pub pre_score: f64,
    pub post_score: f64,
    /// Initial curve in its raw measurement coordinates (returned as the
    /// fourth element of [`MixedModeResult`]).
    pub raw_pre_eq_curve: Curve,
    /// Final corrected curve in its raw coordinates (returned as the fifth
    /// element of [`MixedModeResult`]).
    pub raw_post_eq_curve: Curve,
    /// Initial curve extended to the full display range (20 Hz – 20 kHz).
    pub pre_eq_curve: Curve,
    /// Final corrected curve extended to the full display range.
    pub post_eq_curve: Curve,
    /// EQ correction magnitude response (final − initial).
    pub eq_curve: CurveData,
    /// Effective target curve in absolute SPL coordinates, if any.
    pub target_curve: Option<CurveData>,
    /// Filters returned for the caller/report (the EQ filters for IIR modes,
    /// or the approximate biquads for KautzModal).
    pub filters: Vec<Biquad>,
    pub mean_spl: f64,
    pub arrival_time_ms: Option<f64>,
}

/// Mode-specific optimizer result used by the DSP-chain and report assembly
/// stages.
pub(in crate::roomeq) enum OptimizerOutput {
    /// Phase-linear FIR correction.
    PhaseLinear {
        coeffs: Vec<f64>,
        wav_filename: String,
    },
    /// Sequential hybrid: IIR correction followed by FIR on the residual.
    Hybrid {
        eq_filters: Vec<Biquad>,
        coeffs: Vec<f64>,
        wav_filename: String,
    },
    /// Mixed-phase: IIR minimum-phase EQ plus optional excess-phase FIR.
    MixedPhase {
        eq_filters: Vec<Biquad>,
        fir_coeffs: Option<Vec<f64>>,
        fir_filename: Option<String>,
        report: Option<crate::roomeq::mixed_phase::MixedPhaseCorrectionReport>,
    },
}

#[allow(dead_code)]
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
