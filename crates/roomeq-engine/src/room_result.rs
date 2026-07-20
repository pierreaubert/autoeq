//! Shared in-memory RoomEQ execution results.

use std::collections::HashMap;

use autoeq_core::Curve;
use autoeq_optim::optim::OptimizerRunEvidence;
use math_audio_iir_fir::Biquad;
use roomeq_model::{ChannelDspChain, DspChainOutput, OptimizationMetadata};

/// Result for a single channel optimization.
#[derive(Debug, Clone)]
pub struct ChannelOptimizationResult {
    pub name: String,
    pub pre_score: f64,
    pub post_score: f64,
    pub initial_curve: Curve,
    pub final_curve: Curve,
    pub biquads: Vec<Biquad>,
    pub fir_coeffs: Option<Vec<f64>>,
    pub optimizer_evidence: Vec<OptimizerRunEvidence>,
}

/// Result for a single speaker optimization.
#[derive(Debug, Clone)]
pub struct SpeakerOptimizationResult {
    pub chain: ChannelDspChain,
    pub pre_score: f64,
    pub post_score: f64,
    pub initial_curve: Curve,
    pub final_curve: Curve,
    pub biquads: Vec<Biquad>,
    pub fir_coeffs: Option<Vec<f64>>,
    pub optimizer_evidence: Vec<OptimizerRunEvidence>,
}

/// Complete in-memory result of a RoomEQ optimization workflow.
#[derive(Debug, Clone)]
pub struct RoomOptimizationResult {
    pub channels: HashMap<String, ChannelDspChain>,
    pub channel_results: HashMap<String, ChannelOptimizationResult>,
    pub combined_pre_score: f64,
    pub combined_post_score: f64,
    pub metadata: OptimizationMetadata,
}

impl RoomOptimizationResult {
    /// Convert the result into the serializable DSP-chain contract.
    pub fn to_dsp_chain_output(&self) -> DspChainOutput {
        crate::output::create_dsp_chain_output(self.channels.clone(), Some(self.metadata.clone()))
    }
}
