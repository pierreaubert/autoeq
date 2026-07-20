//! Shared result and artifact-reference types for single-channel processing.

use autoeq_core::{AutoeqError, Curve, Result};
use autoeq_optim::optim::OptimizerRunEvidence;
use math_audio_iir_fir::Biquad;
use roomeq_model::ChannelDspChain;

use crate::channel_target::TargetContext;

pub(crate) fn subtract_target_tilt(curve: &Curve, target: &TargetContext) -> Curve {
    if let Some(tilt_curve) = &target.target_tilt_curve {
        Curve {
            freq: curve.freq.clone(),
            spl: &curve.spl - &tilt_curve.spl,
            phase: curve.phase.clone(),
            ..Curve::default()
        }
    } else {
        curve.clone()
    }
}

/// Logical filename for a workflow-owned convolution sidecar.
///
/// This type deliberately cannot carry a directory or filesystem path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConvolutionSidecarReference {
    filename: String,
}

impl ConvolutionSidecarReference {
    pub fn new(filename: impl Into<String>) -> Result<Self> {
        let filename = filename.into();
        if filename.is_empty()
            || filename == "."
            || filename == ".."
            || filename.contains('/')
            || filename.contains('\\')
        {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!(
                    "convolution sidecar reference must be a logical filename, got '{filename}'"
                ),
            });
        }
        Ok(Self { filename })
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }
}

/// Metadata for coefficients that the workflow must persist after execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedConvolutionSidecar {
    pub reference: ConvolutionSidecarReference,
    /// Whether a write failure invalidates the channel result.
    pub required: bool,
}

/// Complete in-memory result for one processed channel.
pub struct ChannelProcessingResult {
    pub channel: ChannelDspChain,
    pub pre_score: f64,
    pub post_score: f64,
    pub raw_pre_eq_curve: Curve,
    pub raw_post_eq_curve: Curve,
    pub filters: Vec<Biquad>,
    pub mean_spl: f64,
    pub arrival_time_ms: Option<f64>,
    pub fir_coeffs: Option<Vec<f64>>,
    pub convolution_sidecar: Option<GeneratedConvolutionSidecar>,
    pub optimizer_evidence: Vec<OptimizerRunEvidence>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_reference_accepts_filename_but_rejects_paths() {
        assert_eq!(
            ConvolutionSidecarReference::new("left_fir_48000hz.wav")
                .unwrap()
                .filename(),
            "left_fir_48000hz.wav"
        );
        for invalid in [
            "",
            ".",
            "..",
            "/tmp/filter.wav",
            "nested/filter.wav",
            "C:\\filter.wav",
        ] {
            assert!(ConvolutionSidecarReference::new(invalid).is_err());
        }
    }
}
