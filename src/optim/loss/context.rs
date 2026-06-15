use crate::PeqModel;
use crate::optim::compute::compute_smoothness_penalty;
use crate::optim::misc::{apply_audibility_deadband, maybe_smooth_error};
use crate::x2peq::x2spl;
use ndarray::Array1;

/// Shared evaluation context passed to every [`super::Objective`] strategy.
///
/// This struct borrows data from [`crate::optim::ObjectiveData`] so that the
/// strategy implementations do not need to know about the full god object.
#[derive(Debug, Clone, Copy)]
pub struct ObjectiveContext<'a> {
    /// Frequency grid for evaluation.
    pub freqs: &'a Array1<f64>,
    /// Target SPL curve.
    pub target: &'a Array1<f64>,
    /// Deviation curve (`target - input`).
    pub deviation: &'a Array1<f64>,
    /// Sample rate in Hz.
    pub srate: f64,
    /// PEQ parameter layout.
    pub peq_model: PeqModel,
    /// Minimum frequency for evaluation.
    pub min_freq: f64,
    /// Maximum frequency for evaluation.
    pub max_freq: f64,
    /// Whether to apply 1/N octave smoothing to the error curve.
    pub smooth: bool,
    /// Smoothing resolution as 1/N octave.
    pub smooth_n: usize,
    /// Optional JND deadband applied to residual errors.
    pub audibility_deadband: Option<&'a crate::roomeq::AudibilityDeadbandConfig>,
    /// Optional TV² smoothness regularizer on the correction curve.
    pub smoothness_penalty: Option<&'a crate::optim::SmoothnessPenaltyConfig>,
}

impl ObjectiveContext<'_> {
    /// Compute the cascaded PEQ magnitude response for `x` on the context grid.
    pub fn peq_spl(&self, x: &[f64]) -> Array1<f64> {
        x2spl(self.freqs, x, self.srate, self.peq_model)
    }

    /// Optionally smooth an error curve using the context smoothing settings.
    pub fn smooth_error(&self, error: Array1<f64>) -> Array1<f64> {
        maybe_smooth_error(self.freqs, error, self.smooth, self.smooth_n)
    }

    /// Apply the configured audibility deadband to an error curve.
    pub fn apply_deadband(&self, error: &Array1<f64>) -> Array1<f64> {
        apply_audibility_deadband(
            self.freqs,
            error,
            self.min_freq,
            self.max_freq,
            self.audibility_deadband,
        )
    }

    /// Compute the smoothness penalty for a PEQ magnitude response.
    pub fn smoothness_penalty(&self, peq_spl: &Array1<f64>) -> f64 {
        self.smoothness_penalty
            .map(|cfg| {
                compute_smoothness_penalty(
                    peq_spl,
                    self.freqs,
                    self.min_freq,
                    self.max_freq,
                    cfg,
                )
            })
            .unwrap_or(0.0)
    }
}
