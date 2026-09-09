use crate::loss::{AsymmetricLossConfig, weighted_mse_asymmetric};
use crate::optim::loss::{Objective, ObjectiveContext};
use ndarray::Array1;
use std::sync::Arc;

/// Asymmetric flat objective for [`LossType::SpeakerFlatAsymmetric`].
#[derive(Debug, Clone)]
pub struct AsymmetricStrategy {
    pub config: AsymmetricLossConfig,
    pub null_suppression: Option<Arc<Array1<f64>>>,
}

impl Objective for AsymmetricStrategy {
    fn compute(&self, x: &[f64], ctx: &ObjectiveContext) -> f64 {
        self.compute_response(&ctx.peq_spl(x), ctx).expect("scalar response objective")
    }

    fn compute_response(&self, peq_spl: &ndarray::Array1<f64>, ctx: &ObjectiveContext) -> Option<f64> {
        if peq_spl.len() != ctx.freqs.len() || peq_spl.len() != ctx.deviation.len()
            || peq_spl.iter().any(|value| !value.is_finite()) {
            return Some(f64::INFINITY);
        }
        let error = peq_spl - ctx.deviation;
        let error = ctx.apply_deadband(&error);
        let base_loss = weighted_mse_asymmetric(
            ctx.freqs,
            &error,
            ctx.min_freq,
            ctx.max_freq,
            &self.config,
            self.null_suppression.as_deref(),
        );
        Some(base_loss + ctx.smoothness_penalty(peq_spl))
    }
}
