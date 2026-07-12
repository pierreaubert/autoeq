use crate::loss::{AsymmetricLossConfig, weighted_mse_asymmetric};
use crate::optim::loss::{Objective, ObjectiveContext};
use ndarray::Array1;

/// Asymmetric flat objective for [`LossType::SpeakerFlatAsymmetric`].
#[derive(Debug, Clone)]
pub struct AsymmetricStrategy {
    pub config: AsymmetricLossConfig,
    pub null_suppression: Option<Array1<f64>>,
}

impl Objective for AsymmetricStrategy {
    fn compute(&self, x: &[f64], ctx: &ObjectiveContext) -> f64 {
        let peq_spl = ctx.peq_spl(x);
        let error = &peq_spl - ctx.deviation;
        let error = ctx.smooth_error(error);
        let error = ctx.apply_deadband(&error);
        let base_loss = weighted_mse_asymmetric(
            ctx.freqs,
            &error,
            ctx.min_freq,
            ctx.max_freq,
            &self.config,
            self.null_suppression.as_ref(),
        );
        base_loss + ctx.smoothness_penalty(&peq_spl)
    }
}
