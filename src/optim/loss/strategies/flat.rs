use crate::loss::flat_loss;
use crate::optim::loss::{Objective, ObjectiveContext};

/// Flat-response objective for [`LossType::SpeakerFlat`] and
/// [`LossType::HeadphoneFlat`].
#[derive(Debug, Clone, Copy)]
pub struct FlatStrategy;

impl Objective for FlatStrategy {
    fn compute(&self, x: &[f64], ctx: &ObjectiveContext) -> f64 {
        let peq_spl = ctx.peq_spl(x);
        let error = &peq_spl - ctx.deviation;
        let error = ctx.smooth_error(error);
        let error = ctx.apply_deadband(&error);
        let base_loss = flat_loss(ctx.freqs, &error, ctx.min_freq, ctx.max_freq);
        base_loss + ctx.smoothness_penalty(&peq_spl)
    }
}
