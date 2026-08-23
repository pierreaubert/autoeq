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
        let error = ctx.apply_deadband(&error);
        let base_loss = flat_loss(ctx.freqs, &error, ctx.min_freq, ctx.max_freq);
        base_loss + ctx.smoothness_penalty(&peq_spl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PeqModel;
    use ndarray::Array1;

    fn context<'a>(
        frequencies: &'a Array1<f64>,
        deviation: &'a Array1<f64>,
    ) -> ObjectiveContext<'a> {
        ObjectiveContext {
            freqs: frequencies,
            target: deviation,
            deviation,
            srate: 48_000.0,
            peq_model: PeqModel::Pk,
            min_freq: 20.0,
            max_freq: 20_000.0,
            smooth: true,
            smooth_n: 1,
            audibility_deadband: None,
            smoothness_penalty: None,
        }
    }

    #[test]
    fn alternating_signed_residual_cannot_cancel_before_loss() {
        let frequencies =
            Array1::from_iter((0..64).map(|index| 20.0 * 1000.0_f64.powf(index as f64 / 63.0)));
        let alternating =
            Array1::from_iter((0..64).map(|index| if index % 2 == 0 { 6.0 } else { -6.0 }));
        let constant = Array1::from_elem(64, 6.0);

        let alternating_loss = FlatStrategy.compute(&[], &context(&frequencies, &alternating));
        let constant_loss = FlatStrategy.compute(&[], &context(&frequencies, &constant));
        assert!((alternating_loss - constant_loss).abs() < 1e-12);
        assert!((alternating_loss - 6.0).abs() < 1e-12);
    }
}
