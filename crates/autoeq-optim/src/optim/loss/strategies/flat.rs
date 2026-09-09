use crate::loss::flat_loss;
use crate::optim::loss::{Objective, ObjectiveContext};

/// Flat-response objective for [`LossType::SpeakerFlat`] and
/// [`LossType::HeadphoneFlat`].
#[derive(Debug, Clone, Copy)]
pub struct FlatStrategy;

impl Objective for FlatStrategy {
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
        let base_loss = flat_loss(ctx.freqs, &error, ctx.min_freq, ctx.max_freq);
        Some(base_loss + ctx.smoothness_penalty(peq_spl))
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
    fn realized_response_keeps_signed_error_and_rejects_invalid_response() {
        let frequencies = Array1::from_iter((0..64).map(|i| 20.0 * 1000.0_f64.powf(i as f64 / 63.0)));
        let deviation = Array1::zeros(64);
        let ctx = context(&frequencies, &deviation);
        // A correction alternating +6/-6 dB has RMS 6 dB, not zero.
        // No PEQ fit or parameter reconstruction participates in this oracle.
        let correction = Array1::from_iter((0..64).map(|i| if i % 2 == 0 { 6.0 } else { -6.0 }));
        let score = FlatStrategy.compute_response(&correction, &ctx).unwrap();
        assert!((score - 6.0).abs() < 1e-12);
        assert_eq!(FlatStrategy.compute_response(&Array1::zeros(63), &ctx), Some(f64::INFINITY));
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut broken = correction.clone();
            broken[31] = invalid;
            assert_eq!(FlatStrategy.compute_response(&broken, &ctx), Some(f64::INFINITY));
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
