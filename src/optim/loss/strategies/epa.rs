use crate::loss::epa::score::{EpaConfig, TemporalMaskingMode, epa_flatness, epa_loss_normalized, temporal_masking_penalty};
use crate::optim::loss::{Objective, ObjectiveContext};

/// EPA perceptual objective for [`LossType::Epa`].
#[derive(Debug, Clone)]
pub struct EpaStrategy {
    pub config: EpaConfig,
    pub temporal_masking_modes: Vec<TemporalMaskingMode>,
}

impl Objective for EpaStrategy {
    fn compute(&self, x: &[f64], ctx: &ObjectiveContext) -> f64 {
        let peq_spl = ctx.peq_spl(x);
        let error = &peq_spl - ctx.deviation;
        let error = ctx.smooth_error(error);
        let error = ctx.apply_deadband(&error);

        let flatness = epa_flatness(
            ctx.freqs,
            &error,
            ctx.min_freq,
            ctx.max_freq,
            &self.config,
        );

        let freqs_vec: Vec<f64> = ctx.freqs.iter().copied().collect();
        // corrected SPL = target + deviation (measurement) + peq correction
        let corrected_spl: Vec<f64> = ctx
            .freqs
            .iter()
            .enumerate()
            .map(|(i, _)| ctx.target[i] + ctx.deviation[i] + peq_spl[i])
            .collect();

        let base_loss = epa_loss_normalized(
            &freqs_vec,
            &corrected_spl,
            &self.config,
            flatness,
        );
        let temporal_masking = temporal_masking_penalty(
            &freqs_vec,
            peq_spl.as_slice().unwrap_or(&[]),
            &self.temporal_masking_modes,
            &self.config.temporal_masking,
        );

        base_loss + temporal_masking + ctx.smoothness_penalty(&peq_spl)
    }
}
