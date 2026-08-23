use crate::loss::epa::score::{EpaConfig, TemporalMaskingMode, epa_flatness};
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
        let error = ctx.apply_deadband(&error);

        let flatness = epa_flatness(ctx.freqs, &error, ctx.min_freq, ctx.max_freq, &self.config);

        let _ = &self.temporal_masking_modes;
        flatness + ctx.smoothness_penalty(&peq_spl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PeqModel;
    use ndarray::Array1;

    #[test]
    fn audit_epa_optimizer_uses_spectral_flatness_only() {
        let freqs = Array1::<f64>::linspace(20.0, 20_000.0, 128);
        let target = Array1::from_elem(freqs.len(), 75.0);
        let deviation = freqs.mapv(|f| 8.0 * (f / 20_000.0).sqrt());
        let config = EpaConfig::default();
        let strategy = EpaStrategy {
            config: config.clone(),
            temporal_masking_modes: Vec::new(),
        };
        let ctx = ObjectiveContext {
            freqs: &freqs,
            target: &target,
            deviation: &deviation,
            srate: 48_000.0,
            peq_model: PeqModel::Pk,
            min_freq: 20.0,
            max_freq: 20_000.0,
            smooth: false,
            smooth_n: 3,
            audibility_deadband: None,
            smoothness_penalty: None,
        };

        let peq_spl = Array1::zeros(freqs.len());
        let error = &peq_spl - &deviation;
        let flatness = epa_flatness(&freqs, &error, 20.0, 20_000.0, &config);
        let expected = flatness;
        let actual = strategy.compute(&[], &ctx);

        assert!(
            (actual - expected).abs() < 1e-10,
            "EPA optimizer must use only spectral flatness: actual={actual}, expected={expected}"
        );
    }
}
