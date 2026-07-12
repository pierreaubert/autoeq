use crate::loss::epa::score::{
    EpaConfig, TemporalMaskingMode, epa_flatness, epa_loss_normalized, temporal_masking_penalty,
};
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

        let flatness = epa_flatness(ctx.freqs, &error, ctx.min_freq, ctx.max_freq, &self.config);

        let freqs_vec: Vec<f64> = ctx.freqs.iter().copied().collect();
        // deviation = target - measurement, so measurement = target - deviation.
        let corrected_spl: Vec<f64> = ctx
            .freqs
            .iter()
            .enumerate()
            .map(|(i, _)| ctx.target[i] - ctx.deviation[i] + peq_spl[i])
            .collect();

        let base_loss = epa_loss_normalized(&freqs_vec, &corrected_spl, &self.config, flatness);
        let temporal_masking = temporal_masking_penalty(
            &freqs_vec,
            peq_spl.as_slice().unwrap_or(&[]),
            &self.temporal_masking_modes,
            &self.config.temporal_masking,
        );

        base_loss + temporal_masking + ctx.smoothness_penalty(&peq_spl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PeqModel;
    use ndarray::Array1;

    #[test]
    fn audit_epa_uses_measurement_not_mirrored_deviation() {
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
        let corrected: Vec<f64> = target
            .iter()
            .zip(deviation.iter())
            .map(|(&target, &deviation)| target - deviation)
            .collect();
        let expected = epa_loss_normalized(&freqs.to_vec(), &corrected, &config, flatness);
        let actual = strategy.compute(&[], &ctx);

        assert!(
            (actual - expected).abs() < 1e-10,
            "EPA strategy used the wrong corrected SPL: actual={actual}, expected={expected}"
        );
    }
}
