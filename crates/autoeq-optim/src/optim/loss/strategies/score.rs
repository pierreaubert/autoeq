use crate::Curve;
use crate::loss::{
    HeadphoneLossData, SpeakerLossData, flat_loss, headphone_loss, speaker_score_loss,
};
use crate::optim::loss::{Objective, ObjectiveContext};

/// Speaker preference-score objective for [`LossType::SpeakerScore`].
#[derive(Debug, Clone)]
pub struct SpeakerScoreStrategy {
    pub score_data: SpeakerLossData,
}

impl Objective for SpeakerScoreStrategy {
    fn compute(&self, x: &[f64], ctx: &ObjectiveContext) -> f64 {
        let peq_spl = ctx.peq_spl(x);
        let error = &peq_spl - ctx.deviation;
        let s = speaker_score_loss(&self.score_data, ctx.freqs, &peq_spl);
        let error = ctx.apply_deadband(&error);
        let p = flat_loss(ctx.freqs, &error, ctx.min_freq, ctx.max_freq) / 3.0;
        // SpeakerScore fitness: minimize (100 - score + flatness/3 + smoothness)
        100.0 - s + p + ctx.smoothness_penalty(&peq_spl)
    }
}

/// Headphone preference-score objective for [`LossType::HeadphoneScore`].
#[derive(Debug, Clone)]
pub struct HeadphoneScoreStrategy {
    pub data: HeadphoneLossData,
}

impl Objective for HeadphoneScoreStrategy {
    fn compute(&self, x: &[f64], ctx: &ObjectiveContext) -> f64 {
        let peq_spl = ctx.peq_spl(x);
        let error = ctx.deviation - &peq_spl;
        let error = ctx.apply_deadband(&error);
        let error_curve = Curve {
            freq: ctx.freqs.clone(),
            spl: error.clone(),
            phase: None,
            ..Default::default()
        };
        let s = headphone_loss(&error_curve);
        let p = flat_loss(ctx.freqs, &error, ctx.min_freq, ctx.max_freq);
        // HeadphoneScore fitness: minimize (1000 - score + flatness*20 + smoothness)
        1000.0 - s + p * 20.0 + ctx.smoothness_penalty(&peq_spl)
    }
}
