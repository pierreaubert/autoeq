//! Prepared resources for in-memory EQ execution.

use crate::Curve;

/// A target resolved by the workflow before engine execution.
#[derive(Debug, Clone)]
pub enum PreparedEqTarget {
    /// Historical named target generated deterministically on the input grid.
    Predefined(String),
    /// Curve loaded by the workflow from an external resource.
    Curve(Box<Curve>),
}

/// A decoded mono impulse response supplied by the workflow.
#[derive(Debug, Clone)]
pub struct PreparedImpulseResponse {
    pub samples: Vec<f32>,
    pub sample_rate: f64,
}

/// Optional resources used by the EQ engine.
#[derive(Debug, Clone, Default)]
pub struct EqResources {
    pub target: Option<PreparedEqTarget>,
    pub impulse_response: Option<PreparedImpulseResponse>,
}

pub(super) fn target_curve(normalized_curve: &Curve, resources: Option<&EqResources>) -> Curve {
    match resources.and_then(|resources| resources.target.as_ref()) {
        Some(PreparedEqTarget::Curve(target)) => {
            autoeq_core::normalize_and_interpolate_response(&normalized_curve.freq, target)
        }
        Some(PreparedEqTarget::Predefined(name)) => {
            autoeq_core::build_target_curve_by_name(name, &normalized_curve.freq, normalized_curve)
        }
        None => Curve {
            freq: normalized_curve.freq.clone(),
            spl: ndarray::Array1::zeros(normalized_curve.freq.len()),
            phase: None,
            ..Default::default()
        },
    }
}

pub(super) fn analyze_ssir(
    resources: Option<&EqResources>,
) -> Option<(math_rir::SsirResult, &[f32], f64)> {
    let impulse = resources?.impulse_response.as_ref()?;
    if impulse.samples.is_empty()
        || !impulse.sample_rate.is_finite()
        || impulse.sample_rate <= 0.0
        || impulse.samples.len() < (0.010 * impulse.sample_rate) as usize
    {
        return None;
    }

    let config = math_rir::SsirConfig::new(impulse.sample_rate);
    let result = math_rir::analyze_rir(&impulse.samples, &config);
    (result.num_events() >= 1).then_some((result, impulse.samples.as_slice(), impulse.sample_rate))
}
