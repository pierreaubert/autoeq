//! Multi-sub source loading and engine invocation.

use autoeq_measurements::Curve;
use roomeq_model::{MeasurementSource, OptimizerConfig};
use std::error::Error;

pub use roomeq_engine::multisub::{
    MultiSubAllPassResult, MultiSubCombinedResponse, MultiSubOptimizationResult,
};
pub use roomeq_engine::multisub::{
    optimize_multisub as optimize_multisub_prepared,
    optimize_multisub_detailed as optimize_multisub_detailed_prepared,
    optimize_multisub_with_allpass as optimize_multisub_with_allpass_prepared,
};

fn load_measurements(sources: &[MeasurementSource]) -> Result<Vec<Curve>, Box<dyn Error>> {
    sources
        .iter()
        .map(autoeq_measurements::read::load_source)
        .collect::<Result<Vec<_>, _>>()
}

pub fn optimize_multisub_detailed(
    measurements: &[MeasurementSource],
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<MultiSubOptimizationResult, Box<dyn Error>> {
    let curves = load_measurements(measurements)?;
    roomeq_engine::multisub::optimize_multisub_detailed(&curves, config, sample_rate)
}

pub fn optimize_multisub(
    measurements: &[MeasurementSource],
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<MultiSubOptimizationResult, Box<dyn Error>> {
    optimize_multisub_detailed(measurements, config, sample_rate)
}

pub fn optimize_multisub_with_allpass(
    measurements: &[MeasurementSource],
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<MultiSubAllPassResult, Box<dyn Error>> {
    let curves = load_measurements(measurements)?;
    roomeq_engine::multisub::optimize_multisub_with_allpass(&curves, config, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn source_adapter_returns_owned_prepared_curves() {
        let curve = Curve {
            freq: Array1::from_vec(vec![20.0, 100.0]),
            spl: Array1::from_vec(vec![80.0, 81.0]),
            ..Default::default()
        };
        let prepared = load_measurements(&[MeasurementSource::InMemory(curve.clone())]).unwrap();
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].freq, curve.freq);
        assert_eq!(prepared[0].spl, curve.spl);
    }
}
