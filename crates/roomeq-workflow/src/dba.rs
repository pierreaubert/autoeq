//! DBA source loading and engine invocation.

use crate::measurement::load_source_with_frequency_samples;
use autoeq_measurements::Curve;
use roomeq_model::{DBAConfig, MeasurementSource, OptimizerConfig};
use std::error::Error;

pub use roomeq_engine::dba::{DbaOptimizationResult, DbaPreparedInput};
pub use roomeq_engine::dba::{
    optimize_dba as optimize_dba_prepared, optimize_dba_detailed as optimize_dba_detailed_prepared,
    sum_array_response as sum_array_response_prepared,
};

fn load_sources_with_frequency_samples(
    sources: &[MeasurementSource],
    frequency_samples: usize,
) -> Result<Vec<Curve>, Box<dyn Error>> {
    Ok(sources
        .iter()
        .map(|source| load_source_with_frequency_samples(source, frequency_samples))
        .collect::<Result<Vec<_>, _>>()?)
}

pub(crate) fn prepare_dba_with_frequency_samples(
    config: &DBAConfig,
    frequency_samples: usize,
) -> Result<DbaPreparedInput, Box<dyn Error>> {
    Ok(DbaPreparedInput {
        front: load_sources_with_frequency_samples(&config.front, frequency_samples)?,
        rear: load_sources_with_frequency_samples(&config.rear, frequency_samples)?,
    })
}

pub fn optimize_dba(
    dba_config: &DBAConfig,
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<DbaOptimizationResult, Box<dyn Error>> {
    optimize_dba_with_frequency_samples(
        dba_config,
        config,
        sample_rate,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

pub fn optimize_dba_with_frequency_samples(
    dba_config: &DBAConfig,
    config: &OptimizerConfig,
    sample_rate: f64,
    frequency_samples: usize,
) -> Result<DbaOptimizationResult, Box<dyn Error>> {
    optimize_dba_detailed_with_frequency_samples(dba_config, config, sample_rate, frequency_samples)
}

pub fn optimize_dba_detailed(
    dba_config: &DBAConfig,
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<DbaOptimizationResult, Box<dyn Error>> {
    optimize_dba_detailed_with_frequency_samples(
        dba_config,
        config,
        sample_rate,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

pub fn optimize_dba_detailed_with_frequency_samples(
    dba_config: &DBAConfig,
    config: &OptimizerConfig,
    sample_rate: f64,
    frequency_samples: usize,
) -> Result<DbaOptimizationResult, Box<dyn Error>> {
    let input = prepare_dba_with_frequency_samples(dba_config, frequency_samples)?;
    roomeq_engine::dba::optimize_dba_detailed(&input, config, sample_rate)
}

pub fn sum_array_response(sources: &[MeasurementSource]) -> Result<Curve, Box<dyn Error>> {
    sum_array_response_with_frequency_samples(sources, crate::DEFAULT_FREQUENCY_SAMPLES)
}

pub fn sum_array_response_with_frequency_samples(
    sources: &[MeasurementSource],
    frequency_samples: usize,
) -> Result<Curve, Box<dyn Error>> {
    let curves = load_sources_with_frequency_samples(sources, frequency_samples)?;
    roomeq_engine::dba::sum_array_response(&curves)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn source_adapter_prepares_dba_curves_before_engine_call() {
        let curve = Curve {
            freq: Array1::from_vec(vec![50.0, 100.0]),
            spl: Array1::from_vec(vec![80.0, 80.0]),
            phase: Some(Array1::zeros(2)),
            ..Default::default()
        };
        let summed = sum_array_response(&[MeasurementSource::InMemory(curve)]).unwrap();
        assert_eq!(summed.spl.to_vec(), vec![80.0, 80.0]);
    }
}
