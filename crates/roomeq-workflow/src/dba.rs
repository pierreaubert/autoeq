//! DBA source loading and engine invocation.

use autoeq_measurements::Curve;
use roomeq_model::{DBAConfig, MeasurementSource, OptimizerConfig};
use std::error::Error;

pub use roomeq_engine::dba::{DbaOptimizationResult, DbaPreparedInput};
pub use roomeq_engine::dba::{
    optimize_dba as optimize_dba_prepared, optimize_dba_detailed as optimize_dba_detailed_prepared,
    sum_array_response as sum_array_response_prepared,
};

fn load_sources(sources: &[MeasurementSource]) -> Result<Vec<Curve>, Box<dyn Error>> {
    sources
        .iter()
        .map(autoeq_measurements::read::load_source)
        .collect::<Result<Vec<_>, _>>()
}

fn prepare_dba(config: &DBAConfig) -> Result<DbaPreparedInput, Box<dyn Error>> {
    Ok(DbaPreparedInput {
        front: load_sources(&config.front)?,
        rear: load_sources(&config.rear)?,
    })
}

pub fn optimize_dba(
    dba_config: &DBAConfig,
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<DbaOptimizationResult, Box<dyn Error>> {
    optimize_dba_detailed(dba_config, config, sample_rate)
}

pub fn optimize_dba_detailed(
    dba_config: &DBAConfig,
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<DbaOptimizationResult, Box<dyn Error>> {
    let input = prepare_dba(dba_config)?;
    roomeq_engine::dba::optimize_dba_detailed(&input, config, sample_rate)
}

pub fn sum_array_response(sources: &[MeasurementSource]) -> Result<Curve, Box<dyn Error>> {
    let curves = load_sources(sources)?;
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
