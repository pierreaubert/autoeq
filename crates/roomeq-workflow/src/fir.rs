//! Target resolution for engine-owned FIR correction design.

use autoeq_measurements::Curve;
use roomeq_model::{OptimizerConfig, TargetCurveConfig};
use std::error::Error;

fn resolve_fir_target_curve(
    measurement: &Curve,
    config: &OptimizerConfig,
    target_config: Option<&TargetCurveConfig>,
) -> Result<Curve, Box<dyn Error>> {
    let resources = roomeq_engine::eq::EqResources {
        target: crate::prepare_eq_target(target_config)?,
        impulse_response: None,
    };
    Ok(roomeq_engine::fir::prepared_fir_target_curve(
        measurement,
        config,
        &resources,
    ))
}

pub fn generate_fir_correction(
    measurement: &Curve,
    config: &OptimizerConfig,
    target_config: Option<&TargetCurveConfig>,
    sample_rate: f64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let target = resolve_fir_target_curve(measurement, config, target_config)?;
    roomeq_engine::fir::generate_fir_correction_prepared(measurement, config, &target, sample_rate)
}

pub fn generate_fir_correction_with_gd_target(
    measurement: &Curve,
    config: &OptimizerConfig,
    target_config: Option<&TargetCurveConfig>,
    sample_rate: f64,
    gd_target: Option<&roomeq_engine::gd_opt::GdAlignmentTarget>,
    channel_index: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let target = resolve_fir_target_curve(measurement, config, target_config)?;
    roomeq_engine::fir::generate_fir_correction_with_gd_target_prepared(
        measurement,
        config,
        &target,
        sample_rate,
        gd_target,
        channel_index,
    )
}

pub use roomeq_engine::fir::{
    FirPhase, apply_gd_delay_to_fir_coefficients, generate_fir_correction_prepared,
    generate_fir_correction_with_gd_target_prepared,
};

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;

    #[test]
    fn default_target_is_prepared_in_memory() {
        let measurement = Curve {
            freq: Array1::from_vec(vec![20.0, 100.0, 1000.0]),
            spl: Array1::from_vec(vec![70.0, 80.0, 90.0]),
            ..Default::default()
        };
        let config = OptimizerConfig {
            min_freq: 20.0,
            max_freq: 1000.0,
            ..Default::default()
        };
        let target = resolve_fir_target_curve(&measurement, &config, None).unwrap();
        assert_eq!(target.spl.to_vec(), vec![80.0; 3]);
    }
}
