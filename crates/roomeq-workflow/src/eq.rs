//! Filesystem-backed adapters for the in-memory RoomEQ equalization engine.

use std::error::Error;

use roomeq_engine::Curve;
use roomeq_engine::eq::{self as engine_eq, EqResources};
use roomeq_model::{MultiMeasurementConfig, OptimizerConfig, TargetCurveConfig};

pub use roomeq_engine::eq::{
    EqOptimizationResult, MultiEqAutoOptimizerContext,
    resolve_multi_measurement_auto_optimizer_config,
};

fn prepare_resources(
    config: &OptimizerConfig,
    target: Option<&TargetCurveConfig>,
) -> Result<EqResources, Box<dyn Error>> {
    crate::prepare_eq_resources(config, target)
}

pub fn optimize_channel_eq_detailed(
    curve: &Curve,
    config: &OptimizerConfig,
    target: Option<&TargetCurveConfig>,
    sample_rate: f64,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    let resources = prepare_resources(config, target)?;
    engine_eq::optimize_channel_eq_detailed(curve, config, Some(&resources), sample_rate)
}

pub fn optimize_channel_eq_with_callback_detailed(
    curve: &Curve,
    config: &OptimizerConfig,
    target: Option<&TargetCurveConfig>,
    sample_rate: f64,
    callback: roomeq_engine::OptimProgressCallback,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    let resources = prepare_resources(config, target)?;
    engine_eq::optimize_channel_eq_with_callback_detailed(
        curve,
        config,
        Some(&resources),
        sample_rate,
        callback,
    )
}

pub fn optimize_channel_eq_multi_with_auto_optimizer_detailed(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    target: Option<&TargetCurveConfig>,
    sample_rate: f64,
    auto_context: MultiEqAutoOptimizerContext,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    let resources = prepare_resources(config, target)?;
    engine_eq::optimize_channel_eq_multi_with_auto_optimizer_detailed(
        curves,
        config,
        multi_config,
        Some(&resources),
        sample_rate,
        auto_context,
    )
}
