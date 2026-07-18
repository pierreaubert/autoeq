//! Temporary compatibility adapters for the crate-owned EQ engine.
//!
//! Root RoomEQ orchestration still passes filesystem-backed configuration.
//! Resolve those resources in the workflow crate, then call the in-memory
//! engine. This module should disappear when the remaining orchestration moves
//! to `roomeq-workflow`.

use std::collections::HashMap;
use std::error::Error;

use math_audio_iir_fir::Biquad;
use roomeq_engine::eq::{self as engine_eq, EqResources};
use roomeq_model::{MultiMeasurementConfig, OptimizerConfig, TargetCurveConfig};

use crate::Curve;

pub(crate) use roomeq_engine::eq::{
    EqOptimizationResult, MultiEqAutoOptimizerContext,
    resolve_multi_measurement_auto_optimizer_config,
};

fn prepare_resources(
    config: &OptimizerConfig,
    target: Option<&TargetCurveConfig>,
) -> Result<EqResources, Box<dyn Error>> {
    roomeq_workflow::prepare_eq_resources(config, target)
}

pub fn optimize_channel_eq(
    curve: &Curve,
    config: &OptimizerConfig,
    target: Option<&TargetCurveConfig>,
    sample_rate: f64,
) -> Result<(Vec<Biquad>, f64), Box<dyn Error>> {
    let resources = prepare_resources(config, target)?;
    engine_eq::optimize_channel_eq(curve, config, Some(&resources), sample_rate)
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

pub fn optimize_channel_eq_with_spin_detailed(
    curve: &Curve,
    spin_data: &HashMap<String, Curve>,
    config: &OptimizerConfig,
    target: Option<&TargetCurveConfig>,
    sample_rate: f64,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    let resources = prepare_resources(config, target)?;
    engine_eq::optimize_channel_eq_with_spin_detailed(
        curve,
        spin_data,
        config,
        Some(&resources),
        sample_rate,
    )
}

pub fn optimize_channel_eq_with_callback_detailed(
    curve: &Curve,
    config: &OptimizerConfig,
    target: Option<&TargetCurveConfig>,
    sample_rate: f64,
    callback: autoeq_optim::optim::OptimProgressCallback,
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

pub fn optimize_channel_eq_multi_detailed(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    target: Option<&TargetCurveConfig>,
    sample_rate: f64,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    let resources = prepare_resources(config, target)?;
    engine_eq::optimize_channel_eq_multi_detailed(
        curves,
        config,
        multi_config,
        Some(&resources),
        sample_rate,
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

pub fn optimize_channel_eq_multi_with_callback_detailed(
    curves: &[Curve],
    config: &OptimizerConfig,
    multi_config: &MultiMeasurementConfig,
    target: Option<&TargetCurveConfig>,
    sample_rate: f64,
    callback: autoeq_optim::optim::OptimProgressCallback,
) -> Result<EqOptimizationResult, Box<dyn Error>> {
    let resources = prepare_resources(config, target)?;
    engine_eq::optimize_channel_eq_multi_with_callback_detailed(
        curves,
        config,
        multi_config,
        Some(&resources),
        sample_rate,
        callback,
    )
}
