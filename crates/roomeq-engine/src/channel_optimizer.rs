//! Shared single- and multi-measurement PEQ dispatch for channel modes.

use autoeq_core::{AutoeqError, Curve, Result};
use autoeq_optim::optim::OptimProgressCallback;
use log::info;
use roomeq_model::{MultiMeasurementStrategy, OptimizerConfig};

use crate::PreparedChannelInput;
use crate::eq::{self, EqOptimizationResult, EqResources};

#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_maybe_multi(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    optimization_curve: &Curve,
    optimizer_config: &OptimizerConfig,
    eq_resources: &EqResources,
    sample_rate: f64,
    callback: Option<OptimProgressCallback>,
    target_tilt_curve: Option<&Curve>,
) -> Result<EqOptimizationResult> {
    let measurements = prepared.measurements();
    let use_multi = measurements.is_multi_measurement_source()
        && optimizer_config
            .multi_measurement
            .as_ref()
            .is_some_and(|config| config.strategy != MultiMeasurementStrategy::Average);

    if use_multi {
        let multi_config = optimizer_config
            .multi_measurement
            .as_ref()
            .expect("multi-measurement config checked above");
        let tilted_curves;
        let curves = if let Some(tilt) = target_tilt_curve {
            tilted_curves = measurements
                .individual()
                .iter()
                .map(|curve| Curve {
                    freq: curve.freq.clone(),
                    spl: &curve.spl - &tilt.spl,
                    phase: curve.phase.clone(),
                    ..Curve::default()
                })
                .collect::<Vec<_>>();
            tilted_curves.as_slice()
        } else {
            measurements.individual()
        };
        info!(
            "  Multi-measurement optimization ({:?}) with {} curves{}",
            multi_config.strategy,
            curves.len(),
            if target_tilt_curve.is_some() {
                " (tilt applied)"
            } else {
                ""
            }
        );

        if let Some(callback) = callback {
            eq::optimize_channel_eq_multi_with_callback_detailed(
                curves,
                optimizer_config,
                multi_config,
                Some(eq_resources),
                sample_rate,
                callback,
            )
        } else {
            eq::optimize_channel_eq_multi_detailed(
                curves,
                optimizer_config,
                multi_config,
                Some(eq_resources),
                sample_rate,
            )
        }
        .map_err(|error| AutoeqError::OptimizationFailed {
            message: format!(
                "Multi-measurement EQ optimization failed for channel {channel_name}: {error}"
            ),
        })
    } else {
        if let Some(callback) = callback {
            eq::optimize_channel_eq_with_callback_detailed(
                optimization_curve,
                optimizer_config,
                Some(eq_resources),
                sample_rate,
                callback,
            )
        } else {
            eq::optimize_channel_eq_detailed(
                optimization_curve,
                optimizer_config,
                Some(eq_resources),
                sample_rate,
            )
        }
        .map_err(|error| AutoeqError::OptimizationFailed {
            message: format!("EQ optimization failed for channel {channel_name}: {error}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use roomeq_model::MultiMeasurementConfig;

    use super::*;
    use crate::{PreparedCea2034, PreparedChannelMeasurements};

    fn curve(level: f64) -> Curve {
        Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(500.0), 48),
            spl: Array1::from_elem(48, level),
            ..Curve::default()
        }
    }

    fn optimizer() -> OptimizerConfig {
        OptimizerConfig {
            num_filters: 1,
            max_iter: 10,
            population: 6,
            min_freq: 20.0,
            max_freq: 500.0,
            psychoacoustic: false,
            refine: false,
            ..OptimizerConfig::default()
        }
    }

    fn prepared(curves: Vec<Curve>, multi: bool) -> PreparedChannelInput {
        PreparedChannelInput::new(
            PreparedChannelMeasurements::new(curves[0].clone(), curves, multi),
            None,
            PreparedCea2034::default(),
            EqResources::default(),
        )
    }

    #[test]
    fn single_curve_dispatch_succeeds() {
        let response = curve(80.0);
        let prepared = prepared(vec![response.clone()], false);
        assert!(
            optimize_maybe_multi(
                "left",
                &prepared,
                &response,
                &optimizer(),
                &EqResources::default(),
                48_000.0,
                None,
                None,
            )
            .is_ok()
        );
    }

    #[test]
    fn weighted_multi_measurement_dispatch_succeeds() {
        let response = curve(80.0);
        let prepared = prepared(vec![response.clone(), curve(81.0)], true);
        let optimizer = OptimizerConfig {
            multi_measurement: Some(MultiMeasurementConfig {
                strategy: MultiMeasurementStrategy::WeightedSum,
                ..MultiMeasurementConfig::default()
            }),
            ..optimizer()
        };
        assert!(
            optimize_maybe_multi(
                "left",
                &prepared,
                &response,
                &optimizer,
                &EqResources::default(),
                48_000.0,
                None,
                None,
            )
            .is_ok()
        );
    }
}
