use autoeq_core::{AutoeqError, Curve, Result};
use autoeq_optim::optim::{OptimProgressCallback, OptimizerRunEvidence};
use log::info;
use math_audio_iir_fir::Biquad;
use roomeq_model::{MultiMeasurementStrategy, OptimizerConfig};

use crate::PreparedChannelInput;
use crate::eq::{self, EqResources};

#[allow(clippy::too_many_arguments)]
pub(super) fn optimize_iir_eq(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    optimization_curve: &Curve,
    optimizer_config: &OptimizerConfig,
    eq_resources: &EqResources,
    sample_rate: f64,
    callback: Option<OptimProgressCallback>,
    target_tilt_curve: Option<&Curve>,
) -> Result<(Vec<Biquad>, Vec<OptimizerRunEvidence>)> {
    if let Some(schroeder_config) = optimizer_config
        .schroeder_split
        .as_ref()
        .filter(|config| config.enabled)
    {
        let schroeder_frequency = schroeder_config
            .room_dimensions
            .as_ref()
            .map(|dimensions| {
                let frequency = dimensions.schroeder_frequency();
                info!(
                    "  Schroeder split: calculated frequency {:.1} Hz from room dimensions",
                    frequency
                );
                frequency
            })
            .unwrap_or(schroeder_config.schroeder_freq);
        info!(
            "  Schroeder split: optimizing below {:.1} Hz with max_q={:.1}, above with max_q={:.1}",
            schroeder_frequency,
            schroeder_config.low_freq_config.max_q,
            schroeder_config.high_freq_config.max_q
        );

        let result = eq::optimize_with_schroeder_split_detailed(
            optimization_curve,
            optimizer_config,
            schroeder_config,
            sample_rate,
        )?;
        let low_filter_count = result.low_filters.len();
        let high_filter_count = result.high_filters.len();
        let mut filters = result.low_filters;
        filters.extend(result.high_filters);
        info!(
            "  Schroeder split: {} low-freq filters + {} high-freq filters",
            low_filter_count, high_filter_count
        );
        return Ok((filters, result.optimizer_evidence));
    }

    optimize_maybe_multi(
        channel_name,
        prepared,
        optimization_curve,
        optimizer_config,
        eq_resources,
        sample_rate,
        callback,
        target_tilt_curve,
    )
}

#[allow(clippy::too_many_arguments)]
fn optimize_maybe_multi(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    optimization_curve: &Curve,
    optimizer_config: &OptimizerConfig,
    eq_resources: &EqResources,
    sample_rate: f64,
    callback: Option<OptimProgressCallback>,
    target_tilt_curve: Option<&Curve>,
) -> Result<(Vec<Biquad>, Vec<OptimizerRunEvidence>)> {
    let measurements = prepared.measurements();
    let use_multi = measurements.is_multi_measurement_source()
        && optimizer_config
            .multi_measurement
            .as_ref()
            .is_some_and(|config| config.strategy != MultiMeasurementStrategy::Average);

    let result = if use_multi {
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
        })?
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
        })?
    };

    Ok((result.filters, result.optimizer_evidence))
}
