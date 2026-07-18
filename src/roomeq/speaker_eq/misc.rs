use super::super::types::{OptimizerConfig, RoomConfig};
use crate::Curve;
use crate::error::{AutoeqError, Result};
use log::info;
use roomeq_engine::PreparedChannelMeasurements;
use roomeq_engine::eq::{self as engine_eq, EqResources};

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn optimize_eq_maybe_multi(
    measurements: &PreparedChannelMeasurements,
    optimization_curve: &Curve,
    optimizer_config: &OptimizerConfig,
    eq_resources: &EqResources,
    sample_rate: f64,
    channel_name: &str,
    callback: Option<crate::optim::OptimProgressCallback>,
    target_tilt_curve: Option<&Curve>,
) -> Result<engine_eq::EqOptimizationResult> {
    use super::super::types::MultiMeasurementStrategy;

    let use_multi = measurements.is_multi_measurement_source()
        && optimizer_config
            .multi_measurement
            .as_ref()
            .is_some_and(|mc| mc.strategy != MultiMeasurementStrategy::Average);

    if use_multi {
        let multi_config = optimizer_config.multi_measurement.as_ref().unwrap();
        let raw_curves = measurements.individual();

        // Apply target tilt to each individual curve (same as single-measurement path).
        // Without this, multi-measurement optimization sees untilted curves while the
        // averaged curve was tilted, causing variance to increase instead of decrease.
        let tilted_curves;
        let curves = if let Some(tilt) = target_tilt_curve {
            tilted_curves = raw_curves
                .iter()
                .map(|c| Curve {
                    freq: c.freq.clone(),
                    spl: &c.spl - &tilt.spl,
                    phase: c.phase.clone(),
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            tilted_curves.as_slice()
        } else {
            raw_curves
        };

        info!(
            "  Multi-measurement optimization ({:?}) with {} curves{}",
            multi_config.strategy,
            curves.len(),
            if target_tilt_curve.is_some() {
                " (tilt applied)"
            } else {
                ""
            },
        );

        if let Some(cb) = callback {
            engine_eq::optimize_channel_eq_multi_with_callback_detailed(
                curves,
                optimizer_config,
                multi_config,
                Some(eq_resources),
                sample_rate,
                cb,
            )
        } else {
            engine_eq::optimize_channel_eq_multi_detailed(
                curves,
                optimizer_config,
                multi_config,
                Some(eq_resources),
                sample_rate,
            )
        }
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!(
                "Multi-measurement EQ optimization failed for channel {}: {}",
                channel_name, e
            ),
        })
    } else {
        if let Some(cb) = callback {
            engine_eq::optimize_channel_eq_with_callback_detailed(
                optimization_curve,
                optimizer_config,
                Some(eq_resources),
                sample_rate,
                cb,
            )
        } else {
            engine_eq::optimize_channel_eq_detailed(
                optimization_curve,
                optimizer_config,
                Some(eq_resources),
                sample_rate,
            )
        }
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!("EQ optimization failed for channel {}: {}", channel_name, e),
        })
    }
}
pub(super) fn is_subwoofer_measurement_channel(
    channel_name: &str,
    room_config: &RoomConfig,
) -> bool {
    super::super::home_cinema::role_for_channel(channel_name).is_sub_or_lfe()
        || room_config
            .system
            .as_ref()
            .and_then(|sys| {
                let subs = sys.subwoofers.as_ref()?;
                let meas_key = sys.speakers.get(channel_name)?;
                Some(subs.mapping.contains_key(meas_key))
            })
            .unwrap_or(false)
}

/// Determine optimization frequency bands for each driver
///
/// Returns a vector of (min_freq, max_freq) tuples for each driver.
/// Bandwidth extends 1 octave beyond the intended crossover region.
pub(in super::super) fn determine_optimization_bands(
    n_drivers: usize,
    room_config: &RoomConfig,
    crossover_config: &super::super::types::CrossoverConfig,
) -> Vec<(f64, f64)> {
    let global_min = room_config.optimizer.min_freq;
    let global_max = room_config.optimizer.max_freq;

    let mut bands = Vec::with_capacity(n_drivers);

    // Determine fixed crossover point estimates. A `frequency_range` is not a
    // fixed point; it is the search range for each crossover.
    let xover_points = if let Some(ref freqs) = crossover_config.frequencies {
        freqs.clone()
    } else if let Some(freq) = crossover_config.frequency {
        vec![freq]
    } else {
        Vec::new() // No info
    };

    // Helper to get safe crossover bounds
    let get_xover_bounds = |idx: usize| -> (f64, f64) {
        if let Some((min, max)) = crossover_config.frequency_range {
            return (min, max);
        }

        if !xover_points.is_empty() && idx < xover_points.len() {
            let f = xover_points[idx];
            return (f, f);
        }

        // Fallback: log-distribute between 80Hz and 3000Hz
        // This is a rough heuristic if no info is present
        (80.0, 3000.0)
    };

    for i in 0..n_drivers {
        let min_f = if i == 0 {
            global_min
        } else {
            // Highpass: 1 octave below crossover
            let (xover_min, _) = get_xover_bounds(i - 1);
            xover_min * 0.5
        };

        let max_f = if i == n_drivers - 1 {
            global_max
        } else {
            // Lowpass: 1 octave above crossover
            let (_, xover_max) = get_xover_bounds(i);
            xover_max * 2.0
        };

        bands.push((min_f.max(global_min), max_f.min(global_max)));
    }

    bands
}
