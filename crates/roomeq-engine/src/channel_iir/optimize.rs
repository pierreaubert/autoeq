use autoeq_core::{Curve, Result};
use autoeq_optim::optim::{OptimProgressCallback, OptimizerRunEvidence};
use log::info;
use math_audio_iir_fir::Biquad;
use roomeq_model::OptimizerConfig;

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
    if optimizer_config.num_filters == 0 {
        info!("  Skipping PEQ optimization because num_filters is 0");
        return Ok((Vec::new(), Vec::new()));
    }

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

    let result = crate::channel_optimizer::optimize_maybe_multi(
        channel_name,
        prepared,
        optimization_curve,
        optimizer_config,
        eq_resources,
        sample_rate,
        callback,
        target_tilt_curve,
    )?;
    Ok((result.filters, result.optimizer_evidence))
}
