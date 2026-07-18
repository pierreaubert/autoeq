use super::super::auto_tune::{self, AutoOptimizerContext};
use super::super::excursion;
use super::super::types::{OptimizerConfig, RoomConfig};
use super::misc::is_subwoofer_measurement_channel;
use crate::Curve;
use log::{debug, info};

#[allow(clippy::too_many_arguments)]
pub(super) fn build_clamped_optimizer(
    channel_name: &str,
    room_config: &RoomConfig,
    curve_raw: &Curve,
    curve_for_optim: &Curve,
    min_freq: f64,
    max_freq: f64,
    target_tilt_curve: Option<&Curve>,
    broadband_enabled: bool,
) -> OptimizerConfig {
    let is_sub_channel = is_subwoofer_measurement_channel(channel_name, room_config);
    let mut opt = room_config.optimizer.clone();
    if min_freq != room_config.optimizer.min_freq {
        opt.min_freq = min_freq;
    }
    // The workflow has already decoded any SSIR resource into the prepared
    // channel input. Do not pass a filesystem path into engine execution.
    opt.ssir_wav_path = None;

    // For sub channels, clamp the optimizer's UPPER frequency bound to the
    // actual usable bandwidth.
    if is_sub_channel {
        let measured_upper =
            super::super::optimize::detect_sub_passband_3db(curve_raw).map(|(_lo, hi)| hi);
        let crossover_upper = super::super::home_cinema::effective_bass_management(room_config)
            .and_then(|bm| bm.crossover_frequency_hz)
            .map(|xo| 2.0 * xo);
        const SUB_UPPER_FALLBACK_HZ: f64 = 160.0;
        let upper = match (measured_upper, crossover_upper) {
            (Some(m), Some(xo)) => m.max(xo),
            (Some(m), None) => m,
            (None, Some(xo)) => xo,
            (None, None) => SUB_UPPER_FALLBACK_HZ,
        };
        info!(
            "  Sub channel '{}': clamping optimizer upper bound to {:.1} Hz (measured -3dB high={}, 2*crossover={})",
            channel_name,
            upper,
            measured_upper
                .map(|h| format!("{:.1} Hz", h))
                .unwrap_or_else(|| "n/a".to_string()),
            crossover_upper
                .map(|h| format!("{:.1} Hz", h))
                .unwrap_or_else(|| "n/a".to_string()),
        );
        opt.max_freq = opt.max_freq.min(upper);
    }

    if is_sub_channel && let Some(sub_cfg) = &room_config.optimizer.sub_config {
        info!(
            "  Applying sub_config overrides: num_filters={}, max_db={:+.1}, min_db={:+.1}, max_q={:.1}",
            sub_cfg.num_filters, sub_cfg.max_db, sub_cfg.min_db, sub_cfg.max_q,
        );
        opt.num_filters = sub_cfg.num_filters;
        opt.max_db = sub_cfg.max_db;
        opt.min_db = sub_cfg.min_db;
        opt.min_q = sub_cfg.min_q;
        opt.max_q = sub_cfg.max_q;
    }

    if opt.auto_optimizer.as_ref().is_some_and(|auto| auto.enabled) {
        let detected_f3_hz = match excursion::detect_f3_with_config(
            curve_for_optim,
            None,
            opt.excursion_protection.as_ref(),
        ) {
            Ok(f3_result) if f3_result.f3_hz > min_freq && f3_result.f3_hz < max_freq => {
                Some(f3_result.f3_hz)
            }
            Ok(_) => None,
            Err(e) => {
                debug!("  Auto optimizer: F3 detection skipped: {}", e);
                None
            }
        };

        let auto_context = AutoOptimizerContext {
            is_sub_channel,
            effective_min_freq: min_freq,
            effective_max_freq: max_freq,
            detected_f3_hz,
            schroeder_hz: auto_tune::resolved_schroeder_hz(&opt),
            target_tilt_active: target_tilt_curve.is_some(),
            broadband_enabled,
        };
        opt = auto_tune::resolve_auto_optimizer_config(curve_for_optim, &opt, &auto_context);
    }

    opt
}
