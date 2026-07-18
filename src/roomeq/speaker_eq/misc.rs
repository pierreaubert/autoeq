use super::super::eq;
use super::super::excursion;
use super::super::types::{MeasurementSource, OptimizerConfig, RoomConfig};
use crate::Curve;
use crate::error::{AutoeqError, Result};
use log::{debug, info, warn};
use math_audio_dsp::analysis::compute_average_response;
use math_audio_iir_fir::Biquad;
use roomeq_engine::PreparedChannelMeasurements;

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn optimize_eq_maybe_multi(
    measurements: &PreparedChannelMeasurements,
    optimization_curve: &Curve,
    optimizer_config: &OptimizerConfig,
    target_config: Option<&super::super::types::TargetCurveConfig>,
    sample_rate: f64,
    channel_name: &str,
    callback: Option<crate::optim::OptimProgressCallback>,
    target_tilt_curve: Option<&Curve>,
) -> Result<eq::EqOptimizationResult> {
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
            eq::optimize_channel_eq_multi_with_callback_detailed(
                curves,
                optimizer_config,
                multi_config,
                target_config,
                sample_rate,
                cb,
            )
        } else {
            eq::optimize_channel_eq_multi_detailed(
                curves,
                optimizer_config,
                multi_config,
                target_config,
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
            eq::optimize_channel_eq_with_callback_detailed(
                optimization_curve,
                optimizer_config,
                target_config,
                sample_rate,
                cb,
            )
        } else {
            eq::optimize_channel_eq_detailed(
                optimization_curve,
                optimizer_config,
                target_config,
                sample_rate,
            )
        }
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!("EQ optimization failed for channel {}: {}", channel_name, e),
        })
    }
}

/// Decide whether a broadband pre-correction result should be rejected.
///
/// The shelf fit can be confused by room modes or HPF rolloff, producing
/// a result that is worse than the raw measurement. Rejecting it prevents
/// the optimizer from compounding the error.
pub(super) fn broadband_correction_rejected(pre_bb_score: f64, post_bb_score: f64) -> bool {
    // Tight threshold: anything more than 20 % worse is rejected.
    // The old 1.5× threshold was too permissive — a 40 % worse result
    // would still be accepted, causing audible degradation.
    const MAX_WORSENING_RATIO: f64 = 1.2;
    post_bb_score > pre_bb_score * MAX_WORSENING_RATIO
}

pub(super) fn create_kautz_filter_config(sections: &[(f64, f64, f64)]) -> serde_json::Value {
    let kautz_sections: Vec<serde_json::Value> = sections
        .iter()
        .map(|(pole_freq, q, gain)| {
            serde_json::json!({
                "pole_freq": pole_freq,
                "q": q,
                "gain": gain,
            })
        })
        .collect();
    let (freq, q, _) = sections.first().copied().unwrap_or((100.0, 1.0, 0.0));

    serde_json::json!({
        "topology": "kautz_filter",
        "filter_type": "peak",
        "freq": freq,
        "q": q,
        "db_gain": 0.0,
        "kautz_sections": kautz_sections,
    })
}

pub(super) fn cea2034_correction_active(room_config: &RoomConfig) -> bool {
    room_config
        .optimizer
        .cea2034_correction
        .as_ref()
        .is_some_and(|c| c.enabled)
}

pub(super) fn generate_excursion_filters(
    room_config: &RoomConfig,
    curve: &Curve,
    sample_rate: f64,
) -> Vec<Biquad> {
    let Some(exc_config) = &room_config.optimizer.excursion_protection else {
        return Vec::new();
    };
    if !exc_config.enabled {
        return Vec::new();
    }

    info!("  Applying excursion protection...");
    match excursion::generate_excursion_protection(curve, exc_config, sample_rate) {
        Ok(result) => {
            info!(
                "  Excursion protection: F3={:.1}Hz, HPF={:.1}Hz ({} filters)",
                result.f3_hz,
                result.hpf_frequency,
                result.filters.len()
            );
            result.filters
        }
        Err(e) => {
            warn!(
                "  Excursion protection failed: {}. Continuing without protection.",
                e
            );
            Vec::new()
        }
    }
}

pub(super) fn system_has_subwoofer(room_config: &RoomConfig) -> bool {
    room_config
        .system
        .as_ref()
        .map(|sys| {
            sys.subwoofers
                .as_ref()
                .is_some_and(|s| !s.mapping.is_empty())
        })
        .unwrap_or_else(|| {
            // Legacy: check if any speaker name looks like a sub
            room_config
                .speakers
                .keys()
                .any(|k| k.eq_ignore_ascii_case("lfe") || k.to_lowercase().starts_with("sub"))
        })
}

pub(super) fn maybe_clamp_min_freq_for_target_tilt(
    channel_name: &str,
    room_config: &RoomConfig,
    curve: &Curve,
    target_tilt_curve: Option<&Curve>,
    min_freq: f64,
    max_freq: f64,
) -> f64 {
    if target_tilt_curve.is_some() && system_has_subwoofer(room_config) {
        match excursion::detect_f3_with_config(
            curve,
            None,
            room_config.optimizer.excursion_protection.as_ref(),
        ) {
            Ok(f3_result) => {
                // Only clamp if F3 is above the configured min_freq but still
                // well below max_freq. A very high "F3" (e.g., on a tilted curve
                // with no real rolloff) would invalidate the frequency range.
                if f3_result.f3_hz > min_freq && f3_result.f3_hz < max_freq * 0.5 {
                    info!(
                        "  Tilt active + subwoofer: clamping min_freq from {:.1}Hz to F3={:.1}Hz \
                         to prevent bass over-boost below rolloff",
                        min_freq, f3_result.f3_hz
                    );
                    return f3_result.f3_hz;
                }
            }
            Err(e) => {
                debug!(
                    "  F3 detection failed for tilt clamping: {}. Using configured min_freq.",
                    e
                );
            }
        }
    } else if target_tilt_curve.is_some() {
        debug!(
            "  Tilt active but no subwoofer: skipping F3 min_freq clamping for '{}' (full-range speakers)",
            channel_name
        );
    }

    min_freq
}

pub(super) fn mean_response_in_range(curve: &Curve, min_freq: f64, max_freq: f64) -> f64 {
    let freqs_f32: Vec<f32> = curve.freq.iter().map(|&f| f as f32).collect();
    let spl_f32: Vec<f32> = curve.spl.iter().map(|&s| s as f32).collect();
    compute_average_response(
        &freqs_f32,
        &spl_f32,
        Some((min_freq as f32, max_freq as f32)),
    ) as f64
}

pub(super) fn flatness_score_in_range(curve: &Curve, min_freq: f64, max_freq: f64) -> f64 {
    let mean = mean_response_in_range(curve, min_freq, max_freq);
    let normalized_spl = &curve.spl - mean;
    crate::loss::flat_loss(&curve.freq, &normalized_spl, min_freq, max_freq)
}

pub(super) fn target_mean_spl(
    channel_name: &str,
    channel_mean_spl: f64,
    shared_mean_spl: Option<f64>,
) -> f64 {
    if let Some(shared) = shared_mean_spl {
        debug!(
            "  Using shared target level {:.1} dB (channel mean was {:.1} dB, delta {:.1} dB)",
            shared,
            channel_mean_spl,
            shared - channel_mean_spl
        );
        shared
    } else {
        debug!(
            "  Using channel '{}' target level {:.1} dB",
            channel_name, channel_mean_spl
        );
        channel_mean_spl
    }
}

pub(super) fn existing_ssir_wav_path(source: &MeasurementSource) -> Option<std::path::PathBuf> {
    source.wav_path().and_then(|wav_path| {
        let path = std::path::PathBuf::from(wav_path);
        if path.exists() { Some(path) } else { None }
    })
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
