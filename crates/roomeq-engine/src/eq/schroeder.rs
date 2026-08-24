use super::optimize_channel_eq_detailed;
use crate::{AutoeqError, Curve};
use autoeq_core::{Result, response};
use log::debug;
use math_audio_iir_fir::Biquad;
use roomeq_model::{LowFreqFilterConfig, OptimizerConfig, SchroederSplitConfig, TargetShape};

/// Optimized low- and high-frequency filters for a Schroeder split.
#[derive(Debug)]
pub struct SchroederOptimizationResult {
    pub low_filters: Vec<Biquad>,
    pub high_filters: Vec<Biquad>,
    pub optimizer_evidence: Vec<autoeq_optim::optim::OptimizerRunEvidence>,
}

/// Optimize EQ with Schroeder frequency split
///
/// Performs two-pass optimization with different Q constraints:
/// - Below Schroeder: high-Q narrow filters for room modes
/// - Above Schroeder: low-Q broad filters for tonal adjustment
pub fn optimize_with_schroeder_split_detailed(
    curve: &Curve,
    optimizer: &OptimizerConfig,
    schroeder_config: &SchroederSplitConfig,
    sample_rate: f64,
) -> Result<SchroederOptimizationResult> {
    if optimizer.num_filters < 2 {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "Schroeder split requires at least 2 filters (one per band), got {}",
                optimizer.num_filters
            ),
        });
    }

    let schroeder_freq = if let Some(ref dims) = schroeder_config.room_dimensions {
        dims.schroeder_frequency()
    } else {
        schroeder_config.schroeder_freq
    };

    let low_config = &schroeder_config.low_freq_config;
    let high_config = &schroeder_config.high_freq_config;
    let high_min_q = optimizer.min_q.max(0.3);
    if low_config.min_q > low_config.max_q {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "Schroeder low-frequency min_q ({}) exceeds max_q ({})",
                low_config.min_q, low_config.max_q
            ),
        });
    }
    if high_min_q > high_config.max_q {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "Schroeder high-frequency min_q ({high_min_q}) exceeds max_q ({})",
                high_config.max_q
            ),
        });
    }

    let has_non_flat_target = optimizer
        .target_response
        .as_ref()
        .is_some_and(|tr| tr.shape != TargetShape::Flat);

    // A split outside the configured optimization band has only one real
    // side. Do not manufacture an inverted second band (for example
    // [400, 80] Hz); optimize the available side with the full filter budget.
    if schroeder_freq >= optimizer.max_freq {
        let (min_db, max_db) = low_freq_gain_bounds(optimizer, low_config, has_non_flat_target);
        let low_optimizer = OptimizerConfig {
            min_q: low_config.min_q,
            max_q: low_config.max_q,
            min_db,
            max_db,
            ..optimizer.clone()
        };
        let result = optimize_channel_eq_detailed(curve, &low_optimizer, None, sample_rate)
            .map_err(|e| AutoeqError::OptimizationFailed {
                message: format!("Low-frequency EQ optimization failed: {e}"),
            })?;
        return Ok(SchroederOptimizationResult {
            low_filters: clamp_filter_q(result.filters, low_config.min_q, low_config.max_q),
            high_filters: Vec::new(),
            optimizer_evidence: result.optimizer_evidence,
        });
    }
    if schroeder_freq <= optimizer.min_freq {
        let high_optimizer = OptimizerConfig {
            num_filters: if high_config.shelving_only {
                optimizer.num_filters.min(2)
            } else {
                optimizer.num_filters
            },
            min_q: high_min_q,
            max_q: high_config.max_q,
            peq_model: if high_config.shelving_only {
                "ls-pk-hs".to_string()
            } else {
                optimizer.peq_model.clone()
            },
            ..optimizer.clone()
        };
        let result = optimize_channel_eq_detailed(curve, &high_optimizer, None, sample_rate)
            .map_err(|e| AutoeqError::OptimizationFailed {
                message: format!("High-frequency EQ optimization failed: {e}"),
            })?;
        return Ok(SchroederOptimizationResult {
            low_filters: Vec::new(),
            high_filters: clamp_filter_q(result.filters, high_min_q, high_config.max_q),
            optimizer_evidence: result.optimizer_evidence,
        });
    }

    // Determine filter allocation (roughly proportional to frequency range)
    let total_filters = optimizer.num_filters;
    let log_range_total = (optimizer.max_freq / optimizer.min_freq).log2();
    let log_range_low = (schroeder_freq / optimizer.min_freq).max(1.0).log2();
    let low_ratio = log_range_low / log_range_total;

    let low_filters = ((total_filters as f64 * low_ratio).round() as usize)
        .max(1)
        .min(total_filters - 1);
    let high_filters = total_filters - low_filters;

    debug!(
        "  Schroeder split: {} filters below {:.1}Hz, {} filters above",
        low_filters, schroeder_freq, high_filters
    );

    // Each sub-pass gets the full maxeval budget. With fewer filters (lower
    // dimensionality) the optimizer converges faster, so the same budget is
    // adequate for each pass independently.
    // When target_tilt is active, the optimizer works on a tilt-adjusted curve
    // where following the tilt may require both boosts and cuts. Allow limited
    // boost (half the configured max) to give the optimizer enough freedom.
    let (low_min_db, low_max_db) = low_freq_gain_bounds(optimizer, low_config, has_non_flat_target);
    let low_optimizer = OptimizerConfig {
        num_filters: low_filters,
        min_freq: optimizer.min_freq,
        max_freq: schroeder_freq,
        min_q: low_config.min_q,
        max_q: low_config.max_q,
        min_db: low_min_db,
        max_db: low_max_db,
        ..optimizer.clone()
    };

    let low_result = optimize_channel_eq_detailed(
        curve,
        &low_optimizer,
        None, // No additional target for split optimization
        sample_rate,
    )
    .map_err(|e| AutoeqError::OptimizationFailed {
        message: format!("Low-frequency EQ optimization failed: {}", e),
    })?;
    let low_eq_filters = clamp_filter_q(low_result.filters, low_config.min_q, low_config.max_q);

    // High frequency optimization (above Schroeder)
    let high_optimizer = OptimizerConfig {
        num_filters: if high_config.shelving_only {
            high_filters.min(2)
        } else {
            high_filters
        },
        min_freq: schroeder_freq,
        max_freq: optimizer.max_freq,
        min_q: high_min_q, // Ensure minimum Q for broad filters
        max_q: high_config.max_q,
        peq_model: if high_config.shelving_only {
            "ls-pk-hs".to_string()
        } else {
            optimizer.peq_model.clone()
        },
        ..optimizer.clone()
    };

    // Apply low-freq correction first, then optimize high-freq on residual
    let low_resp =
        response::compute_peq_complex_response(&low_eq_filters, &curve.freq, sample_rate);
    let curve_with_low_correction = response::apply_complex_response(curve, &low_resp);

    let high_result = optimize_channel_eq_detailed(
        &curve_with_low_correction,
        &high_optimizer,
        None,
        sample_rate,
    )
    .map_err(|e| AutoeqError::OptimizationFailed {
        message: format!("High-frequency EQ optimization failed: {}", e),
    })?;
    let high_eq_filters = high_result.filters;

    // Post-optimization Q clamping: optimizer backends can violate bounds
    // slightly (or significantly with low maxeval). Enforce the configured Q
    // constraints on the returned filters to guarantee the Schroeder split
    // invariant.
    let high_eq_filters = clamp_filter_q(high_eq_filters, high_min_q, high_config.max_q);

    let mut optimizer_evidence = low_result.optimizer_evidence;
    optimizer_evidence.extend(high_result.optimizer_evidence);
    Ok(SchroederOptimizationResult {
        low_filters: low_eq_filters,
        high_filters: high_eq_filters,
        optimizer_evidence,
    })
}

fn low_freq_gain_bounds(
    optimizer: &OptimizerConfig,
    low_config: &LowFreqFilterConfig,
    has_non_flat_target: bool,
) -> (f64, f64) {
    if let Some(configured_max) = low_config.max_db {
        let configured_abs = configured_max.abs();
        let max_db = if low_config.allow_boost {
            configured_abs
        } else {
            0.0
        };
        return (-configured_abs, max_db);
    }

    if low_config.allow_boost {
        (optimizer.min_db, optimizer.max_db)
    } else if has_non_flat_target {
        (optimizer.min_db, (optimizer.max_db / 2.0).min(3.0))
    } else {
        (optimizer.min_db, 0.0)
    }
}

/// Clamp Q values of filters to [min_q, max_q], recomputing biquad coefficients.
fn clamp_filter_q(filters: Vec<Biquad>, min_q: f64, max_q: f64) -> Vec<Biquad> {
    filters
        .into_iter()
        .map(|f| {
            let clamped_q = f.q.clamp(min_q, max_q);
            if (clamped_q - f.q).abs() > 1e-6 {
                debug!(
                    "  Clamping filter Q at {:.0} Hz: {:.2} -> {:.2}",
                    f.freq, f.q, clamped_q
                );
                Biquad::new(f.filter_type, f.freq, f.srate, clamped_q, f.db_gain)
            } else {
                f
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn curve_with_bass_peak_and_treble_tilt() -> Curve {
        let freq = Array1::logspace(10.0, f64::log10(20.0), f64::log10(20000.0), 128);
        let spl = freq.mapv(|f| {
            let bass_peak = 8.0 * (-(f / 80.0).log2().powi(2) / (2.0 * 0.20_f64.powi(2))).exp();
            let treble_tilt = if f > 300.0 {
                2.0 * (f / 300.0).log2()
            } else {
                0.0
            };
            80.0 + bass_peak + treble_tilt
        });

        Curve {
            freq,
            spl,
            phase: None,
            ..Default::default()
        }
    }

    #[test]
    fn explicit_low_freq_max_db_respects_cuts_only_setting() {
        let optimizer = OptimizerConfig {
            min_db: -12.0,
            max_db: 4.0,
            ..Default::default()
        };
        let low_config = LowFreqFilterConfig {
            allow_boost: false,
            max_db: Some(14.0),
            ..Default::default()
        };

        assert_eq!(
            low_freq_gain_bounds(&optimizer, &low_config, false),
            (-14.0, 0.0)
        );
    }

    #[test]
    fn explicit_low_freq_max_db_allows_symmetric_range_when_boost_enabled() {
        let optimizer = OptimizerConfig::default();
        let low_config = LowFreqFilterConfig {
            allow_boost: true,
            max_db: Some(14.0),
            ..Default::default()
        };

        assert_eq!(
            low_freq_gain_bounds(&optimizer, &low_config, false),
            (-14.0, 14.0)
        );
    }

    #[test]
    fn schroeder_split_rejects_fewer_than_two_filters_without_panicking() {
        let curve = curve_with_bass_peak_and_treble_tilt();
        let split = SchroederSplitConfig {
            enabled: true,
            schroeder_freq: 200.0,
            ..Default::default()
        };

        for num_filters in [0, 1] {
            let optimizer = OptimizerConfig {
                num_filters,
                min_freq: 20.0,
                max_freq: 2_000.0,
                ..Default::default()
            };
            let error =
                optimize_with_schroeder_split_detailed(&curve, &optimizer, &split, 48_000.0)
                    .expect_err("undersized split must be rejected");
            assert!(error.to_string().contains("at least 2 filters"));
        }
    }

    #[test]
    fn schroeder_split_above_band_uses_only_low_frequency_pass() {
        let curve = curve_with_bass_peak_and_treble_tilt();
        let optimizer = OptimizerConfig {
            num_filters: 2,
            max_iter: 20,
            population: 6,
            refine: false,
            min_freq: 20.0,
            max_freq: 80.0,
            psychoacoustic: false,
            ..Default::default()
        };
        let split = SchroederSplitConfig {
            enabled: true,
            schroeder_freq: 400.0,
            ..Default::default()
        };

        let result =
            optimize_with_schroeder_split_detailed(&curve, &optimizer, &split, 48_000.0).unwrap();
        assert!(!result.low_filters.is_empty());
        assert!(result.high_filters.is_empty());
        assert!(result.low_filters.iter().all(|filter| {
            filter.q >= split.low_freq_config.min_q && filter.q <= split.low_freq_config.max_q
        }));
    }

    #[test]
    fn schroeder_split_rejects_inverted_high_q_bounds_without_panicking() {
        let curve = curve_with_bass_peak_and_treble_tilt();
        let optimizer = OptimizerConfig {
            num_filters: 2,
            min_q: 2.0,
            ..Default::default()
        };
        let split = SchroederSplitConfig {
            enabled: true,
            ..Default::default()
        };

        let error = optimize_with_schroeder_split_detailed(&curve, &optimizer, &split, 48_000.0)
            .unwrap_err();
        assert!(error.to_string().contains("min_q (2) exceeds max_q (1)"));
    }

    #[test]
    fn shelving_only_high_pass_emits_no_peak_filters() {
        let curve = curve_with_bass_peak_and_treble_tilt();
        let optimizer = OptimizerConfig {
            num_filters: 4,
            max_iter: 20,
            population: 6,
            refine: false,
            min_freq: 400.0,
            max_freq: 2_000.0,
            psychoacoustic: false,
            ..Default::default()
        };
        let mut split = SchroederSplitConfig {
            enabled: true,
            schroeder_freq: 200.0,
            ..Default::default()
        };
        split.high_freq_config.shelving_only = true;

        let result =
            optimize_with_schroeder_split_detailed(&curve, &optimizer, &split, 48_000.0).unwrap();
        assert!(result.low_filters.is_empty());
        assert!(!result.high_filters.is_empty());
        assert!(result.high_filters.len() <= 2);
        assert!(result.high_filters.iter().all(|filter| matches!(
            filter.filter_type,
            math_audio_iir_fir::BiquadFilterType::Lowshelf
                | math_audio_iir_fir::BiquadFilterType::Highshelf
        )));
    }
}
