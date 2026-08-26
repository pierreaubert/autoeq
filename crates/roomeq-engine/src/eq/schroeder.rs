use super::optimize::optimize_channel_eq_detailed_with_normalization_mean;
use crate::{AutoeqError, Curve};
use autoeq_core::{Result, response};
use log::debug;
use math_audio_iir_fir::Biquad;
use roomeq_model::{LowFreqFilterConfig, OptimizerConfig, SchroederSplitConfig};

fn schroeder_filter_allocation(
    total_filters: usize,
    proportional_low_filters: usize,
    shelving_only: bool,
) -> (usize, usize) {
    if shelving_only {
        (total_filters - 2, 2)
    } else {
        (
            proportional_low_filters,
            total_filters - proportional_low_filters,
        )
    }
}

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
    if high_config.shelving_only && optimizer.num_filters < 3 {
        return Err(AutoeqError::InvalidConfiguration {
            message: "Schroeder shelving_only requires at least 3 filters so the high band can contain a true high shelf".into(),
        });
    }
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

    // Both sub-passes must use one full-band reference. Re-normalizing each
    // side independently erases real level offsets across the split.
    let (normalization_sum, normalization_count) = curve
        .freq
        .iter()
        .zip(curve.spl.iter())
        .filter(|(frequency, _)| {
            **frequency >= optimizer.min_freq && **frequency <= optimizer.max_freq
        })
        .fold((0.0, 0usize), |(sum, count), (_, level)| {
            (sum + *level, count + 1)
        });
    let normalization_mean_spl = if normalization_count > 0 {
        normalization_sum / normalization_count as f64
    } else {
        0.0
    };

    // A split outside the configured optimization band has only one real
    // side. Do not manufacture an inverted second band (for example
    // [400, 80] Hz); optimize the available side with the full filter budget.
    if schroeder_freq >= optimizer.max_freq {
        let (min_db, max_db) = low_freq_gain_bounds(optimizer, low_config);
        let low_optimizer = OptimizerConfig {
            min_q: low_config.min_q,
            max_q: low_config.max_q,
            min_db,
            max_db,
            ..optimizer.clone()
        };
        let result = optimize_channel_eq_detailed_with_normalization_mean(
            curve,
            &low_optimizer,
            None,
            sample_rate,
            normalization_mean_spl,
        )
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
        let result = optimize_channel_eq_detailed_with_normalization_mean(
            curve,
            &high_optimizer,
            None,
            sample_rate,
            normalization_mean_spl,
        )
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

    let proportional_low_filters = ((total_filters as f64 * low_ratio).round() as usize)
        .max(1)
        .min(total_filters - 1);
    // `ls-pk-hs` needs at least two high-band filters to include a real
    // high shelf rather than producing only its leading low shelf.
    let (low_filters, high_filters) = schroeder_filter_allocation(
        total_filters,
        proportional_low_filters,
        high_config.shelving_only,
    );

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
    let (low_min_db, low_max_db) = low_freq_gain_bounds(optimizer, low_config);
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

    let low_result = optimize_channel_eq_detailed_with_normalization_mean(
        curve,
        &low_optimizer,
        None, // No additional target for split optimization
        sample_rate,
        normalization_mean_spl,
    )
    .map_err(|e| AutoeqError::OptimizationFailed {
        message: format!("Low-frequency EQ optimization failed: {}", e),
    })?;
    let low_eq_filters = clamp_filter_q(low_result.filters, low_config.min_q, low_config.max_q);

    // High frequency optimization (above Schroeder)
    let high_optimizer = OptimizerConfig {
        num_filters: high_filters,
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

    let high_result = optimize_channel_eq_detailed_with_normalization_mean(
        &curve_with_low_correction,
        &high_optimizer,
        None,
        sample_rate,
        normalization_mean_spl,
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

        assert_eq!(low_freq_gain_bounds(&optimizer, &low_config), (-14.0, 0.0));
    }

    #[test]
    fn explicit_low_freq_max_db_allows_symmetric_range_when_boost_enabled() {
        let optimizer = OptimizerConfig::default();
        let low_config = LowFreqFilterConfig {
            allow_boost: true,
            max_db: Some(14.0),
            ..Default::default()
        };

        assert_eq!(low_freq_gain_bounds(&optimizer, &low_config), (-14.0, 14.0));
    }

    #[test]
    fn implicit_low_freq_bounds_never_override_cuts_only_policy() {
        let optimizer = OptimizerConfig {
            min_db: -12.0,
            max_db: 12.0,
            ..OptimizerConfig::default()
        };
        let low_config = LowFreqFilterConfig {
            allow_boost: false,
            max_db: None,
            ..LowFreqFilterConfig::default()
        };

        assert_eq!(low_freq_gain_bounds(&optimizer, &low_config), (-12.0, 0.0));
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

    #[test]
    fn shelving_only_reserves_two_high_band_filters_and_empty_curves_error() {
        assert_eq!(schroeder_filter_allocation(3, 2, true), (1, 2));
        assert_eq!(schroeder_filter_allocation(8, 3, true), (6, 2));
        assert_eq!(schroeder_filter_allocation(8, 3, false), (3, 5));

        let error = optimize_with_schroeder_split_detailed(
            &Curve::default(),
            &OptimizerConfig::default(),
            &SchroederSplitConfig {
                enabled: true,
                ..SchroederSplitConfig::default()
            },
            48_000.0,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("at least two aligned frequency/SPL samples"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn schroeder_split_preserves_inter_band_level_information() {
        let freq = Array1::logspace(10.0, f64::log10(20.0), f64::log10(2_000.0), 96);
        let curve = Curve {
            spl: freq.mapv(|frequency| if frequency < 200.0 { 70.0 } else { 90.0 }),
            freq,
            phase: None,
            ..Default::default()
        };
        let optimizer = OptimizerConfig {
            num_filters: 4,
            max_iter: 40,
            population: 8,
            refine: false,
            min_freq: 20.0,
            max_freq: 2_000.0,
            min_db: -12.0,
            max_db: 12.0,
            psychoacoustic: false,
            ..Default::default()
        };
        let mut split = SchroederSplitConfig {
            enabled: true,
            schroeder_freq: 200.0,
            ..Default::default()
        };
        split.low_freq_config.allow_boost = true;

        let result =
            optimize_with_schroeder_split_detailed(&curve, &optimizer, &split, 48_000.0).unwrap();
        let filters = result
            .low_filters
            .iter()
            .chain(result.high_filters.iter())
            .cloned()
            .collect::<Vec<_>>();
        let filter_response =
            autoeq_core::response::compute_peq_complex_response(&filters, &curve.freq, 48_000.0);
        let corrected = autoeq_core::response::apply_complex_response(&curve, &filter_response);
        let band_mean = |low: f64, high: f64| {
            let values = corrected
                .freq
                .iter()
                .zip(corrected.spl.iter())
                .filter_map(|(&frequency, &level)| {
                    (frequency >= low && frequency < high).then_some(level)
                })
                .collect::<Vec<_>>();
            values.iter().sum::<f64>() / values.len() as f64
        };

        let corrected_step = band_mean(40.0, 160.0) - band_mean(250.0, 1_500.0);
        assert!(
            corrected_step.abs() < 15.0,
            "shared normalization should reduce the original 20 dB step, got {corrected_step:.2} dB"
        );
    }
}
