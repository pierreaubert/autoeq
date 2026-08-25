//! Crossover optimization for multi-driver groups
//!
//! # Phase Data Requirement
//!
//! Multi-driver crossover optimization uses complex summation to model
//! interference between drivers at crossover frequencies. For accurate
//! optimization, measurements should include phase data. Without phase data,
//! the optimizer assumes 0° phase, which may result in suboptimal crossover
//! frequencies, gains, and delays.

use crate::Curve;
use autoeq_optim::CrossoverType;
use autoeq_optim::loss::{DriverMeasurement, DriversLossData};
use log::{info, warn};
use ndarray::Array1;
use std::error::Error;

use roomeq_model::{CrossoverConfig, OptimizerConfig, RoomConfig};

/// Determine one optimizer frequency band per ordered speaker driver.
pub fn determine_optimization_bands(
    driver_count: usize,
    room_config: &RoomConfig,
    crossover_config: &CrossoverConfig,
) -> Vec<(f64, f64)> {
    let global_min = room_config.optimizer.min_freq;
    let global_max = room_config.optimizer.max_freq;
    let crossover_points = if let Some(frequencies) = &crossover_config.frequencies {
        frequencies.clone()
    } else if let Some(frequency) = crossover_config.frequency {
        vec![frequency]
    } else {
        Vec::new()
    };
    let crossover_bounds = |index: usize| -> (f64, f64) {
        if let Some(range) = crossover_config.frequency_range {
            return range;
        }
        crossover_points
            .get(index)
            .copied()
            .map(|frequency| (frequency, frequency))
            .unwrap_or((80.0, 3_000.0))
    };

    (0..driver_count)
        .map(|index| {
            let minimum = if index == 0 {
                global_min
            } else {
                crossover_bounds(index - 1).0 * 0.5
            };
            let maximum = if index + 1 == driver_count {
                global_max
            } else {
                crossover_bounds(index).1 * 2.0
            };
            (minimum.max(global_min), maximum.min(global_max))
        })
        .collect()
}

/// Apply polarity inversion to a driver curve.
///
/// When phase data is present, adds 180° to model polarity inversion.
/// When phase is missing, uses a constant 180° phase (pure polarity inversion)
/// rather than adding 180° to minimum-phase reconstruction, which would break
/// the Hilbert-transform relationship between log-magnitude and phase.
fn apply_polarity_inversion_to_driver(curve: &Curve, inverted: bool) -> DriverMeasurement {
    let mut new_curve = curve.clone();
    if inverted {
        let n = new_curve.freq.len();
        let phase = new_curve
            .phase
            .clone()
            .unwrap_or_else(|| Array1::from_elem(n, 0.0));
        new_curve.phase = Some(phase.mapv(|x| x + 180.0));
    }

    DriverMeasurement {
        freq: new_curve.freq,
        spl: new_curve.spl,
        phase: new_curve.phase,
    }
}

/// Optimize crossover for a group of driver measurements using autoeq's workflow
///
/// # Arguments
/// * `drivers` - Vector of driver measurements. This generic entry point sorts
///   by measurement span; use [`optimize_main_sub_crossover`] when every curve
///   shares the same RoomEQ grid and the acoustic roles are known.
/// * `crossover_type` - Type of crossover to use
/// * `sample_rate` - Sample rate for filter design
/// * `config` - Optimizer configuration
/// * `fixed_freqs` - Optional fixed crossover frequencies (skips frequency optimization)
/// * `crossover_freq_range` - Optional (min, max) frequency range for crossover optimization
///   (overrides config.min_freq/max_freq for the crossover search bounds)
///
/// # Returns
/// * Tuple of (optimal_gains, optimal_delays, optimal_crossover_freqs, combined_curve, inversions)
///
/// # Note on Phase Data
/// For accurate crossover optimization, measurements should include phase data.
/// The optimizer uses complex summation to model interference between drivers
/// at crossover frequencies.
#[allow(clippy::type_complexity)]
pub fn optimize_crossover(
    drivers: Vec<Curve>,
    crossover_type: CrossoverType,
    sample_rate: f64,
    config: &OptimizerConfig,
    fixed_freqs: Option<Vec<f64>>,
    crossover_freq_range: Option<(f64, f64)>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Curve, Vec<bool>), Box<dyn Error>> {
    optimize_crossover_impl(
        drivers,
        crossover_type,
        sample_rate,
        config,
        fixed_freqs,
        crossover_freq_range,
        false,
        0,
    )
}

/// Optimize a crossover while preserving an explicitly declared low-to-high order.
/// Driver 0 receives the low-pass branch and the final driver the high-pass.
#[allow(clippy::type_complexity)]
pub fn optimize_crossover_ordered(
    drivers: Vec<Curve>,
    crossover_type: CrossoverType,
    sample_rate: f64,
    config: &OptimizerConfig,
    fixed_freqs: Option<Vec<f64>>,
    crossover_freq_range: Option<(f64, f64)>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Curve, Vec<bool>), Box<dyn Error>> {
    optimize_crossover_impl(
        drivers,
        crossover_type,
        sample_rate,
        config,
        fixed_freqs,
        crossover_freq_range,
        true,
        0,
    )
}

/// Named result for a two-way main/sub crossover optimization.
///
/// The generic driver optimizer is ordered from the lowest acoustic band to
/// the highest: driver 0 receives the low-pass and driver 1 the high-pass.
/// This result maps that driver order back to the physical main/sub roles used
/// by RoomEQ's bass-management realization.
#[derive(Debug, Clone)]
pub struct MainSubCrossoverOptimization {
    pub main_gain_db: f64,
    pub main_delay_ms: f64,
    pub sub_gain_db: f64,
    pub sub_delay_ms: f64,
    pub sub_inverted: bool,
    pub crossover_frequency_hz: f64,
    pub combined_curve: Curve,
}

/// Physical roles for a two-way bass-management crossover.
///
/// Naming the branches here prevents positional `Vec<Curve>` call sites from
/// accidentally optimizing the electrical mirror of the deployed system.
#[derive(Debug, Clone)]
pub struct MainSubCrossoverInput {
    /// Main loudspeaker response that receives the high-pass branch.
    pub main_highpass: Curve,
    /// Subwoofer response that receives the low-pass branch.
    pub sub_lowpass: Curve,
}

/// Optimize a physical two-way bass-management crossover.
///
/// The subwoofer is deliberately driver 0 (low-pass) and the main is driver 1
/// (high-pass). The main is held as the polarity reference so any relative
/// inversion is returned directly as `sub_inverted`, matching the exported
/// RoomEQ signal path.
#[allow(clippy::too_many_arguments)]
pub fn optimize_main_sub_crossover(
    input: MainSubCrossoverInput,
    crossover_type: CrossoverType,
    sample_rate: f64,
    config: &OptimizerConfig,
    fixed_freqs: Option<Vec<f64>>,
    crossover_freq_range: Option<(f64, f64)>,
) -> Result<MainSubCrossoverOptimization, Box<dyn Error>> {
    let (gains, delays, crossover_freqs, combined_curve, inversions) = optimize_crossover_impl(
        vec![input.sub_lowpass, input.main_highpass],
        crossover_type,
        sample_rate,
        config,
        fixed_freqs,
        crossover_freq_range,
        true,
        1,
    )?;

    debug_assert_eq!(gains.len(), 2);
    debug_assert_eq!(delays.len(), 2);
    debug_assert_eq!(inversions.len(), 2);
    debug_assert_eq!(crossover_freqs.len(), 1);

    Ok(MainSubCrossoverOptimization {
        main_gain_db: gains[1],
        main_delay_ms: delays[1],
        sub_gain_db: gains[0],
        sub_delay_ms: delays[0],
        sub_inverted: inversions[0],
        crossover_frequency_hz: crossover_freqs[0],
        combined_curve,
    })
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn optimize_crossover_impl(
    drivers: Vec<Curve>,
    crossover_type: CrossoverType,
    sample_rate: f64,
    config: &OptimizerConfig,
    fixed_freqs: Option<Vec<f64>>,
    crossover_freq_range: Option<(f64, f64)>,
    preserve_order: bool,
    polarity_reference_index: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Curve, Vec<bool>), Box<dyn Error>> {
    // Check for missing phase data and warn
    let missing_phase_count = drivers.iter().filter(|c| c.phase.is_none()).count();
    if missing_phase_count > 0 {
        warn!(
            "Crossover optimization: {} of {} driver measurements are missing phase data. \
            This may result in suboptimal crossover frequencies and driver alignment. \
            For best results, include phase data in your measurements.",
            missing_phase_count,
            drivers.len()
        );
    }

    let n_drivers = drivers.len();
    if n_drivers == 0 {
        return Err("No drivers provided".into());
    }
    if polarity_reference_index >= n_drivers {
        return Err(format!(
            "Polarity reference driver {} is outside the {}-driver input",
            polarity_reference_index, n_drivers
        )
        .into());
    }

    // 1. Determine sort order (Low to High freq)
    // We need to pass sorted drivers to the optimizer, but return results in original order.
    let mut permutation: Vec<usize> = (0..n_drivers).collect();

    // Helper to get mean freq of a curve
    let get_mean_freq = |c: &Curve| {
        let min_f = c.freq.iter().copied().fold(f64::INFINITY, f64::min);
        let max_f = c.freq.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (min_f * max_f).sqrt()
    };

    if !preserve_order {
        permutation.sort_by(|&a, &b| {
            get_mean_freq(&drivers[a])
                .partial_cmp(&get_mean_freq(&drivers[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let sorted_drivers: Vec<Curve> = permutation.iter().map(|&i| drivers[i].clone()).collect();

    // 2. Try polarity combinations on SORTED drivers
    // For N drivers, we have 2^(N-1) combinations (one driver fixed as reference).
    let num_combinations = 1 << (n_drivers - 1);

    struct OptimizationResult {
        result: autoeq_optim::DriverOptimizationResult,
        inversions: Vec<bool>,
        data: DriversLossData,
    }

    let mut best_opt: Option<OptimizationResult> = None;

    // Use crossover-specific frequency range if provided, otherwise fall back to config
    let (xover_min_freq, xover_max_freq) =
        crossover_freq_range.unwrap_or((config.min_freq, config.max_freq));

    // Validate fixed frequencies size match once; it does not depend on polarity.
    if let Some(ref freqs) = fixed_freqs {
        let expected = n_drivers - 1;
        if freqs.len() != expected {
            return Err(format!(
                "Expected {} crossover frequencies for {} drivers, got {}",
                expected,
                n_drivers,
                freqs.len()
            )
            .into());
        }
    }

    for i in 0..num_combinations {
        let mut inversions = vec![false; n_drivers];
        let mut combination_bit = 0;
        for (driver_index, inverted) in inversions.iter_mut().enumerate() {
            if driver_index == polarity_reference_index {
                continue;
            }
            *inverted = (i >> combination_bit) & 1 == 1;
            combination_bit += 1;
        }

        // Create modified drivers with inverted phase where needed
        let modified_drivers: Vec<DriverMeasurement> = sorted_drivers
            .iter()
            .enumerate()
            .map(|(idx, curve)| apply_polarity_inversion_to_driver(curve, inversions[idx]))
            .collect();

        let drivers_data = DriversLossData::new_ordered(modified_drivers, crossover_type);

        // Run optimization
        let result = autoeq_optim::optimize_drivers_crossover(
            drivers_data.clone(),
            xover_min_freq,
            xover_max_freq,
            sample_rate,
            &config.algorithm,
            config.max_iter,
            config.population,
            config.min_db,
            config.max_db,
            fixed_freqs.clone(),
            config.seed,
        )?;

        match best_opt {
            None => {
                best_opt = Some(OptimizationResult {
                    result,
                    inversions,
                    data: drivers_data,
                });
            }
            Some(ref current_best) => {
                if result.post_objective < current_best.result.post_objective {
                    best_opt = Some(OptimizationResult {
                        result,
                        inversions,
                        data: drivers_data,
                    });
                }
            }
        }
    }

    let best = best_opt.ok_or("Optimization failed to produce any result")?;
    let result = best.result;
    let sorted_inversions = best.inversions;
    let drivers_data = best.data; // Use the data that produced the best result (includes correct phases)

    info!(
        "  Optimizing crossover for {} drivers ({:?}){}",
        n_drivers,
        crossover_type,
        if fixed_freqs.is_some() {
            " with fixed frequencies"
        } else {
            ""
        }
    );

    // Compute the combined response (using the best modified data)
    let combined_complex = autoeq_optim::loss::compute_drivers_combined_response_complex(
        &drivers_data,
        &result.gains,
        &result.crossover_freqs,
        Some(&result.delays),
        sample_rate,
    );
    let combined_spl = combined_complex.mapv(|z| 20.0 * z.norm().max(1e-12).log10());
    let combined_phase = combined_complex.mapv(|z| z.arg().to_degrees());

    let combined_curve = Curve {
        freq: drivers_data.freq_grid.clone(),
        spl: combined_spl,
        phase: Some(combined_phase),
        ..Default::default()
    };

    info!(
        "  Crossover optimization: gains={:?}, delays={:?} ms, freqs={:?}, inverts={:?}, final loss={:.6}",
        result
            .gains
            .iter()
            .map(|g| format!("{:+.2}", g))
            .collect::<Vec<_>>(),
        result
            .delays
            .iter()
            .map(|d| format!("{:.2}", d))
            .collect::<Vec<_>>(),
        result
            .crossover_freqs
            .iter()
            .map(|f| format!("{:.0}", f))
            .collect::<Vec<_>>(),
        sorted_inversions,
        result.post_objective
    );

    // 3. Map results back to original order
    let mut final_gains = vec![0.0; n_drivers];
    let mut final_delays = vec![0.0; n_drivers];
    let mut final_inversions = vec![false; n_drivers];

    for (sorted_idx, &original_idx) in permutation.iter().enumerate() {
        final_gains[original_idx] = result.gains[sorted_idx];
        final_delays[original_idx] = result.delays[sorted_idx];
        final_inversions[original_idx] = sorted_inversions[sorted_idx];
    }

    Ok((
        final_gains,
        final_delays,
        result.crossover_freqs,
        combined_curve,
        final_inversions,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crossover_config() -> CrossoverConfig {
        CrossoverConfig {
            crossover_type: "LR24".to_string(),
            frequency: None,
            frequencies: None,
            frequency_range: None,
        }
    }

    #[test]
    fn optimization_bands_cover_fixed_frequency_range() {
        let room_config = RoomConfig {
            optimizer: OptimizerConfig {
                min_freq: 20.0,
                max_freq: 20_000.0,
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };
        let crossover = CrossoverConfig {
            frequency_range: Some((200.0, 3_000.0)),
            ..crossover_config()
        };

        assert_eq!(
            determine_optimization_bands(3, &room_config, &crossover),
            vec![(20.0, 6_000.0), (100.0, 6_000.0), (100.0, 20_000.0)]
        );
    }

    #[test]
    fn optimization_bands_accept_one_or_many_fixed_frequencies() {
        let room_config = RoomConfig {
            optimizer: OptimizerConfig {
                min_freq: 20.0,
                max_freq: 20_000.0,
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };
        let one = CrossoverConfig {
            frequency: Some(1_000.0),
            ..crossover_config()
        };
        let many = CrossoverConfig {
            frequencies: Some(vec![200.0, 2_000.0]),
            ..crossover_config()
        };

        assert_eq!(determine_optimization_bands(2, &room_config, &one).len(), 2);
        assert_eq!(
            determine_optimization_bands(3, &room_config, &many).len(),
            3
        );
    }

    #[test]
    fn optimization_bands_fall_back_without_crossover_frequency() {
        let room_config = RoomConfig {
            optimizer: OptimizerConfig {
                min_freq: 20.0,
                max_freq: 20_000.0,
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        };
        let bands = determine_optimization_bands(2, &room_config, &crossover_config());

        assert_eq!(bands[0].0, 20.0);
        assert_eq!(bands[1].1, 20_000.0);
    }
    use ndarray::Array1;

    #[test]
    fn polarity_inversion_with_missing_phase_uses_constant_180_deg() {
        let curve = Curve {
            freq: Array1::from_vec(vec![100.0, 1000.0]),
            spl: Array1::from_vec(vec![0.0, 0.0]),
            phase: None,
            ..Default::default()
        };

        let driver = apply_polarity_inversion_to_driver(&curve, true);

        let phase = driver.phase.expect("phase should be present");
        assert!((phase[0] - 180.0).abs() < 1e-9);
        assert!((phase[1] - 180.0).abs() < 1e-9);
    }

    #[test]
    fn polarity_inversion_with_existing_phase_adds_180_deg() {
        let curve = Curve {
            freq: Array1::from_vec(vec![100.0, 1000.0]),
            spl: Array1::from_vec(vec![0.0, 0.0]),
            phase: Some(Array1::from_vec(vec![30.0, -45.0])),
            ..Default::default()
        };

        let driver = apply_polarity_inversion_to_driver(&curve, true);

        let phase = driver.phase.expect("phase should be present");
        assert!((phase[0] - 210.0).abs() < 1e-9);
        assert!((phase[1] - 135.0).abs() < 1e-9);
    }

    #[test]
    fn no_polarity_inversion_preserves_missing_phase() {
        let curve = Curve {
            freq: Array1::from_vec(vec![100.0, 1000.0]),
            spl: Array1::from_vec(vec![0.0, 0.0]),
            phase: None,
            ..Default::default()
        };

        let driver = apply_polarity_inversion_to_driver(&curve, false);

        assert!(driver.phase.is_none());
    }

    #[test]
    fn combined_curve_preserves_phase_from_complex_sum() {
        let drivers = vec![
            Curve {
                freq: Array1::from_vec(vec![100.0, 1000.0]),
                spl: Array1::from_vec(vec![0.0, 0.0]),
                phase: Some(Array1::from_vec(vec![0.0, 0.0])),
                ..Default::default()
            },
            Curve {
                freq: Array1::from_vec(vec![100.0, 1000.0]),
                spl: Array1::from_vec(vec![0.0, 0.0]),
                phase: Some(Array1::from_vec(vec![180.0, 180.0])),
                ..Default::default()
            },
        ];

        let result = optimize_crossover(
            drivers,
            CrossoverType::None,
            48000.0,
            &OptimizerConfig {
                num_filters: 1,
                max_iter: 10,
                population: 4,
                seed: Some(42),
                ..Default::default()
            },
            None,
            None,
        );

        assert!(result.is_ok());
        let (_, _, _, combined_curve, _) = result.unwrap();
        assert!(
            combined_curve.phase.is_some(),
            "combined curve should preserve phase"
        );
    }

    #[test]
    fn main_sub_optimizer_matches_realized_highpass_main_lowpass_sub() {
        let sample_rate = 48_000.0;
        let point_count = 100;
        let freq = Array1::from_iter((0..point_count).map(|index| {
            let ratio = index as f64 / (point_count - 1) as f64;
            20.0 * 1_000.0_f64.powf(ratio)
        }));
        let main = Curve {
            freq: freq.clone(),
            spl: Array1::zeros(point_count),
            phase: Some(Array1::zeros(point_count)),
            ..Default::default()
        };
        let sub = Curve {
            freq,
            spl: Array1::zeros(point_count),
            phase: Some(Array1::from_iter(
                (0..point_count).map(|index| 35.0 + index as f64 * 0.7),
            )),
            ..Default::default()
        };

        let optimized = optimize_main_sub_crossover(
            MainSubCrossoverInput {
                main_highpass: main.clone(),
                sub_lowpass: sub.clone(),
            },
            CrossoverType::LinkwitzRiley4,
            sample_rate,
            &OptimizerConfig {
                num_filters: 1,
                max_iter: 10,
                population: 4,
                min_db: 0.0,
                max_db: 0.0,
                seed: Some(0x5ab),
                ..Default::default()
            },
            Some(vec![120.0]),
            None,
        )
        .expect("main/sub crossover optimization should succeed");

        let realized = crate::topology::predict_bass_management_sum(
            &main,
            &sub,
            CrossoverType::LinkwitzRiley4.to_plugin_string(),
            optimized.crossover_frequency_hz,
            sample_rate,
            optimized.main_gain_db,
            optimized.sub_gain_db,
            optimized.main_delay_ms,
            optimized.sub_delay_ms,
            optimized.sub_inverted,
        )
        .expect("phase-aware realized crossover sum should be available");

        assert_eq!(
            optimized.combined_curve.freq.len(),
            optimized.combined_curve.spl.len(),
            "optimized crossover response must contain one SPL value per frequency",
        );

        let mut mirrored_highpass_sub = sub.clone();
        if optimized.sub_inverted {
            mirrored_highpass_sub
                .phase
                .as_mut()
                .unwrap()
                .mapv_inplace(|phase| phase + 180.0);
        }
        let mirrored = crate::topology::predict_bass_management_sum(
            &mirrored_highpass_sub,
            &main,
            CrossoverType::LinkwitzRiley4.to_plugin_string(),
            optimized.crossover_frequency_hz,
            sample_rate,
            optimized.sub_gain_db,
            optimized.main_gain_db,
            optimized.sub_delay_ms,
            optimized.main_delay_ms,
            false,
        )
        .expect("mirrored crossover sum should be available for regression comparison");

        let max_magnitude_error = optimized
            .combined_curve
            .spl
            .iter()
            .zip(realized.spl.iter())
            .map(|(modeled, realized)| (modeled - realized).abs())
            .fold(0.0_f64, f64::max);
        // The optimizer and deployed topology use independent interpolation and
        // response-realization paths, so allow only sub-audible numerical drift.
        assert!(
            max_magnitude_error < 0.1,
            "optimizer and realized crossover magnitude differ by up to {max_magnitude_error} dB"
        );
        let modeled_phase = optimized.combined_curve.phase.as_ref().unwrap();
        let realized_phase = realized.phase.as_ref().unwrap();
        let max_phase_error = modeled_phase
            .iter()
            .zip(realized_phase.iter())
            .map(|(modeled, realized)| {
                ((modeled - realized + 180.0).rem_euclid(360.0) - 180.0).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            max_phase_error < 1.0,
            "optimizer and realized crossover phase differ by up to {max_phase_error} degrees"
        );

        let mirrored_magnitude_error = optimized
            .combined_curve
            .spl
            .iter()
            .zip(mirrored.spl.iter())
            .map(|(modeled, mirrored)| (modeled - mirrored).abs())
            .fold(0.0_f64, f64::max);
        let mirrored_phase = mirrored.phase.as_ref().unwrap();
        let mirrored_phase_error = modeled_phase
            .iter()
            .zip(mirrored_phase.iter())
            .map(|(modeled, mirrored)| {
                ((modeled - mirrored + 180.0).rem_euclid(360.0) - 180.0).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            mirrored_magnitude_error > max_magnitude_error + 1.0
                || mirrored_phase_error > max_phase_error + 10.0,
            "regression fixture does not distinguish physical and mirrored roles: physical={max_magnitude_error:.3} dB/{max_phase_error:.3} deg, mirrored={mirrored_magnitude_error:.3} dB/{mirrored_phase_error:.3} deg"
        );
    }

    #[test]
    fn test_parse_crossover_type() {
        assert!(matches!(
            "lr24".parse::<CrossoverType>(),
            Ok(CrossoverType::LinkwitzRiley4)
        ));
        assert!(matches!(
            "LR4".parse::<CrossoverType>(),
            Ok(CrossoverType::LinkwitzRiley4)
        ));
        assert!(matches!(
            "butterworth2".parse::<CrossoverType>(),
            Ok(CrossoverType::Butterworth2)
        ));
        assert!(matches!(
            "lr48".parse::<CrossoverType>(),
            Ok(CrossoverType::LinkwitzRiley8)
        ));
        assert!(matches!(
            "LinearPhase".parse::<CrossoverType>(),
            Ok(CrossoverType::LinearPhase)
        ));
        assert!("invalid".parse::<CrossoverType>().is_err());
    }
}
