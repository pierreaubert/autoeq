use super::super::artifacts::{self, ConvolutionArtifactKind};
use super::super::crossover;
use super::super::dba;
use super::super::eq;
use super::super::fir;
use super::super::multiseat::{self, MultiSeatMeasurements};
use super::super::multisub;
use super::super::optimize::detect_passband_and_mean;
use super::super::output;
use super::super::speaker_eq::determine_optimization_bands;
use super::super::types::{
    LEGACY_SPEAKER_GROUP_ADVISORY, MixedModeConfig, MultiSeatConfig, MultiSubGroup,
    OptimizerConfig, RoomConfig, SpeakerGroup, SpeakerTopology,
};
use super::misc::apply_per_sub_filters;
use super::misc::average_power_curve;
use super::misc::compute_lr24_crossover_responses;
use super::misc::eq_score_regressed;
use super::misc::flat_loss_score;
use super::misc::identity_multiseat_result;
use super::misc::load_multisub_seat_measurements;
use super::misc::multiseat_peq_config;
use super::misc::split_curve_at_frequency;
use super::types::MixedModeResult;
use crate::Curve;
use crate::error::{AutoeqError, Result};
use crate::read as load;
use crate::response;
use log::{debug, info, warn};
use math_audio_dsp::analysis::compute_average_response;
use std::path::Path;

struct AggregatedTopologyBands {
    curves: Vec<Curve>,
    driver_band_indices: Vec<usize>,
    relative_gains: Vec<f64>,
    relative_delays: Vec<f64>,
    relative_inversions: Vec<bool>,
}

fn aggregate_topology_bands(
    topology: &SpeakerTopology,
    drivers: &[Curve],
    phase_trustworthy: &[bool],
    sample_rate: f64,
    optimizer: &OptimizerConfig,
) -> Result<AggregatedTopologyBands> {
    let bands = topology
        .acoustic_bands()
        .map_err(|message| AutoeqError::InvalidConfiguration { message })?;
    let mut driver_band_indices = vec![0; drivers.len()];
    let mut acoustic_drivers = Vec::with_capacity(bands.len());
    let mut relative_gains = vec![0.0; drivers.len()];
    let mut relative_delays = vec![0.0; drivers.len()];
    let mut relative_inversions = vec![false; drivers.len()];

    for (band_index, members) in bands.iter().enumerate() {
        for &driver_index in members {
            driver_band_indices[driver_index] = band_index;
        }
        if members.len() == 1 {
            acoustic_drivers.push(drivers[members[0]].clone());
            continue;
        }

        let preserve_phase = members
            .iter()
            .all(|&driver_index| phase_trustworthy[driver_index]);
        if preserve_phase {
            let member_curves = members
                .iter()
                .map(|&driver_index| drivers[driver_index].clone())
                .collect::<Vec<_>>();
            let (gains, delays, _, combined, inversions) = crossover::optimize_crossover_ordered(
                member_curves,
                crate::loss::CrossoverType::None,
                sample_rate,
                optimizer,
                Some(vec![0.0; members.len() - 1]),
                None,
            )
            .map_err(|error| AutoeqError::OptimizationFailed {
                message: format!("parallel-driver alignment failed: {error}"),
            })?;
            for (member_index, &driver_index) in members.iter().enumerate() {
                relative_gains[driver_index] = gains[member_index];
                relative_delays[driver_index] = delays[member_index];
                relative_inversions[driver_index] = inversions[member_index];
            }
            acoustic_drivers.push(combined);
            continue;
        }

        warn!(
            "  Parallel driver band contains missing phase; relative delay and polarity controls are disabled"
        );
        let measurements = members
            .iter()
            .map(|&driver_index| {
                let curve = &drivers[driver_index];
                crate::loss::DriverMeasurement {
                    freq: curve.freq.clone(),
                    spl: curve.spl.clone(),
                    phase: curve.phase.clone(),
                }
            })
            .collect::<Vec<_>>();
        let data = crate::loss::DriversLossData::new_ordered(
            measurements,
            crate::loss::CrossoverType::None,
        );
        let response = crate::loss::compute_drivers_combined_response_complex(
            &data,
            &vec![0.0; members.len()],
            &[],
            Some(&vec![0.0; members.len()]),
            sample_rate,
        );
        acoustic_drivers.push(Curve {
            freq: data.freq_grid,
            spl: response.mapv(|value| 20.0 * value.norm().max(1e-12).log10()),
            phase: preserve_phase.then(|| response.mapv(|value| value.arg().to_degrees())),
            ..Default::default()
        });
    }
    Ok(AggregatedTopologyBands {
        curves: acoustic_drivers,
        driver_band_indices,
        relative_gains,
        relative_delays,
        relative_inversions,
    })
}

pub(in super::super) fn process_speaker_group(
    channel_name: &str,
    group: &SpeakerGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
) -> Result<MixedModeResult> {
    warn!("  {LEGACY_SPEAKER_GROUP_ADVISORY}");
    process_speaker_topology_impl(
        channel_name,
        &group.to_legacy_topology(),
        room_config,
        sample_rate,
        _output_dir,
        true,
    )
}

pub(in super::super) fn process_speaker_topology(
    channel_name: &str,
    topology: &SpeakerTopology,
    room_config: &RoomConfig,
    sample_rate: f64,
    output_dir: &Path,
) -> Result<MixedModeResult> {
    process_speaker_topology_impl(
        channel_name,
        topology,
        room_config,
        sample_rate,
        output_dir,
        false,
    )
}

fn process_speaker_topology_impl(
    channel_name: &str,
    group: &SpeakerTopology,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
    legacy_ordering: bool,
) -> Result<MixedModeResult> {
    group
        .validate()
        .map_err(|message| AutoeqError::InvalidConfiguration { message })?;
    // 1. Load all measurements in the group
    let mut drivers = Vec::new();
    for (i, driver) in group.drivers.iter().enumerate() {
        let curve = load::load_source(&driver.measurement).map_err(|e| {
            AutoeqError::InvalidMeasurement {
                message: format!(
                    "Failed to load driver '{}' ({}) measurement for channel {}: {}",
                    driver.id, i, channel_name, e
                ),
            }
        })?;
        drivers.push((driver.clone(), curve));
    }

    debug!("  Loaded {} driver measurements", drivers.len());

    // Legacy inputs retain the old inferred ordering. Explicit topology order is authoritative.
    if legacy_ordering {
        drivers.sort_by(|(_, a), (_, b)| {
            let get_mean = |c: &Curve| {
                let (passband, _) = detect_passband_and_mean(c);
                let (min_f, max_f) = passband.unwrap_or_else(|| {
                    let min_f = c.freq.iter().copied().fold(f64::INFINITY, f64::min);
                    let max_f = c.freq.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    (min_f, max_f)
                });
                (min_f * max_f).sqrt()
            };
            get_mean(a)
                .partial_cmp(&get_mean(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let driver_ids = drivers
        .iter()
        .map(|(driver, _)| driver.id.clone())
        .collect::<Vec<_>>();
    let driver_bands = drivers
        .iter()
        .map(|(driver, _)| driver.crossover_band)
        .collect::<Vec<_>>();
    let driver_curves = drivers
        .into_iter()
        .map(|(_, curve)| curve)
        .collect::<Vec<_>>();
    let driver_phase_trustworthy = driver_curves
        .iter()
        .map(|curve| curve.phase.is_some())
        .collect::<Vec<_>>();

    let acoustic_band_count = group
        .acoustic_bands()
        .map_err(|message| AutoeqError::InvalidConfiguration { message })?
        .len();

    // 3. Get crossover config. A topology containing one acoustic band (for
    // example a parallel woofer array) does not need a crossover.
    let crossover_config = if let Some(crossover_ref) = &group.crossover {
        Some(
            room_config
                .crossovers
                .as_ref()
                .and_then(|xovers| xovers.get(crossover_ref))
                .ok_or_else(|| AutoeqError::InvalidConfiguration {
                    message: format!("Crossover configuration '{}' not found", crossover_ref),
                })?,
        )
    } else if acoustic_band_count > 1 || legacy_ordering {
        return Err(AutoeqError::InvalidConfiguration {
            message:
                "Speaker topology with multiple acoustic bands requires crossover configuration"
                    .to_string(),
        });
    } else {
        None
    };

    // 4. Per-Driver Linearization (Pre-Correction)
    info!("  Linearizing {} drivers...", driver_curves.len());
    let mut optimization_bands = crossover_config
        .map(|crossover| determine_optimization_bands(driver_curves.len(), room_config, crossover))
        .unwrap_or_else(|| {
            vec![
                (
                    room_config.optimizer.min_freq,
                    room_config.optimizer.max_freq,
                );
                driver_curves.len()
            ]
        });
    for (band, explicit) in optimization_bands.iter_mut().zip(driver_bands) {
        if let Some(explicit) = explicit {
            *band = (
                explicit.min_hz.max(room_config.optimizer.min_freq),
                explicit.max_hz.min(room_config.optimizer.max_freq),
            );
            if band.0 >= band.1 {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "explicit driver band [{:.1}, {:.1}] Hz does not overlap optimizer range [{:.1}, {:.1}] Hz",
                        explicit.min_hz,
                        explicit.max_hz,
                        room_config.optimizer.min_freq,
                        room_config.optimizer.max_freq
                    ),
                });
            }
        }
    }
    let mut linearized_drivers = Vec::with_capacity(driver_curves.len());
    let mut per_driver_filters = Vec::with_capacity(driver_curves.len());
    let mut optimizer_evidence = Vec::new();

    for (i, curve) in driver_curves.iter().enumerate() {
        let (min_f, max_f) = optimization_bands[i];
        info!(
            "    Driver {}: optimizing bandwidth {:.1}-{:.1} Hz",
            i, min_f, max_f
        );

        // Create driver-specific config
        let mut driver_opt_config = room_config.optimizer.clone();
        driver_opt_config.min_freq = min_f;
        driver_opt_config.max_freq = max_f;

        // Optimize EQ for this driver
        let result = eq::optimize_channel_eq_detailed(
            curve,
            &driver_opt_config,
            room_config.target_curve.as_ref(), // Use global target (usually flat)
            sample_rate,
        )
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!("Linearization failed for driver {}: {}", i, e),
        })?;
        let filters = result.filters;
        optimizer_evidence.extend(result.optimizer_evidence);

        // Apply filters to get linearized curve
        let resp = response::compute_peq_complex_response(&filters, &curve.freq, sample_rate);
        let linear_curve = response::apply_complex_response(curve, &resp);

        linearized_drivers.push(linear_curve);
        per_driver_filters.push(filters);
    }

    let topology_bands = aggregate_topology_bands(
        group,
        &linearized_drivers,
        &driver_phase_trustworthy,
        sample_rate,
        &room_config.optimizer,
    )?;
    let acoustic_drivers = topology_bands.curves;
    let driver_band_indices = topology_bands.driver_band_indices;

    // 5. Setup Crossover Optimization
    let crossover_type: crate::loss::CrossoverType = crossover_config
        .map(|crossover| crossover.crossover_type.parse())
        .transpose()
        .map_err(|e: String| AutoeqError::InvalidConfiguration { message: e })?
        .unwrap_or(crate::loss::CrossoverType::None);

    let fixed_freqs: Option<Vec<f64>> = crossover_config.and_then(|crossover| {
        crossover
            .frequencies
            .clone()
            .or_else(|| crossover.frequency.map(|freq| vec![freq]))
    });

    // 6. Compute pre-score (using linearized drivers)
    let n_drivers = acoustic_drivers.len();
    let initial_gains = vec![0.0; n_drivers];
    let mut initial_xover_freqs = Vec::new();
    // Simple geometric mean estimate for initial guess
    for _ in 0..(n_drivers - 1) {
        let (min, max) = match crossover_config.and_then(|crossover| crossover.frequency_range) {
            Some((a, b)) => (a, b),
            None => (80.0, 3000.0),
        };
        initial_xover_freqs.push((min * max).sqrt());
    }

    let driver_measurements: Vec<crate::loss::DriverMeasurement> = acoustic_drivers
        .iter()
        .map(|curve| crate::loss::DriverMeasurement {
            freq: curve.freq.clone(),
            spl: curve.spl.clone(),
            phase: curve.phase.clone(),
        })
        .collect();

    let initial_delays = vec![0.0; n_drivers];

    let pre_score = if n_drivers == 1 {
        flat_loss_score(
            &acoustic_drivers[0],
            room_config.optimizer.min_freq,
            room_config.optimizer.max_freq,
        )
    } else {
        let drivers_data =
            crate::loss::DriversLossData::new_ordered(driver_measurements, crossover_type);
        crate::loss::drivers_flat_loss(
            &drivers_data,
            &initial_gains,
            &initial_xover_freqs,
            Some(&initial_delays),
            sample_rate,
            room_config.optimizer.min_freq,
            room_config.optimizer.max_freq,
        )
    };

    // 7. Optimize Crossover (using linearized drivers)
    let (gains, delays, crossover_freqs, combined_curve, inversions) = if n_drivers == 1 {
        (
            vec![0.0],
            vec![0.0],
            Vec::new(),
            acoustic_drivers[0].clone(),
            vec![false],
        )
    } else {
        crossover::optimize_crossover_ordered(
            acoustic_drivers,
            crossover_type,
            sample_rate,
            &room_config.optimizer,
            fixed_freqs,
            crossover_config.and_then(|crossover| crossover.frequency_range),
        )
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!("Crossover optimization failed: {}", e),
        })?
    };

    info!(
        "  Optimized crossover: freqs={:?}, gains={:?}, delays={:?}, inversions={:?}",
        crossover_freqs, gains, delays, inversions
    );

    let driver_gains = driver_band_indices
        .iter()
        .enumerate()
        .map(|(driver_index, &band_index)| {
            topology_bands.relative_gains[driver_index] + gains[band_index]
        })
        .collect::<Vec<_>>();
    let driver_delays = driver_band_indices
        .iter()
        .enumerate()
        .map(|(driver_index, &band_index)| {
            topology_bands.relative_delays[driver_index] + delays[band_index]
        })
        .collect::<Vec<_>>();
    let driver_inversions = driver_band_indices
        .iter()
        .enumerate()
        .map(|(driver_index, &band_index)| {
            topology_bands.relative_inversions[driver_index] ^ inversions[band_index]
        })
        .collect::<Vec<_>>();

    // 8. Global EQ (Optional Touch-up)
    // Run global EQ on the combined response to fix any remaining issues
    // but constrain it to be gentle if possible, or normal full optimization.
    let min_freq = room_config.optimizer.min_freq;
    let max_freq = room_config.optimizer.max_freq;
    let pre_global_eq_score = flat_loss_score(&combined_curve, min_freq, max_freq);

    let global_result = eq::optimize_channel_eq_detailed(
        &combined_curve,
        &room_config.optimizer,
        room_config.target_curve.as_ref(),
        sample_rate,
    )
    .map_err(|e| AutoeqError::OptimizationFailed {
        message: format!(
            "Global EQ optimization failed for channel {}: {}",
            channel_name, e
        ),
    })?;
    let global_eq_filters = global_result.filters;
    let post_score = global_result.loss;
    let global_evidence_start = optimizer_evidence.len();
    optimizer_evidence.extend(global_result.optimizer_evidence);

    let (global_eq_filters, post_score, final_curve) =
        if eq_score_regressed(pre_global_eq_score, post_score) {
            for evidence in &mut optimizer_evidence[global_evidence_start..] {
                evidence.selected_for_output = false;
            }
            warn!(
                "  Global EQ rejected for speaker group {}: flat loss {:.6} -> {:.6}",
                channel_name, pre_global_eq_score, post_score
            );
            (Vec::new(), pre_global_eq_score, combined_curve.clone())
        } else {
            info!("  Optimized {} Global EQ filters", global_eq_filters.len());
            info!(
                "  Pre-score: {:.6}, Post-score: {:.6}",
                pre_global_eq_score, post_score
            );
            let global_resp = response::compute_peq_complex_response(
                &global_eq_filters,
                &combined_curve.freq,
                sample_rate,
            );
            let final_curve = response::apply_complex_response(&combined_curve, &global_resp);
            (global_eq_filters, post_score, final_curve)
        };

    // 9. Build Output DSP Chain
    // We now have per-driver filters AND global filters.

    // Prepare display curves (raw drivers extended)
    let driver_curves_for_display: Vec<Curve> = driver_curves
        .iter()
        .map(output::extend_curve_to_full_range)
        .collect();

    let mut chain = output::build_topology_dsp_chain_with_curves(
        channel_name,
        &driver_ids,
        &driver_band_indices,
        &driver_gains,
        &driver_delays,
        &driver_inversions,
        &crossover_freqs,
        crossover_type.to_plugin_string(),
        &global_eq_filters,
        &per_driver_filters,
        &driver_curves_for_display,
    );

    // Detect passband
    let (norm_range, _passband_mean) = detect_passband_and_mean(&combined_curve);

    // Extend curves for display
    let display_initial = output::extend_curve_to_full_range(&combined_curve);
    let display_resp = response::compute_peq_complex_response(
        &global_eq_filters,
        &display_initial.freq,
        sample_rate,
    );
    let display_final = response::apply_complex_response(&display_initial, &display_resp);

    let mut initial_data: super::super::types::CurveData = (&display_initial).into();
    initial_data.norm_range = norm_range;
    let mut final_data: super::super::types::CurveData = (&display_final).into();
    final_data.norm_range = norm_range;

    chain.initial_curve = Some(initial_data.clone());
    chain.final_curve = Some(final_data.clone());
    chain.eq_response = Some(output::compute_eq_response(&initial_data, &final_data));

    // Use global mean for level alignment
    let min_freq = room_config.optimizer.min_freq;
    let max_freq = room_config.optimizer.max_freq;
    let freqs_f32: Vec<f32> = combined_curve.freq.iter().map(|&f| f as f32).collect();
    let spl_f32: Vec<f32> = combined_curve.spl.iter().map(|&s| s as f32).collect();
    let mean_spl = compute_average_response(
        &freqs_f32,
        &spl_f32,
        Some((min_freq as f32, max_freq as f32)),
    ) as f64;

    Ok((
        chain,
        pre_score,
        post_score,
        combined_curve.clone(),
        final_curve,
        global_eq_filters,
        mean_spl,
        None, // No single WAV for speaker groups
        None, // IIR-only for speaker groups
        optimizer_evidence,
    ))
}

/// Process multi-subwoofer group
///
/// Returns: (DSP chain, pre_score, post_score, initial_curve, final_curve, biquads, mean_spl, arrival_time_ms)
pub(in super::super) fn process_multisub_group(
    channel_name: &str,
    group: &MultiSubGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
) -> Result<MixedModeResult> {
    if let Some(multi_seat_config) = room_config
        .optimizer
        .multi_seat
        .as_ref()
        .filter(|config| config.enabled)
    {
        match load_multisub_seat_measurements(group)? {
            Some(seat_measurements) => {
                return process_multisub_group_multiseat(
                    channel_name,
                    group,
                    room_config,
                    multi_seat_config,
                    sample_rate,
                    seat_measurements,
                );
            }
            None => warn!(
                "  Multi-seat optimization is enabled for multi-sub group '{}' but subwoofer sources do not contain at least two seat measurements each; using single-seat multi-sub path",
                group.name
            ),
        }
    }

    let (result, combined_response, allpass_filters) = if group.allpass_optimization {
        // All-pass enhanced optimization
        info!("  Using all-pass enhanced multi-sub optimization");
        let ap_result = multisub::optimize_multisub_with_allpass(
            &group.subwoofers,
            &room_config.optimizer,
            sample_rate,
        )
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!("Multi-sub all-pass optimization failed: {}", e),
        })?;

        for (i, (freq, q)) in ap_result.allpass_filters.iter().enumerate() {
            info!(
                "  Sub {}: gain={:.1} dB, delay={:.1} ms, all-pass: {:.0} Hz Q={:.2}",
                i, ap_result.base.gains[i], ap_result.base.delays[i], freq, q
            );
        }

        (
            ap_result.base,
            ap_result.combined_response,
            Some(ap_result.allpass_filters),
        )
    } else {
        // Standard gain + delay optimization
        let detailed = multisub::optimize_multisub_detailed(
            &group.subwoofers,
            &room_config.optimizer,
            sample_rate,
        )
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!("Multi-sub optimization failed: {}", e),
        })?;
        (detailed.base, detailed.combined_response, None)
    };
    let combined_curve = combined_response.spatial_magnitude;
    let primary_curve = combined_response.primary_seat_complex;

    info!(
        "  Multi-sub optimization: gains={:?}, delays={:?} ms",
        result.gains, result.delays
    );

    let multisub_eq_optimizer = eq::resolve_multi_measurement_auto_optimizer_config(
        std::slice::from_ref(&combined_curve),
        &room_config.optimizer,
        eq::MultiEqAutoOptimizerContext::sub_channel(),
    );
    let eq_result = eq::optimize_channel_eq_detailed(
        &combined_curve,
        &multisub_eq_optimizer,
        room_config.target_curve.as_ref(),
        sample_rate,
    )
    .map_err(|e| AutoeqError::OptimizationFailed {
        message: format!("EQ optimization failed for multi-sub sum: {}", e),
    })?;
    let eq_filters = eq_result.filters;
    let post_score = eq_result.loss;
    let optimizer_evidence = eq_result.optimizer_evidence;

    info!(
        "  Global EQ: {} filters, score={:.6}",
        eq_filters.len(),
        post_score
    );

    // Load individual sub curves for per-driver display
    let driver_curves_for_display: Vec<Curve> = group
        .subwoofers
        .iter()
        .filter_map(|source| {
            load::load_source(source)
                .ok()
                .map(|c| output::extend_curve_to_full_range(&c))
        })
        .collect();
    let driver_display_ref = if driver_curves_for_display.len() == group.subwoofers.len() {
        Some(driver_curves_for_display.as_slice())
    } else {
        None
    };

    let mut chain = output::build_multisub_dsp_chain_with_allpass(
        channel_name,
        &group.name,
        group.subwoofers.len(),
        &result.gains,
        &result.delays,
        &eq_filters,
        None,
        None,
        driver_display_ref,
        allpass_filters.as_deref(),
        sample_rate,
    );

    let iir_resp =
        response::compute_peq_complex_response(&eq_filters, &combined_curve.freq, sample_rate);
    let final_curve = response::apply_complex_response(&combined_curve, &iir_resp);
    let final_primary_curve = primary_curve.as_ref().map(|primary| {
        let response =
            response::compute_peq_complex_response(&eq_filters, &primary.freq, sample_rate);
        response::apply_complex_response(primary, &response)
    });

    // Detect passband for normalization (used for display curves)
    let (norm_range, _passband_mean) = detect_passband_and_mean(&combined_curve);

    // Level alignment: use mean SPL within the EQ optimization range
    let min_freq = room_config.optimizer.min_freq;
    let max_freq = room_config.optimizer.max_freq;
    let freqs_f32: Vec<f32> = combined_curve.freq.iter().map(|&f| f as f32).collect();
    let spl_f32: Vec<f32> = combined_curve.spl.iter().map(|&s| s as f32).collect();
    let mean_spl = compute_average_response(
        &freqs_f32,
        &spl_f32,
        Some((min_freq as f32, max_freq as f32)),
    ) as f64;

    // Extend curves to 20 Hz – 20 kHz for display output
    let display_initial = output::extend_curve_to_full_range(&combined_curve);
    let display_resp =
        response::compute_peq_complex_response(&eq_filters, &display_initial.freq, sample_rate);
    let display_final = response::apply_complex_response(&display_initial, &display_resp);

    let mut initial_data: super::super::types::CurveData = (&display_initial).into();
    initial_data.norm_range = norm_range;
    let mut final_data: super::super::types::CurveData = (&display_final).into();
    final_data.norm_range = norm_range;

    chain.initial_curve = Some(initial_data.clone());
    chain.final_curve = Some(final_data.clone());
    chain.eq_response = Some(output::compute_eq_response(&initial_data, &final_data));

    Ok((
        chain,
        result.pre_objective,
        post_score,
        primary_curve.unwrap_or_else(|| combined_curve.clone()),
        final_primary_curve.unwrap_or(final_curve),
        eq_filters,
        mean_spl,
        None, // No single WAV for multi-sub groups
        None, // IIR-only for multi-sub groups
        optimizer_evidence,
    ))
}

fn process_multisub_group_multiseat(
    channel_name: &str,
    group: &MultiSubGroup,
    room_config: &RoomConfig,
    multi_seat_config: &MultiSeatConfig,
    sample_rate: f64,
    seat_measurements: Vec<Vec<Curve>>,
) -> Result<MixedModeResult> {
    let seat_count = seat_measurements.first().map(Vec::len).unwrap_or_default();
    let phase_trustworthy = seat_measurements
        .iter()
        .flatten()
        .all(|curve| curve.phase.is_some());
    let primary_seat = multi_seat_config
        .primary_seat
        .min(seat_count.saturating_sub(1));
    let peq_config = multiseat_peq_config(multi_seat_config, seat_count);
    let min_freq = room_config.optimizer.min_freq;
    let max_freq = room_config.optimizer.max_freq;
    let mut optimizer_evidence = Vec::new();

    let raw_measurements = MultiSeatMeasurements::new(seat_measurements.clone())?;
    let raw_identity = identity_multiseat_result(&raw_measurements, multi_seat_config);
    let raw_seat_curves = multiseat::compute_multiseat_combined_curves(
        &raw_measurements,
        &raw_identity,
        (min_freq, max_freq),
        sample_rate,
    )?;
    let raw_combined_curve = average_power_curve(&raw_seat_curves)?;
    let raw_primary_curve = phase_trustworthy.then(|| raw_seat_curves[primary_seat].clone());
    let pre_score = flat_loss_score(&raw_combined_curve, min_freq, max_freq);

    let per_sub_filters = if multi_seat_config.per_sub_peq {
        let mut filters = Vec::with_capacity(seat_measurements.len());
        for (sub_idx, sub_curves) in seat_measurements.iter().enumerate() {
            info!(
                "  Multi-seat per-sub PEQ: optimizing sub {} across {} seats ({:?})",
                sub_idx, seat_count, peq_config.strategy
            );
            let result = eq::optimize_channel_eq_multi_with_auto_optimizer_detailed(
                sub_curves,
                &room_config.optimizer,
                &peq_config,
                None,
                sample_rate,
                eq::MultiEqAutoOptimizerContext::sub_channel(),
            )
            .map_err(|e| AutoeqError::OptimizationFailed {
                message: format!("Per-sub multi-seat PEQ failed for sub {}: {}", sub_idx, e),
            })?;
            let sub_filters = result.filters;
            let sub_loss = result.loss;
            optimizer_evidence.extend(result.optimizer_evidence);
            info!(
                "  Sub {} per-seat PEQ: {} filters, loss={:.6}",
                sub_idx,
                sub_filters.len(),
                sub_loss
            );
            filters.push(sub_filters);
        }
        filters
    } else {
        vec![Vec::new(); seat_measurements.len()]
    };

    let corrected_measurements =
        apply_per_sub_filters(&seat_measurements, &per_sub_filters, sample_rate);
    let measurements = MultiSeatMeasurements::new(corrected_measurements)?;
    let mso_result = multiseat::optimize_multiseat(
        &measurements,
        multi_seat_config,
        (
            room_config.optimizer.min_freq,
            room_config.optimizer.max_freq,
        ),
        sample_rate,
    )?;
    info!(
        "  Multi-seat multi-sub optimization: gains={:?}, delays={:?} ms, polarities={:?}",
        mso_result.gains, mso_result.delays, mso_result.polarities
    );

    for (sub_idx, filters) in mso_result.allpass_filters.iter().enumerate() {
        for (filter_idx, (freq, q)) in filters.iter().enumerate() {
            info!(
                "  Sub {} all-pass {}: {:.0} Hz Q={:.2}",
                sub_idx, filter_idx, freq, q
            );
        }
    }

    let combined_seat_curves = multiseat::compute_multiseat_combined_curves(
        &measurements,
        &mso_result,
        (
            room_config.optimizer.min_freq,
            room_config.optimizer.max_freq,
        ),
        sample_rate,
    )?;
    let combined_curve = average_power_curve(&combined_seat_curves)?;
    let primary_curve = phase_trustworthy.then(|| combined_seat_curves[primary_seat].clone());

    let mut global_evidence_start = None;
    let mut eq_filters = if multi_seat_config.global_eq {
        let result = eq::optimize_channel_eq_multi_with_auto_optimizer_detailed(
            &combined_seat_curves,
            &room_config.optimizer,
            &peq_config,
            room_config.target_curve.as_ref(),
            sample_rate,
            eq::MultiEqAutoOptimizerContext::sub_channel(),
        )
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!("Global multi-seat EQ failed for multi-sub sum: {}", e),
        })?;
        let filters = result.filters;
        let loss = result.loss;
        global_evidence_start = Some(optimizer_evidence.len());
        optimizer_evidence.extend(result.optimizer_evidence);
        info!(
            "  Global multi-seat EQ: {} filters, score={:.6}",
            filters.len(),
            loss
        );
        filters
    } else {
        Vec::new()
    };

    let (norm_range, _passband_mean) = detect_passband_and_mean(&combined_curve);

    let global_eq_pre_score = flat_loss_score(&combined_curve, min_freq, max_freq);
    let iir_resp =
        response::compute_peq_complex_response(&eq_filters, &combined_curve.freq, sample_rate);
    let mut final_curve = response::apply_complex_response(&combined_curve, &iir_resp);
    let mut final_primary_curve = primary_curve.as_ref().map(|primary| {
        let response =
            response::compute_peq_complex_response(&eq_filters, &primary.freq, sample_rate);
        response::apply_complex_response(primary, &response)
    });
    let mut post_score = flat_loss_score(&final_curve, min_freq, max_freq);
    if multi_seat_config.global_eq && eq_score_regressed(global_eq_pre_score, post_score) {
        if let Some(start) = global_evidence_start {
            for evidence in &mut optimizer_evidence[start..] {
                evidence.selected_for_output = false;
            }
        }
        warn!(
            "  Global multi-seat EQ rejected for multi-sub sum: flat loss {:.6} -> {:.6}",
            global_eq_pre_score, post_score
        );
        eq_filters.clear();
        final_curve = combined_curve.clone();
        final_primary_curve = primary_curve.clone();
        post_score = global_eq_pre_score;
    }
    let freqs_f32: Vec<f32> = combined_curve.freq.iter().map(|&f| f as f32).collect();
    let spl_f32: Vec<f32> = combined_curve.spl.iter().map(|&s| s as f32).collect();
    let mean_spl = compute_average_response(
        &freqs_f32,
        &spl_f32,
        Some((min_freq as f32, max_freq as f32)),
    ) as f64;

    let driver_curves_for_display: Vec<Curve> = group
        .subwoofers
        .iter()
        .filter_map(|source| {
            load::load_source(source)
                .ok()
                .map(|c| output::extend_curve_to_full_range(&c))
        })
        .collect();
    let driver_display_ref = if driver_curves_for_display.len() == group.subwoofers.len() {
        Some(driver_curves_for_display.as_slice())
    } else {
        None
    };

    let mut chain = output::build_multisub_dsp_chain_advanced(
        channel_name,
        &group.name,
        group.subwoofers.len(),
        &mso_result.gains,
        &mso_result.delays,
        &eq_filters,
        None,
        None,
        driver_display_ref,
        Some(&per_sub_filters),
        Some(&mso_result.polarities),
        Some(&mso_result.allpass_filters),
        sample_rate,
    );

    let display_initial = output::extend_curve_to_full_range(&combined_curve);
    let display_resp =
        response::compute_peq_complex_response(&eq_filters, &display_initial.freq, sample_rate);
    let display_final = response::apply_complex_response(&display_initial, &display_resp);

    let mut initial_data: super::super::types::CurveData = (&display_initial).into();
    initial_data.norm_range = norm_range;
    let mut final_data: super::super::types::CurveData = (&display_final).into();
    final_data.norm_range = norm_range;

    chain.initial_curve = Some(initial_data.clone());
    chain.final_curve = Some(final_data.clone());
    chain.eq_response = Some(output::compute_eq_response(&initial_data, &final_data));

    Ok((
        chain,
        pre_score,
        post_score,
        raw_primary_curve.unwrap_or(raw_combined_curve),
        final_primary_curve.unwrap_or(final_curve),
        eq_filters,
        mean_spl,
        None,
        None,
        optimizer_evidence,
    ))
}

/// Process DBA configuration
///
/// Returns: (DSP chain, pre_score, post_score, initial_curve, final_curve, biquads, mean_spl, arrival_time_ms)
pub(in super::super) fn process_dba(
    channel_name: &str,
    dba_config: &super::super::types::DBAConfig,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
) -> Result<MixedModeResult> {
    let dba_result = dba::optimize_dba_detailed(dba_config, &room_config.optimizer, sample_rate)
        .map_err(|e| AutoeqError::OptimizationFailed {
            message: format!("DBA optimization failed: {}", e),
        })?;
    let result = dba_result.driver;
    let combined_curve = dba_result.combined_curve;
    let mut optimizer_evidence = vec![dba_result.optimizer_evidence];

    info!(
        "  DBA Optimization: Front Gain={:.2}dB, Rear Gain={:.2}dB, Rear Delay={:.2}ms",
        result.gains[0], result.gains[1], result.delays[1]
    );

    let eq_result = eq::optimize_channel_eq_detailed(
        &combined_curve,
        &room_config.optimizer,
        room_config.target_curve.as_ref(),
        sample_rate,
    )
    .map_err(|e| AutoeqError::OptimizationFailed {
        message: format!("EQ optimization failed for DBA sum: {}", e),
    })?;
    let eq_filters = eq_result.filters;
    let post_score = eq_result.loss;
    optimizer_evidence.extend(eq_result.optimizer_evidence);

    info!(
        "  Global EQ: {} filters, score={:.6}",
        eq_filters.len(),
        post_score
    );

    // Load front/rear array curves for per-driver display
    // DBA has 2 "drivers": front aggregate and rear aggregate
    let driver_display_ref = match (
        dba::sum_array_response(&dba_config.front),
        dba::sum_array_response(&dba_config.rear),
    ) {
        (Ok(front), Ok(rear)) => Some(vec![
            output::extend_curve_to_full_range(&front),
            output::extend_curve_to_full_range(&rear),
        ]),
        _ => None,
    };
    let driver_display_slice = driver_display_ref.as_deref();

    let mut chain = output::build_dba_dsp_chain_with_curves(
        channel_name,
        &result.gains,
        &result.delays,
        &eq_filters,
        None,
        None,
        driver_display_slice,
    );

    let iir_resp =
        response::compute_peq_complex_response(&eq_filters, &combined_curve.freq, sample_rate);
    let final_curve = response::apply_complex_response(&combined_curve, &iir_resp);

    // Detect passband for normalization (used for display curves)
    let (norm_range, _passband_mean) = detect_passband_and_mean(&combined_curve);

    // Level alignment: use mean SPL within the EQ optimization range
    let min_freq = room_config.optimizer.min_freq;
    let max_freq = room_config.optimizer.max_freq;
    let freqs_f32: Vec<f32> = combined_curve.freq.iter().map(|&f| f as f32).collect();
    let spl_f32: Vec<f32> = combined_curve.spl.iter().map(|&s| s as f32).collect();
    let mean_spl = compute_average_response(
        &freqs_f32,
        &spl_f32,
        Some((min_freq as f32, max_freq as f32)),
    ) as f64;

    // Extend curves to 20 Hz – 20 kHz for display output
    let display_initial = output::extend_curve_to_full_range(&combined_curve);
    let display_resp =
        response::compute_peq_complex_response(&eq_filters, &display_initial.freq, sample_rate);
    let display_final = response::apply_complex_response(&display_initial, &display_resp);

    let mut initial_data: super::super::types::CurveData = (&display_initial).into();
    initial_data.norm_range = norm_range;
    let mut final_data: super::super::types::CurveData = (&display_final).into();
    final_data.norm_range = norm_range;

    chain.initial_curve = Some(initial_data.clone());
    chain.final_curve = Some(final_data.clone());
    chain.eq_response = Some(output::compute_eq_response(&initial_data, &final_data));

    Ok((
        chain,
        result.pre_objective,
        post_score,
        combined_curve.clone(),
        final_curve,
        eq_filters,
        mean_spl,
        None, // No single WAV for DBA
        None, // IIR-only for DBA
        optimizer_evidence,
    ))
}

/// Process mixed mode with frequency-based crossover
///
/// This mode applies FIR correction to one frequency band (default: low frequencies)
/// and IIR correction to the other band (default: high frequencies), separated by
/// a configurable crossover frequency.
///
/// Returns: (DSP chain, pre_score, post_score, initial_curve, final_curve, biquads, mean_spl, arrival_time_ms)
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn process_mixed_mode_crossover(
    channel_name: &str,
    curve: &Curve,
    room_config: &RoomConfig,
    mixed_config: &MixedModeConfig,
    sample_rate: f64,
    output_dir: &Path,
    min_freq: f64,
    max_freq: f64,
    mean: f64,
    pre_score: f64,
    arrival_time_ms: Option<f64>,
    callback: Option<crate::optim::OptimProgressCallback>,
) -> Result<MixedModeResult> {
    let crossover_freq = mixed_config.crossover_freq;
    let fir_uses_low = mixed_config.fir_band.to_lowercase() == "low";

    info!(
        "  Mixed mode crossover at {} Hz (FIR on {} band, IIR on {} band)",
        crossover_freq,
        if fir_uses_low { "low" } else { "high" },
        if fir_uses_low { "high" } else { "low" }
    );

    // Split the curve at crossover frequency
    let (low_curve, high_curve) = split_curve_at_frequency(curve, crossover_freq);

    // Determine which curve gets FIR and which gets IIR
    let (fir_curve, iir_curve) = if fir_uses_low {
        (&low_curve, &high_curve)
    } else {
        (&high_curve, &low_curve)
    };

    // Create band-specific optimizer configs with appropriate frequency ranges
    let fir_min_freq = fir_curve.freq.first().copied().unwrap_or(min_freq);
    let fir_max_freq = fir_curve.freq.last().copied().unwrap_or(crossover_freq);
    let iir_min_freq = iir_curve.freq.first().copied().unwrap_or(crossover_freq);
    let iir_max_freq = iir_curve.freq.last().copied().unwrap_or(max_freq);

    info!(
        "  FIR band: {:.1}-{:.1} Hz, IIR band: {:.1}-{:.1} Hz",
        fir_min_freq, fir_max_freq, iir_min_freq, iir_max_freq
    );

    // Optimize IIR on designated band
    let iir_config = OptimizerConfig {
        min_freq: iir_min_freq,
        max_freq: iir_max_freq,
        ..room_config.optimizer.clone()
    };

    let eq_result = if let Some(cb) = callback {
        eq::optimize_channel_eq_with_callback_detailed(
            iir_curve,
            &iir_config,
            room_config.target_curve.as_ref(),
            sample_rate,
            cb,
        )
    } else {
        eq::optimize_channel_eq_detailed(
            iir_curve,
            &iir_config,
            room_config.target_curve.as_ref(),
            sample_rate,
        )
    }
    .map_err(|e| AutoeqError::OptimizationFailed {
        message: format!(
            "IIR optimization failed for {} band: {}",
            if fir_uses_low { "high" } else { "low" },
            e
        ),
    })?;
    let eq_filters = eq_result.filters;
    let optimizer_evidence = eq_result.optimizer_evidence;

    info!(
        "  IIR stage: {} filters for {} band",
        eq_filters.len(),
        if fir_uses_low { "high" } else { "low" }
    );

    // Optimize FIR on designated band
    let fir_config = OptimizerConfig {
        min_freq: fir_min_freq,
        max_freq: fir_max_freq,
        ..room_config.optimizer.clone()
    };

    let fir_coeffs = fir::generate_fir_correction(
        fir_curve,
        &fir_config,
        room_config.target_curve.as_ref(),
        sample_rate,
    )
    .map_err(|e| AutoeqError::OptimizationFailed {
        message: format!(
            "FIR generation failed for {} band: {}",
            if fir_uses_low { "low" } else { "high" },
            e
        ),
    })?;

    // Save FIR to WAV
    let (fir_filename, wav_path) = artifacts::reserve_convolution_artifact_path(
        output_dir,
        channel_name,
        ConvolutionArtifactKind::BandFir,
        sample_rate,
    );
    crate::fir::save_fir_to_wav(&fir_coeffs, sample_rate as u32, &wav_path).map_err(|e| {
        AutoeqError::OptimizationFailed {
            message: format!("Failed to save FIR WAV: {}", e),
        }
    })?;

    info!("  Saved FIR filter to {}", wav_path.display());

    // Build DSP chain with band split/merge
    let mut chain = output::build_mixed_mode_crossover_chain(
        channel_name,
        mixed_config,
        &eq_filters,
        &fir_filename,
        fir_uses_low,
        None,
    );

    // Compute combined response for scoring
    // For proper scoring, we need to simulate what the full chain does:
    // - Split into bands at crossover
    // - Apply FIR to one band, IIR to the other
    // - Sum bands back together
    let iir_resp = response::compute_peq_complex_response(&eq_filters, &curve.freq, sample_rate);
    let fir_resp = response::compute_fir_complex_response(&fir_coeffs, &curve.freq, sample_rate);

    // Compute crossover filter responses (LR24 = 4th order Butterworth)
    let (lp_resp, hp_resp) =
        compute_lr24_crossover_responses(&curve.freq, crossover_freq, sample_rate);

    // Combine responses: low_band * fir_or_iir + high_band * iir_or_fir
    let combined_resp: Vec<num_complex::Complex<f64>> = curve
        .freq
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if fir_uses_low {
                lp_resp[i] * fir_resp[i] + hp_resp[i] * iir_resp[i]
            } else {
                lp_resp[i] * iir_resp[i] + hp_resp[i] * fir_resp[i]
            }
        })
        .collect();

    let final_curve = response::apply_complex_response(curve, &combined_resp);

    // Detect passband for normalization
    let (norm_range, mean_final) = detect_passband_and_mean(&final_curve);

    // Compute post-score
    let normalized_final_spl = &final_curve.spl - mean_final;
    let post_score =
        crate::loss::flat_loss(&final_curve.freq, &normalized_final_spl, min_freq, max_freq);

    info!(
        "  Pre-score: {:.6}, Post-score: {:.6}",
        pre_score, post_score
    );

    // Extend curves to 20 Hz – 20 kHz for display output
    let display_initial = output::extend_curve_to_full_range(curve);
    let display_iir_resp =
        response::compute_peq_complex_response(&eq_filters, &display_initial.freq, sample_rate);
    let display_fir_resp =
        response::compute_fir_complex_response(&fir_coeffs, &display_initial.freq, sample_rate);
    let (display_lp, display_hp) =
        compute_lr24_crossover_responses(&display_initial.freq, crossover_freq, sample_rate);
    let display_combined: Vec<num_complex::Complex<f64>> = display_initial
        .freq
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if fir_uses_low {
                display_lp[i] * display_fir_resp[i] + display_hp[i] * display_iir_resp[i]
            } else {
                display_lp[i] * display_iir_resp[i] + display_hp[i] * display_fir_resp[i]
            }
        })
        .collect();
    let display_final = response::apply_complex_response(&display_initial, &display_combined);

    let mut initial_data: super::super::types::CurveData = (&display_initial).into();
    initial_data.norm_range = norm_range;
    let mut final_data: super::super::types::CurveData = (&display_final).into();
    final_data.norm_range = norm_range;

    chain.initial_curve = Some(initial_data.clone());
    chain.final_curve = Some(final_data.clone());
    chain.eq_response = Some(output::compute_eq_response(&initial_data, &final_data));

    Ok((
        chain,
        pre_score,
        post_score,
        curve.clone(),
        final_curve,
        eq_filters,
        mean,
        arrival_time_ms,
        Some(fir_coeffs),
        optimizer_evidence,
    ))
}

/// Process Gradient Cardioid configuration
///
/// Returns: (DSP chain, pre_score, post_score, initial_curve, final_curve, biquads, mean_spl, arrival_time_ms)
pub(in super::super) fn process_cardioid(
    channel_name: &str,
    config: &super::super::types::CardioidConfig,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &Path,
) -> Result<MixedModeResult> {
    // 1. Load measurements
    let front_curve =
        load::load_source(&config.front).map_err(|e| AutoeqError::InvalidMeasurement {
            message: format!("Failed to load Front measurement: {}", e),
        })?;
    let rear_curve =
        load::load_source(&config.rear).map_err(|e| AutoeqError::InvalidMeasurement {
            message: format!("Failed to load Rear measurement: {}", e),
        })?;
    if front_curve.phase.is_none() || rear_curve.phase.is_none() {
        return Err(AutoeqError::InvalidMeasurement {
            message: "Cardioid processing requires measured phase for front and rear drivers"
                .to_string(),
        });
    }
    let rear_curve =
        if super::super::frequency_grid::same_frequency_grid(&front_curve.freq, &rear_curve.freq) {
            rear_curve
        } else {
            crate::read::interpolate_log_space(&front_curve.freq, &rear_curve)
        };

    // 2. Calculate Delay
    let delay_ms = config.separation_meters / 343.0 * 1000.0;
    info!(
        "  Cardioid: Separation {:.2}m -> Delay {:.2}ms",
        config.separation_meters, delay_ms
    );

    // 3. Simulate Combined Response
    use num_complex::Complex;
    let n_points = front_curve.freq.len();
    let mut combined_complex = Vec::with_capacity(n_points);
    let front_phase =
        front_curve
            .phase
            .as_ref()
            .ok_or_else(|| AutoeqError::InvalidMeasurement {
                message: "Cardioid front phase missing after validation".to_string(),
            })?;
    let rear_phase = rear_curve
        .phase
        .as_ref()
        .ok_or_else(|| AutoeqError::InvalidMeasurement {
            message: "Cardioid rear phase missing after interpolation".to_string(),
        })?;

    for i in 0..n_points {
        let f = front_curve.freq[i];
        let omega = 2.0 * std::f64::consts::PI * f;

        // Front
        let f_mag = 10.0_f64.powf(front_curve.spl[i] / 20.0);
        let f_phi = front_phase[i].to_radians();
        let f_c = Complex::from_polar(f_mag, f_phi);

        // Rear (Inverted + Delayed)
        let r_mag = 10.0_f64.powf(rear_curve.spl[i] / 20.0);
        let r_phi_meas = rear_phase[i].to_radians();

        // Delay phase shift: -omega * delay
        let delay_s = delay_ms / 1000.0;
        let delay_phi = -omega * delay_s;

        // Inversion: +180 deg (PI rad)
        let invert_phi = std::f64::consts::PI;

        let r_phi_total = r_phi_meas + delay_phi + invert_phi;
        let r_c = Complex::from_polar(r_mag, r_phi_total);

        let sum = f_c + r_c;
        combined_complex.push(sum);
    }

    let combined_curve = Curve {
        freq: front_curve.freq.clone(),
        spl: ndarray::Array1::from_iter(
            combined_complex
                .iter()
                .map(|z| 20.0 * z.norm().max(1e-12).log10()),
        ),
        phase: Some(ndarray::Array1::from_iter(
            combined_complex.iter().map(|z| z.arg().to_degrees()),
        )),
        ..Default::default()
    };

    // 4. Optimize EQ
    let min_freq = room_config.optimizer.min_freq;
    let max_freq = room_config.optimizer.max_freq;
    let pre_score = flat_loss_score(&combined_curve, min_freq, max_freq);

    let eq_result = eq::optimize_channel_eq_detailed(
        &combined_curve,
        &room_config.optimizer,
        room_config.target_curve.as_ref(),
        sample_rate,
    )
    .map_err(|e| AutoeqError::OptimizationFailed {
        message: format!("EQ optimization failed for Cardioid sum: {}", e),
    })?;
    let eq_filters = eq_result.filters;
    let post_score = eq_result.loss;
    let mut optimizer_evidence = eq_result.optimizer_evidence;

    let (eq_filters, post_score, final_curve) = if eq_score_regressed(pre_score, post_score) {
        for evidence in &mut optimizer_evidence {
            evidence.selected_for_output = false;
        }
        warn!(
            "  Global EQ rejected for Cardioid sum {}: flat loss {:.6} -> {:.6}",
            channel_name, pre_score, post_score
        );
        (Vec::new(), pre_score, combined_curve.clone())
    } else {
        info!(
            "  Global EQ: {} filters, pre={:.6}, post={:.6}",
            eq_filters.len(),
            pre_score,
            post_score
        );
        let eq_resp =
            response::compute_peq_complex_response(&eq_filters, &combined_curve.freq, sample_rate);
        let final_curve = response::apply_complex_response(&combined_curve, &eq_resp);
        (eq_filters, post_score, final_curve)
    };

    // 5. Build Output Chain
    // Prepare display curves
    let driver_curves_for_display = vec![
        output::extend_curve_to_full_range(&front_curve),
        output::extend_curve_to_full_range(&rear_curve),
    ];

    let mut chain = output::build_cardioid_dsp_chain_with_curves(
        channel_name,
        &[0.0, 0.0],      // Gains (0 for now)
        &[0.0, delay_ms], // Delays
        &eq_filters,
        None,
        None,
        Some(&driver_curves_for_display),
    );

    // Populate initial/final curves in chain
    let (norm_range, _) = detect_passband_and_mean(&combined_curve);
    let display_initial = output::extend_curve_to_full_range(&combined_curve);
    let display_resp =
        response::compute_peq_complex_response(&eq_filters, &display_initial.freq, sample_rate);
    let display_final = response::apply_complex_response(&display_initial, &display_resp);

    let mut initial_data: super::super::types::CurveData = (&display_initial).into();
    initial_data.norm_range = norm_range;
    let mut final_data: super::super::types::CurveData = (&display_final).into();
    final_data.norm_range = norm_range;

    chain.initial_curve = Some(initial_data.clone());
    chain.final_curve = Some(final_data.clone());
    chain.eq_response = Some(output::compute_eq_response(&initial_data, &final_data));

    // Mean SPL
    let freqs_f32: Vec<f32> = combined_curve.freq.iter().map(|&f| f as f32).collect();
    let spl_f32: Vec<f32> = combined_curve.spl.iter().map(|&s| s as f32).collect();
    let mean_spl = compute_average_response(
        &freqs_f32,
        &spl_f32,
        Some((min_freq as f32, max_freq as f32)),
    ) as f64;

    Ok((
        chain,
        pre_score,
        post_score,
        combined_curve,
        final_curve,
        eq_filters,
        mean_spl,
        None,
        None, // IIR-only for cardioid
        optimizer_evidence,
    ))
}
