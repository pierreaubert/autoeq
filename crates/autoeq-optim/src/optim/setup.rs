//! Shared optimization setup helpers
//!
//! Functions for preparing objective data, computing parameter bounds, building
//! initial guesses, and orchestrating optimization runs. These are used by both
//! the CLI workflows in [`crate::workflow`] and by the room EQ subsystem.

use super::ObjectiveData;
use crate::AutoeqError;
use crate::Curve;
use crate::HeadphoneLossData;
use crate::SpeakerLossData;
use crate::loss::DriversLossData;
use std::collections::HashMap;

mod drivers;
mod misc;
mod perform;
mod progress_callback_config;
mod types;

pub use drivers::*;
pub use misc::*;
pub use perform::*;
pub use progress_callback_config::*;
pub use types::*;

/// Prepare the ObjectiveData and whether CEA2034-based scoring is active.
///
/// # Errors
///
/// Returns `AutoeqError::MissingCea2034Curve` if spin_data is provided but missing required curves.
/// Returns `AutoeqError::CurveLengthMismatch` if spin_data curves have inconsistent lengths.
pub fn setup_objective_data(
    params: &crate::OptimParams,
    input_curve: &Curve,
    target_curve: &Curve,
    deviation_curve: &Curve,
    spin_data: &Option<HashMap<String, Curve>>,
) -> Result<(ObjectiveData, bool), AutoeqError> {
    // CEA2034 data is available if spin_data was provided.
    // This can happen either via:
    // 1. CLI path: args.measurement=CEA2034 and args.speaker/version are set
    // 2. Library path: spin_data passed directly from API fetch
    // The key requirement is just having spin_data available.
    let use_cea = spin_data.is_some();

    let speaker_score_data_opt = if let Some(spin) = spin_data {
        Some(SpeakerLossData::try_new(spin)?)
    } else {
        None
    };

    // Headphone score data is available when NOT using CEA2034 speaker data
    let headphone_score_data_opt = if !use_cea {
        Some(HeadphoneLossData::new(params.smooth, params.smooth_n))
    } else {
        None
    };

    let mut builder = crate::optim::ObjectiveDataBuilder::new(
        input_curve.freq.clone(),
        target_curve.spl.clone(),
        deviation_curve.spl.clone(), // This is the deviation to be corrected
        params.sample_rate,
        params.peq_model,
        params.loss,
    )
    .min_spacing_oct(params.min_spacing_oct)
    .spacing_weight(params.spacing_weight)
    .max_db(params.max_db)
    .min_db(params.min_db)
    .freq_range(params.min_freq, params.max_freq)
    .smoothing(params.smooth, params.smooth_n)
    .smoothness_penalty_opt(params.smoothness_penalty.clone())
    .audibility_deadband_opt(params.audibility_deadband);

    if let Some(sd) = speaker_score_data_opt {
        builder = builder.speaker_score_data(sd);
    }
    if let Some(hd) = headphone_score_data_opt {
        builder = builder.headphone_score_data(hd);
    }
    if !use_cea {
        builder = builder.input_curve(input_curve.clone());
    }

    let objective_data = builder.build()?;

    Ok((objective_data, use_cea))
}

/// Set up objective data for multi-driver crossover optimization
///
/// # Arguments
/// * `params` - Optimization parameters
/// * `drivers_data` - Multi-driver measurement data
///
/// # Returns
/// * ObjectiveData configured for multi-driver optimization
pub fn setup_drivers_objective_data(
    params: &crate::OptimParams,
    drivers_data: DriversLossData,
) -> ObjectiveData {
    crate::optim::ObjectiveDataBuilder::drivers_flat(
        drivers_data.freq_grid.clone(),
        params.sample_rate,
        params.peq_model,
        drivers_data,
    )
    .max_db(params.max_db)
    .min_db(params.min_db)
    .freq_range(params.min_freq, params.max_freq)
    .smoothness_penalty_opt(params.smoothness_penalty.clone())
    .audibility_deadband_opt(params.audibility_deadband)
    .build()
    .expect("invariant: drivers_flat builder has all required data")
}

/// Build optimization parameter bounds for multi-driver crossover optimization
///
/// # Arguments
/// * `params` - Optimization parameters
/// * `drivers_data` - Multi-driver measurement data
///
/// # Returns
/// * Tuple of (lower_bounds, upper_bounds)
///
/// # Parameter Vector Layout
/// For N drivers: [gain1, gain2, ..., gainN, xover_freq1, xover_freq2, ..., xover_freq(N-1)]
/// - Gains are in dB, bounded by [-max_db, max_db]
/// - Crossover frequencies are in Hz (log10 space), bounded by driver frequency ranges
pub fn setup_drivers_bounds(
    params: &crate::OptimParams,
    drivers_data: &DriversLossData,
) -> (Vec<f64>, Vec<f64>) {
    let n_drivers = drivers_data.drivers.len();
    let n_params = n_drivers * 2 + (n_drivers - 1); // N gains + N delays + (N-1) crossovers

    let mut lower_bounds = Vec::with_capacity(n_params);
    let mut upper_bounds = Vec::with_capacity(n_params);

    // Bounds for gains: [-max_db, max_db]
    for _ in 0..n_drivers {
        lower_bounds.push(-params.max_db);
        upper_bounds.push(params.max_db);
    }

    // Bounds for delays: [-20.0, 20.0] ms
    for _ in 0..n_drivers {
        lower_bounds.push(-20.0);
        upper_bounds.push(20.0);
    }

    // Bounds for crossover frequencies
    // Each crossover should be between the mean frequencies of adjacent drivers
    for i in 0..(n_drivers - 1) {
        let driver_low = &drivers_data.drivers[i];
        let driver_high = &drivers_data.drivers[i + 1];

        // Use geometric mean frequencies as characteristic frequencies of each driver
        let mean_low = driver_low.mean_freq();
        let mean_high = driver_high.mean_freq();

        // Crossover should be between the geometric means with reasonable margin
        // Use log10 space for better optimization
        // Allow range from 0.5x mean_low to 2x mean_high, centered on geometric mean of the two means
        let geometric_center = (mean_low * mean_high).sqrt();
        let xover_min = (geometric_center * 0.5).max(params.min_freq).log10();
        let xover_max = (geometric_center * 2.0).min(params.max_freq).log10();

        // Ensure bounds are valid
        let xover_min = xover_min.min(xover_max - 0.1);

        lower_bounds.push(xover_min);
        upper_bounds.push(xover_max);
    }

    (lower_bounds, upper_bounds)
}

/// Build optimization parameter bounds for multi-driver optimization with fixed crossover frequencies
///
/// When crossover frequencies are fixed, we only optimize gains and delays.
///
/// # Arguments
/// * `params` - Optimization parameters
/// * `drivers_data` - Multi-driver measurement data
///
/// # Returns
/// * Tuple of (lower_bounds, upper_bounds)
///
/// # Parameter Vector Layout
/// For N drivers: [gain1, gain2, ..., gainN, delay1, delay2, ..., delayN]
/// - Gains are in dB, bounded by [-max_db, max_db]
/// - Delays are in ms, bounded by [-5, 5]
pub fn setup_drivers_bounds_fixed_freqs(
    params: &crate::OptimParams,
    drivers_data: &DriversLossData,
) -> (Vec<f64>, Vec<f64>) {
    let n_drivers = drivers_data.drivers.len();
    let n_params = n_drivers * 2; // N gains + N delays (no crossovers)

    let mut lower_bounds = Vec::with_capacity(n_params);
    let mut upper_bounds = Vec::with_capacity(n_params);

    // Bounds for gains: [-max_db, max_db]
    for _ in 0..n_drivers {
        lower_bounds.push(-params.max_db);
        upper_bounds.push(params.max_db);
    }

    // Bounds for delays: [-20.0, 20.0] ms
    for _ in 0..n_drivers {
        lower_bounds.push(-20.0);
        upper_bounds.push(20.0);
    }

    (lower_bounds, upper_bounds)
}

/// Resolve the `(LS, HS)` hinge bands (Hz) for the `PkLsHs` tilt pair.
///
/// Explicit bands win; each missing band falls back to its geometric half of
/// the correction band. Callers clamp the result into the band.
fn tilt_hinge_bands(params: &crate::OptimParams) -> ([f64; 2], [f64; 2]) {
    let mid = (params.min_freq * params.max_freq).sqrt();
    let bands = params.tilt_bands_hz;
    let ls = bands.and_then(|b| b.ls).unwrap_or([params.min_freq, mid]);
    let hs = bands.and_then(|b| b.hs).unwrap_or([mid, params.max_freq]);
    (ls, hs)
}

/// Build optimization parameter bounds for the optimizer.
pub fn setup_bounds(params: &crate::OptimParams) -> (Vec<f64>, Vec<f64>) {
    use crate::PeqModel;

    let model = params.peq_model;
    let ppf = crate::param_utils::params_per_filter(model);
    let num_params = params.num_filters * ppf;
    let mut lower_bounds = Vec::with_capacity(num_params);
    let mut upper_bounds = Vec::with_capacity(num_params);

    let spacing = 1.0; // Overlap factor - allows adjacent filters to overlap
    let gain_lower = if params.min_db < 0.0 {
        params.min_db
    } else {
        -3.0 * params.max_db
    };
    let q_lower = params.min_q.max(0.1);
    // PkLsHs reserves the trailing groups for the tilt pair; the progressive
    // PK banding only spans the leading groups so enabling tilt never moves
    // the main filters' bounds.
    let n_tilt = match model {
        PeqModel::PkLsHs => params.num_filters.min(2),
        _ => 0,
    };
    let n_main = params.num_filters - n_tilt;
    let range = if n_main > 0 {
        (params.max_freq.log10() - params.min_freq.log10()) / (n_main as f64)
    } else {
        params.max_freq.log10() - params.min_freq.log10()
    };

    for i in 0..n_main {
        // Center frequency for this filter in log space
        let f_center = params.min_freq.log10() + (i as f64) * range;

        // Calculate bounds with overlap
        // Each filter can range from (center - spacing*range) to (center + spacing*range)
        let f_low = (f_center - spacing * range).max(params.min_freq.log10());
        let f_high = (f_center + spacing * range).min(params.max_freq.log10());

        // Ensure progressive increase: each filter's lower bound should be >= previous filter's lower bound
        let f_low_adjusted = if i > 0 {
            // Get the frequency lower bound of the previous filter
            let prev_freq_idx = if ppf == 3 {
                (i - 1) * 3
            } else {
                (i - 1) * 4 + 1
            };
            f_low.max(lower_bounds[prev_freq_idx])
        } else {
            f_low
        };

        // Ensure upper bound is also progressive (but can overlap)
        let f_high_adjusted = if i > 0 {
            let prev_freq_idx = if ppf == 3 {
                (i - 1) * 3
            } else {
                (i - 1) * 4 + 1
            };
            f_high.max(upper_bounds[prev_freq_idx])
        } else {
            f_high
        };

        // Ensure lower bound never exceeds upper bound (can happen when
        // progressive adjustment pushes f_low past f_high with many filters
        // in a narrow range).
        let f_high_adjusted = f_high_adjusted.max(f_low_adjusted);

        // Add bounds based on model type.  A frequency-dependent policy is
        // resolved for the complete rectangular frequency interval so a
        // filter cannot move into a guarded band with an unsafe Q.
        let q_upper = q_max_for_frequency_range(
            params,
            10.0_f64.powf(f_low_adjusted),
            10.0_f64.powf(f_high_adjusted),
        );
        match model {
            PeqModel::Pk
            | PeqModel::HpPk
            | PeqModel::HpPkLp
            | PeqModel::LsPk
            | PeqModel::LsPkHs
            | PeqModel::PkLsHs => {
                // Fixed filter types: [freq, Q, gain]
                lower_bounds.extend_from_slice(&[f_low_adjusted, q_lower, gain_lower]);
                upper_bounds.extend_from_slice(&[f_high_adjusted, q_upper, params.max_db]);
            }
            PeqModel::FreePkFree | PeqModel::Free => {
                // Free filter types: [type, freq, Q, gain]
                let (type_low, type_high) = if model == PeqModel::Free
                    || (model == PeqModel::FreePkFree && (i == 0 || i == params.num_filters - 1))
                {
                    crate::param_utils::filter_type_bounds()
                } else {
                    (0.0, 0.999) // Peak filter only
                };
                lower_bounds.extend_from_slice(&[type_low, f_low_adjusted, q_lower, gain_lower]);
                upper_bounds.extend_from_slice(&[
                    type_high,
                    f_high_adjusted,
                    q_upper,
                    params.max_db,
                ]);
            }
        }
    }

    // Apply model-specific constraints
    match model {
        PeqModel::HpPk | PeqModel::HpPkLp => {
            // First filter is highpass - fixed 3-param layout
            lower_bounds[0] = 20.0_f64.max(params.min_freq).log10();
            upper_bounds[0] = 120.0_f64.min(params.min_freq + 20.0).log10();
            lower_bounds[1] = 1.0_f64.max(q_lower).min(params.max_q);
            upper_bounds[1] = 1.5_f64.max(lower_bounds[1]).min(params.max_q);
            lower_bounds[2] = 0.0;
            upper_bounds[2] = 0.0;
        }
        PeqModel::LsPk | PeqModel::LsPkHs => {
            // First filter is low shelves - fixed 3-param layout
            lower_bounds[0] = 20.0_f64.max(params.min_freq).log10();
            upper_bounds[0] = 120.0_f64.min(params.min_freq + 20.0).log10();
            // Standard RBJ shelves in the DSP core do not use Q.  Pin the
            // serialized value so the optimizer does not spend a dead
            // dimension or imply a slope control that is not implemented.
            let shelf_q = 1.0_f64.clamp(params.min_q, params.max_q);
            lower_bounds[1] = shelf_q;
            upper_bounds[1] = shelf_q;
            lower_bounds[2] = -params.max_db;
            upper_bounds[2] = params.max_db;
        }
        _ => {}
    }

    if params.num_filters > 1 {
        if matches!(model, PeqModel::HpPkLp) {
            // Last filter is lowpass - fixed 3-param layout
            let last_idx = (params.num_filters - 1) * ppf;
            if ppf == 3 {
                lower_bounds[last_idx] = (params.max_freq - 2000.0).max(5000.0).log10();
                upper_bounds[last_idx] = params.max_freq.log10();
                lower_bounds[last_idx + 1] = 1.0_f64.max(q_lower).min(params.max_q);
                upper_bounds[last_idx + 1] =
                    1.5_f64.max(lower_bounds[last_idx + 1]).min(params.max_q);
                lower_bounds[last_idx + 2] = 0.0;
                upper_bounds[last_idx + 2] = 0.0;
            }
        }

        if matches!(model, PeqModel::LsPkHs) {
            // Last filter is lowpass - fixed 3-param layout
            let last_idx = (params.num_filters - 1) * ppf;
            if ppf == 3 {
                lower_bounds[last_idx] = (params.max_freq - 2000.0).max(5000.0).log10();
                upper_bounds[last_idx] = params.max_freq.log10();
                let shelf_q = 1.0_f64.clamp(params.min_q, params.max_q);
                lower_bounds[last_idx + 1] = shelf_q;
                upper_bounds[last_idx + 1] = shelf_q;
                lower_bounds[last_idx + 2] = -params.max_db;
                upper_bounds[last_idx + 2] = params.max_db;
            }
        }
    }

    // Trailing tilt pair for PkLsHs: fixed LS/HS triplets with free hinges.
    // Gains follow the shelf convention (+/-max_db); Q is pinned because the
    // DSP core's RBJ shelves do not use it. The clamp loop below keeps explicit
    // hinge bands inside the correction band.
    if n_tilt > 0 {
        let shelf_q = 1.0_f64.clamp(params.min_q, params.max_q);
        let (ls_band, hs_band) = tilt_hinge_bands(params);
        // The degenerate single-group layout keeps only the treble shelf.
        if n_tilt == 2 {
            lower_bounds.extend_from_slice(&[ls_band[0].log10(), shelf_q, -params.max_db]);
            upper_bounds.extend_from_slice(&[ls_band[1].log10(), shelf_q, params.max_db]);
        }
        lower_bounds.extend_from_slice(&[hs_band[0].log10(), shelf_q, -params.max_db]);
        upper_bounds.extend_from_slice(&[hs_band[1].log10(), shelf_q, params.max_db]);
    }

    // Model-specific fixed HP/LP/shelf anchors must not escape the measured
    // optimization band. Clamp both ends before repairing any collapsed range
    // so every candidate remains meaningful for the available data.
    let min_log_freq = params.min_freq.log10();
    let max_log_freq = params.max_freq.log10();
    for i in 0..params.num_filters {
        let freq_idx = if ppf == 3 { i * ppf } else { i * ppf + 1 };
        lower_bounds[freq_idx] = lower_bounds[freq_idx].clamp(min_log_freq, max_log_freq);
        upper_bounds[freq_idx] = upper_bounds[freq_idx].clamp(min_log_freq, max_log_freq);
        if lower_bounds[freq_idx] > upper_bounds[freq_idx] {
            upper_bounds[freq_idx] = lower_bounds[freq_idx];
        }
    }

    // Debug: Display bounds for each filter (unless in QA mode)
    if !params.quiet {
        log::info!("\n📏 Parameter Bounds (Model: {}):", model);
        log::info!("+----+-------------------+---------------+-----------------+--------+");
        log::info!("|  # | Freq Range (Hz)   | Q Range       | Gain Range (dB) | Type   |");
        log::info!("+----+-------------------+---------------+-----------------+--------+");
        for i in 0..params.num_filters {
            let offset = i * ppf;
            let (freq_idx, q_idx, gain_idx) = if ppf == 3 {
                (offset, offset + 1, offset + 2)
            } else {
                (offset + 1, offset + 2, offset + 3)
            };
            let freq_low_hz = 10f64.powf(lower_bounds[freq_idx]);
            let freq_high_hz = 10f64.powf(upper_bounds[freq_idx]);
            let q_low = lower_bounds[q_idx];
            let q_high = upper_bounds[q_idx];
            let gain_low = lower_bounds[gain_idx];
            let gain_high = upper_bounds[gain_idx];

            let filter_type = match model {
                PeqModel::Pk => "PK",
                PeqModel::HpPk if i == 0 => "HP",
                PeqModel::HpPk => "PK",
                PeqModel::HpPkLp if i == 0 => "HP",
                PeqModel::HpPkLp if i == params.num_filters - 1 => "LP",
                PeqModel::HpPkLp => "PK",
                PeqModel::LsPk if i == 0 => "LS",
                PeqModel::LsPk => "PK",
                PeqModel::LsPkHs if i == 0 => "LS",
                PeqModel::LsPkHs if i == params.num_filters - 1 => "HS",
                PeqModel::LsPkHs => "PK",
                PeqModel::PkLsHs if params.num_filters >= 2 && i == params.num_filters - 2 => "LS",
                PeqModel::PkLsHs if i == params.num_filters - 1 => "HS",
                PeqModel::PkLsHs => "PK",
                PeqModel::FreePkFree if i == 0 || i == params.num_filters - 1 => "??",
                PeqModel::FreePkFree => "PK",
                PeqModel::Free => "??",
            };

            log::info!(
                "| {:2} | {:7.1} - {:7.1} | {:5.2} - {:5.2} | {:+6.2} - {:+6.2} | {:6} |",
                i + 1,
                freq_low_hz,
                freq_high_hz,
                q_low,
                q_high,
                gain_low,
                gain_high,
                filter_type
            );
        }
        log::info!("+----+-------------------+---------------+-----------------+--------+\n");
    }

    (lower_bounds, upper_bounds)
}

/// Resolve the safest Q ceiling for a filter whose rectangular frequency
/// bounds are `[f_low_hz, f_high_hz]`.
///
/// A PEQ parameter vector cannot express a coupled `Q(frequency)` constraint.
/// Applying the ceiling to the whole interval is therefore deliberately
/// conservative: any candidate that remains inside the optimizer bounds is
/// safe even when its centre frequency moves to the guarded edge.  The
/// transition rule keeps high-Q modal freedom below Schroeder while using the
/// upper-band ceiling for intervals that cross the transition.
pub fn q_max_for_frequency_range(
    params: &crate::OptimParams,
    f_low_hz: f64,
    f_high_hz: f64,
) -> f64 {
    let mut upper = params.max_q;
    let high = f_low_hz.max(f_high_hz);
    if let Some(policy) = params.frequency_q_policy {
        if let Some(transition) = policy
            .schroeder_hz
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            let low_max_q = policy
                .low_max_q
                .filter(|value| value.is_finite() && *value > 0.0);
            let high_max_q = policy
                .high_max_q
                .filter(|value| value.is_finite() && *value > 0.0);

            // A rectangular PEQ bound cannot express Q as a function of
            // centre frequency. Use a log-frequency blend at the upper edge
            // of the interval instead of a discontinuous Schroeder switch.
            // The guarded upper band is reached at `high_start_hz`, or one
            // octave above transition when no explicit start is configured.
            let blend_end = policy
                .high_start_hz
                .filter(|value| value.is_finite() && *value > transition)
                .unwrap_or(transition * 2.0);
            if let (Some(low_q), Some(high_q)) = (low_max_q, high_max_q) {
                if high <= transition {
                    upper = upper.min(low_q);
                } else if high >= blend_end {
                    upper = upper.min(high_q);
                } else {
                    let t = (high / transition).log2() / (blend_end / transition).log2();
                    let t = t.clamp(0.0, 1.0);
                    // Smoothstep avoids a slope discontinuity at the modal /
                    // upper-band boundary while remaining monotone.
                    let t = t * t * (3.0 - 2.0 * t);
                    upper = upper.min(low_q + (high_q - low_q) * t);
                }
            } else if high <= transition {
                if let Some(low_q) = low_max_q {
                    upper = upper.min(low_q);
                }
            } else if let Some(high_q) = high_max_q {
                upper = upper.min(high_q);
            }
        }
        if let Some(start) = policy
            .high_start_hz
            .filter(|value| value.is_finite() && *value > 0.0)
            && high >= start
            && let Some(high_max_q) = policy
                .high_max_q
                .filter(|value| value.is_finite() && *value > 0.0)
        {
            upper = upper.min(high_max_q);
        }
    }
    upper.max(params.min_q.max(0.1))
}

/// Set up objective data for multi-subwoofer optimization.
///
/// Creates the objective function configuration for optimizing gain and delay
/// parameters across multiple subwoofers to achieve a flat combined response.
///
/// # Arguments
///
/// * `params` - Optimization parameters with optimization parameters
/// * `drivers_data` - Multi-driver measurement and configuration data
pub fn setup_multisub_objective_data(
    params: &crate::OptimParams,
    drivers_data: DriversLossData,
) -> ObjectiveData {
    crate::optim::ObjectiveDataBuilder::multi_sub_flat(
        drivers_data.freq_grid.clone(),
        params.sample_rate,
        params.peq_model,
        drivers_data,
    )
    .max_db(params.max_db)
    .min_db(params.min_db)
    .freq_range(params.min_freq, params.max_freq)
    .smoothness_penalty_opt(params.smoothness_penalty.clone())
    .audibility_deadband_opt(params.audibility_deadband)
    .build()
    .expect("invariant: multi-sub-flat builder has all required data")
}

/// Set up parameter bounds for multi-subwoofer optimization.
///
/// Creates lower and upper bounds for gain and delay parameters.
/// Gains are bounded by `[-max_db, max_db]` and delays by `[0, 20]` ms.
///
/// # Arguments
///
/// * `params` - Optimization parameters with `max_db` setting
/// * `n_drivers` - Number of subwoofers
///
/// # Returns
///
/// Tuple of (lower_bounds, upper_bounds) vectors.
pub fn setup_multisub_bounds(
    params: &crate::OptimParams,
    n_drivers: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_params = n_drivers * 2; // gains + delays
    let mut lower_bounds = Vec::with_capacity(n_params);
    let mut upper_bounds = Vec::with_capacity(n_params);

    // Gains
    for _ in 0..n_drivers {
        lower_bounds.push(-params.max_db);
        upper_bounds.push(params.max_db);
    }

    // Delays (0 to 20ms)
    for _ in 0..n_drivers {
        lower_bounds.push(0.0);
        upper_bounds.push(20.0);
    }

    (lower_bounds, upper_bounds)
}
