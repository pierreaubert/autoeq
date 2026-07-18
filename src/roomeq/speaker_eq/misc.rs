use super::super::types::RoomConfig;

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
