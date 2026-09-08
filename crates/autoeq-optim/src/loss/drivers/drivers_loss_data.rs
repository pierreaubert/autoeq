use super::super::flat::flat_loss;
use super::compute::compute_drivers_combined_response;
use super::crossover_type::CrossoverType;
use super::driver_measurement::DriverMeasurement;
use ndarray::Array1;

/// Data required for multi-driver crossover optimization
#[derive(Debug, Clone)]
pub struct DriversLossData {
    /// Measurements for each driver (sorted by frequency range, lowest first)
    pub drivers: Vec<DriverMeasurement>,
    /// Crossover type to use between driver pairs
    pub crossover_type: CrossoverType,
    /// Common frequency grid for evaluation
    pub freq_grid: Array1<f64>,
    /// Calibrated power reference, cached outside optimizer evaluations.
    pub power_reference: Array1<f64>,
}

impl DriversLossData {
    /// Create a new DriversLossData instance
    ///
    /// # Arguments
    /// * `drivers` - Vector of driver measurements (will be sorted by frequency)
    /// * `crossover_type` - Type of crossover filter to use
    pub fn new(mut drivers: Vec<DriverMeasurement>, crossover_type: CrossoverType) -> Self {
        let max_drivers = if crossover_type == CrossoverType::None {
            usize::MAX
        } else {
            4
        };
        assert!(
            drivers.len() >= 2 && drivers.len() <= max_drivers,
            "Must have at least 2 drivers and at most 4 when using crossovers, got {}",
            drivers.len()
        );

        // Sort drivers by their mean frequency (woofer -> midrange -> tweeter)
        drivers.sort_by(|a, b| {
            a.mean_freq()
                .partial_cmp(&b.mean_freq())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Self::from_ordered(drivers, crossover_type)
    }

    /// Create loss data while preserving the caller's explicit acoustic-band order.
    pub fn new_ordered(drivers: Vec<DriverMeasurement>, crossover_type: CrossoverType) -> Self {
        let max_drivers = if crossover_type == CrossoverType::None {
            usize::MAX
        } else {
            4
        };
        assert!(
            drivers.len() >= 2 && drivers.len() <= max_drivers,
            "Must have at least 2 drivers and at most 4 when using crossovers, got {}",
            drivers.len()
        );
        Self::from_ordered(drivers, crossover_type)
    }

    fn from_ordered(drivers: Vec<DriverMeasurement>, crossover_type: CrossoverType) -> Self {
        // Create a common frequency grid spanning all drivers
        // Use logarithmic spacing from lowest to highest frequency
        let min_freq = drivers
            .iter()
            .map(|d| d.freq_range().0)
            .fold(f64::INFINITY, f64::min);
        let max_freq = drivers
            .iter()
            .map(|d| d.freq_range().1)
            .fold(f64::NEG_INFINITY, f64::max);

        let (min_freq, max_freq) = if crossover_type == CrossoverType::None {
            (
                drivers
                    .iter()
                    .map(|d| d.freq_range().0)
                    .fold(f64::NEG_INFINITY, f64::max),
                drivers
                    .iter()
                    .map(|d| d.freq_range().1)
                    .fold(f64::INFINITY, f64::min),
            )
        } else {
            (min_freq, max_freq)
        };
        // Create log-spaced frequency grid (10 points per octave)
        let freq_grid = crate::read::create_log_frequency_grid(
            10 * 10, // 10 octaves * 10 points per octave
            min_freq.max(20.0),
            max_freq.min(20000.0),
        );

        let freq_grid = if crossover_type == CrossoverType::None {
            let mut frequencies = freq_grid.to_vec();
            frequencies.extend(
                drivers
                    .iter()
                    .flat_map(|driver| driver.freq.iter().copied())
                    .filter(|&frequency| {
                        frequency >= min_freq.max(20.0) && frequency <= max_freq.min(20000.0)
                    }),
            );
            frequencies.sort_by(f64::total_cmp);
            frequencies.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
            Array1::from_vec(frequencies)
        } else {
            freq_grid
        };
        let mut power_reference = Array1::<f64>::zeros(freq_grid.len());
        for driver in &drivers {
            let curve = autoeq_core::Curve {
                freq: driver.freq.clone(),
                spl: driver.spl.clone(),
                phase: driver.phase.clone(),
                ..Default::default()
            };
            let aligned = autoeq_core::interpolate_log_space(&freq_grid, &curve);
            power_reference += &aligned.spl.mapv(|level| 10.0_f64.powf(level / 10.0));
        }
        power_reference.mapv_inplace(|power| 10.0 * power.max(1e-24).log10());
        Self {
            power_reference,
            drivers,
            crossover_type,
            freq_grid,
        }
    }
}

/// Compute the loss for multi-driver crossover optimization
///
/// # Arguments
/// * `data` - DriversLossData containing driver measurements
/// * `gains` - Gain in dB for each driver
/// * `crossover_freqs` - Crossover frequencies between successive driver pairs
/// * `sample_rate` - Sample rate for filter design
/// * `min_freq` - Minimum frequency for loss evaluation
/// * `max_freq` - Maximum frequency for loss evaluation
///
/// # Returns
/// * Loss value (lower is better)
pub fn drivers_flat_loss(
    data: &DriversLossData,
    gains: &[f64],
    crossover_freqs: &[f64],
    delays: Option<&[f64]>,
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
) -> f64 {
    // Compute combined response
    let combined_response =
        compute_drivers_combined_response(data, gains, crossover_freqs, delays, sample_rate);

    // Normalize the response (subtract the mean in the evaluation range)
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..data.freq_grid.len() {
        let freq = data.freq_grid[i];
        if freq >= min_freq && freq <= max_freq {
            sum += combined_response[i];
            count += 1;
        }
    }
    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    let normalized = &combined_response - mean;

    // Compute flatness loss (RMS deviation from zero)
    flat_loss(&data.freq_grid, &normalized, min_freq, max_freq)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measurement(offset: f64) -> DriverMeasurement {
        DriverMeasurement {
            freq: ndarray::Array1::from_vec(vec![20.0, 80.0, 200.0]),
            spl: ndarray::Array1::from_vec(vec![70.0 + offset; 3]),
            phase: None,
        }
    }

    #[test]
    fn parallel_array_grid_keeps_narrow_measured_cancellation() {
        let frequencies = ndarray::array![20.0, 49.9, 50.0, 50.1, 200.0];
        let first = DriverMeasurement {
            freq: frequencies.clone(),
            spl: ndarray::Array1::from_elem(5, 80.0),
            phase: Some(ndarray::Array1::zeros(5)),
        };
        let second = DriverMeasurement {
            phase: Some(ndarray::array![0.0, 0.0, 180.0, 0.0, 0.0]),
            ..first.clone()
        };
        let data = DriversLossData::new_ordered(vec![first, second], CrossoverType::None);
        for frequency in frequencies {
            assert!(data.freq_grid.iter().any(|&f| f == frequency));
        }
        let combined =
            compute_drivers_combined_response(&data, &[0.0, 0.0], &[], Some(&[0.0, 0.0]), 48000.0);
        let index = data.freq_grid.iter().position(|&f| f == 50.0).unwrap();
        assert!(
            combined[index] < -200.0,
            "native cancellation was lost: {}",
            combined[index]
        );
    }

    #[test]
    fn no_crossover_supports_eight_subwoofers() {
        let drivers = (0..8).map(|index| measurement(index as f64)).collect();
        let data = DriversLossData::new(drivers, CrossoverType::None);
        assert_eq!(data.drivers.len(), 8);
    }

    #[test]
    #[should_panic(expected = "at most 4 when using crossovers")]
    fn crossover_still_rejects_more_than_four_drivers() {
        let drivers = (0..5).map(|index| measurement(index as f64)).collect();
        let _ = DriversLossData::new(drivers, CrossoverType::LinkwitzRiley4);
    }
}
