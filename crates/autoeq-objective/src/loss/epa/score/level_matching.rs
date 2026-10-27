//! Fixed pressure-energy references for level-matched transfer-spectrum diagnostics.

// Rust guideline compliant 2026-02-21

/// Compute the weighted log-frequency pressure-energy mean over 1–2 kHz.
///
/// All channels share `grid`. The grid must explicitly contain both reference
/// endpoints; this function neither interpolates nor extrapolates measurements.
/// Channel weights preserve relative energy balance. A common SPL offset shifts
/// the returned reference by the same amount. Subtracting that reference supplies
/// level-relative spectra, not calibrated programme audio or listener preference.
///
/// Returns `None` for missing endpoints, inconsistent shapes, unordered or invalid
/// frequencies, nonfinite levels, negative weights, or zero total weighted energy.
///
/// # Examples
/// ```
/// use autoeq_objective::loss::epa::score::reference_band_energy_db;
/// let channels = vec![vec![75.0, 75.0]];
/// let reference = reference_band_energy_db(&[1000.0, 2000.0], &channels, &[1.0]);
/// assert_eq!(reference, Some(75.0));
/// ```
pub fn reference_band_energy_db(
    grid: &[f64],
    channels: &[Vec<f64>],
    weights: &[f64],
) -> Option<f64> {
    if grid.len() < 2
        || channels.is_empty()
        || channels.len() != weights.len()
        || grid
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || grid.windows(2).any(|pair| pair[0] >= pair[1])
        || channels.iter().any(|channel| {
            channel.len() != grid.len() || channel.iter().any(|level| !level.is_finite())
        })
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return None;
    }
    // This octave is the existing delivered-EPA matching contract. Do not
    // substitute a fit-dependent band or a fitted spectral tilt.
    let first = grid
        .binary_search_by(|frequency| frequency.total_cmp(&1_000.0))
        .ok()?;
    let last = grid
        .binary_search_by(|frequency| frequency.total_cmp(&2_000.0))
        .ok()?;
    let weight_sum = weights.iter().sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return None;
    }
    let anchor = channels
        .iter()
        .zip(weights)
        .filter(|(_, weight)| **weight > 0.0)
        .flat_map(|(channel, _)| channel[first..=last].iter().copied())
        .fold(f64::NEG_INFINITY, f64::max);
    let energy = |index: usize| {
        channels
            .iter()
            .zip(weights)
            .filter(|(_, weight)| **weight > 0.0)
            .map(|(channel, weight)| weight * 10.0_f64.powf((channel[index] - anchor) / 10.0))
            .sum::<f64>()
            / weight_sum
    };
    let area = (first..last)
        .map(|index| {
            0.5 * (energy(index) + energy(index + 1)) * (grid[index + 1] / grid[index]).ln()
        })
        .sum::<f64>();
    let mean = area / 2.0_f64.ln();
    (mean.is_finite() && mean > 0.0).then(|| anchor + 10.0 * mean.log10())
}

#[cfg(test)]
mod tests {
    use super::reference_band_energy_db;

    #[test]
    fn reference_band_energy_db_uses_energy_and_preserves_common_offset() {
        let grid = [1000.0, 2000.0];
        let channels = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let expected = 7.75_f64.log10() * 10.0;
        let reference = reference_band_energy_db(&grid, &channels, &[1.0, 3.0]).unwrap();
        assert!((reference - expected).abs() < 1e-12);
        // Large offsets must not overflow the pressure-energy calculation.
        let shifted: Vec<_> = channels
            .iter()
            .map(|channel| channel.iter().map(|level| level + 4000.0).collect())
            .collect();
        let shifted_reference = reference_band_energy_db(&grid, &shifted, &[1.0, 3.0]).unwrap();
        assert!((shifted_reference - reference - 4000.0).abs() < 1e-12);
    }

    #[test]
    fn reference_band_energy_db_integrates_in_log_frequency() {
        let grid = [500.0, 1000.0, 2.0_f64.sqrt() * 1000.0, 2000.0, 4000.0];
        let channels = vec![vec![100.0, 0.0, 10.0, 0.0, 100.0]];
        let reference = reference_band_energy_db(&grid, &channels, &[1.0]).unwrap();
        // Equal log-width trapezoids have mean energy (1 + 2*10 + 1)/4.
        assert!((reference - 5.5_f64.log10() * 10.0).abs() < 1e-12);
    }

    #[test]
    fn reference_band_energy_db_excludes_zero_weight_channels_from_anchor() {
        let channels = vec![vec![75.0, 75.0], vec![4000.0, 4000.0]];
        assert_eq!(
            reference_band_energy_db(&[1000.0, 2000.0], &channels, &[1.0, 0.0]),
            Some(75.0)
        );
    }

    #[test]
    fn reference_band_energy_db_refuses_unsupported_or_invalid_inputs() {
        let levels = vec![vec![0.0, 0.0]];
        for grid in [
            [1001.0, 2000.0],
            [1000.0, 1999.0],
            [2000.0, 1000.0],
            [1000.0, f64::NAN],
        ] {
            assert!(reference_band_energy_db(&grid, &levels, &[1.0]).is_none());
        }
        let grid = [1000.0, 2000.0];
        for weights in [
            vec![],
            vec![0.0],
            vec![-1.0],
            vec![f64::NAN],
            vec![f64::INFINITY],
        ] {
            assert!(reference_band_energy_db(&grid, &levels, &weights).is_none());
        }
        assert!(reference_band_energy_db(&grid, &[vec![0.0]], &[1.0]).is_none());
        assert!(reference_band_energy_db(&grid, &[vec![0.0, f64::NAN]], &[1.0]).is_none());
        assert!(reference_band_energy_db(&grid, &[], &[]).is_none());
    }
}
