//! Baseline-relative coherent efficiency for program and physical array sums.

// Rust guideline compliant 2026-02-21
use autoeq_core::Curve;
use std::borrow::Cow;

/// Measure the worst coherent/RSS efficiency loss over shared measured bass bins.
///
/// A common gain or phase change cancels in the ratio. The returned pair is
/// `(loss_db, frequency_hz)`. A -240 dB relative-power floor bounds phase roundoff
/// at exact cancellation; it is a numerical floor, not an acoustic threshold.
/// Mismatched grids are explicitly interpolated on common measured support.
///
/// # Errors
///
/// Refuses inconsistent input counts, invalid curves, missing measured phase,
/// an invalid upper frequency, or fewer than two common supported bass bins.
pub fn correlated_bass_efficiency_regression(
    before: &[Curve],
    after: &[Curve],
    max_hz: f64,
) -> Result<(f64, f64), String> {
    if before.len() < 2 || before.len() != after.len() || !max_hz.is_finite() || max_hz <= 0.0 {
        return Err(
            "correlated efficiency needs matching multiple inputs and a positive band".into(),
        );
    }
    let curves: Vec<_> = before.iter().chain(after).collect();
    for curve in &curves {
        curve
            .validate("correlated bass")
            .map_err(|error| error.to_string())?;
        if curve.freq.len() < 2
            || curve.phase.as_ref().is_none_or(|phase| {
                phase.len() != curve.freq.len() || phase.iter().any(|value| !value.is_finite())
            })
        {
            return Err("correlated bass needs measured phase on every input".into());
        }
    }
    let joined = if curves.iter().all(|curve| curve.freq == before[0].freq) {
        None
    } else {
        let low = curves.iter().map(|curve| curve.freq[0]).fold(0.0, f64::max);
        let high = curves
            .iter()
            .map(|curve| *curve.freq.last().expect("nonempty curve"))
            .fold(f64::INFINITY, f64::min);
        if low >= high {
            return Err("correlated bass has no common measured support".into());
        }
        let mut frequencies: Vec<_> = curves
            .iter()
            .flat_map(|curve| curve.freq.iter().copied())
            .filter(|frequency| *frequency >= low && *frequency <= high)
            .collect();
        frequencies.sort_by(f64::total_cmp);
        frequencies.dedup();
        Some(ndarray::Array1::from(frequencies))
    };
    let grid = joined.as_ref().unwrap_or(&before[0].freq);
    fn align<'a>(curves: &'a [Curve], grid: &ndarray::Array1<f64>) -> Vec<Cow<'a, Curve>> {
        curves
            .iter()
            .map(|curve| {
                if curve.freq == *grid {
                    Cow::Borrowed(curve)
                } else {
                    Cow::Owned(autoeq_core::interpolate_log_space(grid, curve))
                }
            })
            .collect()
    }
    let before = align(before, grid);
    let after = align(after, grid);
    let efficiency = |curves: &[Cow<'_, Curve>], bin: usize| {
        let maximum = curves
            .iter()
            .map(|curve| curve.spl[bin])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut coherent = num_complex::Complex64::new(0.0, 0.0);
        let mut power = 0.0;
        for curve in curves {
            let magnitude = 10.0_f64.powf((curve.spl[bin] - maximum) / 20.0);
            coherent += num_complex::Complex64::from_polar(
                magnitude,
                curve.phase.as_ref().expect("validated phase")[bin].to_radians(),
            );
            power += magnitude * magnitude;
        }
        10.0 * (coherent.norm_sqr() / power).max(1e-24).log10()
    };
    let mut worst: Option<(f64, f64)> = None;
    let mut bins = 0;
    for (bin, &frequency) in grid.iter().enumerate() {
        if frequency > max_hz {
            continue;
        }
        bins += 1;
        let regression = (efficiency(&before, bin) - efficiency(&after, bin)).max(0.0);
        if !regression.is_finite() {
            return Err("nonfinite correlated bass efficiency".into());
        }
        if worst.is_none_or(|(previous, _)| regression > previous) {
            worst = Some((regression, frequency));
        }
    }
    if bins < 2 {
        return Err("correlated bass needs at least two common bass bins".into());
    }
    Ok(worst.expect("two supported bins"))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn response() -> Curve {
        Curve {
            freq: ndarray::array![20.0, 50.0, 80.0, 150.0, 300.0],
            spl: ndarray::Array1::from_elem(5, 80.0),
            phase: Some(ndarray::Array1::zeros(5)),
            ..Default::default()
        }
    }
    #[test]
    fn common_equalization_and_delay_do_not_spend_coherent_efficiency() {
        let before = vec![response(); 3];
        let mut after = before.clone();
        for curve in &mut after {
            curve.spl.mapv_inplace(|level| level - 40.0);
            let freq = curve.freq.clone();
            for (phase, frequency) in curve.phase.as_mut().unwrap().iter_mut().zip(freq) {
                *phase = -360.0 * frequency * 0.004;
            }
        }
        assert!(
            correlated_bass_efficiency_regression(&before, &after, 300.0)
                .unwrap()
                .0
                < 1e-10
        );
    }
    #[test]
    fn interpolation_is_explicit_and_level_scaling_remains_invariant() {
        let before = vec![response(); 2];
        let mut after = before.clone();
        for curve in &mut after {
            curve.freq = ndarray::array![20.0, 40.0, 80.0, 160.0, 300.0];
            curve.spl.fill(5.0);
        }
        assert!(
            correlated_bass_efficiency_regression(&before, &after, 300.0)
                .unwrap()
                .0
                < 1e-10
        );
        after[1].phase = None;
        assert!(correlated_bass_efficiency_regression(&before, &after, 300.0).is_err());
    }
    #[test]
    fn inconsistent_input_counts_and_unsupported_bands_are_refused() {
        let before = vec![response(); 2];
        assert!(correlated_bass_efficiency_regression(&before, &before[..1], 300.0).is_err());
        assert!(correlated_bass_efficiency_regression(&before, &before, 20.0).is_err());
        assert!(correlated_bass_efficiency_regression(&before, &before, f64::NAN).is_err());
        let mut invalid = before.clone();
        invalid[0].spl[0] = f64::NAN;
        assert!(correlated_bass_efficiency_regression(&before, &invalid, 300.0).is_err());
    }
}
