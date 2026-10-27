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

type AlignedPair<'a> = (Vec<Cow<'a, Curve>>, Vec<Cow<'a, Curve>>);

/// Validate a before/after pair and align both sets on one common grid.
fn aligned_pair<'a>(
    before: &'a [Curve],
    after: &'a [Curve],
    max_hz: f64,
) -> Result<AlignedPair<'a>, String> {
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
    Ok((align(before, grid), align(after, grid)))
}

pub fn correlated_bass_efficiency_regression(
    before: &[Curve],
    after: &[Curve],
    max_hz: f64,
) -> Result<(f64, f64), String> {
    let (before, after) = aligned_pair(before, after, max_hz)?;
    let grid = &before[0].freq;
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

/// Measure the band-integrated coherent/RSS efficiency loss over shared bass bins.
///
/// Programme-plane counterpart of [`correlated_bass_efficiency_regression`]:
/// coherent and RSS power are integrated over the whole band before the
/// ratio is taken, so a single-bin summation null shifted by a relative
/// delay change cannot dominate the verdict. A systematic decorrelation
/// (for example one inverted channel) still collapses the integrated
/// coherent power across the band. The returned pair is
/// `(loss_db, frequency_hz)` where the frequency is the worst single-bin
/// regression, kept for diagnostics only. The physical-array plane, where
/// every output carries the same signal and a worst-bin null is a real
/// array defect, should keep using the worst-bin measure.
///
/// # Errors
///
/// Refuses inconsistent input counts, invalid curves, missing measured phase,
/// an invalid upper frequency, or fewer than two common supported bass bins.
pub fn correlated_bass_band_efficiency_regression(
    before: &[Curve],
    after: &[Curve],
    max_hz: f64,
) -> Result<(f64, f64), String> {
    let (before, after) = aligned_pair(before, after, max_hz)?;
    let grid = &before[0].freq;
    // One global level reference for both sets: the efficiency ratio of
    // each set is invariant to a common scale, and band integration then
    // weights bins by their actual programme power.
    let reference = before
        .iter()
        .chain(&after)
        .flat_map(|curve| curve.spl.iter().copied())
        .fold(f64::NEG_INFINITY, f64::max);
    let powers = |curves: &[Cow<'_, Curve>], bin: usize| {
        let mut coherent = num_complex::Complex64::new(0.0, 0.0);
        let mut rss = 0.0;
        for curve in curves {
            let magnitude = 10.0_f64.powf((curve.spl[bin] - reference) / 20.0);
            coherent += num_complex::Complex64::from_polar(
                magnitude,
                curve.phase.as_ref().expect("validated phase")[bin].to_radians(),
            );
            rss += magnitude * magnitude;
        }
        (coherent.norm_sqr(), rss)
    };
    let mut bins = 0;
    let (mut coherent_before, mut rss_before) = (0.0, 0.0);
    let (mut coherent_after, mut rss_after) = (0.0, 0.0);
    let mut worst: Option<(f64, f64)> = None;
    for (bin, &frequency) in grid.iter().enumerate() {
        if frequency > max_hz {
            continue;
        }
        bins += 1;
        let (coherent, rss) = powers(&before, bin);
        coherent_before += coherent;
        rss_before += rss;
        let before_db = 10.0 * (coherent / rss).max(1e-24).log10();
        let (coherent, rss) = powers(&after, bin);
        coherent_after += coherent;
        rss_after += rss;
        let after_db = 10.0 * (coherent / rss).max(1e-24).log10();
        let regression = (before_db - after_db).max(0.0);
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
    let efficiency_db = |coherent: f64, rss: f64| 10.0 * (coherent / rss).max(1e-24).log10();
    let loss = (efficiency_db(coherent_before, rss_before)
        - efficiency_db(coherent_after, rss_after))
    .max(0.0);
    if !loss.is_finite() {
        return Err("nonfinite correlated bass efficiency".into());
    }
    Ok((loss, worst.expect("two supported bins").1))
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
    fn dense_response() -> Curve {
        let freq: Vec<f64> = (0..=28).map(|i| 20.0 + 10.0 * f64::from(i)).collect();
        let count = freq.len();
        Curve {
            freq: freq.into(),
            spl: ndarray::Array1::from_elem(count, 80.0),
            phase: Some(ndarray::Array1::zeros(count)),
            ..Default::default()
        }
    }
    #[test]
    fn band_measure_ignores_a_shifted_single_bin_null_but_catches_inversion() {
        let before = vec![dense_response(); 2];
        // A 2 ms relative delay rotates the pair through antiphase near
        // 250 Hz: the worst-bin measure explodes, the band measure keeps
        // most of the coherent energy because low bins stay aligned.
        let mut delayed = before.clone();
        let freq = delayed[1].freq.clone();
        for (phase, frequency) in delayed[1].phase.as_mut().unwrap().iter_mut().zip(freq) {
            *phase = -360.0 * frequency * 0.002;
        }
        let worst = correlated_bass_efficiency_regression(&before, &delayed, 300.0)
            .unwrap()
            .0;
        let band = correlated_bass_band_efficiency_regression(&before, &delayed, 300.0)
            .unwrap()
            .0;
        assert!(worst > 20.0, "worst-bin loss {worst}");
        // Flat-magnitude pair: 2 ms across 20-300 Hz genuinely spends
        // most of the 3 dB two-channel coherent advantage (measured
        // 4.13 dB), but stays orders of magnitude below the worst-bin
        // verdict driven by the single antiphase bin.
        assert!((3.0..5.0).contains(&band), "band loss {band}");
        assert!(band < worst * 0.1, "band {band} vs worst {worst}");
        // A broadband inversion collapses integrated coherent power.
        let mut inverted = before.clone();
        inverted[1]
            .phase
            .as_mut()
            .unwrap()
            .mapv_inplace(|phase| phase + 180.0);
        let inverted_band = correlated_bass_band_efficiency_regression(&before, &inverted, 300.0)
            .unwrap()
            .0;
        assert!(inverted_band > 20.0, "inverted band loss {inverted_band}");
        // Common delay/gain remains free, exactly like the worst-bin measure.
        let mut common = before.clone();
        for curve in &mut common {
            curve.spl.mapv_inplace(|level| level - 12.0);
            let freq = curve.freq.clone();
            for (phase, frequency) in curve.phase.as_mut().unwrap().iter_mut().zip(freq) {
                *phase = -360.0 * frequency * 0.007;
            }
        }
        assert!(
            correlated_bass_band_efficiency_regression(&before, &common, 300.0)
                .unwrap()
                .0
                < 1e-10
        );
        let mut missing_phase = before.clone();
        missing_phase[0].phase = None;
        assert!(
            correlated_bass_band_efficiency_regression(&before, &missing_phase, 300.0).is_err()
        );
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
