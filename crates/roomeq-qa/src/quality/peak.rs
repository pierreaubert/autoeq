use roomeq_model::Curve;

/// Fix support from the uncorrected measurement and the declared physical-role
/// band. Only outer roll-offs reduce support; internal nulls remain evidence.
pub(super) fn measured_band(reference: &Curve, fmin: f64, fmax: f64) -> Option<(f64, f64)> {
    let peak = reference
        .freq
        .iter()
        .zip(&reference.spl)
        .filter(|(f, spl)| **f >= fmin && **f <= fmax && spl.is_finite())
        .map(|(_, spl)| *spl)
        .reduce(f64::max)?;
    let mut supported = reference
        .freq
        .iter()
        .zip(&reference.spl)
        .filter(|(f, spl)| **f >= fmin && **f <= fmax && **spl >= peak - 20.0)
        .map(|(f, _)| *f);
    let low = supported.next()?;
    let high = supported.next_back().unwrap_or(low);
    (high > low).then_some((low, high))
}

fn band_values(curve: &Curve, fmin: f64, fmax: f64) -> Vec<f64> {
    if curve.freq.first().is_none_or(|f| *f > fmin) || curve.freq.last().is_none_or(|f| *f < fmax) {
        return Vec::new();
    }
    curve
        .freq
        .iter()
        .zip(&curve.spl)
        .filter(|(f, _)| **f >= fmin && **f <= fmax)
        .map(|(_, spl)| *spl)
        .collect()
}

/// Bin RMS above and below the mean on fixed support. A candidate cannot
/// hide a new cancellation by moving a response-dependent eligibility floor.
pub(super) fn peak_dip_from_mean(curve: &Curve, fmin: f64, fmax: f64) -> (f64, f64) {
    let values = band_values(curve, fmin, fmax);
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return (f64::INFINITY, f64::INFINITY);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut peak_sum = 0.0;
    let mut peak_count = 0usize;
    let mut dip_sum = 0.0;
    let mut dip_count = 0usize;
    for value in values {
        let dev = value - mean;
        if dev > 0.0 {
            peak_sum += dev * dev;
            peak_count += 1;
        } else if dev < 0.0 {
            dip_sum += dev * dev;
            dip_count += 1;
        }
    }
    (
        (peak_sum / peak_count.max(1) as f64).sqrt(),
        (dip_sum / dip_count.max(1) as f64).sqrt(),
    )
}

/// Maximum positive deviation on fixed support. This measures peaks only;
/// dip RMS and target deficits must be assessed separately.
pub(super) fn peak_deviation_db(curve: &Curve, fmin: f64, fmax: f64) -> f64 {
    let values = band_values(curve, fmin, fmax);
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.into_iter().map(|v| v - mean).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat() -> Curve {
        Curve {
            freq: (20..=500).map(|f| f as f64).collect(),
            spl: vec![80.0; 481].into(),
            ..Default::default()
        }
    }

    #[test]
    fn deeper_native_bin_holes_cannot_disappear_below_a_candidate_floor() {
        let reference = flat();
        let band = measured_band(&reference, 20.0, 500.0).unwrap();
        for range in [60..=120, 60..=60, 61..=61, 119..=119] {
            let mut previous_dip = 0.0;
            for depth in [10.0, 19.9, 20.1, 30.0, 60.0] {
                let mut candidate = reference.clone();
                for (f, spl) in candidate.freq.iter().zip(candidate.spl.iter_mut()) {
                    if range.contains(&(*f as i32)) {
                        *spl -= depth;
                    }
                }
                let (_, dip) = peak_dip_from_mean(&candidate, band.0, band.1);
                assert!(dip > previous_dip, "depth {depth}: {dip} <= {previous_dip}");
                assert!(peak_deviation_db(&candidate, band.0, band.1) > 0.0);
                previous_dip = dip;
            }
        }
    }

    #[test]
    fn measurement_rolloff_excludes_stopband_but_keeps_internal_nulls() {
        let mut reference = flat();
        for (f, spl) in reference.freq.iter().zip(reference.spl.iter_mut()) {
            if *f > 200.0 || (60.0..=120.0).contains(f) {
                *spl -= 30.0;
            }
        }
        let band = measured_band(&reference, 20.0, 500.0).unwrap();
        assert_eq!(band, (20.0, 200.0));
        assert!(peak_dip_from_mean(&reference, band.0, band.1).1 > 10.0);
        let mut candidate = flat();
        for (f, spl) in candidate.freq.iter().zip(candidate.spl.iter_mut()) {
            if *f > 200.0 {
                *spl -= 60.0;
            }
        }
        assert_eq!(peak_dip_from_mean(&candidate, band.0, band.1), (0.0, 0.0));
    }
}
