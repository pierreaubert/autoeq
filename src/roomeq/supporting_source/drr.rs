//! Direct-to-Reverberant Ratio (DRR) computation for supporting-source diagnostics.

use crate::Curve;

#[cfg(test)]
use ndarray::Array1;

/// Compute DRR before and after applying the supporting-source filter.
///
/// This is a magnitude-only approximation. We treat the direct sound as the
/// first arrival (primary source) and the reverberant field as the sum of the
/// primary reverberant tail plus the filtered supporting source.
///
/// # Arguments
/// * `primary` - Averaged primary loudspeaker magnitude response (dB).
/// * `support` - Averaged supporting loudspeaker magnitude response (dB).
/// * `support_gain_db` - Supporting-source filter gain in dB per frequency bin.
/// * `direct_window_ratio` - Fraction of the primary magnitude treated as direct
///   sound. The paper implicitly assumes the primary's direct field dominates
///   first arrival. Default: 0.5 (-6 dB).
///
/// # Returns
/// `(drr_before_db, drr_after_db)` vectors aligned with `primary.freq`.
pub fn compute_drr(
    primary: &Curve,
    support: &Curve,
    support_gain_db: &[f64],
    direct_window_ratio: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = primary.freq.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let dir_lin2 = primary
        .spl
        .iter()
        .map(|&db| {
            let mag = 10.0_f64.powf(db / 20.0);
            (mag * direct_window_ratio).powi(2)
        })
        .collect::<Vec<_>>();

    let primary_rev_lin2 = primary
        .spl
        .iter()
        .map(|&db| {
            let mag = 10.0_f64.powf(db / 20.0);
            (mag * (1.0 - direct_window_ratio)).powi(2)
        })
        .collect::<Vec<_>>();

    let support_lin = support
        .spl
        .iter()
        .zip(support_gain_db.iter())
        .map(|(&s_db, &g_db)| {
            let s_mag = 10.0_f64.powf(s_db / 20.0);
            let g_mag = 10.0_f64.powf(g_db / 20.0);
            s_mag * g_mag
        })
        .collect::<Vec<_>>();

    let mut before = Vec::with_capacity(n);
    let mut after = Vec::with_capacity(n);
    for i in 0..n {
        let rev_before = primary_rev_lin2[i];
        let rev_after = primary_rev_lin2[i] + support_lin[i].powi(2);
        before.push(ratio_to_db(dir_lin2[i], rev_before));
        after.push(ratio_to_db(dir_lin2[i], rev_after));
    }

    (before, after)
}

fn ratio_to_db(direct: f64, reverberant: f64) -> f64 {
    if reverberant <= 0.0 || !reverberant.is_finite() {
        return 40.0;
    }
    if direct <= 0.0 || !direct.is_finite() {
        return -40.0;
    }
    10.0 * (direct / reverberant).log10()
}

/// Simple statistical summary of a dB-valued curve.
#[allow(dead_code)]
pub fn db_summary(values: &[f64]) -> (f64, f64) {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len().max(1) as f64;
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_curve(spl_db: f64, n: usize) -> Curve {
        Curve {
            freq: Array1::linspace(100.0, 10000.0, n),
            spl: Array1::from_elem(n, spl_db),
            ..Default::default()
        }
    }

    #[test]
    fn drr_decreases_when_support_adds_energy() {
        let primary = flat_curve(80.0, 50);
        let support = flat_curve(80.0, 50);
        let gain_db = vec![0.0; 50];
        let (before, after) = compute_drr(&primary, &support, &gain_db, 0.5);
        assert!(before.iter().all(|v| v.is_finite()));
        assert!(after.iter().all(|v| v.is_finite()));
        assert!(
            after.iter().sum::<f64>() < before.iter().sum::<f64>(),
            "DRR should decrease when supporting source adds energy"
        );
    }

    #[test]
    fn empty_curve_returns_empty() {
        let empty = Curve::default();
        let (before, after) = compute_drr(&empty, &empty, &[], 0.5);
        assert!(before.is_empty() && after.is_empty());
    }
}
