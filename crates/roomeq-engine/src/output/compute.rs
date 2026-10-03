use super::misc::same_frequency_grid;
use autoeq_optim::loss::epa::score::{
    compute_epa_multichannel_normalized, compute_epa_normalized, epa_channel_energy_weight,
    infer_epa_channel_role,
};
use roomeq_model::EpaConfig;
use roomeq_model::{ChannelDspChain, CurveData, EpaChannelMetrics, EpaMultichannelMetrics};
use std::collections::HashMap;

/// Compute per-channel EPA metrics (pre-EQ and post-EQ) from each
/// channel's `initial_curve` and `final_curve`.
///
/// `CurveData.spl` is mean-subtracted around 1–2 kHz (level-relative),
/// so we call [`compute_epa_normalized`] which denormalizes the curve
/// against `config.listening_level_phon` before running the
/// psychoacoustic model. Without that calibration step the loudness
/// and loudness-balance components would be dominated by the absolute
/// threshold of hearing.
///
/// Returns `None` if no channel has both curves populated.
pub fn compute_epa_per_channel(
    channels: &HashMap<String, ChannelDspChain>,
    config: &EpaConfig,
) -> Option<HashMap<String, EpaChannelMetrics>> {
    let config = crate::config_adapter::to_optimizer_epa(config);
    let mut out: HashMap<String, EpaChannelMetrics> = HashMap::new();
    for (name, chain) in channels {
        let (Some(initial), Some(final_)) = (&chain.initial_curve, &chain.final_curve) else {
            continue;
        };
        let pre = crate::report_adapter::to_epa_score(compute_epa_normalized(
            &initial.freq,
            &initial.spl,
            &config,
        ));
        let post = crate::report_adapter::to_epa_score(compute_epa_normalized(
            &final_.freq,
            &final_.spl,
            &config,
        ));
        out.insert(name.clone(), EpaChannelMetrics { pre, post });
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Compute aggregate EPA metrics from all channel curves using BS.1770-style
/// channel energy weights.
///
/// This is a frequency-response approximation for room-EQ reports. It does
/// not replace time-domain LUFS metering, but it avoids treating stereo or
/// surround systems as unrelated monaural measurements.
pub fn compute_epa_multichannel(
    channels: &HashMap<String, ChannelDspChain>,
    config: &EpaConfig,
) -> Option<EpaMultichannelMetrics> {
    let config = crate::config_adapter::to_optimizer_epa(config);
    let mut entries: Vec<_> = channels
        .iter()
        .filter_map(|(name, chain)| {
            let (Some(initial), Some(final_)) = (&chain.initial_curve, &chain.final_curve) else {
                return None;
            };
            let role = infer_epa_channel_role(name);
            (epa_channel_energy_weight(role) > 0.0).then_some((
                name.as_str(),
                initial,
                final_,
                role,
            ))
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let (_, first_initial, _, _) = entries.first()?;
    let freqs = first_initial.freq.as_slice();
    if freqs.is_empty() {
        return None;
    }

    let grids_match = entries.iter().all(|(_, initial, final_, _)| {
        same_frequency_grid(freqs, &initial.freq) && same_frequency_grid(freqs, &final_.freq)
    });
    let aligned = if grids_match {
        None
    } else {
        let curves: Vec<_> = entries
            .iter()
            .flat_map(|(_, initial, final_, _)| [*initial, *final_])
            .collect();
        Some(align_epa_curves(&curves)?)
    };
    let freqs = aligned.as_ref().map_or(freqs, |(grid, _)| grid.as_slice());

    let pre_channels: Vec<_> = entries
        .iter()
        .enumerate()
        .map(|(i, (_, initial, _, role))| {
            (
                aligned
                    .as_ref()
                    .map_or(initial.spl.as_slice(), |(_, spl)| spl[2 * i].as_slice()),
                *role,
            )
        })
        .collect();
    let post_channels: Vec<_> = entries
        .iter()
        .enumerate()
        .map(|(i, (_, _, final_, role))| {
            (
                aligned
                    .as_ref()
                    .map_or(final_.spl.as_slice(), |(_, spl)| spl[2 * i + 1].as_slice()),
                *role,
            )
        })
        .collect();

    let pre = crate::report_adapter::to_epa_score(compute_epa_multichannel_normalized(
        freqs,
        &pre_channels,
        &config,
    )?);
    let post = crate::report_adapter::to_epa_score(compute_epa_multichannel_normalized(
        freqs,
        &post_channels,
        &config,
    )?);

    Some(EpaMultichannelMetrics {
        pre,
        post,
        standard: "BS.1770-style channel energy aggregation over EPA spectra".to_string(),
    })
}

/// Align EPA magnitudes to the union of native bins within shared support.
/// Never extrapolate a channel or change its SPL normalization.
fn align_epa_curves(curves: &[&CurveData]) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    if curves.is_empty()
        || curves.iter().any(|curve| {
            curve.freq.len() < 2
                || curve.freq.len() != curve.spl.len()
                || curve.freq.iter().any(|f| !f.is_finite() || *f <= 0.0)
                || curve.freq.windows(2).any(|pair| pair[0] >= pair[1])
                || curve.spl.iter().any(|spl| !spl.is_finite())
        })
    {
        return None;
    }
    let low = curves
        .iter()
        .map(|c| c.freq[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let high = curves
        .iter()
        .map(|c| c.freq[c.freq.len() - 1])
        .fold(f64::INFINITY, f64::min);
    if low >= high {
        return None;
    }
    let mut grid: Vec<_> = curves
        .iter()
        .flat_map(|c| c.freq.iter().copied())
        .filter(|f| *f >= low && *f <= high)
        .collect();
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    let frequencies = ndarray::Array1::from(grid.clone());
    let levels = curves
        .iter()
        .map(|curve| {
            let source = crate::Curve {
                freq: ndarray::Array1::from(curve.freq.clone()),
                spl: ndarray::Array1::from(curve.spl.clone()),
                ..Default::default()
            };
            autoeq_core::interpolate_log_space(&frequencies, &source)
                .spl
                .to_vec()
        })
        .collect();
    Some((grid, levels))
}

/// Compute the EQ filter response curve from initial and final curves.
///
/// Returns a `CurveData` whose SPL values are `final - initial` (the correction in dB).
pub fn compute_eq_response(initial: &CurveData, final_curve: &CurveData) -> CurveData {
    let spl: Vec<f64> = final_curve
        .spl
        .iter()
        .zip(initial.spl.iter())
        .map(|(&f, &i)| f - i)
        .collect();
    CurveData {
        freq: initial.freq.clone(),
        spl,
        phase: None,
        norm_range: None,
        ..Default::default()
    }
}

#[cfg(test)]
mod alignment_tests {
    use super::*;

    fn curve(freq: Vec<f64>) -> CurveData {
        CurveData {
            spl: freq.iter().map(|f| 3.0 * f.log2()).collect(),
            freq,
            phase: None,
            norm_range: None,
            ..Default::default()
        }
    }

    #[test]
    fn alignment_interpolates_log_frequency_only_inside_shared_support() {
        let a = curve(vec![20.0, 40.0, 80.0, 160.0]);
        let b = curve(vec![30.0, 60.0, 120.0, 240.0]);
        let (grid, levels) = align_epa_curves(&[&a, &b]).unwrap();
        assert_eq!(grid, vec![30.0, 40.0, 60.0, 80.0, 120.0, 160.0]);
        for spl in levels {
            for (f, value) in grid.iter().zip(spl) {
                assert!((value - 3.0 * f.log2()).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn alignment_rejects_disjoint_or_invalid_data() {
        let a = curve(vec![20.0, 40.0]);
        assert!(align_epa_curves(&[&a, &curve(vec![80.0, 160.0])]).is_none());
        assert!(align_epa_curves(&[&a, &curve(vec![40.0, 80.0])]).is_none());
        for freq in [
            vec![],
            vec![20.0],
            vec![40.0, 20.0],
            vec![20.0, 20.0],
            vec![0.0, 40.0],
            vec![20.0, f64::NAN],
        ] {
            assert!(align_epa_curves(&[&a, &curve(freq)]).is_none());
        }
        let mut bad = a.clone();
        bad.spl.pop();
        assert!(align_epa_curves(&[&a, &bad]).is_none());
        bad = a.clone();
        bad.spl[0] = f64::NAN;
        assert!(align_epa_curves(&[&a, &bad]).is_none());
    }
}
