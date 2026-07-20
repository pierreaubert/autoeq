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

    if !entries.iter().all(|(_, initial, final_, _)| {
        same_frequency_grid(freqs, &initial.freq) && same_frequency_grid(freqs, &final_.freq)
    }) {
        log::warn!("Skipping multichannel EPA aggregation: channel frequency grids do not match");
        return None;
    }

    let pre_channels: Vec<_> = entries
        .iter()
        .map(|(_, initial, _, role)| (initial.spl.as_slice(), *role))
        .collect();
    let post_channels: Vec<_> = entries
        .iter()
        .map(|(_, _, final_, role)| (final_.spl.as_slice(), *role))
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
    }
}
