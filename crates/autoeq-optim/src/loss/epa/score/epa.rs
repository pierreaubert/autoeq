use super::epa_config::EpaConfig;
use super::types::EpaChannelRole;

/// BS.1770-style channel energy weight for a coarse EPA role.
pub fn epa_channel_energy_weight(role: EpaChannelRole) -> f64 {
    match role {
        EpaChannelRole::Main => 1.0,
        EpaChannelRole::Surround => 1.41,
        EpaChannelRole::Lfe => 0.0,
    }
}

/// EPA optimizer loss.
///
/// Transfer magnitude alone cannot support programme loudness, roughness, or
/// measured-decay claims. Those EPA descriptors remain available in reports,
/// while filter optimization uses only the configured spectral flatness term.
pub fn epa_loss(_freqs: &[f64], _spl_db: &[f64], _config: &EpaConfig, flatness_loss: f64) -> f64 {
    flatness_loss
}

/// Level-relative counterpart of [`epa_loss`].
pub fn epa_loss_normalized(
    freqs: &[f64],
    spl_rel: &[f64],
    config: &EpaConfig,
    flatness_loss: f64,
) -> f64 {
    epa_loss(freqs, spl_rel, config, flatness_loss)
}

/// Compute the configured EPA spectral-flatness component.
///
/// Frequencies outside `[min_freq, max_freq]` are excluded. Returns
/// `f64::INFINITY` when the active range contains no samples.
pub fn epa_flatness(
    freqs: &ndarray::Array1<f64>,
    error: &ndarray::Array1<f64>,
    min_freq: f64,
    max_freq: f64,
    config: &EpaConfig,
) -> f64 {
    use crate::loss::enhanced_weights::combined_weighted_loss;

    let mut active_frequencies = Vec::new();
    let mut active_error = Vec::new();
    for (&frequency, &value) in freqs.iter().zip(error.iter()) {
        if frequency >= min_freq && frequency <= max_freq {
            active_frequencies.push(frequency);
            active_error.push(value);
        }
    }
    if active_frequencies.is_empty() {
        return f64::INFINITY;
    }

    combined_weighted_loss(
        &ndarray::Array1::from(active_frequencies),
        &ndarray::Array1::from(active_error),
        &config.flatness_band_weights,
        config.flatness_erb_weight,
        config.flatness_band_weight,
    )
}
