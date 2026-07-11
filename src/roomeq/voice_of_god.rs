//! Inter-channel timbre matching.
//!
//! After per-channel EQ optimization, different speakers may have residual tonal
//! differences (e.g., one satellite has brighter timbre than another). This stage applies
//! broadband spectral alignment (lowshelf + highshelf + gain) to push each satellite
//! channel toward the reference channel's tonal balance.
//!
//! Reuses [`spectral_align::compute_target_alignment`] for the fitting.

use crate::Curve;
use crate::error::{AutoeqError, Result};
use log::info;
use std::collections::HashMap;

use super::spectral_align::{
    SpectralAlignmentResult, compute_target_alignment, create_alignment_filters,
    create_alignment_plugins,
};
use super::types::PluginConfigWrapper;

/// Structured outcome for one channel in the timbre-matching stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimbreMatchingChannelStatus {
    /// Reference channel; no correction is applied.
    Reference,
    /// A correction was computed.
    Applied,
    /// The channel needed no material correction.
    Skipped,
    /// Invalid channel data prevented a correction.
    Failed,
}

/// Result of inter-channel timbre analysis for a single channel.
#[derive(Debug, Clone)]
pub struct InterChannelTimbreMatchingResult {
    /// Channel name
    pub channel_name: String,
    /// Spectral alignment corrections (None for the reference channel)
    pub alignment: Option<SpectralAlignmentResult>,
    /// Whether this channel is the reference
    pub is_reference: bool,
    /// Structured per-channel stage outcome.
    pub status: TimbreMatchingChannelStatus,
    /// Machine-readable advisories for degraded or skipped paths.
    pub advisories: Vec<String>,
    /// Normalized pairwise timbre spread before candidate DSP.
    pub timbre_spread_before_db: Option<f64>,
    /// Normalized pairwise timbre spread after candidate DSP.
    pub timbre_spread_after_db: Option<f64>,
}

/// Compute inter-channel timbre corrections for all channels.
///
/// For each non-reference channel, fits lowshelf + highshelf + flat gain to match
/// the reference channel's spectral shape. The reference channel itself receives
/// no corrections.
///
/// # Errors
/// Returns an error if `reference_channel` is not found in `corrected_curves`.
pub fn compute_inter_channel_timbre_matching_with_threshold(
    corrected_curves: &HashMap<String, Curve>,
    reference_channel: &str,
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
    min_improvement_db: f64,
) -> Result<HashMap<String, InterChannelTimbreMatchingResult>> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "inter-channel timbre matching requires a positive sample rate, got {sample_rate}"
            ),
        });
    }
    if !min_freq.is_finite() || !max_freq.is_finite() || min_freq <= 0.0 || max_freq <= min_freq {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "inter-channel timbre matching requires 0 < min_freq < max_freq, got {min_freq}..{max_freq}"
            ),
        });
    }
    if !min_improvement_db.is_finite() || min_improvement_db < 0.0 {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "inter-channel timbre matching min_improvement_db must be finite and non-negative, got {min_improvement_db}"
            ),
        });
    }
    let reference_curve = corrected_curves.get(reference_channel).ok_or_else(|| {
        AutoeqError::InvalidConfiguration {
            message: format!(
                "Timbre-matching reference channel '{}' not found. Available: {:?}",
                reference_channel,
                corrected_curves.keys().collect::<Vec<_>>()
            ),
        }
    })?;
    validate_curve(reference_curve, "reference", reference_channel)?;

    let mut results = HashMap::new();

    for (name, curve) in corrected_curves {
        if name == reference_channel {
            results.insert(
                name.clone(),
                InterChannelTimbreMatchingResult {
                    channel_name: name.clone(),
                    alignment: None,
                    is_reference: true,
                    status: TimbreMatchingChannelStatus::Reference,
                    advisories: vec!["reference_channel".to_string()],
                    timbre_spread_before_db: None,
                    timbre_spread_after_db: None,
                },
            );
            continue;
        }

        if let Err(error) = validate_curve(curve, "channel", name) {
            results.insert(
                name.clone(),
                InterChannelTimbreMatchingResult {
                    channel_name: name.clone(),
                    alignment: None,
                    is_reference: false,
                    status: TimbreMatchingChannelStatus::Failed,
                    advisories: vec![error.to_string()],
                    timbre_spread_before_db: None,
                    timbre_spread_after_db: None,
                },
            );
            continue;
        }

        let grids_differ =
            !super::frequency_grid::same_frequency_grid(&curve.freq, &reference_curve.freq);
        // The target-alignment helper interpolates the reference onto this
        // channel's grid and restricts the fit to their measured overlap.
        let candidate_alignment =
            compute_target_alignment(curve, reference_curve, min_freq, max_freq, sample_rate);
        let mut advisories = Vec::new();
        if grids_differ {
            advisories.push("frequency_grid_interpolated".to_string());
        }
        let timbre_spread_before_db =
            pairwise_normalized_timbre_spread_db(curve, reference_curve, min_freq, max_freq);
        let (alignment, timbre_spread_after_db) = if let Some(candidate) = candidate_alignment {
            let corrected = apply_alignment_to_curve(curve, &candidate, sample_rate);
            let after = pairwise_normalized_timbre_spread_db(
                &corrected,
                reference_curve,
                min_freq,
                max_freq,
            );
            let improvement = timbre_spread_before_db
                .zip(after)
                .map(|(before, after)| before - after);
            if improvement.is_some_and(|value| value >= min_improvement_db) {
                (Some(candidate), after)
            } else {
                advisories.push("insufficient_timbre_spread_improvement".to_string());
                (None, after)
            }
        } else {
            (None, timbre_spread_before_db)
        };
        let status = if alignment.is_some() {
            TimbreMatchingChannelStatus::Applied
        } else {
            TimbreMatchingChannelStatus::Skipped
        };
        if alignment.is_none() && advisories.is_empty() {
            advisories.push("no_material_correction".to_string());
        }

        results.insert(
            name.clone(),
            InterChannelTimbreMatchingResult {
                channel_name: name.clone(),
                alignment,
                is_reference: false,
                status,
                advisories,
                timbre_spread_before_db,
                timbre_spread_after_db,
            },
        );
    }

    results
        .values()
        .filter(|r| !r.is_reference && r.alignment.is_some())
        .for_each(|r| {
            let a = r.alignment.as_ref().unwrap();
            info!(
                "  Timbre match '{}': LS={:+.2} dB, HS={:+.2} dB, gain={:+.2} dB (residual {:.2} dB RMS)",
                r.channel_name,
                a.lowshelf_gain_db,
                a.highshelf_gain_db,
                a.flat_gain_db,
                a.residual_rms_db,
            );
        });

    Ok(results)
}

pub fn compute_inter_channel_timbre_matching(
    corrected_curves: &HashMap<String, Curve>,
    reference_channel: &str,
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
) -> Result<HashMap<String, InterChannelTimbreMatchingResult>> {
    compute_inter_channel_timbre_matching_with_threshold(
        corrected_curves,
        reference_channel,
        sample_rate,
        min_freq,
        max_freq,
        0.05,
    )
}

fn validate_curve(curve: &Curve, role: &str, name: &str) -> Result<()> {
    if !super::frequency_grid::is_valid_frequency_grid(&curve.freq)
        || curve.spl.len() != curve.freq.len()
        || curve.spl.iter().any(|value| !value.is_finite())
    {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!(
                "Timbre-matching {role} channel '{name}' has invalid frequency/SPL data"
            ),
        });
    }
    Ok(())
}

/// Create DSP plugins for an inter-channel timbre-matching result.
///
/// Returns a list of plugins (EQ with shelves, gain) to apply to the channel.
/// Returns an empty list for the reference channel or channels with negligible corrections.
pub fn create_timbre_matching_plugins(
    result: &InterChannelTimbreMatchingResult,
    sample_rate: f64,
) -> Vec<PluginConfigWrapper> {
    let mut plugins = Vec::new();

    if let Some(alignment) = &result.alignment {
        let (eq_plugin, gain_plugin) = create_alignment_plugins(alignment, sample_rate);
        if let Some(eq) = eq_plugin {
            plugins.push(eq);
        }
        if let Some(gain) = gain_plugin {
            plugins.push(gain);
        }
    }

    plugins
}

pub(crate) fn apply_alignment_to_curve(
    curve: &Curve,
    alignment: &SpectralAlignmentResult,
    sample_rate: f64,
) -> Curve {
    let filters = create_alignment_filters(alignment, sample_rate);
    let response =
        crate::response::compute_peq_complex_response(&filters, &curve.freq, sample_rate);
    let mut corrected = crate::response::apply_complex_response(curve, &response);
    corrected.spl += alignment.flat_gain_db;
    corrected
}

/// Mean absolute, level-normalized timbre difference over measured overlap.
pub fn pairwise_normalized_timbre_spread_db(
    channel: &Curve,
    reference: &Curve,
    min_freq: f64,
    max_freq: f64,
) -> Option<f64> {
    validate_curve(channel, "channel", "metric").ok()?;
    validate_curve(reference, "reference", "metric").ok()?;
    let overlap_min = min_freq.max(channel.freq[0]).max(reference.freq[0]);
    let overlap_max = max_freq
        .min(channel.freq[channel.freq.len() - 1])
        .min(reference.freq[reference.freq.len() - 1]);
    if overlap_max <= overlap_min {
        return None;
    }
    let reference = if super::frequency_grid::same_frequency_grid(&channel.freq, &reference.freq) {
        reference.clone()
    } else {
        crate::read::interpolate_log_space(&channel.freq, reference)
    };
    let values = channel
        .freq
        .iter()
        .enumerate()
        .filter(|(_, frequency)| **frequency >= overlap_min && **frequency <= overlap_max)
        .filter(|(index, _)| channel.spl[*index].is_finite() && reference.spl[*index].is_finite())
        .map(|(index, _)| (channel.spl[index], reference.spl[index]))
        .collect::<Vec<_>>();
    if values.len() < 2 {
        return None;
    }
    let channel = values.iter().map(|(value, _)| *value).collect::<Vec<_>>();
    let reference = values.iter().map(|(_, value)| *value).collect::<Vec<_>>();
    super::acoustic_qa::normalized_timbre_spread_db(&[channel, reference])
}

#[deprecated(since = "0.4.47", note = "use TimbreMatchingChannelStatus")]
pub type VoGChannelStatus = TimbreMatchingChannelStatus;

#[deprecated(since = "0.4.47", note = "use InterChannelTimbreMatchingResult")]
pub type VoGResult = InterChannelTimbreMatchingResult;

#[deprecated(since = "0.4.47", note = "use compute_inter_channel_timbre_matching")]
pub fn compute_voice_of_god(
    corrected_curves: &HashMap<String, Curve>,
    reference_channel: &str,
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
) -> Result<HashMap<String, InterChannelTimbreMatchingResult>> {
    compute_inter_channel_timbre_matching_with_threshold(
        corrected_curves,
        reference_channel,
        sample_rate,
        min_freq,
        max_freq,
        0.05,
    )
}

#[deprecated(since = "0.4.47", note = "use create_timbre_matching_plugins")]
pub fn create_vog_plugins(
    result: &InterChannelTimbreMatchingResult,
    sample_rate: f64,
) -> Vec<PluginConfigWrapper> {
    create_timbre_matching_plugins(result, sample_rate)
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_curve(spl_fn: impl Fn(f64) -> f64) -> Curve {
        let n = 200;
        let log_start = 20f64.log10();
        let log_end = 20000f64.log10();
        let freq: Vec<f64> = (0..n)
            .map(|i| 10f64.powf(log_start + (log_end - log_start) * i as f64 / (n - 1) as f64))
            .collect();
        let spl: Vec<f64> = freq.iter().map(|&f| spl_fn(f)).collect();
        Curve {
            freq: Array1::from(freq),
            spl: Array1::from(spl),
            phase: None,
            ..Default::default()
        }
    }

    fn make_curve_on_grid(freq: Vec<f64>, spl_fn: impl Fn(f64) -> f64) -> Curve {
        let spl = freq.iter().map(|&f| spl_fn(f)).collect::<Vec<_>>();
        Curve {
            freq: Array1::from(freq),
            spl: Array1::from(spl),
            phase: None,
            ..Default::default()
        }
    }

    const SR: f64 = 48000.0;

    #[test]
    fn test_vog_identical_channels() {
        // All flat channels → corrections should be None (negligible)
        let mut curves = HashMap::new();
        curves.insert("L".to_string(), make_curve(|_| 0.0));
        curves.insert("R".to_string(), make_curve(|_| 0.0));
        curves.insert("C".to_string(), make_curve(|_| 0.0));

        let results = compute_voice_of_god(&curves, "C", SR, 20.0, 20000.0).unwrap();

        assert_eq!(results.len(), 3);
        assert!(results["C"].is_reference);
        assert_eq!(results["C"].status, VoGChannelStatus::Reference);
        // L and R should have no alignment (identical to reference)
        assert!(
            results["L"].alignment.is_none(),
            "L should need no correction"
        );
        assert!(
            results["R"].alignment.is_none(),
            "R should need no correction"
        );
    }

    #[test]
    fn test_vog_bass_mismatch() {
        // Reference (C) is flat, satellite (L) has +3dB bass → expect negative lowshelf correction
        let mut curves = HashMap::new();
        curves.insert("C".to_string(), make_curve(|_| 0.0));
        curves.insert(
            "L".to_string(),
            make_curve(|f| if f < 200.0 { 3.0 } else { 0.0 }),
        );

        let results = compute_voice_of_god(&curves, "C", SR, 20.0, 20000.0).unwrap();

        assert!(results["C"].is_reference);
        let l_result = results["L"]
            .alignment
            .as_ref()
            .expect("L should have corrections");
        // L has more bass than reference → needs negative lowshelf to cut bass
        assert!(
            l_result.lowshelf_gain_db < -0.3,
            "L should need LS cut, got {:.2}",
            l_result.lowshelf_gain_db
        );
        assert!(
            results["L"].timbre_spread_after_db.unwrap()
                < results["L"].timbre_spread_before_db.unwrap()
        );
    }

    #[test]
    fn test_vog_reference_not_found() {
        let mut curves = HashMap::new();
        curves.insert("L".to_string(), make_curve(|_| 0.0));

        let result = compute_voice_of_god(&curves, "NONEXISTENT", SR, 20.0, 20000.0);
        assert!(
            result.is_err(),
            "Should error when reference channel not found"
        );
    }

    #[test]
    fn test_vog_mismatched_grids_are_interpolated() {
        let mut curves = HashMap::new();
        curves.insert(
            "C".to_string(),
            make_curve_on_grid(vec![20.0, 50.0, 100.0, 500.0, 2_000.0, 10_000.0], |_| 0.0),
        );
        curves.insert(
            "L".to_string(),
            make_curve_on_grid(
                vec![25.0, 40.0, 80.0, 200.0, 1_000.0, 5_000.0, 12_000.0],
                |f| if f < 200.0 { 3.0 } else { 0.0 },
            ),
        );

        let results = compute_voice_of_god(&curves, "C", SR, 20.0, 20_000.0).unwrap();
        assert_ne!(results["L"].status, VoGChannelStatus::Failed);
        assert!(
            results["L"]
                .advisories
                .contains(&"frequency_grid_interpolated".to_string())
        );
    }

    #[test]
    fn test_vog_invalid_reference_is_structured_error() {
        let mut curves = HashMap::new();
        curves.insert(
            "C".to_string(),
            Curve {
                freq: Array1::from(vec![100.0, 200.0]),
                spl: Array1::from(vec![0.0]),
                ..Default::default()
            },
        );

        let error = compute_voice_of_god(&curves, "C", SR, 20.0, 20_000.0).unwrap_err();
        assert!(error.to_string().contains("invalid frequency/SPL data"));
    }

    #[test]
    fn test_vog_single_channel() {
        // Only the reference exists → no corrections needed
        let mut curves = HashMap::new();
        curves.insert("C".to_string(), make_curve(|_| 0.0));

        let results = compute_voice_of_god(&curves, "C", SR, 20.0, 20000.0).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results["C"].is_reference);
        assert!(results["C"].alignment.is_none());
    }

    #[test]
    fn test_vog_three_channels() {
        // L (reference, flat), C (+2dB treble), R (+3dB bass)
        // Verify corrections push C and R toward L
        let mut curves = HashMap::new();
        curves.insert("L".to_string(), make_curve(|_| 0.0));
        curves.insert(
            "C".to_string(),
            make_curve(|f| if f > 4000.0 { 2.0 } else { 0.0 }),
        );
        curves.insert(
            "R".to_string(),
            make_curve(|f| if f < 200.0 { 3.0 } else { 0.0 }),
        );

        let results = compute_voice_of_god(&curves, "L", SR, 20.0, 20000.0).unwrap();

        assert!(results["L"].is_reference);

        // C has excess treble → needs negative highshelf
        let c_align = results["C"]
            .alignment
            .as_ref()
            .expect("C should have corrections");
        assert!(
            c_align.highshelf_gain_db < -0.3,
            "C should need HS cut for excess treble, got {:.2}",
            c_align.highshelf_gain_db
        );

        // R has excess bass → needs negative lowshelf
        let r_align = results["R"]
            .alignment
            .as_ref()
            .expect("R should have corrections");
        assert!(
            r_align.lowshelf_gain_db < -0.3,
            "R should need LS cut for excess bass, got {:.2}",
            r_align.lowshelf_gain_db
        );
    }

    #[test]
    fn timbre_matching_rejects_candidate_without_required_metric_improvement() {
        let curves = HashMap::from([
            ("C".to_string(), make_curve(|_| 0.0)),
            (
                "L".to_string(),
                make_curve(|frequency| if frequency < 200.0 { 1.0 } else { 0.0 }),
            ),
        ]);
        let results = compute_inter_channel_timbre_matching_with_threshold(
            &curves, "C", SR, 20.0, 20_000.0, 100.0,
        )
        .unwrap();
        assert_eq!(results["L"].status, TimbreMatchingChannelStatus::Skipped);
        assert!(results["L"].alignment.is_none());
        assert!(
            results["L"]
                .advisories
                .contains(&"insufficient_timbre_spread_improvement".to_string())
        );
    }

    #[test]
    fn normalized_timbre_spread_uses_only_measured_overlap() {
        let channel = make_curve_on_grid(vec![20.0, 50.0, 100.0, 200.0], |frequency| {
            if frequency < 50.0 { 100.0 } else { 0.0 }
        });
        let reference = make_curve_on_grid(vec![50.0, 100.0, 200.0, 500.0], |_| 0.0);

        assert_eq!(
            pairwise_normalized_timbre_spread_db(&channel, &reference, 20.0, 500.0),
            Some(0.0)
        );
    }

    #[test]
    fn timbre_matching_rejects_non_finite_improvement_threshold() {
        let curves = HashMap::from([
            ("C".to_string(), make_curve(|_| 0.0)),
            ("L".to_string(), make_curve(|_| 1.0)),
        ]);

        let error = compute_inter_channel_timbre_matching_with_threshold(
            &curves,
            "C",
            SR,
            20.0,
            20_000.0,
            f64::NAN,
        )
        .unwrap_err();
        assert!(error.to_string().contains("min_improvement_db"));
    }
}
