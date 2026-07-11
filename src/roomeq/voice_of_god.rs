//! Voice of God (VoG) — timbre-match satellite channels to a reference channel.
//!
//! After per-channel EQ optimization, different speakers may have residual tonal
//! differences (e.g., one satellite has brighter timbre than another). VoG applies
//! broadband spectral alignment (lowshelf + highshelf + gain) to push each satellite
//! channel toward the reference channel's tonal balance.
//!
//! Reuses [`spectral_align::compute_target_alignment`] for the fitting.

use crate::Curve;
use crate::error::{AutoeqError, Result};
use log::info;
use std::collections::HashMap;

use super::spectral_align::{
    SpectralAlignmentResult, compute_target_alignment, create_alignment_plugins,
};
use super::types::PluginConfigWrapper;

/// Structured outcome for one channel in the timbre-matching stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoGChannelStatus {
    /// Reference channel; no correction is applied.
    Reference,
    /// A correction was computed.
    Applied,
    /// The channel needed no material correction.
    Skipped,
    /// Invalid channel data prevented a correction.
    Failed,
}

/// Result of VoG analysis for a single channel.
#[derive(Debug, Clone)]
pub struct VoGResult {
    /// Channel name
    pub channel_name: String,
    /// Spectral alignment corrections (None for the reference channel)
    pub alignment: Option<SpectralAlignmentResult>,
    /// Whether this channel is the reference
    pub is_reference: bool,
    /// Structured per-channel stage outcome.
    pub status: VoGChannelStatus,
    /// Machine-readable advisories for degraded or skipped paths.
    pub advisories: Vec<String>,
}

/// Compute Voice of God corrections for all channels.
///
/// For each non-reference channel, fits lowshelf + highshelf + flat gain to match
/// the reference channel's spectral shape. The reference channel itself receives
/// no corrections.
///
/// # Errors
/// Returns an error if `reference_channel` is not found in `corrected_curves`.
pub fn compute_voice_of_god(
    corrected_curves: &HashMap<String, Curve>,
    reference_channel: &str,
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
) -> Result<HashMap<String, VoGResult>> {
    let reference_curve = corrected_curves.get(reference_channel).ok_or_else(|| {
        AutoeqError::InvalidConfiguration {
            message: format!(
                "VoG reference channel '{}' not found. Available: {:?}",
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
                VoGResult {
                    channel_name: name.clone(),
                    alignment: None,
                    is_reference: true,
                    status: VoGChannelStatus::Reference,
                    advisories: vec!["reference_channel".to_string()],
                },
            );
            continue;
        }

        if let Err(error) = validate_curve(curve, "channel", name) {
            results.insert(
                name.clone(),
                VoGResult {
                    channel_name: name.clone(),
                    alignment: None,
                    is_reference: false,
                    status: VoGChannelStatus::Failed,
                    advisories: vec![error.to_string()],
                },
            );
            continue;
        }

        let grids_differ =
            !super::frequency_grid::same_frequency_grid(&curve.freq, &reference_curve.freq);
        // The target-alignment helper interpolates the reference onto this
        // channel's grid and restricts the fit to their measured overlap.
        let alignment =
            compute_target_alignment(curve, reference_curve, min_freq, max_freq, sample_rate);
        let status = if alignment.is_some() {
            VoGChannelStatus::Applied
        } else {
            VoGChannelStatus::Skipped
        };
        let mut advisories = Vec::new();
        if grids_differ {
            advisories.push("frequency_grid_interpolated".to_string());
        }
        if alignment.is_none() {
            advisories.push("no_material_correction".to_string());
        }

        results.insert(
            name.clone(),
            VoGResult {
                channel_name: name.clone(),
                alignment,
                is_reference: false,
                status,
                advisories,
            },
        );
    }

    results
        .values()
        .filter(|r| !r.is_reference && r.alignment.is_some())
        .for_each(|r| {
            let a = r.alignment.as_ref().unwrap();
            info!(
                "  VoG '{}': LS={:+.2} dB, HS={:+.2} dB, gain={:+.2} dB (residual {:.2} dB RMS)",
                r.channel_name,
                a.lowshelf_gain_db,
                a.highshelf_gain_db,
                a.flat_gain_db,
                a.residual_rms_db,
            );
        });

    Ok(results)
}

fn validate_curve(curve: &Curve, role: &str, name: &str) -> Result<()> {
    if !super::frequency_grid::is_valid_frequency_grid(&curve.freq)
        || curve.spl.len() != curve.freq.len()
        || curve.spl.iter().any(|value| !value.is_finite())
    {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!("VoG {role} channel '{name}' has invalid frequency/SPL data"),
        });
    }
    Ok(())
}

/// Create DSP plugins for a VoG correction result.
///
/// Returns a list of plugins (EQ with shelves, gain) to apply to the channel.
/// Returns an empty list for the reference channel or channels with negligible corrections.
pub fn create_vog_plugins(result: &VoGResult, sample_rate: f64) -> Vec<PluginConfigWrapper> {
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

#[cfg(test)]
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
}
