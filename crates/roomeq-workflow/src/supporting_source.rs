//! Supporting-source room-compensation processing.

use crate::measurement::{
    load_source_individual_with_frequency_samples, load_source_with_frequency_samples,
};
use autoeq_measurements::read::{interpolate_log_space, read_curve_from_csv};
use roomeq_engine::Curve;
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_engine::room_result::ChannelOptimizationResult;
use roomeq_engine::supporting_source::{compute_supporting_source_filter, db_summary};
use roomeq_model::{
    ChannelDspChain, MeasurementSource, OptimizationMetadata, RoomConfig, StatisticalSummary,
    SupportingSourceGroup, SupportingSourceOutputNaming, SupportingSourceReport, TargetCurveConfig,
};
use std::collections::HashMap;
use std::path::Path;

/// Compute mean per-frequency standard deviation (in dB) across multiple
/// measurement positions inside the compensation band.
fn spatial_variance_db_with_frequency_samples(
    source: &MeasurementSource,
    band_hz: (f64, f64),
    frequency_samples: usize,
) -> Option<f64> {
    let curves = load_source_individual_with_frequency_samples(source, frequency_samples).ok()?;
    if curves.len() < 2 {
        return None;
    }
    let ref_freqs = curves[0].freq.clone();
    let interpolated: Vec<Curve> = curves
        .iter()
        .map(|c| interpolate_log_space(&ref_freqs, c))
        .collect();
    let in_band: Vec<usize> = ref_freqs
        .iter()
        .enumerate()
        .filter(|&(_, f)| *f >= band_hz.0 && *f <= band_hz.1)
        .map(|(i, _)| i)
        .collect();
    if in_band.is_empty() {
        return None;
    }
    let per_freq_std: Vec<f64> = in_band
        .iter()
        .map(|&i| {
            let values: Vec<f64> = interpolated.iter().map(|c| c.spl[i]).collect();
            let (mean, _std) = db_summary(&values);
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            var.sqrt()
        })
        .collect();
    let (mean_std, _) = db_summary(&per_freq_std);
    Some(mean_std)
}

/// Build spatial-robustness advisories for a supporting-source measurement.
fn spatial_robustness_advisories_with_frequency_samples(
    source: &MeasurementSource,
    band_hz: (f64, f64),
    frequency_samples: usize,
) -> Vec<String> {
    match spatial_variance_db_with_frequency_samples(source, band_hz, frequency_samples) {
        Some(var_db) if var_db > 6.0 => vec!["high_spatial_variance".to_string()],
        Some(var_db) if var_db > 3.0 => vec!["moderate_spatial_variance".to_string()],
        Some(_) => Vec::new(),
        None => vec!["single_position_measurement".to_string()],
    }
}

/// Compute the target curve for a supporting-source channel.
///
/// Returns the target curve as a `Curve`. The target is resolved from:
/// 1. `group.supporting_source.target_response` if set.
/// 2. `room_config.target_curve` otherwise.
/// 3. A flat 0 dB fallback if neither is set.
pub fn resolve_supporting_source_target(
    group: &SupportingSourceGroup,
    room_config: &RoomConfig,
) -> Result<Curve> {
    if let Some(ref target_name) = group.supporting_source.target_response {
        // Target name is a reference to a target curve. For now we only support
        // the room-level target_curve by special name.
        if target_name == "target_curve" {
            return resolve_room_target(room_config);
        }
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "Unsupported supporting_source.target_response '{}'. Use 'target_curve' or omit.",
                target_name
            ),
        });
    }
    resolve_room_target(room_config)
}

fn resolve_room_target(room_config: &RoomConfig) -> Result<Curve> {
    let flat_target = || {
        let freq = ndarray::Array1::logspace(10.0, f64::log10(20.0), f64::log10(20000.0), 200);
        Curve {
            freq,
            spl: ndarray::Array1::from_elem(200, 0.0),
            ..Default::default()
        }
    };

    let Some(config) = room_config.target_curve.as_ref() else {
        return Ok(flat_target());
    };

    match config {
        TargetCurveConfig::Predefined(name) => {
            let canonical = match name.to_ascii_lowercase().as_str() {
                "flat" => "flat",
                "harman" => "harman",
                "listening window" => "Listening Window",
                "sound power" => "Sound Power",
                "early reflections" => "Early Reflections",
                "estimated in-room response" => "Estimated In-Room Response",
                _ => {
                    return Err(AutoeqError::InvalidConfiguration {
                        message: format!(
                            "Unsupported predefined target '{name}' for supporting source"
                        ),
                    });
                }
            };
            let reference = flat_target();
            Ok(roomeq_engine::build_target_curve_by_name(
                canonical,
                &reference.freq,
                &reference,
            ))
        }
        TargetCurveConfig::Path(path) => {
            read_curve_from_csv(path).map_err(|e| AutoeqError::InvalidMeasurement {
                message: format!("Failed to read target curve: {}", e),
            })
        }
    }
}

/// Compute the support output channel name from a logical role.
pub fn support_channel_name(
    logical_role: &str,
    naming: Option<&SupportingSourceOutputNaming>,
) -> String {
    let suffix = naming.map(|n| n.suffix.as_str()).unwrap_or("_support");
    format!("{}{}", logical_role, suffix)
}

/// Process a single supporting-source channel.
///
/// Loads primary/support measurements, computes the supporting-source filter,
/// writes the FIR to a WAV file, and returns the primary and support DSP chains
/// plus a report.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn process_supporting_source_channel(
    logical_role: &str,
    group: &SupportingSourceGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    output_dir: &Path,
    naming: Option<&SupportingSourceOutputNaming>,
) -> Result<(
    (ChannelDspChain, ChannelDspChain),
    (ChannelOptimizationResult, ChannelOptimizationResult),
    SupportingSourceReport,
)> {
    process_supporting_source_channel_with_frequency_samples(
        logical_role,
        group,
        room_config,
        sample_rate,
        output_dir,
        naming,
        crate::DEFAULT_FREQUENCY_SAMPLES,
    )
}

/// Process a supporting-source channel using a configurable RoomEQ frequency grid.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn process_supporting_source_channel_with_frequency_samples(
    logical_role: &str,
    group: &SupportingSourceGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    output_dir: &Path,
    naming: Option<&SupportingSourceOutputNaming>,
    frequency_samples: usize,
) -> Result<(
    (ChannelDspChain, ChannelDspChain),
    (ChannelOptimizationResult, ChannelOptimizationResult),
    SupportingSourceReport,
)> {
    let primary =
        load_source_with_frequency_samples(&group.primary, frequency_samples).map_err(|e| {
            AutoeqError::InvalidMeasurement {
                message: format!(
                    "Failed to load primary measurement for '{}': {}",
                    logical_role, e
                ),
            }
        })?;
    let support =
        load_source_with_frequency_samples(&group.support, frequency_samples).map_err(|e| {
            AutoeqError::InvalidMeasurement {
                message: format!(
                    "Failed to load support measurement for '{}': {}",
                    logical_role, e
                ),
            }
        })?;

    let target = resolve_supporting_source_target(group, room_config)?;

    let filter = compute_supporting_source_filter(
        &primary,
        &support,
        &target,
        &group.supporting_source,
        sample_rate,
    )
    .map_err(|e| AutoeqError::OptimizationFailed {
        message: format!(
            "Supporting-source filter failed for '{}': {}",
            logical_role, e
        ),
    })?;

    // Write FIR to WAV.
    let support_name = support_channel_name(logical_role, naming);
    let wav_name = format!("{}_fir.wav", support_name);
    let wav_path = output_dir.join(&wav_name);
    math_audio_iir_fir::save_fir_to_wav(&filter.taps, sample_rate as u32, &wav_path).map_err(
        |e| AutoeqError::InvalidConfiguration {
            message: format!("Failed to write supporting-source FIR: {}", e),
        },
    )?;

    let wav_relative = wav_name; // the DSP chain references the file by basename

    let applied_delay_ms = if room_config.optimizer.allow_delay() {
        group.supporting_source.delay_ms
    } else {
        0.0
    };
    let (primary_chain, support_chain) = roomeq_engine::output::build_supporting_source_dsp_chains(
        logical_role,
        &support_name,
        applied_delay_ms,
        filter.normalization_gain_db,
        &wav_relative,
        Some(&primary),
        Some(&support),
        Some(&filter.constrained_target),
    );

    let drr_before_db = filter.drr_before_db.as_deref().map(|values| {
        let (mean, std) = db_summary(values);
        StatisticalSummary { mean, std }
    });
    let drr_after_db = filter.drr_after_db.as_deref().map(|values| {
        let (mean, std) = db_summary(values);
        StatisticalSummary { mean, std }
    });

    let gain_curve = Curve {
        freq: filter.constrained_target.freq.clone(),
        spl: ndarray::Array1::from(filter.support_gain_db.clone()),
        ..Default::default()
    };
    let gain_on_support_grid =
        autoeq_measurements::read::interpolate_log_space(&support.freq, &gain_curve);
    let support_final_spl = &support.spl + &gain_on_support_grid.spl;
    let support_final_curve = Curve {
        freq: support.freq.clone(),
        spl: support_final_spl,
        ..Default::default()
    };

    let primary_result = ChannelOptimizationResult {
        name: logical_role.to_string(),
        pre_score: 0.0,
        post_score: 0.0,
        initial_curve: primary.clone(),
        final_curve: primary.clone(),
        biquads: Vec::new(),
        fir_coeffs: None,
        optimizer_evidence: Vec::new(),
    };
    let support_result = ChannelOptimizationResult {
        name: support_name.clone(),
        pre_score: 0.0,
        post_score: 0.0,
        initial_curve: support.clone(),
        final_curve: support_final_curve,
        biquads: Vec::new(),
        fir_coeffs: Some(filter.taps.clone()),
        optimizer_evidence: Vec::new(),
    };

    let band_hz = group.supporting_source.freq_range_hz;
    let mut advisories = vec![
        "primary_eq_bypassed_to_preserve_direct_sound".to_string(),
        "scores_not_computed_for_supporting_source".to_string(),
    ];
    if applied_delay_ms != group.supporting_source.delay_ms {
        advisories.push("support_delay_disabled_by_allow_delay".to_string());
    }
    advisories.extend(
        spatial_robustness_advisories_with_frequency_samples(
            &group.primary,
            band_hz,
            frequency_samples,
        )
        .into_iter()
        .map(|a| format!("primary:{}", a)),
    );
    if drr_before_db.is_none() || drr_after_db.is_none() {
        advisories.push("drr_unavailable_without_time_gated_ir".to_string());
    }
    advisories.extend(
        spatial_robustness_advisories_with_frequency_samples(
            &group.support,
            band_hz,
            frequency_samples,
        )
        .into_iter()
        .map(|a| format!("support:{}", a)),
    );

    let report = SupportingSourceReport {
        enabled: true,
        primary_output: logical_role.to_string(),
        support_output: support_name,
        delay_ms: applied_delay_ms,
        fir_length: filter.taps.len(),
        compensation_band_hz: band_hz,
        drr_before_db,
        drr_after_db,
        target_constraints_active: filter.precedence_limit_hits > 0,
        precedence_limit_hits: filter.precedence_limit_hits,
        advisories,
    };

    Ok((
        (primary_chain, support_chain),
        (primary_result, support_result),
        report,
    ))
}

/// Merge a supporting-source report into optimization metadata.
pub fn merge_supporting_source_report(
    metadata: &mut OptimizationMetadata,
    logical_role: String,
    report: SupportingSourceReport,
) {
    let map = metadata.supporting_source.get_or_insert_with(HashMap::new);
    map.insert(logical_role, report);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use roomeq_model::{
        MeasurementSource, OptimizerConfig, SupportingSourceConfig, SupportingSourceDecorrelation,
        TargetCurveConfig, default_config_version,
    };

    fn flat_curve(spl_db: f64) -> Curve {
        Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20000.0), 64),
            spl: Array1::from_elem(64, spl_db),
            phase: None,
            ..Default::default()
        }
    }

    #[test]
    fn support_channel_name_uses_suffix() {
        assert_eq!(super::support_channel_name("L", None), "L_support");
        assert_eq!(
            super::support_channel_name(
                "L",
                Some(&SupportingSourceOutputNaming {
                    suffix: "_room".to_string()
                })
            ),
            "L_room"
        );
    }

    #[test]
    fn resolve_target_defaults_to_flat() {
        let group = SupportingSourceGroup {
            name: "test".to_string(),
            speaker_name: None,
            primary: MeasurementSource::InMemory(flat_curve(80.0)),
            support: MeasurementSource::InMemory(flat_curve(80.0)),
            supporting_source: SupportingSourceConfig::default(),
        };
        let room_config = RoomConfig {
            version: default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: None,
            crossovers: None,
            provenance: Default::default(),
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        };
        let target = resolve_supporting_source_target(&group, &room_config).unwrap();
        assert!(!target.freq.is_empty());
        assert!(target.spl.iter().all(|&v| (v - 0.0).abs() < 1e-9));
    }

    #[test]
    fn resolve_target_uses_room_target_curve() {
        let group = SupportingSourceGroup {
            name: "test".to_string(),
            speaker_name: None,
            primary: MeasurementSource::InMemory(flat_curve(80.0)),
            support: MeasurementSource::InMemory(flat_curve(80.0)),
            supporting_source: SupportingSourceConfig {
                target_response: Some("target_curve".to_string()),
                ..Default::default()
            },
        };
        let room_config = RoomConfig {
            version: default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: Some(TargetCurveConfig::Predefined("flat".to_string())),
            crossovers: None,
            provenance: Default::default(),
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        };
        let target = resolve_supporting_source_target(&group, &room_config).unwrap();
        assert!(target.spl.iter().all(|&v| (v - 0.0).abs() < 1e-9));
    }

    #[test]
    fn resolve_target_errors_on_unsupported_reference() {
        let group = SupportingSourceGroup {
            name: "test".to_string(),
            speaker_name: None,
            primary: MeasurementSource::InMemory(flat_curve(80.0)),
            support: MeasurementSource::InMemory(flat_curve(80.0)),
            supporting_source: SupportingSourceConfig {
                target_response: Some("unknown".to_string()),
                ..Default::default()
            },
        };
        let room_config = RoomConfig {
            version: default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: None,
            crossovers: None,
            provenance: Default::default(),
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        };
        assert!(resolve_supporting_source_target(&group, &room_config).is_err());
    }

    #[test]
    fn resolve_room_target_loads_csv_path() {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "frequency,spl").unwrap();
        for f in [20.0, 100.0, 1000.0, 20000.0] {
            writeln!(tmp, "{},0.0", f).unwrap();
        }
        tmp.flush().unwrap();
        let room_config = RoomConfig {
            version: default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: Some(TargetCurveConfig::Path(tmp.path().to_path_buf())),
            crossovers: None,
            provenance: Default::default(),
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        };
        let target = resolve_room_target(&room_config).unwrap();
        assert_eq!(target.freq.len(), 4);
    }

    #[test]
    fn resolve_room_target_supports_harman_predefined() {
        let room_config = RoomConfig {
            version: default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: Some(TargetCurveConfig::Predefined("harman".to_string())),
            crossovers: None,
            provenance: Default::default(),
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        };
        let target = resolve_room_target(&room_config).unwrap();
        let index_1khz = target
            .freq
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                (*left - 1_000.0).abs().total_cmp(&(*right - 1_000.0).abs())
            })
            .map(|(index, _)| index)
            .unwrap();
        assert!(target.spl[index_1khz].abs() < 0.1);
        assert!(target.spl[0] > target.spl[target.spl.len() - 1]);
    }

    #[test]
    fn process_channel_emits_chains_results_and_report() {
        let group = SupportingSourceGroup {
            name: "test".to_string(),
            speaker_name: None,
            primary: MeasurementSource::InMemory(flat_curve(80.0)),
            support: MeasurementSource::InMemory(flat_curve(80.0)),
            supporting_source: SupportingSourceConfig {
                delay_ms: 3.0,
                fir_taps: 128,
                decorrelation: SupportingSourceDecorrelation::None,
                ..Default::default()
            },
        };
        let room_config = RoomConfig {
            version: default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: None,
            crossovers: None,
            provenance: Default::default(),
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        };
        let output_dir = std::env::temp_dir();
        let ((primary_chain, support_chain), (primary_result, support_result), report) =
            process_supporting_source_channel(
                "L",
                &group,
                &room_config,
                48000.0,
                &output_dir,
                None,
            )
            .unwrap();

        assert_eq!(primary_chain.channel, "L");
        assert_eq!(support_chain.channel, "L_support");
        assert!(
            support_chain
                .plugins
                .iter()
                .any(|p| p.plugin_type == "convolution")
        );

        assert_eq!(primary_result.name, "L");
        assert_eq!(support_result.name, "L_support");
        assert_eq!(support_result.fir_coeffs.as_ref().unwrap().len(), 128);

        assert_eq!(report.primary_output, "L");
        assert_eq!(report.support_output, "L_support");
        assert_eq!(report.fir_length, 128);
        assert!(report.drr_before_db.is_none());
        assert!(report.drr_after_db.is_none());
        assert!(
            report
                .advisories
                .iter()
                .any(|advisory| advisory == "drr_unavailable_without_time_gated_ir")
        );
        assert!(
            report
                .advisories
                .iter()
                .any(|advisory| { advisory == "primary_eq_bypassed_to_preserve_direct_sound" })
        );
        assert!(
            report
                .advisories
                .iter()
                .any(|advisory| { advisory == "scores_not_computed_for_supporting_source" })
        );
    }

    #[test]
    fn supporting_source_obeys_allow_delay() {
        let mut room_config = RoomConfig {
            version: default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: None,
            crossovers: None,
            provenance: Default::default(),
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        };
        room_config.optimizer.allow_delay = Some(false);
        let group = SupportingSourceGroup {
            name: "test".to_string(),
            speaker_name: None,
            primary: MeasurementSource::InMemory(flat_curve(80.0)),
            support: MeasurementSource::InMemory(flat_curve(80.0)),
            supporting_source: SupportingSourceConfig {
                delay_ms: 3.0,
                fir_taps: 128,
                decorrelation: SupportingSourceDecorrelation::None,
                ..Default::default()
            },
        };
        let output_dir = std::env::temp_dir();
        let ((_, support_chain), _, report) = process_supporting_source_channel(
            "L",
            &group,
            &room_config,
            48_000.0,
            &output_dir,
            None,
        )
        .unwrap();
        assert_eq!(report.delay_ms, 0.0);
        assert!(
            !support_chain
                .plugins
                .iter()
                .any(|plugin| plugin.plugin_type == "delay")
        );
        assert!(
            report
                .advisories
                .iter()
                .any(|advisory| { advisory == "support_delay_disabled_by_allow_delay" })
        );
    }
}
