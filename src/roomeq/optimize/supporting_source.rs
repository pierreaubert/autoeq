//! Supporting-source room-compensation processing.

use crate::Curve;
use crate::error::{AutoeqError, Result};
use crate::roomeq::optimize::types::ChannelOptimizationResult;
use crate::roomeq::supporting_source::compute_supporting_source_filter;
use crate::roomeq::types::{
    ChannelDspChain, OptimizationMetadata, RoomConfig, StatisticalSummary, SupportingSourceGroup,
    SupportingSourceOutputNaming, SupportingSourceReport,
};
use std::collections::HashMap;
use std::path::Path;

/// Compute mean per-frequency standard deviation (in dB) across multiple
/// measurement positions inside the compensation band.
fn spatial_variance_db(source: &crate::MeasurementSource, band_hz: (f64, f64)) -> Option<f64> {
    let curves = crate::read::load_source_individual(source).ok()?;
    if curves.len() < 2 {
        return None;
    }
    let ref_freqs = curves[0].freq.clone();
    let interpolated: Vec<crate::Curve> = curves
        .iter()
        .map(|c| crate::read::interpolate_log_space(&ref_freqs, c))
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
            let (mean, _std) = crate::roomeq::supporting_source::db_summary(&values);
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            var.sqrt()
        })
        .collect();
    let (mean_std, _) = crate::roomeq::supporting_source::db_summary(&per_freq_std);
    Some(mean_std)
}

/// Build spatial-robustness advisories for a supporting-source measurement.
fn spatial_robustness_advisories(
    source: &crate::MeasurementSource,
    band_hz: (f64, f64),
) -> Vec<String> {
    match spatial_variance_db(source, band_hz) {
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
        crate::roomeq::types::TargetCurveConfig::Predefined(name) => {
            if name.eq_ignore_ascii_case("flat") {
                Ok(flat_target())
            } else {
                Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "Predefined target '{}' not yet supported for supporting source",
                        name
                    ),
                })
            }
        }
        crate::roomeq::types::TargetCurveConfig::Path(path) => {
            crate::read::read_curve_from_csv(path).map_err(|e| AutoeqError::InvalidMeasurement {
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
    let primary =
        crate::read::load_source(&group.primary).map_err(|e| AutoeqError::InvalidMeasurement {
            message: format!(
                "Failed to load primary measurement for '{}': {}",
                logical_role, e
            ),
        })?;
    let support =
        crate::read::load_source(&group.support).map_err(|e| AutoeqError::InvalidMeasurement {
            message: format!(
                "Failed to load support measurement for '{}': {}",
                logical_role, e
            ),
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
    crate::fir::save_fir_to_wav(&filter.taps, sample_rate as u32, &wav_path).map_err(|e| {
        AutoeqError::InvalidConfiguration {
            message: format!("Failed to write supporting-source FIR: {}", e),
        }
    })?;

    let wav_relative = wav_name; // the DSP chain references the file by basename

    let (primary_chain, support_chain) = crate::roomeq::output::build_supporting_source_dsp_chains(
        logical_role,
        &support_name,
        group.supporting_source.delay_ms,
        &wav_relative,
        Some(&primary),
        Some(&support),
        Some(&filter.constrained_target),
    );

    let (drr_before_mean, drr_before_std) =
        crate::roomeq::supporting_source::db_summary(&filter.drr_before_db);
    let (drr_after_mean, drr_after_std) =
        crate::roomeq::supporting_source::db_summary(&filter.drr_after_db);

    let support_final_spl: ndarray::Array1<f64> = support
        .spl
        .iter()
        .zip(&filter.support_gain_db)
        .map(|(&s, &g)| s + g)
        .collect();
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
    };
    let support_result = ChannelOptimizationResult {
        name: support_name.clone(),
        pre_score: 0.0,
        post_score: 0.0,
        initial_curve: support.clone(),
        final_curve: support_final_curve,
        biquads: Vec::new(),
        fir_coeffs: Some(filter.taps.clone()),
    };

    let band_hz = group.supporting_source.freq_range_hz;
    let mut advisories = Vec::new();
    advisories.extend(
        spatial_robustness_advisories(&group.primary, band_hz)
            .into_iter()
            .map(|a| format!("primary:{}", a)),
    );
    advisories.extend(
        spatial_robustness_advisories(&group.support, band_hz)
            .into_iter()
            .map(|a| format!("support:{}", a)),
    );

    let report = SupportingSourceReport {
        enabled: true,
        primary_output: logical_role.to_string(),
        support_output: support_name,
        delay_ms: group.supporting_source.delay_ms,
        fir_length: filter.taps.len(),
        compensation_band_hz: band_hz,
        drr_before_db: StatisticalSummary {
            mean: drr_before_mean,
            std: drr_before_std,
        },
        drr_after_db: StatisticalSummary {
            mean: drr_after_mean,
            std: drr_after_std,
        },
        target_constraints_active: filter.precedence_limit_hits > 0
            || group.supporting_source.precedence_limits.len() > 1,
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
    use crate::read::MeasurementSource;
    use crate::roomeq::types::{
        OptimizerConfig, SupportingSourceConfig, SupportingSourceDecorrelation,
    };
    use ndarray::Array1;

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
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: None,
            crossovers: None,
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
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: Some(crate::roomeq::types::TargetCurveConfig::Predefined(
                "flat".to_string(),
            )),
            crossovers: None,
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
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: None,
            crossovers: None,
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
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: Some(crate::roomeq::types::TargetCurveConfig::Path(
                tmp.path().to_path_buf(),
            )),
            crossovers: None,
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        };
        let target = resolve_room_target(&room_config).unwrap();
        assert_eq!(target.freq.len(), 4);
    }

    #[test]
    fn resolve_room_target_rejects_non_flat_predefined() {
        let room_config = RoomConfig {
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: Some(crate::roomeq::types::TargetCurveConfig::Predefined(
                "harman".to_string(),
            )),
            crossovers: None,
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        };
        assert!(resolve_room_target(&room_config).is_err());
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
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers: HashMap::new(),
            optimizer: OptimizerConfig::default(),
            target_curve: None,
            crossovers: None,
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
    }
}
