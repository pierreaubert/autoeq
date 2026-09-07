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
) -> std::result::Result<Option<f64>, String> {
    let curves = load_source_individual_with_frequency_samples(source, frequency_samples)
        .map_err(|error| error.to_string())?;
    if curves.len() < 2 {
        return Ok(None);
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
        return Ok(None);
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
    Ok(Some(mean_std))
}

/// Build spatial-robustness advisories for a supporting-source measurement.
fn spatial_robustness_advisories_with_frequency_samples(
    source: &MeasurementSource,
    band_hz: (f64, f64),
    frequency_samples: usize,
) -> Vec<String> {
    match spatial_variance_db_with_frequency_samples(source, band_hz, frequency_samples) {
        Ok(Some(var_db)) if var_db > 6.0 => vec!["high_spatial_variance".to_string()],
        Ok(Some(var_db)) if var_db > 3.0 => vec!["moderate_spatial_variance".to_string()],
        Ok(Some(_)) => Vec::new(),
        Ok(None) => vec!["single_position_measurement".to_string()],
        Err(_) => vec!["position_load_failed".to_string()],
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

fn anchor_supporting_source_target_to_primary(
    target: &mut Curve,
    primary: &Curve,
    min_freq: f64,
    max_freq: f64,
) {
    let primary_level = roomeq_engine::analysis::response_metrics::mean_response_in_range(
        primary, min_freq, max_freq,
    );
    let target_level = roomeq_engine::analysis::response_metrics::mean_response_in_range(
        target, min_freq, max_freq,
    );
    if primary_level.is_finite() && target_level.is_finite() && target_level.abs() <= 20.0 {
        target.spl.mapv_inplace(|level| level + primary_level);
    } else if target_level.is_finite() {
        log::info!(
            "  Supporting-source target mean {:.1} dB looks absolute; keeping its level anchor",
            target_level
        );
    }
}

fn resolve_support_delay(
    config: &roomeq_model::SupportingSourceConfig,
    allow_delay: bool,
) -> Result<f64> {
    let invalid = |message: &str| AutoeqError::InvalidConfiguration {
        message: message.into(),
    };
    if !config.delay_ms.is_finite()
        || config.delay_ms <= 0.0
        || !config.max_coherent_cancellation_db.is_finite()
        || config.max_coherent_cancellation_db < 0.0
    {
        return Err(invalid(
            "supporting-source delay and cancellation budget must be finite and valid",
        ));
    }
    let offset = match config.acoustic_arrival_offset_ms {
        Some(offset) if offset.is_finite() => offset,
        None if config.allow_unverified_acoustics => 0.0,
        _ => {
            return Err(invalid(
                "supporting source needs finite acoustic_arrival_offset_ms from a common time reference, or explicit allow_unverified_acoustics",
            ));
        }
    };
    let electrical = config.delay_ms - offset;
    if !electrical.is_finite() || electrical < 0.0 {
        return Err(invalid(
            "supporting-source requested relative arrival requires a noncausal advance; change placement or timing request",
        ));
    }
    if !allow_delay && electrical > 1e-9 {
        return Err(invalid(
            "supporting-source arrival requirement conflicts with allow_delay=false",
        ));
    }
    Ok(electrical)
}

fn coherent_support_sum(
    primary: &Curve,
    support: &Curve,
    band: (f64, f64),
) -> Result<(Curve, f64)> {
    primary.validate("coherent supporting-source primary")?;
    support.validate("coherent supporting-source support")?;
    let primary_on_grid = autoeq_measurements::read::interpolate_log_space(&support.freq, primary);
    let (Some(pp), Some(sp)) = (&primary_on_grid.phase, &support.phase) else {
        return Err(AutoeqError::InvalidMeasurement {
            message: "coherent support sum requires both measured phases".into(),
        });
    };
    let mut freq = Vec::new();
    let mut spl = Vec::new();
    let mut phase = Vec::new();
    let mut worst = 0.0_f64;
    for i in 0..support.freq.len() {
        let f = support.freq[i];
        if f < band.0 || f > band.1 || f < primary.freq[0] || f > *primary.freq.last().unwrap() {
            continue;
        }
        let p = num_complex::Complex64::from_polar(
            10.0_f64.powf(primary_on_grid.spl[i] / 20.0),
            pp[i].to_radians(),
        );
        let s = num_complex::Complex64::from_polar(
            10.0_f64.powf(support.spl[i] / 20.0),
            sp[i].to_radians(),
        );
        let sum = p + s;
        let level = 20.0 * sum.norm().max(1e-30).log10();
        if !level.is_finite() || !pp[i].is_finite() || !sp[i].is_finite() {
            return Err(AutoeqError::InvalidMeasurement {
                message: "nonfinite coherent supporting-source evidence".into(),
            });
        }
        worst = worst.max(primary_on_grid.spl[i].max(support.spl[i]) - level);
        freq.push(f);
        spl.push(level);
        phase.push(sum.arg().to_degrees());
    }
    if freq.len() < 3 {
        return Err(AutoeqError::InvalidMeasurement {
            message: "insufficient coherent supporting-source overlap".into(),
        });
    }
    Ok((
        Curve {
            freq: freq.into(),
            spl: spl.into(),
            phase: Some(phase.into()),
            ..Default::default()
        },
        worst,
    ))
}

fn deployed_fir_coefficients(normalized_taps: &[f64], gain_db: f64) -> Vec<f64> {
    let linear_gain = 10.0_f64.powf(gain_db / 20.0);
    normalized_taps
        .iter()
        .map(|tap| tap * linear_gain)
        .collect()
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

    primary.validate("supporting-source primary measurement")?;
    support.validate("supporting-source support measurement")?;
    let applied_delay_ms = resolve_support_delay(
        &group.supporting_source,
        room_config.optimizer.allow_delay(),
    )?;
    let coherent = group.supporting_source.shared_phase_reference
        && primary.phase.is_some()
        && support.phase.is_some();
    if !coherent && !group.supporting_source.allow_unverified_acoustics {
        return Err(AutoeqError::InvalidConfiguration {
            message: "supporting source requires measured phases with shared_phase_reference, or explicit allow_unverified_acoustics".into(),
        });
    }
    let mut target = resolve_supporting_source_target(group, room_config)?;
    // Supporting-source gain is solved in an absolute pressure frame.  Room
    // targets, however, are conventionally level-relative (flat = 0 dB),
    // while loaded measurements retain their calibrated SPL (typically
    // ~80 dB).  Anchor the target to the primary's measured mean so the
    // subtraction does not interpret a valid target as silence.
    anchor_supporting_source_target_to_primary(
        &mut target,
        &primary,
        room_config.optimizer.min_freq,
        room_config.optimizer.max_freq,
    );

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
    let collides_with_logical_channel = room_config.system.as_ref().is_some_and(|system| {
        system.speakers.contains_key(&support_name)
            || system
                .subwoofers
                .as_ref()
                .is_some_and(|subwoofers| subwoofers.mapping.contains_key(&support_name))
    });
    if support_name == logical_role || collides_with_logical_channel {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "Supporting-source output suffix must produce a distinct, unused channel name for '{logical_role}', got '{support_name}'"
            ),
        });
    }
    let wav_name = format!("{}_fir.wav", support_name);
    let wav_path = output_dir.join(&wav_name);

    let wav_relative = wav_name; // the DSP chain references the file by basename

    let (primary_chain, mut support_chain) =
        roomeq_engine::output::build_supporting_source_dsp_chains(
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

    // Replay the realized FIR, normalization gain, and electrical delay.
    let mut support_final_curve = crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
        &support_chain,
        &support,
        sample_rate,
        output_dir,
        &HashMap::from([(wav_relative.clone(), filter.taps.clone())]),
    )?;
    if support.phase.is_none() {
        support_final_curve.phase = None;
    }
    let coherent_evidence = if coherent {
        Some(coherent_support_sum(
            &primary,
            &support_final_curve,
            group.supporting_source.freq_range_hz,
        )?)
    } else {
        None
    };
    if let Some((_, dip)) = &coherent_evidence
        && *dip > group.supporting_source.max_coherent_cancellation_db
        && !group.supporting_source.allow_unverified_acoustics
    {
        return Err(AutoeqError::OptimizationFailed {
            message: format!(
                "supporting-source coherent cancellation {dip:.2} dB exceeds {:.2} dB budget",
                group.supporting_source.max_coherent_cancellation_db
            ),
        });
    }
    math_audio_iir_fir::save_fir_to_wav(&filter.taps, sample_rate as u32, &wav_path).map_err(
        |e| AutoeqError::InvalidConfiguration {
            message: format!("Failed to write supporting-source FIR: {}", e),
        },
    )?;
    support_chain.final_curve = Some((&support_final_curve).into());
    let deployed_fir_coeffs = deployed_fir_coefficients(&filter.taps, filter.normalization_gain_db);

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
        fir_coeffs: Some(deployed_fir_coeffs),
        optimizer_evidence: Vec::new(),
    };

    let band_hz = group.supporting_source.freq_range_hz;
    let mut advisories = vec![
        "power_average_design_is_not_coherent_sum".to_string(),
        "reference_seat_timing_excludes_fir_energy_spread_and_requires_listening_validation"
            .to_string(),
        "primary_eq_bypassed_to_preserve_direct_sound".to_string(),
        "scores_not_computed_for_supporting_source".to_string(),
    ];
    if !coherent {
        advisories.push("coherent_sum_unverified".into());
    }
    if group.supporting_source.acoustic_arrival_offset_ms.is_none() {
        advisories.push("acoustic_arrival_unverified_electrical_delay_only".into());
    }
    if group.supporting_source.allow_unverified_acoustics {
        advisories.push("experimental_acoustics_explicitly_acknowledged".into());
    }
    if coherent_evidence
        .as_ref()
        .is_some_and(|(_, dip)| *dip > group.supporting_source.max_coherent_cancellation_db)
    {
        advisories.push("coherent_cancellation_budget_exceeded".into());
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
        summation_model: "power_average_design".into(),
        propagation_relative_arrival_ms: group
            .supporting_source
            .acoustic_arrival_offset_ms
            .map(|offset| offset + applied_delay_ms),
        coherent_sum: coherent_evidence.as_ref().map(|(curve, _)| curve.into()),
        max_coherent_cancellation_db: coherent_evidence.as_ref().map(|(_, dip)| *dip),
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
    fn spatial_advisory_distinguishes_measurement_load_failure() {
        let source = MeasurementSource::Single(autoeq_core::MeasurementSingle {
            measurement: autoeq_core::MeasurementRef::Path(
                "/definitely/not/a/real/supporting-source.csv".into(),
            ),
            speaker_name: None,
        });

        assert_eq!(
            spatial_robustness_advisories_with_frequency_samples(&source, (20.0, 200.0), 32),
            vec!["position_load_failed".to_string()]
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
                // Synthetic fixture has no shared-time acoustic evidence.
                allow_unverified_acoustics: true,
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
                // Synthetic fixture has no shared-time acoustic evidence.
                allow_unverified_acoustics: true,
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
    fn supporting_source_target_is_anchored_to_primary_spl() {
        let primary = flat_curve(82.0);
        let mut target = flat_curve(0.0);
        anchor_supporting_source_target_to_primary(&mut target, &primary, 20.0, 20_000.0);
        assert!(target.spl.iter().all(|level| (*level - 82.0).abs() < 1e-4));
    }

    #[test]
    fn absolute_supporting_source_target_keeps_its_level_anchor() {
        let primary = flat_curve(82.0);
        let mut target = flat_curve(75.0);

        anchor_supporting_source_target_to_primary(&mut target, &primary, 20.0, 20_000.0);

        assert!(target.spl.iter().all(|level| (*level - 75.0).abs() < 1e-12));
    }

    #[test]
    fn deployed_fir_evidence_includes_the_normalization_gain_stage() {
        let gain_db = 20.0 * 2.0_f64.log10();

        assert_eq!(
            deployed_fir_coefficients(&[0.5, -1.0, 0.25], gain_db),
            vec![1.0, -2.0, 0.5]
        );
    }

    #[test]
    fn process_channel_emits_chains_results_and_report() {
        let primary_curve = flat_curve(80.0);
        let support_curve = flat_curve(80.0);
        let group = SupportingSourceGroup {
            name: "test".to_string(),
            speaker_name: None,
            primary: MeasurementSource::InMemory(primary_curve.clone()),
            support: MeasurementSource::InMemory(support_curve.clone()),
            supporting_source: SupportingSourceConfig {
                // Synthetic fixture has no shared-time acoustic evidence.
                allow_unverified_acoustics: true,
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
        let mut room_config = room_config;
        room_config.optimizer.allow_delay = Some(true);
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
        assert!(
            support_chain
                .final_curve
                .as_ref()
                .is_some_and(|curve| !curve.spl.is_empty())
        );
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
    fn support_timing_requires_calibration_and_preserves_relative_arrival() {
        let mut config = SupportingSourceConfig::default();
        assert!(resolve_support_delay(&config, true).is_err());
        config.acoustic_arrival_offset_ms = Some(-10.0);
        assert_eq!(resolve_support_delay(&config, true).unwrap(), 20.0);
        assert!(resolve_support_delay(&config, false).is_err());
        config.acoustic_arrival_offset_ms = Some(10.0);
        assert_eq!(resolve_support_delay(&config, false).unwrap(), 0.0);
        config.acoustic_arrival_offset_ms = Some(11.0);
        assert!(resolve_support_delay(&config, true).is_err());
        config.acoustic_arrival_offset_ms = Some(f64::NAN);
        assert!(resolve_support_delay(&config, true).is_err());
    }

    #[test]
    fn unverified_acoustics_are_rejected_before_writing_an_artifact() {
        let mut group = SupportingSourceGroup {
            name: "unverified".into(),
            speaker_name: None,
            primary: MeasurementSource::InMemory(flat_curve(80.0)),
            support: MeasurementSource::InMemory(flat_curve(80.0)),
            supporting_source: SupportingSourceConfig::default(),
        };
        let room = RoomConfig {
            optimizer: OptimizerConfig {
                allow_delay: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };
        let directory = tempfile::tempdir().unwrap();
        let error =
            process_supporting_source_channel("L", &group, &room, 48_000.0, directory.path(), None)
                .unwrap_err();
        assert!(error.to_string().contains("acoustic_arrival_offset_ms"));
        group.supporting_source.acoustic_arrival_offset_ms = Some(0.0);
        group.supporting_source.shared_phase_reference = true;
        let error =
            process_supporting_source_channel("L", &group, &room, 48_000.0, directory.path(), None)
                .unwrap_err();
        assert!(error.to_string().contains("measured phases"));
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[test]
    fn coherent_sum_exposes_interference_not_power_average() {
        let mut primary = flat_curve(80.0);
        primary.phase = Some(ndarray::Array1::zeros(primary.freq.len()));
        let mut support = primary.clone();
        support.phase.as_mut().unwrap().fill(180.0);
        let (cancelled, dip) = coherent_support_sum(&primary, &support, (100.0, 1000.0)).unwrap();
        assert!(dip > 100.0);
        assert!(cancelled.spl.iter().all(|level| *level < 0.0));
        support.phase.as_mut().unwrap().fill(0.0);
        let (summed, dip) = coherent_support_sum(&primary, &support, (100.0, 1000.0)).unwrap();
        assert!(dip < 1e-9);
        assert!(
            summed
                .spl
                .iter()
                .all(|level| (*level - 86.0206).abs() < 1e-4)
        );
        support.phase = None;
        assert!(coherent_support_sum(&primary, &support, (100.0, 1000.0)).is_err());
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
                // Synthetic fixture has no shared-time acoustic evidence.
                allow_unverified_acoustics: true,
                delay_ms: 3.0,
                fir_taps: 128,
                decorrelation: SupportingSourceDecorrelation::None,
                ..Default::default()
            },
        };
        let output_dir = std::env::temp_dir();
        let error = process_supporting_source_channel(
            "L",
            &group,
            &room_config,
            48_000.0,
            &output_dir,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("allow_delay=false"));
    }
}
