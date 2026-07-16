//! Supporting-source workflow helpers.

use crate::error::{AutoeqError, Result};
use crate::roomeq::optimize::supporting_source::{
    merge_supporting_source_report, process_supporting_source_channel,
};
use crate::roomeq::types::{OptimizationMetadata, RoomConfig, SpeakerConfig, SystemConfig};
use log::info;
use std::path::Path;

/// Partition logical roles into single-source channels and supporting-source groups.
#[allow(clippy::type_complexity)]
pub(in super::super) fn partition_roles(
    config: &RoomConfig,
    sys: &SystemConfig,
) -> Result<(
    Vec<String>,
    Vec<(String, crate::roomeq::types::SupportingSourceGroup)>,
)> {
    let mut single_roles = Vec::new();
    let mut supporting = Vec::new();

    for role in sys.speakers.keys() {
        let meas_key = sys
            .speakers
            .get(role)
            .ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: format!("Missing speaker mapping for '{}'", role),
            })?;
        let cfg =
            config
                .speakers
                .get(meas_key)
                .ok_or_else(|| AutoeqError::InvalidConfiguration {
                    message: format!("Missing speaker config for key '{}'", meas_key),
                })?;
        match cfg {
            SpeakerConfig::Single(_) => single_roles.push(role.clone()),
            SpeakerConfig::SupportingSource(group) => {
                supporting.push((role.clone(), group.clone()));
            }
            _ => {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "Workflow cannot handle speaker config variant for '{}'",
                        role
                    ),
                });
            }
        }
    }

    Ok((single_roles, supporting))
}

/// Load single-source curves for alignment, skipping supporting-source channels.
pub(in super::super) fn load_single_source_curves(
    config: &RoomConfig,
    sys: &SystemConfig,
    single_roles: &[String],
) -> Result<std::collections::HashMap<String, crate::Curve>> {
    let mut curves = std::collections::HashMap::new();
    for role in single_roles {
        let source = super::bass_management::resolve_single_source(role, config, sys)?;
        let curve =
            crate::read::load_source(source).map_err(|e| AutoeqError::InvalidMeasurement {
                message: e.to_string(),
            })?;
        curves.insert(role.clone(), curve);
    }
    Ok(curves)
}

/// Process all supporting-source groups and attach their outputs to the result.
pub(in super::super) fn process_supporting_source_channels(
    config: &RoomConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: &Path,
    channel_chains: &mut std::collections::HashMap<String, crate::roomeq::types::ChannelDspChain>,
    channel_results: &mut std::collections::HashMap<
        String,
        crate::roomeq::ChannelOptimizationResult,
    >,
    metadata: &mut OptimizationMetadata,
) -> Result<()> {
    let naming = sys.supporting_source_outputs.as_ref();
    for (role, group) in &sys
        .speakers
        .iter()
        .filter_map(|(role, meas_key)| {
            config.speakers.get(meas_key).and_then(|cfg| match cfg {
                SpeakerConfig::SupportingSource(g) => Some((role.clone(), g.clone())),
                _ => None,
            })
        })
        .collect::<Vec<_>>()
    {
        info!(
            "  Processing supporting-source channel '{}' -> support output",
            role
        );

        let ((primary_chain, support_chain), (primary_result, support_result), report) =
            process_supporting_source_channel(
                role,
                group,
                config,
                sample_rate,
                output_dir,
                naming,
            )?;

        channel_chains.insert(role.clone(), primary_chain);
        channel_results.insert(role.clone(), primary_result);
        let support_name = report.support_output.clone();
        channel_chains.insert(support_name.clone(), support_chain);
        channel_results.insert(support_name, support_result);

        merge_supporting_source_report(metadata, role.clone(), report);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::MeasurementSource;
    use crate::roomeq::types::{
        OptimizerConfig, SupportingSourceConfig, SupportingSourceDecorrelation,
        SupportingSourceGroup, SystemModel,
    };
    use ndarray::Array1;

    fn flat_curve(spl_db: f64) -> crate::Curve {
        crate::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20000.0), 64),
            spl: Array1::from_elem(64, spl_db),
            phase: None,
            ..Default::default()
        }
    }

    fn room_config_with_speakers(
        speakers: std::collections::HashMap<String, SpeakerConfig>,
    ) -> RoomConfig {
        RoomConfig {
            version: crate::roomeq::types::default_config_version(),
            system: None,
            speakers,
            optimizer: OptimizerConfig::default(),
            target_curve: None,
            crossovers: None,
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        }
    }

    fn supporting_group() -> SupportingSourceGroup {
        SupportingSourceGroup {
            name: "Left pair".to_string(),
            speaker_name: None,
            primary: MeasurementSource::InMemory(flat_curve(80.0)),
            support: MeasurementSource::InMemory(flat_curve(80.0)),
            supporting_source: SupportingSourceConfig {
                delay_ms: 2.0,
                fir_taps: 128,
                decorrelation: SupportingSourceDecorrelation::None,
                ..Default::default()
            },
        }
    }

    #[test]
    fn partition_roles_separates_single_and_supporting() {
        let mut speakers = std::collections::HashMap::new();
        speakers.insert(
            "left".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(80.0))),
        );
        speakers.insert(
            "ss".to_string(),
            SpeakerConfig::SupportingSource(supporting_group()),
        );
        let config = room_config_with_speakers(speakers);
        let sys = SystemConfig {
            model: SystemModel::Stereo,
            speakers: std::collections::HashMap::from([
                ("L".to_string(), "left".to_string()),
                ("S".to_string(), "ss".to_string()),
            ]),
            ..Default::default()
        };
        let (single, supporting) = partition_roles(&config, &sys).unwrap();
        assert_eq!(single, vec!["L"]);
        assert_eq!(supporting.len(), 1);
        assert_eq!(supporting[0].0, "S");
    }

    #[test]
    fn partition_roles_errors_on_missing_config() {
        let config = room_config_with_speakers(std::collections::HashMap::new());
        let sys = SystemConfig {
            model: SystemModel::Stereo,
            speakers: std::collections::HashMap::from([("L".to_string(), "missing".to_string())]),
            ..Default::default()
        };
        assert!(partition_roles(&config, &sys).is_err());
    }

    #[test]
    fn partition_roles_errors_on_unsupported_variant() {
        let mut speakers = std::collections::HashMap::new();
        speakers.insert(
            "left".to_string(),
            SpeakerConfig::Group(crate::roomeq::types::SpeakerGroup {
                name: "group".to_string(),
                speaker_name: None,
                measurements: vec![],
                crossover: None,
            }),
        );
        let config = room_config_with_speakers(speakers);
        let sys = SystemConfig {
            model: SystemModel::Stereo,
            speakers: std::collections::HashMap::from([("L".to_string(), "left".to_string())]),
            ..Default::default()
        };
        assert!(partition_roles(&config, &sys).is_err());
    }

    #[test]
    fn load_single_source_curves_loads_curves() {
        let mut speakers = std::collections::HashMap::new();
        speakers.insert(
            "left".to_string(),
            SpeakerConfig::Single(MeasurementSource::InMemory(flat_curve(80.0))),
        );
        let config = room_config_with_speakers(speakers);
        let sys = SystemConfig {
            model: SystemModel::Stereo,
            speakers: std::collections::HashMap::from([("L".to_string(), "left".to_string())]),
            ..Default::default()
        };
        let curves = load_single_source_curves(&config, &sys, &["L".to_string()]).unwrap();
        assert!(curves.contains_key("L"));
    }

    #[test]
    fn process_supporting_source_channels_adds_outputs_and_metadata() {
        let mut speakers = std::collections::HashMap::new();
        speakers.insert(
            "ss".to_string(),
            SpeakerConfig::SupportingSource(supporting_group()),
        );
        let config = room_config_with_speakers(speakers);
        let sys = SystemConfig {
            model: SystemModel::Stereo,
            speakers: std::collections::HashMap::from([("L".to_string(), "ss".to_string())]),
            ..Default::default()
        };
        let output_dir = tempfile::tempdir().unwrap();
        let mut chains = std::collections::HashMap::new();
        let mut results = std::collections::HashMap::new();
        let mut metadata = crate::roomeq::types::OptimizationMetadata {
            pre_score: 0.0,
            post_score: 0.0,
            algorithm: "cobyla".to_string(),
            loss_type: None,
            iterations: 0,
            timestamp: chrono::Utc::now().to_rfc3339(),
            inter_channel_deviation: None,
            epa_per_channel: None,
            epa_multichannel: None,
            group_delay: None,
            mixed_phase_per_channel: None,
            perceptual_metrics: None,
            home_cinema_layout: None,
            multi_seat_coverage: None,
            multi_seat_correction: None,
            bass_management: None,
            timing_diagnostics: None,
            ctc: None,
            perceptual_policy: None,
            bootstrap_uncertainty: None,
            validation_bundle: None,
            supporting_source: None,
            correction_acceptance: None,
            optimizer_evidence: None,
            stage_outcomes: Vec::new(),
        };

        process_supporting_source_channels(
            &config,
            &sys,
            48000.0,
            output_dir.path(),
            &mut chains,
            &mut results,
            &mut metadata,
        )
        .unwrap();

        assert!(chains.contains_key("L"));
        assert!(chains.contains_key("L_support"));
        assert!(results.contains_key("L"));
        assert!(results.contains_key("L_support"));
        assert!(
            metadata
                .supporting_source
                .as_ref()
                .unwrap()
                .contains_key("L")
        );
    }

    #[test]
    fn process_supporting_source_channels_noop_when_empty() {
        let config = room_config_with_speakers(std::collections::HashMap::new());
        let sys = SystemConfig {
            model: SystemModel::Stereo,
            speakers: std::collections::HashMap::new(),
            ..Default::default()
        };
        let output_dir = tempfile::tempdir().unwrap();
        let mut chains = std::collections::HashMap::new();
        let mut results = std::collections::HashMap::new();
        let mut metadata = crate::roomeq::types::OptimizationMetadata {
            pre_score: 0.0,
            post_score: 0.0,
            algorithm: "cobyla".to_string(),
            loss_type: None,
            iterations: 0,
            timestamp: chrono::Utc::now().to_rfc3339(),
            inter_channel_deviation: None,
            epa_per_channel: None,
            epa_multichannel: None,
            group_delay: None,
            mixed_phase_per_channel: None,
            perceptual_metrics: None,
            home_cinema_layout: None,
            multi_seat_coverage: None,
            multi_seat_correction: None,
            bass_management: None,
            timing_diagnostics: None,
            ctc: None,
            perceptual_policy: None,
            bootstrap_uncertainty: None,
            validation_bundle: None,
            supporting_source: None,
            correction_acceptance: None,
            optimizer_evidence: None,
            stage_outcomes: Vec::new(),
        };

        process_supporting_source_channels(
            &config,
            &sys,
            48000.0,
            output_dir.path(),
            &mut chains,
            &mut results,
            &mut metadata,
        )
        .unwrap();

        assert!(chains.is_empty());
        assert!(results.is_empty());
        assert!(metadata.supporting_source.is_none());
    }
}
