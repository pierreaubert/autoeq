use super::enable::enable_gd_missing_coherence_measurements;
use super::enable::enable_gd_trusted_measurements;
use super::enable::enable_multi_measurement_paths;
use super::enable::enable_multisub_multi_seat_paths;
use super::option_override::OptionOverride;
use super::test_case::{RegisteredTestCase, TestCase};
use crate::registry::{QaTier, QualityCaseKind, load_registry};
use anyhow::Result;
use roomeq_model::{MultiMeasurementConfig, MultiMeasurementStrategy, ProcessingMode, RoomConfig};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GroupDelayQaProfile {
    MissingCoherenceDelayOnly,
    TrustedDelayOnly,
    FixedAllPass,
    AdaptiveAllPass,
    PhaseLinearFir,
    MixedPhase,
}

impl GroupDelayQaProfile {
    pub(super) fn label(self) -> &'static str {
        match self {
            GroupDelayQaProfile::MissingCoherenceDelayOnly => "missing_coherence_delay_only",
            GroupDelayQaProfile::TrustedDelayOnly => "trusted_delay_only",
            GroupDelayQaProfile::FixedAllPass => "fixed_allpass",
            GroupDelayQaProfile::AdaptiveAllPass => "adaptive_allpass",
            GroupDelayQaProfile::PhaseLinearFir => "phase_linear_fir",
            GroupDelayQaProfile::MixedPhase => "mixed_phase",
        }
    }

    pub(super) fn needs_trusted_measurements(self) -> bool {
        !matches!(self, GroupDelayQaProfile::MissingCoherenceDelayOnly)
    }

    pub(super) fn needs_multi_measurement_paths(self) -> bool {
        matches!(self, GroupDelayQaProfile::AdaptiveAllPass)
    }
}

/// Disable the option in config to create a clean baseline
pub(super) fn disable_option(config: &mut RoomConfig, option: &OptionOverride) {
    match option {
        OptionOverride::TargetTilt { .. } => {
            config.optimizer.target_response = None;
        }
        OptionOverride::ExcursionProtection => {
            config.optimizer.excursion_protection = None;
        }
        OptionOverride::SchroederSplit { .. } => {
            config.optimizer.schroeder_split = None;
        }
        OptionOverride::AsymmetricLoss => {
            config.optimizer.asymmetric_loss = false;
        }
        OptionOverride::Psychoacoustic => {
            config.optimizer.psychoacoustic = false;
        }
        OptionOverride::BroadbandTargetMatching => {
            if let Some(ref mut tr) = config.optimizer.target_response {
                tr.broadband_precorrection = false;
            }
        }
        OptionOverride::PhaseAlignment => {
            config.optimizer.phase_alignment = None;
            config.optimizer.allow_delay = Some(false);
        }
        OptionOverride::MultiMeasurementMinimax
        | OptionOverride::MultiMeasurementVariancePenalized { .. } => {
            config.optimizer.multi_measurement = Some(MultiMeasurementConfig {
                strategy: MultiMeasurementStrategy::Average,
                ..Default::default()
            });
        }
        OptionOverride::ProductionMultiSubMultiSeat => {
            config.optimizer.multi_seat = None;
        }
        OptionOverride::InterChannelTimbreMatching { .. } => {
            config.optimizer.inter_channel_timbre_matching = None;
        }
        OptionOverride::SpatialRobustness => {
            config.optimizer.multi_measurement = Some(MultiMeasurementConfig {
                strategy: MultiMeasurementStrategy::Average,
                ..Default::default()
            });
        }
        OptionOverride::PreRinging => {
            if let Some(ref mut fir) = config.optimizer.fir {
                fir.pre_ringing = None;
            }
        }
        OptionOverride::MixedPhaseMode => {
            config.optimizer.processing_mode = ProcessingMode::LowLatency;
            config.optimizer.mixed_phase = None;
        }
        OptionOverride::DecomposedCorrection => {
            config.optimizer.decomposed_correction = None;
        }
        OptionOverride::GroupDelay { profile } => {
            config.optimizer.group_delay = None;
            match profile {
                GroupDelayQaProfile::PhaseLinearFir => {
                    config.optimizer.processing_mode = ProcessingMode::LowLatency;
                    config.optimizer.fir = None;
                }
                GroupDelayQaProfile::MixedPhase => {
                    config.optimizer.processing_mode = ProcessingMode::LowLatency;
                    config.optimizer.mixed_phase = None;
                }
                _ => {}
            }
        }
    }
}

pub(super) fn prepare_option_measurement_paths(
    config: &mut RoomConfig,
    fem_dir: &Path,
    fem_subdir: &str,
    needs_multi_measurement: bool,
    needs_gd_trusted_measurements: bool,
    needs_multisub_multi_seat: bool,
    gd_profile: Option<GroupDelayQaProfile>,
) -> Result<()> {
    if needs_multisub_multi_seat {
        enable_multisub_multi_seat_paths(config, fem_dir, fem_subdir);
    }
    if gd_profile == Some(GroupDelayQaProfile::MissingCoherenceDelayOnly) {
        enable_gd_missing_coherence_measurements(config, fem_dir, fem_subdir)?;
    } else if needs_gd_trusted_measurements {
        enable_gd_trusted_measurements(
            config,
            fem_dir,
            fem_subdir,
            needs_multi_measurement,
            gd_profile,
        )?;
    } else if needs_multi_measurement {
        enable_multi_measurement_paths(config, fem_dir, fem_subdir);
    }
    Ok(())
}

pub(super) fn all_test_cases() -> Vec<RegisteredTestCase> {
    let registry = load_registry().expect("RoomEQ QA registry must be valid");
    registry
        .quality_cases_for(QaTier::Nightly)
        .map(|spec| {
            let options = spec
                .options
                .iter()
                .map(|name| {
                    OptionOverride::from_registry_name(name).unwrap_or_else(|| {
                        panic!(
                            "unknown quality option '{name}' in registry case {}",
                            spec.id
                        )
                    })
                })
                .collect();
            let case = match &spec.kind {
                QualityCaseKind::Workflow => TestCase::Workflow {
                    name: spec.name.clone(),
                    fem_subdir: spec.fem_subdir.clone(),
                    optim_subdir: spec.optim_subdir.clone(),
                },
                QualityCaseKind::Generic => TestCase::Generic {
                    name: spec.name.clone(),
                    fem_subdir: spec.fem_subdir.clone(),
                    optim_subdir: spec.optim_subdir.clone(),
                },
                QualityCaseKind::CrossModeConvergence => TestCase::CrossModeConvergence {
                    name: spec.name.clone(),
                    fem_subdir: spec.fem_subdir.clone(),
                    optim_subdir: spec.optim_subdir.clone(),
                    config_path: spec.config_path.clone(),
                    override_dir: spec.override_dir.clone(),
                    preserve_system: spec.preserve_system,
                    strict: spec.strict_cross_mode,
                },
                QualityCaseKind::OptionEffect => TestCase::OptionEffect {
                    name: spec.name.clone(),
                    fem_subdir: spec.fem_subdir.clone(),
                    optim_subdir: spec.optim_subdir.clone(),
                    options,
                },
            };
            RegisteredTestCase {
                id: spec.id.clone(),
                claims: spec.claims.clone(),
                expect: spec.expect,
                case,
            }
        })
        .collect()
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn quality_matrix_is_registry_driven_and_all_options_parse() {
        let cases = all_test_cases();
        assert_eq!(cases.len(), 37);
        assert!(
            cases
                .iter()
                .any(|case| matches!(&case.case, TestCase::CrossModeConvergence { .. }))
        );
        assert!(cases.iter().any(
            |case| matches!(&case.case, TestCase::OptionEffect { options, .. } if options.len() >= 6)
        ));
        let variance_labels = cases
            .iter()
            .flat_map(|case| match &case.case {
                TestCase::OptionEffect { options, .. } => options
                    .iter()
                    .map(ToString::to_string)
                    .filter(|label| label.starts_with("multi_measurement_variance"))
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect::<Vec<_>>();
        assert!(variance_labels.iter().any(|label| label.contains("0.25")));
        assert!(variance_labels.iter().any(|label| label.contains("0.10")));
    }
}
