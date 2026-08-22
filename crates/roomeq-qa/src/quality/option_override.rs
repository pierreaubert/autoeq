use super::group_delay_qa_profile::GroupDelayQaProfile;

#[derive(Clone, Debug)]
pub(super) enum OptionOverride {
    TargetTilt {
        slope_db_per_octave: f64,
    },
    ExcursionProtection,
    SchroederSplit {
        schroeder_freq: f64,
        low_max_q: f64,
        high_max_q: f64,
    },
    AsymmetricLoss,
    Psychoacoustic,
    BroadbandTargetMatching,
    PhaseAlignment,
    MultiMeasurementMinimax,
    MultiMeasurementVariancePenalized {
        variance_lambda: f64,
    },
    ProductionMultiSubMultiSeat,
    InterChannelTimbreMatching {
        reference_channel: String,
    },
    SpatialRobustness,
    PreRinging,
    MixedPhaseMode,
    DecomposedCorrection,
    GroupDelay {
        profile: GroupDelayQaProfile,
    },
}

impl OptionOverride {
    pub(super) fn from_registry_name(name: &str) -> Option<Self> {
        let option = match name {
            "target_tilt" => Self::TargetTilt {
                slope_db_per_octave: -0.8,
            },
            "excursion_protection" => Self::ExcursionProtection,
            "schroeder_split" => Self::SchroederSplit {
                schroeder_freq: 300.0,
                low_max_q: 10.0,
                high_max_q: 1.0,
            },
            "asymmetric_loss" => Self::AsymmetricLoss,
            "psychoacoustic" => Self::Psychoacoustic,
            "broadband_target_matching" => Self::BroadbandTargetMatching,
            "phase_alignment" => Self::PhaseAlignment,
            "multi_measurement_minimax" => Self::MultiMeasurementMinimax,
            "multi_measurement_variance_low" => Self::MultiMeasurementVariancePenalized {
                variance_lambda: 0.25,
            },
            "multi_measurement_variance" => Self::MultiMeasurementVariancePenalized {
                variance_lambda: 0.1,
            },
            "production_multi_sub_multi_seat" => Self::ProductionMultiSubMultiSeat,
            "inter_channel_timbre_matching" => Self::InterChannelTimbreMatching {
                reference_channel: "L".to_string(),
            },
            "spatial_robustness" => Self::SpatialRobustness,
            "pre_ringing" => Self::PreRinging,
            "mixed_phase" => Self::MixedPhaseMode,
            "decomposed_correction" => Self::DecomposedCorrection,
            "gd_missing_coherence" => Self::GroupDelay {
                profile: GroupDelayQaProfile::MissingCoherenceDelayOnly,
            },
            "gd_trusted_delay" => Self::GroupDelay {
                profile: GroupDelayQaProfile::TrustedDelayOnly,
            },
            "gd_fixed_allpass" => Self::GroupDelay {
                profile: GroupDelayQaProfile::FixedAllPass,
            },
            "gd_adaptive_allpass" => Self::GroupDelay {
                profile: GroupDelayQaProfile::AdaptiveAllPass,
            },
            "gd_phase_linear_fir" => Self::GroupDelay {
                profile: GroupDelayQaProfile::PhaseLinearFir,
            },
            "gd_mixed_phase" => Self::GroupDelay {
                profile: GroupDelayQaProfile::MixedPhase,
            },
            _ => return None,
        };
        Some(option)
    }

    /// Options that deliberately reshape the optimization target away from
    /// "flat". Flat-loss score ratios against a flat-target baseline are not
    /// a meaningful acceptance gate when these are active: the optimizer is
    /// minimizing deviation from a shaped target, so flat loss worsens by
    /// design (a -0.8 dB/oct tilt alone adds ~2.3 dB RMS of flat deviation).
    pub(super) fn reshapes_target(&self) -> bool {
        matches!(
            self,
            OptionOverride::TargetTilt { .. } | OptionOverride::BroadbandTargetMatching
        )
    }
}

impl std::fmt::Display for OptionOverride {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionOverride::TargetTilt {
                slope_db_per_octave,
            } => {
                write!(f, "target_tilt(slope={})", slope_db_per_octave)
            }
            OptionOverride::ExcursionProtection => write!(f, "excursion_protection"),
            OptionOverride::SchroederSplit { schroeder_freq, .. } => {
                write!(f, "schroeder_split(fs={})", schroeder_freq)
            }
            OptionOverride::AsymmetricLoss => write!(f, "asymmetric_loss"),
            OptionOverride::Psychoacoustic => write!(f, "psychoacoustic"),
            OptionOverride::BroadbandTargetMatching => write!(f, "broadband_target_matching"),
            OptionOverride::PhaseAlignment => write!(f, "phase_alignment"),
            OptionOverride::MultiMeasurementMinimax => write!(f, "multi_measurement_minimax"),
            OptionOverride::MultiMeasurementVariancePenalized { variance_lambda } => {
                write!(f, "multi_measurement_variance({variance_lambda:.2})")
            }
            OptionOverride::ProductionMultiSubMultiSeat => {
                write!(f, "production_multi_sub_multi_seat")
            }
            OptionOverride::InterChannelTimbreMatching { reference_channel } => {
                write!(f, "inter_channel_timbre_matching(ref={reference_channel})")
            }
            OptionOverride::SpatialRobustness => write!(f, "spatial_robustness"),
            OptionOverride::PreRinging => write!(f, "pre_ringing"),
            OptionOverride::MixedPhaseMode => write!(f, "mixed_phase"),
            OptionOverride::DecomposedCorrection => write!(f, "decomposed_correction"),
            OptionOverride::GroupDelay { profile } => {
                write!(f, "group_delay({})", profile.label())
            }
        }
    }
}
