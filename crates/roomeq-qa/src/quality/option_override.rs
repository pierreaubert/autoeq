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
    MultiMeasurementVariancePenalized,
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
            OptionOverride::MultiMeasurementVariancePenalized => {
                write!(f, "multi_measurement_variance")
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
