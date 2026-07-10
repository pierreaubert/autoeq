use super::bass_anchor_results_legacy::BassAnchorResultsLegacy;
use super::cardioid_config::CardioidConfig;
use super::channel_matching_config::ChannelMatchingConfig;
use super::ctc_config::CtcConfig;
use super::ctc_hrtf_config::CtcHrtfConfig;
use super::ctc_measurement_config::CtcMeasurementConfig;
use super::dbaconfig::DBAConfig;
use super::decomposed_correction_serde_config::DecomposedCorrectionSerdeConfig;
use super::default::default_high_freq_guard_max_q;
use super::default::default_high_freq_guard_start_hz;
use super::default::default_high_freq_smoothing_n;
use super::default::default_max_freq;
use super::high_frequency_correction_config::HighFrequencyCorrectionConfig;
use super::multi_sub_group::MultiSubGroup;
use super::optimizer_config::OptimizerConfig;
use super::perceptual_policy_config::PerceptualPolicyConfig;
use super::policy::{
    policy_asymmetric_loss, policy_audibility_deadband, policy_decomposed_correction,
    policy_early_late_correction, policy_high_frequency_guard, policy_multi_measurement,
    policy_psychoacoustic_smoothing, policy_smoothness_penalty, policy_target_response,
};
use super::room_config::RoomConfig;
use super::speaker_config::SpeakerConfig;
use super::speaker_group::SpeakerGroup;
use super::spl_calibration::SplCalibration;
use super::types::{
    CrossoverConfig, CtcHeadPositionConfig, CtcHrtfSpeakerConfig, CtcMeasurementFileConfig,
    MultiMeasurementStrategy, PerceptualPolicyPreset, RecordingConfiguration, SystemConfig,
    SystemModel, TargetCurveConfig, TargetShape,
};
use crate::loss::AsymmetricLossConfig;
use crate::read::PsychoacousticSmoothingConfig;
use crate::{MeasurementRef, MeasurementSingle, MeasurementSource};
use std::collections::HashMap;
use std::path::PathBuf;

include!("tests/core.rs");
include!("tests/roundtrip.rs");
include!("tests/policies.rs");
