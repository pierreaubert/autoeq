use crate::{
    BassManagementConfig, RoleTargetConfig, RoomConfig, SpeakerConfig, TargetResponseConfig,
    TargetShape, UserPreference,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::HashMap;

/// Resolve the active bass-management crossover frequency from room config.
pub fn bass_management_crossover_frequency_hz(config: &RoomConfig) -> Option<f64> {
    let system = config.system.as_ref()?;
    let subwoofers = system.subwoofers.as_ref()?;
    if !system.bass_management.clone().unwrap_or_default().enabled {
        return None;
    }
    let crossover = config
        .crossovers
        .as_ref()?
        .get(subwoofers.crossover.as_deref()?)?;
    crossover.frequency.or_else(|| {
        crossover
            .frequency_range
            .map(|(minimum, maximum)| (minimum * maximum).sqrt())
    })
}

/// Map a user-facing channel label to its canonical home-cinema role.
pub fn role_for_channel(channel_name: &str) -> HomeCinemaRole {
    fn normalized(channel_name: &str) -> String {
        channel_name
            .trim()
            .to_ascii_lowercase()
            .chars()
            .filter(|ch| !matches!(ch, ' ' | '-' | '_' | '.'))
            .collect()
    }
    fn has_supported_suffix(value: &str, prefix: &str) -> bool {
        let Some(suffix) = value.strip_prefix(prefix) else {
            return false;
        };
        !suffix.is_empty()
            && (suffix.chars().all(|c| c.is_ascii_digit())
                || matches!(suffix, "rear" | "front" | "left" | "right"))
    }
    let normalized = normalized(channel_name);
    match normalized.as_str() {
        "l" | "fl" | "left" | "frontleft" => HomeCinemaRole::FrontLeft,
        "r" | "fr" | "right" | "frontright" => HomeCinemaRole::FrontRight,
        "c" | "center" | "centre" => HomeCinemaRole::Center,
        "lfe" | "lf" => HomeCinemaRole::Lfe,
        "sub" | "subwoofer" | "sw" | "sub1" | "sub2" => HomeCinemaRole::Subwoofer,
        "sl" | "ls" | "surroundleft" | "sideleft" => HomeCinemaRole::SideSurroundLeft,
        "sr" | "rs" | "surroundright" | "sideright" => HomeCinemaRole::SideSurroundRight,
        "bl" | "rl" | "sbl" | "rearleft" | "backleft" | "surroundbackleft" => {
            HomeCinemaRole::RearSurroundLeft
        }
        "br" | "rr" | "sbr" | "rearright" | "backright" | "surroundbackright" => {
            HomeCinemaRole::RearSurroundRight
        }
        "wl" | "wideleft" | "frontwideleft" => HomeCinemaRole::WideLeft,
        "wr" | "wideright" | "frontwideright" => HomeCinemaRole::WideRight,
        "tfl" | "fhl" | "topfrontleft" | "frontheightleft" => HomeCinemaRole::TopFrontLeft,
        "tfr" | "fhr" | "topfrontright" | "frontheightright" => HomeCinemaRole::TopFrontRight,
        "tml" | "topmiddleleft" => HomeCinemaRole::TopMiddleLeft,
        "tmr" | "topmiddleright" => HomeCinemaRole::TopMiddleRight,
        "tbl" | "trl" | "rhl" | "topbackleft" | "toprearleft" | "rearheightleft" => {
            HomeCinemaRole::TopRearLeft
        }
        "tbr" | "trr" | "rhr" | "topbackright" | "toprearright" | "rearheightright" => {
            HomeCinemaRole::TopRearRight
        }
        _ if normalized.starts_with("subwoofer") || has_supported_suffix(&normalized, "sub") => {
            HomeCinemaRole::Subwoofer
        }
        _ if has_supported_suffix(&normalized, "lfe") => HomeCinemaRole::Lfe,
        _ => HomeCinemaRole::Unknown,
    }
}

/// Return the semantic matching group for a channel label, when it is safe to
/// match that role as part of a multi-channel group.
pub fn matching_group_key(channel_name: &str) -> Option<&'static str> {
    matching_group_key_for_role(role_for_channel(channel_name))
}

/// Return the semantic matching group for a canonical home-cinema role.
pub fn matching_group_key_for_role(role: HomeCinemaRole) -> Option<&'static str> {
    match role.group() {
        HomeCinemaRoleGroup::FrontLr => Some("front_lr"),
        HomeCinemaRoleGroup::SideSurrounds => Some("side_surrounds"),
        HomeCinemaRoleGroup::RearSurrounds => Some("rear_surrounds"),
        HomeCinemaRoleGroup::Wides => Some("wides"),
        HomeCinemaRoleGroup::TopFront => Some("top_front"),
        HomeCinemaRoleGroup::TopMiddle => Some("top_middle"),
        HomeCinemaRoleGroup::TopRear => Some("top_rear"),
        HomeCinemaRoleGroup::Unknown => Some("generic"),
        HomeCinemaRoleGroup::Center | HomeCinemaRoleGroup::Lfe | HomeCinemaRoleGroup::Subwoofer => {
            None
        }
    }
}

/// Determine the logical channels that a RoomEQ configuration exposes.
pub fn logical_channel_names(config: &RoomConfig) -> Vec<String> {
    if let Some(system) = config.system.as_ref() {
        let mut pairs: Vec<_> = system.speakers.keys().cloned().collect();
        pairs.sort();
        pairs
    } else if let Some(recording) = config.recording_config.as_ref()
        && let Some(names) = recording.channel_names.as_ref()
        && !names.is_empty()
    {
        names.clone()
    } else {
        let mut names: Vec<_> = config.speakers.keys().cloned().collect();
        names.sort();
        names
    }
}

/// Resolve physical speaker configurations into their logical channel names.
pub fn logical_speaker_configs(config: &RoomConfig) -> HashMap<String, SpeakerConfig> {
    if let Some(system) = config.system.as_ref() {
        system
            .speakers
            .iter()
            .filter_map(|(role, key)| {
                config
                    .speakers
                    .get(key)
                    .cloned()
                    .map(|speaker| (role.clone(), speaker))
            })
            .collect()
    } else {
        config.speakers.clone()
    }
}

/// Clamp a channel's optimization band to the role-appropriate target band.
pub fn role_score_band(config: &RoomConfig, channel_name: &str) -> (f64, f64) {
    let (role_min, role_max) = role_for_channel(channel_name).default_target_band_hz();
    let min = config.optimizer.min_freq.max(role_min);
    let max = config.optimizer.max_freq.min(role_max).max(min);
    (min, max)
}

/// Apply enabled role-specific target adjustments to a base target response.
pub fn role_adjusted_target_response(
    channel_name: &str,
    base: &TargetResponseConfig,
) -> TargetResponseConfig {
    let Some(role_targets) = base.role_targets.as_ref().filter(|cfg| cfg.enabled) else {
        return base.clone();
    };
    let mut adjusted = base.clone();
    apply_role_target_adjustment(role_for_channel(channel_name), role_targets, &mut adjusted);
    adjusted
}

/// Whether the channel's role-specific target changes the curve shape.
pub fn role_target_curve_shape_active(channel_name: &str, target: &TargetResponseConfig) -> bool {
    let Some(role_targets) = target.role_targets.as_ref().filter(|cfg| cfg.enabled) else {
        return false;
    };
    let role = role_for_channel(channel_name);
    (role == HomeCinemaRole::Center && role_targets.center_dialog_boost_db.abs() > 0.001)
        || (role_targets.cinema_x_curve_enabled
            && role_targets.cinema_x_curve_db_per_octave.abs() > 0.001)
        || (role_targets.listening_distance_m.is_some()
            && role_targets.distance_treble_rolloff_db_per_doubling.abs() > 0.001)
}

/// Stable profile key for a canonical role.
pub fn role_profile_base(role: HomeCinemaRole) -> &'static str {
    match role {
        HomeCinemaRole::FrontLeft | HomeCinemaRole::FrontRight => "front_lr",
        HomeCinemaRole::Center => "center_dialog",
        HomeCinemaRole::Lfe => "lfe",
        HomeCinemaRole::Subwoofer => "subwoofer",
        HomeCinemaRole::SideSurroundLeft
        | HomeCinemaRole::SideSurroundRight
        | HomeCinemaRole::RearSurroundLeft
        | HomeCinemaRole::RearSurroundRight
        | HomeCinemaRole::WideLeft
        | HomeCinemaRole::WideRight => "surround",
        HomeCinemaRole::TopFrontLeft
        | HomeCinemaRole::TopFrontRight
        | HomeCinemaRole::TopMiddleLeft
        | HomeCinemaRole::TopMiddleRight
        | HomeCinemaRole::TopRearLeft
        | HomeCinemaRole::TopRearRight => "height",
        HomeCinemaRole::Unknown => "generic",
    }
}

/// Stable profile key for a canonical role group.
pub fn role_group_key(group: HomeCinemaRoleGroup) -> &'static str {
    match group {
        HomeCinemaRoleGroup::FrontLr => "front_lr",
        HomeCinemaRoleGroup::Center => "center",
        HomeCinemaRoleGroup::Lfe => "lfe",
        HomeCinemaRoleGroup::Subwoofer => "subwoofer",
        HomeCinemaRoleGroup::SideSurrounds => "side_surrounds",
        HomeCinemaRoleGroup::RearSurrounds => "rear_surrounds",
        HomeCinemaRoleGroup::Wides => "wides",
        HomeCinemaRoleGroup::TopFront => "top_front",
        HomeCinemaRoleGroup::TopMiddle => "top_middle",
        HomeCinemaRoleGroup::TopRear => "top_rear",
        HomeCinemaRoleGroup::Unknown => "unknown",
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum HomeCinemaRole {
    FrontLeft,
    FrontRight,
    Center,
    Lfe,
    SideSurroundLeft,
    SideSurroundRight,
    RearSurroundLeft,
    RearSurroundRight,
    WideLeft,
    WideRight,
    TopFrontLeft,
    TopFrontRight,
    TopMiddleLeft,
    TopMiddleRight,
    TopRearLeft,
    TopRearRight,
    Subwoofer,
    Unknown,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum HomeCinemaRoleGroup {
    FrontLr,
    Center,
    Lfe,
    Subwoofer,
    SideSurrounds,
    RearSurrounds,
    Wides,
    TopFront,
    TopMiddle,
    TopRear,
    Unknown,
}

impl HomeCinemaRole {
    pub fn group(self) -> HomeCinemaRoleGroup {
        match self {
            Self::FrontLeft | Self::FrontRight => HomeCinemaRoleGroup::FrontLr,
            Self::Center => HomeCinemaRoleGroup::Center,
            Self::Lfe => HomeCinemaRoleGroup::Lfe,
            Self::Subwoofer => HomeCinemaRoleGroup::Subwoofer,
            Self::SideSurroundLeft | Self::SideSurroundRight => HomeCinemaRoleGroup::SideSurrounds,
            Self::RearSurroundLeft | Self::RearSurroundRight => HomeCinemaRoleGroup::RearSurrounds,
            Self::WideLeft | Self::WideRight => HomeCinemaRoleGroup::Wides,
            Self::TopFrontLeft | Self::TopFrontRight => HomeCinemaRoleGroup::TopFront,
            Self::TopMiddleLeft | Self::TopMiddleRight => HomeCinemaRoleGroup::TopMiddle,
            Self::TopRearLeft | Self::TopRearRight => HomeCinemaRoleGroup::TopRear,
            Self::Unknown => HomeCinemaRoleGroup::Unknown,
        }
    }

    pub fn is_height(self) -> bool {
        matches!(
            self,
            Self::TopFrontLeft
                | Self::TopFrontRight
                | Self::TopMiddleLeft
                | Self::TopMiddleRight
                | Self::TopRearLeft
                | Self::TopRearRight
        )
    }

    pub fn is_sub_or_lfe(self) -> bool {
        matches!(self, Self::Subwoofer | Self::Lfe)
    }

    pub fn is_bed_channel(self) -> bool {
        !self.is_height() && !self.is_sub_or_lfe() && self != Self::Unknown
    }

    pub fn is_bass_managed_candidate(self) -> bool {
        self.is_bed_channel() || self.is_height()
    }

    pub fn default_target_band_hz(self) -> (f64, f64) {
        match self {
            Self::Lfe | Self::Subwoofer => (20.0, 160.0),
            Self::Center => (80.0, 16_000.0),
            Self::SideSurroundLeft
            | Self::SideSurroundRight
            | Self::RearSurroundLeft
            | Self::RearSurroundRight
            | Self::WideLeft
            | Self::WideRight => (80.0, 12_000.0),
            role if role.is_height() => (120.0, 10_000.0),
            Self::Unknown => (20.0, 20_000.0),
            _ => (40.0, 18_000.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HomeCinemaChannelReport {
    pub name: String,
    pub role: HomeCinemaRole,
    pub role_group: HomeCinemaRoleGroup,
    pub is_bass_managed: bool,
    pub matching_group: Option<String>,
    pub target_band_hz: (f64, f64),
    pub target_profile: String,
    pub target_advisory: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HomeCinemaLayoutReport {
    pub layout: String,
    pub bed_channels: usize,
    pub lfe_channels: usize,
    pub height_channels: usize,
    pub subwoofer_channels: usize,
    pub channels: Vec<HomeCinemaChannelReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MultiSeatCoverageReport {
    pub channels_with_multiple_measurements: usize,
    pub non_sub_channel_count: usize,
    pub non_sub_channels_with_multiple_measurements: usize,
    pub max_seat_count: usize,
    pub by_role_group: BTreeMap<String, usize>,
    pub all_channel_correction_ready: bool,
    pub recommended_scope: String,
    pub advisories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MultiSeatCorrectionReport {
    pub enabled: bool,
    pub applied: bool,
    pub strategy: String,
    pub seat_count: usize,
    pub primary_seat: usize,
    pub seat_weights: Vec<f64>,
    pub channels: Vec<MultiSeatChannelCorrectionReport>,
    pub role_groups: Vec<MultiSeatRoleGroupCorrectionReport>,
    pub advisories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MultiSeatChannelCorrectionReport {
    pub channel: String,
    pub role: HomeCinemaRole,
    pub role_group: HomeCinemaRoleGroup,
    pub status: String,
    pub seat_count: usize,
    pub target_band_hz: (f64, f64),
    pub rms_target_error_db: Option<f64>,
    pub max_abs_deviation_db: Option<f64>,
    pub primary_pass: Option<bool>,
    pub non_primary_pass: Option<bool>,
    pub spatial_variance_peak_db: Option<f64>,
    pub min_correction_depth: Option<f64>,
    pub seats: Vec<MultiSeatPredictionReport>,
    pub advisories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MultiSeatPredictionReport {
    pub seat_index: usize,
    pub weight: f64,
    pub is_primary: bool,
    pub rms_target_error_db: f64,
    pub max_abs_deviation_db: f64,
    pub pass: bool,
    pub null_risk: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MultiSeatRoleGroupCorrectionReport {
    pub role_group: HomeCinemaRoleGroup,
    pub channel_count: usize,
    pub applied_channel_count: usize,
    pub pass: bool,
    pub worst_rms_target_error_db: Option<f64>,
    pub worst_max_abs_deviation_db: Option<f64>,
    pub advisories: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AllChannelMultiSeatAcceptance {
    pub accepted: bool,
    pub advisories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BassManagementReport {
    pub enabled: bool,
    pub crossover_type: String,
    pub crossover_frequency_hz: Option<f64>,
    pub redirected_bass_enabled: bool,
    pub lfe_channel: String,
    pub lfe_playback_gain_db: f64,
    pub lfe_low_pass_hz: f64,
    pub lfe_gain_applied_to_chain: bool,
    pub sub_trim_db: f64,
    pub max_sub_boost_db: f64,
    pub headroom_margin_db: f64,
    pub applied_sub_gain_db: Option<f64>,
    pub gain_limited: bool,
    pub physical_sub_output: String,
    pub redirected_bass_channel_count: usize,
    pub main_high_pass_hz: Option<f64>,
    pub sub_low_pass_hz: Option<f64>,
    pub lfe_headroom_required_db: f64,
    pub signal_flow: Vec<BassManagementSignalFlowEntry>,
    pub signal_flow_advisories: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_graph: Option<BassManagementRoutingGraph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimization: Option<BassManagementOptimizationReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<BassManagementGroupReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sub_outputs: Vec<BassManagementSubOutputReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headroom_simulation: Option<BassBusHeadroomSimulationReport>,
    pub advisory: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BassManagementOptimizationReport {
    pub applied: bool,
    pub phase_required: bool,
    pub phase_available: bool,
    pub configured_crossover_hz: Option<f64>,
    pub optimized_crossover_hz: Option<f64>,
    pub crossover_range_hz: Option<(f64, f64)>,
    pub crossover_type: String,
    pub main_delay_ms: f64,
    pub sub_delay_ms: f64,
    pub relative_sub_delay_ms: f64,
    pub sub_polarity_inverted: bool,
    pub requested_sub_gain_db: f64,
    pub applied_sub_gain_db: f64,
    pub gain_limited: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_bass_bus_peak_gain_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_before: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_after: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_results: Vec<BassManagementGroupReport>,
    /// Per-logical-input route alignment. Crossover frequency and type remain
    /// shared by the referenced speaker group.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_results: Vec<BassManagementSourceReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sub_output_results: Vec<BassManagementSubOutputReport>,
    pub advisories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BassManagementSignalFlowEntry {
    pub source_channel: String,
    pub role: HomeCinemaRole,
    pub destination: String,
    pub high_pass_hz: Option<f64>,
    pub low_pass_hz: Option<f64>,
    pub lfe_gain_db: f64,
    pub redirects_bass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BassManagementRoutingGraph {
    pub physical_sub_output: String,
    pub input_channels: Vec<String>,
    pub output_channels: Vec<String>,
    pub routes: Vec<BassManagementRoute>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix: Option<BassManagementMatrix>,
    /// Final down-only calibration trims per logical input channel.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub input_trim_db: HashMap<String, f64>,
    pub advisories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BassManagementRoute {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    pub source_channel: String,
    pub source_index: usize,
    pub destination: String,
    pub destination_index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_chain_channel: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_chain_channel: Option<String>,
    pub route_kind: String,
    pub crossover_type: String,
    pub high_pass_hz: Option<f64>,
    pub low_pass_hz: Option<f64>,
    pub gain_db: f64,
    pub gain_linear: f64,
    #[serde(default = "default_route_matrix_gain")]
    pub matrix_gain: f64,
    pub delay_ms: f64,
    pub polarity_inverted: bool,
}

fn default_route_matrix_gain() -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CrossoverConfig, SubwooferSystemConfig, SystemConfig};

    #[test]
    fn default_route_matrix_gain_is_unity() {
        assert_eq!(default_route_matrix_gain(), 1.0);
    }

    #[test]
    fn bass_management_crossover_uses_geometric_range_center() {
        let mut crossovers = HashMap::new();
        crossovers.insert(
            "main".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: None,
                frequencies: None,
                frequency_range: Some((80.0, 125.0)),
            },
        );
        let config = RoomConfig {
            system: Some(SystemConfig {
                bass_management: Some(BassManagementConfig {
                    enabled: true,
                    ..BassManagementConfig::default()
                }),
                subwoofers: Some(SubwooferSystemConfig {
                    config: crate::SubwooferStrategy::default(),
                    crossover: Some("main".to_string()),
                    mapping: HashMap::new(),
                }),
                ..SystemConfig::default()
            }),
            crossovers: Some(crossovers),
            ..RoomConfig::default()
        };

        assert_eq!(bass_management_crossover_frequency_hz(&config), Some(100.0));
        let mut default_policy = config;
        default_policy.system.as_mut().unwrap().bass_management = None;
        assert_eq!(
            bass_management_crossover_frequency_hz(&default_policy),
            Some(100.0)
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BassManagementGroupReport {
    pub group_id: String,
    pub roles: Vec<String>,
    pub crossover_type: String,
    pub selected_crossover_hz: Option<f64>,
    pub configured_crossover_hz: Option<f64>,
    pub main_delay_ms: f64,
    pub bass_route_delay_ms: f64,
    pub polarity_inverted: bool,
    pub trim_db: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_before: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_after: Option<f64>,
    pub advisories: Vec<String>,
}

/// Bass-management alignment for one logical input source.
///
/// Independent programme channels are intentionally not coherently summed
/// when these values are optimized. The source references a group for the
/// shared crossover type and frequency, but owns its route delay, polarity,
/// and trim.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BassManagementSourceReport {
    pub source_channel: String,
    pub group_id: String,
    pub main_delay_ms: f64,
    pub bass_route_delay_ms: f64,
    pub polarity_inverted: bool,
    pub trim_db: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_before: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_after: Option<f64>,
    #[serde(default)]
    pub accepted: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub advisories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BassManagementSubOutputReport {
    pub output_role: String,
    pub gain_db: f64,
    pub delay_ms: f64,
    pub polarity_inverted: bool,
    pub strategy_source: String,
    pub headroom_contribution_db: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BassBusOutputHeadroomReport {
    pub output_role: String,
    pub rms_bus_gain_db: f64,
    pub coherent_peak_gain_db: f64,
    pub lfe_contribution_db: f64,
    pub pass: bool,
    pub margin_db: f64,
    pub worst_frequency_hz: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BassBusHeadroomSimulationReport {
    pub model: String,
    pub frequency_range_hz: (f64, f64),
    pub rms_bus_gain_db: f64,
    pub coherent_peak_gain_db: f64,
    pub lfe_contribution_db: f64,
    pub headroom_margin_db: f64,
    pub pass: bool,
    pub margin_db: f64,
    pub worst_frequency_hz: f64,
    pub per_output: Vec<BassBusOutputHeadroomReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BassManagementMatrix {
    pub input_channel_map: Vec<usize>,
    pub output_channel_map: Vec<usize>,
    pub matrix: Vec<f32>,
    pub route_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelTimingReport {
    pub name: String,
    pub role: HomeCinemaRole,
    pub measured_arrival_ms: f64,
    pub acoustic_distance_m: f64,
    pub applied_delay_ms: f64,
    pub final_arrival_ms: f64,
    pub final_offset_from_reference_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TimingDiagnosticsReport {
    pub reference_channel: Option<String>,
    pub reference_arrival_ms: Option<f64>,
    pub arrival_spread_before_ms: f64,
    pub arrival_spread_after_ms: f64,
    pub channels: Vec<ChannelTimingReport>,
    pub advisories: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct EffectiveBassManagement {
    pub config: BassManagementConfig,
    pub crossover_type: String,
    pub crossover_frequency_hz: Option<f64>,
    pub advisory: String,
}

pub fn add_slope_offset(target: &mut TargetResponseConfig, slope_offset_db_per_octave: f64) {
    let base_slope = match target.shape {
        TargetShape::Flat => 0.0,
        TargetShape::Harman => -0.8,
        TargetShape::Custom => target.slope_db_per_octave,
        TargetShape::File | TargetShape::FromMeasurement => target.slope_db_per_octave,
    };
    target.shape = TargetShape::Custom;
    target.slope_db_per_octave = base_slope + slope_offset_db_per_octave;
}

pub fn apply_role_target_adjustment(
    role: HomeCinemaRole,
    role_targets: &RoleTargetConfig,
    target: &mut TargetResponseConfig,
) {
    let slope_offset = role_slope_offset(role, role_targets);
    if slope_offset.abs() > 0.001 {
        add_slope_offset(target, slope_offset);
    }

    match role {
        HomeCinemaRole::Center => {
            target.preference.treble_shelf_db += role_targets.center_treble_shelf_db;
        }
        HomeCinemaRole::SideSurroundLeft
        | HomeCinemaRole::SideSurroundRight
        | HomeCinemaRole::RearSurroundLeft
        | HomeCinemaRole::RearSurroundRight
        | HomeCinemaRole::WideLeft
        | HomeCinemaRole::WideRight => {
            target.preference.treble_shelf_db += role_targets.surround_treble_shelf_db;
        }
        role if role.is_height() => {
            target.preference.treble_shelf_db += role_targets.height_treble_shelf_db;
        }
        HomeCinemaRole::Lfe => {
            target.preference.bass_shelf_db += role_targets.lfe_bass_shelf_db;
        }
        HomeCinemaRole::Subwoofer => {
            target.preference.bass_shelf_db += role_targets.subwoofer_bass_shelf_db;
        }
        _ => {}
    }

    if target.preference.treble_shelf_freq <= 0.0 {
        target.preference.treble_shelf_freq = UserPreference::default().treble_shelf_freq;
    }
    if target.preference.bass_shelf_freq <= 0.0 {
        target.preference.bass_shelf_freq = UserPreference::default().bass_shelf_freq;
    }
}

pub fn role_slope_offset(role: HomeCinemaRole, role_targets: &RoleTargetConfig) -> f64 {
    match role {
        HomeCinemaRole::FrontLeft | HomeCinemaRole::FrontRight => {
            role_targets.front_slope_offset_db_per_octave
        }
        HomeCinemaRole::Center => role_targets.center_slope_offset_db_per_octave,
        HomeCinemaRole::SideSurroundLeft
        | HomeCinemaRole::SideSurroundRight
        | HomeCinemaRole::RearSurroundLeft
        | HomeCinemaRole::RearSurroundRight
        | HomeCinemaRole::WideLeft
        | HomeCinemaRole::WideRight => role_targets.surround_slope_offset_db_per_octave,
        HomeCinemaRole::TopFrontLeft
        | HomeCinemaRole::TopFrontRight
        | HomeCinemaRole::TopMiddleLeft
        | HomeCinemaRole::TopMiddleRight
        | HomeCinemaRole::TopRearLeft
        | HomeCinemaRole::TopRearRight => role_targets.height_slope_offset_db_per_octave,
        HomeCinemaRole::Subwoofer => role_targets.subwoofer_slope_offset_db_per_octave,
        HomeCinemaRole::Lfe => role_targets.lfe_slope_offset_db_per_octave,
        HomeCinemaRole::Unknown => 0.0,
    }
}
