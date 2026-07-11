use super::cardioid_config::CardioidConfig;
use super::dbaconfig::DBAConfig;
use super::multi_sub_group::MultiSubGroup;
use super::speaker_group::SpeakerGroup;
use super::speaker_topology::SpeakerTopology;
use super::supporting_source_group::SupportingSourceGroup;
use crate::MeasurementSource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Speaker configuration (can be single measurement or group)
///
/// Variant order matters for serde untagged deserialization: serde tries each variant
/// in order. `SupportingSource` is first because it requires the unique `primary` and
/// `support` fields. Topology is before legacy Group because its unique `drivers`
/// field selects the explicit representation. Named group variants are tried before
/// `Single`, which remains the catch-all.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
#[allow(
    clippy::large_enum_variant,
    reason = "SpeakerConfig::Single is the dominant variant in 100+ call sites; boxing would create churn for marginal memory savings"
)]
pub enum SpeakerConfig {
    /// Supporting-source room compensation (primary + delayed support loudspeaker).
    SupportingSource(SupportingSourceGroup),
    /// Group of measurements (multi-driver case)
    Topology(SpeakerTopology),
    /// Legacy group of measurements (multi-driver case)
    Group(SpeakerGroup),
    /// Multiple subwoofers optimization
    MultiSub(MultiSubGroup),
    /// Double Bass Array (DBA) optimization
    Dba(DBAConfig),
    /// Gradient Cardioid subwoofer optimization
    Cardioid(Box<CardioidConfig>),
    /// Single channel (simple case)
    Single(MeasurementSource),
}

impl SpeakerConfig {
    pub fn speaker_name(&self) -> Option<&str> {
        match self {
            SpeakerConfig::SupportingSource(group) => group.speaker_name.as_deref(),
            SpeakerConfig::Single(source) => source.speaker_name(),
            SpeakerConfig::Topology(topology) => topology.speaker_name.as_deref(),
            SpeakerConfig::Group(group) => group.speaker_name.as_deref(),
            SpeakerConfig::MultiSub(ms) => ms.speaker_name.as_deref(),
            SpeakerConfig::Dba(dba) => dba.speaker_name.as_deref(),
            SpeakerConfig::Cardioid(c) => c.speaker_name.as_deref(),
        }
    }

    pub fn resolve_paths(&mut self, base_dir: &std::path::Path) {
        match self {
            SpeakerConfig::SupportingSource(group) => group.resolve_paths(base_dir),
            SpeakerConfig::Single(source) => source.resolve_paths(base_dir),
            SpeakerConfig::Topology(topology) => topology.resolve_paths(base_dir),
            SpeakerConfig::Group(group) => group.resolve_paths(base_dir),
            SpeakerConfig::MultiSub(group) => group.resolve_paths(base_dir),
            SpeakerConfig::Dba(config) => config.resolve_paths(base_dir),
            SpeakerConfig::Cardioid(config) => config.resolve_paths(base_dir),
        }
    }
}
