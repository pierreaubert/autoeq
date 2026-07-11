use crate::MeasurementSource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{SpeakerDriver, SpeakerDriverRole, SpeakerTopology};

pub const LEGACY_SPEAKER_GROUP_ADVISORY: &str =
    "legacy_speaker_group_measurements_deprecated_use_explicit_topology";

/// Group of measurements for a single speaker (multi-driver)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SpeakerGroup {
    /// Name of the group
    pub name: String,
    /// Optional speaker model name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
    /// Measurements in this group
    pub measurements: Vec<MeasurementSource>,
    /// Crossover configuration for this group
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crossover: Option<String>,
}

impl SpeakerGroup {
    pub fn resolve_paths(&mut self, base_dir: &std::path::Path) {
        for m in &mut self.measurements {
            m.resolve_paths(base_dir);
        }
    }

    /// Translate the legacy measurements array into stable driver entries.
    ///
    /// The processing adapter retains the historical bandwidth-based ordering;
    /// new configurations should use [`SpeakerTopology`] to declare order and roles.
    pub fn to_legacy_topology(&self) -> SpeakerTopology {
        let count = self.measurements.len();
        let drivers = self
            .measurements
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, measurement)| SpeakerDriver {
                id: format!("legacy_driver_{}", index + 1),
                role: legacy_role(index, count),
                measurement,
                crossover_band: None,
            })
            .collect();
        SpeakerTopology {
            name: self.name.clone(),
            speaker_name: self.speaker_name.clone(),
            drivers,
            parallel_groups: Vec::new(),
            crossover: self.crossover.clone(),
        }
    }
}

fn legacy_role(index: usize, count: usize) -> SpeakerDriverRole {
    match (count, index) {
        (1, _) => SpeakerDriverRole::FullRange,
        (2, 0) => SpeakerDriverRole::Woofer,
        (2, _) => SpeakerDriverRole::Tweeter,
        (3, 0) => SpeakerDriverRole::Woofer,
        (3, 1) => SpeakerDriverRole::Midrange,
        (3, _) => SpeakerDriverRole::Tweeter,
        (_, 0) => SpeakerDriverRole::Woofer,
        (_, index) if index + 1 == count => SpeakerDriverRole::Tweeter,
        _ => SpeakerDriverRole::Other,
    }
}
