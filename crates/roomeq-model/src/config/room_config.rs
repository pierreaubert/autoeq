use super::ctc_config::CtcConfig;
use super::default::default_config_version;
use super::optimizer_config::OptimizerConfig;
use super::speaker_config::SpeakerConfig;
use super::types::CrossoverConfig;
use super::types::RecordingConfiguration;
use super::types::SystemConfig;
use super::types::TargetCurveConfig;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete room configuration
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct RoomConfig {
    /// Configuration version (semantic versioning, e.g. "1.0.0")
    #[serde(default = "default_config_version")]
    pub version: String,
    /// System configuration (v2.1) - Decouples logical roles from measurements
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemConfig>,
    /// Map of channel name to speaker configuration
    pub speakers: HashMap<String, SpeakerConfig>,
    /// Optional crossover configuration for multi-driver groups
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossovers: Option<HashMap<String, CrossoverConfig>>,
    /// Optional target curve (freq, spl)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_curve: Option<TargetCurveConfig>,
    /// Optimizer configuration
    #[serde(default)]
    pub optimizer: OptimizerConfig,
    /// Recording configuration (device settings, signal parameters used during capture)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recording_config: Option<RecordingConfiguration>,
    /// Cross-talk cancellation / binaural-aware correction configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctc: Option<CtcConfig>,
    /// Pre-fetched CEA2034 data (runtime only, not serialized).
    #[serde(skip)]
    #[schemars(skip)]
    pub cea2034_cache: Option<HashMap<String, crate::read::Cea2034Data>>,
}

impl RoomConfig {
    /// Validate invariants required before any RoomEQ engine run.
    pub fn validate_structure(&self) -> Result<(), String> {
        if self.speakers.is_empty() {
            return Err("room configuration requires at least one speaker".to_string());
        }
        for (name, speaker) in &self.speakers {
            if name.trim().is_empty() {
                return Err("speaker names must not be empty".to_string());
            }
            if let SpeakerConfig::Topology(topology) = speaker {
                topology.validate().map_err(|error| {
                    format!("speaker group '{name}' has invalid topology: {error}")
                })?;
            }
        }
        if !self.optimizer.min_freq.is_finite()
            || !self.optimizer.max_freq.is_finite()
            || self.optimizer.min_freq <= 0.0
            || self.optimizer.min_freq >= self.optimizer.max_freq
        {
            return Err(format!(
                "optimizer frequency range must be finite, positive, and increasing; got [{}, {}] Hz",
                self.optimizer.min_freq, self.optimizer.max_freq
            ));
        }
        Ok(())
    }

    /// Resolve relative paths in this room configuration against a base directory
    pub fn resolve_paths(&mut self, base_dir: &std::path::Path) {
        for speaker in self.speakers.values_mut() {
            speaker.resolve_paths(base_dir);
        }
        if let Some(TargetCurveConfig::Path(ref mut path)) = self.target_curve
            && path.is_relative()
        {
            *path = base_dir.join(&*path);
        }
        if let Some(ctc) = &mut self.ctc {
            ctc.resolve_paths(base_dir);
        }
    }
}
