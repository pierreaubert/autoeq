use super::ctc_config::CtcConfig;
use super::default::{default_config_version, validate_config_version};
use super::optimizer_config::OptimizerConfig;
use super::speaker_config::SpeakerConfig;
use super::types::CrossoverConfig;
use super::types::RecordingConfiguration;
use super::types::SystemConfig;
use super::types::TargetCurveConfig;
use super::{ConfigValidationReport, ValidationStage};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete room configuration
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RoomConfig {
    /// Configuration schema version. Supported lines are 1.0.x–1.2.x and
    /// 2.0.x–2.1.x; the current version is 2.1.0.
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

impl Default for RoomConfig {
    fn default() -> Self {
        Self {
            version: default_config_version(),
            system: None,
            speakers: HashMap::new(),
            crossovers: None,
            target_curve: None,
            optimizer: OptimizerConfig::default(),
            recording_config: None,
            ctc: None,
            cea2034_cache: None,
        }
    }
}

impl RoomConfig {
    /// Validate the serialized configuration schema version.
    pub fn validate_version(&self) -> Result<(), String> {
        validate_config_version(&self.version)
    }

    /// Validate invariants required before any RoomEQ engine run.
    pub fn validate_structure(&self) -> Result<(), String> {
        let report = self.validation_report();
        report.errors().next().cloned().map_or(Ok(()), Err)
    }

    /// Run the engine-neutral portion of the canonical validation pipeline.
    ///
    /// This intentionally runs only schema/version and structural validation.
    /// Callers must inspect [`ConfigValidationReport::production_ready`] rather
    /// than treating this report as resolved-resource/acoustic/export evidence.
    pub fn validation_report(&self) -> ConfigValidationReport {
        let mut report = ConfigValidationReport::new();
        let version_errors = self.validate_version().err().into_iter().collect();
        report.record(ValidationStage::SchemaVersion, version_errors, Vec::new());
        report.record(
            ValidationStage::Structural,
            self.structural_errors(),
            Vec::new(),
        );
        report
    }

    fn structural_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.speakers.is_empty() {
            errors.push("room configuration requires at least one speaker".to_string());
        }
        for (name, speaker) in &self.speakers {
            if name.trim().is_empty() {
                errors.push("speaker names must not be empty".to_string());
            }
            if let SpeakerConfig::Topology(topology) = speaker
                && let Err(error) = topology.validate()
            {
                errors.push(format!(
                    "speaker group '{name}' has invalid topology: {error}"
                ));
            }
        }
        if let Some(system) = &self.system {
            for (role, measurement_key) in &system.speakers {
                if !self.speakers.contains_key(measurement_key) {
                    errors.push(format!(
                        "system.speakers role '{role}' references missing speaker measurement '{measurement_key}'"
                    ));
                }
            }
        }
        if let Some(crossovers) = &self.crossovers {
            for (name, crossover) in crossovers {
                if crossover
                    .crossover_type
                    .parse::<crate::loss::CrossoverType>()
                    .is_err()
                {
                    errors.push(format!(
                        "Crossover '{name}' has unsupported type '{}'",
                        crossover.crossover_type
                    ));
                }
            }
        }
        errors.extend(self.optimizer.gain_envelope_errors());
        if !self.optimizer.min_freq.is_finite()
            || !self.optimizer.max_freq.is_finite()
            || self.optimizer.min_freq <= 0.0
            || self.optimizer.min_freq >= self.optimizer.max_freq
        {
            errors.push(format!(
                "optimizer frequency range must be finite, positive, and increasing; got [{}, {}] Hz",
                self.optimizer.min_freq, self.optimizer.max_freq
            ));
        }
        errors
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
