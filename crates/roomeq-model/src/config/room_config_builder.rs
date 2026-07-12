//! Builder for [`RoomConfig`].
//!
//! This builder is intentionally minimal: a room configuration is a container
//! for speakers plus an optimizer, and the fluent API focuses on the fields that
//! tests and workflows actually vary.

use super::{OptimizerConfig, RoomConfig, SpeakerConfig, SystemConfig};
use std::collections::HashMap;
use std::path::Path;

/// Fluent builder for [`RoomConfig`].
#[derive(Debug, Clone, Default)]
pub struct RoomConfigBuilder(RoomConfig);

impl RoomConfigBuilder {
    /// Create a builder with an empty room configuration.
    pub fn new() -> Self {
        Self(RoomConfig::default())
    }

    /// Create a builder seeded with an existing configuration.
    pub fn from_config(config: RoomConfig) -> Self {
        Self(config)
    }

    /// Add or replace a speaker channel.
    #[must_use]
    pub fn speaker(mut self, name: impl Into<String>, config: SpeakerConfig) -> Self {
        self.0.speakers.insert(name.into(), config);
        self
    }

    /// Set the entire speaker map.
    #[must_use]
    pub fn speakers(mut self, speakers: HashMap<String, SpeakerConfig>) -> Self {
        self.0.speakers = speakers;
        self
    }

    /// Set the optimizer configuration.
    #[must_use]
    pub fn optimizer(mut self, optimizer: OptimizerConfig) -> Self {
        self.0.optimizer = optimizer;
        self
    }

    /// Set the system configuration.
    #[must_use]
    pub fn system(mut self, system: SystemConfig) -> Self {
        self.0.system = Some(system);
        self
    }

    /// Set the target curve configuration.
    #[must_use]
    pub fn target_curve(mut self, target_curve: super::TargetCurveConfig) -> Self {
        self.0.target_curve = Some(target_curve);
        self
    }

    /// Set the crossover map.
    #[must_use]
    pub fn crossovers(mut self, crossovers: HashMap<String, super::CrossoverConfig>) -> Self {
        self.0.crossovers = Some(crossovers);
        self
    }

    /// Set the recording configuration.
    #[must_use]
    pub fn recording_config(mut self, recording_config: super::RecordingConfiguration) -> Self {
        self.0.recording_config = Some(recording_config);
        self
    }

    /// Set the CTC configuration.
    #[must_use]
    pub fn ctc(mut self, ctc: super::CtcConfig) -> Self {
        self.0.ctc = Some(ctc);
        self
    }

    /// Resolve relative paths against a base directory.
    ///
    /// This is a convenience that delegates to [`RoomConfig::resolve_paths`].
    #[must_use]
    pub fn resolve_paths(mut self, base_dir: &Path) -> Self {
        self.0.resolve_paths(base_dir);
        self
    }

    /// Consume the builder and return the configured [`RoomConfig`].
    pub fn build(self) -> RoomConfig {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::RoomConfigBuilder;
    use crate::Curve;
    use crate::MeasurementSource;
    use crate::roomeq::types::{OptimizerConfigBuilder, SpeakerConfig};

    #[test]
    fn builder_collects_speakers_and_optimizer() {
        let config = RoomConfigBuilder::new()
            .speaker(
                "L",
                SpeakerConfig::Single(MeasurementSource::InMemory(Curve::default())),
            )
            .optimizer(OptimizerConfigBuilder::new().num_filters(8).build())
            .build();
        assert!(config.speakers.contains_key("L"));
        assert_eq!(config.optimizer.num_filters, 8);
    }
}
