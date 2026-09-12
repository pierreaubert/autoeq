//! Neutral policy types consumed by the optimizer without depending on RoomEQ.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct AudibilityDeadbandConfig {
    pub enabled: bool,
    pub bass_db: f64,
    pub mid_db: f64,
    pub treble_db: f64,
    pub bass_mid_hz: f64,
    pub mid_treble_hz: f64,
    pub disable_below_schroeder: bool,
    pub schroeder_hz: f64,
}

/// Explicit evidence for a measured modal resonance.
///
/// This is deliberately separate from an optimizer seed tuple: a mode is
/// usable by a proximity safety rule only when its frequency, narrowness and
/// prominence came from the measurement/decomposition stage.  Missing
/// evidence means no mode-proximity veto is asserted.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ModeProximityEvidence {
    pub frequency_hz: f64,
    pub q: f64,
    pub prominence_db: f64,
    /// Optional measured temporal severity above the decay threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_severity_db: Option<f64>,
}

impl Default for AudibilityDeadbandConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bass_db: 0.25,
            mid_db: 0.75,
            treble_db: 1.0,
            bass_mid_hz: 250.0,
            mid_treble_hz: 2_000.0,
            disable_below_schroeder: true,
            schroeder_hz: 300.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AudibilityDeadbandConfig;

    #[test]
    fn deadband_default_uses_canonical_schroeder_fallback() {
        assert_eq!(AudibilityDeadbandConfig::default().schroeder_hz, 300.0);
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MultiMeasurementStrategy {
    #[default]
    Average,
    WeightedSum,
    Minimax,
    VariancePenalized,
    SpatialRobustness,
    MinimaxUncertainty,
}
