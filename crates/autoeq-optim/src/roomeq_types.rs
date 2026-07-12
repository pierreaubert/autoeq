//! Neutral policy types consumed by the optimizer without depending on RoomEQ.

use serde::{Deserialize, Serialize};
use schemars::JsonSchema;

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
            schroeder_hz: 250.0,
        }
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
