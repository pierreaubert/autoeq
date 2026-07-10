use super::bootstrap_uncertainty_config::BootstrapUncertaintyConfig;
use super::default::default_variance_lambda;
use super::types::{MultiMeasurementStrategy, SpatialRobustnessSerdeConfig};
use crate::roomeq::rir_prototype::RirPrototypeConfig;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Configuration for multi-measurement optimization
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MultiMeasurementConfig {
    /// Strategy for combining per-measurement losses
    #[serde(default)]
    pub strategy: MultiMeasurementStrategy,
    /// Weights for WeightedSum (normalized internally). Equal if omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<Vec<f64>>,
    /// Lambda for VariancePenalized (default 1.0). Higher = more consistent across positions.
    #[serde(default = "default_variance_lambda")]
    pub variance_lambda: f64,
    /// Spatial robustness configuration (used when strategy = SpatialRobustness)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_robustness: Option<SpatialRobustnessSerdeConfig>,
    /// Bootstrap uncertainty configuration (used when strategy = MinimaxUncertainty).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_uncertainty: Option<BootstrapUncertaintyConfig>,
    /// Optional distance- and directivity-weighted RIR prototype.
    /// When set, measurements are collapsed into a single prototype before
    /// the multi-measurement strategy is applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rir_prototype: Option<RirPrototypeConfig>,
}

impl Default for MultiMeasurementConfig {
    fn default() -> Self {
        Self {
            strategy: MultiMeasurementStrategy::default(),
            weights: None,
            variance_lambda: default_variance_lambda(),
            spatial_robustness: None,
            bootstrap_uncertainty: None,
            rir_prototype: None,
        }
    }
}
