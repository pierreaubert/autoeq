use super::default::default_flatness_erb_weight;
use super::temporal_masking_config::TemporalMaskingConfig;
use crate::loss::enhanced_weights::FrequencyBandWeights;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Experimental EPA diagnostics and optimizer flatness configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EpaConfig {
    /// Listening level in phon for diagnostic loudness computation.
    pub listening_level_phon: f64,
    /// Diagnostic target sharpness in acum (1.0 = natural broadband noise character).
    pub target_sharpness: f64,
    /// Diagnostic roughness threshold; transfer-only roughness does not steer optimization.
    pub max_roughness: f64,
    /// Diagnostic weights for the three reported EPA dimensions.
    pub evaluation_weight: f64,
    pub potency_weight: f64,
    pub activity_weight: f64,
    /// Band weights used for the flatness component of the EPA loss.
    /// Only consulted when `flatness_band_weight > 0`.
    #[serde(default)]
    pub flatness_band_weights: FrequencyBandWeights,
    /// ERB-rate weight for the active optimizer flatness component.
    /// The default is 1.0 (pure ERB-rate flatness).
    #[serde(default = "default_flatness_erb_weight")]
    pub flatness_erb_weight: f64,
    /// Band weight for the flatness component of the EPA loss.
    /// Default 0.0 (see `flatness_erb_weight`).
    #[serde(default)]
    pub flatness_band_weight: f64,
    /// Experimental temporal diagnostics for modal decay and FIR phase behavior.
    /// These settings do not steer EPA optimization. Confident measured SSIR
    /// decay is preferred for modal reporting; magnitude Q is only a fallback.
    #[serde(default)]
    pub temporal_masking: TemporalMaskingConfig,
}

impl Default for EpaConfig {
    fn default() -> Self {
        Self {
            listening_level_phon: 75.0,
            target_sharpness: 1.2,
            max_roughness: 0.5,
            evaluation_weight: 0.6,
            potency_weight: 0.2,
            activity_weight: 0.2,
            flatness_band_weights: FrequencyBandWeights::default(),
            flatness_erb_weight: 1.0,
            flatness_band_weight: 0.0,
            temporal_masking: TemporalMaskingConfig::default(),
        }
    }
}
