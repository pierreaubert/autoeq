use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Configuration for building a distance- and directivity-weighted RIR prototype.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RirPrototypeConfig {
    /// Optimal listening position (e.g., center of the head at the main seat).
    pub reference_position: [f64; 3],
    /// Position of the main loudspeaker that defines the forward/directivity axis.
    pub source_position: [f64; 3],
    /// Position of each microphone, in the same order as the measurements.
    pub microphone_positions: Vec<[f64; 3]>,
    /// How distance translates into a weight.
    #[serde(default)]
    pub distance_mode: DistanceWeightMode,
    /// Which directivity model to apply.
    #[serde(default)]
    pub directivity: DirectivityModel,
    /// If true, apply directivity per frequency bin; otherwise use a broadband factor at 1 kHz.
    #[serde(default)]
    pub frequency_dependent_directivity: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DistanceWeightMode {
    /// w = 1 / d^2, clipped at a minimum distance to avoid infinities.
    #[default]
    InverseSquare,
    /// w = exp(-d^2 / (2 * sigma^2)).
    Gaussian { sigma_m: f64 },
    /// All microphones weighted equally (useful for baselines).
    Uniform,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DirectivityModel {
    /// No directivity correction.
    #[default]
    Omnidirectional,
    /// Rigid-sphere head-shadow approximation.
    SphericalHead { radius_m: f64 },
}
