//! Recorded arrival evidence shared by capture tools and RoomEQ consumers.

use serde::{Deserialize, Serialize};

/// A detected event with explicit limits on its measured direction.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub struct CaptureArrival {
    /// Arrival time relative to the shared sweep origin, in milliseconds.
    pub arrival_ms: f64,
    /// Delay relative to the detected direct sound, in milliseconds.
    pub relative_ms: f64,
    /// First microphone event peak energy relative to direct sound, in dB.
    pub level_db: f64,
    /// Integrated event energy relative to each microphones direct segment, in capture order.
    #[serde(default)]
    pub microphone_energy_db: Vec<f64>,
    /// Unit vector toward the arriving source/image source, when identifiable.
    pub direction: Option<[f64; 3]>,
    /// True when a planar array leaves the side of arrival unresolved.
    pub mirror_ambiguous: bool,
    /// RMS pair-delay fit residual in samples; not an angular uncertainty.
    pub residual_samples: Option<f64>,
    /// Band used for the pair-delay estimate, in Hz.
    pub band_hz: Option<[f64; 2]>,
    /// Missing or ambiguous direction evidence is explained here.
    pub issues: Vec<String>,
}

/// Measured arrival candidates for one simultaneously captured source.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub struct CaptureReflectionReport {
    /// Sequentially excited source identity.
    pub source_id: String,
    /// SSIR direct-arrival candidate; detection is not a surveyed source label.
    pub direct_sound: Option<CaptureArrival>,
    /// Early reflection candidates, ordered by arrival time.
    pub early_reflections: Vec<CaptureArrival>,
    /// Conditions that prevented source-level analysis.
    pub issues: Vec<String>,
}
