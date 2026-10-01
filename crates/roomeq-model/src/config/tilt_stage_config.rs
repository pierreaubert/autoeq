use super::default::default_true;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Appends an optimizer-driven LS plus HS tilt pair at the DSP chain end.
///
/// The pair extends `num_filters` by two instead of replacing peak slots, so
/// enabling tilt never moves the main filters' bounds. Both shelves use the
/// existing shelf limits (+/-`max_db` gain, pinned Q) and flow through the
/// same HF guard, audibility veto, and headroom stages as every other filter.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct TiltStageConfig {
    /// Enable the trailing tilt pair.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// LS hinge band `[lo, hi]` in Hz (`None` selects the lower geometric
    /// half of the correction band).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ls_band_hz: Option<[f64; 2]>,
    /// HS hinge band `[lo, hi]` in Hz (`None` selects the upper geometric
    /// half of the correction band).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hs_band_hz: Option<[f64; 2]>,
}

impl Default for TiltStageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ls_band_hz: None,
            hs_band_hz: None,
        }
    }
}
