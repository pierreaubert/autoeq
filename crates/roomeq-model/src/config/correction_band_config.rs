use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Optional, explicit frequency range in which RoomEQ is allowed to shape
/// the response.
///
/// The optimizer's observation/evaluation band remains unchanged. When the
/// range is narrower than that band, bins outside it are retained in the
/// measured graph and scorecard; with `allow_natural_rolloff`, the existing
/// source response is allowed to remain there instead of being flattened.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CorrectionBandPolicy {
    /// Lower edge of the correction support in Hz.
    pub min_hz: f64,
    /// Upper edge of the correction support in Hz.
    pub max_hz: f64,
    /// Permit measured response outside the correction support to remain
    /// naturally rolled off. This never installs a high-pass or low-pass.
    #[serde(default)]
    pub allow_natural_rolloff: bool,
}

impl CorrectionBandPolicy {
    /// Validate the policy against the configured observation band.
    pub fn validate_against(
        &self,
        configured_min_hz: f64,
        configured_max_hz: f64,
    ) -> Result<(), String> {
        if !self.min_hz.is_finite()
            || !self.max_hz.is_finite()
            || self.min_hz <= 0.0
            || self.max_hz <= self.min_hz
        {
            return Err(
                "correction_band min_hz/max_hz must be finite with 0 < min_hz < max_hz".into(),
            );
        }
        if !configured_min_hz.is_finite()
            || !configured_max_hz.is_finite()
            || configured_min_hz <= 0.0
            || configured_max_hz <= configured_min_hz
        {
            return Err("optimizer observation band is invalid".into());
        }
        if self.min_hz < configured_min_hz || self.max_hz > configured_max_hz {
            return Err(format!(
                "correction_band [{:.1}, {:.1}] Hz must lie inside optimizer observation band [{:.1}, {:.1}] Hz",
                self.min_hz, self.max_hz, configured_min_hz, configured_max_hz
            ));
        }
        if !self.allow_natural_rolloff
            && (self.min_hz > configured_min_hz || self.max_hz < configured_max_hz)
        {
            return Err(
                "narrow correction_band requires allow_natural_rolloff=true (otherwise it would silently leave scored bins uncorrected)"
                    .into(),
            );
        }
        Ok(())
    }
}

