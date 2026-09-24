use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Report-policy inputs that need an explicit operator declaration.
///
/// These change no acceptance math: they declare the policy the viewer uses
/// to render summary cells from emitted evidence. Absent stays pending.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ReportingConfig {
    /// Declared ±tolerance in seconds for the report Section 1 T60 flatness
    /// share (nine measured octave fits within tolerance of the
    /// complete-channel room mean). When `None`, the viewer applies the
    /// ITU-R BS.1116-2 §8.2.3.1 Fig. 1 midband default of ±0.05 s and labels
    /// the cell accordingly. A present value must be finite and positive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t60_flatness_tolerance_s: Option<f64>,
}

impl ReportingConfig {
    /// Validate the declared report policy.
    ///
    /// # Errors
    ///
    /// Returns a reason when the declared T60 tolerance is nonfinite or not
    /// positive.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(tolerance) = self.t60_flatness_tolerance_s
            && (!tolerance.is_finite() || tolerance <= 0.0)
        {
            return Err(format!(
                "reporting.t60_flatness_tolerance_s must be finite and positive (got {tolerance})"
            ));
        }
        Ok(())
    }
}
