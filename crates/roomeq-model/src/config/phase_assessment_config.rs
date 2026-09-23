//! Gate parameters for the excess-phase assessment behind phase correction.
//!
//! The assessment itself lives in `roomeq-analysis` and deliberately offers
//! no defaults: whoever authorizes phase correction must state the evidence
//! bar explicitly. These workflow-side defaults mirror the analysis-validated
//! values so a configured phase action is assessable out of the box; every
//! field stays user-overridable and is reported with the decision.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

fn is_false(value: &bool) -> bool {
    !*value
}

fn default_taper_oct() -> f64 {
    0.5
}
fn default_snr_floor_db() -> f64 {
    10.0
}
fn default_min_valid_fraction() -> f64 {
    0.8
}
fn default_smooth_narrow_oct() -> f64 {
    1.0 / 6.0
}
fn default_smooth_wide_oct() -> f64 {
    1.0 / 2.0
}
fn default_consistency_tol_ms() -> f64 {
    0.5
}
fn default_dip_depth_db() -> f64 {
    6.0
}

/// Evidence bar for excess-phase assessment before phase correction.
///
/// Bins below `snr_floor_db` are excluded; assessment returns unknown
/// unless `min_valid_fraction` of the band stays usable. Narrow-vs-wide
/// group-delay disagreement above `consistency_tol_ms` marks the phase
/// window-sensitive. `strict_dips` forces unknown on any uncertain dip.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PhaseAssessmentConfig {
    /// Retain existing magnitude processing when per-driver phase evidence is refused.
    ///
    /// Opt-in workflow policy: no replacement FIR is designed, and a phase
    /// refusal is recorded. Shared-FIR behavior is unchanged. Invalid capture
    /// timing, malformed input, and operational failures still abort.
    #[serde(default, skip_serializing_if = "is_false")]
    pub retain_magnitude_on_refusal: bool,
    /// Cosine taper width in octaves at the assessment band edges.
    #[serde(default = "default_taper_oct")]
    pub taper_oct: f64,
    /// Bins below this SNR in dB are excluded from the assessment.
    #[serde(default = "default_snr_floor_db")]
    pub snr_floor_db: f64,
    /// Minimum usable fraction of in-band bins.
    #[serde(default = "default_min_valid_fraction")]
    pub min_valid_fraction: f64,
    /// Fractional-octave smoothing width for excess group delay.
    #[serde(default = "default_smooth_narrow_oct")]
    pub smooth_narrow_oct: f64,
    /// Second wider smoothing width for the window-sensitivity check.
    #[serde(default = "default_smooth_wide_oct")]
    pub smooth_wide_oct: f64,
    /// Maximum tolerated narrow-vs-wide group-delay disagreement in ms.
    #[serde(default = "default_consistency_tol_ms")]
    pub consistency_tol_ms: f64,
    /// When true, any uncertain dip forces an unknown verdict.
    #[serde(default)]
    pub strict_dips: bool,
    /// Depth below the wide-smoothed magnitude that counts as a dip in dB.
    #[serde(default = "default_dip_depth_db")]
    pub dip_depth_db: f64,
}

impl Default for PhaseAssessmentConfig {
    fn default() -> Self {
        Self {
            retain_magnitude_on_refusal: false,
            taper_oct: default_taper_oct(),
            snr_floor_db: default_snr_floor_db(),
            min_valid_fraction: default_min_valid_fraction(),
            smooth_narrow_oct: default_smooth_narrow_oct(),
            smooth_wide_oct: default_smooth_wide_oct(),
            consistency_tol_ms: default_consistency_tol_ms(),
            strict_dips: false,
            dip_depth_db: default_dip_depth_db(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnitude_retention_is_explicit_and_roundtrips() {
        let default: PhaseAssessmentConfig = serde_json::from_str("{}").unwrap();
        assert!(!default.retain_magnitude_on_refusal);
        assert!(
            serde_json::to_value(default)
                .unwrap()
                .get("retain_magnitude_on_refusal")
                .is_none()
        );
        let enabled: PhaseAssessmentConfig =
            serde_json::from_str(r#"{"retain_magnitude_on_refusal":true}"#).unwrap();
        let restored: PhaseAssessmentConfig =
            serde_json::from_value(serde_json::to_value(enabled).unwrap()).unwrap();
        assert!(restored.retain_magnitude_on_refusal);
    }
}
