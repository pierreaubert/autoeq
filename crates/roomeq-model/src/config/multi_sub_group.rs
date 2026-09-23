use crate::MeasurementSource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Configuration for multiple subwoofers
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MultiSubGroup {
    /// Name of the subwoofer group (e.g. "subs")
    pub name: String,
    /// Optional speaker model name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
    /// Measurements for each subwoofer
    pub subwoofers: Vec<MeasurementSource>,
    /// Enable per-subwoofer all-pass filter optimization.
    /// Measurement-backed workflows require shared stationary timing evidence.
    #[serde(default)]
    pub allpass_optimization: bool,
    /// Select joint coherent multi-sub optimization over the
    /// source-by-seat transfer matrix instead of the variation-only
    /// detailed mode. Requires per-seat measurements with phase on
    /// every sub and a verified shared timing reference. Refuses missing
    /// evidence without running an alternate optimizer. Takes precedence over legacy
    /// `optimizer.multi_seat` processing for this group. Default: false.
    /// Routed systems must select subwoofer strategy `mso`; `single` would
    /// select independent subs and is rejected when joint mode is requested.
    #[serde(default)]
    pub joint_optimization: bool,
}

impl MultiSubGroup {
    pub fn resolve_paths(&mut self, base_dir: &std::path::Path) {
        for m in &mut self.subwoofers {
            m.resolve_paths(base_dir);
        }
    }
}
