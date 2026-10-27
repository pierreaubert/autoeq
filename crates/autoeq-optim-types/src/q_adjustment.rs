//! Local-Q projection evidence independent of room configuration.

// Rust guideline compliant 2026-02-21
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// One local-Q projection at a decoded filter center.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QAdjustment {
    /// Index of the adjusted filter.
    pub filter_index: usize,
    /// Decoded filter center in Hz.
    pub center_hz: f64,
    /// Q before projection.
    pub q_before: f64,
    /// Q after projection (the effective cap).
    pub q_after: f64,
    /// Global cap supplied by the caller.
    pub global_cap: f64,
    /// Local envelope cap at the center (`+inf` when unconfigured).
    pub local_cap: f64,
}

impl QAdjustment {
    /// Whether the global cap bound this filter rather than the local one.
    pub fn bound_by_global(&self) -> bool {
        self.global_cap <= self.local_cap
    }
}
