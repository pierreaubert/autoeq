//! Supporting-source room compensation (Brooks-Park et al., JASA 159(4), 2026).
//!
//! This module computes a delayed, decorrelated supporting-source FIR filter
//! that adds reverberant energy to the room without touching the primary
//! loudspeaker's direct sound.

mod drr;
mod filter;
mod velvet;

pub use drr::{compute_drr, db_summary};
pub use filter::compute_supporting_source_filter;
pub use velvet::generate_velvet_noise;

use crate::Curve;

/// Result of computing a supporting-source filter.
#[derive(Debug, Clone)]
pub struct SupportingSourceFilter {
    /// Minimum-phase FIR taps applied to the supporting source.
    pub taps: Vec<f64>,
    /// Effective target after constraints (for diagnostics).
    pub constrained_target: Curve,
    /// Supporting-source gain response in dB.
    pub support_gain_db: Vec<f64>,
    /// DRR before compensation (dB per frequency bin).
    pub drr_before_db: Vec<f64>,
    /// DRR after compensation (dB per frequency bin).
    pub drr_after_db: Vec<f64>,
    /// Number of precedence-limit hits (diagnostic).
    pub precedence_limit_hits: usize,
}
