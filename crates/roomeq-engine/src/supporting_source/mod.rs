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
    /// Gain removed by peak-normalizing `taps`, restored as a separate DSP
    /// gain stage before convolution.
    pub normalization_gain_db: f64,
    /// Effective target after constraints (for diagnostics).
    pub constrained_target: Curve,
    /// Band-windowed supporting-source gain response in dB, on the same
    /// frequency grid as `constrained_target`.
    pub support_gain_db: Vec<f64>,
    /// DRR before compensation (dB per frequency bin), when it was derived
    /// from time-gated impulse-response evidence.
    pub drr_before_db: Option<Vec<f64>>,
    /// DRR after compensation (dB per frequency bin), when it was derived
    /// from time-gated impulse-response evidence.
    pub drr_after_db: Option<Vec<f64>>,
    /// Number of precedence-limit hits (diagnostic).
    pub precedence_limit_hits: usize,
}
