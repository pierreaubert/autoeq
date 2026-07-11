use crate::Curve;
use crate::workflow::DriverOptimizationResult;

/// Result of multi-sub optimization with all-pass filters.
#[derive(Debug, Clone)]
pub struct MultiSubAllPassResult {
    /// Standard optimization result (gains, delays)
    pub base: DriverOptimizationResult,
    /// Per-subwoofer all-pass filter parameters: Vec of (frequency_hz, q)
    pub allpass_filters: Vec<(f64, f64)>,
    /// Combined frequency response after optimization
    pub combined_curve: Curve,
    /// Whether delay and all-pass controls were enabled from trustworthy phase data.
    pub phase_controls_enabled: bool,
    /// Machine-readable advisories raised while selecting the optimization mode.
    pub advisories: Vec<String>,
}
