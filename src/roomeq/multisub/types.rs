use crate::Curve;
use crate::workflow::DriverOptimizationResult;

/// Separate spatial and temporal views of an optimized multi-sub system.
#[derive(Debug, Clone)]
pub struct MultiSubCombinedResponse {
    /// Magnitude-only spatial aggregate used for global EQ and score reporting.
    pub spatial_magnitude: Curve,
    /// Complex response at the configured primary seat, when phase is trustworthy.
    pub primary_seat_complex: Option<Curve>,
}

impl MultiSubCombinedResponse {
    /// Compatibility view used by callers that historically consumed one curve.
    pub fn legacy_combined_curve(&self) -> Curve {
        self.primary_seat_complex
            .clone()
            .unwrap_or_else(|| self.spatial_magnitude.clone())
    }
}

/// Detailed result for standard multi-sub optimization.
#[derive(Debug, Clone)]
pub struct MultiSubOptimizationResult {
    pub base: DriverOptimizationResult,
    pub combined_response: MultiSubCombinedResponse,
    pub phase_controls_enabled: bool,
    pub advisories: Vec<String>,
}

/// Result of multi-sub optimization with all-pass filters.
#[derive(Debug, Clone)]
pub struct MultiSubAllPassResult {
    /// Standard optimization result (gains, delays)
    pub base: DriverOptimizationResult,
    /// Per-subwoofer all-pass filter parameters: Vec of (frequency_hz, q)
    pub allpass_filters: Vec<(f64, f64)>,
    /// Combined frequency response after optimization
    pub combined_curve: Curve,
    /// Separate spatial magnitude and primary-seat complex responses.
    pub combined_response: MultiSubCombinedResponse,
    /// Whether delay and all-pass controls were enabled from trustworthy phase data.
    pub phase_controls_enabled: bool,
    /// Machine-readable advisories raised while selecting the optimization mode.
    pub advisories: Vec<String>,
}
