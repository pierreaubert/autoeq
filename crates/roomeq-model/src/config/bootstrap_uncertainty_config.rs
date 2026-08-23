use super::default::default_bootstrap_alpha;
use super::default::default_bootstrap_cvar_alpha;
use super::default::default_bootstrap_num_resamples;
use super::default::default_bootstrap_seed;
use super::types::BootstrapScalarisation;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Serializable spatial case-bootstrap configuration for JSON config files.
/// It estimates seat-sampling variability, not repeat-sweep or calibration uncertainty.
///
/// Drives `MultiMeasurementStrategy::MinimaxUncertainty`. At optimizer-setup
/// time, the input N measurement curves are case-bootstrap resampled B times;
/// each resampled mean becomes its own per-measurement objective. The outer
/// optimizer then scalarises the B objectives per `scalarisation`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct BootstrapUncertaintyConfig {
    /// Optional effective number of independent spatial positions.
    /// Values below the nominal seat count widen the spatial bootstrap by
    /// drawing fewer independent cases per resample.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_spatial_sample_size: Option<f64>,
    /// Separately measured repeat-sweep noise (dB standard deviation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_sweep_noise_std_db: Option<f64>,
    /// Separately supplied calibration uncertainty (dB standard deviation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_uncertainty_std_db: Option<f64>,
    /// Number of bootstrap resamples B. Typical: 200..1000. Default: 400.
    #[serde(default = "default_bootstrap_num_resamples")]
    pub num_resamples: usize,
    /// Two-sided confidence level α; band covers `[α/2, 1-α/2]`. Default: 0.10.
    #[serde(default = "default_bootstrap_alpha")]
    pub alpha: f64,
    /// PRNG seed for determinism.
    #[serde(default = "default_bootstrap_seed")]
    pub seed: u64,
    /// Scalarisation across the B resamples.
    #[serde(default)]
    pub scalarisation: BootstrapScalarisation,
    /// Tail fraction for CVaR (only used when `scalarisation = Cvar`). Default 0.20.
    #[serde(default = "default_bootstrap_cvar_alpha")]
    pub cvar_alpha: f64,
}

impl Default for BootstrapUncertaintyConfig {
    fn default() -> Self {
        Self {
            effective_spatial_sample_size: None,
            repeat_sweep_noise_std_db: None,
            calibration_uncertainty_std_db: None,
            num_resamples: default_bootstrap_num_resamples(),
            alpha: default_bootstrap_alpha(),
            seed: default_bootstrap_seed(),
            scalarisation: BootstrapScalarisation::default(),
            cvar_alpha: default_bootstrap_cvar_alpha(),
        }
    }
}
