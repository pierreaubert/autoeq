//! Warm-start checkpoint save/restore for optimization runs.
//!
//! This stores a **warm-start record, not an exact-continuation checkpoint**:
//! no production path restores optimizer internals (problem/config/data
//! identity, population, adaptation state, RNG state), so resuming from this
//! file restarts the optimizer from the saved parameters with a fresh
//! population/adaptation/RNG stream. It must never be presented as
//! bit-identical continuation of an uninterrupted run. Implementing exact
//! DE/NSGA resume is explicitly out of scope.
//!
//! To keep warm starts sound, each record carries the measurement identity,
//! normalization hash, sample rate, and parameter bounds it was saved with.
//! Call [`OptimizerState::check_warm_start_compatible`] before reusing a
//! record and reject the checkpoint on any mismatch.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Schema version of [`OptimizerState`]. Bump when the serialized contract
/// changes in an incompatible way.
pub const WARM_START_STATE_VERSION: u32 = 1;

/// Identity a warm-start record must match before it may seed a new run.
#[derive(Debug, Clone)]
pub struct WarmStartIdentity<'a> {
    /// Hash identifying the measurement data (e.g. content hash).
    pub measurement_identity: &'a str,
    /// Hash of the normalization applied to the measurement.
    pub normalization_hash: Option<&'a str>,
    /// Sample rate in Hz; controls Nyquist/filter realization.
    pub sample_rate: f64,
    /// Parameter bounds of the run to seed.
    pub lower_bounds: &'a [f64],
    /// Parameter bounds of the run to seed.
    pub upper_bounds: &'a [f64],
}

/// Saved warm-start record for optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Best parameters found so far (warm-start point, not live population).
    pub best_params: Vec<f64>,
    /// Current parameters
    pub current_params: Vec<f64>,
    /// Best loss value
    pub best_loss: f64,
    /// Current iteration
    pub iteration: usize,
    /// Total iterations requested
    pub total_iterations: usize,
    /// Whether optimization was converging
    pub converged: bool,
    /// Random seed used
    pub seed: Option<u64>,
    /// Timestamp of save
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Schema version; defaults on read for records saved before versioning.
    #[serde(default = "default_state_version")]
    pub state_version: u32,
    /// Hash identifying the measurement data this record was saved with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement_identity: Option<String>,
    /// Hash of the normalization applied to the measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization_hash: Option<String>,
    /// Sample rate in Hz at save time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<f64>,
    /// Parameter bounds at save time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lower_bounds: Option<Vec<f64>>,
    /// Parameter bounds at save time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upper_bounds: Option<Vec<f64>>,
    /// Optimizer algorithm name at save time (informational).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
}

const fn default_state_version() -> u32 {
    WARM_START_STATE_VERSION
}

impl OptimizerState {
    /// Reject this record as a warm-start seed when it does not describe the
    /// requested run. Any identity mismatch (measurement, normalization,
    /// sample rate, bounds, or unknown schema version) returns `Err` and the
    /// caller must start fresh instead of reusing the saved parameters.
    pub fn check_warm_start_compatible(
        &self,
        identity: &WarmStartIdentity<'_>,
    ) -> Result<(), String> {
        if self.state_version != WARM_START_STATE_VERSION {
            return Err(format!(
                "unsupported warm-start state version {}, expected {WARM_START_STATE_VERSION}",
                self.state_version
            ));
        }
        match &self.measurement_identity {
            Some(recorded) if recorded == identity.measurement_identity => {}
            _ => {
                return Err(format!(
                    "warm-start measurement identity mismatch: recorded {:?}, requested {:?}",
                    self.measurement_identity, identity.measurement_identity
                ));
            }
        }
        if let (Some(recorded), Some(requested)) =
            (&self.normalization_hash, identity.normalization_hash)
            && recorded != requested
        {
            return Err(format!(
                "warm-start normalization mismatch: recorded {recorded:?}, requested {requested:?}"
            ));
        }
        if let Some(recorded) = self.sample_rate
            && recorded != identity.sample_rate
        {
            return Err(format!(
                "warm-start sample-rate mismatch: recorded {recorded}, requested {}",
                identity.sample_rate
            ));
        }
        if let Some(recorded) = &self.lower_bounds
            && recorded.as_slice() != identity.lower_bounds
        {
            return Err("warm-start lower-bounds mismatch".to_string());
        }
        if let Some(recorded) = &self.upper_bounds
            && recorded.as_slice() != identity.upper_bounds
        {
            return Err("warm-start upper-bounds mismatch".to_string());
        }
        Ok(())
    }
}

/// Save a warm-start record to file.
pub fn save_optimizer_state(
    state: &OptimizerState,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(state)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load a warm-start record from file.
///
/// Returns `Ok(None)` when the file does not exist. The caller must still
/// pass the result through [`OptimizerState::check_warm_start_compatible`]
/// before using it: loading alone does not prove the record matches the
/// current measurement, normalization, sample rate, or bounds.
pub fn load_optimizer_state(
    path: &Path,
) -> Result<Option<OptimizerState>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(None);
    }

    let json = std::fs::read_to_string(path)?;
    let state = serde_json::from_str(&json)?;
    Ok(Some(state))
}

/// Get default state file path for a given output directory
pub fn get_state_file_path(output_dir: &Path) -> PathBuf {
    output_dir.join("optimizer_state.json")
}
