//! Warm-start checkpoint save/restore for optimization runs.
//!
//! These files store a warm-start candidate, not exact optimizer continuation.
//! A resumed run starts with the saved candidate and a fresh optimizer
//! population, adaptation state, and random stream.
//!
//! Reuse requires a complete measurement and configuration identity, an
//! explicit normalization decision, matching sample rate and bounds, and a
//! finite feasible candidate. Legacy records remain readable, but incomplete
//! records are rejected with a reason when used for a warm start.

// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Schema version of OptimizerState. Bump when its serialized contract changes.
pub const WARM_START_STATE_VERSION: u32 = 2;

const LEGACY_WARM_START_STATE_VERSION: u32 = 1;

/// Records whether normalization was applied and its identity when present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", content = "hash", rename_all = "snake_case")]
pub enum NormalizationMetadata {
    /// The input measurement was used without normalization.
    NotApplied,
    /// Normalization was applied; hash identifies its output.
    Applied { hash: String },
}

/// Identity a saved candidate must match before it may seed a new run.
#[derive(Debug, Clone)]
pub struct WarmStartIdentity<'a> {
    /// Hash identifying the measurement data (for example, a content hash).
    pub measurement_identity: &'a str,
    /// Hash of the search and correction configuration.
    pub config_identity: &'a str,
    /// Hash of applied normalization; None means normalization was not applied.
    pub normalization_hash: Option<&'a str>,
    /// Sample rate in Hz; controls Nyquist and filter realization.
    pub sample_rate: f64,
    /// Lower parameter bounds for this run.
    pub lower_bounds: &'a [f64],
    /// Upper parameter bounds for this run.
    pub upper_bounds: &'a [f64],
    /// Optimizer algorithm requested for this run.
    pub algorithm: &'a str,
    /// Optimizer implementation version requested for this run.
    pub algorithm_version: &'a str,
    /// Evaluation or iteration budget requested for this run.
    pub budget: usize,
}

/// Saved warm-start candidate and the identities needed to validate its reuse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Best feasible parameters found so far.
    pub best_params: Vec<f64>,
    /// Current parameters at the time of the checkpoint.
    pub current_params: Vec<f64>,
    /// Objective value for best_params.
    pub best_loss: f64,
    /// Current optimizer iteration or evaluation count.
    pub iteration: usize,
    /// Iteration or evaluation budget for this run.
    pub total_iterations: usize,
    /// Whether optimization had converged when the state was saved.
    pub converged: bool,
    /// Random seed used for the original run.
    pub seed: Option<u64>,
    /// Timestamp of save.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Checkpoint schema version; absent legacy values default to version 1.
    #[serde(default = "default_state_version")]
    pub state_version: u32,
    /// Hash identifying the measurement data this record was saved with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement_identity: Option<String>,
    /// Hash identifying all search and correction settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_identity: Option<String>,
    /// Explicit normalization decision; absent means unknown, not unnormalized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization_metadata: Option<NormalizationMetadata>,
    /// Legacy normalization hash retained for loading version 1 checkpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization_hash: Option<String>,
    /// Sample rate in Hz at save time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<f64>,
    /// Parameter lower bounds at save time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lower_bounds: Option<Vec<f64>>,
    /// Parameter upper bounds at save time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upper_bounds: Option<Vec<f64>>,
    /// Optimizer algorithm name at save time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    /// Optimizer implementation version at save time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm_version: Option<String>,
    /// Whether the saved candidate passed the run's declared feasibility checks.
    #[serde(default)]
    pub best_feasible: bool,
}

const fn default_state_version() -> u32 {
    LEGACY_WARM_START_STATE_VERSION
}

impl OptimizerState {
    /// Build a versioned checkpoint from a candidate whose feasibility was checked.
    ///
    /// The caller must set candidate_feasible only after validating the
    /// candidate against the run's constraints.
    ///
    /// # Examples
    ///
    /// ~~~rust
    /// use autoeq_workflow::workflow::resume::{
    ///     NormalizationMetadata, OptimizerState, WarmStartIdentity,
    /// };
    ///
    /// let identity = WarmStartIdentity {
    ///     measurement_identity: "measurement-hash",
    ///     config_identity: "config-hash",
    ///     normalization_hash: None,
    ///     sample_rate: 48_000.0,
    ///     lower_bounds: &[-1.0, -1.0],
    ///     upper_bounds: &[1.0, 1.0],
    ///     algorithm: "autoeq:de",
    ///     algorithm_version: "0.5.6",
    ///     budget: 100,
    /// };
    /// let state = OptimizerState::from_candidate(
    ///     &[0.0, 0.5],
    ///     0.25,
    ///     12,
    ///     100,
    ///     false,
    ///     Some(7),
    ///     true,
    ///     &identity,
    /// );
    /// assert_eq!(state.normalization_metadata, Some(NormalizationMetadata::NotApplied));
    /// ~~~
    #[expect(
        clippy::too_many_arguments,
        reason = "the complete saved-run identity is explicit at checkpoint construction"
    )]
    pub fn from_candidate(
        candidate: &[f64],
        loss: f64,
        iteration: usize,
        total_iterations: usize,
        converged: bool,
        seed: Option<u64>,
        candidate_feasible: bool,
        identity: &WarmStartIdentity<'_>,
    ) -> Self {
        let normalization_metadata = Some(match identity.normalization_hash {
            Some(hash) => NormalizationMetadata::Applied {
                hash: hash.to_owned(),
            },
            None => NormalizationMetadata::NotApplied,
        });
        Self {
            best_params: candidate.to_vec(),
            current_params: candidate.to_vec(),
            best_loss: loss,
            iteration,
            total_iterations,
            converged,
            seed,
            timestamp: chrono::Utc::now(),
            state_version: WARM_START_STATE_VERSION,
            measurement_identity: Some(identity.measurement_identity.to_owned()),
            config_identity: Some(identity.config_identity.to_owned()),
            normalization_metadata,
            normalization_hash: None,
            sample_rate: Some(identity.sample_rate),
            lower_bounds: Some(identity.lower_bounds.to_vec()),
            upper_bounds: Some(identity.upper_bounds.to_vec()),
            algorithm: Some(identity.algorithm.to_owned()),
            algorithm_version: Some(identity.algorithm_version.to_owned()),
            best_feasible: candidate_feasible,
        }
    }

    /// Reject a checkpoint that cannot safely seed the requested run.
    ///
    /// This validates both the saved candidate and the requested identity.
    ///
    /// # Errors
    ///
    /// Returns an actionable reason when metadata is missing or invalid, the
    /// schema is unsupported, or any requested identity differs.
    pub fn check_warm_start_compatible(
        &self,
        identity: &WarmStartIdentity<'_>,
    ) -> Result<(), String> {
        validate_identity(identity)?;
        self.validate_candidate_and_metadata()?;

        if !(LEGACY_WARM_START_STATE_VERSION..=WARM_START_STATE_VERSION)
            .contains(&self.state_version)
        {
            return Err(format!(
                "unsupported warm-start state version {}; expected a version from {LEGACY_WARM_START_STATE_VERSION} through {WARM_START_STATE_VERSION}",
                self.state_version
            ));
        }

        compare_identity(
            "measurement",
            self.measurement_identity.as_deref(),
            identity.measurement_identity,
        )?;
        compare_identity(
            "configuration",
            self.config_identity.as_deref(),
            identity.config_identity,
        )?;

        let expected_normalization = expected_normalization(identity);
        let recorded_normalization = self.recorded_normalization()?;
        if recorded_normalization != expected_normalization {
            return Err(format!(
                "warm-start normalization mismatch: recorded {recorded_normalization:?}, requested {expected_normalization:?}"
            ));
        }

        compare_sample_rate(self.sample_rate, identity.sample_rate)?;
        compare_bounds(self.lower_bounds.as_deref(), "lower", identity.lower_bounds)?;
        compare_bounds(self.upper_bounds.as_deref(), "upper", identity.upper_bounds)?;
        compare_identity(
            "optimizer algorithm",
            self.algorithm.as_deref(),
            identity.algorithm,
        )?;
        compare_identity(
            "optimizer version",
            self.algorithm_version.as_deref(),
            identity.algorithm_version,
        )?;
        if self.total_iterations != identity.budget {
            return Err(format!(
                "warm-start budget mismatch: recorded {}, requested {}",
                self.total_iterations, identity.budget
            ));
        }
        Ok(())
    }

    fn validate_for_save(&self) -> Result<(), String> {
        if self.state_version != WARM_START_STATE_VERSION {
            return Err(format!(
                "cannot save warm-start state version {}; expected current version {WARM_START_STATE_VERSION}",
                self.state_version
            ));
        }
        self.validate_candidate_and_metadata()?;
        if self.normalization_metadata.is_none() {
            return Err(
                "warm-start checkpoint is missing normalization decision metadata; choose NotApplied or Applied(hash)"
                    .to_string(),
            );
        }
        self.recorded_normalization()?;
        Ok(())
    }

    fn validate_candidate_and_metadata(&self) -> Result<(), String> {
        let measurement = self.measurement_identity.as_deref().ok_or_else(|| {
            "warm-start checkpoint is missing measurement identity; start a fresh run or save a new checkpoint".to_string()
        })?;
        validate_nonempty("measurement identity", measurement)?;

        let config = self.config_identity.as_deref().ok_or_else(|| {
            "warm-start checkpoint is missing configuration identity; start a fresh run or save a new checkpoint".to_string()
        })?;
        validate_nonempty("configuration identity", config)?;

        let rate = self.sample_rate.ok_or_else(|| {
            "warm-start checkpoint is missing sample rate; start a fresh run or save a new checkpoint".to_string()
        })?;
        validate_sample_rate(rate)?;

        let lower = self.lower_bounds.as_deref().ok_or_else(|| {
            "warm-start checkpoint is missing lower bounds; start a fresh run or save a new checkpoint".to_string()
        })?;
        let upper = self.upper_bounds.as_deref().ok_or_else(|| {
            "warm-start checkpoint is missing upper bounds; start a fresh run or save a new checkpoint".to_string()
        })?;

        validate_bounds(lower, upper)?;

        if self.best_params.is_empty() || self.best_params.len() != self.current_params.len() {
            return Err(format!(
                "warm-start parameter dimensions are invalid: best has {}, current has {}; expected equal nonzero dimensions",
                self.best_params.len(),
                self.current_params.len()
            ));
        }
        if self.best_params.len() != lower.len() {
            return Err(format!(
                "warm-start parameter dimension {} does not match bound dimension {}",
                self.best_params.len(),
                lower.len()
            ));
        }
        for (name, params) in [
            ("best", self.best_params.as_slice()),
            ("current", self.current_params.as_slice()),
        ] {
            for (index, (&value, (&min, &max))) in
                params.iter().zip(lower.iter().zip(upper)).enumerate()
            {
                if !value.is_finite() {
                    return Err(format!("warm-start {name} parameter {index} is not finite"));
                }
                if value < min || value > max {
                    return Err(format!(
                        "warm-start {name} parameter {index}={value} is outside saved bounds [{min}, {max}]"
                    ));
                }
            }
        }
        if !self.best_loss.is_finite() {
            return Err("warm-start best loss is not finite".to_string());
        }
        if self.total_iterations == 0 || self.iteration > self.total_iterations {
            return Err(format!(
                "warm-start iteration/budget is invalid: iteration {} of {}",
                self.iteration, self.total_iterations
            ));
        }
        let algorithm = self.algorithm.as_deref().ok_or_else(|| {
            "warm-start checkpoint is missing optimizer algorithm; start a fresh run or save a new checkpoint".to_string()
        })?;
        validate_nonempty("optimizer algorithm", algorithm)?;
        let algorithm_version = self.algorithm_version.as_deref().ok_or_else(|| {
            "warm-start checkpoint is missing optimizer version; start a fresh run or save a new checkpoint".to_string()
        })?;
        validate_nonempty("optimizer version", algorithm_version)?;
        if !self.best_feasible {
            return Err(
                "warm-start checkpoint does not identify its candidate as feasible; start a fresh run or save a feasible candidate"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn recorded_normalization(&self) -> Result<NormalizationMetadata, String> {
        if let Some(metadata) = &self.normalization_metadata {
            if let NormalizationMetadata::Applied { hash } = metadata {
                validate_nonempty("normalization hash", hash)?;
            }
            if let Some(legacy_hash) = self.normalization_hash.as_deref() {
                match metadata {
                    NormalizationMetadata::Applied { hash } if hash == legacy_hash => {}
                    _ => {
                        return Err(
                            "warm-start checkpoint has conflicting normalization metadata"
                                .to_string(),
                        );
                    }
                }
            }
            return Ok(metadata.clone());
        }

        if self.state_version == LEGACY_WARM_START_STATE_VERSION {
            if let Some(hash) = self.normalization_hash.as_deref() {
                validate_nonempty("legacy normalization hash", hash)?;
                return Ok(NormalizationMetadata::Applied {
                    hash: hash.to_owned(),
                });
            }
            return Err(
                "legacy warm-start checkpoint cannot distinguish no normalization from missing metadata; start a fresh run and save a version 2 checkpoint"
                    .to_string(),
            );
        }

        Err(
            "warm-start checkpoint is missing normalization decision metadata; start a fresh run and save a version 2 checkpoint"
                .to_string(),
        )
    }
}

/// Hash ordered configuration components into a stable SHA-256 identity.
///
/// Components are length-delimited, so different sequences cannot collide by
/// concatenation alone.
///
/// # Examples
///
/// ~~~
/// use autoeq_workflow::workflow::resume::config_identity_digest;
///
/// let identity = config_identity_digest(["speaker-flat", "48 kHz", "7 filters"]);
/// assert_eq!(identity.len(), 64);
/// ~~~
pub fn config_identity_digest<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    let mut digest = Sha256::new();
    digest.update(b"autoeq-warm-start-config-v1");
    for part in parts {
        digest.update((part.len() as u64).to_be_bytes());
        digest.update(part.as_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Resolve a requested algorithm name to the concrete backend identity saved
/// in a warm-start checkpoint.
pub fn canonical_optimizer_identity(requested: &str) -> Result<String, String> {
    autoeq_optim::optim::backend::resolve(requested)
        .map(|backend| backend.name().to_owned())
        .ok_or_else(|| format!("unknown optimizer backend: {requested}"))
}

/// Save a complete warm-start checkpoint with atomic file replacement.
///
/// The temporary file is created beside path, flushed, and renamed over the
/// destination. Failures before replacement leave an existing checkpoint
/// untouched. After replacement, a Unix parent-directory flush reports whether
/// the rename is durable; if that flush fails, the complete new checkpoint is
/// already visible but crash durability is uncertain. On other platforms,
/// directory-entry durability follows the filesystem API.
///
/// # Errors
///
/// Returns an error for incomplete or invalid checkpoint data, serialization
/// failures, or filesystem failures.
pub fn save_optimizer_state(
    state: &OptimizerState,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    state
        .validate_for_save()
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    let json = serde_json::to_vec_pretty(state)?;
    atomic_replace(path, |file| file.write_all(&json))?;
    Ok(())
}

/// Load a warm-start checkpoint, preserving compatibility with legacy files.
///
/// Returns None when the file does not exist. Loading does not prove the
/// record can be reused; pass it through OptimizerState::check_warm_start_compatible.
///
/// # Errors
///
/// Returns an error when the file cannot be read or its JSON is malformed.
pub fn load_optimizer_state(
    path: &Path,
) -> Result<Option<OptimizerState>, Box<dyn std::error::Error>> {
    let json = match fs::read_to_string(path) {
        Ok(json) => json,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let state = serde_json::from_str(&json)?;
    Ok(Some(state))
}

/// Get the default state file path for an output directory.
///
/// # Examples
///
/// ~~~
/// use std::path::Path;
/// use autoeq_workflow::workflow::resume::get_state_file_path;
///
/// assert_eq!(
///     get_state_file_path(Path::new("out")),
///     Path::new("out/optimizer_state.json")
/// );
/// ~~~
pub fn get_state_file_path(output_dir: &Path) -> PathBuf {
    output_dir.join("optimizer_state.json")
}

fn atomic_replace(path: &Path, write: impl FnOnce(&mut File) -> io::Result<()>) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    write(temporary.as_file_mut())?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    #[cfg(unix)]
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn validate_identity(identity: &WarmStartIdentity<'_>) -> Result<(), String> {
    validate_nonempty(
        "requested measurement identity",
        identity.measurement_identity,
    )?;
    validate_nonempty("requested configuration identity", identity.config_identity)?;
    if let Some(hash) = identity.normalization_hash {
        validate_nonempty("requested normalization hash", hash)?;
    }
    validate_sample_rate(identity.sample_rate)?;
    validate_bounds(identity.lower_bounds, identity.upper_bounds)?;
    validate_nonempty("requested optimizer algorithm", identity.algorithm)?;
    validate_nonempty("requested optimizer version", identity.algorithm_version)?;
    if identity.budget == 0 {
        return Err("warm-start requested budget must be greater than zero".to_string());
    }
    Ok(())
}

fn validate_nonempty(name: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("warm-start {name} is empty"));
    }
    Ok(())
}

fn validate_sample_rate(sample_rate: f64) -> Result<(), String> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(format!(
            "warm-start sample rate must be finite and positive, got {sample_rate}"
        ));
    }
    Ok(())
}

fn validate_bounds(lower: &[f64], upper: &[f64]) -> Result<(), String> {
    if lower.is_empty() || lower.len() != upper.len() {
        return Err(format!(
            "warm-start bounds have invalid dimensions: lower has {}, upper has {}; expected equal nonzero dimensions",
            lower.len(),
            upper.len()
        ));
    }
    for (index, (&min, &max)) in lower.iter().zip(upper).enumerate() {
        if !min.is_finite() || !max.is_finite() || min > max {
            return Err(format!(
                "warm-start bounds at parameter {index} must be finite and ordered, got [{min}, {max}]"
            ));
        }
    }
    Ok(())
}

fn expected_normalization(identity: &WarmStartIdentity<'_>) -> NormalizationMetadata {
    match identity.normalization_hash {
        Some(hash) => NormalizationMetadata::Applied {
            hash: hash.to_owned(),
        },
        None => NormalizationMetadata::NotApplied,
    }
}

fn compare_identity(name: &str, recorded: Option<&str>, requested: &str) -> Result<(), String> {
    match recorded {
        Some(recorded) if recorded == requested => Ok(()),
        Some(recorded) => Err(format!(
            "warm-start {name} identity mismatch: recorded {recorded:?}, requested {requested:?}"
        )),
        None => Err(format!(
            "warm-start checkpoint is missing {name} identity; start a fresh run and save a new checkpoint"
        )),
    }
}

fn compare_sample_rate(recorded: Option<f64>, requested: f64) -> Result<(), String> {
    match recorded {
        Some(recorded) if recorded == requested => Ok(()),
        Some(recorded) => Err(format!(
            "warm-start sample-rate mismatch: recorded {recorded}, requested {requested}"
        )),
        None => Err(
            "warm-start checkpoint is missing sample rate; start a fresh run and save a new checkpoint"
                .to_string(),
        ),
    }
}

fn compare_bounds(recorded: Option<&[f64]>, name: &str, requested: &[f64]) -> Result<(), String> {
    let recorded = recorded.ok_or_else(|| {
        format!(
            "warm-start checkpoint is missing {name}-bounds identity; start a fresh run and save a new checkpoint"
        )
    })?;
    if recorded != requested {
        return Err(format!("warm-start {name}-bounds mismatch"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{atomic_replace, canonical_optimizer_identity};
    use std::io::{self, Write};

    #[test]
    fn optimizer_aliases_record_the_resolved_backend_name() {
        assert_eq!(canonical_optimizer_identity("de").unwrap(), "autoeq:de");
        assert_eq!(
            canonical_optimizer_identity("nlopt:crs2lm").unwrap(),
            "autoeq:de"
        );
    }

    #[test]
    fn interrupted_atomic_write_preserves_previous_checkpoint() {
        let directory = tempfile::tempdir().expect("test directory should be created");
        let path = directory.path().join("optimizer_state.json");
        std::fs::write(&path, b"previous valid checkpoint")
            .expect("previous checkpoint should be created");

        let result = atomic_replace(&path, |file| {
            file.write_all(b"partial replacement")?;
            Err(io::Error::other("simulated interrupted write"))
        });

        assert!(result.is_err());
        assert_eq!(
            std::fs::read(&path).expect("previous checkpoint should remain readable"),
            b"previous valid checkpoint"
        );
        let leftovers: Vec<_> = std::fs::read_dir(directory.path())
            .expect("test directory should be readable")
            .filter_map(Result::ok)
            .collect();
        assert_eq!(leftovers.len(), 1, "temporary checkpoint should be removed");
    }
}
