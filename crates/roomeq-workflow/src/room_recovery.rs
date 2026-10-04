//! Durable recovery for the narrowly supported single-channel RoomEQ DE path.
//!
//! A recovery file atomically couples the exact math checkpoint with the
//! workflow disposition. A successful call to the math checkpoint callback is
//! the caller's durability acknowledgement; the optimizer itself does not
//! provide an `fsync` guarantee.

use autoeq_core::{Curve, MeasurementRef, MeasurementSource};
use roomeq_engine::eq::exact_recovery::{ExactDERecoveryOptions, ExactDERecoveryState};
use roomeq_model::{ProcessingMode, RoomConfig, SpeakerConfig, TargetCurveConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tempfile::NamedTempFile;

/// File containing the single-writer RoomEQ recovery journal.
pub const ROOM_RECOVERY_FILE_NAME: &str = "room-recovery.json";
const ROOM_RECOVERY_SCHEMA_VERSION: u32 = 2;
const ROOM_RECOVERY_LOCK_FILE_NAME: &str = ".room-recovery.lock";
const MAX_RECOVERY_ERROR_CHARS: usize = 2048;
const MAX_ROOM_RECOVERY_JOURNAL_BYTES: usize = 32 * 1024 * 1024;
const CALLBACK_STOP_MESSAGE: &str = "Optimization stopped by callback";

/// Durable state of the supported room run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoomRecoveryStatus {
    /// A supported run was created but has not admitted optimizer work.
    Prepared,
    /// The exact DE pass or downstream RoomEQ stages are active.
    Running,
    /// A prior process ended while the journal still described active work.
    Interrupted,
    /// The exact DE solver has a finalized terminal checkpoint.
    OptimizerComplete,
    /// RoomPipeline finalization and durable candidate staging completed; the
    /// canonical native bundle is not published yet.
    StageComplete,
    /// An exact candidate bundle is identified and publication is in progress.
    Publishing,
    /// The intended native graph and manifest are present at the canonical path.
    Committed,
    /// The operator or observer requested terminal cancellation.
    Cancelled,
    /// A non-recoverable workflow failure was recorded.
    Failed,
}

/// Inputs needed to open a single-writer recovery session.
pub struct RoomRecoveryOpen<'a> {
    /// Private directory holding the journal and lock.
    pub directory: &'a Path,
    /// Canonical output root that will be published after finalization.
    pub output_path: &'a Path,
    /// Fully resolved RoomEQ configuration.
    pub room_config: &'a RoomConfig,
    /// Sample rate used for optimization and realization.
    pub sample_rate_hz: f64,
    /// Number of log-frequency samples used by the workflow.
    pub frequency_samples: usize,
    /// Whether to resume an existing interrupted or incomplete run.
    pub resume: bool,
}

type OutputIdentity = crate::output_bundle::BundleContentIdentity;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputIntent {
    previous: Option<OutputIdentity>,
    intended: OutputIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StageCandidate {
    path: PathBuf,
    identity: OutputIdentity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RoomRecoveryJournal {
    schema_version: u32,
    run_identity: String,
    channel_id: String,
    config_sha256: String,
    input_sha256_by_path: BTreeMap<String, String>,
    measurement_curve_sha256: String,
    sample_rate_bits: String,
    frequency_samples: usize,
    status: RoomRecoveryStatus,
    attempt: u64,
    recovery_count: u64,
    last_resume_from: Option<RoomRecoveryStatus>,
    logical_search_evaluations: usize,
    checkpointed_search_evaluations_this_process: usize,
    checkpoint: Option<ExactDERecoveryState>,
    checkpoint_run_identity: Option<String>,
    stage_candidate: Option<StageCandidate>,
    previous_output: Option<OutputIdentity>,
    output_intent: Option<OutputIntent>,
    failure_reason: Option<String>,
}

struct SessionState {
    directory: PathBuf,
    output_path: PathBuf,
    journal: RoomRecoveryJournal,
    process_start_evaluations: usize,
    already_committed: bool,
}

/// A locked, single-writer recovery session shared by RoomPipeline and the CLI.
#[derive(Clone)]
pub struct RoomRecoverySession {
    state: Arc<Mutex<SessionState>>,
    // Holding this file keeps the OS lock for the entire optimization and
    // publication lifetime. The kernel releases it if the process is killed.
    _lock: Arc<File>,
}

impl RoomRecoverySession {
    /// Open or create an exact RoomEQ recovery session.
    ///
    /// This lane accepts one generic single-measurement channel, LowLatency
    /// processing, seeded AutoEQ DE, and a positive PEQ count. It hashes the
    /// resolved config and all declared measurement/target/calibration/IR files
    /// before allowing the pipeline to start.
    ///
    /// # Errors
    /// Returns an error for unsupported configurations, changed input/output
    /// identity, malformed recovery state, concurrent writers, or failed
    /// filesystem operations.
    pub fn open(request: RoomRecoveryOpen<'_>) -> Result<Self, String> {
        let mut identity_config = request.room_config.clone();
        identity_config.resolve_room_dimensions();
        validate_supported_config(&identity_config, request.sample_rate_hz)?;
        if request.frequency_samples == 0 {
            return Err("RoomEQ recovery requires a positive frequency sample count".into());
        }
        let (config_sha256, input_sha256_by_path, measurement_curve_sha256) =
            config_and_input_identity(&identity_config, true)?;
        let run_identity = make_run_identity(
            &identity_config,
            request.sample_rate_hz,
            request.frequency_samples,
            request.output_path,
            &config_sha256,
            &input_sha256_by_path,
            &measurement_curve_sha256,
        )?;

        fs::create_dir_all(request.directory).map_err(|error| {
            format!(
                "cannot create RoomEQ recovery directory {}: {error}",
                request.directory.display()
            )
        })?;
        let directory_metadata = fs::symlink_metadata(request.directory).map_err(|error| {
            format!(
                "cannot inspect RoomEQ recovery directory {}: {error}",
                request.directory.display()
            )
        })?;
        if directory_metadata.file_type().is_symlink() || !directory_metadata.is_dir() {
            return Err("RoomEQ recovery path must be a regular directory, not a symlink".into());
        }
        let lock_path = request.directory.join(ROOM_RECOVERY_LOCK_FILE_NAME);
        if fs::symlink_metadata(&lock_path).is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err("RoomEQ recovery lock path cannot be a symbolic link".into());
        }
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|error| {
                format!("cannot open recovery lock {}: {error}", lock_path.display())
            })?;
        lock.try_lock().map_err(|error| {
            format!(
                "RoomEQ recovery directory is already active ({}): {error}",
                request.directory.display()
            )
        })?;

        crate::output_bundle::recover_output_bundle_transactions(request.output_path)
            .map_err(|error| format!("cannot reconcile RoomEQ output transactions: {error}"))?;

        let journal_path = request.directory.join(ROOM_RECOVERY_FILE_NAME);
        let loaded = read_journal(&journal_path)?;
        let mut already_committed = false;
        let mut process_start_evaluations = 0;
        let journal = match loaded {
            None if request.resume => {
                return Err(format!(
                    "cannot resume missing RoomEQ recovery file {}",
                    journal_path.display()
                ));
            }
            None => {
                let previous_output = capture_output_identity(request.output_path, false)?;
                let journal = RoomRecoveryJournal {
                    schema_version: ROOM_RECOVERY_SCHEMA_VERSION,
                    run_identity: run_identity.clone(),
                    channel_id: only_channel_name(&identity_config)?.to_owned(),
                    config_sha256,
                    input_sha256_by_path,
                    measurement_curve_sha256,
                    sample_rate_bits: format!("{:016x}", request.sample_rate_hz.to_bits()),
                    frequency_samples: request.frequency_samples,
                    status: RoomRecoveryStatus::Prepared,
                    attempt: 1,
                    recovery_count: 0,
                    last_resume_from: None,
                    logical_search_evaluations: 0,
                    checkpointed_search_evaluations_this_process: 0,
                    checkpoint: None,
                    checkpoint_run_identity: None,
                    stage_candidate: None,
                    previous_output,
                    output_intent: None,
                    failure_reason: None,
                };
                validate_journal(&journal)?;
                write_journal(&journal_path, &journal).map_err(|error| error.to_string())?;
                journal
            }
            Some(mut journal) => {
                validate_journal(&journal)?;
                if journal.run_identity != run_identity
                    || journal.config_sha256 != config_sha256
                    || journal.input_sha256_by_path != input_sha256_by_path
                    || journal.measurement_curve_sha256 != measurement_curve_sha256
                    || journal.sample_rate_bits
                        != format!("{:016x}", request.sample_rate_hz.to_bits())
                    || journal.frequency_samples != request.frequency_samples
                    || journal.channel_id != only_channel_name(&identity_config)?
                {
                    return Err("RoomEQ recovery input/config/output identity mismatch".into());
                }
                if !request.resume {
                    return Err(
                        "RoomEQ recovery state already exists; pass --resume-recovery".into(),
                    );
                }
                if matches!(
                    journal.status,
                    RoomRecoveryStatus::Cancelled | RoomRecoveryStatus::Failed
                ) {
                    return Err(format!(
                        "RoomEQ recovery is terminal ({:?}); cancelled or failed runs cannot be resumed",
                        journal.status
                    ));
                }

                let current_output = capture_output_identity(request.output_path, false)?;
                if matches!(
                    journal.status,
                    RoomRecoveryStatus::Publishing | RoomRecoveryStatus::Committed
                ) {
                    let intent = journal
                        .output_intent
                        .as_ref()
                        .ok_or("publishing recovery state is missing its output intent")?;
                    if current_output.as_ref() == Some(&intent.intended) {
                        let validated = capture_output_identity(request.output_path, true)?;
                        if validated.as_ref() != Some(&intent.intended) {
                            return Err(
                                "published RoomEQ bundle no longer matches its manifested resources"
                                    .into(),
                            );
                        }
                        journal.status = RoomRecoveryStatus::Committed;
                        journal.failure_reason = None;
                        already_committed = true;
                    } else if journal.status == RoomRecoveryStatus::Committed {
                        return Err(
                            "committed RoomEQ output no longer matches its recorded identity"
                                .into(),
                        );
                    } else if current_output == intent.previous {
                        journal.status = RoomRecoveryStatus::StageComplete;
                        journal.output_intent = None;
                    } else {
                        return Err("RoomEQ output matches neither the prior nor intended recovery identity".into());
                    }
                } else if current_output != journal.previous_output {
                    return Err(
                        "canonical RoomEQ output changed since this recovery run began".into(),
                    );
                }

                if journal.status == RoomRecoveryStatus::StageComplete {
                    validate_stage_candidate(&journal, request.output_path)?;
                }

                if already_committed {
                    write_journal(&journal_path, &journal).map_err(|error| error.to_string())?;
                } else {
                    let interrupted = journal.status == RoomRecoveryStatus::Running;
                    if interrupted {
                        journal.status = RoomRecoveryStatus::Interrupted;
                        write_journal(&journal_path, &journal)
                            .map_err(|error| error.to_string())?;
                    }
                    journal.last_resume_from = Some(journal.status);
                    if journal.status != RoomRecoveryStatus::StageComplete {
                        journal.status = RoomRecoveryStatus::Running;
                    }
                    journal.attempt = journal.attempt.saturating_add(1);
                    if interrupted
                        || journal.last_resume_from == Some(RoomRecoveryStatus::Interrupted)
                    {
                        journal.recovery_count = journal.recovery_count.saturating_add(1);
                    }
                    journal.checkpointed_search_evaluations_this_process = 0;
                    process_start_evaluations = journal
                        .checkpoint
                        .as_ref()
                        .map(|state| state.checkpoint().evaluations)
                        .unwrap_or_default();
                    write_journal(&journal_path, &journal).map_err(|error| error.to_string())?;
                }
                journal
            }
        };

        Ok(Self {
            state: Arc::new(Mutex::new(SessionState {
                directory: request.directory.to_path_buf(),
                output_path: request.output_path.to_path_buf(),
                journal,
                process_start_evaluations,
                already_committed,
            })),
            _lock: Arc::new(lock),
        })
    }

    /// Whether recovery found that the intended publication had already completed.
    #[must_use]
    pub fn already_committed(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.already_committed)
            .unwrap_or(false)
    }

    /// Current persisted recovery status.
    ///
    /// # Errors
    /// Returns an error if the session mutex was poisoned.
    pub fn status(&self) -> Result<RoomRecoveryStatus, String> {
        self.state
            .lock()
            .map(|state| state.journal.status)
            .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())
    }

    /// Cumulative exact DE search evaluations, including those from prior processes.
    ///
    /// # Errors
    /// Returns an error if the session mutex was poisoned.
    pub fn logical_search_evaluations(&self) -> Result<usize, String> {
        self.state
            .lock()
            .map(|state| state.journal.logical_search_evaluations)
            .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())
    }

    /// Barrier-confirmed DE search evaluations completed in this process.
    ///
    /// A killed process may have admitted work after its last generation
    /// barrier; that uncheckpointed work is intentionally not guessed.
    ///
    /// # Errors
    /// Returns an error if the session mutex was poisoned.
    pub fn checkpointed_search_evaluations_this_process(&self) -> Result<usize, String> {
        self.state
            .lock()
            .map(|state| state.journal.checkpointed_search_evaluations_this_process)
            .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())
    }

    /// Verify that a loaded configuration still identifies this session.
    ///
    /// # Errors
    /// Returns an error when config values or any declared input bytes changed.
    pub fn verify_configuration(
        &self,
        room_config: &RoomConfig,
        sample_rate_hz: f64,
        frequency_samples: usize,
    ) -> Result<(), String> {
        let mut room_config = room_config.clone();
        room_config.resolve_room_dimensions();
        validate_supported_config(&room_config, sample_rate_hz)?;
        if frequency_samples == 0 {
            return Err("RoomEQ recovery requires a positive frequency sample count".into());
        }
        let (config_sha256, input_sha256_by_path, measurement_curve_sha256) =
            config_and_input_identity(&room_config, false)?;
        let state = self
            .state
            .lock()
            .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())?;
        let run_identity = make_run_identity(
            &room_config,
            sample_rate_hz,
            frequency_samples,
            &state.output_path,
            &config_sha256,
            &input_sha256_by_path,
            &measurement_curve_sha256,
        )?;
        if run_identity != state.journal.run_identity
            || config_sha256 != state.journal.config_sha256
            || input_sha256_by_path != state.journal.input_sha256_by_path
            || measurement_curve_sha256 != state.journal.measurement_curve_sha256
            || state.journal.sample_rate_bits != format!("{:016x}", sample_rate_hz.to_bits())
            || state.journal.frequency_samples != frequency_samples
            || state.journal.channel_id != only_channel_name(&room_config)?
        {
            return Err("RoomEQ recovery input/config/output identity changed".into());
        }
        Ok(())
    }

    /// Freeze and verify the one numerical measurement used by this recovery run.
    ///
    /// The returned snapshot is the exact object the caller must pass to channel
    /// preparation. This prevents a preloaded curve from changing between the
    /// identity check and optimizer dispatch.
    ///
    /// # Errors
    /// Returns an error if loading fails or the loaded numerical curve differs
    /// from the curve bound at session creation.
    pub fn freeze_and_verify_measurement_source(
        &self,
        source: &MeasurementSource,
    ) -> Result<MeasurementSource, String> {
        let snapshot = autoeq_measurements::read::snapshot_source(source)
            .map_err(|error| format!("cannot freeze exact-recovery measurement: {error}"))?;
        let MeasurementSource::Single(single) = &snapshot else {
            return Err("exact RoomEQ recovery requires one frozen measurement source".into());
        };
        let MeasurementRef::Loaded {
            loaded_response, ..
        } = &single.measurement
        else {
            return Err("exact RoomEQ recovery measurement was not frozen before scoring".into());
        };
        let actual = curve_sha256(loaded_response)?;
        let state = self
            .state
            .lock()
            .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())?;
        if actual != state.journal.measurement_curve_sha256 {
            return Err("prepared RoomEQ measurement curve identity changed".into());
        }
        Ok(snapshot)
    }

    /// Return whether a finalized candidate is waiting for publication-only recovery.
    #[must_use]
    pub fn has_completed_stage(&self) -> bool {
        self.state
            .lock()
            .is_ok_and(|state| state.journal.status == RoomRecoveryStatus::StageComplete)
    }

    /// Prepare this run's durable candidate path before pipeline execution.
    ///
    /// The candidate directory is beside canonical output so A11 can publish
    /// on the same filesystem. An incomplete candidate for this locked run is
    /// removed before recomputation; a completed candidate must be reused.
    ///
    /// # Errors
    /// Returns an error for terminal/completed states or unsafe filesystem paths.
    pub fn prepare_stage_output(&self) -> Result<PathBuf, String> {
        let state = self
            .state
            .lock()
            .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())?;
        if state.journal.status == RoomRecoveryStatus::StageComplete {
            return Err("completed RoomEQ stage must be reused, not recomputed".into());
        }
        if matches!(
            state.journal.status,
            RoomRecoveryStatus::Cancelled
                | RoomRecoveryStatus::Failed
                | RoomRecoveryStatus::Committed
                | RoomRecoveryStatus::Publishing
        ) {
            return Err(format!(
                "cannot prepare RoomEQ stage output from {:?}",
                state.journal.status
            ));
        }
        let (directory, output) = stage_paths(&state.output_path, &state.journal.run_identity)?;
        match fs::symlink_metadata(&directory) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err("RoomEQ recovery stage path is not a regular directory".into());
            }
            Ok(_) => fs::remove_dir_all(&directory)
                .map_err(|error| format!("cannot clear incomplete RoomEQ stage: {error}"))?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("cannot inspect RoomEQ stage path: {error}")),
        }
        fs::create_dir_all(&directory)
            .map_err(|error| format!("cannot create RoomEQ recovery stage: {error}"))?;
        Ok(output)
    }

    /// Return the validated durable candidate recorded by a completed stage.
    ///
    /// # Errors
    /// Returns an error if no completed stage exists or its files changed.
    pub fn completed_stage_output(&self) -> Result<PathBuf, String> {
        let state = self
            .state
            .lock()
            .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())?;
        if state.journal.status != RoomRecoveryStatus::StageComplete {
            return Err("RoomEQ recovery has no completed candidate stage".into());
        }
        let candidate = state
            .journal
            .stage_candidate
            .as_ref()
            .ok_or("RoomEQ completed stage is missing its candidate identity")?;
        validate_stage_candidate(&state.journal, &state.output_path)?;
        Ok(candidate.path.clone())
    }

    /// Build the exact-DE options used by the supported single-channel pass.
    ///
    /// The callback atomically stores both the checkpoint and its typed
    /// workflow disposition. A successful callback return is the caller's
    /// durability acknowledgement.
    ///
    /// # Errors
    /// Returns an error for a terminal session, invalid checkpoint identity,
    /// or a poisoned recovery state.
    pub fn exact_de_options(&self) -> Result<ExactDERecoveryOptions, String> {
        let state = self
            .state
            .lock()
            .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())?;
        if matches!(
            state.journal.status,
            RoomRecoveryStatus::Cancelled
                | RoomRecoveryStatus::Failed
                | RoomRecoveryStatus::Committed
        ) {
            return Err(format!(
                "cannot start exact DE from terminal RoomEQ recovery state {:?}",
                state.journal.status
            ));
        }
        let checkpoint = state.journal.checkpoint.clone();
        if checkpoint
            .as_ref()
            .is_some_and(|saved| is_callback_cancelled_checkpoint(saved))
        {
            return Err("RoomEQ exact checkpoint records terminal observer cancellation".into());
        }
        let run_identity = state.journal.run_identity.clone();
        drop(state);
        let session = self.clone();
        ExactDERecoveryOptions::new(
            run_identity,
            checkpoint,
            Box::new(move |checkpoint| session.record_checkpoint(checkpoint)),
        )
    }

    /// Mark the complete RoomPipeline result as ready for native-bundle staging.
    ///
    /// # Errors
    /// Returns an error if no finalized successful exact checkpoint exists,
    /// if cancellation was recorded, or if persisting the state fails.
    pub fn mark_stage_complete(&self, staged_output_path: &Path) -> Result<(), String> {
        let expected_path = {
            let state = self
                .state
                .lock()
                .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())?;
            stage_paths(&state.output_path, &state.journal.run_identity)?.1
        };
        if staged_output_path != expected_path {
            return Err("completed RoomEQ candidate is outside its run-owned stage path".into());
        }
        let identity = capture_output_identity(staged_output_path, true)?
            .ok_or("completed RoomEQ stage has no native output bundle")?;
        self.update_journal(|journal| {
            if !allows_stage_complete(journal.status) {
                return Err(format!(
                    "cannot mark RoomEQ stage complete from {:?}",
                    journal.status
                ));
            }
            if journal.checkpoint.as_ref().is_none_or(|state| {
                state.checkpoint().terminal.as_ref().is_none_or(|terminal| {
                    !terminal.finalized || terminal.message == CALLBACK_STOP_MESSAGE
                })
            }) {
                return Err(
                    "RoomEQ stage completion requires a finalized, non-cancelled DE checkpoint"
                        .into(),
                );
            }
            let candidate = StageCandidate {
                path: staged_output_path.to_path_buf(),
                identity,
            };
            if journal.status == RoomRecoveryStatus::StageComplete
                && journal.stage_candidate.as_ref() != Some(&candidate)
            {
                return Err("completed RoomEQ stage identity changed".into());
            }
            journal.stage_candidate = Some(candidate);
            journal.status = RoomRecoveryStatus::StageComplete;
            journal.failure_reason = None;
            Ok(())
        })
    }

    /// Publish the validated staged candidate only while the canonical output
    /// still matches the prior identity captured when this recovery run opened.
    ///
    /// A11 repeats that comparison while holding its canonical-path lock through
    /// the native-bundle transaction.
    ///
    /// # Errors
    /// Returns an error if the candidate, publication intent, or prior output
    /// identity changed, or if the atomic bundle transaction fails.
    pub fn publish_candidate_output(&self, staged_output_path: &Path) -> Result<(), String> {
        self.mark_publishing(staged_output_path)?;
        let (output_path, expected_previous, intended) = {
            let state = self
                .state
                .lock()
                .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())?;
            let intent = state
                .journal
                .output_intent
                .as_ref()
                .ok_or("RoomEQ recovery publication intent was not recorded")?;
            (
                state.output_path.clone(),
                intent.previous.clone(),
                intent.intended.clone(),
            )
        };
        if capture_output_identity(&output_path, false)?.as_ref() == Some(&intended) {
            return Ok(());
        }
        crate::output_bundle::publish_output_bundle_from_if_previous(
            staged_output_path,
            &output_path,
            expected_previous,
        )
        .map_err(|error| format!("cannot publish recovered RoomEQ bundle: {error:#}"))
    }

    /// Record an ordinary workflow failure while preserving terminal cancellation.
    ///
    /// # Errors
    /// Returns an error when the journal cannot be atomically updated.
    pub fn mark_failed(&self, reason: &str) -> Result<(), String> {
        let bounded_reason = reason
            .chars()
            .take(MAX_RECOVERY_ERROR_CHARS)
            .collect::<String>();
        self.update_journal(|journal| {
            if preserves_failure_status(journal.status) {
                // Preserve terminal or reconcilable publication state.
                journal.failure_reason = Some(bounded_reason);
            } else {
                journal.status = RoomRecoveryStatus::Failed;
                journal.failure_reason = Some(bounded_reason);
            }
            Ok(())
        })
    }

    /// Record a terminal CLI/user cancellation without publishing a candidate.
    ///
    /// # Errors
    /// Returns an error when the journal cannot be atomically updated.
    pub fn mark_cancelled(&self, reason: &str) -> Result<(), String> {
        let bounded_reason = reason
            .chars()
            .take(MAX_RECOVERY_ERROR_CHARS)
            .collect::<String>();
        self.update_journal(|journal| {
            match journal.status {
                RoomRecoveryStatus::Committed
                | RoomRecoveryStatus::Publishing
                | RoomRecoveryStatus::Cancelled
                | RoomRecoveryStatus::Failed => {
                    journal.failure_reason = Some(bounded_reason);
                }
                _ => {
                    journal.status = RoomRecoveryStatus::Cancelled;
                    journal.failure_reason = Some(bounded_reason);
                }
            }
            Ok(())
        })
    }

    /// Persist the exact graph and manifest identities before native publication.
    ///
    /// # Errors
    /// Returns an error unless RoomPipeline is complete and the staged A11 bundle
    /// has both a graph and integrity manifest.
    pub fn mark_publishing(&self, staged_output_path: &Path) -> Result<(), String> {
        let staged = capture_output_identity(staged_output_path, true)?
            .ok_or("staged RoomEQ bundle is missing its native graph")?;
        let (output_path, expected_previous, current_intent) = {
            let state = self
                .state
                .lock()
                .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())?;
            (
                state.output_path.clone(),
                state.journal.previous_output.clone(),
                state.journal.output_intent.clone(),
            )
        };
        let current = capture_output_identity(&output_path, false)?;
        if current != expected_previous
            && !current_intent
                .as_ref()
                .is_some_and(|intent| current.as_ref() == Some(&intent.intended))
        {
            return Err("canonical RoomEQ output changed since this recovery run began".into());
        }
        self.update_journal(|journal| {
            if !allows_mark_publishing(journal.status) {
                return Err(format!(
                    "cannot publish RoomEQ recovery state {:?}; expected StageComplete or Publishing",
                    journal.status
                ));
            }
            let candidate = journal
                .stage_candidate
                .as_ref()
                .ok_or("RoomEQ publication has no durable completed stage")?;
            if candidate.path != staged_output_path || candidate.identity != staged {
                return Err("RoomEQ staged candidate changed before publication".into());
            }
            let intent = OutputIntent {
                previous: journal.previous_output.clone(),
                intended: staged,
            };
            if journal.status == RoomRecoveryStatus::Publishing
                && journal.output_intent.as_ref() != Some(&intent)
            {
                return Err("RoomEQ publication intent changed during retry".into());
            }
            journal.output_intent = Some(intent);
            journal.status = RoomRecoveryStatus::Publishing;
            Ok(())
        })
    }

    /// Verify the canonical graph/manifest pair and mark publication committed.
    ///
    /// # Errors
    /// Returns an error if output bytes differ from the durably recorded intent.
    pub fn mark_committed(&self) -> Result<(), String> {
        let current = {
            let state = self
                .state
                .lock()
                .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())?;
            capture_output_identity(&state.output_path, true)?
        };
        let current = current.ok_or("canonical RoomEQ bundle is missing")?;
        self.update_journal(|journal| {
            if !allows_mark_committed(journal.status) {
                return Err(format!(
                    "cannot mark RoomEQ publication committed from {:?}",
                    journal.status
                ));
            }
            let intent = journal
                .output_intent
                .as_ref()
                .ok_or("RoomEQ publication has no recorded output intent")?;
            if intent.intended != current {
                return Err(
                    "published RoomEQ graph or manifest differs from the staged intent".into(),
                );
            }
            journal.status = RoomRecoveryStatus::Committed;
            journal.failure_reason = None;
            Ok(())
        })?;
        if let Ok(mut state) = self.state.lock() {
            state.already_committed = true;
        }
        Ok(())
    }

    fn record_checkpoint(&self, saved_state: &ExactDERecoveryState) -> Result<(), String> {
        let checkpoint = saved_state.checkpoint();
        let mut session_state = self
            .state
            .lock()
            .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())?;
        if matches!(
            session_state.journal.status,
            RoomRecoveryStatus::Failed
                | RoomRecoveryStatus::Publishing
                | RoomRecoveryStatus::Committed
        ) {
            return Err(format!(
                "cannot write a DE checkpoint after RoomEQ recovery entered {:?}",
                session_state.journal.status
            ));
        }
        let incoming_cancelled = is_callback_cancelled_checkpoint(saved_state);
        if session_state.journal.status == RoomRecoveryStatus::Cancelled {
            let Some(previous) = session_state.journal.checkpoint.as_ref() else {
                return Err("RoomEQ recovery already recorded terminal cancellation".into());
            };
            if !is_callback_cancelled_checkpoint(previous) {
                return Err("RoomEQ recovery already recorded terminal cancellation".into());
            }
            let (Some(previous_terminal), Some(incoming_terminal)) =
                (&previous.checkpoint().terminal, &checkpoint.terminal)
            else {
                return Err("RoomEQ recovery already recorded terminal cancellation".into());
            };
            if !incoming_cancelled
                || checkpoint.evaluations != previous.checkpoint().evaluations
                || checkpoint.generation < previous.checkpoint().generation
                || (previous_terminal.finalized && !incoming_terminal.finalized)
                || incoming_terminal.polish_evaluations < previous_terminal.polish_evaluations
            {
                return Err("RoomEQ recovery already recorded terminal cancellation".into());
            }
        }
        let checkpoint_run_identity = saved_state.run_identity().to_owned();
        saved_state.check_compatible(&checkpoint_run_identity)?;
        let mut next = session_state.journal.clone();
        next.logical_search_evaluations = checkpoint.evaluations.saturating_add(
            checkpoint
                .terminal
                .as_ref()
                .map(|terminal| terminal.polish_evaluations)
                .unwrap_or_default(),
        );
        next.checkpointed_search_evaluations_this_process = checkpoint
            .evaluations
            .saturating_sub(session_state.process_start_evaluations);
        next.status = if checkpoint
            .terminal
            .as_ref()
            .is_some_and(|terminal| terminal.message == CALLBACK_STOP_MESSAGE)
        {
            RoomRecoveryStatus::Cancelled
        } else if checkpoint
            .terminal
            .as_ref()
            .is_some_and(|terminal| terminal.finalized)
        {
            RoomRecoveryStatus::OptimizerComplete
        } else {
            RoomRecoveryStatus::Running
        };
        next.failure_reason = None;
        next.checkpoint = Some(saved_state.clone());
        next.checkpoint_run_identity = Some(checkpoint_run_identity);
        validate_journal(&next)?;
        let journal_path = session_state.directory.join(ROOM_RECOVERY_FILE_NAME);
        write_journal(&journal_path, &next)
            .map_err(|error| format!("cannot durably store RoomEQ checkpoint/journal: {error}"))?;
        session_state.journal = next;
        Ok(())
    }

    fn update_journal(
        &self,
        update: impl FnOnce(&mut RoomRecoveryJournal) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "RoomEQ recovery session state was poisoned".to_owned())?;
        let mut next = state.journal.clone();
        update(&mut next)?;
        validate_journal(&next)?;
        let path = state.directory.join(ROOM_RECOVERY_FILE_NAME);
        write_journal(&path, &next)
            .map_err(|error| format!("cannot atomically update RoomEQ recovery state: {error}"))?;
        state.journal = next;
        Ok(())
    }
}

fn validate_supported_config(config: &RoomConfig, sample_rate_hz: f64) -> Result<(), String> {
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err("RoomEQ recovery sample rate must be finite and positive".into());
    }
    if config.speakers.len() != 1 {
        return Err("RoomEQ exact recovery supports exactly one generic channel".into());
    }
    if config.system.is_some() {
        return Err("RoomEQ exact recovery does not support a system topology".into());
    }
    let speaker = config
        .speakers
        .values()
        .next()
        .ok_or("RoomEQ exact recovery requires one configured speaker")?;
    let SpeakerConfig::Single(source) = speaker else {
        return Err("RoomEQ exact recovery supports only a generic single-channel source".into());
    };
    if !matches!(source, MeasurementSource::Single(_)) {
        return Err("RoomEQ exact recovery supports exactly one measurement curve".into());
    }
    if config.optimizer.processing_mode != ProcessingMode::LowLatency {
        return Err("RoomEQ exact recovery requires LowLatency processing".into());
    }
    if !config.optimizer.algorithm.eq_ignore_ascii_case("autoeq:de") {
        return Err("RoomEQ exact recovery supports only the autoeq:de backend".into());
    }
    if config.optimizer.seed.is_none() {
        return Err("RoomEQ exact recovery requires an explicit optimizer seed".into());
    }
    if config.optimizer.num_filters == 0 {
        return Err("RoomEQ exact recovery requires at least one PEQ filter".into());
    }
    if config.optimizer.refine {
        return Err("RoomEQ exact recovery does not support local refinement".into());
    }
    if config.optimizer.min_filter_improvement > 0.0 && config.optimizer.num_filters > 1 {
        return Err("RoomEQ exact recovery does not support adaptive filter selection".into());
    }
    if config.optimizer.multi_measurement.is_some() {
        return Err("RoomEQ exact recovery does not support multi-measurement objectives".into());
    }
    if config
        .optimizer
        .schroeder_split
        .as_ref()
        .is_some_and(|split| split.enabled)
    {
        return Err("RoomEQ exact recovery does not support a Schroeder split".into());
    }
    Ok(())
}

fn only_channel_name(config: &RoomConfig) -> Result<&str, String> {
    config
        .speakers
        .keys()
        .next()
        .map(String::as_str)
        .ok_or_else(|| "RoomEQ exact recovery requires one configured speaker".into())
}

fn config_and_input_identity(
    config: &RoomConfig,
    reject_preloaded: bool,
) -> Result<(String, BTreeMap<String, String>, String), String> {
    // Workflow input snapshots wrap the declared source reference with a
    // parsed response. Hash the exact loaded curve as well as the original
    // declaration so Path -> Loaded remains stable without trusting a caller's
    // preloaded numerical response.
    let measurement_curve_sha256 = configured_measurement_curve_sha256(config, reject_preloaded)?;
    let mut identity_config = config.clone();
    if let Some(SpeakerConfig::Single(MeasurementSource::Single(single))) =
        identity_config.speakers.values_mut().next()
    {
        single.measurement = single.measurement.original().clone();
    }
    let config_value = serde_json::to_value(&identity_config)
        .map_err(|error| format!("cannot serialize exact RoomEQ config identity: {error}"))?;
    let config_bytes = serde_json::to_vec(&config_value)
        .map_err(|error| format!("cannot encode exact RoomEQ config identity: {error}"))?;
    let config_sha256 = sha256_bytes(&config_bytes);
    let mut inputs = BTreeMap::new();
    let speaker = identity_config
        .speakers
        .values()
        .next()
        .ok_or("RoomEQ exact recovery requires one configured speaker")?;
    let SpeakerConfig::Single(MeasurementSource::Single(single)) = speaker else {
        return Err("RoomEQ exact recovery requires one serializable single measurement".into());
    };
    add_measurement_reference_files(&single.measurement, &mut inputs)?;
    if let MeasurementRef::Inline(inline) = single.measurement.original() {
        if let Some(path) = &inline.csv_path {
            add_file(Path::new(path), &mut inputs)?;
        }
        if let Some(path) = &inline.wav_path {
            add_file(Path::new(path), &mut inputs)?;
        }
    }
    if let Some(TargetCurveConfig::Path(path)) = &config.target_curve {
        add_file(Path::new(path), &mut inputs)?;
    }
    for source in config.measured_impulse_responses.values() {
        add_file(&source.path, &mut inputs)?;
    }
    if let Some(recording) = &config.recording_config {
        if let Some(path) = &recording.mic_calibration_path {
            add_file(Path::new(path), &mut inputs)?;
        }
        for path in recording.mic_calibration_paths.iter().flatten().flatten() {
            add_file(Path::new(path), &mut inputs)?;
        }
    }
    Ok((config_sha256, inputs, measurement_curve_sha256))
}

fn configured_measurement_curve_sha256(
    config: &RoomConfig,
    reject_preloaded: bool,
) -> Result<String, String> {
    let source = config
        .speakers
        .values()
        .next()
        .ok_or("RoomEQ exact recovery requires one configured speaker")?;
    let SpeakerConfig::Single(MeasurementSource::Single(single)) = source else {
        return Err("RoomEQ exact recovery requires one serializable single measurement".into());
    };
    let curve = match &single.measurement {
        MeasurementRef::Loaded {
            loaded_response, ..
        } if reject_preloaded => {
            let _ = loaded_response;
            return Err(
                "exact RoomEQ recovery cannot start from a caller-preloaded measurement curve"
                    .into(),
            );
        }
        MeasurementRef::Loaded {
            loaded_response, ..
        } => loaded_response.as_ref().clone(),
        reference => autoeq_measurements::read::load_measurement_strict(reference)
            .map_err(|error| format!("cannot load exact-recovery measurement curve: {error}"))?,
    };
    curve_sha256(&curve)
}

fn curve_sha256(curve: &Curve) -> Result<String, String> {
    curve
        .content_hash()
        .map_err(|error| format!("cannot hash exact-recovery measurement curve: {error}"))
}

fn add_measurement_reference_files(
    measurement: &MeasurementRef,
    inputs: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    if let Some(path) = measurement.path() {
        add_file(path, inputs)?;
    }
    Ok(())
}

fn add_file(path: &Path, inputs: &mut BTreeMap<String, String>) -> Result<(), String> {
    let name = path.to_string_lossy().into_owned();
    if !inputs.contains_key(&name) {
        inputs.insert(name.clone(), sha256_file(path)?);
    }
    Ok(())
}

fn make_run_identity(
    config: &RoomConfig,
    sample_rate_hz: f64,
    frequency_samples: usize,
    output_path: &Path,
    config_sha256: &str,
    input_sha256_by_path: &BTreeMap<String, String>,
    measurement_curve_sha256: &str,
) -> Result<String, String> {
    let identity = serde_json::json!({
        "schema": "roomeq-exact-de-run-v1",
        "channel_id": only_channel_name(config)?,
        "config_sha256": config_sha256,
        "input_sha256_by_path": input_sha256_by_path,
        "measurement_curve_sha256": measurement_curve_sha256,
        "sample_rate_bits": format!("{:016x}", sample_rate_hz.to_bits()),
        "frequency_samples": frequency_samples,
        "output_path": output_path.to_string_lossy(),
    });
    let bytes = serde_json::to_vec(&identity)
        .map_err(|error| format!("cannot encode exact RoomEQ run identity: {error}"))?;
    Ok(sha256_bytes(&bytes))
}

fn stage_paths(output_path: &Path, run_identity: &str) -> Result<(PathBuf, PathBuf), String> {
    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = output_path
        .file_name()
        .ok_or("canonical RoomEQ output path must include a file name")?;
    let digest = run_identity
        .strip_prefix("sha256:")
        .filter(|digest| digest.len() >= 16)
        .ok_or("RoomEQ recovery run identity has an invalid SHA-256 encoding")?;
    let directory = parent.join(format!(
        ".{}.recovery-{}",
        file_name.to_string_lossy(),
        &digest[..16]
    ));
    let candidate = directory.join(file_name);
    Ok((directory, candidate))
}

fn validate_stage_candidate(
    journal: &RoomRecoveryJournal,
    canonical_output_path: &Path,
) -> Result<(), String> {
    let candidate = journal
        .stage_candidate
        .as_ref()
        .ok_or("completed RoomEQ stage is missing its candidate identity")?;
    let expected = stage_paths(canonical_output_path, &journal.run_identity)?.1;
    if candidate.path != expected {
        return Err("RoomEQ staged candidate path does not match its run identity".into());
    }
    let actual = capture_output_identity(&candidate.path, true)?
        .ok_or("RoomEQ staged candidate bundle is missing")?;
    if actual != candidate.identity {
        return Err("RoomEQ staged candidate graph or manifested resources changed".into());
    }
    Ok(())
}

fn allows_stage_complete(status: RoomRecoveryStatus) -> bool {
    matches!(
        status,
        RoomRecoveryStatus::Running
            | RoomRecoveryStatus::OptimizerComplete
            | RoomRecoveryStatus::StageComplete
    )
}

fn allows_mark_publishing(status: RoomRecoveryStatus) -> bool {
    matches!(
        status,
        RoomRecoveryStatus::StageComplete | RoomRecoveryStatus::Publishing
    )
}

fn allows_mark_committed(status: RoomRecoveryStatus) -> bool {
    matches!(
        status,
        RoomRecoveryStatus::Publishing | RoomRecoveryStatus::Committed
    )
}

fn preserves_failure_status(status: RoomRecoveryStatus) -> bool {
    matches!(
        status,
        RoomRecoveryStatus::StageComplete
            | RoomRecoveryStatus::Publishing
            | RoomRecoveryStatus::Committed
            | RoomRecoveryStatus::Cancelled
            | RoomRecoveryStatus::Failed
    )
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("sha256:{}", hex_bytes(digest.as_slice()))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("cannot read recovery input {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot hash recovery input {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!(
        "sha256:{}",
        hex_bytes(hasher.finalize().as_slice())
    ))
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn is_callback_cancelled_checkpoint(state: &ExactDERecoveryState) -> bool {
    state
        .checkpoint()
        .terminal
        .as_ref()
        .is_some_and(|terminal| terminal.message == CALLBACK_STOP_MESSAGE)
}

fn validate_journal(journal: &RoomRecoveryJournal) -> Result<(), String> {
    if journal.schema_version != ROOM_RECOVERY_SCHEMA_VERSION {
        return Err(format!(
            "unsupported RoomEQ recovery schema {}; expected {ROOM_RECOVERY_SCHEMA_VERSION}",
            journal.schema_version
        ));
    }
    if journal.run_identity.trim().is_empty() || journal.channel_id.trim().is_empty() {
        return Err("RoomEQ recovery journal is missing run or channel identity".into());
    }
    if let Some(state) = &journal.checkpoint {
        let checkpoint_run_identity = journal
            .checkpoint_run_identity
            .as_deref()
            .ok_or("RoomEQ recovery checkpoint is missing its prepared-objective identity")?;
        state.check_compatible(checkpoint_run_identity)?;
        let logical = state.checkpoint().evaluations.saturating_add(
            state
                .checkpoint()
                .terminal
                .as_ref()
                .map(|terminal| terminal.polish_evaluations)
                .unwrap_or_default(),
        );
        if journal.logical_search_evaluations != logical {
            return Err(
                "RoomEQ recovery logical evaluation count disagrees with DE checkpoint".into(),
            );
        }
        let callback_cancelled = is_callback_cancelled_checkpoint(state);
        if callback_cancelled && journal.status != RoomRecoveryStatus::Cancelled {
            return Err(
                "RoomEQ recovery status disagrees with checkpoint terminal disposition".into(),
            );
        }
    } else {
        if journal.logical_search_evaluations != 0 {
            return Err(
                "RoomEQ recovery journal reports evaluations without a DE checkpoint".into(),
            );
        }
        if journal.checkpoint_run_identity.is_some() {
            return Err(
                "RoomEQ recovery journal has checkpoint identity without a checkpoint".into(),
            );
        }
    }
    if matches!(
        journal.status,
        RoomRecoveryStatus::OptimizerComplete
            | RoomRecoveryStatus::StageComplete
            | RoomRecoveryStatus::Publishing
            | RoomRecoveryStatus::Committed
    ) && journal.checkpoint.as_ref().is_none_or(|state| {
        state
            .checkpoint()
            .terminal
            .as_ref()
            .is_none_or(|terminal| !terminal.finalized || terminal.message == CALLBACK_STOP_MESSAGE)
    }) {
        return Err(format!(
            "RoomEQ recovery status {:?} requires a finalized successful DE checkpoint",
            journal.status
        ));
    }
    let publication_state = matches!(
        journal.status,
        RoomRecoveryStatus::Publishing | RoomRecoveryStatus::Committed
    );
    if publication_state != journal.output_intent.is_some() {
        return Err(format!(
            "RoomEQ recovery status {:?} has inconsistent publication intent",
            journal.status
        ));
    }
    let completed_stage = matches!(
        journal.status,
        RoomRecoveryStatus::StageComplete
            | RoomRecoveryStatus::Publishing
            | RoomRecoveryStatus::Committed
    );
    if completed_stage != journal.stage_candidate.is_some()
        && !(journal.status == RoomRecoveryStatus::Cancelled && journal.stage_candidate.is_some())
    {
        return Err(format!(
            "RoomEQ recovery status {:?} has inconsistent staged-candidate identity",
            journal.status
        ));
    }
    if let (Some(candidate), Some(intent)) = (&journal.stage_candidate, &journal.output_intent) {
        if candidate.identity != intent.intended {
            return Err("RoomEQ output intent disagrees with completed stage identity".into());
        }
    }
    Ok(())
}

fn read_journal(path: &Path) -> Result<Option<RoomRecoveryJournal>, String> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot inspect recovery journal: {error}")),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("RoomEQ recovery journal must be a regular file".into());
    }
    if metadata.len() > MAX_ROOM_RECOVERY_JOURNAL_BYTES as u64 {
        return Err("RoomEQ recovery journal exceeds the 32 MiB size limit".into());
    }
    let file =
        File::open(path).map_err(|error| format!("cannot read recovery journal: {error}"))?;
    let mut bytes = Vec::with_capacity((metadata.len() as usize).min(64 * 1024));
    file.take(MAX_ROOM_RECOVERY_JOURNAL_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read recovery journal: {error}"))?;
    if bytes.len() as u64 != metadata.len() || bytes.len() > MAX_ROOM_RECOVERY_JOURNAL_BYTES {
        return Err("RoomEQ recovery journal changed size while being read".into());
    }
    let journal: RoomRecoveryJournal = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid RoomEQ recovery journal: {error}"))?;
    Ok(Some(journal))
}

fn write_journal(path: &Path, journal: &RoomRecoveryJournal) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = NamedTempFile::new_in(parent)?;
    let mut writer = BoundedWriter {
        file: temporary.as_file_mut(),
        written: 0,
        limit: MAX_ROOM_RECOVERY_JOURNAL_BYTES,
        exceeded: false,
    };
    serde_json::to_writer_pretty(&mut writer, journal).map_err(|error| {
        if writer.exceeded {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "RoomEQ recovery journal exceeds the 32 MiB size limit",
            )
        } else {
            io::Error::new(io::ErrorKind::InvalidData, error)
        }
    })?;
    writer.flush()?;
    drop(writer);
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    #[cfg(unix)]
    File::open(parent)?.sync_all()?;
    Ok(())
}

struct BoundedWriter<'a> {
    file: &'a mut File,
    written: usize,
    limit: usize,
    exceeded: bool,
}

impl Write for BoundedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.written.saturating_add(bytes.len()) > self.limit {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "serialized RoomEQ recovery journal exceeds its size limit",
            ));
        }
        let written = self.file.write(bytes)?;
        self.written += written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

fn capture_output_identity(
    path: &Path,
    require_manifest: bool,
) -> Result<Option<OutputIdentity>, String> {
    crate::output_bundle::capture_bundle_content_identity(path, require_manifest)
        .map_err(|error| format!("cannot identify RoomEQ output bundle: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::{OptimizerConfig, ProcessingMode};

    fn config() -> RoomConfig {
        RoomConfig {
            speakers: std::collections::HashMap::from([(
                "front_left".into(),
                SpeakerConfig::Single(MeasurementSource::InMemory(autoeq_core::Curve::default())),
            )]),
            optimizer: OptimizerConfig {
                processing_mode: ProcessingMode::LowLatency,
                algorithm: "autoeq:de".into(),
                seed: Some(42),
                num_filters: 1,
                max_iter: 128,
                population: 4,
                refine: false,
                min_filter_improvement: 0.0,
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        }
    }

    fn inline_config() -> RoomConfig {
        RoomConfig {
            speakers: std::collections::HashMap::from([(
                "front_left".into(),
                SpeakerConfig::Single(MeasurementSource::Single(autoeq_core::MeasurementSingle {
                    measurement: MeasurementRef::Inline(autoeq_core::InlineMeasurement {
                        frequencies: vec![40.0, 80.0, 160.0, 320.0, 640.0, 1_280.0],
                        magnitude_db: vec![76.0, 79.0, 81.0, 80.0, 78.0, 75.0],
                        phase_deg: None,
                        name: Some("recovery-inline".into()),
                        csv_path: None,
                        wav_path: None,
                    }),
                    speaker_name: None,
                    provenance: Default::default(),
                })),
            )]),
            optimizer: OptimizerConfig {
                processing_mode: ProcessingMode::LowLatency,
                algorithm: "autoeq:de".into(),
                seed: Some(42),
                num_filters: 1,
                max_iter: 128,
                population: 4,
                refine: false,
                min_filter_improvement: 0.0,
                ..OptimizerConfig::default()
            },
            ..RoomConfig::default()
        }
    }

    fn run_exact_de_for_session(
        session: &RoomRecoverySession,
        config: &RoomConfig,
    ) -> Result<roomeq_engine::eq::EqOptimizationResult, String> {
        let source = match config.speakers.get("front_left") {
            Some(SpeakerConfig::Single(source)) => source,
            _ => return Err("test config is missing its single source".into()),
        };
        let frozen = session.freeze_and_verify_measurement_source(source)?;
        let MeasurementSource::Single(single) = frozen else {
            return Err("frozen test source is not a single measurement".into());
        };
        let MeasurementRef::Loaded {
            loaded_response, ..
        } = single.measurement
        else {
            return Err("frozen test measurement is not loaded".into());
        };
        let curve = *loaded_response;
        let options = session.exact_de_options()?;
        roomeq_engine::eq::optimize_channel_eq_with_exact_de_checkpoint_detailed(
            &curve,
            &config.optimizer,
            None,
            48_000.0,
            None,
            options,
        )
        .map_err(|error| format!("exact RoomEQ engine optimization failed: {error}"))
    }

    #[test]
    fn exact_recovery_refuses_in_memory_source_before_creating_journal() {
        let directory = tempfile::tempdir().unwrap();
        let recovery_dir = directory.path().join("state");
        let result = RoomRecoverySession::open(RoomRecoveryOpen {
            directory: &recovery_dir,
            output_path: &directory.path().join("room.json"),
            room_config: &config(),
            sample_rate_hz: 48_000.0,
            frequency_samples: 512,
            resume: false,
        });
        assert!(result.is_err());
        assert!(!recovery_dir.join(ROOM_RECOVERY_FILE_NAME).exists());
    }

    #[test]
    fn exact_recovery_rejects_unsupported_backend_before_journal_creation() {
        let directory = tempfile::tempdir().unwrap();
        let recovery_dir = directory.path().join("state");
        let mut config = config();
        config.optimizer.algorithm = "autoeq:cmaes".into();
        let result = RoomRecoverySession::open(RoomRecoveryOpen {
            directory: &recovery_dir,
            output_path: &directory.path().join("room.json"),
            room_config: &config,
            sample_rate_hz: 48_000.0,
            frequency_samples: 512,
            resume: false,
        });
        assert!(result.is_err());
        assert!(!recovery_dir.exists());
    }

    #[test]
    fn loaded_source_snapshot_is_stable_but_tampered_curve_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let recovery_dir = directory.path().join("state");
        let output_path = directory.path().join("room.json");
        let config = inline_config();
        let session = RoomRecoverySession::open(RoomRecoveryOpen {
            directory: &recovery_dir,
            output_path: &output_path,
            room_config: &config,
            sample_rate_hz: 48_000.0,
            frequency_samples: 512,
            resume: false,
        })
        .unwrap();
        session
            .verify_configuration(&config, 48_000.0, 512)
            .expect("original inline configuration matches");

        let source = match config.speakers.get("front_left").unwrap() {
            SpeakerConfig::Single(source) => source,
            _ => unreachable!(),
        };
        let frozen = autoeq_measurements::read::snapshot_source(source).unwrap();
        session
            .freeze_and_verify_measurement_source(&frozen)
            .expect("the frozen response is the curve bound at open");

        let mut tampered = frozen;
        let MeasurementSource::Single(single) = &mut tampered else {
            unreachable!();
        };
        let MeasurementRef::Loaded {
            loaded_response, ..
        } = &mut single.measurement
        else {
            unreachable!();
        };
        loaded_response.spl[2] += 0.25;
        let error = session
            .freeze_and_verify_measurement_source(&tampered)
            .unwrap_err();
        assert!(error.contains("measurement curve identity changed"));
    }

    #[test]
    fn caller_preloaded_measurement_is_refused_before_journal_creation() {
        let directory = tempfile::tempdir().unwrap();
        let recovery_dir = directory.path().join("state");
        let mut config = inline_config();
        let SpeakerConfig::Single(MeasurementSource::Single(single)) =
            config.speakers.get_mut("front_left").unwrap()
        else {
            unreachable!();
        };
        let original = single.measurement.clone();
        let curve = autoeq_measurements::read::load_measurement_strict(&original).unwrap();
        single.measurement = MeasurementRef::Loaded {
            original: Box::new(original),
            loaded_response: Box::new(curve),
        };

        let result = RoomRecoverySession::open(RoomRecoveryOpen {
            directory: &recovery_dir,
            output_path: &directory.path().join("room.json"),
            room_config: &config,
            sample_rate_hz: 48_000.0,
            frequency_samples: 512,
            resume: false,
        });
        assert!(
            result
                .err()
                .expect("preloaded curve must be refused")
                .contains("caller-preloaded")
        );
        assert!(!recovery_dir.join(ROOM_RECOVERY_FILE_NAME).exists());
    }

    #[test]
    fn journal_rechecks_each_identity_field_independently_of_run_digest() {
        let directory = tempfile::tempdir().unwrap();
        let recovery_dir = directory.path().join("state");
        let output_path = directory.path().join("room.json");
        let config = inline_config();
        let session = RoomRecoverySession::open(RoomRecoveryOpen {
            directory: &recovery_dir,
            output_path: &output_path,
            room_config: &config,
            sample_rate_hz: 48_000.0,
            frequency_samples: 512,
            resume: false,
        })
        .unwrap();
        drop(session);

        let path = recovery_dir.join(ROOM_RECOVERY_FILE_NAME);
        let original: RoomRecoveryJournal = read_journal(&path).unwrap().unwrap();
        let tamperers: [fn(&mut RoomRecoveryJournal); 5] = [
            |journal: &mut RoomRecoveryJournal| journal.config_sha256.push('x'),
            |journal: &mut RoomRecoveryJournal| journal.measurement_curve_sha256.push('x'),
            |journal: &mut RoomRecoveryJournal| journal.sample_rate_bits.push('x'),
            |journal: &mut RoomRecoveryJournal| journal.frequency_samples += 1,
            |journal: &mut RoomRecoveryJournal| journal.channel_id.push('x'),
        ];
        for tamper in tamperers {
            let mut changed = original.clone();
            tamper(&mut changed);
            write_journal(&path, &changed).unwrap();
            let result = RoomRecoverySession::open(RoomRecoveryOpen {
                directory: &recovery_dir,
                output_path: &output_path,
                room_config: &config,
                sample_rate_hz: 48_000.0,
                frequency_samples: 512,
                resume: true,
            });
            assert!(result.is_err(), "tampered journal field was accepted");
        }
    }

    #[test]
    fn recovery_status_transition_helpers_fail_closed() {
        for status in [
            RoomRecoveryStatus::Running,
            RoomRecoveryStatus::OptimizerComplete,
            RoomRecoveryStatus::StageComplete,
        ] {
            assert!(allows_stage_complete(status));
        }
        for status in [
            RoomRecoveryStatus::Prepared,
            RoomRecoveryStatus::Interrupted,
            RoomRecoveryStatus::Publishing,
            RoomRecoveryStatus::Committed,
            RoomRecoveryStatus::Cancelled,
            RoomRecoveryStatus::Failed,
        ] {
            assert!(!allows_stage_complete(status));
        }
        assert!(allows_mark_publishing(RoomRecoveryStatus::StageComplete));
        assert!(allows_mark_publishing(RoomRecoveryStatus::Publishing));
        for status in [
            RoomRecoveryStatus::Prepared,
            RoomRecoveryStatus::Running,
            RoomRecoveryStatus::OptimizerComplete,
            RoomRecoveryStatus::Committed,
            RoomRecoveryStatus::Cancelled,
            RoomRecoveryStatus::Failed,
        ] {
            assert!(!allows_mark_publishing(status));
        }
        assert!(allows_mark_committed(RoomRecoveryStatus::Publishing));
        assert!(allows_mark_committed(RoomRecoveryStatus::Committed));
        for status in [
            RoomRecoveryStatus::Prepared,
            RoomRecoveryStatus::Running,
            RoomRecoveryStatus::OptimizerComplete,
            RoomRecoveryStatus::StageComplete,
            RoomRecoveryStatus::Cancelled,
            RoomRecoveryStatus::Failed,
        ] {
            assert!(!allows_mark_committed(status));
        }
        assert!(preserves_failure_status(RoomRecoveryStatus::StageComplete));
        assert!(preserves_failure_status(RoomRecoveryStatus::Publishing));
        assert!(preserves_failure_status(RoomRecoveryStatus::Committed));
        assert!(!preserves_failure_status(RoomRecoveryStatus::Running));
    }

    #[test]
    fn recovery_bundle_identity_ignores_only_a11_generation() {
        let directory = tempfile::tempdir().unwrap();
        let staged_path = directory.path().join("staged/room.json");
        let published_path = directory.path().join("published/room.json");
        let mut graph = roomeq_model::DspGraph::new("1");
        graph.add_channel("front_left", Vec::new());
        crate::output_bundle::save_output_bundle(&mut graph, &staged_path).unwrap();
        let staged_identity = capture_output_identity(&staged_path, true)
            .unwrap()
            .unwrap();
        let staged_manifest_path = crate::output_bundle::assets_dir_for(&staged_path)
            .join(crate::output_bundle::ARTIFACT_BUNDLE_MANIFEST_FILENAME);
        let staged_manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(staged_manifest_path).unwrap()).unwrap();

        crate::output_bundle::publish_output_bundle_from(&staged_path, &published_path).unwrap();
        let published_manifest_path = crate::output_bundle::assets_dir_for(&published_path)
            .join(crate::output_bundle::ARTIFACT_BUNDLE_MANIFEST_FILENAME);
        let published_manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(published_manifest_path).unwrap()).unwrap();
        assert_ne!(
            staged_manifest["generation"], published_manifest["generation"],
            "A11 assigns a new transaction generation when publishing the bundle"
        );
        assert_eq!(
            capture_output_identity(&published_path, true)
                .unwrap()
                .unwrap(),
            staged_identity,
            "recovery identity binds graph and manifest content, not A11's transaction token"
        );

        let mut graph_bytes = std::fs::read(&published_path).unwrap();
        graph_bytes.push(b' ');
        std::fs::write(&published_path, graph_bytes).unwrap();
        let error = capture_output_identity(&published_path, true).unwrap_err();
        assert!(
            error.contains("manifested-resource validation")
                || error.contains("graph does not match artifact bundle manifest"),
            "graph-byte tampering must invalidate the recorded content identity: {error}"
        );
    }

    #[test]
    fn publishing_recovery_reconciles_a11_commit_without_new_de_scores() {
        let directory = tempfile::tempdir().unwrap();
        let recovery_dir = directory.path().join("state");
        let output_path = directory.path().join("room.json");
        let config = inline_config();
        let session = RoomRecoverySession::open(RoomRecoveryOpen {
            directory: &recovery_dir,
            output_path: &output_path,
            room_config: &config,
            sample_rate_hz: 48_000.0,
            frequency_samples: 512,
            resume: false,
        })
        .unwrap();
        let staged_path = session.prepare_stage_output().unwrap();
        run_exact_de_for_session(&session, &config)
            .expect("production engine exact-DE path writes the terminal checkpoint");
        let terminal_state = read_journal(&recovery_dir.join(ROOM_RECOVERY_FILE_NAME))
            .unwrap()
            .unwrap()
            .checkpoint
            .expect("terminal checkpoint is persisted in the recovery journal");
        let terminal_checkpoint = terminal_state.checkpoint();
        assert!(
            terminal_checkpoint
                .terminal
                .as_ref()
                .is_some_and(|terminal| terminal.finalized)
        );
        let logical_evaluations_before_reopen = session.logical_search_evaluations().unwrap();
        assert_eq!(
            logical_evaluations_before_reopen,
            terminal_checkpoint.evaluations.saturating_add(
                terminal_checkpoint
                    .terminal
                    .as_ref()
                    .unwrap()
                    .polish_evaluations
            )
        );

        let mut graph = roomeq_model::DspGraph::new("1");
        graph.add_channel("front_left", Vec::new());
        crate::output_bundle::save_output_bundle(&mut graph, &staged_path).unwrap();
        session.mark_stage_complete(&staged_path).unwrap();
        session.mark_publishing(&staged_path).unwrap();
        crate::output_bundle::publish_output_bundle_from(&staged_path, &output_path).unwrap();
        drop(session);

        let recovered = RoomRecoverySession::open(RoomRecoveryOpen {
            directory: &recovery_dir,
            output_path: &output_path,
            room_config: &config,
            sample_rate_hz: 48_000.0,
            frequency_samples: 512,
            resume: true,
        })
        .expect("Publishing intent recognizes completed A11 output after restart");
        assert!(recovered.already_committed());
        assert_eq!(recovered.status().unwrap(), RoomRecoveryStatus::Committed);
        assert_eq!(
            recovered.logical_search_evaluations().unwrap(),
            logical_evaluations_before_reopen,
            "publication reconciliation preserves the authoritative cumulative DE count"
        );
        let published = crate::output_bundle::load_output_bundle_frozen(&output_path).unwrap();
        assert_eq!(published.output().channels.len(), 1);
    }

    #[test]
    fn changed_prior_bundle_refuses_publication_before_recording_intent() {
        let directory = tempfile::tempdir().unwrap();
        let recovery_dir = directory.path().join("state");
        let output_path = directory.path().join("room.json");
        let config = inline_config();
        let mut prior = roomeq_model::DspGraph::new("1");
        prior.add_channel("front_left", Vec::new());
        crate::output_bundle::save_output_bundle(&mut prior, &output_path).unwrap();

        let session = RoomRecoverySession::open(RoomRecoveryOpen {
            directory: &recovery_dir,
            output_path: &output_path,
            room_config: &config,
            sample_rate_hz: 48_000.0,
            frequency_samples: 512,
            resume: false,
        })
        .unwrap();
        let staged_path = session.prepare_stage_output().unwrap();
        run_exact_de_for_session(&session, &config).expect("production exact-DE pass");

        let mut candidate = roomeq_model::DspGraph::new("1");
        candidate.add_channel("front_left", Vec::new());
        crate::output_bundle::save_output_bundle(&mut candidate, &staged_path).unwrap();
        session.mark_stage_complete(&staged_path).unwrap();

        let mut changed_prior = roomeq_model::DspGraph::new("1");
        changed_prior.add_channel("front_right", Vec::new());
        crate::output_bundle::save_output_bundle(&mut changed_prior, &output_path).unwrap();
        let changed_graph = std::fs::read(&output_path).unwrap();
        let changed_manifest_path = crate::output_bundle::assets_dir_for(&output_path)
            .join(crate::output_bundle::ARTIFACT_BUNDLE_MANIFEST_FILENAME);
        let changed_manifest = std::fs::read(&changed_manifest_path).unwrap();
        let journal_path = recovery_dir.join(ROOM_RECOVERY_FILE_NAME);
        let journal_before = std::fs::read(&journal_path).unwrap();

        let error = session
            .mark_publishing(&staged_path)
            .expect_err("a stale recovery run must not claim publication intent");
        assert!(error.contains("output changed"));
        assert_eq!(session.status().unwrap(), RoomRecoveryStatus::StageComplete);
        assert_eq!(std::fs::read(&journal_path).unwrap(), journal_before);
        assert_eq!(std::fs::read(&output_path).unwrap(), changed_graph);
        assert_eq!(
            std::fs::read(changed_manifest_path).unwrap(),
            changed_manifest
        );
    }

    #[test]
    fn manifested_support_corruption_is_rejected_when_capturing_identity() {
        let directory = tempfile::tempdir().unwrap();
        let output_path = directory.path().join("room.json");
        let source_assets = directory.path().join("source-assets");
        std::fs::create_dir_all(&source_assets).unwrap();
        let pcm = b"fixture convolution resource";
        std::fs::write(source_assets.join("impulse.wav"), pcm).unwrap();
        let expected_sha256 = Sha256::digest(pcm)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let mut graph = roomeq_model::DspGraph::new("1");
        graph.add_channel(
            "front_left",
            vec![roomeq_model::Plugin {
                kind: "convolution".into(),
                parameters: serde_json::json!({"ir_file": "impulse.wav"}),
            }],
        );
        graph.metadata = Some(
            serde_json::from_value(serde_json::json!({
                "pre_score": 0.0,
                "post_score": 0.0,
                "algorithm": "fixture",
                "iterations": 0,
                "timestamp": "fixture",
                "final_convolution_sha256": {"impulse.wav": expected_sha256},
            }))
            .unwrap(),
        );
        graph.channels.get_mut("front_left").unwrap().initial_curve = Some(
            autoeq_core::Curve {
                freq: vec![100.0, 200.0, 400.0].into(),
                spl: vec![80.0, 81.0, 82.0].into(),
                ..Default::default()
            }
            .into(),
        );
        crate::output_bundle::save_output_bundle_with_resources(
            &mut graph,
            &output_path,
            &source_assets,
        )
        .unwrap();
        assert!(
            capture_output_identity(&output_path, true)
                .unwrap()
                .is_some()
        );

        let support_path = crate::output_bundle::assets_dir_for(&output_path)
            .join("resources")
            .join(format!("{expected_sha256}.wav"));
        std::fs::write(&support_path, "corrupted support payload\n").unwrap();
        let error = capture_output_identity(&output_path, true).unwrap_err();
        assert!(
            error.contains("artifact bundle"),
            "unexpected refusal: {error}"
        );
    }
}
