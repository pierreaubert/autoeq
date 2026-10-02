//! Exact AutoEQ DE continuation checkpoints.
//!
//! Unlike [`super::resume::OptimizerState`], this file stores the complete
//! math-layer DE barrier state. It is valid only for the same measurement,
//! objective, normalization, bounds, optimizer settings, seed, math source,
//! and target encoded by the caller's run identity and the DE checkpoint.

use autoeq_optim::de::DECheckpoint;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// Serialized schema version for exact AutoEQ DE state files.
pub const EXACT_DE_STATE_VERSION: u32 = 1;
/// Maximum serialized exact checkpoint size accepted on load or save.
///
/// The cap bounds memory used for JSON parsing and prevents oversized
/// population/archive snapshots from being loaded or published accidentally.
pub const EXACT_DE_STATE_MAX_BYTES: usize = 32 * 1024 * 1024;

/// Complete exact-continuation record for one AutoEQ DE run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactOptimizerState {
    /// Exact-state file schema version.
    pub schema_version: u32,
    /// Caller identity binding measurement, objective, normalization, and config.
    pub run_identity: String,
    /// Safe-generation-barrier DE state from math-optimisation.
    pub checkpoint: DECheckpoint,
    /// RFC3339 UTC timestamp when this file state was constructed.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ExactOptimizerState {
    /// Create a strict exact-state record from a solver checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error when the caller identity is empty, the math
    /// checkpoint carries a different run identity, or its source/build
    /// identity fields are missing.
    pub fn from_checkpoint(checkpoint: DECheckpoint, run_identity: &str) -> Result<Self, String> {
        let state = Self {
            schema_version: EXACT_DE_STATE_VERSION,
            run_identity: run_identity.to_owned(),
            checkpoint,
            timestamp: chrono::Utc::now(),
        };
        state.validate_for_save()?;
        Ok(state)
    }

    /// Reject exact continuation when the saved identity differs from this run.
    ///
    /// This checks the outer measurement/configuration identity. The solver
    /// validates math-source, executable-build, and full checkpoint integrity
    /// before its first objective evaluation.
    ///
    /// # Errors
    ///
    /// Returns an error when the state is malformed or when the requested
    /// identity is empty or differs from the saved identity.
    pub fn check_compatible(&self, run_identity: &str) -> Result<(), String> {
        self.validate_for_save()?;
        if run_identity.trim().is_empty() {
            return Err("requested exact-resume identity is empty".to_owned());
        }
        if self.run_identity != run_identity {
            return Err("exact-resume measurement/objective/config identity mismatch".to_owned());
        }
        Ok(())
    }

    fn validate_for_save(&self) -> Result<(), String> {
        if self.schema_version != EXACT_DE_STATE_VERSION {
            return Err(format!(
                "unsupported exact DE state schema {}; expected {EXACT_DE_STATE_VERSION}",
                self.schema_version
            ));
        }
        if self.run_identity.trim().is_empty() {
            return Err("exact DE state is missing run identity".to_owned());
        }
        if self.checkpoint.run_identity != self.run_identity {
            return Err("exact DE state and math checkpoint identities disagree".to_owned());
        }
        if self.checkpoint.solver_source_identity.trim().is_empty() {
            return Err("exact DE state is missing solver source identity".to_owned());
        }
        if self.checkpoint.build_identity.trim().is_empty() {
            return Err("exact DE state is missing executable build identity".to_owned());
        }
        Ok(())
    }
}

/// Atomically write an exact DE checkpoint beside the destination file.
/// Serialized states larger than [`EXACT_DE_STATE_MAX_BYTES`] are rejected.
///
/// A failure before replacement leaves any previous complete checkpoint
/// untouched. Once rename succeeds, a parent-directory flush failure reports
/// durability uncertainty while the complete new file remains visible.
///
/// # Errors
///
/// Returns an error when the state is invalid or oversized, serialization
/// fails, or a filesystem write, replacement, or durability flush fails.
pub fn save_exact_optimizer_state(state: &ExactOptimizerState, path: &Path) -> io::Result<()> {
    state
        .validate_for_save()
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    atomic_replace(path, |file| {
        write_bounded_state(file, state, EXACT_DE_STATE_MAX_BYTES)
    })
}

/// Load an exact DE state. Reads are capped at [`EXACT_DE_STATE_MAX_BYTES`]
/// before parsing. Candidate-only warm-start files are rejected with an
/// actionable message instead of being treated as exact continuation.
///
/// # Errors
///
/// Returns an error when the file cannot be read, exceeds the size limit,
/// contains malformed or unsupported exact-state data, or contains only a
/// warm-start candidate. A missing file is returned as `Ok(None)`.
pub fn load_exact_optimizer_state(path: &Path) -> io::Result<Option<ExactOptimizerState>> {
    let json = match read_bounded_file(path, EXACT_DE_STATE_MAX_BYTES) {
        Ok(json) => json,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let value: serde_json::Value = serde_json::from_slice(&json)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if value.get("checkpoint").is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file contains a warm-start candidate only; use --resume-state or save an exact DE checkpoint",
        ));
    }
    let state: ExactOptimizerState = serde_json::from_value(value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    state
        .validate_for_save()
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?;
    Ok(Some(state))
}

fn read_bounded_file(path: &Path, limit: usize) -> io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut reader = file.take(limit as u64 + 1);
    let mut bytes = Vec::with_capacity(limit.min(64 * 1024));
    reader.read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("exact DE checkpoint exceeds the {limit}-byte size limit"),
        ));
    }
    Ok(bytes)
}

fn write_bounded_state(
    file: &mut File,
    state: &ExactOptimizerState,
    limit: usize,
) -> io::Result<()> {
    let mut writer = BoundedWriter {
        file,
        written: 0,
        limit,
        exceeded: false,
    };
    serde_json::to_writer_pretty(&mut writer, state).map_err(|error| {
        if writer.exceeded {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("exact DE checkpoint exceeds the {limit}-byte size limit"),
            )
        } else {
            io::Error::new(io::ErrorKind::InvalidData, error)
        }
    })?;
    writer.flush()
}

struct BoundedWriter<'a> {
    file: &'a mut File,
    written: usize,
    limit: usize,
    exceeded: bool,
}

impl Write for BoundedWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.written.saturating_add(buffer.len()) > self.limit {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "serialized checkpoint exceeds configured size limit",
            ));
        }
        let written = self.file.write(buffer)?;
        self.written += written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use autoeq_optim::de::{DEConfigBuilder, DifferentialEvolution};
    use ndarray::{Array1, array};

    fn checkpoint() -> DECheckpoint {
        let objective = |x: &Array1<f64>| x.iter().map(|value| value * value).sum();
        let config = DEConfigBuilder::new()
            .seed(801)
            .maxiter(2)
            .popsize(4)
            .tol(0.0)
            .atol(0.0)
            .build()
            .expect("valid fixture configuration");
        let mut solver = DifferentialEvolution::new(&objective, array![-1.0], array![1.0])
            .expect("valid fixture bounds");
        *solver.config_mut() = config;
        let mut saved = None;
        let mut callback = |state: &DECheckpoint| {
            if state.generation == 1 {
                saved = Some(state.clone());
                Err("stop after test barrier".to_owned())
            } else {
                Ok(())
            }
        };
        let _ = solver.solve_with_checkpoint(None, "fixture-run", Some(&mut callback));
        saved.expect("generation checkpoint").clone()
    }

    #[test]
    fn exact_state_roundtrips_and_rejects_identity_or_legacy_candidate_files() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let path = temporary.path().join("exact-state.json");
        let state = ExactOptimizerState::from_checkpoint(checkpoint(), "fixture-run")
            .expect("consistent identity");
        save_exact_optimizer_state(&state, &path).expect("save exact state");
        let loaded = load_exact_optimizer_state(&path)
            .expect("load exact state")
            .expect("state exists");
        loaded
            .check_compatible("fixture-run")
            .expect("same objective identity");
        assert!(loaded.check_compatible("different-run").is_err());

        let legacy_path = temporary.path().join("warm-start.json");
        std::fs::write(&legacy_path, br#"{"best_params":[0.0]}"#).expect("write candidate fixture");
        let error = load_exact_optimizer_state(&legacy_path).unwrap_err();
        assert!(error.to_string().contains("warm-start candidate only"));
    }

    #[test]
    fn bounded_io_rejects_oversized_state_without_replacing_prior_checkpoint() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let destination = temporary.path().join("exact-state.json");
        std::fs::write(&destination, b"previous valid checkpoint").expect("previous state");
        let state = ExactOptimizerState::from_checkpoint(checkpoint(), "fixture-run")
            .expect("consistent identity");

        let save_error = atomic_replace(&destination, |file| write_bounded_state(file, &state, 32))
            .expect_err("small size limit rejects serialized state");
        assert!(save_error.to_string().contains("size limit"));
        assert_eq!(
            std::fs::read(&destination).expect("prior file survives"),
            b"previous valid checkpoint"
        );

        let oversized = temporary.path().join("oversized.json");
        std::fs::write(&oversized, [b'x'; 65]).expect("oversized fixture");
        let load_error = read_bounded_file(&oversized, 64).expect_err("oversize rejected");
        assert!(load_error.to_string().contains("64-byte size limit"));
    }
}

#[cfg(test)]
mod production_split_resume_tests {
    use super::{ExactOptimizerState, load_exact_optimizer_state, save_exact_optimizer_state};
    use autoeq_optim::cli::Args;
    use autoeq_optim::de::DECheckpoint;
    use autoeq_optim::loss::LossType;
    use autoeq_optim::optim::de::DECheckpointSaveCallback;
    use autoeq_optim::optim::{ObjectiveData, ObjectiveDataBuilder};
    use autoeq_optim::{OptimParams, PeqModel};
    use ndarray::Array1;
    use std::sync::{Arc, Mutex};

    fn params() -> OptimParams {
        let args = Args::speaker_defaults();
        let mut params = OptimParams::from(&args);
        params.peq_model = PeqModel::Pk;
        params.num_filters = 1;
        params.min_freq = 40.0;
        params.max_freq = 20_000.0;
        params.min_q = 0.5;
        params.max_q = 4.0;
        params.min_db = -6.0;
        params.max_db = 6.0;
        params.algo = "autoeq:de".to_owned();
        params.population = 8;
        params.maxeval = 48;
        params.seed = Some(91_827);
        params.no_parallel = true;
        params.parallel_threads = 1;
        params.refine = false;
        params.tolerance = 0.0;
        params.atolerance = 0.0;
        params
    }

    fn analytic_objective(target_offset_db: f64) -> ObjectiveData {
        let freqs = Array1::from_iter(
            (0..12).map(|index| 40.0 * (20_000.0_f64 / 40.0).powf(index as f64 / 11.0)),
        );
        let target = Array1::from_iter(freqs.iter().map(|frequency| {
            let log_ratio = (frequency / 900.0).ln() / 0.45;
            target_offset_db + 3.5 * (-0.5 * log_ratio * log_ratio).exp()
        }));
        let deviation = Array1::from_elem(freqs.len(), 0.25);
        ObjectiveDataBuilder::new(
            freqs,
            target,
            deviation,
            48_000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
        )
        .max_db(6.0)
        .min_db(-6.0)
        .freq_range(40.0, 20_000.0)
        .smoothing(false, 1)
        .build()
        .expect("valid analytic fixture objective")
    }

    fn run_identity(target_offset_db: f64) -> String {
        format!("analytic-speaker-target-offset-db={target_offset_db:.1}-v1")
    }

    fn run_exact(
        params: &OptimParams,
        objective: &ObjectiveData,
        checkpoint: Option<DECheckpoint>,
        run_identity: String,
        save_callback: DECheckpointSaveCallback,
    ) -> Result<(Vec<f64>, f64), Box<dyn std::error::Error>> {
        let output = autoeq_optim::optim::setup::perform_optimization_with_run_descriptor_and_exact_checkpoint(
            params,
            objective,
            autoeq_optim::optim::setup::ExactDECheckpointOptions {
                checkpoint,
                run_identity,
                save_callback,
            },
            Box::new(|_| autoeq_optim::de::CallbackAction::Continue),
        )?;
        Ok((output.parameters, output.objective_value))
    }

    #[test]
    fn production_save_load_resume_matches_uninterrupted_and_rejects_stale_inputs_before_scoring() {
        let params = params();
        let identity = run_identity(0.0);
        let baseline_objective = analytic_objective(0.0);
        let uninterrupted = run_exact(
            &params,
            &baseline_objective,
            None,
            identity.clone(),
            Box::new(|_| Ok(())),
        )
        .expect("uninterrupted AutoEQ setup run");

        let temporary = tempfile::tempdir().expect("temporary checkpoint directory");
        let checkpoint_path = temporary.path().join("exact-state.json");
        let interrupted_objective = analytic_objective(0.0);
        let saved_checkpoint = Arc::new(Mutex::new(None));
        let checkpoint_for_callback = Arc::clone(&saved_checkpoint);
        let callback_path = checkpoint_path.clone();
        let callback_identity = identity.clone();
        let save_and_interrupt: DECheckpointSaveCallback = Box::new(move |checkpoint| {
            if checkpoint.generation >= 2 && checkpoint.terminal.is_none() {
                let state =
                    ExactOptimizerState::from_checkpoint(checkpoint.clone(), &callback_identity)
                        .map_err(|error| error.to_string())?;
                save_exact_optimizer_state(&state, &callback_path)
                    .map_err(|error| error.to_string())?;
                *checkpoint_for_callback
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(checkpoint.clone());
                return Err("intentional stop after saved production barrier".to_owned());
            }
            Ok(())
        });
        assert!(
            run_exact(
                &params,
                &interrupted_objective,
                None,
                identity.clone(),
                save_and_interrupt,
            )
            .is_err(),
            "the saved barrier should interrupt the first run"
        );

        let captured = saved_checkpoint
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .expect("interrupted production run saved a generation barrier");
        let loaded = load_exact_optimizer_state(&checkpoint_path)
            .expect("read actual checkpoint file")
            .expect("saved checkpoint exists");
        assert_eq!(
            loaded.checkpoint, captured,
            "JSON save/load must preserve every DE checkpoint field and float exactly"
        );

        let resumed_objective = analytic_objective(0.0);
        let resumed = run_exact(
            &params,
            &resumed_objective,
            Some(loaded.checkpoint.clone()),
            identity.clone(),
            Box::new(|_| Ok(())),
        )
        .expect("resume through production setup after disk roundtrip");
        assert_eq!(resumed.0, uninterrupted.0, "resumed filters must match");
        assert_eq!(resumed.1, uninterrupted.1, "resumed loss must match");

        let changed_target = analytic_objective(0.5);
        let target_error = run_exact(
            &params,
            &changed_target,
            Some(loaded.checkpoint.clone()),
            run_identity(0.5),
            Box::new(|_| Ok(())),
        )
        .expect_err("a changed target must reject the saved identity");
        assert!(target_error.to_string().contains("identity"));
        assert!(
            changed_target.prepared.get().is_none(),
            "identity mismatch must fail before any objective scoring"
        );

        let mut changed_params = params.clone();
        // Add enough budget to change the native DE generation count for this fixture.
        changed_params.maxeval = params.maxeval.saturating_add(10_000);
        let changed_budget_objective = analytic_objective(0.0);
        let budget_error = run_exact(
            &changed_params,
            &changed_budget_objective,
            Some(loaded.checkpoint),
            identity,
            Box::new(|_| Ok(())),
        )
        .expect_err("a changed DE budget must reject saved exact state");
        assert!(budget_error.to_string().contains("configuration"));
        assert!(
            changed_budget_objective.prepared.get().is_none(),
            "configuration mismatch must fail before objective scoring"
        );
    }
}
