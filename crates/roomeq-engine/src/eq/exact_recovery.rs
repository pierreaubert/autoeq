//! Engine-owned identity and callback contracts for exact single-pass DE recovery.

use autoeq_optim::de::DECheckpoint;
use serde::{Deserialize, Serialize};

const EXACT_DE_RECOVERY_STATE_VERSION: u32 = 1;

/// A complete DE generation-barrier checkpoint accepted by the RoomEQ engine.
///
/// This is a solver state, not a warm-start candidate. Its run identity must
/// bind the prepared objective, bounds, optimizer configuration, and build
/// identity used by the engine before it will resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactDERecoveryState {
    schema_version: u32,
    run_identity: String,
    checkpoint: DECheckpoint,
}

impl ExactDERecoveryState {
    /// Construct an engine recovery state from one math-layer checkpoint.
    ///
    /// # Errors
    /// Returns an error if the identity is empty, differs from the checkpoint,
    /// or the checkpoint lacks solver source/build identity.
    pub fn from_checkpoint(checkpoint: DECheckpoint, run_identity: &str) -> Result<Self, String> {
        let state = Self {
            schema_version: EXACT_DE_RECOVERY_STATE_VERSION,
            run_identity: run_identity.to_owned(),
            checkpoint,
        };
        state.validate()?;
        Ok(state)
    }

    /// Verify that this state is for the requested prepared optimization.
    ///
    /// # Errors
    /// Returns an error when the state is malformed or has a different run
    /// identity.
    pub fn check_compatible(&self, run_identity: &str) -> Result<(), String> {
        self.validate()?;
        if run_identity.trim().is_empty() {
            return Err("requested exact RoomEQ identity is empty".into());
        }
        if self.run_identity != run_identity {
            return Err("exact RoomEQ prepared-objective identity mismatch".into());
        }
        Ok(())
    }

    /// Return the identity stored with this complete solver state.
    #[must_use]
    pub fn run_identity(&self) -> &str {
        &self.run_identity
    }

    /// Return the complete math-layer checkpoint.
    #[must_use]
    pub fn checkpoint(&self) -> &DECheckpoint {
        &self.checkpoint
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_version != EXACT_DE_RECOVERY_STATE_VERSION {
            return Err(format!(
                "unsupported exact RoomEQ checkpoint schema {}; expected {EXACT_DE_RECOVERY_STATE_VERSION}",
                self.schema_version
            ));
        }
        if self.run_identity.trim().is_empty() {
            return Err("exact RoomEQ checkpoint is missing its run identity".into());
        }
        if self.checkpoint.run_identity != self.run_identity {
            return Err("exact RoomEQ state and solver checkpoint identities disagree".into());
        }
        if self.checkpoint.solver_source_identity.trim().is_empty() {
            return Err("exact RoomEQ checkpoint is missing solver source identity".into());
        }
        if self.checkpoint.build_identity.trim().is_empty() {
            return Err("exact RoomEQ checkpoint is missing executable build identity".into());
        }
        Ok(())
    }
}

/// Save callback used by the engine after each complete DE generation barrier.
pub type ExactDERecoverySaveCallback =
    Box<dyn FnMut(&ExactDERecoveryState) -> Result<(), String> + Send>;

/// Exact DE continuation request accepted by the RoomEQ engine.
pub struct ExactDERecoveryOptions {
    run_identity: String,
    checkpoint: Option<ExactDERecoveryState>,
    save_callback: ExactDERecoverySaveCallback,
}

impl ExactDERecoveryOptions {
    /// Create an exact DE request with an optional complete state and durable
    /// generation-barrier callback.
    ///
    /// The callback's successful return is the caller's durability
    /// acknowledgement; the solver itself does not promise filesystem sync.
    ///
    /// # Errors
    /// Returns an error when the caller identity is empty or the supplied
    /// checkpoint state is malformed.
    pub fn new(
        run_identity: impl Into<String>,
        checkpoint: Option<ExactDERecoveryState>,
        save_callback: ExactDERecoverySaveCallback,
    ) -> Result<Self, String> {
        let run_identity = run_identity.into();
        if run_identity.trim().is_empty() {
            return Err("exact RoomEQ run identity is empty".into());
        }
        if let Some(checkpoint) = &checkpoint {
            checkpoint.validate()?;
        }
        Ok(Self {
            run_identity,
            checkpoint,
            save_callback,
        })
    }

    /// Return the outer workflow identity used to bind the prepared objective.
    #[must_use]
    pub fn run_identity(&self) -> &str {
        &self.run_identity
    }

    /// Return the state selected for continuation, when this is a resume.
    #[must_use]
    pub fn checkpoint(&self) -> Option<&ExactDERecoveryState> {
        self.checkpoint.as_ref()
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        String,
        Option<ExactDERecoveryState>,
        ExactDERecoverySaveCallback,
    ) {
        (self.run_identity, self.checkpoint, self.save_callback)
    }
}

impl std::fmt::Debug for ExactDERecoveryOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExactDERecoveryOptions")
            .field("run_identity", &self.run_identity)
            .field("checkpoint", &self.checkpoint)
            .field("save_callback", &"<generation-barrier callback>")
            .finish()
    }
}
