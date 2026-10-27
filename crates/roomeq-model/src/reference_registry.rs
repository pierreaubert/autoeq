//! Approved independent-reference registry for promotion claims.
//!
//! The canonical types live in `autoeq-optim-types` so optimizer and
//! quality gates can share them without depending on this crate
//! (refactor-20261025 Phase 3c). This module path is retained for
//! existing `roomeq_model::reference_registry::` consumers.

pub use autoeq_optim_types::reference_registry::*;
