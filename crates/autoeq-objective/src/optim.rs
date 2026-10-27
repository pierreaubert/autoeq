//! Objective evaluation modules (moved from `autoeq-optim`).

/// Strategy-based objective functions.
pub mod loss;
pub mod prepared_objective;
/// Hard objective-evaluation budgets and cooperative cancellation.
pub mod run_control;

pub mod clamp;
pub mod compute;
pub mod matched_resolution;
pub mod misc;
pub mod objective_data;
pub mod objective_data_builder;
pub mod resolution_guard;
pub mod types;

pub use crate::loss::is_peq_layout_loss;
pub use crate::penalty_mode::*;
pub use crate::smoothness_penalty_config::*;
pub use autoeq_optim_types::QAdjustment;
pub use compute::*;
pub use objective_data::*;
pub use objective_data_builder::*;
pub use resolution_guard::{PeakResolutionGuard, guarded_parameters};
pub use types::*;

/// Penalty configuration mode (crate-root module, re-exported here
/// for the historical `optim::penalty_mode` path).
pub mod penalty_mode {
    pub use crate::penalty_mode::*;
}
/// Smoothness regularizer configuration (crate-root module,
/// re-exported here for the historical path).
pub mod smoothness_penalty_config {
    pub use crate::smoothness_penalty_config::*;
}
