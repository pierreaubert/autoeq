//! Topology workflow orchestration over prepared engine operations.

mod bass_management;
#[cfg(test)]
mod executor_tests;
mod generic;
mod home_cinema;
mod optimize;
mod run;
mod stereo;
mod stereo_sub;
mod supporting_source;
#[cfg(test)]
mod tests;
mod types;
mod workflow;
pub(crate) use home_cinema::CROSSOVER_CANCELLATION_UNASSESSED_ADVISORY;
pub(crate) use home_cinema::RoleSpliceOutcome;
pub(crate) use home_cinema::collect_routed_splice_outcomes;
pub(crate) use home_cinema::crossover_timing_refused;
pub(crate) use home_cinema::main_level_alignment_band;
pub(crate) use home_cinema::reconstruct_deployed_snapshot_best_effort;
pub(crate) use home_cinema::reconstruct_deployed_source_curves;
pub(crate) use home_cinema::reconstruct_deployed_source_curves_unenforced;
pub(crate) use home_cinema::reconstruct_deployed_source_curves_with_evidence;
#[cfg(test)]
pub(crate) use roomeq_engine::topology::*;

pub use optimize::*;
pub use types::{WorkflowProgressCallback, WorkflowProgressCallbackFactory, WorkflowStageCallback};
