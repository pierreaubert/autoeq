//! AutoEQ - A library for audio equalization and filter optimization
//!
//! Copyright (C) 2025-2026 Pierre Aubert pierre(at)spinorama(dot)org
//!
//! This program is free software: you can redistribute it and/or modify
//! it under the terms of the GNU General Public License as published by
//! the Free Software Foundation, either version 3 of the License, or
//! (at your option) any later version.
//!
//! This program is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//! GNU General Public License for more details.
//!
//! You should have received a copy of the GNU General Public License
//! along with this program.  If not, see <https://www.gnu.org/licenses/>.

pub use backend::{AlgorithmType, ConstraintCapabilities, FilterOptimizer, FilterOptimizerOutput};
// Objective evaluation lives in `autoeq-objective` (refactor-20261025
// Phase 3c); these re-exports preserve every historical module path.
use autoeq_objective::optim::{
    clamp, compute, misc, objective_data, objective_data_builder, penalty_mode, resolution_guard,
    smoothness_penalty_config,
};
pub use autoeq_objective::optim::{loss, prepared_objective, run_control};

/// Optimizer termination message and objective value, for success or failure.
///
/// Both variants preserve the backend's reason and last reported objective.
pub type OptimizerOutcome = Result<(String, f64), (String, f64)>;

/// Version of the optimizer implementation used in warm-start identities.
pub const OPTIMIZER_IMPLEMENTATION_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Unified optimizer backend trait and capability descriptors.
pub mod backend;
/// Gaussian-process Bayesian optimisation backend.
pub mod bo;
/// Shared callback utilities for optimization
pub mod callback;
/// Pure-Rust CMA-ES backend.
pub mod cmaes;
/// COBRA surrogate optimization with native inequalities.
pub mod cobra;
/// Pure-Rust COBYLA backend (replaces NLopt's COBYLA when nlopt feature is off).
pub mod cobyla;
/// Frequency-dependent constraint envelopes and neutral diagnostics.
pub mod constraint_envelope;
/// Centralised constraint installation (native vs penalty).
pub mod constraints_install;
/// AutoEQ DE-specific optimization code
pub mod de;
/// Pure-Rust ISRES backend.
pub mod isres;
/// Metaheuristics-specific optimization code.
#[cfg(feature = "backend-metaheuristics")]
pub mod mh;
/// Pure-Rust NSGA-II/III Pareto backends.
pub mod nsga;
/// Testable high-level optimizer seam for EQ pipelines.
pub mod optimizer_backend;
/// Shared optimization parameters (decoupled from CLI args)
pub mod params;
/// Pareto front analysis
pub mod pareto;
/// Algorithm registry — string name → backend.
pub mod registry;
/// Shared bounded scalar-objective optimizer dispatch.
pub mod scalar;
/// Shared optimization setup (bounds, initial guess, objective data)
pub mod setup;
/// Backend progress callbacks and shared objective inputs.
pub mod types;

mod algorithm_info;
mod optimize;
#[cfg(test)]
mod tests;

pub use algorithm_info::*;
pub use clamp::*;
pub use compute::*;
pub use constraint_envelope::{
    BudgetBreach, COMPOSITE_COMPARISON_EPS_DB, ClassifiedOutcome, CompositeBreach,
    ConstrainedCandidate, ConstraintDiagnostic, ConstraintKind, ConstraintSpec, FinalizedCandidate,
    GainAdjustment, JudgedParetoFront, JudgedParetoMember, OptimizationOutcomeKind,
    OwnedConstraintSpec, ParetoFeasibility, ParetoValidationError, QAdjustment,
    VALIDATED_SUBDIVISIONS_PER_BIN, check_composite_gain_envelope, check_pareto_feasibility,
    classify_outcome, constrain_candidate, enforce_local_q_at_centers,
    enforce_local_q_envelope_at_centers, envelope_bound_at, finalize_candidate, is_peq_layout_loss,
    judge_pareto_members, judge_pareto_members_with_stop, policy_local_q_knots,
    project_gains_onto_envelopes, validate_envelope_knots, validated_composite_grid,
    verify_joint_budgets,
};
pub use objective_data::*;
pub use objective_data_builder::*;
pub use optimize::*;
pub use optimizer_backend::*;
pub use penalty_mode::*;
pub use resolution_guard::{PeakResolutionGuard, guarded_parameters};
pub use smoothness_penalty_config::*;
pub use types::*;
pub mod run_descriptor;
