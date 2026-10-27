//! Wolfram oracle group for the optimizer and AutoEQ workflow.
//!
//! Grouped test target (refactor-20261025 Phase 2): each module below
//! is one manifest case file kept at `tests/<module>.rs`. The case ->
//! target/module mapping is declared in `validation-manifest.toml`
//! (`test` names the module, `target` names this group) and enforced
//! by `qa_contract` and `scripts/run_crate_test_matrix.py`.

#[path = "wolfram_aw01_workflow_correction.rs"]
mod wolfram_aw01_workflow_correction;

#[path = "wolfram_o01_asymmetric_weights.rs"]
mod wolfram_o01_asymmetric_weights;

#[path = "wolfram_o01_scalar_weights.rs"]
mod wolfram_o01_scalar_weights;

#[path = "wolfram_o02_scalarization.rs"]
mod wolfram_o02_scalarization;

#[path = "wolfram_o03_tv2_stencil.rs"]
mod wolfram_o03_tv2_stencil;

#[path = "wolfram_o04_bass_boost.rs"]
mod wolfram_o04_bass_boost;

#[path = "wolfram_o04_deadband.rs"]
mod wolfram_o04_deadband;

#[path = "wolfram_o05_constraint_penalties.rs"]
mod wolfram_o05_constraint_penalties;

#[path = "wolfram_o06_initial_guess_bounds.rs"]
mod wolfram_o06_initial_guess_bounds;

#[path = "wolfram_o07_crossover_sum.rs"]
mod wolfram_o07_crossover_sum;

#[path = "wolfram_o08_joint_multisub.rs"]
mod wolfram_o08_joint_multisub;

#[path = "wolfram_o09_group_delay.rs"]
mod wolfram_o09_group_delay;

#[path = "wolfram_o10_bark_sharpness.rs"]
mod wolfram_o10_bark_sharpness;

#[path = "wolfram_o11_temporal_ir.rs"]
mod wolfram_o11_temporal_ir;

#[path = "wolfram_o12_pareto_mixed.rs"]
mod wolfram_o12_pareto_mixed;

#[path = "wolfram_o13_backend_objectives.rs"]
mod wolfram_o13_backend_objectives;

#[path = "wolfram_o14_rerank_ordering.rs"]
mod wolfram_o14_rerank_ordering;
