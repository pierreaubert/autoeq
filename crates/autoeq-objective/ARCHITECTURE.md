# autoeq-objective architecture

- `loss`: closed-form loss computations (flat/score/headphone/
  drivers/multi-sub/EPA) over prepared curves.
- `optim::objective_data` / `objective_data_builder`: objective
  inputs and the `Objective` strategy dispatch.
- `optim::loss`: strategy wrappers adapting loss computations to
  backend evaluation.
- `optim::compute`: fitness composition (base fitness, penalties,
  Pareto objectives, smoothness).
- `optim::prepared_objective`, `optim::run_control`: prepared
  evaluation with hard budgets and cooperative cancellation.
- `constraints`: evaluation-only candidate constraints.

This crate never depends on optimizer backends or RoomEQ crates;
RoomEQ-facing neutral inputs live in `roomeq_types`.
