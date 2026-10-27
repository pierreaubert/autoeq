# autoeq-optim architecture

Backend execution over prepared `autoeq-objective` inputs and neutral
`autoeq-optim-types` configuration/evidence contracts. No command-line parser,
RoomEQ model, or resource loading is owned here.

- `optim::registry`: canonical algorithm names, legacy aliases, and backend
  selection. Shipping defaults preserve the complete registry.
- `optim::{de,cobyla,cobra,isres,cmaes,bo,nsga,mh}`: solver adapters. The
  default-enabled `backend-metaheuristics` feature gates the `mh` implementation,
  dependency, registry entries, and its tests. Built-in-only clients may disable
  default features.
- `optim::params`: `OptimParams`, converted from parser-independent
  `autoeq_optim_types::OptimizationConfig` or a higher-layer RoomEQ adapter.
- `optim::{setup,scalar,run_control}`: bounds, initial guesses, execution budgets,
  cancellation, restart control, and structured termination evidence.
- `driver_optimization`, `perceptual_promotion`, `rerank`: backend orchestration
  and candidate selection. Objective math and evaluation-only constraints are
  owned by `autoeq-objective`; core filter encoding is owned by `autoeq-core`.
- `benches/solver_generation.rs`: backend-generation benchmarks. Shared fixtures
  preserve the evaluator/backend comparison; evaluator-only benches live in
  `autoeq-objective`.

Consumers include `autoeq-workflow`, `autoeq-cli`, `roomeq-engine`, and QA.
Desktop configuration clients import `autoeq-optim-types` directly.
