# autoeq-workflow — Architecture

High-level speaker and headphone equalization workflows.
Composes load → optimize → realize → report into resumable runs.

## Layer

Orchestration over `autoeq-core` + `autoeq-measurements` + `autoeq-optim`.
The `autoeq` binary and `autoeq-cli` adapters are thin callers of this crate.

## Key modules (`workflow/`)

| Module | Owns |
|---|---|
| `load` | Measurement intake and target preparation |
| `config` / `types` | Run configuration and shared workflow types |
| `build` | Objective assembly from config + curves |
| `optimize` | Backend invocation, progress, seed handling |
| `resume` | Deterministic run resumption |
| `misc` | Shared helpers; `qa_println!` debug macro for historical CLI callers |

## Core abstractions

- **Staged run.** A run is `load` (curves + target) → `build` (objective) →
  `optimize` (backend over the registry) → realized filters + report. Each
  stage is a free function over plain config/curve types, so stages are
  testable in isolation.
- **Resumability.** `resume` persists run state (`save_optimizer_state` /
  `load_optimizer_state`) so long optimizations continue where they stopped.
- **`cli` module.** Shared argument shapes consumed by `autoeq-cli`, keeping
  flag parsing identical across binaries.

## API

```rust
// Intake: files + target selection.
let curves = workflow::load::load_driver_measurements_from_files(&paths)?;
let target = workflow::build::build_target_curve_by_name("harman", &grid, &avg)?;

// Optimize by product shape; multisub splits objective prep from the run.
let hp = workflow::optimize::optimize_headphone_with_grid(curves, target, &params)?;
let xover = workflow::optimize::optimize_drivers_crossover(drivers, &params)?;
let obj = workflow::optimize::prepare_multisub_objective(&subs)?;
let ms = workflow::optimize::optimize_multisub_prepared(obj, &params)?;

// Resumability is persisted optimizer state, not checkpoints.
workflow::resume::save_optimizer_state(&dir, &state)?;
```

## Data contracts

- **In:** run configs, measurement paths/IDs, target specifications.
- **Out:** optimized filter sets, scores, HTML/JSON reports (via `autoeq-plot`
  and `autoeq-artifacts` at the CLI boundary).

## Consumers

`autoeq-cli`, the `autoeq` and `benchmark-autoeq-speaker` binaries.
