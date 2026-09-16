# autoeq-optim — Architecture

Objective functions and optimization backends for AutoEQ.
Maps measurements + targets to optimal PEQ parameter vectors.

## Layer

Compute layer over `autoeq-core` + `autoeq-measurements`, using
`math-optimisation` backends. No CLI or artifact I/O here (`cli.rs` only
carries shared argument types).

## Key modules

| Module | Owns |
|---|---|
| `optim/registry` | Backend registry: `autoeq:de` (JADE/L-SHADE variants), `autoeq:cobyla`, `autoeq:isres`, `autoeq:cmaes`, `autoeq:nsga2/3`, `mh:*`. Legacy `nlopt:*` names survive as compatibility aliases mapped to pure-Rust backends. Bare aliases (`de`, `cma-es`, …) resolve here too |
| `optim/loss/strategies` | `Objective` strategy trait: `flat`, `score`, `epa`, `asymmetric`. `ObjectiveData` caches the strategy so it is built once |
| `loss/` | Concrete losses: `speaker`, `headphone`, `drivers` (+crossover), `multisub`, `epa`, `flat`, `slope`, `bass_boost`, `phase_aware`, `enhanced_weights` |
| `optim/objective_data` | Single-curve and multi-objective paths; `multi_objective` scalarizes per-curve objectives |
| `initial_guess`, `param_utils` (via core) | Seed vectors per `PeqLayout` |
| `optim/smoothness_penalty_config` | Optional TV² curvature regularizer in log-frequency (`tv2_weight`, `schroeder_hz`, `modal_weight_scale`, `exponent`) |
| `rerank`, `driver_optimization` | Post-optimization candidate ranking; multi-driver crossover co-optimization |
| `constraints`, `penalty_mode` | Bounds/penalty handling shared by backends |

## Core abstractions

- **`Objective` trait** — one method, `compute(&self, x: &[f64], ctx:
  &ObjectiveContext) -> f64`, plus an optional `compute_response` for scoring
  an already-realized correction (lets FIR stages reuse PEQ loss/deadband/
  regularizer math). Each `LossType` maps to one implementation, so the
  dispatcher in `optim::compute` stays a small match.
- **`ObjectiveData`** — bundles curves, targets, bands, and weights, and
  caches the built `Objective` strategy. Multi-objective runs scalarize
  per-curve objectives through the same type.
- **`FilterOptimizer` + `registry::resolve(name)`** — backends are selected by
  string (`"autoeq:cmaes"`, `"de"`, `"nlopt:cobyla"`, …) and return
  `Option<Box<dyn FilterOptimizer>>`; unknown names fail at resolve time.
- **`OptimParams` + `OptimizerRunEvidence`** — run configuration in,
  structured termination/convergence evidence out
  (`optimize_filters_detailed`).

## API

```rust
// Pick a backend by name (aliases + legacy nlopt names resolve here).
let backend = registry::resolve("autoeq:cmaes").expect("known optimizer");

// Optimize a PEQ vector in place within bounds. Both success and
// failure carry (algorithm, loss), so callers always learn something.
let (algo_used, best_loss) =
    optimize_filters(&mut x, &lower, &upper, objective_data, &params).unwrap_or_else(|e| e);

// Or keep the full convergence evidence.
let evidence: OptimizerRunEvidence =
    optimize_filters_detailed(&mut x, &lower, &upper, objective_data, &params);
```

## Data contracts

- **In:** `Curve`s, target curves, `OptimizerSettings` (algorithm, bands, bounds, seeds).
- **Out:** solution vectors + run evidence (iterations, confidence) decoded via `x2peq`.

## Consumers

`autoeq-workflow`, `autoeq-plot`, `roomeq-engine` (channel optimizers),
`roomeq-qa`.
