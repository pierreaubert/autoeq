# Crate Test Matrix

This is the focused verification inventory for the crate-partition migration.
The executable source of truth is `scripts/crate_partition_policy.json`; the
architecture checker rejects a workspace package without a focused command.

| Package | Focused command | Current responsibility |
|---|---|---|
| `autoeq` | `cargo test -p autoeq --lib` | Compatibility facade and remaining root implementation |
| `autoeq-core` | `cargo test -p autoeq-core --lib` | Curves, PEQ layouts, response math, numerical primitives |
| `autoeq-artifacts` | `cargo test -p autoeq-artifacts --lib` | Artifact-store ports and implementations |
| `autoeq-cli` | `cargo test -p autoeq-cli --lib` | AutoEQ, benchmark, and acquisition command adapters |
| `autoeq-fir` | `cargo test -p autoeq-fir --lib` | FIR design and encoding primitives |
| `autoeq-measurements` | `cargo test -p autoeq-measurements --lib` | Measurement acquisition and normalization |
| `autoeq-optim` | `cargo test -p autoeq-optim --lib` | Objectives and optimizer backends |
| `autoeq-plot` | `cargo test -p autoeq-plot --lib` | Plot and report rendering |
| `autoeq-workflow` | `cargo test -p autoeq-workflow --lib` | Speaker/headphone use cases |
| `roomeq-model` | `cargo test -p roomeq-model --lib` | RoomEQ contracts and validation |
| `roomeq-analysis` | `cargo test -p roomeq-analysis --lib` | In-memory acoustic analysis |
| `roomeq-cli` | `cargo test -p roomeq-cli --lib` | RoomEQ and recording-conversion command adapters |
| `roomeq-quality` | `cargo test -p roomeq-quality --lib` | Metrics and acceptance policy |
| `roomeq-engine` | `cargo test -p roomeq-engine --lib` | Extracted deterministic processing |
| `roomeq-export` | `cargo test -p roomeq-export --lib` | External-format contracts and rendering |
| `roomeq-synthetic` | `cargo test -p roomeq-synthetic --lib` | Deterministic scenarios and measurements |
| `roomeq-qa` | `cargo test -p roomeq-qa --lib` | Scenario matrices, regression runners, reports, and fuzzing |
| `roomeq-workflow` | `cargo test -p roomeq-workflow --lib` | Resource loading and engine invocation |

Run all WP0 fitness and compatibility checks with:

```bash
just check-crate-partition
```

When a new package is added, add its target dependency policy, focused command,
and this table row in the same change. When behavior moves, run the destination
command and the commands for its direct workspace consumers before checking the
root facade.
