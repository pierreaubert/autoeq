# roomeq-cli — Architecture

RoomEQ command-line parsing and user-facing orchestration.
Owns the `roomeq` CLI surface; execution lives downstream.

## Layer

Thin adapter over `roomeq-workflow` + `roomeq-engine` + `roomeq-export` (+
`autoeq-optim` types for optimizer flags). Input files are validated against
`src/bin/roomeq/input_schema.json` before anything runs.

## Key modules

| Module | Owns |
|---|---|
| `roomeq` | `roomeq` subcommands, config/override plumbing, run invocation |
| `convert_recording` | `convert-recording` (measurement format conversion) |

## Core abstractions

- **Validate, then run.** The CLI merges `recordings.json` with
  `--override-config` JSONs, validates the merged `RoomConfig` (schema +
  structural rules), and only then calls `optimize_room`. A bad path or
  out-of-range band fails in seconds, never mid-optimization.
- **Output routing.** `--output` selects the result JSON destination;
  export formats and sidecars fan out from the returned
  `RoomOptimizationResult` via `roomeq-workflow::export` and
  `roomeq-export`.

## API shape

```bash
# Canonical run: config in, result JSON out.
cargo run --release --bin roomeq -- \
  --config tests/data/roomeq/test_config_stereo.json \
  --output /tmp/out.json

# Targeted corpus QA through the same CLI surface:
cargo run --features qa --bin roomeq-qa-acoustic -- \
  --tier nightly --scenario measured_stereo_ascilab1
```

## Consumers

Root binaries `roomeq` and `convert-recording`.
