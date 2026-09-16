# roomeq-workflow — Architecture

RoomEQ application workflows and resource adapters.
Entry point is `optimize_room`: load config → run engine kernel → finalize →
export. Owns everything the engine deliberately does not: files, caches,
sidecars, progress reporting.

## Layer

Orchestration over `roomeq-engine` + `roomeq-model` + `roomeq-export`.
The root `src/roomeq/` tree is a thin compatibility facade over this crate's
partitions.

## Key modules

| Module | Owns |
|---|---|
| `room_optimization.rs` + `room_optimization/` | `optimize_room`; per-channel coordination, `seat_replay` (final-seat verification), `finalization` (electrical-headroom attenuation, correction acceptance), `phase/` + `gd/` (excess-phase / group-delay paths), `reports/`, `validation_scorecard` |
| `topology/` | Route selection: `generic`, `home_cinema`, `bass_management`, `multisub` runners |
| `group_measurements`, `measurement`, `channel_measurements` | Multi-seat / multi-measurement aggregation (weighted, minimax, variance-penalized, spatial-robustness, modal-basis strategies) |
| `supporting_source.rs` | Supporting-source workflow wiring (kernels in `roomeq-engine`) |
| `home_cinema/` | Routed home-cinema workflow logic |
| `eq`, `fir`, `eq_resources` | Filter resources for the optimization passes |
| `export`, `output`, `sidecar`, `wav` | Result serialization, DSP export calls, sidecars, probe WAVs |
| `config_loader`, `executor`, `pipeline` | Config intake, threaded execution, observer/progress plumbing |
| `electrical_headroom` | Unit-peak / 0 dBFS-ceiling verification of the deployed graph |
| `arrival`, `ctc`, `dba`, `cea2034`, `multisub` | Arrival estimation, CTC/DBA wiring, spinorama inputs, multi-sub flows |

## Core abstractions

- **`optimize_room(config, sample_rate, callback, output_dir)`.** The single
  production entry (plus `optimize_room_with_probe_arrivals` when a UI step
  measured tone-burst arrivals). It builds a `RoomPipeline`, runs the kernel
  through the engine boundary, and returns a `RoomOptimizationResult`.
- **Finalization (`room_optimization/finalization.rs`).** After per-channel
  optimization: `install_attenuation` inserts `final_electrical_headroom`
  safety gains so corrected peaks stay under the output ceiling;
  `correction_acceptance` applies the runtime-safety policy (residual caps,
  boost caps, latency/ringing/group-delay budgets) and can only accept or
  degrade — never silently extend the PEQ band.
- **Seat replay (`seat_replay`).** The deployed graph is re-simulated at
  held-out seats; `final_curve`s already include balance trims and headroom
  gains, which is why plotted "After EQ" sits below the raw measurement by
  exactly those flat stages.
- **Multi-seat strategies** (`minimize_variance`, `primary_with_constraints`,
  `average`, complex-domain `modal_basis` SFM) and **multi-measurement**
  strategies (weighted/minimax/variance-penalized, spatial robustness) select
  how seats combine into one correction.

## API

```rust
// Full run from resolved config:
let result: RoomOptimizationResult =
    optimize_room(&config, 48_000.0, Some(callback), Some(&out_dir))?;

// With UI-measured probe arrivals instead of WAV-onset fallback:
let result = optimize_room_with_probe_arrivals(
    &config, 48_000.0, None, Some(&out_dir), &probe_arrival_ms,
)?;

// Intake helpers used before the run (base + optional override merge,
// returning the config, its resolved dir, and a validation report):
let (config, dir, report) = load_config(&recordings_path, Some(&override_path))?;
let merged = load_merged_config_strict(&recordings, &overrides)?;
```

## Data contracts

- **In:** `recordings.json` + optimizer JSONs (see `src/bin/roomeq/INPUT_FORMAT.md`).
- **Out:** `RoomOptimizationResult` (channels, DSP chains, curves, metadata,
  stage outcomes) → JSON + export formats + sidecars.

## Consumers

`roomeq-cli`, `roomeq-qa`, the `roomeq` binary.
