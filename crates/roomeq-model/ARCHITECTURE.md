# roomeq-model — Architecture

Stable RoomEQ configuration and output contracts.
Every RoomEQ crate speaks these types; nothing here executes DSP.

## Layer

Contract crate: `autoeq-core` types plus room-specific config/output schemas
(`serde` + `schemars`; the CLI validates `input_schema.json` against them).

## Key modules

| Module | Owns |
|---|---|
| `config/` | `RoomConfig`: speakers, measurements, optimizer settings, bass management, overrides (≈50 focused `*_config.rs` files: `optimizer_config`, `speaker_config`, `bass_management_config`, `multi_seat_config`, `target_response_config`, `finalization_config`, …) plus `room_config_builder` / `optimizer_config_builder` |
| `optimizer_settings` | Algorithm/band/bound/seed/robustness options (incl. `smoothness_penalty`) |
| `output` | `ChannelDspChain`, `DspGraph`, plugin wrappers — the deployed-DSP description |
| `contracts`, `validation_rules/` | Intake validation: configs are rejected before any optimization runs |
| `preset`, `auto_tune` | Named starting points and automatic config derivation |
| `target_tilt`, `home_cinema*` | Target-curve tilt; home-cinema layout resolution |
| `report_contracts` | Result/scorecard JSON schemas |
| `ir_waveform`, `rir_prototype_config`, `physical_routing` | Time-domain and routing descriptors |

## Core abstractions

- **`RoomConfig`.** The single intake type: `recordings.json` (speakers →
  measurement files) merged with the optimizer JSON (algorithm, `min_freq` /
  `max_freq`, filter counts/bounds, seeds, bass management, topology). Built
  via `room_config_builder`, validated by `validation_rules`.
- **`ChannelDspChain`.** Ordered plugin list per channel (`eq` with `Biquad`
  rows, `gain` with headroom/balance metadata, `delay`, `crossover`,
  convolution). This is what ships: exporters serialize it, `RealizedDsp`
  executes it, QA replays it.
- **`RoomOptimizationResult`.** Channels (`initial_curve` / `final_curve` /
  `target_curve` per channel) + `metadata` (effective config, stage outcomes,
  correction acceptance, bass-management graph). The `dsp-*.json` files are
  this type serialized.

## API shape

```rust
// Intake: merge + validate before any DSP runs.
let config: RoomConfig = load_merged_config_strict(&recordings, &overrides)?;
config.validate_version()?; // schema version first, then structural invariants
config.validate_structure()?; // validation_rules: fail fast on bad intake

// Inspect what will be deployed.
for (name, chain) in &result.channels {
    for plugin in &chain.plugins {
        println!("{name}: {} {:?}", plugin.plugin_type, plugin.parameters);
    }
}
```

## Consumers

All `roomeq-*` crates; `src/bin/roomeq/input_schema.json` mirrors these types.
