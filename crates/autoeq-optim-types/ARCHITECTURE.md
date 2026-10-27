# autoeq-optim-types architecture

- `config`: parser-independent `OptimizationConfig` values and unchanged
  plain/speaker/headphone/room defaults. Version/list/preset-display controls
  remain in terminal `autoeq-cli::cli::Args`.
- `loss`: neutral objective-selection enum, also re-exported by the objective
  and backend libraries for their API signatures.
- `evidence`: Pareto dispatch and input-normalization evidence types
  with their validation logic. Produced by `autoeq-optim` backends,
  consumed by `roomeq-engine` and `roomeq-model` report assembly.
- `reference_registry`: approved independent-reference entries and
  verified-evidence checks used by perceptual promotion gates in
  `autoeq-optim`, `roomeq-quality`, and `roomeq-qa`.

Dependencies: `autoeq-core`, `serde`, and `schemars`. This crate must stay
below both the optimizer and the RoomEQ model in the dependency
graph; its resolved normal/build closure excludes Clap, solver execution,
RoomEQ model/engine/workflow, and terminal packages.
