# AutoEQ / RoomEQ virtual workspace

Core automatic equalization stack for speakers, headphones, and RoomEQ workflows.

## High-level workflow

1. `autoeq-measurements`: ingest and normalize measurements (CSV / API / capture artifacts)
2. `autoeq-objective`: evaluate losses (`flat`, `score`, `epa`, multi-driver, multi-sub)
3. `autoeq-optim`: run optimizer backends over PEQ parameters
4. `autoeq-core`: turn parameter vectors into filters and responses
5. `roomeq-workflow`: multi-channel orchestration, reporting, routing, and export

## Loss types

- `speaker-flat`
- `speaker-flat-asymmetric`
- `speaker-score`
- `headphone-flat`
- `headphone-score`
- `drivers-flat`
- `multi-sub-flat`
- `epa`

## Optimizer backends

Registry path: `crates/autoeq-optim/src/optim/registry.rs`.

- `autoeq:de` (DE, supports JADE/L-SHADE strategy variants)
- `autoeq:cobyla` (pure-Rust COBYLA)
- `autoeq:isres` (pure-Rust ISRES)
- `autoeq:cmaes` (CMA-ES)
- `autoeq:nsga2`, `autoeq:nsga3` (Pareto MO optimizers)
- `mh:*` backends (e.g. `mh:pso`, `mh:firefly`, etc.)

Notes:
- The old C-FFI NLopt backend is removed. `nlopt:*` names are kept as compatibility aliases and are mapped to pure-Rust backends with warnings.
- Common bare aliases (`de`, `cobyla`, `isres`, `cma-es`, `nsga-ii`, `nsga-iii`) resolve through the registry.

## Key objective controls

- `ObjectiveData` carries single-curve and multi-objective paths.
- `multi_objective` delegates scalarization over per-curve objectives.
- The scalar objective is selected through the `Objective` strategy trait
  (`crates/autoeq-objective/src/optim/loss/`). `ObjectiveData` caches the strategy
  so it is built once.
- `PeqLayout` (`crates/autoeq-core/src/param_utils.rs`) abstracts the
  parameter-vector layout for
  each `PeqModel`; filter extraction/encoding and initial-guess generation
  delegate to the trait instead of repeating `match peq_model` blocks.
- `smoothness_penalty` (TV² curvature regularizer in log-frequency) is optional and supported in PEQ-based branches.
  - Config fields: `tv2_weight`, `schroeder_hz`, `modal_weight_scale`, `exponent`.
  - CLI flags: `--smoothness-weight`, `--smoothness-exponent`, `--smoothness-schroeder-hz`, `--smoothness-modal-scale`.
  - RoomEQ JSON path: `optimizer.smoothness_penalty`.

## RoomEQ highlights

Production code is partitioned across `crates/roomeq-{model,analysis,quality,dsp,engine,export,workflow}`.
The root is a virtual workspace; import the owning crates directly.

- Multi-seat strategies include:
  - `minimize_variance`
  - `primary_with_constraints`
  - `average`
  - `modal_basis` (complex-domain SFM modal-basis optimization)
- Multi-measurement strategies include weighted/minimax/variance-penalized paths and spatial robustness mode.
- Supporting-source room compensation (Brooks-Park): a delayed, decorrelated
  supporting loudspeaker fills reverberant energy without altering the primary
  source's direct sound. Config type: `SpeakerConfig::SupportingSource`;
  implementation lives under `crates/roomeq-engine/src/supporting_source/` and
  `crates/roomeq-workflow/src/supporting_source.rs`.
- Bass-management and routed workflow logic live under
  `crates/roomeq-engine/src/bass_management/` and
  `crates/roomeq-workflow/src/{home_cinema,topology}/`; export conformance
  lives in `crates/roomeq-export/`.
- Single-speaker processing is coordinated by
  `crates/roomeq-workflow/src/room_optimization.rs`, with DSP kernels owned by
  `roomeq-engine`.

## Binary entry points

Executables belong to terminal packages and need no root-package feature:

- `autoeq-cli`: `autoeq`, `benchmark-autoeq-speaker`, `autoeq-download-speakers`.
- `roomeq-cli`: `roomeq`, `convert-recording`; schemas and input manual live in
  `crates/roomeq-cli/schemas/`.
- `roomeq-qa`: `roomeq-fuzzer`, `roomeq-qa-quality`, `roomeq-qa-coverage`,
  `roomeq-qa-features`, `roomeq-qa-synthetic`, `roomeq-qa-acoustic`.
- `autoeq-measurements`: `migrate_provenance`.

User-facing launcher versions use `autoeq_core::PRODUCT_VERSION`; serialized
library and optimizer versions keep their existing identities.

## Useful commands

Bare `cargo build/test/check` selects the workspace `default-members`
library set; executable builds select `autoeq-cli`, `roomeq-cli`, or `roomeq-qa`.

```bash
cargo check -p autoeq-optim
cargo clippy -p roomeq-workflow --no-deps
cargo test -p autoeq-core --lib
```

Test lanes (Cargo profiles `test`/`qa`, nextest profiles in
`.config/nextest.toml`):

```bash
just test                 # default-member libraries, fast feedback
just test-crate <package> # one package and its owning tests
just test-pr [base]       # changed packages + reverse dependencies
just test-cli             # real executable tests on the qa profile
just ntest                # complete native inventory (unchanged scope)
just qa-autoeq-quick      # bounded flat+score AutoEQ smoke
```

RoomEQ QA:

```bash
just qa-roomeq
just qa-roomeq-convergence
just qa-roomeq-coverage-gate  # hard 90% library line-coverage gate
```

Targeted RoomEQ run:

```bash
cargo run -p roomeq-cli --release --bin roomeq -- \
  --config tests/data/roomeq/test_config_stereo.json \
  --output /tmp/out.json
```

## Docs to keep in sync

- `CHANGELOG.md`
- `docs/ROOMEQ_MANUAL.md`
- `crates/roomeq-cli/schemas/INPUT_FORMAT.md`
- `crates/roomeq-cli/schemas/input_schema.json`
