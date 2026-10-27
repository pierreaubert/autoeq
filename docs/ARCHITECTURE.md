# Software Architecture

High-level view of the `autoeq` workspace: automatic equalization for
speakers, headphones, and rooms. Per-crate details live in each crate's
`ARCHITECTURE.md` (linked below). The virtual workspace and executable
ownership are described in §5.

Design rules that hold everywhere:

- `autoeq_core::iir::Biquad` is the core filter type.
- Filters stay inside trustworthy measurement-data frequency bounds, and no
  finer than the measurement/smoothing resolution warrants.
- Mismatched response grids are aligned or resampled explicitly, never zipped
  by index.
- Relative-to-peak thresholds drive passband detection.
- Minimum-phase/magnitude correction is kept separate from excess-phase,
  time-domain, and spatial-control claims.
- Exported DSP must round-trip: convolution, CamillaDSP, report, and sidecar
  contracts are conformance-tested.

## 1. Crate map

```mermaid
flowchart LR
    subgraph autoeq["AutoEQ stack (speakers / headphones)"]
        direction LR
        core[autoeq-core]
        meas[autoeq-measurements]
        objective[autoeq-objective]
        neutral[autoeq-optim-types]
        opt[autoeq-optim]
        wf[autoeq-workflow]
        cli[autoeq-cli]
        fir[autoeq-fir]
        plot[autoeq-plot]
        art[autoeq-artifacts]
        oracle[autoeq-qa]
        renderer[autoeq-report-wasm]
        shell[autoeq-report-wasm-shell]
        viewer[autoeq-report-gpui]
        meas --> core
        neutral --> core
        cli --> neutral
        wf --> neutral
        plot --> neutral
        opt --> objective
        objective --> core
        objective --> meas
        objective --> neutral
        opt --> core
        opt --> meas
        wf --> opt
        wf --> meas
        cli --> wf
        fir --> core
        plot --> opt
        art --> core
        plot --> renderer
        renderer --> core
        shell --> renderer
        viewer --> renderer
        oracle --> opt
        oracle --> cli
    end
    subgraph roomeq["RoomEQ stack (rooms)"]
        direction LR
        model[roomeq-model]
        analysis[roomeq-analysis]
        dsp[roomeq-dsp]
        engine[roomeq-engine]
        rworkflow[roomeq-workflow]
        quality[roomeq-quality]
        qa[roomeq-qa]
        synth[roomeq-synthetic]
        rcli[roomeq-cli]
        export[roomeq-export]
        model --> neutral
        dsp --> model
        export --> dsp
        export --> quality
        engine --> dsp
        analysis --> model
        engine --> model
        engine --> analysis
        engine --> opt
        engine --> quality
        rworkflow --> engine
        quality --> model
        qa --> rworkflow
        qa --> quality
        qa --> synth
        rcli --> rworkflow
        rworkflow --> export
    end
    engine -.- core
    rworkflow -.- meas
```

The map includes all 24 workspace packages and selected dependency edges.
Local GPUI demos are excluded from this workspace (see below).

Legend: solid arrows point from a crate to its dependencies (opposite to
runtime data flow); dotted lines indicate shared primitives. `roomeq-export` is last because it only serializes already
finalized DSP chains — runtime data flow is §2, where export is step 11.

| Crate | Role | Doc |
|---|---|---|
| `autoeq-core` | Domain + DSP primitives (`Curve`, `Biquad`, `PeqLayout`, `x2peq`) | [link](../crates/autoeq-core/ARCHITECTURE.md) |
| `autoeq-measurements` | Loaders, conditioning, CEA-2034, provenance | [link](../crates/autoeq-measurements/ARCHITECTURE.md) |
| `autoeq-objective` | Prepared evaluation, losses, objective data | [link](../crates/autoeq-objective/ARCHITECTURE.md) |
| `autoeq-optim-types` | Neutral configuration, objective selection, and optimizer evidence | [link](../crates/autoeq-optim-types/ARCHITECTURE.md) |
| `autoeq-optim` | Optimizer backends + registry | [link](../crates/autoeq-optim/ARCHITECTURE.md) |
| `autoeq-workflow` | Speaker/headphone end-to-end runs | [link](../crates/autoeq-workflow/ARCHITECTURE.md) |
| `autoeq-cli` | `autoeq` CLI adapters | [link](../crates/autoeq-cli/ARCHITECTURE.md) |
| `autoeq-fir` | FIR design on curves | [link](../crates/autoeq-fir/ARCHITECTURE.md) |
| `autoeq-plot` | Native HTML/WASM reports | [link](../crates/autoeq-plot/ARCHITECTURE.md) |
| `autoeq-artifacts` | `ArtifactStore` + sidecar naming | [link](../crates/autoeq-artifacts/ARCHITECTURE.md) |
| `roomeq-model` | Config + DSP-chain contracts | [link](../crates/roomeq-model/ARCHITECTURE.md) |
| `roomeq-analysis` | Measurement-analysis primitives | [link](../crates/roomeq-analysis/ARCHITECTURE.md) |
| `roomeq-dsp` | Shared path-free routing and realization | [link](../crates/roomeq-dsp/ARCHITECTURE.md) |
| `roomeq-engine` | In-memory execution, DSP kernels, pipeline order | [link](../crates/roomeq-engine/ARCHITECTURE.md) |
| `roomeq-workflow` | `optimize_room`, topology routing, finalization | [link](../crates/roomeq-workflow/ARCHITECTURE.md) |
| `roomeq-export` | CamillaDSP/APO/EasyEffects/… exporters | [link](../crates/roomeq-export/ARCHITECTURE.md) |
| `roomeq-quality` | Independent scoring, corpus, gates | [link](../crates/roomeq-quality/ARCHITECTURE.md) |
| `roomeq-qa` | Regression matrices + runners | [link](../crates/roomeq-qa/ARCHITECTURE.md) |
| `roomeq-synthetic` | Deterministic ground-truth curves | [link](../crates/roomeq-synthetic/ARCHITECTURE.md) |
| `roomeq-cli` | `roomeq` CLI surface | [link](../crates/roomeq-cli/ARCHITECTURE.md) |
| `autoeq-qa` | Numerical oracles, negative controls, and optimizer benchmarks | [link](../crates/autoeq-qa/ARCHITECTURE.md) |
| `autoeq-report-wasm` | Shared payload schema, report assembly, and drawing geometry | [link](../crates/autoeq-report-wasm/ARCHITECTURE.md) |
| `autoeq-report-wasm-shell` | Terminal WASM/Canvas export shell | [link](../crates/autoeq-report-wasm-shell/ARCHITECTURE.md) |
| `autoeq-report-gpui` | Host explorer model and optional wasm32 GPUI viewer | [link](../crates/autoeq-report-gpui/ARCHITECTURE.md) |

The separately excluded [GPUI demos](../crates/autoeq-gpui-examples/ARCHITECTURE.md)
use their own manifest and lockfile. Root workspace commands do not select them.

## 2. RoomEQ data flow, step by step

Entry: `roomeq --config <recordings.json> --override-config <optimizer.json>`
(input contract: `crates/roomeq-cli/schemas/INPUT_FORMAT.md` + `input_schema.json`).

```mermaid
flowchart TD
    A["1. Intake\nrecordings.json + optimizer JSON\nvalidated against roomeq-model contracts"] --> B["2. Load + condition\nreaders → Curve on shared grid\nsmooth, clamp, normalize"]
    B --> C["3. Targets\nper-channel target from measurement\naverage + tilt, Schroeder-aware band"]
    C --> D["4. Route selection\ngeneric / home-cinema /\nbass-managed / multi-sub"]
    D --> E["5. Per-channel optimization\nbackend from registry over 20..max_freq\nPEQ solution vectors via x2peq"]
    E --> F["6. Post-optimization stages\ntime + spectral alignment\ntimbre matching, phase / group-delay,\nFIR + mixed-phase generation"]
    F --> G["7. Safety + acceptance\nseat replay at held-out positions\nelectrical headroom: unit-peak inputs,\n0 dBFS ceiling, attenuation as safety gain\ncorrection-acceptance policy"]
    G --> H["8. Final level alignment\ninter-channel trims\nupper-band target reference"]
    H --> I["9. Realize + verify\nRealizedDsp applies the exact deployed chain\nwhat is scored is what ships"]
    I --> J["10. Score\nroomeq-quality scorecard + robustness\nbaseline comparison, gate verdict"]
    J --> K["11. Emit\nresult JSON + DSP exports\n+ sidecars + probe WAVs"]
```

Notes on the flow:

- Steps 5–8 iterate per channel inside `roomeq-workflow::optimize_room`;
  the canonical step order is `PipelineStepId::ALL` in
  `roomeq-engine/src/pipeline.rs`.
- The `final_curve` in the result JSON already includes deployment gains
  (balance trims, `final_electrical_headroom` attenuation), so "After EQ"
  plots sit below the raw measurement by exactly those flat stages — the
  PEQs themselves contribute ~0 dB outside the optimization band.
- Held-out seats are never training inputs; enforced corpus scenarios need
  ≥ 2 of them covering every scored channel.

## 3. AutoEQ (speaker/headphone) data flow

```mermaid
flowchart TD
    A["Measurements\nCSV / Spinorama API / recording"] --> B["autoeq-measurements\ncondition + CEA-2034"]
    B --> C["autoeq-optim\nloss: flat / score / epa / drivers / multi-sub\nbackend: de / cmaes / cobyla / ..."]
    C --> D["x2peq + response\nsolution vectors → Biquad sets"]
    D --> E["autoeq-workflow\nreport + resume"]
```

## 4. QA flow

```mermaid
flowchart TD
    M["data_tests/roomeq/\nacoustic_corpus manifest + baselines\nmeasured captures, FEM fixtures"] --> R["roomeq-qa runners\nacoustic / quality / synthetic /\nfeatures / coverage / fuzzer"]
    R --> Q["roomeq-quality scoring\nscorecards, robustness, gates"]
    Q --> V{"gate verdict"}
    V -->|pass| P["candidate promotion"]
    V -->|fail| F["report-only advisory\nor hard failure by tier"]
```

- PR tier runs the fast subset; nightly/weekly run the full corpus.
- Baselines are platform-scoped (OS + arch); numeric drift across platforms
  is expected and handled by recalibration, not by loosening gates.
- Corpus intake rules (opaque IDs, no personal data, rights classification)
  are documented in `data_tests/roomeq/acoustic_corpus/PROVENANCE.md`.

## 5. Virtual workspace and executable ownership

The root has no package or facade sources. Libraries are the default members;
CLI, QA, and report packages are selected explicitly. Dependency policy rejects
root-package resurrection, production-to-terminal edges, and normal/dev/build
cycles. Terminal/tool/report-shell packages set `publish = false`.

| Package | Executables |
|---|---|
| `autoeq-cli` | `autoeq`, `benchmark-autoeq-speaker`, `autoeq-download-speakers` |
| `roomeq-cli` | `roomeq`, `convert-recording` |
| `roomeq-qa` | Fuzzer and the five `roomeq-qa-*` runners |
| `autoeq-measurements` | `migrate_provenance` |

RoomEQ schemas and their input manual belong to `roomeq-cli/schemas/`.
Integration tests import production owners and live with their subject or a
terminal QA owner. Native report assets remain with `autoeq-report-wasm`.

Use `just test-crate <package>` for focused feedback, `just test-cli` for
executable contracts, and `just ntest` for the complete native inventory.

## 6. Further reading

- `docs/ROOMEQ_MANUAL.md` — user manual; `docs/ROOMEQ_101.md` — concepts.
- `crates/roomeq-cli/schemas/INPUT_FORMAT.md` — input contract.
- `data_tests/roomeq/acoustic_corpus/PROVENANCE.md` — corpus intake rules.
- `CHANGELOG.md` — behavioral history.
