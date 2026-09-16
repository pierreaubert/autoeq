# roomeq-engine — Architecture

In-memory RoomEQ execution and deterministic processing.
Owns the pipeline vocabulary (`PipelineStepId::ALL` canonical order) and the
DSP kernels; knows nothing about filesystems, caches, or artifact stores.

## Layer

Execution boundary. `RoomEngine::run` takes a prepared `EngineRequest`
(resolved `RoomConfig` + sample rate + probe arrivals) plus an observer and a
processing kernel supplied by the workflow layer; as vertical slices migrate,
that kernel shrinks until execution is wholly engine-owned.

## Canonical stage order (`pipeline.rs`)

ConfigPreparation → Validation → TopologyRouteSelection →
TopologyWorkflowExecution → GenericChannelOptimization → FirGeneration →
MixedPhaseFirGeneration → PhaseCorrection → TimeAlignment →
SpectralAlignment → InterChannelTimbreMatching → HeightChannelAlignment →
PhaseAlignment → GroupDelayOptimization → ImpulseResponseComputation →
ChannelMatching → MetadataRefresh → SanityCheck.

## Key modules

| Module | Owns |
|---|---|
| `channel_optimizer`, `channel_iir`, `channel_fir` | Per-channel PEQ/FIR optimization and realization |
| `channel_measurements`, `channel_target`, `channel_preprocessing` | Measurement conditioning, target derivation, broadband pre-alignment |
| `dsp_realization` | `RealizedDsp`: applies a `ChannelDspChain` to curves (what you hear is what is scored) |
| `eq`, `crossover` | Filter application, crossover design |
| `bass_management/`, `topology/` | Routed bass workflows; `align_channels_to_lowest` inter-channel trims |
| `spectral_align` | Broadband shelf/gain alignment, upper-band target reference |
| `supporting_source/` | Brooks–Park supporting-loudspeaker compensation kernels |
| `ctc`, `dba` | Crosstalk-cancellation and double-bass-array paths |
| `excursion` | Driver-excursion safety checks |
| `loss` (re-export) | `autoeq-optim` losses used in-room |
| `pipeline` | `EngineRequest`, `RoomEngine`, `PipelineStepId`, observers |

## Core abstractions

- **`RoomEngine` + `EngineRequest<'a>` + kernel.** The engine never touches
  disk: the workflow resolves files, builds the request, and hands a
  `FnOnce(EngineRequest, observer) -> Result<T>` kernel. Progress UIs observe
  via `PipelineObserver` keyed by `PipelineStepId`.
- **`RealizedDsp`.** `RealizedDsp::new(&chain, sample_rate, &mut NoConvolutionIr)`
  then `apply_to_curve(&base)` replays the *exact deployed plugin chain*
  (PEQs, gains, delays, crossovers) onto a measurement. Scorecards, seat
  replay, and headroom audits all read through this — never through
  optimizer-internal caches.
- **Gain-plugin vocabulary.** Flat corrections travel as `gain` plugins with
  machine-readable flags: `room_eq_correction_gain` (part of the correction,
  removed from baselines on replay), `room_eq_safety_gain` +
  `label = "final_electrical_headroom"` (anti-clip attenuation under the
  0 dBFS ceiling), `label = "final_channel_level_alignment"` (inter-channel
  trims from `align_channels_to_lowest`: every channel gets
  `lowest_mean − own_mean`, i.e. normalize *down* to the quietest channel).
- **Separation of correction kinds.** Minimum-phase PEQs, excess-phase/group-
  delay all-pass paths, and spatial (CTC/DBA/supporting-source) processing
  are distinct stages with distinct evidence — a magnitude win can never
  silently claim a time-domain win.

## API

```rust
// In-memory execution: no paths, no stores.
let engine = RoomEngine;
let result: RoomOptimizationResult = engine.run(
    EngineRequest { config, sample_rate, probe_arrival_overrides: None },
    observer,
    // Workflow-supplied kernel: stages not yet migrated into the engine.
    |request, observer| run_channel_stages(request, observer),
)?;

// Replay the deployed chain onto a raw measurement:
let mut realized = RealizedDsp::new(&chain, sample_rate, &mut NoConvolutionIr)?;
let after_eq: Curve = realized.apply_to_curve(&before_eq)?;

// Match a channel group down to its quietest member (means over band):
let trims: HashMap<String, f64> =
    topology::align_channels_to_lowest(&curves, &ranges);
```

## Consumers

`roomeq-workflow` (supplies the kernel), `roomeq-qa`, `roomeq-export`,
`roomeq-cli`.
