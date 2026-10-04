# roomeq-export — Architecture

Canonical RoomEQ DSP graphs to external audio processing formats, plus
packaging, conformance checks, and round-trip tests.

## Layer

Presentation/export leaf over `roomeq-model` + `roomeq-engine`. Never invents
correction; it serializes the already-finalized `ChannelDspChain`s.

## Supported targets (`lib.rs`)

CamillaDSP (YAML) · Equalizer APO / Peace (text) · EasyEffects (JSON) ·
Wavelet (GraphicEQ text) · PipeWire filter-chain (SPA-JSON) · Roon DSP Engine
(JSON) · REW Generic EQ (reference text for manual entry) · canonical normalized
biquad coefficients (JSON). REW cannot reload Generic EQ text; its saved filter
format is binary `.req`, which this crate does not generate.

## Key modules

| Module | Owns |
|---|---|
| `channel`, `collect`, `extract` | Per-channel plugin harvesting into export rows |
| `format`, `export_format`, `write` | Target renderers and file writers |
| `conformance` | Per-target rule checks (what each engine can express) |
| `roundtrip` (+ `tests/`) | Internal parse-back verification of supported formats; external consumer import requires separate evidence |
| `package` | Multi-file export bundles |
| `delay`, `hash` | Delay encoding, artifact identity |
| `pipewire`, `roon_convolver` | Target-specific adapters |

## Core abstractions

- **`DspGraph` in, text out.** `render_dsp_graph(&graph)` produces a target
  document; `build_export_package(&graph, …)` bundles multi-file outputs
  (e.g. YAML + FIR WAVs) with `convolution_resource_references` tracking
  every sidecar a bundle needs — `checked_…` variants fail instead of
  emitting dangling references.
- **Conformance before bytes.** Each target declares what it can express
  (filter counts, delay ranges, gain limits); `conformance` rejects or
  degrades chains a target cannot represent, so an export is never silently
  lossy.
- **Round-trip tests.** Every renderer has a parse-back test: export →
  re-import must null against the in-memory chain within tolerance.

## API

```rust
// Render one target (format selects the renderer; unsupported graphs fail):
let camilla_yaml: String =
    render_dsp_graph(&graph, ExportFormat::CamillaDsp, sample_rate)?;
let biquads: String =
    render_dsp_graph(&graph, ExportFormat::BiquadCoefficients, sample_rate)?;
// Bundle everything a deployment needs (in-memory; the crate never touches fs):
let package = build_export_package(&graph, format, &main_name, sample_rate, &resources)?;
let refs: Vec<String> = checked_convolution_resource_references(&graph)?;

// Package sidecars (FIR WAVs) alongside their manifests:
package_convolution_sidecars(&graph, &store, &out_dir)?;
```

## Contract invariant

Preserve the convolution, CamillaDSP, report, and sidecar contracts:
exported DSP must null against the in-memory realization within the
round-trip tolerance.
