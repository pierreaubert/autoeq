# RoomEQ - Multi-channel Room Equalization Optimizer

`roomeq` is a command-line tool for optimizing multi-channel speaker systems. It analyzes frequency response measurements and generates optimal DSP chains (EQ, crossovers, gains) for each channel.

## Final convolution resource integrity

Best-effort routed snapshots and correction replay treat a reviewed residual
splice dip as a source-specific exception. They continue checking other logical
inputs and record each retained residual separately. One reviewed source never
exempts the remaining mains from validation. These intermediate exceptions do
not relax the strict final crossover-safety gate or establish final acceptance.

When combined-boost limiting changes a channel's PEQs, its original optimizer
run remains recorded but is no longer marked selected for output. The applied
limiting stage records the gain scale and boost limit separately.

Successful limiting also updates the cached biquad coefficients used by IR
reporting. Previous waveform and early/late reports are invalidated until they
are rebuilt. If phase evidence is unavailable at refresh, the waveform pair is
absent rather than retained from an older measurement or correction. These are
model-derived waveform reports, not a replacement for measured temporal or
listening evidence.

Combined-boost limiting commits a changed channel only after successful DSP
replay. Replay failure retains both its original chain and cached response;
a failed replay is not evidence that the channel meets the boost limit.

When final-seat shape and useful-output checks both fail, the rejection report
retains both violation codes and the error includes both diagnostics. A shape
failure does not suppress evidence of lost output.

Legacy MSO with multiple measured seats now retains one coherent combined
response per seat after selected sub gains/delays for shared sub EQ. The
configured multi-measurement strategy receives those responses. Routing keeps
its separate complex representative for crossover timing. Shared-EQ support
is restricted to common measured support; missing phase or inconsistent
seat identities/counts cannot be replaced by an invented coherent response.
This does not change legacy MSO's primary-seat gain/delay search into a
multi-seat alignment optimizer; use the explicit multi-seat mode for that.

If correction-safety rollback breaks routed playback, the workflow fails.
The restored pre-gate graph is retained only as diagnostic state: restoring
routing does not prove that the rejected correction is safe. No validation
bundle is newly published for this failed workflow.

Final-seat quality rejection sets correction acceptance to `accepted: false`
and `decision: "rejected"`, retaining seat evidence and violation reasons.
The workflow returns an error; this decision does not claim that a rollback or
identity fallback was executed. Consumers of the output decision enum must
handle the additive `rejected` value.

Channel-matching EQ has explicit logical-input (`pre_route`) ownership, so routed
playback applies it to both the main and redirected-bass branches. Deployed-source
curve caches follow the same correction and are restored if matching is discarded.

Routed finalization and direct physical replay reject channel plugins without a
recognized `pre_route`, `post_route` or `route_owned` stage. CamillaDSP's public
export validator already enforces this requirement. Driver plugins are owned by
their explicit physical branch; non-routed chains do not need routing stage tags.

Post-EQ acceptance checks useful-output loss across the representative stage's
declared band, including bass below the mains scoring band. A rejected candidate
is discarded through the existing stage path, preserving previous correction,
routing, gains and delays; its optimizer evidence is not selected for output.
Stage metadata records output-loss rejections. This necessary stage guard does
not replace final native-seat checks for cumulative or position-specific loss.

Routed graphs keep logical source inputs separate from physical output-only sub
channels. Adding independently controlled sub outputs extends the destination
list, not the list of signals to replay or capture. Route source indices address
the logical input list; destination indices address the physical output list.

Enabled validation bundles are written only after final-seat validation succeeds.
They include the workflow sample rate, requested optimizer configuration, final
DSP graph with acceptance/seat evidence and resource identities, and final
combined scores. The embedded graph omits the bundle's own just-created report
pointer. A workflow error does not publish a new bundle; it does not delete an
older bundle already present in a reused output directory. These descriptors
are not rendered, level-matched listening assets or proof of perceptual benefit.

Finalization and packaging require every global, channel and driver convolution
stage to declare a string `ir_file` that is nonblank and contains no NUL byte.
Malformed declarations are errors, not absent resources; valid references to
unavailable files remain explicitly unbound at finalization.

Before binding available resources, finalization decodes every referenced WAV
at the workflow sample rate and requires nonempty, equal-length channels with
finite samples. This applies even without retained coefficient ownership, such
as driver or multi-FIR resources. It validates WAV data, not equivalence to
an accepted acoustic transfer; retained-tap comparisons remain ownership-scoped.

Completed workflows record `metadata.final_convolution_sha256` after the final
DSP stages, using the configured artifact store. Each reference maps to its
SHA-256, or `null` when the resource was unavailable at finalization. Packaged
exports verify this inventory before writing and retain it when sidecars are
renamed. Changed, missing, or unbound resources require a new validated workflow
result; exporting must not silently assign replacement files a new identity.
The export API without sidecar verification rejects bound convolution graphs.

Older/manual graphs without this optional field remain readable and exportable,
but do not carry this final-workflow integrity guarantee. The inventory records
byte identity, not acoustic acceptance, audibility, or protection against an
actor who edits both the file and its recorded digest.

Retained channel FIR coefficients are associated only with that channel's
single declared channel-level convolution reference. A driver's separate FIR
must use its own sidecar; it is neither compared against unrelated parent taps
nor replaced with them when missing. This ownership check does not infer
coefficient ownership for arbitrary multi-FIR chains.

## CamillaDSP delay realization


CamillaDSP exports preserve integer delays in samples. Fractional delays use
a causal FIR rather than silently rounding to whole samples. Every parallel
branch receives the same required stage padding, preserving relative timing.
The YAML `roomeq_delay_realization` declaration records per-stage padding,
total additional latency, sample rate, and usable frequency band. The same
typed report is available through the export API and packaged artifact; the
workflow logs it when fractional-delay processing is present.

The fractional-delay kernel is specified through 0.46 × sample rate, with
at most 0.01 dB magnitude error; Nyquist is not certified. Added backend
padding is **in addition to** requested delays, existing correction-FIR
latency, and device buffering. It is not evidence of acoustic improvement,
transient headroom, or listener preference. The required matrix backend QA
checks sampled electrical transfer independently of optimization scores.

### Physical-routing API

`roomeq-model::PhysicalRoutingGraph` is the backend-neutral contract for resolved
input-to-physical-output routing. Input plugins run before fan-out; every route
retains its own gain, polarity, crossover family/frequency and delay. Contributions
sum at each physical output before its ordered output plugins run. Routes may be
canonicalized for deterministic serialization, but distinct transfers are never
merged solely because they share an input. Port names are graph identities, not
sound-card assignments.

`roomeq-engine::physical_routing::resolve_physical_routing` translates supported
legacy home-cinema channel/driver ownership into this contract. Shared sub EQ is
assigned to each physical sub, while driver gain/delay/polarity already included
in route values is not applied again. Unresolved ownership is an error. Electrical
diagnostics and the sibling SOTF native adapter use this resolver. Native lowering
preserves signed gains and uses explicit wet-only, zero-feedback alignment delays.
Unsupported crossover families are rejected rather than replaced with LR24.
Local PCM regressions cover integer delays, LR24 crossover transfer, shared EQ,
and route-order invariance at 44.1/48/96 kHz and multiple block sizes. Deployment
still requires the matching AutoEQ dependency revision. Arbitrary fractional-delay
and FIR conformance, device routing, acoustic improvement, and transient/continuous
headroom are not certified by those tests or by model replay alone.

This adds a library contract, not a new RoomEQ input configuration field or a
replacement of the existing exported `DspGraph` JSON schema. AutoEQ has no new
dependency on SOTF's player, engine, or plugins.

## Features

- **Single speaker optimization**: Optimize EQ for individual speakers
- **Multi-driver crossover optimization**: Optimize crossovers for multi-driver speakers (woofer + tweeter, etc.)
- **Group delay alignment**: Optimize time alignment between subwoofers and main speakers
- **Multiple optimization algorithms**: Support for COBYLA, Differential Evolution, and other optimizers
- **AudioEngine-compatible output**: Generates JSON DSP chains compatible with the AudioEngine plugin system
- **Target curve tilt**: Harman-style tilted target curves with optional bass shelf
- **Excursion protection**: Automatic F3 detection and highpass filter generation
- **Schroeder frequency split**: Separate EQ strategies for modal and statistical room behavior
- **Phase alignment**: Subwoofer/speaker phase and polarity optimization
- **Multi-seat optimization**: Minimize response variance across multiple listening positions
- **Supporting-source room compensation**: Delayed, decorrelated supporting loudspeaker to fill reverberant energy while preserving the primary source's direct sound

## Usage

```bash
cargo run --features cli --bin roomeq -- --config <config.json> --output <output.json> [OPTIONS]
```

### Options

- `--config <CONFIG>`: Path to room configuration JSON file (required)
- `--output <OUTPUT>`: Path to output DSP chain JSON file (required)
- `--sample-rate <RATE>`: Sample rate for filter design (default: 48000 Hz)
- `--freq-samples <N>`: Number of log-frequency points used when reducing dense
  measurements for interpolation (default: 200)
- `--export-format <FORMAT>`: Also export `camilladsp`, `apo`, `easyeffects`,
  `wavelet`, `pipewire`, `roon`, `rew`, or `coefficients`
- `--export-path <PATH>`: Override the derived external-export path
- `--convert <DSP_CHAIN_JSON>`: Convert an existing RoomEQ DSP chain without
  running optimization
- `--verbose`: Enable verbose output
- `--help`: Print help information

`rew` emits a single-channel REW Generic EQ filter-settings file.
`coefficients` emits normalized `a0=1, a1, a2, b0, b1, b2` sections using the
same canonical biquad implementation as runtime DSP. Both formats reject
convolution, crossovers, routing, or unknown stages instead of silently
dropping them.

## Choosing a DSP Target

RoomEQ has a canonical JSON output and several external exporters. The
canonical JSON is the most complete description of the result; an external
export is a translation into the target's DSP model, not a guarantee that every
RoomEQ feature remains representable.

| Target | Best use | Preserves | Important limitations |
|--------|----------|-----------|-----------------------|
| **RoomEQ JSON / AudioEngine** | Full-fidelity RoomEQ or an AudioEngine-compatible host | Per-channel gain/EQ/delay, multi-driver crossovers, FIR convolution, mixed phase, global matrices, routed bass management, and route metadata | The consumer must implement the RoomEQ output schema, plugin types, channel ordering, sidecar FIR files, and graph routing. |
| **CamillaDSP (`camilladsp`)** | Multichannel playback with subwoofer/bass-management routing | Routed bass-management graphs, channel mixing, serial filters, and convolution sidecars when the generated paths are available | Requires correct channel numbering, sample rate, sidecar WAV paths, and CamillaDSP configuration. Unsupported RoomEQ plugin features or graph details cannot be carried over automatically. |
| **Equalizer APO / Peace (`apo`)** | Windows playback with serial filters and representable channel routing | APO-compatible gain/EQ stages and routing that fits APO's channel model | It is not a general RoomEQ graph host. Complex fan-out, unsupported plugins, FIR packaging, or unusual channel layouts may be rejected or simplified; verify the generated channel mapping. |
| **EasyEffects (`easyeffects`)** | Linux desktop stereo or simple per-channel correction | Serial single-channel-compatible gain and EQ | No full bass-management graph, channel matrix, crossover topology, arbitrary delay, or general FIR/mixed-phase realization. |
| **Wavelet (`wavelet`)** | Textual magnitude EQ for supported Wavelet workflows | Serial GraphicEQ-style magnitude correction | No routing, delay, crossover, convolution, or phase correction. |
| **PipeWire (`pipewire`)** | PipeWire filter-chain playback, including suitable FIR sidecars | Serial filter chains and supported convolution sidecars | The exporter is not a complete substitute for the canonical routed graph. Confirm channel routing, sidecar paths, and filter-chain support before using it for home-cinema bass management. |
| **Roon (`roon`)** | Roon DSP Engine IIR/FIR playback | Serial Roon-supported IIR/FIR stages within Roon's limits | Roon's supported stage set, channel model, latency, and file-handling limits apply; arbitrary RoomEQ matrices, route graphs, or plugin types are not guaranteed. |
| **REW (`rew`)** | Importing one channel of IIR EQ into REW or another compatible tool | One channel of gain plus supported biquad filters, with an explicit preamp | Exactly one channel. No delay, FIR, crossover, bass routing, matrix, or other graph stage. |
| **Normalized coefficients (`coefficients`)** | Integrating RoomEQ filters into custom DSP | Any number of serial channels, gain, delay, and the 12 canonical RoomEQ biquad types | No FIR, crossover, matrix, bass routing, or plugin graph. The host must apply `preamp_gain_db`, `delay_ms`, section order, and the documented coefficient convention. |

### Practical Target Selection

- For a 5.1/7.1 system with redirected bass, separate main/sub delays, or
  multiple crossover groups, use the canonical JSON or CamillaDSP. These are
  the targets intended to preserve a graph with several source branches
  feeding one physical sub output.
- For ordinary stereo IIR room correction, APO, EasyEffects, PipeWire, Roon,
  or normalized coefficients can be appropriate, depending on the playback
  host.
- For FIR or mixed-phase correction, use the canonical JSON or an exporter
  that explicitly supports the generated convolution sidecars. Keep the WAV
  files with the exported configuration and verify sample rate, channel order,
  latency, and pre-ringing policy.
- Use REW, Wavelet, and normalized coefficients only when you intentionally
  want a reduced per-channel magnitude-EQ representation.

Every external export validates the source graph against the target's known
constraints. A successful export means the artifact is representable by that
target; it does not mean that unsupported RoomEQ routing or temporal behavior
was silently preserved.

## Configuration File Format

### Simple Stereo System

```json
{
  "speakers": {
    "left": "measurements/left_speaker.csv",
    "right": "measurements/right_speaker.csv"
  },
  "optimizer": {
    "num_filters": 10,
    "algorithm": "nlopt:cobyla",
    "max_iter": 5000,
    "min_freq": 20.0,
    "max_freq": 20000.0,
    "min_q": 0.5,
    "max_q": 10.0,
    "min_db": -12.0,
    "max_db": 12.0,
    "loss_type": "flat"
  }
}
```

### 2.1 System with Explicit Topology (v2.1)

```json
{
  "system": {
    "model": "stereo",
    "speakers": {
      "L": "left_meas",
      "R": "right_meas",
      "LFE": "sub_meas"
    },
    "subwoofers": {
      "config": "single",
      "crossover": "bass_xo",
      "sub_meas": "L"
    },
    "bass_management": {
      "lfe_low_pass_hz": 120.0
    }
  },
  "crossovers": {
    "bass_xo": {
      "type": "LR24",
      "frequency": 80.0
    }
  },
  "speakers": {
    "left_meas": "measurements/left.csv",
    "right_meas": "measurements/right.csv",
    "sub_meas": "measurements/sub.csv"
  },
  "optimizer": {
    "num_filters": 10,
    "algorithm": "cobyla"
  }
}
```

The LFE programme cutoff is a separate bass-management control: changing or
optimizing `bass_xo` redirects main-channel bass without narrowing the LFE
programme band. The cinema default is 120 Hz.

For routed home-cinema output, RoomEQ optimizes each logical input against its
own high-passed main plus redirected low-passed sub branch. Crossover type and
frequency are shared within the speaker group; route trim, relative delay, and
polarity are reported per source in `bass_management.optimization.source_results`.
The optimizer never uses a coherent sum of independent programme channels for
tonal calibration. Whole-bus aggregation is reserved for the configured
headroom model, which may add one common down-only input safety trim.

#### Per-sub crossovers (multi-sub low-pass per driver)

A multi-subwoofer system may give each physical sub its own crossover key by
writing `system.subwoofers.crossover` as a positional list instead of a single
string:

```json
"subwoofers": {
  "config": "mso",
  "crossover": ["bass_xover1", "bass_xover2"]
}
```

Entry `i` applies to physical sub `i` in driver order. The selector evaluates
all mains as separate logical inputs against the complete shared sub array,
including the actual main high-pass and each driver's
low-pass. The selected filters are included before route delay, polarity,
and trim optimization. A crossover list does not create L-to-left-sub or
R-to-right-sub routing. Each selected frequency stays inside its own range.
Every key must exist, and the list must contain either one shared entry or
one entry per physical sub.

Deployment: `LP_i` is a low-pass `crossover` plugin on each physical sub
(`channels.<SUB>.drivers[i].plugins`, staged `post_route`). Redirected-bass
routes omit a second group low-pass when this filter is present. Main high-pass
frequency remains optimized for the shared main group; the LFE programme keeps
its independent low-pass. The optimizer, replay and export use this same graph.
Reports retain `bass_management.groups[].selected_sub_low_pass_hz`,
`bass_management.optimization.sub_output_results[].selected_low_pass_hz`, and
the `per_sub_lp_deployed_to_drivers:N` advisory. Legacy shared crossover
configurations keep their route low-pass and have no per-driver low-pass.


### Multi-driver Speaker (2-way)

```json
{
  "speakers": {
    "left": {
      "name": "Left Speaker (2-way)",
      "measurements": [
        "measurements/left_woofer.csv",
        "measurements/left_tweeter.csv"
      ],
      "crossover": "default_lr24"
    }
  },
  "crossovers": {
    "default_lr24": {
      "type": "LR24"
    }
  },
  "optimizer": {
    "num_filters": 10,
    "algorithm": "nlopt:cobyla",
    "max_iter": 5000,
    "min_freq": 100.0,
    "max_freq": 10000.0,
    "min_q": 0.5,
    "max_q": 10.0,
    "min_db": -12.0,
    "max_db": 12.0,
    "loss_type": "flat"
  }
}
```

### Measurement CSV Format

The four-column microphone phase-calibration library API treats calibration
support as measured evidence: `sample_at` returns `None` outside that support,
and `apply_to_curve` returns an error without modifying the input if any
frequency is unsupported or evidence is malformed. Successful application
invalidates cached phase decomposition. Callers must handle the returned
`Result`; these helper guarantees do not by themselves establish that a
recording/export workflow applied the calibration, or establish absolute SPL
equivalence between CSV, impulse-response and MDAT sources.

Measurement CSV files should have the following columns:
- `freq`: Frequency in Hz
- `spl`: Sound pressure level in dB

Example:
```csv
freq,spl
20,75.0
50,78.0
100,80.0
200,82.0
...
```

## Optimizer Configuration

For single-channel processing, the requested correction band is intersected
with the measurement's native frequency support before preprocessing, scoring,
and PEQ/FIR design. A measurement starting at 100 Hz cannot authorize a 20 Hz
correction band. Disjoint requested and measured bands fail explicitly; the
processor does not invent measurements outside the captured range. This
constrains the design band, not the natural response tails of finite filters.

Progress reporting does not change filter-selection policy. When adaptive
selection is enabled by `min_filter_improvement`, it remains enabled for
interactive and QA runs with progress callbacks, including Hybrid's IIR stage.
A stop request cancels the adaptive run rather than advancing to another pass.

### Algorithms

Hybrid spatial FIR searches retain the caller's progress/stop callback after
the IIR stage. DE and CMA-ES check it at native generation boundaries as well
as scored FIR basis boundaries. COBYLA and ISRES currently check only at stage
boundaries because the pinned scalar backends lack native stop hooks; their
search can finish before a pending Stop is observed. A result is discarded
once Stop is observed. Completed FIR optimizer evidence records
`callback_cancellation=native_generation_boundaries` or
`callback_cancellation=stage_boundaries_only` when a callback was supplied.
These checks do not interrupt FIR template construction, an initial population,
or an in-flight objective evaluation, and do not promise a maximum stop latency.

- `autoeq:cmaes`: CMA-ES (default global optimizer)
- `autoeq:de`: Differential Evolution
- `autoeq:cobyla`: COBYLA (Constrained Optimization BY Linear Approximations)
- `autoeq:isres`: Improved Stochastic Ranking Evolution Strategy
- Other AutoEQ and metaheuristics algorithms supported by autoeq

### Loss Types

- `flat`: Optimize for flat frequency response
- `score`: Optimize for Harman/Olive score (bass boost + flat PIR)
- `epa`: experimental ERB-rate loss with diagnostic transfer-response EPA descriptors; not a validated programme loudness/roughness or measured-decay model

The neutral flat/asymmetric objective and runtime acceptance use the same
versioned auditory measure, `glasberg-moore-erb-rate-1990-v1`. It integrates
residual energy with discrete ERB-rate cell widths, so changing between linear,
logarithmic, sparse, or dense frequency grids does not silently change the
meaning of the reported RMS. Residual signs are never smoothed before the
nonlinear loss.

For multiple measurements, `spatial_robustness` evaluates every seat directly
with a variance-penalized risk measure. Its spatial-variance correction-depth
mask is applied to each seat; seats are not collapsed to a power-average curve
before optimization. Case-bootstrap uncertainty is explicitly labelled
`spatial_seat_sampling` and assumes independent positions by default. For
correlated nearby seats, configure a reduced `effective_spatial_sample_size`;
the bootstrap then draws fewer cases per resample and produces a wider,
conservative interval. This is not a spatial block bootstrap or covariance
model. Repeat-sweep noise and microphone-calibration uncertainty remain
separate, explicitly supplied nuisance sources.

When `psychoacoustic` is enabled, `psychoacoustic_smoothing` can override the default variable smoothing curve (`1/48` octave below 100 Hz through `1/6` octave above 1 kHz). When `asymmetric_loss` is enabled, `asymmetric_loss_config` can override peak/dip and bass peak/dip weights without changing the default behavior for existing configs.

`perceptual_policy` can fill coherent defaults for `reference`, `music`, `cinema`, `night`, and `speech` use cases. The policy layer maps existing knobs rather than replacing them: target response, EPA/asymmetric weighting, psychoacoustic smoothing, spatial/bootstrap robustness, audibility deadband, high-frequency guardrails, FIR direct/early/late advisories, and validation bundle descriptors remain individually configurable.

### Crossover Types

- `LR24` or `LR4`: Linkwitz-Riley 24 dB/oct (4th order)
- `LR48` or `LR8`: Linkwitz-Riley 48 dB/oct (8th order)
- `Butterworth12` or `BW12`: Butterworth 12 dB/oct (2nd order)
- `Butterworth24` or `BW24`: Butterworth 24 dB/oct (4th order)
- `LinearPhase`, `FIR`, or `LPFIR`: complementary FIR crossover with constant group delay and no crossover-point phase rotation

## Advanced Audio Corrections

RoomEQ provides advanced audio correction features for optimizing room acoustics in two scenarios:

- **Scenario A (WITH Subwoofers)**: Phase alignment and multi-seat variance minimization
- **Scenario B (WITHOUT Subwoofers)**: Schroeder split, excursion protection, and target response shaping

### Target Response

Some listeners prefer a gently downward-sloping house curve. RoomEQ's
**-0.8 dB/octave** Harman-style option is a user preference, not a universal
neutral room-correction target. It is emitted in the separately bypassable
preference layer and excluded from neutral correction quality scores. Target
shaping is configured through the unified `target_response` object, together
with optional user-preference shelves and the broadband pre-correction toggle.

```json
{
  "optimizer": {
    "target_response": {
      "shape": "harman",
      "slope_db_per_octave": -0.8,
      "reference_freq": 1000,
      "preference": {
        "bass_shelf_db": 0,
        "bass_shelf_freq": 200,
        "treble_shelf_db": 0,
        "treble_shelf_freq": 8000
      },
      "broadband_precorrection": false
    }
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | string | `"flat"` | Target shape: `"flat"`, `"harman"`, `"custom"`, `"file"`, `"from_measurement"`; `harman` is a house curve realized in the preference layer |
| `slope_db_per_octave` | number | -0.8 | Slope in dB/octave (negative = downward tilt). Used when `shape == "custom"` |
| `reference_freq` | number | 1000 | Frequency where the target slope passes through 0 dB (Hz) |
| `curve_path` | string (path) | - | CSV target file path (used when `shape == "file"`) |
| `preference.bass_shelf_db` | number | 0 | Bass shelf in the separately bypassable post-correction preference layer (dB) |
| `preference.bass_shelf_freq` | number | 200 | Bass shelf transition frequency (Hz) |
| `preference.treble_shelf_db` | number | 0 | Treble shelf in the separately bypassable post-correction preference layer (dB) |
| `preference.treble_shelf_freq` | number | 8000 | Treble shelf transition frequency (Hz) |
| `broadband_precorrection` | boolean | false | Run a preliminary broadband shelf + gain fit before the fine-grained PEQ pass |

The neutral optimizer target is computed without preference shelves:
```
target_db(f) = slope * log2(f / reference_freq)
```

Preference shelves are realized afterward as a separate output IIR layer. They
remain visible in the final DSP/plugin chain, but are excluded from neutral
post-EQ scores and `raw_post_eq_curve`. Output metadata records both
`neutral_target_response` and `preference_layer`, including
`excluded_from_neutral_quality_score: true`.

**Example: Harman with Bass Boost**

```json
{
  "optimizer": {
    "target_response": {
      "shape": "harman",
      "preference": {
        "bass_shelf_db": 3,
        "bass_shelf_freq": 200
      }
    }
  }
}
```

### Excursion Protection

Bookshelf speakers and small drivers have limited bass extension. Attempting to boost bass below the speaker's F3 point (-3dB frequency) can cause excessive driver excursion, increased distortion, and potential damage.

Excursion protection automatically detects the F3 rolloff and generates a highpass filter to prevent dangerous over-boost.

```json
{
  "optimizer": {
    "excursion_protection": {
      "enabled": true,
      "auto_detect_f3": true,
      "f3_reference_min_hz": 100.0,
      "f3_reference_max_hz": 200.0,
      "filter_order": 4,
      "filter_type": "linkwitzriley",
      "margin_octaves": 0.25
    }
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | false | Enable excursion protection |
| `auto_detect_f3` | boolean | true | Auto-detect F3 from measurement |
| `manual_f3_hz` | number | - | Manual F3 override (Hz) when auto-detect is false |
| `filter_order` | integer | 4 | HPF order: 2=12dB/oct, 4=24dB/oct |
| `filter_type` | string | `"linkwitzriley"` | Filter type: `"linkwitzriley"` or `"butterworth"` |
| `margin_octaves` | number | 0.25 | Safety margin below F3 for HPF placement |

**F3 Detection Algorithm:**
1. Smooth the measurement curve (1/3 octave)
2. Find reference level at 100-200 Hz
3. Search downward for -3dB point
4. Place HPF at `F3 * 2^(-margin_octaves)`

### Schroeder Frequency Split

The **Schroeder frequency** marks the transition between modal (low frequency) and statistical (high frequency) behavior in a room. Below this frequency, room modes dominate and require high-Q narrow filters for correction. Above this frequency, broad tonal adjustments are more appropriate.

Typical Schroeder frequencies:
- Small room (15 m³): ~400 Hz
- Medium room (40 m³): ~250 Hz
- Large room (100 m³): ~160 Hz

```json
{
  "optimizer": {
    "schroeder_split": {
      "enabled": true,
      "schroeder_freq": 300,
      "low_freq_config": {
        "max_q": 5.0,
        "min_q": 0.5,
        "allow_boost": false
      },
      "high_freq_config": {
        "max_q": 1.0,
        "shelving_only": false
      }
    }
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | false | Enable Schroeder split |
| `schroeder_freq` | number | 300 | Schroeder frequency (Hz) |
| `room_dimensions` | object | - | Optional room dimensions for auto-calculation |
| `low_freq_config.max_q` | number | 5.0 | Max Q for low-freq filters |
| `low_freq_config.min_q` | number | 0.5 | Min Q for low-freq filters |
| `low_freq_config.allow_boost` | boolean | false | Allow boosts (not recommended) |
| `high_freq_config.max_q` | number | 1.0 | Max Q for high-freq filters |
| `high_freq_config.shelving_only` | boolean | false | Use only shelving filters |

**Auto-Calculate Schroeder from Room Dimensions:**

```json
{
  "optimizer": {
    "schroeder_split": {
      "enabled": true,
      "room_dimensions": {
        "length": 5.0,
        "width": 4.0,
        "height": 2.5
      }
    }
  }
}
```

When RT60 and room volume are available, RoomEQ calculates the Schroeder
frequency as `f_S ≈ 2000 · √(RT60 / V)`, where RT60 is seconds and V is room
volume in m³.

### Phase Alignment

When integrating a subwoofer with main speakers, proper time/phase alignment in the crossover region is critical. Misalignment causes cancellation dips at crossover, reduced bass output, and poor transient response.

Phase alignment optimizes the delay and polarity to maximize energy sum in the crossover region.

```json
{
  "optimizer": {
    "phase_alignment": {
      "enabled": true,
      "min_freq": 60,
      "max_freq": 100,
      "optimize_polarity": true,
      "max_delay_ms": 30
    }
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | true | Enable phase alignment |
| `min_freq` | number | 60 | Minimum optimization frequency (Hz) |
| `max_freq` | number | 100 | Maximum optimization frequency (Hz) |
| `optimize_polarity` | boolean | true | Test both normal and inverted polarity |
| `max_delay_ms` | number | 3 | Maximum delay search range (ms); the default is refined by the phase-alignment scan and golden-section search |

**Algorithm:**
1. **Global scan**: Test delays from -max_delay to +max_delay with frequency-adaptive sampling (the default range is ±3 ms)
2. **For each candidate**: Compute combined response `|H_sub + H_speaker * e^(-jωτ) * polarity|`
3. **Integrate energy** in [min_freq, max_freq] band
4. **Fine search**: Refine the best scan interval with a golden-section search
5. **Output**: Optimal delay and polarity for maximum energy sum

**Note:** Both subwoofer and speaker measurements must include phase data (export from REW with phase, or measure with calibrated mic).

### Multi-Seat Optimization

In rooms with multiple listening positions, optimizing for one seat often degrades others. Multi-seat optimization finds subwoofer gain/delay settings that minimize variance across all seats.

```json
{
  "optimizer": {
    "multi_seat": {
      "enabled": true,
      "strategy": "minimize_variance",
      "primary_seat": 0,
      "max_deviation_db": 6
    }
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | false | Enable multi-seat optimization |
| `strategy` | string | `"minimize_variance"` | Optimization strategy |
| `primary_seat` | integer | 0 | Primary seat index (0-based) |
| `max_deviation_db` | number | 6 | Max deviation at secondary seats (dB) |

**Strategies:**

| Strategy | Description |
|----------|-------------|
| `minimize_variance` | Minimize standard deviation of SPL across all seats |
| `primary_with_constraints` | Optimize primary seat, constrain others within max_deviation |
| `average` | Optimize for flattest average response across seats |
| `modal_basis` | Complex modal-basis SFM: extracts dominant seat modes from per-sub/per-seat transfer functions and optimizes sub gain/delay/polarity/all-pass controls |

**Measurement Setup:**

For multi-seat optimization, you need measurements of each subwoofer at each seat position:

```json
{
  "speakers": {
    "subs": {
      "name": "Multi-seat Subwoofers",
      "subwoofers": [
        ["sub1_seat1.csv", "sub1_seat2.csv", "sub1_seat3.csv"],
        ["sub2_seat1.csv", "sub2_seat2.csv", "sub2_seat3.csv"]
      ]
    }
  }
}
```

### Supporting-Source Room Compensation

A supporting-source loudspeaker is a delayed, decorrelated loudspeaker placed in
same room as the primary source. It adds reverberant energy to the primary
source without altering its direct sound, improving apparent source width and
room envelopment while preserving imaging (Brooks-Park et al., JASA 159(4),
2026). RoomEQ computes a minimum-phase FIR for the supporting source that fills
in the primary's room-induced spectral notches, subject to a precedence ceiling
and a configurable compensation band.

```json
{
  "system": {
    "model": "stereo",
    "speakers": {
      "L": "left_pair",
      "R": "right_pair"
    }
  },
  "speakers": {
    "left_pair": {
      "name": "Left Main + Support",
      "primary": "measurements/left_primary.csv",
      "support": "measurements/left_support.csv",
      "supporting_source": {
        "delay_ms": 10.0,
        "allow_unverified_acoustics": true,
        "freq_range_hz": [70.0, 20000.0],
        "decorrelation": "velvet_noise",
        "fir_taps": 8192,
        "velvet_noise_taps": 4096,
        "precedence_limits": [
          { "low_hz": 70.0, "high_hz": 500.0, "limit_db": 10.0 },
          { "low_hz": 500.0, "high_hz": 20000.0, "limit_db": 6.0 }
        ]
      }
    },
    "right_pair": {
      "name": "Right Main + Support",
      "primary": "measurements/right_primary.csv",
      "support": "measurements/right_support.csv",
      "supporting_source": {
        "delay_ms": 10.0,
        "allow_unverified_acoustics": true
      }
    }
  },
  "optimizer": {
    "processing_mode": "phase_linear",
    "loss_type": "flat"
  }
}
```

For each `SupportingSourceGroup`, RoomEQ emits two output channels (e.g.
`L` and `L_support`, or `WideLeft` and `WideLeft_support` in a home-cinema
layout). The primary output intentionally bypasses ordinary RoomEQ EQ, so its
direct sound remains unchanged; the configured optimizer does not apply to
that primary path. The supporting channel contains a `convolution` plugin
loading the generated FIR WAV file. `metadata.supporting_source` records this
as `primary_eq_bypassed_to_preserve_direct_sound`, along with precedence-limit
and spatial-robustness advisories. Absolute DRR summaries are present only
when time-gated impulse-response evidence is available.

The example above is explicitly experimental: magnitude curves alone do not
establish arrival timing or coherent interference. By default supporting-source
processing requires `acoustic_arrival_offset_ms` (unfiltered support arrival minus
primary arrival at the reference seat, measured on a common time reference) and
`shared_phase_reference: true` with measured phase on both transfers. Electrical
delay is `delay_ms - acoustic_arrival_offset_ms`: a support source arriving 2.5 ms
earlier needs 12.5 ms electrical delay to achieve a requested 10 ms propagation-plus-
electrical lag. A required advance, or `optimizer.allow_delay: false` when positive
electrical delay is needed, is rejected. Do not set the shared-phase flag merely
because CSVs contain phase columns; independent sweep time origins are insufficient.

The realized FIR, gain, and delay are replayed for the reported coherent sum.
`max_coherent_cancellation_db` defaults to a 3 dB engineering budget below the
louder branch; exceeding it rejects a verified-mode run. Explicit
`allow_unverified_acoustics: true` permits experimental operation with conspicuous
missing-evidence/over-budget advisories. The report labels the **power-average
design** separately from `coherent_sum`; the latter is omitted without shared-phase
evidence. `propagation_relative_arrival_ms` excludes FIR energy spread and is not a
measured perceptual onset. Reference-seat prediction does not prove fusion,
localization, DRR, or multi-seat benefit; validate those with final-chain measurements
and controlled listening. Required evidence is checked before writing the FIR.

### Listening-stimulus resolution

The quality harness's band-limited noise has nominal -6 dB band edges and a
transition allowance of half the smallest of bandwidth, lower edge, and upper-edge
clearance to Nyquist. The FIR design targets at least 40 dB stopband rejection
outside those transitions. Requests exceeding 4095 taps are errors, not silently
widened bands; for example 90–110 Hz at 48 or 96 kHz is unsupported. Minimum
bandwidth and edge clearance are approximately `8 * sample_rate / 4095` Hz.
Deterministic filter-response checks complement seeded waveform tests; finite
stimulus duration still matters when interpreting measured spectra.

### Complete Configuration Examples

**Scenario A: System with Subwoofers**

```json
{
  "speakers": {
    "left": "measurements/left.csv",
    "right": "measurements/right.csv",
    "sub": "measurements/subwoofer.csv"
  },
  "optimizer": {
    "algorithm": "autoeq:cmaes",
    "num_filters": 10,
    "refine": true,

    "target_response": {
      "shape": "harman",
      "preference": {
        "bass_shelf_db": 2
      }
    },

    "phase_alignment": {
      "enabled": true,
      "min_freq": 60,
      "max_freq": 100,
      "optimize_polarity": true,
      "max_delay_ms": 30
    }
  }
}
```

**Scenario B: Bookshelf Speakers without Subwoofer**

```json
{
  "speakers": {
    "left": "measurements/left_bookshelf.csv",
    "right": "measurements/right_bookshelf.csv"
  },
  "optimizer": {
    "algorithm": "autoeq:cmaes",
    "num_filters": 12,
    "refine": true,

    "target_response": {
      "shape": "harman"
    },

    "excursion_protection": {
      "enabled": true,
      "auto_detect_f3": true,
      "filter_order": 4,
      "margin_octaves": 0.25
    },

    "schroeder_split": {
      "enabled": true,
      "schroeder_freq": 300,
      "low_freq_config": {
        "max_q": 10,
        "allow_boost": false
      },
      "high_freq_config": {
        "max_q": 1.0
      }
    }
  }
}
```

### Optimization Flow

When multiple features are enabled, the optimization follows this order:

```
1. Load measurement(s)
2. Build the neutral target curve from `target_response.shape` (Harman house curve and other preferences stripped)
3. [IF excursion_protection] Detect F3, generate protection HPF
4. [IF has_subwoofer && phase_alignment] Optimize delay/polarity for energy max
5. [IF multi_seat] Optimize sub gains/delays for variance minimization
6. [IF schroeder_split] Two-pass EQ (low-Q high freq, high-Q low freq)
   [ELSE] Standard EQ optimization
7. Append the separately bypassable preference/content layer and combine the DSP chain
```

### API Reference

The features are also available programmatically:

```rust
use autoeq::roomeq::{
    // Target Response
    build_complete_target_curve,
    TargetResponseConfig, TargetShape, UserPreference,

    // Excursion Protection
    detect_f3, generate_excursion_protection,
    ExcursionProtectionConfig, ExcursionProtectionResult,

    // Phase Alignment
    optimize_phase_alignment,
    PhaseAlignmentConfig, PhaseAlignmentResult,

    // Multi-Seat
    optimize_multiseat,
    MultiSeatMeasurements, MultiSeatConfig, MultiSeatOptimizationResult,
};
```

## Output Format

The output is a JSON file containing DSP chains for each channel:

```json
{
  "channels": {
    "left": {
      "channel": "left",
      "plugins": [
        {
          "plugin_type": "gain",
          "parameters": {
            "gain_db": -2.5
          }
        },
        {
          "plugin_type": "eq",
          "parameters": {
            "filters": [
              {
                "filter_type": "peak",
                "freq": 1000.0,
                "q": 1.5,
                "db_gain": 3.0
              }
            ]
          }
        }
      ]
    }
  },
  "metadata": {
    "pre_score": 0.0,
    "post_score": 0.0,
    "algorithm": "nlopt:cobyla",
    "iterations": 5000,
    "timestamp": "2025-01-15T12:00:00Z"
  }
}
```

This output can be loaded directly into the AudioEngine plugin system.

## Examples

See the `tests/data/roomeq/` directory for example configurations:
- `test_config_stereo.json`: Simple stereo system
- `test_config_multidriver.json`: Multi-driver speaker with crossover

## Documentation

Detailed format documentation with examples:
- [`ROOMEQ_INPUT_FORMAT.md`](ROOMEQ_INPUT_FORMAT.md): Complete input configuration format
- [`ROOMEQ_OUTPUT_FORMAT.md`](ROOMEQ_OUTPUT_FORMAT.md): Complete DSP chain output format

JSON Schemas for validation:
- [`input_schema.json`](../src/bin/roomeq/input_schema.json): Input configuration schema
- [`output_schema.json`](../src/bin/roomeq/output_schema.json): Output DSP chain schema

Configuration validation is also exposed as a versioned five-stage runtime
report: `schema_version`, `structural`, `resolved_resource`, `acoustic`, and
`export_target`. A structural-only report is intentionally not
`production_ready`; the CLI loader returns the staged report and the production
optimizer reruns the required resource/acoustic gates after path resolution.

## Testing

Run the integration tests:
```bash
cargo test -p autoeq --test roomeq_integration_test
```

Run the unit tests:
```bash
cargo test -p autoeq --bin roomeq
```

## Architecture

Production RoomEQ code is partitioned by responsibility:

- `roomeq-model`: configuration, validation, and output contracts.
- `roomeq-analysis`: measurement, phase, spatial, and acoustic analysis.
- `roomeq-quality`: perceptual metrics and acoustic-corpus acceptance.
- `roomeq-engine`: DSP, filter design, optimization, routing, and safety gates.
- `roomeq-workflow`: configuration loading and complete run orchestration.
- `roomeq-export`: export-target conformance and artifact generation.
- `roomeq-cli`: command-line parsing, schemas, and result serialization.

The historical root `src/roomeq/` tree is a compatibility facade, not the
production implementation boundary.

### QA tiers and scenario registry

Escaped-defect ownership checking now requires a regression identity object with
`package`, library `target`, and fully qualified `name`, resolved against nextest
JSON discovery. Ignored or zero-selected inventories cannot establish ownership.
Recipes are resolved from Just's JSON dump and literal dependency/run commands
reachable from `.github/workflows/ci.yml`; dynamic shell invocations are not
guessed. A mutant fixture needs a manifest and runner, not only a README.
`just qa-roomeq-escaped-defects` resolves and executes the registered exact tests;
`just qa-roomeq-escaped-defect-mutants` separately executes the registered
semantic faults. Their evidence records distinguish a completed run from a
running or failed one and retain per-defect artifact hashes. These recipes are
wired into the blocking RoomEQ CI workflow; local results are not evidence of
a hosted CI run. Ownership discovery alone never establishes killed mutants.
Checker regressions run with
`python3 -m unittest scripts.test_escaped_defect_ownership`.

The opt-in `--parameter-matrix` runner currently establishes finite-output smoke
coverage, not complete pairwise execution or useful correction. Its artifact at
Each parameter-matrix row retains a unique directory under
`target/qa/roomeq-parameter-bundles/`. `request.json` stores the exact single-speaker
measurements separately from `configuration_without_speakers`, since internal
in-memory sources are not serializable CLI measurement references.
`selected-output.json` and its convolution sidecars retain the selected rerun's
DSP for independent replay after execution. The matrix's `replay_bundle` links
these artifacts; earlier bundles are not overwritten by subsequent runs.
This artifact-retention contract does not establish backend or acoustic success.

`target/qa/roomeq-parameter-matrix.json` records requested axes, effective optimizer
and routing settings, measurement descriptors, selected DSP rate, delivered FIR
lengths, stage outcomes, and seed reliability. The selected rate is passed to
optimization; stereo, redirected 2.1, and 5.1 configurations use their actual
topology builders. Crossover values request fixed 160 Hz LR24, automatic
120–220 Hz LR24, and fixed 160 Hz LR48 respectively. The shell contract checks
each reported physical main/sub branch against its selected crossover group;
automatic selection without measured phase must remain explicitly skipped,
not counted as successful search. These route checks are not an independent
render of every matrix row. Full execution and usefulness/mutation coverage
remain unfinished. CLI success/failure checks are in
`scripts/test_roomeq_synthetic_exit.py`; run after building the release QA binary.

The public acoustic quality evaluator now records `useful_output` separately
from normalized shape RMS, for every training and held-out seat. Its default
authorized broadband gain is 0 dB. Library callers can use
`evaluate_acoustic_quality_with_permitted_gain` to declare an intended trim or
headroom attenuation; this gain is not fitted from the candidate. The unexplained
loss metric is log-frequency-weighted RMS below the lesser of baseline and target
(or baseline when no target exists), after the declared gain. This allows removing
above-target peaks without requiring inversion of existing nulls. The public gate
uses a configurable 3 dB engineering budget, not a listening-calibrated threshold.
A separate loss RMS over measured support at or below 200 Hz prevents the main
band from diluting bass loss; its actual evaluated band is reported, and fewer
than two supported bass samples leave that evidence absent.
Absolute target-shortfall RMS remains visible even for authorized attenuation;
exceeding its separate 3 dB advisory budget reports `best_effort_target_shortfall`.
Final multi-seat replay now retains this evidence across logical inputs and both
partitions and enforces the 3 dB loss budget. Explicit per-logical-input correction
gain allowances use `optimizer.permitted_output_gain_db`; see the input format.
Structural routing gain remains in both pre/post baselines. Single-seat-only
paths, structural-route loss, full pipeline/export certification, and electrical
headroom remain outside this completed integration scope.

Final-seat replay preserves a full-range main when a shorter sub capture has an
explicit calibrated `optimizer.upper_band_acoustic_bounds` declaration for that
physical output, partition, and seat. A falling measured tail is not sufficient.
The bound is processed through the actual branch DSP; aggregate omission must
stay within 0.1 dB magnitude uncertainty. Reports retain magnitude and phase
uncertainty without inventing unmeasured phase, and use a conservative improvement
lower bound. Missing or significant upper-band evidence fails validation instead
of truncating the main assessment. See `src/bin/roomeq/INPUT_FORMAT.md` for the
configuration and calibration contract.

Unequal lower endpoints do not silently shorten the assessment band either.
Within the requested band, if one driver or routed output has measured bass
that another physical branch does not cover, replay reports insufficient
summation evidence. Upper-band bounds do not certify missing lower-band
response. Supply the missing capture or explicitly request only the common
supported band; replay does not infer a tweeter's acoustic stopband from its
electrical crossover. ULP-scale endpoint rounding is tolerated, not meaningful
frequency extrapolation.

Routed correction scores use the same measured passband
and correction-only basis for pre and post, removing routing transfer from the
post response. These shape scores do not replace coherent routed-splice,
useful-level, electrical-headroom, or held-out-seat evidence.

Phase-linear and single-measurement Hybrid FIR outputs publish the calibrated design target,
including flat targets. Final correction acceptance preserves explicit target
levels when cropping to a routed passband; only inferred legacy targets are
re-leveled within that band. A selected crossover must not redefine the target
after optimization. Native target grids are aligned explicitly to the measured
response for acceptance. A target that does not cover the evaluated passband is
not extrapolated or accepted on a silently narrower band.

Hybrid residual FIR design keeps the target level prepared from the original
optimization input. The IIR stage's residual does not establish a new target
level; existing FIR boost limits and final acceptance still apply.

For multi-measurement Hybrid with a linear-phase FIR, the residual stage now
scores the complete IIR+FIR response against the same prepared per-measurement
objectives as the IIR stage, including weights, spatial masks and bootstrap
risk. It searches convex combinations of equal-length per-objective,
representative and neutral FIR designs; this is a finite candidate-basis
optimization, not an unrestricted FIR optimum. The displayed channel target
remains a representative target, not a replacement for the per-seat objective
bank. Optimizer evidence identifies this search and its selected candidate.
The bounded scalar backends currently supported by this stage are DE, CMA-ES,
COBYLA and ISRES; another requested backend fails explicitly rather than being
silently substituted. Multi-measurement minimum-phase Hybrid uses an aligned
per-objective dB-correction basis instead: each trial is realized through the
minimum-phase generator before its actual finite-tap response is scored.
It does not mix minimum-phase coefficients. This costs more per evaluation
than the cached linear-phase bank. It currently requires aligned objective
grids and uses the same bounded scalar backends. Kirkeby now uses the same
realized dB-basis search for multi-seat magnitude selection, while retaining
the representative residual as its acoustic phase reference. Requested
excess-phase correction requires actual reference acoustic phase; electrical
IIR phase cannot replace missing measurement evidence. The scratch-buffer and
calibration-phase defects are repaired in integrated math-iir-fir 0.5.23, but
Hybrid reference-phase correction still fails seat-weight separation and temporal
requirements. A separate primitive diagnostic finds arrival-delay attenuation
from finite-tap windowing. Do not interpret magnitude improvement as phase
benefit. Causal-support and phase realization fixes, broader
minimum-phase outcome checks, full temporal/headroom validation and
backend-rendered spatial benefit remain open outcome-audit work.
When a required post-workflow FIR cannot be generated or its WAV cannot be
written, the workflow fails instead of installing a nonexistent sidecar or
silently omitting the requested stage.

Fractional FIR group-delay alignment uses a 129-tap windowed-sinc delay kernel
with a declared usable band up to 0.46 times sample rate and a 0.01 dB magnitude
tolerance. It allocates enough common integer padding to keep the entire delay
kernel causal, including requested negative advances; all playback channels
receive that common latency. For a half-sample delay starting at tap zero,
the padding is 64 samples (1.333 ms at 48 kHz). The new convolution stage is
placed before routing, and retains `gd_requested_delay_ms`,
`gd_effective_delay_ms`, `gd_common_padding_samples`, and
`gd_usable_band_max_hz`. The group-delay summary reports effective delays,
and temporal evidence is recomputed from the resulting coefficients. Existing
FIR sidecars are preserved. Nyquist is outside the fractional-delay guarantee;
an unsupported measured playback band is reported as an unapplied GD stage.

The required `just qa-roomeq-camilladsp-backend` gate also renders packaged
fractional-delay exports with CamillaDSP at 44.1, 48, and 96 kHz. It checks
-0.5, +0.25, and +0.5 sample offsets with original FIR impulses at taps 0 and
17, after removing the source sidecar directory. Each of these 18 renders is
checked at 1,025 frequencies from 20 Hz through 0.46 times sample rate against
an independent pure-delay reference and the reported complex response. This
is sampled-band evidence, not continuous-frequency or Nyquist certification.

The machine-readable QA inventory is
`crates/roomeq-qa/src/registry.json`. Registered
scenarios declare their configuration, solver, processing modes, execution
tier, claimed features, and quantitative expectations. The `pr`, `nightly`,
and `weekly` recipes select cumulative cost tiers and do not maintain separate
case inventories. Stochastic QA runs five deterministic seeds and selects the
median explicitly accepted result (or the median of all runs if none is accepted).
Every seed's acceptance decision, reversion/degradation reasons, optimizer
termination/budget evidence and final scores are retained in
`metadata.qa_seed_distribution`, acoustic current/candidate reports, and the
append-only `target/qa/roomeq-seed-distributions.jsonl` artifact. The
`accepted_useful_rate` counts explicitly accepted runs with improved delivered
scores; `safe_output_rate` conservatively counts finite outputs with explicit
final acceptance and no failed stage. Reverted or missing acceptance does not
establish safety, even when the pipeline returns successfully. This rate is not
independent backend certification. Neither rate is replaced by the selected
median or score spread. Missing acceptance evidence does not count as useful acceptance.
Completed JSONL records retain the optimizer configuration, requested seeds,
sample rate, and a separate `final_artifact` verdict with the rerun's scores,
acceptance, optimizer evidence, and stage outcomes. Population rates describe
the five selection runs, not the final rerun; `final_artifact_delivered` means
the pipeline returned successfully, not that the correction was accepted.
Seed execution errors and non-finite scores fail the QA run after all five
selection seeds have been attempted. The JSONL artifact retains the completed
outcomes and identified errors with `status: failed` and
`phase: seed_selection`; rates retain the five-seed denominator, and failed
seeds count as neither useful nor verified safe. An error while rerunning the
selected seed for final artifacts is recorded separately as
`phase: selected_artifact_run`, preserving the selection population and marking
`final_artifact_delivered: false`. Its population rates do not certify the
failed final rerun. Failure to write evidence also fails QA.
Runtime safety
fallback is reported as the distinct `REVERTED` outcome and passes only when
explicitly permitted by the registry. The full registry-driven FEM matrix and
the retained generated-data integration matrix also run on the scheduled
weekly workflow.

Acoustic corpus current/candidate scoring and robustness rescoring use the
runtime physical-seat replay contract. Native training captures are snapshotted
before optimization; held-out CSVs are not reduced to a display grid. Each
logical source is reconstructed from its contributing physical outputs, with
structural routing retained in the correction-disabled baseline. Missing physical
branches, required phase, or sidecars fail replay. Reports retain
`playback_evidence` with source/seat/output identity, baseline and delivered
curves, and summation support evidence. Robustness perturbations operate on
physical captures before summation; seat dropout removes the same seat across
all outputs. Declared upper-band bounds are conservatively increased by the
perturbation's SPL limit and that derivation is retained in their evidence IDs.
These software replay checks do not supply missing corpus captures or establish
unmeasured spatial performance. Existing acoustic baselines are not automatically
recalibrated when scoring changes.

Held-out corpus descriptors accept `seat_id`, an explicit shared listening
position ID, alongside the physical-output `channel` and capture `path`.
Coherent multi-output evaluation requires it. Named capture sets are sorted by
seat ID before replay; every contributing output must have the same ID set.
Duplicate output/seat pairs, mixed named/unnamed captures, and mismatched sets
are errors. Runtime seat indices (including support-bound declarations) follow
this sorted order; reports retain the ID as `seat_label`, including after
robustness dropout. IDs must come from acquisition provenance, not filename
guessing. Legacy independent-channel descriptors may remain unnamed and do not
thereby establish cross-output position identity. A physical sub capture need
not itself be selected as a logical source for scoring.

Acoustic quality shape normalization fits a constant gain using the same
log-frequency trapezoid weights as its residual RMS and mean seat spread.
Adding redundant bass samples therefore does not move the broadband reference.
Pooled bin percentiles remain explicitly distinct from frequency-integrated
statistics. Shape scores alone do not establish preserved output level.
Quality alignment retains the union of supplied pre/post/target grid points,
so a candidate-only native cancellation bin is not lost by baseline-grid sampling.
Useful-output evidence includes the worst sampled unexplained loss and runs of
consecutive samples exceeding the recorded 3 dB diagnostic threshold. These
`loss_bands` use the fixed evaluation support, calibrated target, and permitted
gain; they do not classify authorized attenuation as loss. Band endpoints are
sample locations, not a continuous-frequency guarantee. The diagnostic threshold
does not replace the separate runtime RMS-loss policy.
QA peak/dip metrics use a fixed band derived from the uncorrected measurement;
outer roll-offs can narrow that band, but internal holes and new candidate
cancellations remain eligible even below 20 dB relative to the response peak.

Gate purpose is part of the acceptance contract. A `safety` case may accept a
runtime `REVERTED` result only when it explicitly enables safe reversion.
`functional` and `quality` cases must retain the requested correction; a safe
fallback is not evidence that the processing mode or quality objective works.
Those tiers therefore require zero unexpected reversion.

`just qa-roomeq-contract-pr` runs deterministic realization, CTC replay,
main/sub role, registry-semantics, and measured Genelec 5.1.4 cross-mode
contracts with an explicit optimizer budget. Nightly and weekly quality runs
use the larger convergence budget, five-seed median, and broader matrices; the
PR contract uses one fixed seed. A reachability test
also fails if a runner declared by the registry disappears from CI and
scheduled workflows.

The blocking `qa-roomeq-ci` companion recipe runs quick safety coverage,
multi-seat guards, and perceptual contracts. The randomized five-seed quality
fuzzer remains scheduled nightly/weekly so its convergence runtime and random
case difficulty cannot stall deterministic PR feedback.

Optimizer, routing, crossover, and realization changes should observe a
48-hour pre-release stabilization window with the deterministic contract on
every change and completed scheduled quality runs. Release reports should
classify escaped defects as optimizer/objective, role/routing, DSP realization,
acceptance/reporting, or automation reachability, and record unexpected
reverts, cross-mode drift, mutation survivors, and suite runtime.

### Staged rollout and release gates

New audibility and acceptance policies roll out in three behaviors:
`Legacy` (policy disabled — the output it replaced, always available),
`Advisory` (report-only: evaluates and records reason-coded verdicts but
never changes output — the default whenever a new policy is selected), and `Enforcing`
(explicit opt-in that changes emitted output). Deleting a policy selection
restores legacy behavior; enforcement is never the default.

Promotion is evidence-gated by four independent release gates
(`crates/roomeq-qa/src/release_gates.rs`): implementation correctness,
physical safety, perceptual-model validation, and demonstrated listening
benefit. Passing one gate never implies the others. Advisory releases and
elapsed warning cycles carry no evidence and never promote. Physical
safeguards promote on correctness plus physical safety alone — the
optional rerank objective (Stage 4) never blocks them. Perceptual claims
additionally need perceptual validation; listening-benefit claims need
recorded listening outcomes.

Every staged artifact carries provenance for its stage: stimulus
manifests record renderer, version, platform, SPL mapping, and file
hashes; validation results record the preregistration hash; rerank
reports record evaluator and loss pins; export round trips
(`roomeq-export/src/roundtrip.rs`) verify biquad coefficients,
routing, preamp normalization, delay, and convolution bytes against the
canonical graph. Demo the gates and round trips without listening
evidence via `roomeq-qa-synthetic --release-gates`.

### Multi-sub measurement resolution and output preservation

Bass measurements ending at or below 500 Hz retain every native sample and
measured phase. Full-range measurements retain their native bass samples while
using the configured reduced grid above 500 Hz. MSO, DBA, and spatial MSO retain
all measured frequency points in their common supported band. Interpolating a
summed magnitude/phase curve is not equivalent to summing individual responses;
optimization and routed reconstruction sum the drivers on the receiving grid.
The measurement resolution remains the limit on knowledge of unmeasured nulls.

A sub measurement ending at 200 Hz does not truncate full-range main analysis.
For prediction above a tail at least 24 dB below its measured peak and falling
at least 12 dB/octave, the model continues a falling envelope (capped at
48 dB/octave). An energetic endpoint is held conservatively. Sub alignment and
EQ remain inside the measured, useful bass band; this prediction extension is
not new measurement evidence and is never used to extend the sub EQ band.

The useful upper bound is the last measured sample within 20 dB of the in-band
peak. An internal room null therefore does not truncate later useful bass.
Crossover evaluation includes the native sub samples even when the receiving
main measurement has a coarser grid.

Independent subs driven by the same input are summed coherently when measured
phase is available. Without phase, preprocessing explicitly reports a power-sum
approximation; phase-critical route verification requires suitable phase data.
All-pass and multi-seat routed processing use the dedicated sub engine, preserve
per-driver PEQ/all-pass, polarity and primary-seat measurements, and retain the
configured global-EQ policy instead of adding another single-seat sub EQ pass.

DBA, ordinary/all-pass MSO and continuous-area scalarizations penalize loss of
useful output, deep new nulls, low-band loss and unnecessary gain. DBA here
optimizes measured magnitude with an inverted rear array; room geometry and
late-energy measurements are still needed to establish rear-wall absorption.
The cardioid path is a fixed geometric delay/inversion recipe; matching and
directional measurements are needed to establish rear rejection. Its legitimate
low-frequency efficiency loss is not treated as an MSO defect.
`primary_with_constraints.max_deviation_db` is a soft penalty threshold, not a
hard guarantee at every seat and frequency.
## Prepared FIR target grid contract

The internal prepared-FIR API requires target frequencies to match the
measurement frequency grid exactly, with one level per frequency in each curve.
It rejects mismatches before pointwise boost capping or filter design. Callers
with independent target grids must align them during target preparation; changing
array order or silently pairing samples by index is not a resampling operation.

## Final-graph electrical headroom assessment

The output metadata includes a `final_graph_sampled_electrical_headroom` stage
after final level alignment, CTC and convolution artifact binding. This is
separate from acoustic/correction acceptance: an accepted correction can still
need an explicit playback gain budget.

The assessment assumes independently phased sinusoidal logical inputs, each
with peak amplitude 1.0. It evaluates complete serialized input/route/output
transfers at 8193 points from DC to Nyquist, plus serialized EQ and crossover
centers. Per-output checks report the sampled amplitude against full scale;
their diagnostics include peak frequency, required attenuation, grid size and
input limits. A `degraded` stage identifies sampled overload or unavailable
electrical evidence. Unsupported global processing or unresolved sidecars is
not silently omitted and does not receive a passing electrical check.

This report does not apply attenuation, alter the configured correction
acceptance policy, or certify frequencies between samples, transient/true-peak
headroom, native-backend agreement, acoustic benefit or device assignment.
Required attenuation must be reconciled with calibrated useful-output goals;
it is not a recommendation to attenuate blindly.

## Per-driver FIR placement (input schema 2.2.0)

Set `optimizer.fir.placement` to `per_driver` to try separate FIRs for the
physical speakers in each independent main/sub group. The default, `shared`,
is unchanged. Phase mode is independent of placement. See
[the input contract](../src/bin/roomeq/INPUT_FORMAT.md#configuration-schema-version)
for the JSON fragment, supported group types, safeguards and limitations.

The engine retains crossover, gain, delay and intentional IIR stages, then
evaluates complete sets of per-driver FIRs against the calibrated acoustic sum.
It does not flatten every physical speaker towards the full-range target.
Each exported driver's plugin list references its own WAV; load all of them on
their respective physical outputs. Mixed-phase FIRs remain phase-only, and all
branches have a common causal support budget. Minimum-phase FIRs remain causal
without imposing a linear-phase centering delay.

Conservative null masks prevent FIR boost into deep local dips, low-coherence
bins and low signal/noise regions. A remaining target deficit at a protected
null is intentional, not an invitation to increase boost. Single-position
measurements cannot definitively distinguish all SBIR/room nulls. Relative
main/sub phase correction can help interference, but cannot guarantee room-wide
improvement. Validate other seats and listen at matched levels.

The measured `2.2_sigberg2/optimiser-fir.json` opts in. Use `placement: shared`
for an A/B run in a separate output directory. Compare the complete replayed
response, not an individual driver's SPL against the whole-system target.
Automatic shared/per-driver selection is not implemented. Routed systems with
shared physical subs are explicitly rejected by this initial option.
