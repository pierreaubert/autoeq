# Roon export and application validation

`--export-format roon` writes `room_eq.json`, a versioned manual setup manifest.
It is not an importable Roon preset. The legacy `channels`,
`headroom_gain_db`, `delay_ms`, `parametric_eq`, and `convolution` fields remain;
`procedural_eq.operations` gives the ordered Volume, Delay, and Parametric EQ
operations to enter in Roon MUSE.

When a chain contains convolution, the path-aware exporter also writes
`room_eq_convolution.zip`. The archive contains one Convolver `.cfg` for the
selected sample rate and layout, relative mono WAV paths, and explicit
one-input-to-one-output routes. Its hexadecimal speaker mask follows canonical
WAVE order. Channels without FIR correction get a same-length unity impulse so
they are not silenced. Export fails on unknown or duplicate channel positions,
unsafe or absolute paths, malformed/non-mono/rate-mismatched WAVs, unequal IR
lengths, and multiple serial convolution plugins.

Portable validation runs with:

```bash
just qa-export-portable
```

The licensed application test is macOS-only, interactive, and excluded from
hosted CI:

```bash
just qa-export-roon-setup
ROOMEQ_ROON_MANIFEST=/absolute/path/room_eq.json just qa-export-roon-app
```

Setup creates a mode-0700 directory under
`~/Library/Application Support/SotF/RoonExportQA`. Roon login/license state is
never automated. The official extension API is used only for exact-zone
discovery and playback control; it has no MUSE configuration API. Persisted
authorization, captures, and diagnostics remain in the private directory.

The repository supplies an accessibility/OCR driver that refuses every action
unless the exact QA zone is uniquely visible. Because Roon labels and control
hierarchies change between releases, setup also requires a version-calibrated
private action executable through `ROOMEQ_ROON_UI_APPLY_CMD`. It receives only
the QA zone, mode, manifest, and archive paths and must implement `baseline`,
`iir`, `fir`, `combined`, and `cleanup`. This keeps private screenshots and
machine-specific accessibility calibration outside the repository.

The app recipe plays the quiet deterministic 48 kHz sweep, captures BlackHole
with FFmpeg, and checks IIR, FIR, and combined magnitude against the manifest
and archive to 0.5 dB from 30 Hz to 18 kHz. It checks relative delay to two
samples, records Roon version and textual AX/OCR diagnostics, deletes transient
OCR screenshots, stops playback, and invokes cleanup after every outcome.
It never uses an approximate zone match or falls back to another zone.

A future self-hosted Mac job must run only in an interactive logged-in session
on scheduled or manual protected-branch events.
