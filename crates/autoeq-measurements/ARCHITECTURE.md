# autoeq-measurements — Architecture

Measurement sources, loading, preprocessing, and speaker metrics.
Turns files, directories, and remote APIs into validated `Curve`s.

## Layer

I/O and conditioning layer over `autoeq-core`. Adds `reqwest` (download APIs),
`directories` (cache locations), `strsim` (speaker-name matching).

## Key modules

| Module | Owns |
|---|---|
| `read/` | Format readers: `read_csv` (header-driven 2–4 col CSV **and** REW-style `*`-comment TXT; phase/coherence/noise-floor columns), `directory`, `source` (strict multi-file loader with shared-grid alignment), `read_api` (Spinorama HTTP API) |
| `read/` (conditioning) | `clamp`, `interpolate` (log-space resampling), `normalize`, `smooth` (1/N-octave) — mismatched grids are aligned/resampled explicitly, never zipped by index |
| `cea2034/` | CEA-2034 spinorama parsing and score inputs |
| `mic_phase_calibration` | Microphone phase-calibration loading |
| `provenance` | Measurement origin/rights tracking |
| `quality` | Curve-quality flags consumed by QA |

## Core abstractions

- **Two reader tiers.** `load_driver_measurement` is header-name driven
  (freq/spl/phase/coherence/noise-floor columns in any order, BOM-tolerant,
  headerless positional fallback); `load_frequency_response` handles legacy
  2-col and 4-col stereo-average files. `read_curve_from_csv` tries the
  driver reader first, falls back to the legacy one, then sorts by frequency
  (rejecting non-finite/duplicate bins) and populates the min/excess-phase
  cache.
- **`load_source` family.** `load_source(&MeasurementSource) -> Curve`
  averages multi-file sources onto a `common_overlap_grid` clipped to the
  physical intersection of supports — disjoint measurements fail instead of
  extrapolating. Variants (`_detailed`, `_with_individual`,
  `_individual_with_support`, `coherent_average_measurement`) expose
  per-curve evidence; `load_measurement_strict` fails closed on any bad file.
- **Comment tolerance.** `#`, `//` lines are skipped structurally; anything
  else unparseable (e.g. REW `*` headers) is skipped by parse failure, so
  REW exports load as freq+spl with phase dropped.

## API

```rust
// Single curve, full pipeline: parse → sort → phase-cache.
let curve: Curve = read_curve_from_csv(&PathBuf::from("L.csv"))?;

// Tracked record (adds origin + source-path identity for QA evidence).
let record: MeasurementRecord = read_record_from_csv(&path)?;

// Multi-file source averages onto the physical overlap grid;
// strict variants fail closed on any bad file.
let avg: Curve = load_source(&source)?;
let strict: Curve = load_measurement_strict(&mref)?;
```

## Data contracts

- **In:** CSV/TXT/WAV paths, `MeasurementRef`s, Spinorama speaker IDs.
- **Out:** `Curve`s on validated, strictly-monotonic grids plus provenance records.

## Consumers

`autoeq-optim` (loss evaluation), `autoeq-workflow`, `roomeq-engine`,
`roomeq-workflow` (via `group_measurements`), `autoeq-plot`.
