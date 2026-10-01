# Report payload schema (`autoeq-report-data-v1`)

Both report producers — the Rust emitter (`autoeq-plot`, via
`autoeq-report-wasm`) and the Python emitter (`scripts/src/wasm_report.py`) —
build this exact JSON document. It is embedded in the HTML shell
(`crates/autoeq-report-wasm/shell/template.html`, `{{PAYLOAD_JSON}}`); the
WASM renderers draw from it and never recompute curves (plan decision D3).

The authoritative field definitions are the Rust types in
`crates/autoeq-report-wasm/src/schema.rs` (serde JSON). This file is the
human-readable contract.

## Top level

The HTML shell accepts optional `bucket` names on sections, rendering one
centered tab per bucket (`DSP analysis`, `Acoustics analysis`,
`Psychoacoustic report`, in that canonical order); unbucketed sections stay
visible above the tabs. Inside a bucket with more than one `subhead`, the
subheads render as tabs sharing the channel-tab styling; a lone subhead
renders directly with no tab bar. `tab` selectors are independent inside
each subhead page; untabbed sections render first. A
`footer: true` section follows its subhead's tab pages, and `flat: true` on
an `html` section skips the shell card chrome for self-framed content.
Optional `group` titles are the legacy equivalent of buckets, grouping
sections into collapsible panels in first-occurrence order; producers should
emit `bucket`/`subhead` instead. These are shell layout hints, ignored by
the Rust chart renderer. Grid `rotation: [elevation, azimuth]` is optional
display-only d3rs camera state in degrees; Reset restores `[65, -12]`.

```json
{
  "schema": "autoeq-report-data-v1",
  "title": "page title",
  "sections": [ ... ],
  "provenance": {
    "roomeq_version": "0.5.74",
    "data_timestamp": "2026-10-01T11:00:00Z",
    "generated_at": "2026-10-01T11:05:00Z"
  }
}
```

`provenance` is optional and additive (omitted payloads render as before).
When present, the shell shows one header line above the renderer status —
`RoomEQ <roomeq_version>`, `Data <data_timestamp>`, `Report <generated_at>`
for whichever fields are set — using text assignment, so values are never
interpreted as HTML. `data_timestamp` is the DSP emission moment
(`metadata.timestamp`); `generated_at` is the HTML render moment.

The shell refuses to render when `schema` does not match
`autoeq-report-wasm-shell::schema_version()`.

## Sections

Tagged on `kind`: `html`, `figure`, `grid`, `bar`, `sankey`. Every section carries
an optional `tab` string: when any section has one, the shell groups
sections under a tab bar (roomeq per-channel tabs); renderers ignore `tab`.

### `html`

```json
{ "kind": "html", "html": "<raw html>", "tab": null }
```

Raw HTML (tables, summaries, filter lists). Inserted verbatim; producers
must escape untrusted text before building it.

The shell also accepts `footer: true` to place an HTML section after the tabs.
HTML may contain `.report-figure` placeholders whose `data-section` attribute
contains an HTML-escaped JSON figure/grid section. The shell mounts those through
the same WASM renderer, inheriting the enclosing tab. Producers must escape the
entire attribute value, not interpolate raw JSON into HTML.

### `grid`

Carries `figure` axis/title/legend metadata and `grid` data:
`x` (positive increasing frequencies), `y` (increasing times in ms),
`z` (finite rectangular values indexed `[time][frequency]`), `surface`
(true for a projected surface view, false for a heatmap), and `zmin`/`zmax`
(fixed colour/display range). Optional `highlights` contains frequency-column
indices; matching `figure.series` entries provide names, colours and visibility.
Surface views accept `colormap` (turbo, viridis, plasma, inferno, magma,
rainbow; unknown names fall back to turbo), `show_surface` (fill visibility,
default off) and `show_contours` (time-slice contours, default on), so the
default surface is an REW-style contour wireframe with ridge highlights.

Both axes require at least two values. Arrays are limited to 4096 per axis and
one million cells. Invalid grids render an explicit unavailable message.
Figure bounds control the viewport without changing the stored level reference.
Surface geometry/projection and axes come from d3rs; the WASM Canvas renderer
does not require a WebGPU adapter. Grid views support zoom, pan, reset and
highlight legend toggles; fractional-octave smoothing is not applied to grids.

### `figure`

Cartesian line chart. `x.scale` is `log` or `linear`; `y` is always linear.

```json
{
  "kind": "figure",
  "figure": {
    "title": "Response",
    "x": { "label": "Frequency (Hz)", "scale": "log", "min": 20, "max": 20000 },
    "y": { "label": "SPL (dB)", "min": null, "max": null },
    "y2": null,
    "series": [
      { "name": "Input", "x": [...], "y": [...],
        "color": null, "width": 2.0, "dash": "solid", "visible": true, "y_axis": 0 }
    ],
    "hlines": [{ "at": 0, "color": "rgba(150,150,150,0.4)", "dash": "dash", "width": 1, "label": null }],
    "vlines": [],
    "xranges": [{ "x0": 20, "x1": 30, "color": "rgba(144,238,144,0.3)" }],
    "annotations": [{ "x": 1000, "y": 3, "text": "note" }],
    "legend": true
  },
  "tab": null
}
```

- `series[].x` / `y` must have equal length; non-finite samples are skipped.
  Log axes ignore `x <= 0` samples.
- `y` entries may be `null`: a null sample is a gap — the renderer breaks
  the line there (the old plotly `connectgaps: false` behavior) and ignores
  nulls when auto-scaling.
- `color: null` selects the `d3rs` category10 slot for the series index.
- `dash` is one of `solid`, `dash`, `dot`, `dashdot`.
- `min`/`max: null` means "data extent with padding".
- The legend is clickable client-side and toggles `visible`.
- `y_axis: 1` maps a series onto the secondary right axis `y2`
  (e.g. directivity index overlaid on SPL). Without `y2`, such series read
  as primary. Reference lines and annotations always use the primary axis.
- `hlines` entries may carry an optional `label`, drawn at the right plot
  edge (e.g. a headroom limit). Bar charts accept `hlines` too.
- A primary-y label of exactly `SPL (dB)` selects the audio SPL grid:
  labelled majors on an adaptive 1/2/5 stride plus a minor horizontal
  gridline every integer dB. Other labels keep the default ~6 ticks.

### `bar`

Grouped categorical bar chart.

```json
{
  "kind": "bar",
  "chart": {
    "title": "Scores",
    "categories": ["a", "b"],
    "groups": [{ "name": "Before", "values": [3.1, 4.2], "color": null }],
    "ylabel": "score",
    "ymin": null,
    "ymax": null,
    "legend": true,
    "hlines": []
  },
  "tab": null
}
```

`values[i]` aligns with `categories[i]`; short vectors read as 0.

### `sankey`

Flow diagram (bass-management routing).

```json
{
  "kind": "sankey",
  "chart": {
    "title": "Routing",
    "nodes": ["in", "xover", "out"],
    "links": [{ "source": 0, "target": 1, "value": 2.0, "color": null }]
  },
  "tab": null
}
```

`source`/`target` index into `nodes`. Non-positive or dangling links are
dropped. Layout comes from the `d3rs` Sankey implementation.
Optional `chart.node_boxes: true` draws fixed-height plugin boxes connected by
arrows instead of quantity-width bands. It defaults to `false`. The HTML shell
accepts section-level `min_width` in CSS pixels for horizontally scrollable
signal-flow diagrams and `min_height` for parallel-channel spacing (both capped
at 6000 pixels). These affect canvas size, not the graph's processing topology.

## Versioning

Breaking changes require a new discriminator (`autoeq-report-data-v2`)
plus shell support for both during transition. Additive optional fields do
not bump the version.
