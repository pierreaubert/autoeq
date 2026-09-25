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

```json
{
  "schema": "autoeq-report-data-v1",
  "title": "page title",
  "sections": [ ... ]
}
```

The shell refuses to render when `schema` does not match
`autoeq-report-wasm::schema_version()`.

## Sections

Tagged on `kind`: `html`, `figure`, `bar`, `sankey`. Every section carries
an optional `tab` string: when any section has one, the shell groups
sections under a tab bar (roomeq per-channel tabs); renderers ignore `tab`.

### `html`

```json
{ "kind": "html", "html": "<raw html>", "tab": null }
```

Raw HTML (tables, summaries, filter lists). Inserted verbatim; producers
must escape untrusted text before building it.

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

## Versioning

Breaking changes require a new discriminator (`autoeq-report-data-v2`)
plus shell support for both during transition. Additive optional fields do
not bump the version.
