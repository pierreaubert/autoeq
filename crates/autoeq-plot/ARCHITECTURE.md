# autoeq-plot — Architecture

Plotly visualizations and report generation for AutoEQ
(HTML reports, spin/driver/filter plots, static export).

## Layer

Presentation leaf over `autoeq-core` + `autoeq-measurements` + `autoeq-optim`.
Optional `resvg` backend renders static images.

## Key modules

| Module | Owns |
|---|---|
| `plot_results`, `plot_filters`, `plot_drivers` | Curve/result/filter figures |
| `plot_spin` | CEA-2034 spinorama views |
| `static_export` | Headless PNG/SVG export |
| `config`, `filter_color`, `ref_lines`, `trend_lines` | Shared styling |

## Core abstractions

- **Figures are values.** Each `plot_*` module builds a `plotly::Plot` from
  curves + filters; nothing is written until the caller chooses HTML
  (`to_html`) or a static raster (`static_export` via `resvg`).
- **Shared styling.** `filter_color` assigns stable per-filter colors,
  `ref_lines` draws target/0 dB references, `trend_lines` overlays
  regression slopes — so every report reads the same.

## API shape

```rust
// One call builds the figure set from domain types (async: spin data may load).
let (main, filters, spin, extra) = plot_results::plot_compute(
    &config, &optimized_params, &input, &target, &deviation, &cea2034,
).await;

// Or a single panel: combined PEQ response over the input grid.
let fig: plotly::Plot =
    plot_filters::plot_filters(&config, &input, &target, &deviation, &params);
std::fs::write("report.html", fig.to_html())?; // interactive; static_export covers PNG/SVG
```

## Consumers

`autoeq-cli` (optional). The RoomEQ HTML path used by
`ui/display-roomeq` is separate (`ui/display-roomeq/report.py`); see the
RoomEQ display flow in `docs/ARCHITECTURE.md`.
