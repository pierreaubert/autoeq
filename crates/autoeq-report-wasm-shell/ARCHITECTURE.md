# Browser export shell ownership

`autoeq-report-wasm-shell` is a terminal, `cdylib`-only package. It adapts the
shared renderer's drawing interface to browser Canvas 2D, retains per-canvas
state for legend interactions, and exposes the `wasm-bindgen` report API.
Payload schema and drawing geometry remain in `autoeq-report-wasm`.

The HTML assembly drives exports including `render_section`, `legend_json`,
`toggle_series`, `crossover_config_html`, `shared_sections_json`, `last_error`,
and `schema_version`. Browser state and DOM handles stay at this boundary.
The optional triangle compositor uses the geometry supplied by the shared core.

No Rust package may depend on this shell. Keeping the reusable API in the
renderer library avoids mixing reusable `rlib` artifacts with export `cdylib`
artifacts across release-abort and test-unwind builds.

`just report-dist` builds the stable wasm32 bundle and generated glue into the
renderer assets. Host tests cannot establish browser drawing behavior; validate
the resulting report interactions as well. See the [usage guide](README.md) and
[workspace architecture](../../docs/ARCHITECTURE.md).
