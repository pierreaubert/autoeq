# autoeq-report-wasm-shell

WASM export shell for the AutoEQ 2D report renderer.

## Ownership

- Owns the `cdylib` bundle built by `just report-dist`: the `wasm-bindgen`
  exports (`render_section`, `legend_json`, `toggle_series`, `last_error`,
  `schema_version`) driving the checked-in `report2d` bundle.
- Draws with the shared draw core and schema from `autoeq-report-wasm`.
- Does not own the report payload schema, the HTML shell assembly, or any
  host-side rendering. Nothing may depend on this crate: `cdylib`-only is
  load-bearing (see `Cargo.toml`).

## Testing

```bash
cargo test -p autoeq-report-wasm-shell --lib
cargo build --release --target wasm32-unknown-unknown -p autoeq-report-wasm-shell
```
