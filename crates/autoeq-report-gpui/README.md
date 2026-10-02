# autoeq-report-gpui

Optional GPUI explorer for AutoEQ report payloads. It reuses the versioned
schema and renderer contracts from `autoeq-report-wasm`, then presents report
sections as filterable cards with per-series counts and statistics. The
explorer model is host-testable; the browser boot and view are compiled for
`wasm32` only.

The regular Canvas report remains available without WebGPU. This crate adds
the GPUI view for browser environments that support the required WebGPU path.
Run the host-side model tests with:

```bash
cargo test -p autoeq-report-gpui --locked --lib
```
