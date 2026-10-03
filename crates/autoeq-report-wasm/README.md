# autoeq-report-wasm

Shared AutoEQ report payload types and the platform-independent renderer core.
The crate owns the versioned JSON schema in `src/schema.rs`, HTML assembly,
Canvas 2D drawing, and grid rendering. `autoeq-plot` produces payloads using
these types. The renderer uses `d3rs` for scales, axes, colors, and Sankey
layout, with Canvas as the fallback when WebGPU is unavailable.

This crate is an `rlib` and does not own the WebAssembly export bundle. The
separate `autoeq-report-wasm-shell` crate provides the `cdylib` exports, while
`autoeq-report-gpui` provides the optional explorer viewer.

The payload contract and examples are documented in
[`SCHEMA.md`](SCHEMA.md). Run its library tests with:

```bash
cargo test -p autoeq-report-wasm --locked --lib
```
