# Shared report renderer ownership

Despite its historical name, `autoeq-report-wasm` is the shared Rust library,
not the WebAssembly export shell. It owns the versioned payload schema,
prepared report data, serialized-chain replay used by reports, section and
figure construction, HTML assembly, and platform-independent drawing geometry.
The explicit public exports are declared in `src/lib.rs`.

`autoeq-plot` and the browser viewers consume this library. Browser exports
belong to `autoeq-report-wasm-shell`; GPUI boot and view code belong to
`autoeq-report-gpui`. The renderer depends on `autoeq-core` and `gpui-d3rs`
with default features disabled, keeping host windowing out of its geometry
boundary. Its separate `report-producer` binary supports report fixtures.

The default report draws through Canvas 2D. Where available, the browser shell
can accelerate projected surface triangles with WebGPU while retaining shared
layout, colors, and interaction. Payload validity and DSP evidence remain
renderer contracts rather than browser-only checks.

Library tests cover host behavior; product validation also builds the stable
WASM shell and nightly GPUI viewer and exercises their browser interactions.
See the [payload schema](SCHEMA.md), [usage guide](README.md), and
[workspace architecture](../../docs/ARCHITECTURE.md).
