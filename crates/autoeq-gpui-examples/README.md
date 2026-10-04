# AutoEQ GPUI demos

This crate owns the Spinorama speaker-data demos built on the sibling
`gpui-toolkit` plotting libraries. Its separate workspace keeps those local
dependencies out of backend installation and QA dependency resolution.

From the AutoEQ checkout, build them with `just demo-d3rs-spinorama` and
`just demo-px-spinorama`. Run either demo with:

```sh
cargo run --manifest-path crates/autoeq-gpui-examples/Cargo.toml --bin d3rs-spinorama --release
cargo run --manifest-path crates/autoeq-gpui-examples/Cargo.toml --bin px-spinorama --release
```

The D3RS demo's interpolation and rendering unit tests remain beside its
source modules. Run them with:

```sh
cargo test --manifest-path crates/autoeq-gpui-examples/Cargo.toml --bin d3rs-spinorama
```
