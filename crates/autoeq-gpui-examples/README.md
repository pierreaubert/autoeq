# AutoEQ GPUI demos

This crate owns the Spinorama speaker-data demos built on the sibling
`gpui-toolkit` plotting libraries.

From the `autoeq` workspace, build them with `just demo-d3rs-spinorama` and
`just demo-px-spinorama`. Run them with
`cargo run -p autoeq-gpui-examples --bin d3rs-spinorama --release` and
`cargo run -p autoeq-gpui-examples --bin px-spinorama --release`.

The D3RS demo's interpolation and rendering unit tests remain beside its
source modules and run as binary-target tests in the AutoEQ workspace.
