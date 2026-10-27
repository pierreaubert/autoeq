# GPUI report viewer ownership

`autoeq-report-gpui` consumes the same versioned payload schema as the default
report renderer. Its host-testable `ExplorerModel` derives section cards,
tab filters, and series statistics. It does not own optimizer execution or a
second report schema.

GPUI application boot, browser DOM integration, and rendering are wasm32-gated.
Their GPUI and web dependencies are target-specific in `Cargo.toml`; native
library tests exercise the model without starting that browser runtime.
The package currently emits both `rlib` and `cdylib` artifacts.

The HTML shell loads the optional viewer when WebGPU is available. The ordinary
Canvas report remains the default path. Product validation therefore includes
both host model tests and a real browser mount of the nightly-built GPUI bundle;
a successful host compilation does not prove that mount works.

See the [usage guide](README.md), [shared renderer architecture](../autoeq-report-wasm/ARCHITECTURE.md),
and [workspace architecture](../../docs/ARCHITECTURE.md).
