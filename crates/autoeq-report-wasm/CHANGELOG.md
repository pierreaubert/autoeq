# Changelog

## Unreleased

- Early-reflection candidate figures now plot the measured 1–8 kHz
  envelope behind the discrete candidate stems, and predicted-after-EQ
  candidate figures are derived from the saved predicted post-EQ IR
  when one exists.
- Render the Crossover Configuration summary from deployed channel, driver,
  and route metadata in the shared Rust/WASM producer. Driver crossover rows
  now remain visible when measured curves are stored externally or absent.
- Documented the report payload schema, shared rendering boundary, and
  separation between the renderer library and WebAssembly export shell.
