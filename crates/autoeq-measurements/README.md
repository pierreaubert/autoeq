# autoeq-measurements

Measurement acquisition, parsing, calibration, and preprocessing for AutoEQ.

## Ownership

- Owns CSV/API loading, provenance, normalization, interpolation, smoothing, and CEA-2034 acquisition.
- Does not own optimizer backends, application orchestration, or duplicate measurement contracts.

Cache discovery remains the production default. Tests and isolated workflows
can use the `*_at_cache_root` acquisition APIs and the pure
`data_dir_for_cache_root`/`headphone_cache_dir_for_cache_root` path helpers
without changing process-global environment variables.

## Testing

```bash
cargo test -p autoeq-measurements --lib
```
