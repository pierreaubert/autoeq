# autoeq-measurements

Measurement acquisition, parsing, calibration, and preprocessing for AutoEQ.

## Ownership

- Owns CSV/API loading, provenance, normalization, interpolation, smoothing, and CEA-2034 acquisition.
- Does not own optimizer backends, application orchestration, or duplicate measurement contracts.

## Testing

```bash
cargo test -p autoeq-measurements --lib
```
