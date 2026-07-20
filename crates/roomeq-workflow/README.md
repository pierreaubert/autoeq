# roomeq-workflow

RoomEQ application workflows and resource adapters.

## Ownership

- Owns configuration loading, path resolution, measurement/resource preparation, cache/resume, artifact persistence, and calls into engine/export crates.
- Does not own deterministic DSP algorithms, stable data contracts, external format rendering, or CLI parsing.

## Testing

```bash
cargo test -p roomeq-workflow --lib
```
