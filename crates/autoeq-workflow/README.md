# autoeq-workflow

Speaker and headphone AutoEQ application workflows.

## Ownership

- Owns speaker/headphone use cases and composition of measurement, optimizer, artifact, and plot services.
- Does not own CLI parsing, RoomEQ orchestration, or low-level numerical implementations.

`optimize_speaker_at_cache_root` lets callers isolate measurement acquisition
without mutating the process environment; the existing entry point preserves
platform cache discovery for production use.

## Testing

```bash
cargo test -p autoeq-workflow --lib
```
