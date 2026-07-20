# autoeq-workflow

Speaker and headphone AutoEQ application workflows.

## Ownership

- Owns speaker/headphone use cases and composition of measurement, optimizer, artifact, and plot services.
- Does not own CLI parsing, RoomEQ orchestration, or low-level numerical implementations.

## Testing

```bash
cargo test -p autoeq-workflow --lib
```
