# roomeq-model

Stable configuration, validation, result, and DSP-graph contracts for RoomEQ.

## Ownership

- Owns neutral RoomEQ data models, serde/schema contracts, validation reports, and topology policy types.
- Does not own resource I/O, optimizer execution, export rendering, or workflow orchestration.

## Testing

```bash
cargo test -p roomeq-model --lib
```
