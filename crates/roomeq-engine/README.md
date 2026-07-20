# roomeq-engine

Prepared-input RoomEQ analysis, optimization, and DSP pipeline execution.

## Ownership

- Owns deterministic in-memory channel, topology, CTC, bass-management, FIR, and DSP-graph execution.
- Does not own filesystem/network I/O, CLI parsing, external export formats, or QA runners.

## Testing

```bash
cargo test -p roomeq-engine --lib
```
