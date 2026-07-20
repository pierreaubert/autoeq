# roomeq-cli

Command-line adapters for RoomEQ and recording-conversion operations.

## Ownership

- Owns argument parsing, command selection, startup, and user-facing diagnostics.
- Delegates production behavior to `roomeq-workflow`; it does not own DSP algorithms or QA policy.

## Testing

```bash
cargo test -p roomeq-cli --lib
```
