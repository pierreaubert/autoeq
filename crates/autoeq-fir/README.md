# autoeq-fir

Pure FIR design and serialization primitives for automatic equalization.

## Ownership

- Owns FIR construction and in-memory WAV/byte encoding helpers.
- Does not own filesystem placement, workflow orchestration, or RoomEQ policy.

## Testing

```bash
cargo test -p autoeq-fir --lib
```
