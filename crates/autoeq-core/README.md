# autoeq-core

Core, I/O-free types and numerical primitives shared by AutoEQ and RoomEQ.

## Ownership

- Owns curves, measurement descriptors, PEQ models and layouts, filter conversion, and response math.
- Does not own measurement acquisition, optimizer backends, application workflows, plotting, or CLI behavior.

## Testing

```bash
cargo test -p autoeq-core --lib
```
