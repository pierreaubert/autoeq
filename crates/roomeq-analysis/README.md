# roomeq-analysis

Deterministic, in-memory measurement analysis for RoomEQ.

## Ownership

- Owns frequency-grid, impulse, timing, spatial, slope, crossover, and robustness analysis primitives.
- Does not own optimization execution, resource loading, QA policy, or application workflows.

## Testing

```bash
cargo test -p roomeq-analysis --lib
```
