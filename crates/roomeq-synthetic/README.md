# roomeq-synthetic

Deterministic synthetic measurements and scenarios for RoomEQ testing.

## Ownership

- Owns reusable synthetic curves, rooms, signals, and scenario-building primitives.
- Does not own production workflows, optimization policy, or QA pass/fail thresholds.

## Testing

```bash
cargo test -p roomeq-synthetic --lib
```
