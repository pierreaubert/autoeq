# roomeq-quality

Acoustic quality metrics and acceptance policy for RoomEQ results.

## Ownership

- Owns corpus-independent quality metrics, scorecards, correction acceptance, and regression policy.
- Does not own production pipeline execution, resource I/O, or scenario orchestration.

## Testing

```bash
cargo test -p roomeq-quality --lib
```
