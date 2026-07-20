# roomeq-qa

RoomEQ scenario matrices, runners, scorecards, and QA command implementations.

## Ownership

- Owns acoustic, coverage, convergence, feature, fuzzer, and synthetic QA orchestration.
- Exercises public production APIs but does not own production pipeline behavior.

## Testing

```bash
cargo test -p roomeq-qa --lib
```
