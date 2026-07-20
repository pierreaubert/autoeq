# autoeq-artifacts

Artifact storage ports and implementations used by AutoEQ and RoomEQ workflows.

## Ownership

- Owns artifact-store contracts plus filesystem and in-memory storage adapters.
- Does not own acoustic policy, optimization, export rendering, or artifact contents.

## Testing

```bash
cargo test -p autoeq-artifacts --lib
```
