# autoeq-optim

Objectives, constraints, and optimizer backends for automatic equalization.

## Ownership

- Owns objective construction, parameter bounds, constraints, optimizer registries, and optimization evidence.
- Does not own filesystem/network acquisition, plotting, CLI parsing, or application workflows.

## Testing

```bash
cargo test -p autoeq-optim --lib
```
