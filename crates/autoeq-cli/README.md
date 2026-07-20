# autoeq-cli

Command-line adapters for user-facing AutoEQ operations.

## Ownership

- Owns argument parsing, command selection, startup, and user-facing diagnostics.
- Delegates application behavior to `autoeq-workflow`; it does not own optimization algorithms.

## Testing

```bash
cargo test -p autoeq-cli --lib
```
