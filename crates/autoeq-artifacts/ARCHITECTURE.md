# autoeq-artifacts — Architecture

Artifact storage contracts for reports, exports, and sidecars.

## Layer

Ports-and-adapters boundary: the `ArtifactStore` trait (create dirs, atomic
write, read) with `FsArtifactStore` as the production filesystem
implementation. Keeps workflow/engine code testable without touching disk.

## Key modules

| Module | Owns |
|---|---|
| `lib` | `ArtifactStore` trait + `FsArtifactStore` |
| `roomeq` | Deterministic RoomEQ convolution-sidecar naming and reservation |

## Core abstractions

- **`ArtifactStore: Send + Sync`.** `create_dir_all`, `write(path,
  contents)`, `read(path) -> Option<Vec<u8>>`. Tests inject an in-memory
  fake; production passes `FsArtifactStore`. Nothing in the DSP path knows
  which one is wired.
- **Sidecar reservation.** `roomeq` helpers hand out deterministic
  convolution-sidecar paths so parallel channel writers never collide and
  reruns overwrite byte-identically.

## API

```rust
let store = FsArtifactStore::new();
store.create_dir_all(&out_dir)?;
store.write(&out_dir.join("dsp-iir.json"), &json_bytes)?;
let back: Option<Vec<u8>> = store.read(&path)?;
```

## Consumers

`roomeq-workflow` (pipeline artifact writes), export flows.
