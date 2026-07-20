# roomeq-export

Pure external-format rendering and package construction for RoomEQ DSP graphs.

## Ownership

- Owns CamillaDSP, Equalizer APO, PipeWire, Roon, REW, Wavelet, EasyEffects, coefficient, and sidecar packaging.
- Returns deterministic in-memory package members; it does not perform production filesystem I/O or optimization.

## Testing

```bash
cargo test -p roomeq-export --lib
```
