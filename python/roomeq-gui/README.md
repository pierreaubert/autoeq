# RoomEQ GPUI

`autoeq-roomeq-gui` is the native GPUI client for the RoomEQ JSON workflow. It edits complete input schemas, asks the existing `roomeq` executable to validate and optimize, and opens output JSON for review. It does not build Cargo or reimplement RoomEQ acoustics.

Install locally with `python -m pip install ./python/roomeq-gui`. Build the binary separately with `just prod-roomeq`, then run `roomeq-gui [--roomeq PATH] [--config FILE] [--result FILE]`.

Load an existing configuration directly at startup with:

```bash
roomeq-gui --config path/to/room.json
```

The app finds `ROOMEQ_BIN`, `roomeq` on `PATH`, then `target/release/roomeq` or `target/debug/roomeq`. It obtains both schemas from that binary and displays a compatibility warning when it falls back to bundled baselines. Configurations are never autosaved: Save/Save As is required before validation or optimization; output defaults to `<config-stem>.result.json` beside the configuration. Relative measurement paths retain the configuration directory as their base.
