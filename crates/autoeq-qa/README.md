# autoeq-qa

Independent cross-validation of AutoEQ / RoomEQ against the Wolfram Engine.

Each case pairs a closed-form Wolfram oracle (`wolfram/*.wls`) with a
Rust comparison test (`tests/wolfram_*.rs`). References resolve live
when `WOLFRAMSCRIPT` is set, otherwise from checked-in goldens
(`wolfram/goldens/`, produced with `just qa-wolfram-goldens`).

Covered (catalogue families from `reviews/catalogue-20260923.md`):
C01 PEQ + FIR complex transfer, C03 log-frequency interpolation,
C06 ERB quadrature and timing conversion.

See [AGENTS.md](AGENTS.md) and `validation-manifest.toml` for the case
list, tiers, and tolerances.
