# roomeq-dsp

Shared DSP realization helpers used by both `roomeq-engine` and
`roomeq-export`: band-limited fractional-delay FIR realization
(`fir`), physical routing resolution (`physical_routing`), and
portable plugin-config construction (`output`).

Extracted from `roomeq-engine` (refactor-20261025 Phase 3d) so
exporters do not compile the optimization engine.
`roomeq-engine` re-exports every module here.
