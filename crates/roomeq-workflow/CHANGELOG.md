# Changelog

## Unreleased

- CTC runs with room-seat captures now go through ordinary final
  selection: finalization, seat replay, and the configured-reference
  comparison no longer refuse or void their evidence solely because a
  CTC artifact exists (the global CTC stage keeps its own report and
  electrical bound). The 5.0.4 Genelec modes deliver `safe_fallback`
  DSP chains instead of diagnostic-only errors.
- Carry each branch's deployed FIR latency prior into acceptance-replay
  scorecards so induced group delay unwraps against the real bulk delay
  instead of a zero prior on sparse grids.
- Strip stale staged final-channel alignment in post-DSP recalibration
  before re-deriving input trims, fixing a ~5.5 dB useful-output
  transplant loss on routed graphs.
- Release FIR coefficients and temporal masking on Hybrid-stage reverts,
  and rejoin reverted branches to their crossover partner's surviving
  bulk with a standalone latency alignment that tracks the live gap
  across safety gates.

- Optimize stereo 2.1/2.2 bass routing independently from home-cinema bass
  management: evaluate dual-mono, direct-pair, and crossed-pair matrices, reject
  candidates that fail serialized-graph splice or electrical-headroom replay,
  apply deterministic robust tie-breaking, and report every candidate plus the
  selected topology, coefficients, physical outputs, and selection evidence.

- Replay the serialized routed graph until every mains splice sums safely, reverting splice-breaking post-route correction stages role by role (mains-only FIR first, then post-route correction EQ) instead of failing the whole run; unfixable splices keep the hard error, and reverts are recorded as degraded stage outcomes.
- Drop home-cinema Post-EQ candidates that improve the predicted splice sum while regressing the mains published channel score, preserving the main correction the final safety gate would otherwise revert wholesale.
- Carry the correction-acceptance report and gate-added stage outcomes across routed-safety restores, marked as restored rather than blessed.

- Add explicit stage outcomes, serialized routed-response replay, stale-curve refresh after reversion, passband-safe correction, and final down-only role-pair level alignment.
- Delegate CTC/replay transfer evaluation to the engine's canonical serialized
  DSP realization while retaining workflow ownership of convolution sidecars,
  removing the divergent private plugin evaluator.

- Embed optimized group-delay polarity alongside existing phase-linear FIR delay and reconcile both realized controls into result metadata.
- Prepare file-backed targets before engine execution and preserve calibrated SPL across IIR, FIR, and hybrid paths.
- Stage main-channel correction after redirected-bass routing, preserve residual height delays, and reject duplicate phase mappings and supporting-source output collisions.
- Keep mixed-phase ownership, CTC response caches, broadband correction, and reported final curves consistent with the exported DSP chain.

## 0.5.34

- Compute full indirect IR-only STI from declared native measured impulse responses with `math-rir` 0.5.12. Preserve channel results in native bundles and physical-driver results in measured-acoustics sidecars; retain legacy JSON compatibility and warn on short captures.

## 0.4.53

- Keep canonical final curves aligned with topology gain and tolerate only
  sub-millidecibel FIR realization noise in the final correction safety gate.
- Label group-delay polarity, delay, all-pass, and phase-linear FIR stages and
  reconcile optimization metadata with the DSP controls that remain exported
  after safety reversion.
- Preserve bounded configured-objective improvements when symmetric target RMS
  changes slightly, without weakening runtime realization or acoustic safety.

## 0.4.52

- Validate file-backed acoustic resources before declaring a production
  configuration ready and preserve sidecar names across repeated exports.
- Inherited the workspace policy forbidding unsafe Rust code.
- Documented crate ownership and verification expectations.

## 0.4.51

- Established `roomeq-workflow` as the production composition boundary over RoomEQ resources, engine, and exporters.
