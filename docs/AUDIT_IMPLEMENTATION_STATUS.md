# Audit implementation status — 2026-10-02

Tracking: private Gitea [issue 7](http://192.168.1.32:3001/pierre/autoeq/issues/7).
Baseline: `e3baa14a7b71c6b6d1266771c1196920c0f2aaf2`.
The full local audit is `reviews/audit-20261002.md` (ignored by Git).

## Order and completion rule

Finish AutoEQ and RoomEQ backend work before UI implementation. The first
batch does not complete the audit. A work package closes only when its
production path and acceptance evidence are present. Hardware, listening,
and research outcomes require their stated evidence; synthetic fixtures
cannot replace them.

| Package | Current state | Remaining acceptance work |
| --- | --- | --- |
| A00 QA/build coverage | In implementation | Complete Cargo matrix, schema parity, registered numerical runners, CI evidence |
| A01 native UI actions | Deferred | Begin after backend work |
| A02 canonical UI result loading | Deferred | Begin after backend work; backend bundle loading belongs to A11 |
| A03 capture/backend handoff | Open | Versioned acquisition contract, provenance, strict measured-session loading and cancellation |
| A04 measurement/live analysis | Open | Calibrated units, routing, quality and bounded streaming with device validation |
| A05 configuration/review UI | Deferred | Begin after backend work |
| A06 speaker/headphone workflows | Open | Backend source/rig/target/device contracts and reference comparisons; UI deferred |
| A07 optimizer quality | Partial | Common feasibility hardening implemented; fixed-budget backend/seed benchmarks and derived presets remain |
| A08 recoverable jobs | In implementation | Strict identity, atomic persistence, production seed reuse, interruption evidence; exact continuation needs separate equivalence evidence |
| A09 realized correction | Partial | Production FIR and shared-grid validation implemented; full mode/rate/complex/time-domain witnesses remain |
| A10 calibrated joint bass | Open | Preserve existing complete-graph demand checks; integrate calibrated demand search and demonstrate output/seat/routing tradeoffs |
| A11 bundles/export | In implementation | Transaction recovery, resource identity/capability query, consumer transfer/relocation witnesses |
| A12 applied playback/verification | Open | Native host boundary, exact graph activation, device stress and associated measured capture |
| A13 rendering/accessibility | Deferred | Begin after backend work |
| A14 perceptual/listening evidence | Open | Independent references/domain registry and relevant blinded listening evidence |
| A15 spatial/adaptive studies | Open | Separate held-out measured studies with explicit implement/constrain/decline outcomes |
| A16 persistent regression corpus | In implementation | Numerical runner foundation first; capability mapping, held-out cases, historical canaries and release evidence remain |

## Verified backend changes

Commit `106fbe8` rejects nonfinite optimizer candidates and invalid bounds,
including the shared gate for PEQ, multi-driver and multi-sub layouts.
It validates joint-bass measurement arrays/output identities and requires
shared EQ to use identical valid seat frequency grids.

Production FIR entry points now validate rates, tap budgets, correction
bands, measurement/target arrays and policy values before numerical design.
Excess-phase inversion requires phase data. Realized tap arrays must have
the requested length and finite coefficients. The mixed-mode sidecar fixture
supplies explicit analytic phase for its requested phase inversion.

Verification on the isolated implementation worktree:

- `cargo test -p autoeq-optim --lib evidence_validation_tests`: 2 passed.
- `cargo test -p autoeq-optim --lib all_candidate_layouts_reject_nonfinite_parameters_without_envelopes`: 1 passed.
- `cargo test -p roomeq-engine --lib fir:: -- --quiet`: 69 passed.
- `cargo test -p roomeq-engine --lib mixed_crossover:: -- --quiet`: 6 passed.
- `cargo test -p roomeq-engine --lib multisub::joint_objective -- --quiet`: 24 passed.
- An earlier full engine run had 865 passed, 2 failed, 1 ignored. Both failures
  were corrected and their affected suites rerun above. A fresh complete
  integrated run is still required before final audit completion.

Builds used offline Cargo resolution and
`CARGO_TARGET_DIR=/Volumes/home_tmp/tmp/autoeq-audit-target`. Cargo resolved
local `math-iir-fir` and `math-dsp` from the existing sibling math-audio worktree;
these test results include those configured dependency sources.
