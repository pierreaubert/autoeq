# RoomEQ recovery — 2026-09-12

Branch: `fix/roomeq-reorg-recovery`, based on `5c58887`.

User request: compare both uncommitted reorganization attempts, recover useful
changes on a new branch, implement the reorganization plan, and validate useful
measured RoomEQ output through `scripts/test_roomeq_measured.sh`.

The original worktrees and the user's uncommitted edits in main are preserved.
The first attempt is `.worktrees/reorg-20260910` at `32f5a38`; Muse is registered
at `/Volumes/home_tmp/tmp/worktrees/autoeq/muse-reorg-20260910` at `5c58887`.
Gitea issue lookup failed (tea panicked, direct API connection timed out), so
this document records the local recovery task pending server availability.

While validation was running, main acquired unrelated T7V fixture renames
(`2.0_t7v` to `2.0_t7v_2024` / `2.0_t7v_2026`). They are not incorporated or
modified here. This branch's default matrix uses its original tracked
`2.0_t7v` fixture; reconcile scenario names when integrating those separate
main-worktree changes.

## Review findings

- The first attempt is the provisional base: explicit measurement and routing
  contracts, shared scorecards, correction-band policy, frequency-dependent
  constraints, processing-mode evidence, and export/report ownership changes.
  Imported code is provisional until the new branch's verification passes.
- Its inferred -120 dB subwoofer acoustic tail is unsupported. An electrical
  low-pass cannot prove an acoustic capability bound. Remove that inference;
  retain explicit declarations and report missing support as insufficient evidence.
- Its no-candidate fallback removes stages without replaying the published graph.
  Reconcile fallback curves, artifacts, outcomes, and the frozen baseline.
- Muse's endpoint-derived acoustic bound is also an assumption, not independent
  source-capability evidence. Do not import automatic acceptance based on it.
- Muse's output-budget relaxation can accept a candidate outside the requested
  budget. It is not needed to satisfy the recovery plan and is not imported.
- The cumulative-refinement trigger counted identical EQ across parallel outputs
  as repeated processing. Restrict duplicate detection to each serial signal path.
- Fallback electrical headroom was unchecked, and the published electrical stage
  still described the discarded candidate. Assess the structural fallback, apply
  common safety attenuation when needed, replay that level change against the
  frozen baseline, and refresh the electrical stage after final selection. An
  attenuated fallback is rejected/insufficient evidence, never unchanged.
- The measured Hybrid path exposed an incomplete `[band_split, delay, band_merge]`
  after rollback removed only its FIR. Baseline replay now shares the existing
  correction-stage classifier, removing the owned Hybrid block atomically and
  retaining explicit excursion protection and external arrival alignment.
- A passing identity candidate was labelled accepted despite zero improvement.
  Final selection now requires positive primary-metric improvement for a
  nonidentity candidate; realized identity is explicitly unchanged. Runtime
  rejection reasons remain in the stage ledger after rollback (for example
  Fidelia FIR's `pre_ringing_limit_exceeded`).
- Retain Muse's measurement-path diagnostics. Review its additional topology
  regressions and correction-mode policy coverage independently.
- Both scripts only replace the local venv interpreter with `python3`; neither
  change verifies dependencies, selects bounded cases, or isolates sidecars by mode.

## Validation ledger

Recovery follow-up:

- The four failures in the first full nextest run (3,178/3,182 passing) are now
  fixed and individually green: supporting-source convolution ownership,
  excursion-protection retention, multidriver safety-gain ownership, and final
  electrical-stage evidence in the validation bundle.
- Shared routed `fir.placement=per_driver` is supported by distributing the
  common post-route kernel onto physical outputs with distinct byte-identical
  artifacts. This preserves each logical-input transfer, retains source-owned
  pre-route filters, and is explicitly labelled
  `shared_kernel_per_physical_output`; it is not independent matrix inversion.
  Regression coverage checks source/physical-branch magnitude and phase,
  latency metadata, filesystem and memory stores, and configuration validation.
- The formerly failing `2.2_sigberg2` FIR invocation now completes and passes
  the artifact audit. It reports `insufficient_evidence` because the sub captures
  end at 199.951172 Hz while the requested observation band reaches 16 kHz.
- The plan's short-subwoofer paragraph permits an inferred -120 dB tail, but
  this conflicts with its primary no-invented-source-evidence contract. Recovery
  follows that primary contract: only explicit acoustic bounds can support
  the omitted tail; an electrical low-pass alone is not acoustic evidence.
- Final full workspace verification passed:
  `cargo nextest run --release --offline --locked --workspace --all-targets --all-features --no-fail-fast`
  ran 3,184 tests: 3,184 passed, 27 skipped (207.865 seconds of tests).
  Both generated schema baselines, the four Python artifact-audit regressions,
  scoped Rust formatting, shell syntax, and `git diff --check HEAD` also pass.
  A subsequent placement safety review added an explicit guard against moving
  convolution out of a routed frequency-split hybrid block. Its focused tests
  pass. The final guarded workspace rerun completed in 179.005 seconds:
  3,185 tests passed, 27 skipped, zero failures, with the same full-target,
  full-feature nextest command above.
- The first full measured attempt stopped at the paired-sub hybrid fixture:
  it inherited `max_freq=200` but requested a 300 Hz FIR/IIR split. The validator
  remains strict. `2.2_sigberg3` and `2.1_sigberg2` now explicitly use a 100 Hz
  processing split; physical crossovers, processing modes, and observation bands
  are unchanged. The corrected paired hybrid case improves 8.893330 to 2.667844
  dB over 20–200 Hz and exports two verified FIRs. Both corrected scenarios pass
  bounded hybrid and mixed-phase runs.
- Full-matrix preflight also found that KEF's ignored CSV derivatives were absent
  in fresh worktrees and its fixture referenced manually shortened filenames.
  Regeneration from the tracked `.mdat` gives byte-identical measurement CSVs;
  references now use the converter's actual filenames. The harness prepares
  derivatives with `mdat2csv.py --no-clobber`, then dry-runs every requested
  config before any optimization. Existing CSVs and generated configs are not
  overwritten. Converter regressions: 12 passed, 1 skipped; artifact-audit
  regressions: 4 passed.

The latest default measured run completed 44/48 cases before stopping at
Genelec 5.1.4 IIR. Its correction-free structural graph requires 14.536 dB of
additional attenuation under independently phased unit-peak inputs, exceeding
`finalization.max_attenuation_db=12`. This is a real configured-policy conflict,
not an acoustic acceptance failure that can safely be ignored. User approval
has been requested for an explicit 18 dB allowance on this fixture only; no
acoustic output-loss budget, input-peak assumption, or clipping ceiling would
change. No allowance has been changed pending that choice. Existing 5.1.4
artifacts from earlier provisional runs are not validated results of this run.

The [machine-readable measured snapshot](ROOMEQ_MEASURED_RECOVERY_RESULTS.json)
contains only the 44 cases completed by that run: 13 accepted, 9 unchanged,
10 rejected, and 12 insufficient-evidence outcomes. Every included case passed
the native-manifest, electrical-stage, outcome-consistency, and FIR-byte audit.
Supplementary `2.1_sigberg2` IIR and FIR runs also completed successfully on the
guarded binary, completing this scenario's four-mode re-audit together with the
earlier hybrid/mixed-phase runs. These checks do not override the 5.1.4 blocker.

- Provisional imported code: `cargo check --offline --locked -p roomeq-workflow
  -p roomeq-cli` passed (17.18 s), before recovery-specific fixes.
- The initial focused gates below are historical; final workspace verification
  is recorded above, and final measured results are recorded separately.

Initial focused gates: optimizer 303 passed; engine 694 passed / 2 ignored;
model 253 passed; quality 139 passed / 1 ignored; export 115 passed / 2 ignored;
workflow 698 passed / 6 ignored. After the electrical fallback fix, finalization
tests: 12 passed / 3 ignored. Artifact audit regressions: 4 passed.
The broad debug engine run was stopped and superseded by the release suite.

Pre-electrical-fallback measured audit (superseded for fallbacks): 2.0_d3v IIR
accepted, 5.210673 -> 4.564537 dB over the configured 20–1200 Hz band; paired
2.2_sigberg3 IIR accepted, 8.893330 -> 3.578203 dB over 20–200 Hz; 5.0_genelec
IIR accepted, 3.910919 -> 3.524082 dB over 20–400 Hz. Each had per-input final
seat replay evidence. 2.1_sigberg2, shared-MSO 2.2_genelec, and 5.1.4_genelec
fell back; these must be re-audited after electrical normalization was repaired.
2.2_sigberg2 requires missing subwoofer acoustic support above 199.951172 Hz
to assess its requested 20–16000 Hz band; no upper-tail capability was invented.

## Measurement interpretation

The audit retains each capture's native grid and phase in degrees and realizes
the configured graph at 48 kHz. CSV magnitude labels do not establish absolute
SPL calibration or an acoustic source-capability ceiling; for example d3v's CSV
contains negative dB levels. Reported target errors are relative, measured
frequency-response errors, not calibrated loudness or listening-test claims.
The configured observation bands differ by case (1200 Hz for d3v, 200 Hz for
paired Sigberg, 400 Hz for 5.0 Genelec, 16000 Hz for shared-sub cases). The audit
records requested and evaluated bands separately and leaves full-graph metrics
unavailable when branch support is missing. Single captures establish no
held-out-seat claim. FIR byte hashes and serialized electrical paths are checked
independently of the acoustic objective.

The initial TokenSave context retrieval saved approximately 1,118 tokens. Graph
queries established entry points; imported and modified code was inspected in
the new worktree directly rather than assuming the main-branch index contained it.
