# RoomEQ Audibility Contract — Stage 0 Record

Date: 2026-09-05. Plan: `reviews/next-20260905.md`, Stage 0 ("Resolve the
scientific and engineering contract").

This document records the decisions, references, acceptance criteria, and
known implementation issues that later stages build on. It changes no DSP
and no defaults. Where evidence is missing the entry says `unknown` —
missing data is never evidence of safety or inaudibility.

Machine-readable vocabulary for this contract lives in
`crates/roomeq-model/src/config/report_outcome.rs`
(`ReportOutcome`, `AssessmentConfidence`, `EnforcementState`,
`AssessmentProvenance`, `AppliedThreshold`, `AssessmentRecord`,
`BudgetAggregation`, `PruningBudget`).

## 1. Fixed references

- **Pruning reference.** Before any simplification is evaluated, accept and
  freeze a full chain `F0` (filters, gains, delays, crossovers, FIRs,
  routing). Every simplified chain `G` compares against `y(F0, c)` over
  the declared conditions — never against the previous pruning step.
  `F0` identity: content hash of the realized chain plus config version;
  record it in `AssessmentProvenance.reference`.
- **Quality reference.** Declared per workflow before correction is
  judged: room response, direct sound, spatial cues, target tilt, and
  latency to preserve. There is no single global quality reference; each
  workflow documents its own (see §3, first decision).
- **Identity solution.** The zero-filter chain is always an admissible
  candidate. Removing an audibly harmful filter is a quality-changing
  decision with its own acceptance checks, not pruning.
- **No uncorrected-vs-corrected minimization.** Identity is optimal for
  that term, so it cannot be the quality objective.

## 2. Decisions on the plan's Open Decisions list

| # | Question | Stage 0 resolution |
|---|----------|-------------------|
| 1 | Desired reference above/below the room transition when direct/anechoic data is absent | **Deferred to Stage 3** (`unknown`). Until then: below-transition corrections are cuts-biased and seat-RMS-referenced; above-transition corrections keep the existing target pipeline unchanged. No new target is introduced in Stage 0. |
| 2 | Auditory model/version, attributes, reference implementation; detection vs equivalence vs preference | **No model selected.** The only available distance is the experimental heuristic proxy (approx masked-loudness delta, sones-labeled, unvalidated). It may nominate candidates and screen regressions; it settles nothing. Model selection is Stage 2 work with published reference cases. |
| 3 | Physical calibration contract, assumed-level fallback, programme corpus, level range, listener population, test design | **Nominal-level fallback with explicit assumption.** `listening_level_phon: 75` is a nominal setting, not SPL calibration. Reports carry `calibration: "nominal-75phon"` or `"unknown-spl"`. No masking claim and no removal may rest solely on the nominal level. Programme corpus, level range, and listener population are **deferred to Stage 2** (`unknown`); until then no listener-benefit claim is made. |
| 4 | Cumulative pruning budget, aggregation, confidence margins, worst-seat tradeoffs, engineering limits | **Resolved as an opt-in declared budget.** `PruningBudget` (`max_cumulative_delta`, `aggregation: sum\|max`, `conditions`) is validated for shape and reported against; enforcement arrives with Stage 1 cumulative checks. Default: disabled. Worst-seat metrics (p95/max) must define frequency support, weighting, bin-vs-seat aggregation order, and uncertainty before use — **deferred to Stage 3**. |
| 5 | Measurement-quality requirements; magnitude-only, noisy, sparse, phase-uncertain inputs | **Magnitude-only supports spectral screening only.** Temporal and phase claims require trustworthy complex responses or time-referenced RIRs. Missing phase/time data degrades assessments to `InsufficientEvidence`, never to a passing verdict. Concrete per-input gates are Stage 3 work. |
| 6 | Compute budget, reranking vs optimization, versioned policy/config design, evidence for future preset recommendations | **Reranking is shortlist-only; no auditory model inside the optimization loop** until Stage 4 demonstrates benefit. New policies are explicitly selected and versioned; disabling a policy restores legacy behavior. A warning/advisory release never authorizes a silent default flip. |

## 3. Report outcomes (normative vocabulary)

| Outcome | Meaning |
|---|---|
| `keep` | Full-chain comparison supports keeping current processing. |
| `candidate_removal` | Nominated only; needs the cumulative F0 check before anything is applied. |
| `accepted_removal` | Passed acceptance and applied; compare against frozen F0. |
| `risk_limited_correction` | Proceeds under explicit recorded limits (caps, bands, seats) on partial evidence. |
| `insufficient_evidence` | Cannot decide; the filter stays. Default for unassessed items. |

Prediction, enforcement, and confidence are separate fields
(`AssessmentRecord.outcome` / `.enforcement` / `.confidence`). Reports
additionally carry model/version, calibration or assumptions, comparison
reference, applied thresholds with units, and a reason
(`AssessmentProvenance`, `AppliedThreshold`). Sones-labeled proxy values
keep their units with an experimental marker; they are not portable
thresholds and not reference-grade ISO loudness.

## 4. Supported inputs and fallbacks

| Input | Supported assessment | Fallback |
|---|---|---|
| Magnitude response, calibrated SPL, known grid | Spectral screening, heuristic nomination | — |
| Magnitude-only, SPL unknown | Screening with `calibration: unknown-spl`; no masking claims | Sensitivity note required |
| Complex response / time-referenced RIR | Temporal and phase diagnostics (Stage 3) | Without them: `insufficient_evidence` for temporal/phase questions |
| Multi-seat measurements | Seat-wise and worst-seat reporting (Stage 3) | Single seat: no spatial claim |
| Sparse/noisy measurement | Screening with widened uncertainty | Below trust threshold: `insufficient_evidence` |

Do not optimize finer features than measurement resolution supports;
align grids explicitly; never normalize away candidate level,
interchannel, or timing changes under evaluation.

## 4a. Acoustic and psychoacoustic operating rules

These rules constrain both candidate generation and acceptance. They are
engineering policies, not claims that a single microphone can establish a
listener's complete perceptual experience.

- **Use the room transition explicitly.** Below the estimated Schroeder
  frequency, the response is treated as modal and position-sensitive. A
  narrow, high-Q cut is allowed when a peak is supported as a real resonance
  (and not merely a cancellation), while boost into a suspected cancellation
  is independently limited. There is no global Q limit.
- **Broaden above the transition.** Above Schroeder, ordinary correction is
  limited to broad level and slope alignment, normally with Q < 1 shelves or
  bells. A higher-Q operation is an exception for a measured, repeatable
  resonance and is normally a cut. Narrow comb or seat-specific dips are not
  inverted routinely.
- **Do not promise treble inversion.** Head position, microphone placement,
  loudspeaker directivity, and the listener's two-ear HRTF make detailed
  high-frequency room correction unstable. The default upper-band objective is
  the broad target/slope; phase, group delay, and sharp treble features are
  reported as evidence and guardrails rather than forced to a flat trace.
- **Keep magnitude primary, timing protective.** Tonal and low-frequency
  modal improvement is the primary acoustic objective. Phase, group delay,
  latency, pre-ringing, and source arrival are constraints on the realized
  solution. A phase change is not rejected merely because it is nonzero, and
  a flat magnitude result is not accepted when it violates a declared timing,
  routing, or electrical limit.
- **Treat precedence as an evidence boundary.** The precedence effect is a
  property of direct sound, reflections, programme content, and binaural
  hearing. A conventional magnitude/phase microphone trace cannot establish
  a universal Haas delay or HRTF outcome. Such claims require time-referenced
  multi-channel evidence and a declared listening protocol; otherwise the
  result is `insufficient_evidence`.
- **Allow achievable extension compromises.** A subwoofer or main speaker may
  not support a ruler-flat 20--20 kHz target. Keep a fixed observation band,
  expose the requested and effective correction bands, and allow a natural
  roll-off (for example declining to force 20--40 Hz) when that is the safer
  supported choice. Do not hide a damaged band by shrinking the evaluation
  support, and do not add a high-pass unless protection or configuration
  explicitly requires it.
- **Separate audibility from engineering limits.** ERB/Bark weighting and
  temporal masking are analysis dimensions, not proof that a sharp feature is
  inaudible. Deep narrow dips are not automatically harmless, and a sones-
  labeled proxy is not a calibrated threshold. Every accepted decision states
  its calibration, support, uncertainty, and whether the result is a physical
  safeguard, an experimental perceptual screen, or a listening finding.

## 5. Acceptance criteria per stage

- **Stage 0 (this document).** Contract recorded; outcome vocabulary and
  budget types exist with wire-format pins and schema coverage;
  advisory-by-default locked by tests
  (`test_optimizer_config_stage0_advisory_defaults`,
  `rule_pruning_budget_rejects_nonsense_shapes_only`,
  `report_outcome_tests`). No enforcement widened: `cargo test -p
  roomeq-model` green, stereo fixture output unchanged vs baseline.
- **Stage 1.** Heuristic nominations separated from validated acceptance;
  one-removal-at-a-time with interaction recompute; cumulative F0
  comparison; identity solution admissible; entry-point audit complete
  (single/multi measurement, adaptive/single-pass, Pareto-selected,
  multi-seat, multi-sub/bass-managed, hybrid IIR/FIR); routed/exported
  chain verified with stable ids, order, reason codes, loss, reference,
  and rollback. Implemented: adjudication in `eq/audibility_veto.rs`
  (tests pin identity, resonance retention, cancelling/overlapping
  pairs, accumulation guards, budget binding, rollback); propagation
  test `veto_enforcement_propagates_to_exported_chain`; audit in §7
  below. Multi/sub paths: per-channel optimization adjudicates; joint
  multi-measurement and spatial-robustness strategies are recorded gaps
  with preserved legacy behavior.
- **Stage 2.** Named auditory comparison against published cases or an
  independent reference implementation; calibrated level sweeps;
  rendered audio stimuli (speech, music, tones, transients); held-out
  seats/programmes/levels/rooms; predefined blinded protocol with power
  analysis. Synthetic fixtures cover implementation behavior only.
  Implemented (staging, not listening): seeded renderer
  (`roomeq-quality/stimuli.rs` — tones, bursts, sweeps, clicks, shaped
  and band-limited noise, harmonic complexes for timbre variants,
  masker-probe temporal-masking stimuli; speech/music as hash-pinned
  external files only) with the named `affine-fs-spl-v1` conversion,
  clip-refusing level sweeps, and hashed manifests; blinded protocol
  with preregistration hash, exact ABX arithmetic, power sizing,
  detectability/equivalence/preference intents (equivalence needs a
  prespecified bound; preference cannot claim inaudibility), and results
  sidecars (`protocol.rs`); deterministic staged holdout splits plus the
  required case-kind inclusion list — repeatability, no-correction,
  sub/main phase, channel balance/timing, and the separate
  detectability/preference arms (`validation_corpus.rs`);
  order-explicit seat aggregation with worst-seat support rules and
  seeded bootstrap intervals (`metrics.rs`); `roomeq-qa-synthetic
  --stimuli` renders the demo set.
- **Stage 3.** Conservative physical/spatial acceptance with
  false-classification and missing-data tests; opt-in band-split
  policies; defined seat-wise metrics; held-out-seat and sub/main-sum
  checks. Implemented (policy staging, not listening): confidence-aware
  inversion support — minimum-phase classification alone never authorizes
  a boost; confidence, measurement-depth, and cross-seat/bin gates must
  also clear, misclassified/uncertain nulls are cuts-only, absent
  evidence refuses (`inversion_support.rs`); opt-in band-split policy —
  disabled by default, enabled requires a positive-width
  confidence-dependent transition (hard boundaries refused), cuts-only
  bass posture, distinguished direct vs in-room tilts, and
  early/direct/late plus group-delay diagnostics that are advisory by
  construction (`band_policy.rs`); final-chain stability/headroom/
  latency/export constraints with missing-evidence-fails-closed, and
  temporal gates that enforce only with trusted timing plus a stated
  engineering or validated-perceptual basis, with a fully specified
  pre-ringing measurement definition (`chain_constraints.rs`); declared
  seat-wise metric definitions (support band, weighting, bins-vs-seats
  percentile domain, aggregation order, uncertainty, permitted
  degradation) with final validation against the declared baseline, pruned
  candidates against the accepted full chain, in-band sub/main summation
  checks, and aggregate gain plus worst supported seat
  (`final_check.rs`); `roomeq-qa-synthetic --stage3-policies` demos all
  four on synthetic fixtures.
- **Stage 4.** Shortlist reranking against EPA/flat baselines on held-out
  data and listening outcomes; simpler objective kept unless benefit is
  demonstrated; budgets and reproducibility declared. Implemented
  (pipeline staging, not listening): bounded shortlists from fast
  objectives with Pareto and identity inclusion
  (`autoeq-optim/rerank.rs` — `build_shortlist`); rerank under a pinned
  auditory evaluator and pinned loss (mid-run switches abort), with
  staged-metric basis today and listening basis gated on recorded
  outcomes; evaluation/wall-time/memory budgets with cancellation, and a
  content-hashed transform cache so ablation reuses unaffected
  components; explicit refinement records carrying the pin; EPA/flat
  baseline comparison on one held-out set with a keep-simpler default
  below the margin; coarse-screening-nomination vs validated-final
  resolution discipline; new loss options require stated semantics,
  finite domain, and reference test, with temporal masking mapped to a
  supported model stage. Demo via `roomeq-qa-synthetic --stage4-rerank`.
- **Stage 5.** Only Stages 0–3-supported rules promoted; legacy presets
  stable; provenance in reports/sidecars; export round trips verified;
  docs and schemas updated per stage. Implemented (rollout mechanics,
  not listening): four independent release gates — correctness,
  physical safety, perceptual validation, listening benefit — with
  promotion rules in `roomeq-qa/release_gates.rs` (advisory and
  evidence-free passes never promote; safeguards promote without Stage 4
  evidence; legacy rollback is never gated); versioned policy selection
  with a legacy-restore guarantee, grounded in the real veto config
  (`None`/disabled → legacy, default selection → advisory, explicit
  opt-in → enforcing); export round-trip verifiers
  (`roomeq-export/roundtrip.rs`) proving biquad JSON coefficients,
  routing, preamp, and delay against the graph and convolution WAV bytes
  sample-exact with tamper-evident mismatch errors; staged-rollout docs
  in `ROOMEQ_MANUAL.md` and `INPUT_FORMAT.md` (schema already carries
  the veto selection).

## 6. Known implementation issues (historical pre-coding ledger)

Current closure map (2026-09-12): items 1–5 and 8 below are covered by the
one-removal-at-a-time adjudication, multi-measurement/spatial post-pass,
experimental-unit provenance, mode-evidence tests, serialized output schema,
and frozen-chain local-deviation checks in this worktree. Item 6 remains an
evidence requirement rather than a universal ban. Item 7 is a documentation
erratum/no-entry-point clarification. CEA-2034 source-model filters carry an
explicit exemption marker and are still included in complete realized-graph
scorecards. See `reorg-20260910.md` for the exact tests and measured audit;
the detailed historical issue text below is retained unchanged.

1. **Batch enforcement.** `enforce_veto_verdicts` removes all `Remove`
   verdicts at once. Stage 1 must move to one-removal-at-a-time with
   recompute; until then enforcement stays off by default.
2. **Sones labeling.** Veto verdicts and thresholds use sones-labeled
   proxy units without an experimental marker in every surface. Required:
   experimental labeling plus provenance retention for legacy fields
   (no silent reinterpretation).
3. **Entry-point coverage.** Veto post-pass is wired for single-channel
   adaptive/single-pass paths. Multi-seat, multi-sub/bass-managed,
   hybrid IIR/FIR, and Pareto-selected paths are unaudited (Stage 1).
4. **Reserved hooks.** `VetoReason::ModeProximityBan` and
   `mode_proximity_ban_erbs` are recorded but unenforced (Stage 3
   consumers).
5. **Verdict schema gap.** Emitted veto verdicts are not yet covered by
   `output_schema.json`. Close when verdicts reach reports/sidecars
   (Stage 1 wiring, Stage 5 conformance).
6. **T60 linewidth bar.** A T60-derived ban width needs an adequately
   resolved, approximately isolated decaying mode; it informs confidence,
   never a universal radius.
7. **Review-doc errata.** `reviews/next-20260905.md` validation section
   named `just qa-roomeq`, which has no such recipe (fixed in-place to
   `just qa-roomeq-all`); its Stage 1 audit list mentions
   "continuous-refinement updates," for which no matching entry point
   was found — needs author clarification.
8. **Loudness-only elimination.** `backward_eliminate_veto_units` (and
   the legacy raw-loss `backward_eliminate`) judge removals by total
   impact only; a narrow deep feature can be locally audible with
   negligible total impact. Mitigation: elimination always retains the
   finalist; only post-pass adjudication (with the local-deviation
   guard) may empty the set. A local guard inside elimination itself is
   future work.
9. **Reason codes stop at the engine boundary.** Veto verdicts and
   acceptance records live on `EqOptimizationResult` with no downstream
   consumers yet: channel results, reports, sidecars, and exports carry
   filters but not reason codes. Full report-chain reason-code
   stability is Stage 5 work.
10. **No reoptimization after removal exists.** Verified by search: no
    reoptimization path follows the post-pass on any workflow. The
    F0-threading invariant is recorded on `apply_veto_postpass` for any
    future path.

## 7. Entry-point audit (Stage 1)

Covered = heuristic nomination plus cumulative adjudication run on the
emitted set. Uncovered paths keep their exact legacy behavior; the
`audibility_veto: Vec::new()` markers at those sites are explicit, not
oversights.

| Entry point | Path | Veto | Evidence |
|---|---|---|---|
| Single-channel adaptive | `optimize_channel_eq_adaptive` → post-pass | Covered | `crates/roomeq-engine/src/eq/optimize.rs` post-pass call |
| Single-pass legacy + callback/live variants | `optimize_channel_eq_inner` → post-pass | Covered | Same file, second post-pass call; callbacks force single-pass |
| Spinorama variants | Same inner dispatch | Covered | Shared post-pass tail |
| Per-channel/sub across stereo, stereo-sub, home cinema | `run_post_eq` → single dispatch | Covered | `topology/run.rs`, `stereo_sub.rs`, `home_cinema.rs` via `run_post_eq`; `channel_optimizer.rs` single branch; workflow `eq.rs` single bridge |
| Pareto candidates via single-channel runs | Same as above | Covered | NSGA backends resolve through the same passes |
| Hybrid IIR/FIR | `channel_fir` consumes adjudicated biquads | Transitive | Emits taps, not biquads; nothing new to nominate |
| Joint multi-measurement | `optimize_channel_eq_multi_inner` | **Gap** | Explicit empty verdicts + no summary; via multi dispatch and workflow multi bridge |
| Spatial robustness (incl. seat-aggregated strategies) | `optimize_spatial_robustness` | **Gap** | Explicit empty verdicts + no summary; strategy selected in `run.rs`, `home_cinema.rs`, `home_cinema/multi.rs` |
| CEA2034 speaker prefilters | Own path in `cea2034.rs` | **Gap** | Bounded by its own `max_db` (default 3.0); out of veto scope |
| Continuous refinement | No such entry point | N/A | Contract issue 7; transition-artifact checks outstanding for live updates |

Routed-chain verification performed: twin stereo runs (report-only vs
enforced, identical seed) prove enforced-kept sets are ordered
subsequences of the emitted sets and each routed channel's exported EQ
plugin contains exactly its kept set
(`veto_enforcement_propagates_to_exported_chain`). Render fidelity of
the export formats remains covered by the existing realized-transfer
conformance tests; full routed verification with gains, delays, and
crossovers is Stage 5 conformance work.
