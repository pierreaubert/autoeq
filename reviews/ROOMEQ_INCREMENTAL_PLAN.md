# RoomEQ incremental implementation plan

Date: 2026-09-10. Status (2026-09-12): Stages 0–9 simplification increments
are implemented; the full measured topology/mode matrix remains deferred by
design. The detailed current ledger is in `reorg-20260910.md`.

## Goal and delivery rules

Improve the original measured system with useful, physically supported correction. Implement in the existing Rust codebase through small independently verified changes. Restore the failing tests first. Discuss and implement a separately scoped simplification phase once the required behavior works; simplification is a completion requirement, not an optional follow-up.

- One defect or coherent behavior change per increment; document its reason and verification.
- Do not combine test recovery with a redesign, new default policy, or broad refactor.
- Repair tests when their fixtures or assertions contradict the intended contract, with the contradiction documented. Otherwise fix production code. Do not turn expected success into expected failure merely to obtain green results.
- Do not lower thresholds, ignore failures, increase optimizer budgets/seeds until they pass, or alter measured recordings to conceal defects.
- Preserve pre-existing user changes. Existing working-tree edits are part of the initial test state, not changes made by this plan.
- Tests demonstrate their stated properties, not universal acoustic or perceptual benefit. No dataset-specific acceptance exceptions.
- Retain existing modes. No wholesale Python rewrite, new framework, or speculative HRTF model is required.
- Update this ledger after each increment. A later step starts only after its prerequisites pass; the final simplification phase also requires a discussion of scope.

Tracking: Gitea is unreachable. Public GitHub issue publication was blocked by automatic approval review; this document is the local tracking record pending explicit publication approval.

## Agreed behavior

### Topology and processing

| Configuration | Logical programme inputs | Physical routing |
|---|---|---|
| Stereo 2.0 | L, R | Two mains |
| Stereo 2.1 | L, R | Two mains plus shared sub; no third input |
| Stereo 2.2 paired | L, R | Each input drives its own main/sub pair |
| Stereo 2.2 shared | L, R | Two mains plus a shared two-sub system |
| Cinema | 5/7/9 bed and 2/4/6 height channels, plus LFE when carried by the programme | Corresponding speakers and 1-N physical subs |

LFE programme content and redirected bass are distinct. Subwoofer count does not determine input count. MSO/DBA and other existing strategies configure cooperation between physical sources, independently of IIR/FIR/mixed-phase correction. DBA requires applicable geometry/evidence; it is not an arbitrary array label.

Every required topology must support IIR, FIR, and mixed-phase/excess-phase correction when its required evidence and realizable support are present. Retain the existing Hybrid and other modes. Report effective behavior: the existing `phase_linear` configuration can request Kirkeby/excess-phase processing and must not be treated as proof of a linear-phase result.

### Correction and assessment

- Below Schroeder: allow detailed correction and high-Q bass filters when justified, particularly resonance cuts. Limit cancellation-filling boost independently.
- Around Schroeder: transition smoothly toward broader correction; use the existing configured transition rather than inventing a universal frequency.
- Above Schroeder: broad slope/level alignment, normally Q<1 bells or suitable shelves/tilt; exceptionally a targeted cut for a reliably supported resonance. No detailed treble room-response inversion or routine dip filling.
- There is no global Q range. Filter width follows frequency region and correction purpose.
- Permit explicit bass-extension compromises and natural roll-off, for example declining to flatten 20-40 Hz. This does not automatically authorize a high-pass or removing existing bass.
- Preserve a fixed observation band while reporting the selected correction band and any lost extension separately. Do not improve a score by hiding damaged bins.
- Evaluate original and final complete realized playback with the same metric definitions, measurement support, target, level reference, and spatial priorities.
- Prioritize useful tonal/bass improvement; protect temporal behavior, source integration, output capacity, and spatial relationships. Do not demand strict improvement in every metric or conflate safe unchanged output with successful correction.
- ERB analysis is an auditory-resolution model, not a guarantee that narrow features are inaudible. Retain acoustic extrema and energy-based band evidence; do not invent a calibrated audibility model.
- Ordinary microphones do not establish HRTF/binaural or precedence-effect performance. Retain existing spatial modes with their specific evidence requirements; no universal Haas delay or group-delay audibility threshold.
- A bounded unsuccessful search means no satisfactory candidate was found within that problem and budget, not proof of physical infeasibility.

## Ordered increments

### 1. Recover the current tests (mandatory first gate)

1.1 Reproduce the complete failure inventory with the existing workspace command, retaining full diagnostics, revision, features, skips, and pre-existing modifications.

1.2 Fix isolated diagnostic and event-contract defects first (missing measurement path/cause and cancellation reporting). Run the exact failing regression after each change, then its owning test target.

1.3 Fix artifact ownership and finalization defects (supporting-source retained taps versus persisted WAV, missing acceptance evidence, electrical attenuation and useful-output comparison). Preserve exact artifact binding and electrical limits.

1.4 Resolve integration fixtures and assertions for topology, phase evidence, structural gain/protection filters, target handling, and reversion reporting. Synthetic fixtures may explicitly declare their known phase; real missing phase must not be invented. Preserve the intent of each success test.

1.5 Diagnose and repair the Hybrid reference-phase and continuous-area decision failures against their actual intended objective. Preserve the distinction between minimum phase, excess phase, and full-reference cancellation. Any required upstream math fix receives its own bounded change and integration check.

1.6 Resolve the cinema scaling benchmark failures reached by `--all-targets`; do not remove the benchmarks from the verification command to obtain success.

**Exit:** the original broad command passes with no new skips; existing intentional skips are listed. No feature work begins while the recovery gate remains red. Group related failures by cause, not one patch per failing test.

### 2. Pin input/output and bass-routing contracts

Small increments: stereo 2.1; paired 2.2; shared 2.2; cinema LFE/redirected bass. Convert legacy role labels at the boundary. Use the existing physical routing model.

**Exit:** independently calculable per-input transfer and simultaneous-input electrical tests verify counts, gains, polarity, delays, crossovers, and each physical branch exactly once. No manufactured third stereo input. Preserve native/export behavior.

### 3. Make before/after assessment consistent

Reuse existing quality types and metric kernels. First make target/level/support definitions explicit, then ensure final graph evaluation uses them, then align runtime/QA/report consumers. Retain separately named auditory and acoustic metrics rather than mixing ERB-rate and log-frequency RMS under the same name.

Assess bass target error, broad upper-band level/slope, excess peaks and uncertain boost, seat outcomes, realized temporal behavior, and electrical limits. Original and final use the same definitions. Keep current engineering limits unless an explicit policy change is separately justified. Missing mandatory evidence cannot receive an accepted verdict.

Selection preserves configured seat priorities. Data used for candidate selection is validation data; do not present it as untouched generalization evidence.

**Exit:** identity/known-correction/known-damage properties, grid-density and ordering invariance, calibrated gain changes, full-graph/sidecar identity, and consistent runtime/QA/report decisions. No large evaluator consolidation yet.

### 4. Implement frequency-dependent correction policy

4.1 Correct bounds and permission handling below/around Schroeder; remove any global Q restriction that prevents justified sharp bass cuts.

4.2 Restrict ordinary upper-band correction to broad Q<1 slope/level behavior; keep a separate evidence-supported resonance-cut exception.

4.3 Evaluate the combined low-frequency correction and broad alignment; prevent repeated ownership or post-alignment invalidation of an earlier result.

**Exit:** analytic narrow bass resonances can receive appropriately sharp cuts; low-frequency cancellation boost remains restrained; upper-band comb features do not trigger local inversion; broad tilt/level can improve without new narrow artifacts. Small perturbations do not create unexplained policy jumps.

### 5. Support achievable extension and target compromises

Add an explicit optional acceptable correction-band range/roll-off policy to the existing optimizer configuration. Existing explicit frequency limits remain fixed unless the adaptive policy is requested. Do not impose 40 Hz as a universal floor.

Try a bounded set of supported correction-band/strength alternatives using the same evaluator. Retain natural response outside correction support, unless separately configured protection requires filtering. Report extension and output changes alongside target improvement. No claim of maximum clean speaker output from a single uncalibrated response.

**Exit:** limited-extension examples can produce useful accepted correction; unnecessary bass loss, hidden evaluation-band shrinking, and attenuation disguised as improvement are rejected. Requested and effective bands are visible and reproducible.

### 6. Carry the policy through IIR, FIR, and mixed phase

6.1 IIR: verify the policy on the complete corrected topology, not per-filter gain alone.

6.2 FIR: apply equivalent magnitude/band intentions; evaluate actual taps and the declared phase target, support, latency, normalization, and fractional delay.

6.3 Mixed phase: retain IIR magnitude correction and correct only trustworthy excess phase under explicit support/pre-ringing constraints. Preserve common arrival and relative-source alignment. Re-evaluate the complete IIR+FIR graph.

Each mode lands separately. Keep filter representation, phase objective, and subwoofer strategy independent. Compare modes under declared resource budgets; do not silently increase taps or switch phase semantics to make a fixture pass.

**Exit:** equivalent intentions and honest before/after evidence across all three methods; missing phase, insufficient causal support, and native realization mismatch are explicitly handled. No unannounced method substitution.

### 7. Exercise the required topology/mode combinations

Use the measured corpus under `data_tests/roomeq/measured`, including paired-sub variants, as application inputs for the fixed policy. Record actual measurement/seat coverage; do not claim measured 7/9-bed or every height/sub layout merely because a generated scenario exists.

Run stereo 2.0, 2.1, both 2.2 arrangements, then cinema/multi-sub strategies incrementally across IIR/FIR/mixed phase. Retain generated topology/property coverage and independent native/export replay. Include changes of measurement order/density, plausible timing/level uncertainty, and conflicting/new seats. No optimization against a fixed corpus acceptance percentage.

**Exit:** every selected case actually executes, has the correct input/output contract and effective method, and yields a justified improvement/unchanged/rejected/unassessed outcome. A correct rejection is not reported as acoustic success. Independent evaluation data remains untouched by selection.

### 8. Discuss and implement simplification (required final phase)

Only after the functional gates pass, inventory production code separately from tests/generated files and present a concrete deletion/consolidation proposal. Discuss the scope with the user before broad restructuring.

Targets: duplicate acceptance definitions, repeated response reconstruction, stale report caches, overlapping rollback/control ownership, and redundant topology plumbing. Preserve independent verification oracles and distinct regressions. File moves alone do not count as simplification.

Implement approved removals incrementally with unchanged behavior and meaningful before/after size/ownership measurements. No speculative promise to return to 60k lines while preserving all functionality.

**Exit:** agreed simplification is implemented and verified. The overall plan remains incomplete if this phase is merely deferred.

## Verification and documentation

- Exact failed test, then owning test target/crate after each small repair.
- Broad recovery gate: `cargo nextest run --offline --locked --release --no-fail-fast --workspace --all-targets --all-features` (same selection as `just ntest`; offline/locked avoid accidental dependency changes).
- Compile gate when affected: `cargo check --offline --locked --workspace --all-targets --all-features`.
- Run relevant measured/native/export/QA recipes at the increment that changes their behavior. Full cinema/acoustic matrices are milestone gates, not repeated after every diagnostic fix. Record unavailable external datasets/backends explicitly.
- Keep `CHANGELOG.md`, `docs/ROOMEQ_MANUAL.md`, `src/bin/roomeq/INPUT_FORMAT.md`, and `src/bin/roomeq/input_schema.json` synchronized when behavior/configuration changes; include the corresponding output-schema checks where needed.
- Record physical/acoustic limits separately from perceptual claims. Calibrating new audibility thresholds or collecting missing measurements is separate work, not a reason to fabricate passing evidence.

## Progress ledger

### Initial recovery run

- Revision: `7b5ecef`, plus five pre-existing modified files reported by `git status`.
- Command: broad recovery gate above.
- Result: **3,134 executed; 3,107 passed; 27 failed; 27 skipped**, exit 100, 137.540 s test execution.
- Full log: `/private/tmp/roomeq-incremental-initial-tests.log` (local diagnostic artifact).
- Failures: 16 root integration tests (including generated-data and broadband-matching cases), one engine Hybrid phase test, three workflow finalization tests, one QA nightly decision test, and six cinema scaling benchmark instances.
- No production changes were made before this run. Existing skipped cases are not a new workaround.

### Recovery increment 1: missing measurement diagnostic

- Preserved the underlying missing-file cause through raw seat capture.
- The existing `test_roomeq_missing_measurement` regression now passes, with its original path and no-output assertions unchanged.
- Verification: `cargo nextest run --offline --locked --release -p autoeq --all-features --test roomeq_integration_test -E 'test(test_roomeq_missing_measurement)'`: 1 passed, 4 outside the selected filter. Log: `/private/tmp/roomeq-incremental-missing-measurement.log`.
- This is a diagnostic repair only; no acoustic policy or test threshold changes.

### Recovery increment 2: FIR ownership and a stale integration assertion

- Supporting-source retained FIR taps now describe the persisted convolution kernel; normalization remains in the existing gain plugin. The prior helper test only asserted gain multiplication and did not check artifact ownership; the channel test now compares retained taps to actual WAV samples.
- The multidriver CLI assertion now permits the existing explicitly labelled final channel-level alignment gain, at most once; crossovers must still appear on each driver branch, and other channel-level plugin types remain prohibited.
- Verification: `cargo nextest run --offline --locked --release -p autoeq -p roomeq-workflow --all-features --lib --tests -E 'binary(roomeq_supporting_source) | binary(roomeq_integration_test) | (package(roomeq-workflow) & test(supporting_source::))'`: **28 passed**; 670 outside the selected filter. Log: `/private/tmp/roomeq-incremental-recovery-batch1.log`.
- This covers five of the original failures, including increment 1. Step 1 remains open; the broad suite has not been rerun after these repairs.

### Recovery increment 3: observer cancellation propagation

- Check the observer's stop flag before propagating a channel-optimizer error, on both the first attempt and the multi-seat retry. Ordinary errors still propagate when the observer has not stopped.
- This follows the existing Post-EQ callback contract; the original cancellation assertion is unchanged.
- Pipeline-target and topology-run verification pending. The pipeline target also contains the independently failing CTC final-seat evidence test.

### Next recovery investigation: missing acceptance evidence

- Final selection currently returns success with a skipped stage if no acceptance report exists. This bypass explains the three failing finalization regressions: missing scorecard, electrically unbounded output, and attenuation not checked against the original system.
- Do not fix those regressions by fabricating acceptance reports in the tests. Trace and repair evidence creation for the supported routes, then enforce selection. The current safety gate also deliberately excludes supporting-source outputs and some multi-driver paths; removing the bypass alone is not a complete repair.
- No change to this acceptance path has been made in the first three increments.


### Reorg worktree implementation log — 2026-09-10

- Worktree: `reorg-20260910` at `.worktrees/reorg-20260910`; branch created from `5c58887`.
- Added the Stage 0 audibility contract to this worktree as `reviews/ROOMEQ_AUDIBILITY_CONTRACT.md`. It is the governing contract for physical limits, psychoacoustic claims, evidence states, and before/after comparisons; it does not silently change DSP defaults.
- Added this implementation log to `reviews/ROOMEQ_INCREMENTAL_PLAN.md`. The test-recovery gate remains first; contract-driven algorithm changes are gated on a green recovery run.
- The user-requested execution note is `reorg-20260910.md`; it lists the staged worktree sequence and verification gates.
- Current known state inherited from the parent worktree: the initial broad recovery run had 27 failures; the focused first repair batch passed 28 selected tests. No claim of a green broad gate is made here.

### Reorg worktree implementation log — contract clarification — 2026-09-10

- Added the normative acoustic/psychoacoustic operating rules to
  `reviews/ROOMEQ_AUDIBILITY_CONTRACT.md`: Schroeder-dependent filter width,
  broad upper-band Q<1 behavior with supported resonance-cut exceptions,
  treble/HRTF and precedence-effect evidence limits, magnitude-first timing
  guardrails, and explicit natural bass-extension compromises.
- These rules are documentation and contract state only in this increment;
  they do not change optimizer defaults or claim that ERB/masking proxies are
  calibrated audibility thresholds.
- The implementation remains gated by Stage 1 test recovery. The first code
  change after the recovery gate must add/verify metrics on the complete
  realized graph with identical before/after support.
