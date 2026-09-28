# RoomEQ subwoofer attenuation policy — issue #73

## Requirement

Physical subwoofer attenuation must not consume the main-output attenuation
budget. Keep the electrical output ceiling and acoustic acceptance checks.
Diagnostics must identify the physical output responsible for a rejection.

## Implementation

- Candidate output attenuation and structural fallback use one role-aware budget
  check. Routed output roles come from the bass-management graph; independent
  outputs use their owning channel's existing subwoofer classification.
- `max_attenuation_db` continues to bound non-subwoofer outputs. A common cut
  remains bounded because it also reduces main output.
- Spectral trials with a subwoofer requirement above the budget apply that cut
  at the physical subwoofer output before the common spectral refinement.
  Existing spectral trials within budget remain unchanged.
- Every physical output is still replayed against the electrical ceiling.
  Declared physical-drive limits and final acoustic checks remain enforced.
- Rejections list required attenuation for each physical output and mark exempt
  subwoofers. No optimizer filter bounds or acoustic acceptance thresholds changed.

## Measured verification assumptions

The existing measured configurations, target curves and CSV measurements are
unchanged. Sigberg uses seed 42, 48 kHz output, and the supplied IIR/mixed overrides.
Mixed uses 4096 FIR taps and a 300 Hz hybrid split. Its measured main/sub timing
remains unverified; this change makes no new coherent-cancellation claim.

## Tests

- Finalization library tests: 41 passed, 3 ignored.
- Workflow library: 958 passed, 7 ignored, one existing routed-pruning failure
  (`pruning_qa::qa_roomeq_pruning_conditions_exported_routed_matrix`, issue #72).
- The three new regressions cover an unbounded sub cut with bounded main/common
  cuts, a 30 dB structural subwoofer fallback, and electrical replay proving that
  subwoofer attenuation preserves the main transfer and enforces the ceiling.
- `mbx clippy -p roomeq-workflow -p roomeq-model --lib --no-deps`: passed with
  nine existing workflow warnings; none added by this change.
- Changed Rust files pass rustfmt; `git diff --check` passes.

Commands use the repository's `rtk proxy` wrapper. Detailed measured results and
binary hashes are retained under `/Volumes/home_tmp/tmp/autoeq-issue-73/`.

## Final measured results

Final binary SHA-256: `b76e025b72ef0cbf59bfc26931ae69df145aa4d8662dcc8fa9e6c9421fb68891`. All three final runs recorded this hash.

| Case | Result | Evidence |
| --- | --- | --- |
| 2.2_sigberg1 / IIR | Accepted, 16.90 s | Score 2.8381773205974468; serialized channels exactly match the original suite. |
| 2.2_sigberg1 / mixed | Rejected, 317.00 s | All 726 candidates assessed; delivered fallback channels exactly match the original suite. |
| 5.1.4_genelec / mixed | Diagnostic fallback saved, 192.76 s | The former hard error at 19.585 dB sub attenuation is gone. All ten output ceiling checks pass. Acoustic regression/output-loss checks still reject playback. |

Sigberg mixed full-correction output requirements:

| Output | Required attenuation | Policy |
| --- | ---: | --- |
| L | 14.403 dB | Main; exceeds 12 dB budget |
| R | 20.935 dB | Main; exceeds 12 dB budget |
| Sub1 | 9.845 dB | Subwoofer; exempt |
| Sub2 | 10.737 dB | Subwoofer; exempt |

The reported 20.935 dB came from the right main. The exemption therefore does
not make this mixed setup accepted. Lower-strength alternatives remain subject
to the existing acoustic/output-loss and processing-mode checks.

The Genelec fallback applies 19.585278 dB of subwoofer safety attenuation with
its 18 dB main budget unchanged. Its sub output and all mains satisfy the
electrical ceiling. No accepted-improvement claim is made for that fallback.

Source exploration used TokenSave (approximately 2,000 tokens saved).
