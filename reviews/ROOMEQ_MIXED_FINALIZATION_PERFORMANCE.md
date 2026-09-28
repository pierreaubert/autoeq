# Mixed-mode finalization performance

Tracking issue: https://github.com/pierreaubert/autoeq/issues/71

## Plan

1. Preserve the complete candidate search and acceptance requirements.
2. Evaluate FIR responses with an FFT on exact, complete one-sided FFT grids,
   including grids with extra electrical peak anchors. Keep scalar evaluation
   for off-grid anchors and arbitrary grids. Fold long impulse responses
   modulo the FFT length so no taps are discarded.
3. Reuse temporal evidence only when the safety gate leaves all evidence inputs
   unchanged; continue replay after changes.
   Evaluate scalar FIR polynomials with Horner's method, comparing against the
   original direct phasor sum, to avoid per-tap trigonometry on native seat grids.
4. Log finalization candidate counts and elapsed time, including slow-run
   progress visible with warning-level logging.
5. Test FFT/scalar parity, long filters, parallel and mixed chains, fallback,
   and finalization safety. Rerun the full measured case and compare decisions.

## Baseline

`2.2_sigberg1`, mixed override, seed 42, 48 kHz, 4096 FIR taps, 200 optimizer
frequency samples. The waveform grid has 32769 bins (65536-point FFT).
The unmodified reproduction took 1535.1 seconds and exited 1 with rejected
playback. Stack samples at 21 and 163 seconds show direct FIR evaluation in
temporal report refreshes during finalization. Artifacts:
`/Volumes/home_tmp/tmp/autoeq-sigberg-mixed-stall/`.

The existing unrelated Cargo.lock updates are outside this change.

## Profile follow-up

The first FFT implementation accelerated waveform replay but profiling then
showed electrical validation dominating. Its 8193 uniform samples include
extra EQ/crossover center frequencies, which initially defeated exact-grid
recognition. The final implementation recognizes the uniform grid inside that
ordered set and evaluates the extra frequencies directly, without interpolation.
The intermediate run was stopped after 171.7 seconds to apply that fix; its
logs and stack sample remain in `/Volumes/home_tmp/tmp/autoeq-issue-71/mixed/`.

After accelerating electrical grids, profiling showed work spread through
native seat replay, level alignment, and report rebuilding. Those paths still
used per-tap trigonometry on nonuniform measurement grids. The scalar evaluator
now uses Horner's method for the same FIR polynomial, with an independent
direct-sum regression oracle. The second intermediate benchmark was stopped
before completion; its artifacts remain under `mixed-final/`.

## Regression checks

- DSP realization: 12 tests passed, including FFT/direct parity, added electrical
  frequencies, mixed bands, parallel branches, long filters, empty FIR bypass,
  changed sidecars, arbitrary-grid fallback, and scalar/direct-phasor parity.
- Full workflow library: 956 passed, 7 ignored, 1 failed.
- The failing `pruning_qa::qa_roomeq_pruning_conditions_exported_routed_matrix`
  also fails on pre-fix commit `560631c`, with the same Cargo.lock. The isolated
  checkout changes only local UI dependency paths to resolve to the same source.
  Both failures retain the zero-gain sub filter because final acoustic acceptance
  rejects pruning. This performance change does not modify pruning policy.
  Follow-up: https://github.com/pierreaubert/autoeq/issues/72.

## Final measured comparison

| Metric | Before | After |
|---|---:|---:|
| Complete mixed-mode runtime | 1535.1 s | 326.34 s |
| Candidate count | 726 | 726 |
| Accepted candidates | 0 | 0 |
| Exit code | 1 | 1 |

The measured run is 4.7 times faster, saving about 20 minutes. Every candidate
identity, pass/fail decision, and rejection diagnostic matches exactly. The
serialized channel objects are identical, including numeric fields. Pre/post
aggregate scores remain 3.005740658580901, and the final violations remain
`baseline_requires_safety_attenuation` and
`no_candidate_within_electrical_acoustic_limits`.

Warning-only logging now reports advancing candidate counts about every 30
seconds. This remains a several-minute full search; acceptance policy and search
coverage are preserved. The existing rejection is not presented as a successful
playback result.

Final artifacts: `/Volumes/home_tmp/tmp/autoeq-issue-71/verified-mixed/`.
The independent comparison script is `/Volumes/home_tmp/tmp/autoeq-issue-71/compare.py`.
Clippy completed with the same 8 engine and 9 workflow warnings in existing code.

The IIR check finished in 17.19 seconds with exit 0. All 66 candidate decisions
and diagnostics and the serialized channel objects match its pre-fix output.
The accepted aggregate score remains 2.8381773205974468. IIR artifacts are under
`/Volumes/home_tmp/tmp/autoeq-issue-71/verified-iir/`.

Changed Rust files pass rustfmt checks; `git diff --check` passes.
