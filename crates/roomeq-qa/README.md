# roomeq-qa

RoomEQ scenario matrices, runners, scorecards, and QA command implementations.

## Ownership

- Owns acoustic, coverage, convergence, feature, fuzzer, and synthetic QA orchestration.
- Exercises public production APIs but does not own production pipeline behavior.

## Testing

```bash
cargo test -p roomeq-qa --lib
just qa-roomeq-contract-pr
just qa-roomeq-ci
```

## Gate purposes

Every registry expectation declares what it proves:

- `safety` verifies that execution is finite and bounded. It may accept a
  runtime `REVERTED` result only when `allow_safe_revert` is also true.
- `functional` verifies that a requested processing path is present and
  retained. A revert is a failure even when the uncorrected fallback is safe.
- `quality` verifies retained correction against quantitative score,
  improvement, and boost limits. A revert is a failure.

Runtime fallback remains a product safety mechanism; it is not evidence that
FIR, hybrid, crossover, multi-seat, or other requested behavior works. The
quick matrix is the explicit safety tier. Functional and quality matrices have
a zero-unexpected-reversion policy.

## Contract and convergence tiers

`qa-roomeq-contract-pr` is the deterministic pull-request gate. It checks the
canonical serialized-DSP realization, typed main/sub crossover roles, CTC
replay, registry gate semantics, and the measured Genelec 5.1.4 IIR/FIR/hybrid
contract. Its quality optimizer budget is explicitly bounded with `--maxeval`.
The contract uses one fixed optimizer seed; nightly and weekly convergence use
the five-seed median and report the full score spread.

`qa-roomeq-ci` adds quick safety coverage, multi-seat guards, and perceptual
contracts. The randomized five-seed quality fuzzer is intentionally scheduled
only in nightly/weekly QA, where its runtime and convergence distribution do
not make blocking PR feedback slow or stochastic.

Nightly and weekly runs retain the larger convergence budget and broad scenario
matrix. They measure optimizer quality and stochastic robustness; they do not
replace deterministic correctness contracts. The reachability test fails when
a suite declared in `registry.json` is no longer invoked by CI or a schedule.

## Release stabilization and defect accounting

For optimizer, crossover, routing, or DSP-realization changes, keep a 48-hour
stabilization window before release. During that window, run the deterministic
contract on every change and complete the nightly/weekly quality suites without
unexpected reversion or unexplained baseline drift.

Track escaped defects by the boundary that failed:

1. objective/optimizer convergence;
2. driver role, routing, or topology ownership;
3. serialized DSP realization versus the optimized model;
4. acceptance oracle or report reconstruction;
5. registry-to-CI reachability.

For each release, record unexpected revert count, strict cross-mode drift,
mutation survivors in realization/role/gate-purpose contracts, suite runtime,
and each escaped defect together with the deterministic test added for it.
