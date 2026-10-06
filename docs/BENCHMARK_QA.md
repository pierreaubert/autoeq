# Speaker benchmark QA

`just bench-autoeq-speaker` uses `--qa --jobs 1` on every cached speaker and all six original scenarios. Bare `--qa` now applies the shared QA predicate with threshold zero: the selected optimization stage must converge, filter spacing must pass, and the post preference score must be strictly greater than the pre score. Supplied numeric thresholds retain the same strict comparison.

This corrects a previously non-executable bare flag; zero is not claimed as a historical default. It does not change optimizer algorithms, seeds, quotas, refinement, targets, or measurements.

QA retains actual scores and per-scenario failures, drains all started work, and returns failure after complete reporting if any scenario fails, any speaker is incomplete, or the corpus is absent. Incomplete local measurements remain failures. Benchmark results on a current cache do not establish historical corpus identity or real-time audio performance.

QA requires finite pre-score, post-score, explicit threshold and pre + threshold.
Nonfinite values or arithmetic overflow refuse qualification. Finite explicit
numeric thresholds retain the existing strict post > pre + threshold comparison,
including negative finite thresholds.
