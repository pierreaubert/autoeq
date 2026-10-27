# Numerical QA ownership

`autoeq-qa` is a terminal QA package. Its library supplies reference loading,
comparison helpers, result records, and optimizer benchmark support. Production
libraries must not depend on it.

[`validation-manifest.toml`](validation-manifest.toml) maps 124 logical cases:
118 Rust cases and six separately executed Python cases. Six explicit oracle
targets group the Rust cases by core, optimization, engine, analysis/quality,
workflow, and export. Supplemental `negative_controls`, `qa_contract`, and
`optimizer_benchmark` targets remain distinct. `autotests = false` prevents
retained case files from also becoming standalone Cargo targets.

The package currently declares broad normal dependencies on production and
terminal owners for its tests and optimizer benchmark. Grouping integration
roots does not isolate those package dependencies. The group-layout and build
cost decisions require the controlled refactor measurements.

References come from the configured Wolfram Engine or checked-in goldens.
Missing references fail required comparisons. The manifest runner also rejects
missing or duplicate case records; test-count equality alone is insufficient.
Goldens, tolerance policy, and provenance remain independent of target layout.

See the [usage guide](README.md) and [workspace architecture](../../docs/ARCHITECTURE.md).
