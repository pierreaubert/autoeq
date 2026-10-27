# autoeq-objective

Objective-function evaluation for AutoEQ: loss computations,
objective data and builders, prepared objectives, evaluation
budgets (`run_control`), and evaluation-only constraints.

Extracted from `autoeq-optim` (refactor-20261025 Phase 3c) so
prepared evaluation can be reused without compiling optimizer
backends. `autoeq-optim` re-exports these modules, so existing
`autoeq_optim::` paths keep resolving.
