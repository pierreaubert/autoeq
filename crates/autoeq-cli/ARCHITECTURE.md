# autoeq-cli — Architecture

AutoEQ command-line adapters. Argument parsing and subcommand dispatch only;
DSP orchestration lives in `autoeq-workflow`.

## Layer

Terminal presentation surface owning `cli::Args`, enum parsing adapters,
validation messages, and list/process-exit helpers.

## Key modules

| Module | Owns |
|---|---|
| `cli` | Shared argument parsing, validation, and neutral-config adapters |
| `autoeq_command` | `autoeq` subcommands (optimize speakers/headphones) |
| `benchmark` | `benchmark-autoeq-speaker` harness |
| `download` | `autoeq-download-speakers` (Spinorama fetch) |

## Core abstractions

- **Adapter, not logic.** Each module parses flags into `autoeq-workflow`
  config types and calls one workflow entry point; no DSP math here.
- **Shared `cli` shapes.** Because argument types live in this terminal crate,
  every binary accepts the same optimizer/target/seed flags.

## API shape

No DSP entry points here by design. The `autoeq/` modules (`runopt`, `load`,
`save`, `prescore`/`postscore`, `qa`, `spacing`, `progress`) are clap argument
shapes plus small adapters: each parses flags into `autoeq-workflow` config
types and calls one workflow function. Adding a flag means extending the arg
struct and threading the field into the existing workflow call — never new
math in this crate.

## Consumers

Owns `autoeq`, `benchmark-autoeq-speaker`, and `autoeq-download-speakers`
under `src/bin/`, plus their executable integration tests under `tests/`.

`Args` converts explicitly to `autoeq_optim_types::OptimizationConfig` and the
workflow/plot/backend request types. Desktop and library clients construct the
neutral configuration directly; they do not depend on this terminal package.
