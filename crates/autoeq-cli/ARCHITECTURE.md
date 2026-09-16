# autoeq-cli — Architecture

AutoEQ command-line adapters. Argument parsing and subcommand dispatch only;
all behavior lives in `autoeq-workflow`.

## Layer

Thin presentation surface. Re-exports `autoeq-workflow::cli`.

## Key modules

| Module | Owns |
|---|---|
| `autoeq_command` | `autoeq` subcommands (optimize speakers/headphones) |
| `benchmark` | `benchmark-autoeq-speaker` harness |
| `download` | `autoeq-download-speakers` (Spinorama fetch) |

## Core abstractions

- **Adapter, not logic.** Each module parses flags into `autoeq-workflow`
  config types and calls one workflow entry point; no DSP math here.
- **Shared `cli` shapes.** Because argument types live in `autoeq-workflow`,
  every binary accepts the same optimizer/target/seed flags.

## API shape

No DSP entry points here by design. The `autoeq/` modules (`runopt`, `load`,
`save`, `prescore`/`postscore`, `qa`, `spacing`, `progress`) are clap argument
shapes plus small adapters: each parses flags into `autoeq-workflow` config
types and calls one workflow function. Adding a flag means extending the arg
struct and threading the field into the existing workflow call — never new
math in this crate.

## Consumers

Root binaries `autoeq`, `benchmark-autoeq-speaker`, `autoeq-download-speakers`.
