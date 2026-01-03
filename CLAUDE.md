# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoEQ is a Rust CLI toolkit for computing parametric EQ corrections for speakers, headphones, and room acoustics. It uses global optimization algorithms (Differential Evolution, metaheuristics, NLopt) to find optimal IIR/FIR filter parameters that minimize a loss function (flat response or Harman score optimization).

## Build Commands

```bash
# Install just (required task runner)
cargo install just

# Build release binaries
just prod              # builds autoeq, roomeq, and benchmarking tools
just prod-autoeq       # builds only autoeq binary
just prod-roomeq       # builds only roomeq binary

# Run tests
export AUTOEQ_DIR=$(pwd)   # required for tests to locate data files
cargo test --workspace --lib
just test                  # runs cargo check + lib tests

# Quality assurance (runs optimization scenarios with --qa flag)
just qa

# Format code
just fmt

# Download speaker data from spinorama.org for benchmarking
just download
```

## Running Single Tests

```bash
# Run a specific test
AUTOEQ_DIR=$(pwd) cargo test --lib test_name

# Run tests in a specific crate
AUTOEQ_DIR=$(pwd) cargo test -p autoeq --lib
AUTOEQ_DIR=$(pwd) cargo test -p autoeq-cea2034 --lib
```

## Architecture

### Workspace Crates

- **autoeq**: Main crate with CLI binaries and optimization logic
- **autoeq-cea2034**: CEA2034/Spinorama calculations (listening window, early reflections, predicted in-room response, speaker scores)
- **autoeq-env**: Environment utilities and constants shared across crates
- **autoeq-roomsim**: WASM-targeted room acoustic simulator using BEM (boundary element method)

### Key Binaries (in autoeq crate)

- `autoeq`: Main CLI for headphone/speaker EQ optimization
- `roomeq`: Multi-channel/subwoofer/DBA optimization with JSON config
- `benchmark-autoeq-speaker`: Benchmarking tool for algorithm comparison
- `autoeq-download-speakers`: Downloads speaker data from spinorama.org

### Core Modules (autoeq/src/)

- `loss.rs`: Loss functions (flat, score, mixed) for speakers and headphones
- `optim.rs` + `optim_de.rs` + `optim_mh.rs` + `optim_nlopt.rs`: Optimization backends (custom DE, metaheuristics-nature, NLopt)
- `workflow.rs`: High-level optimization workflows used by binaries
- `cli.rs`: Clap CLI argument definitions
- `x2peq.rs`: Converts optimizer parameter vectors to PEQ filter specifications
- `constraints/`: Constraint functions for optimization (frequency spacing, Q limits, etc.)

### External Math Libraries

The project uses custom math crates via `[patch.crates-io]`:
- `math-iir-fir`: IIR/FIR filter design (aliased as `iir` in code)
- `math-differential-evolution`: Custom DE optimizer (aliased as `de`)
- `math-bem`: Boundary element method for room simulation

## Coding Conventions

- Rust edition 2024 with strict Clippy lints (pedantic, cargo, complexity, correctness, perf)
- Uses `thiserror` for error types, `anyhow` for binary error handling
- `ndarray` for numerical arrays with BLAS/rayon support
- Tests require `AUTOEQ_DIR` environment variable pointing to project root
