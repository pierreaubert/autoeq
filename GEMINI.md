# AutoEQ Project Overview

AutoEQ is a Rust-based CLI and library for audio equalization and filter optimization. It helps users find the best equalization settings for their speakers or headphones based on measurements. It also includes tools for room simulation and CEA2034 (Spinorama) metrics calculation.

## Project Structure

The project is organized as a Rust workspace with the following members:

*   **`autoeq`**: The main CLI application for computing EQs. It supports various optimization algorithms and loss functions.
*   **`autoeq-cea2034`**: Implementation of CEA2034 (Spinorama) standards for loudspeaker performance metrics.
*   **`autoeq-roomsim`**: A room simulator to analyze speaker response in a given room.
*   **`autoeq-env`**: Shared constants and environment utilities.

## Building and Running

The project uses `just` as a command runner. Ensure you have `just` installed (`cargo install just`).

### Common Commands

*   **Build Release Binaries:**
    ```bash
    just prod
    ```
    This builds `autoeq`, `roomeq`, and other binaries in release mode.

*   **Run Tests:**
    ```bash
    just test
    ```
    Runs all unit tests in the workspace.

*   **Run QA (Quality Assurance):**
    ```bash
    just qa
    ```
    Runs a series of integration tests and standard optimization scenarios (e.g., JBL M2, Beyerdynamic DT1990 Pro) to verify performance and correctness.

*   **Format Code:**
    ```bash
    just fmt
    ```

*   **Run Demos:**
    ```bash
    just demo
    ```

### CLI Usage

**AutoEQ (Headphone/Speaker EQ):**

```bash
# General usage
cargo run --bin autoeq --release -- [OPTIONS]

# Example: Optimize using Spinorama data
cargo run --bin autoeq --release -- --speaker="MAG Theatron S6" --version asr --measurement CEA2034 --algo cobyla

# Example: Optimize using a local measurement file
cargo run --bin autoeq --release -- --curve path/to/measurement.csv --target path/to/target.csv --algo autoeq:de

# Help
cargo run --bin autoeq --release -- --help
```

**RoomEQ (Room Correction):**

```bash
cargo run --bin roomeq --release -- --config room_config.json --output dsp_chain.json
```

**Download Data:**

```bash
# Download measurement data from Spinorama.org
cargo run --bin autoeq-download-speakers --release
```

## Development Conventions

*   **Language:** Rust (Edition 2024).
*   **Build System:** Cargo with `Justfile` for workflow automation.
*   **Testing:** Standard `cargo test` for unit tests. Integration and performance tests are handled via `just qa` and specific binary targets (e.g., `benchmark-autoeq-speaker`).
*   **Code Style:** Standard Rust formatting (`cargo fmt`).
*   **Environment Variables:** `AUTOEQ_DIR` is used for setting the project root during development/testing.

## Key Configuration Files

*   **`Cargo.toml`**: Workspace root configuration defining members and dependencies.
*   **`Justfile`**: Defines project-specific commands for building, testing, and maintenance.
*   **`autoeq/src/cli.rs`**: Defines the command-line arguments and validation logic for the main `autoeq` binary.
*   **`autoeq/README.md`**: Detailed documentation for the `autoeq` CLI.
