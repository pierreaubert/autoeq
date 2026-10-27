//! Executable entry point for crate-owned speaker acquisition.

// Rust guideline compliant 2026-02-21

fn main() -> Result<(), Box<dyn std::error::Error>> {
    autoeq_cli::download::run_command()
}
