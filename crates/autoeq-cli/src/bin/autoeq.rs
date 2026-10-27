//! Executable entry point for the crate-owned AutoEQ command.

// Rust guideline compliant 2026-02-21

fn main() -> anyhow::Result<()> {
    if autoeq_cli::program_version::print_autoeq_version_if_requested(autoeq_core::PRODUCT_VERSION)
    {
        return Ok(());
    }
    autoeq_cli::autoeq_command::run_command()
}
