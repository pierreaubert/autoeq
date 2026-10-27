//! Executable entry point for the crate-owned AutoEQ benchmark.

// Rust guideline compliant 2026-02-21

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if autoeq_cli::program_version::print_benchmark_version_if_requested(
        autoeq_core::PRODUCT_VERSION,
    ) {
        return Ok(());
    }
    autoeq_cli::benchmark::run_command()
}
