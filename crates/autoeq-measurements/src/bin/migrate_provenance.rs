//! Executable entry point for the measurements-owned provenance migration command.

// Rust guideline compliant 2026-02-21

fn main() -> Result<(), Box<dyn std::error::Error>> {
    autoeq_measurements::provenance_migration_cli::run()
}
