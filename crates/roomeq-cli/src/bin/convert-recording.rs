//! Executable entry point for crate-owned recording conversion.

// Rust guideline compliant 2026-02-21

fn main() -> Result<(), Box<dyn std::error::Error>> {
    roomeq_cli::convert_recording::run()
}
