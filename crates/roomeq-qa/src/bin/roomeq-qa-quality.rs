//! Executable entry point for crate-owned RoomEQ quality QA.

// Rust guideline compliant 2026-02-21

fn main() -> anyhow::Result<()> {
    if roomeq_qa::quality::print_program_version_if_requested(autoeq_core::PRODUCT_VERSION) {
        return Ok(());
    }
    if roomeq_qa::quality::run()? {
        std::process::exit(1);
    }
    Ok(())
}
