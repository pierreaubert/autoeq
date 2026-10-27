//! Executable entry point for crate-owned RoomEQ coverage QA.

// Rust guideline compliant 2026-02-21

fn main() -> anyhow::Result<()> {
    if roomeq_qa::coverage::run()? {
        std::process::exit(1);
    }
    Ok(())
}
