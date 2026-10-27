//! Executable entry point for crate-owned synthetic RoomEQ QA.

// Rust guideline compliant 2026-02-21

fn main() -> anyhow::Result<()> {
    if roomeq_qa::synthetic::run()? {
        std::process::exit(1);
    }
    Ok(())
}
