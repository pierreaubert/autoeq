//! Executable entry point for the crate-owned RoomEQ command.

// Rust guideline compliant 2026-02-21

fn main() -> anyhow::Result<()> {
    roomeq_cli::supervisor::run()
}
