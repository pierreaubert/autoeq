//! Executable entry point for crate-owned feature-progression QA.

// Rust guideline compliant 2026-02-21

fn main() -> anyhow::Result<()> {
    roomeq_qa::features::run()
}
