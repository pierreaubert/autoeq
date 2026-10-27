//! Executable entry point for crate-owned acoustic-corpus QA.

// Rust guideline compliant 2026-02-21

fn main() -> anyhow::Result<()> {
    roomeq_qa::acoustic::run()
}
