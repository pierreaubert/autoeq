//! Thin compatibility launcher for crate-owned acoustic-corpus QA.

fn main() -> anyhow::Result<()> {
    roomeq_qa::acoustic::run()
}
