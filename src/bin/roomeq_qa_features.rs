//! Thin compatibility launcher for crate-owned feature-progression QA.

fn main() -> anyhow::Result<()> {
    roomeq_qa::features::run()
}
