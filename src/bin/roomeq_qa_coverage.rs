//! Thin compatibility launcher for crate-owned RoomEQ coverage QA.

fn main() -> anyhow::Result<()> {
    if roomeq_qa::coverage::run()? {
        std::process::exit(1);
    }
    Ok(())
}
