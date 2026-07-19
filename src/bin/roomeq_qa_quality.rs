//! Thin compatibility launcher for crate-owned RoomEQ quality QA.

fn main() -> anyhow::Result<()> {
    if roomeq_qa::quality::run()? {
        std::process::exit(1);
    }
    Ok(())
}
