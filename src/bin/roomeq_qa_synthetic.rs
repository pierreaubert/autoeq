//! Thin compatibility launcher for crate-owned synthetic RoomEQ QA.

fn main() -> anyhow::Result<()> {
    if roomeq_qa::synthetic::run()? {
        std::process::exit(1);
    }
    Ok(())
}
