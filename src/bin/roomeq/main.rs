//! Thin compatibility launcher for the crate-owned RoomEQ command.

fn main() -> anyhow::Result<()> {
    roomeq_cli::roomeq::run_command()
}
