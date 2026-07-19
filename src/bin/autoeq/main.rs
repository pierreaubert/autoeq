//! Thin compatibility launcher for the crate-owned AutoEQ command.

fn main() -> anyhow::Result<()> {
    autoeq_cli::autoeq_command::run_command()
}
