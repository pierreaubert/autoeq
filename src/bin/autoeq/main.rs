//! Thin compatibility launcher for the crate-owned AutoEQ command.

fn main() -> anyhow::Result<()> {
    if autoeq_cli::program_version::print_autoeq_version_if_requested(env!("CARGO_PKG_VERSION")) {
        return Ok(());
    }
    autoeq_cli::autoeq_command::run_command()
}
