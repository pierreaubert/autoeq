//! Thin compatibility launcher for the crate-owned AutoEQ benchmark.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if autoeq_cli::program_version::print_benchmark_version_if_requested(env!("CARGO_PKG_VERSION"))
    {
        return Ok(());
    }
    autoeq_cli::benchmark::run_command()
}
