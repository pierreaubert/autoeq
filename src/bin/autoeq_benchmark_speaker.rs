//! Thin compatibility launcher for the crate-owned AutoEQ benchmark.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    autoeq_cli::benchmark::run_command()
}
