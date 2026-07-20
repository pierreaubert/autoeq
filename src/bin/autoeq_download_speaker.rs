//! Thin compatibility launcher for crate-owned speaker acquisition.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    autoeq_cli::download::run_command()
}
