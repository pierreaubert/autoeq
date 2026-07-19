//! Thin compatibility launcher for crate-owned recording conversion.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    roomeq_cli::convert_recording::run()
}
