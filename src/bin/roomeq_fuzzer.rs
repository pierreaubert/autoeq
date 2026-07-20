//! Thin compatibility launcher for crate-owned RoomEQ fuzzing.

use autoeq_optim::loss::DriversLossData;
use roomeq_qa::fuzzer::DriverPlotter;
use std::error::Error;
use std::path::Path;

struct Plotter;

impl DriverPlotter for Plotter {
    fn plot_drivers_results(
        &self,
        data: &DriversLossData,
        gains: &[f64],
        crossover_freqs: &[f64],
        sample_rate: f64,
        output: &Path,
    ) -> anyhow::Result<()> {
        autoeq_plot::plot_drivers_results(data, gains, crossover_freqs, None, sample_rate, output)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    if roomeq_qa::fuzzer::run(&Plotter)? {
        std::process::exit(1);
    }
    Ok(())
}
