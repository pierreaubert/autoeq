use autoeq_qa::optimizer_benchmark::{
    BenchmarkCancellation, BenchmarkCliArgs, BenchmarkRunOptions, run_optimizer_benchmark,
    write_report,
};
use clap::Parser;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("optimizer benchmark failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args = BenchmarkCliArgs::parse();
    let cancellation = BenchmarkCancellation::default();
    let worker_cancellation = cancellation.clone();
    let time_budget_millis = args
        .time_budget_seconds
        .map(|seconds| {
            seconds
                .checked_mul(1_000)
                .ok_or_else(|| "time budget seconds overflow milliseconds".to_string())
        })
        .transpose()?;
    let options = BenchmarkRunOptions {
        evaluation_budget: args.evaluation_budget,
        time_budget_millis,
        seeds: (!args.seeds.is_empty()).then_some(args.seeds.clone()),
        limit_cells: args.limit_cells,
    };
    let mut worker =
        tokio::task::spawn_blocking(move || run_optimizer_benchmark(options, worker_cancellation));

    let report = tokio::select! {
        joined = &mut worker => joined
            .map_err(|error| format!("benchmark worker failed: {error}"))??,
        signal = tokio::signal::ctrl_c() => {
            signal.map_err(|error| format!("could not listen for Ctrl+C: {error}"))?;
            eprintln!("cancelling: closing objective scoring and waiting for active work");
            cancellation.request_and_wait_for_scores();
            worker.await
                .map_err(|error| format!("benchmark worker failed while stopping: {error}"))??
        }
    };

    write_report(args.output.as_deref(), &report)?;
    eprintln!(
        "matrix complete={} cancelled={} cells={} distributions={}",
        report.complete_matrix,
        report.cancelled,
        report.cells.len(),
        report.distributions.len()
    );
    Ok(())
}
