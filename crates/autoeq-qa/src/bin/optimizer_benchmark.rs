use autoeq_qa::optimizer_benchmark::{
    BenchmarkCancellation, BenchmarkCliArgs, BenchmarkRunOptions, run_benchmark_cell_spec_file,
    run_optimizer_benchmark, write_cell_specs, write_controlled_cell_result, write_report,
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
    if args.list_cell_specs {
        if args.output.is_some()
            || args.evaluation_budget.is_some()
            || args.time_budget_seconds.is_some()
            || !args.seeds.is_empty()
            || args.limit_cells.is_some()
        {
            return Err(
                "--list-cell-specs cannot be combined with matrix overrides or --output".into(),
            );
        }
        return write_cell_specs();
    }
    if let Some(path) = args.cell_spec.as_deref() {
        if args.evaluation_budget.is_some()
            || args.time_budget_seconds.is_some()
            || !args.seeds.is_empty()
            || args.limit_cells.is_some()
        {
            return Err("--cell-spec cannot be combined with matrix overrides".into());
        }
        let result = run_benchmark_cell_spec_file(path)?;
        write_controlled_cell_result(args.output.as_deref(), &result)?;
        eprintln!(
            "cell={} outcome={:?} root_search={}/{} validation={} elapsed_ms={}",
            result.cell_id,
            result.outcome,
            result.root_counters.evaluations_started,
            result.root_counters.evaluation_budget,
            result.root_counters.validation_evaluations_started,
            result.elapsed_millis
        );
        return Ok(());
    }
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
